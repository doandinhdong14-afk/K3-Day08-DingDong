"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

# Cache module-level: bge-m3 nặng ~2.2GB, load lại mỗi lần gọi mất ~15s và đó là
# phần lớn thời gian chạy eval (xem results.md, đề xuất cải thiện #4). Cache không
# làm đổi kết quả — cùng model, cùng vector, cùng thứ hạng.
_MODEL = None
_COLLECTION = None


def _task4_config():
    """Lấy config của Task 4, chạy được cả `python -m src.task5...` lẫn chạy trực tiếp."""
    try:
        from .task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL
    except ImportError:
        from task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL
    return EMBEDDING_MODEL, COLLECTION_NAME, CHROMA_DIR


def _get_model():
    """Load (một lần) SentenceTransformer dùng chung cho mọi truy vấn."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        embedding_model, _, _ = _task4_config()
        _MODEL = SentenceTransformer(embedding_model)
    return _MODEL


def _get_collection():
    """Mở (một lần) collection ChromaDB đã index ở Task 4."""
    global _COLLECTION
    if _COLLECTION is None:
        import chromadb

        _, COLLECTION_NAME, CHROMA_DIR = _task4_config()
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _COLLECTION = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _COLLECTION


def generate_hypothetical_document(query: str) -> str:
    """
    Sinh tài liệu giả định (Hypothetical Document) từ câu hỏi bằng LLM.
    """
    import os
    import requests
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return query
        
    try:
        prompt = (
            f"Hãy viết một câu trả lời giả định hoặc một đoạn thông tin ngắn, chính xác bằng tiếng Việt "
            f"cho câu hỏi: '{query}'. "
            f"Đoạn văn này sẽ được dùng để tìm kiếm tài liệu tương đồng ngữ nghĩa."
        )
        
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "google/gemini-2.5-flash",
                "messages": [
                    {"role": "system", "content": "You are a university administration assistant who writes draft university rules in Vietnamese."},
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=8
        )
        if response.status_code == 200:
            res_json = response.json()
            choices = res_json.get("choices", [])
            if choices:
                draft = choices[0].get("message", {}).get("content", "").strip()
                if draft:
                    print(f"[HyDE] Da tao van ban gia dinh: {draft[:80]}...")
                    return draft
    except Exception as e:
        print(f"[WARNING] Loi khi sinh tai lieu gia dinh HyDE: {e}")
        
    return query


def semantic_search(query: str, top_k: int = 10, use_hyde: bool = False) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa
        use_hyde: Có sử dụng tài liệu giả định HyDE hay không

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    # Nếu bật HyDE, tạo tài liệu giả định trước khi tìm kiếm ngữ nghĩa
    search_query = generate_hypothetical_document(query) if use_hyde else query

    # Chuyển câu hỏi sang vector (model đã cache ở module level)
    query_vector = _get_model().encode(search_query).tolist()

    # Truy vấn cơ sở dữ liệu vector ChromaDB
    collection = _get_collection()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    if not results or not results["documents"] or not results["documents"][0]:
        return []

    output = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        # Tính điểm tương đồng từ khoảng cách cosine
        score = max(0.0, 1.0 - dist)
        output.append({
            "content": doc,
            "score": round(float(score), 4),
            "metadata": meta
        })

    # Sắp xếp từ cao xuống thấp
    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("what is the tuition fee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
