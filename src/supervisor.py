"""
Supervisor + Workers song song — pattern nâng cao cho retrieval.

Task 9 (`task9_retrieval_pipeline.retrieve`) chạy TUẦN TỰ: semantic → lexical → merge.
Module này là bản mở rộng do Role 1 (RAG Architect) phụ trách:

    Query
      │
      ├─ Supervisor: phân rã câu hỏi thành nhiều "góc truy vấn"
      │     ├─ Worker: dense  (semantic_search trên query gốc)
      │     ├─ Worker: sparse (BM25 trên query gốc)
      │     ├─ Worker: dense  (HyDE — tài liệu giả định)          [tuỳ chọn]
      │     └─ Worker: dense  (query expansion — biến thể 1..n)   [tuỳ chọn]
      │        ↑ tất cả chạy SONG SONG trong ThreadPoolExecutor
      │
      ├─ Fusion: RRF gộp mọi ranked list của các worker
      ├─ Rerank (Task 7)
      └─ Fallback PageIndex nếu cosine gốc tốt nhất < SCORE_THRESHOLD

Vì sao dùng thread chứ không phải process:
    Các worker đều là I/O-bound hoặc nằm trong thư viện đã nhả GIL (ChromaDB query,
    HTTP call tới LLM, forward pass của torch). ThreadPoolExecutor cho song song thật
    mà không phải trả giá pickle/spawn của multiprocessing.

Suy giảm mượt (graceful degradation):
    Không có API key → bỏ qua worker HyDE / query expansion, chạy đúng như Task 9.
    Một worker lỗi → supervisor ghi nhận và tiếp tục với các worker còn lại.

Chạy thử:
    python -m src.supervisor
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .task5_semantic_search import (
        _get_model, generate_hypothetical_document, semantic_search,
    )
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
    from .task9_retrieval_pipeline import (
        DEFAULT_TOP_K, RERANK_METHOD, SCORE_THRESHOLD,
    )
except ImportError:  # khi chạy trực tiếp `python src/supervisor.py`
    from task5_semantic_search import (
        _get_model, generate_hypothetical_document, semantic_search,
    )
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank, rerank_rrf
    from task8_pageindex_vectorless import pageindex_search
    from task9_retrieval_pipeline import (
        DEFAULT_TOP_K, RERANK_METHOD, SCORE_THRESHOLD,
    )


# Số biến thể câu hỏi mà supervisor yêu cầu LLM sinh ra. 2 là đủ: thêm nữa thì
# các biến thể bắt đầu trùng ý nhau mà vẫn tốn thêm 1 lượt gọi LLM.
N_QUERY_VARIANTS = 2

# Giới hạn worker chạy đồng thời. Corpus của bài lab nhỏ, 6 là thừa sức.
MAX_WORKERS = 6


# =============================================================================
# QUERY EXPANSION (Multi-Query Retrieval)
# =============================================================================

def expand_query(query: str, n: int = N_QUERY_VARIANTS) -> list[str]:
    """
    Sinh n biến thể/cách diễn đạt khác của câu hỏi bằng LLM.

    Mục đích: câu hỏi của sinh viên thường dùng từ đời thường ("đóng tiền học"),
    còn văn bản quy định dùng từ hành chính ("thanh toán học phí"). Search riêng
    từng biến thể rồi gộp bằng RRF giúp bắt được cả hai cách diễn đạt.

    Trả về [] nếu không có API key hoặc gọi LLM lỗi — supervisor vẫn chạy bình thường.
    """
    import os

    import requests
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key or api_key.startswith("your_") or "sk-or-v1-..." in api_key:
        return []

    prompt = (
        f"Viết {n} cách diễn đạt khác của câu hỏi sau, giữ nguyên ý định tìm kiếm "
        f"nhưng đổi từ vựng (dùng thuật ngữ hành chính/quy định của trường đại học). "
        f"Mỗi biến thể một dòng, KHÔNG đánh số, KHÔNG giải thích.\n\n"
        f"Câu hỏi: {query}"
    )

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "google/gemini-2.5-flash",
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=10,
        )
        if response.status_code != 200:
            return []

        choices = response.json().get("choices", [])
        if not choices:
            return []

        raw = choices[0].get("message", {}).get("content", "")
        variants = [
            line.strip(" -•*\t")
            for line in raw.splitlines()
            if line.strip() and line.strip().lower() != query.lower()
        ]
        return variants[:n]
    except Exception as e:
        err_msg = str(e).encode("ascii", errors="ignore").decode("ascii")
        print(f"  [WARNING] Query expansion that bai: {err_msg}")
        return []


# =============================================================================
# SUPERVISOR
# =============================================================================

def _run_workers(tasks: list[tuple[str, callable]]) -> dict[str, list[dict]]:
    """
    Chạy song song danh sách (tên_worker, hàm_không_tham_số).

    Worker nào ném exception thì ghi log và trả list rỗng cho worker đó —
    một nhánh hỏng không được kéo sập cả pipeline.
    """
    outputs: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks}
        for future in as_completed(futures):
            name = futures[future]
            try:
                outputs[name] = future.result() or []
            except Exception as e:
                err_msg = str(e).encode("ascii", errors="ignore").decode("ascii")
                print(f"  [WARNING] Worker '{name}' loi: {err_msg}")
                outputs[name] = []
    return outputs


def supervisor_retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_hyde: bool = True,
    use_expansion: bool = True,
    verbose: bool = False,
) -> list[dict]:
    """
    Retrieval song song do supervisor điều phối.

    Args:
        query: Câu truy vấn
        top_k: Số kết quả cuối cùng
        score_threshold: Ngưỡng cosine GỐC để quyết định fallback (giống Task 9)
        use_hyde: Bật worker HyDE (cần OPENROUTER_API_KEY)
        use_expansion: Bật worker query expansion (cần OPENROUTER_API_KEY)
        verbose: In số lượng kết quả từng worker

    Returns:
        List of {'content', 'score', 'metadata', 'source'} — giống hợp đồng của Task 9.
    """
    fetch_k = top_k * 2

    # Nạp sẵn embedding model TRƯỚC khi mở thread: nhiều thread cùng gọi _get_model()
    # lần đầu sẽ cùng khởi tạo bge-m3 song song (tốn RAM gấp nhiều lần và có thể race).
    _get_model()

    # --- Lớp 1: các worker luôn chạy ---
    tasks: list[tuple[str, callable]] = [
        ("dense", lambda: semantic_search(query, top_k=fetch_k)),
        ("sparse", lambda: lexical_search(query, top_k=fetch_k)),
    ]

    # --- Lớp 2: worker HyDE ---
    if use_hyde:
        tasks.append(("dense_hyde", lambda: semantic_search(query, top_k=fetch_k, use_hyde=True)))

    # --- Lớp 3: worker cho từng biến thể câu hỏi ---
    # expand_query gọi LLM nên chạy trước, tuần tự — số biến thể quyết định số worker.
    if use_expansion:
        for i, variant in enumerate(expand_query(query), 1):
            tasks.append((
                f"dense_variant_{i}",
                lambda v=variant: semantic_search(v, top_k=fetch_k),
            ))
            tasks.append((
                f"sparse_variant_{i}",
                lambda v=variant: lexical_search(v, top_k=fetch_k),
            ))

    outputs = _run_workers(tasks)

    if verbose:
        for name, res in sorted(outputs.items()):
            print(f"  [worker] {name:20} -> {len(res)} ket qua")

    # --- Fusion: RRF trên mọi ranked list thu được ---
    ranked_lists = [res for res in outputs.values() if res]
    if not ranked_lists:
        return []

    merged = rerank_rrf(ranked_lists, top_k=fetch_k)
    for item in merged:
        item["source"] = "hybrid"

    final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)

    # --- Fallback: dựa trên điểm COSINE GỐC, không phải điểm RRF đã fuse ---
    # Lấy điểm cosine tốt nhất trong các worker dense (dense / dense_hyde / variant).
    dense_scores = [
        res[0]["score"]
        for name, res in outputs.items()
        if name.startswith("dense") and res
    ]
    best_score = max(dense_scores) if dense_scores else 0.0

    if best_score < score_threshold:
        print(f"  [WARNING] Diem cosine cao nhat ({best_score:.3f}) < nguong "
              f"({score_threshold:.3f}). Kich hoat PageIndex fallback.")
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                for item in fallback:
                    item.setdefault("source", "pageindex")
                return fallback[:top_k]
        except Exception as e:
            err_msg = str(e).encode("ascii", errors="ignore").decode("ascii")
            print(f"  [WARNING] PageIndex fallback khong kha dung: {err_msg}")

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Sinh viên đóng học phí online bằng cách nào?",
        "Trả sách trễ bị phạt bao nhiêu?",
        "xyzabc123nonsense",  # lạc đề → test fallback
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Query: {q}")
        print("=" * 70)
        results = supervisor_retrieve(q, top_k=3, verbose=True)
        for i, r in enumerate(results, 1):
            src = r.get("metadata", {}).get("source", "?")
            print(f"  {i}. [{r['score']:.4f}] [{r['source']}|{src}] {r['content'][:70]}...")
