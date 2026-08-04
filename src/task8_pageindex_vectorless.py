import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents trong data/standardized/ lên PageIndex.
    """
    if not PAGEINDEX_API_KEY or "your_pageindex" in PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY chưa được cấu hình trong file .env.")
        return

    try:
        from pageindex import PageIndex

        pi = PageIndex(api_key=PAGEINDEX_API_KEY)

        if not STANDARDIZED_DIR.exists():
            print(f"⚠ Thư mục {STANDARDIZED_DIR} không tồn tại.")
            return

        # Tìm toàn bộ file .md trong standardized directory
        md_files = list(STANDARDIZED_DIR.glob("**/*.md"))
        if not md_files:
            print(f"⚠ Không tìm thấy file .md nào trong {STANDARDIZED_DIR}")
            return

        print(f"🔄 Đang upload {len(md_files)} tài liệu lên PageIndex...")

        for filepath in md_files:
            try:
                # Upload từng document lên PageIndex Cloud Index
                res = pi.upload_document(file_path=str(filepath))
                print(f"   ✓ Uploaded: {filepath.name} (Doc ID: {res.get('doc_id', 'N/A')})")
            except Exception as e:
                print(f"   ❌ Lỗi khi upload {filepath.name}: {e}")

        print("✓ Tải lên tài liệu hoàn tất!")

    except ImportError:
        print("❌ Chưa cài đặt thư viện pageindex. Hãy chạy: pip install pageindex")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo PageIndex: {e}")


def pageindex_search(query: str, top_k: int = 5, debug: bool = False) -> List[Dict[str, Any]]:
    """
    Vectorless retrieval sử dụng PageIndex API.
    Dùng làm fallback khi hybrid search không trả về kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa
        debug: Nếu True, sẽ in raw JSON response nhận được từ PageIndex

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # marker để Task 9 / UI biết kết quả đến từ nhánh fallback
        }
    """
    if not PAGEINDEX_API_KEY or "your_pageindex" in PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY không khả dụng.")
        return []

    try:
        from pageindex import PageIndex

        pi = PageIndex(api_key=PAGEINDEX_API_KEY)

        # Gửi retrieval query tới PageIndex SDK
        response = pi.retrieval(query=query, top_k=top_k)

        # In response thực tế dạng JSON nếu bật flag debug
        if debug:
            print("\n--- DEBUG PAGEINDEX RAW RESPONSE ---")
            print(json.dumps(response, indent=2, ensure_ascii=False))
            print("------------------------------------\n")

        results = []

        # Case 1: Parse response theo schema mới/mặc định (retrieved_nodes)
        retrieved_nodes = response.get("retrieved_nodes", [])
        for node in retrieved_nodes:
            # Struct: relevant_contents = list[list[{section_title, relevant_content}]]
            rel_contents_groups = node.get("relevant_contents", [])
            doc_id = node.get("doc_id", "Unknown")
            node_score = float(node.get("score", 1.0))

            for group in rel_contents_groups:
                for item in group:
                    sec_title = item.get("section_title", "")
                    content_text = item.get("relevant_content", "")

                    if content_text:
                        full_content = f"### {sec_title}\n{content_text}" if sec_title else content_text
                        results.append(
                            {
                                "content": full_content.strip(),
                                "score": round(node_score, 4),
                                "metadata": {
                                    "doc_id": doc_id,
                                    "section_title": sec_title,
                                    "source": doc_id,
                                    "type": "pageindex_vectorless",
                                },
                                "source": "pageindex",
                            }
                        )

        # Case 2: Parse fallback nếu API trả về danh sách 'results' phẳng
        if not results and "results" in response:
            for item in response.get("results", []):
                results.append(
                    {
                        "content": item.get("content", item.get("text", "")),
                        "score": float(item.get("score", 0.0)),
                        "metadata": item.get("metadata", {"source": "pageindex"}),
                        "source": "pageindex",
                    }
                )

        # Sắp xếp lại theo điểm số giảm dần và lấy top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    except ImportError:
        print("❌ Chưa cài đặt thư viện pageindex. Hãy chạy: pip install pageindex")
        return []
    except Exception as e:
        print(f"❌ Lỗi khi thực thi pageindex_search: {e}")
        return []


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("=" * 50)
        print("Task 8: PageIndex Vectorless RAG Test")
        print("=" * 50)

        # 1. Upload tài liệu
        upload_documents()

        # 2. Test Query (bật debug=True để kiểm tra JSON thực tế)
        test_query = "tuition fee payment methods"
        print(f"\n🔍 Executing query: '{test_query}'")
        results = pageindex_search(test_query, top_k=3, debug=True)

        print(f"\n--- Tìm thấy {len(results)} kết quả ---")
        for i, r in enumerate(results, 1):
            print(f"[{i}] Score: {r['score']:.3f} | Doc ID: {r['metadata'].get('doc_id', 'N/A')}")
            print(f"    Content: {r['content'][:150]}...\n")