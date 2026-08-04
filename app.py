"""
RAG Chatbot — University Services.

Streamlit app nối RAG Retrieval (Task 9 / Supervisor) và Generation (Task 10).

Chạy:
    streamlit run app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="University Services RAG Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# HELPERS
# =============================================================================

def render_sources(sources: list[dict], key_prefix: str):
    """
    Hiển thị danh sách chunks đã dùng làm evidence: tên file, loại, điểm số,
    và thanh score trực quan để thấy ngay chunk nào đóng góp nhiều nhất.
    """
    if not sources:
        return

    max_score = max((s.get("score", 0) or 0) for s in sources) or 1.0

    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)"):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {})
            source_name = meta.get("source", "Unknown")
            doc_type = meta.get("type", "unknown")
            score = src.get("score", 0) or 0
            retrieval_branch = src.get("source", "hybrid")

            badge = "🔀 hybrid" if retrieval_branch == "hybrid" else "📄 pageindex"
            st.markdown(
                f"**[{i}] {source_name}** &nbsp;`{doc_type}`&nbsp; {badge} &nbsp;|&nbsp; "
                f"score: `{score:.4f}`"
            )
            st.progress(min(1.0, score / max_score))
            st.text(src.get("content", "")[:400] + "...")
            st.divider()


def build_history(messages: list[dict]) -> list[dict]:
    """
    Chuyển lịch sử chat của Streamlit thành format hội thoại cho Task 10.
    Bỏ metadata phụ (sources, timing) — LLM chỉ cần role + content.
    """
    return [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]


# =============================================================================
# SIDEBAR — INFO & SETTINGS
# =============================================================================

with st.sidebar:
    st.title("🎓 University Services RAG")
    st.caption(
        "Trợ lý hỏi đáp về dịch vụ và chính sách đại học "
        "(học phí, học bổng, thư viện, đăng ký học phần, sinh viên quốc tế)"
    )

    st.divider()

    st.subheader("💡 Câu hỏi gợi ý")
    # Các câu hỏi bám đúng corpus đang index (tài liệu công khai của RMIT Vietnam).
    suggestions = [
        "Sinh viên đóng học phí trực tuyến bằng những cách nào?",
        "Trả sách trễ hạn ở thư viện bị phạt bao nhiêu?",
        "Điều kiện duy trì học bổng RMIT là gì?",
        "Mỗi học kỳ được đăng ký tối thiểu bao nhiêu tín chỉ?",
        "Sinh viên quốc tế mua eSIM ở đâu?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{s[:24]}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.subheader("⚙️ Thiết lập")
    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, 5)
    use_memory = st.toggle(
        "Ghi nhớ hội thoại (follow-up)",
        value=True,
        help="Viết lại câu hỏi follow-up thành câu hỏi độc lập trước khi retrieval. "
             "Tắt đi thì mỗi câu hỏi được xử lý độc lập.",
    )
    use_supervisor = st.toggle(
        "Supervisor mode (workers song song)",
        value=False,
        help="Chạy thêm worker HyDE + query expansion song song rồi gộp bằng RRF. "
             "Chất lượng tốt hơn nhưng chậm hơn và tốn thêm lượt gọi LLM.",
    )

    if st.button("🗑️ Xoá lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("**Kiến trúc hệ thống:**")
    st.caption(
        "Hybrid Retrieval (Semantic bge-m3 + BM25) → RRF Rerank → "
        "PageIndex Fallback (cosine < 0.48) → LLM Generation có Citation"
    )

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.title("🎓 University Services RAG Chatbot")
st.caption("Hệ thống hỏi đáp thông tin dịch vụ đại học (Học phí, Học bổng, Thư viện, Đăng ký học phần)")

# Hiển thị lịch sử chat
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("rewritten_query"):
                st.caption(f"🔎 Câu hỏi dùng để tìm kiếm: _{msg['rewritten_query']}_")
            render_sources(msg.get("sources", []), key_prefix=f"hist_{idx}")

# =============================================================================
# QUERY HANDLING
# =============================================================================

user_input = st.chat_input("Nhập câu hỏi của bạn về chính sách/dịch vụ đại học...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    # Lịch sử TRƯỚC câu hỏi hiện tại — dùng để viết lại câu hỏi follow-up
    history = build_history(st.session_state.messages) if use_memory else None

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
            started = time.time()
            rewritten_query = None
            try:
                from src.task10_generation import (
                    condense_question, generate_answer_from_chunks, generate_with_citation,
                )

                if use_supervisor:
                    # Nhánh Supervisor: retrieval song song, generation vẫn dùng lại
                    # đúng SYSTEM_PROMPT / reorder / tham số sinh của Task 10.
                    from src.supervisor import supervisor_retrieve

                    search_query = condense_question(query, history) if use_memory else query
                    sources = supervisor_retrieve(search_query, top_k=top_k)
                    if sources:
                        answer = generate_answer_from_chunks(query, sources, history) or (
                            "⚠️ LLM không phản hồi (chưa có API key hoặc hết quota). "
                            "Xem trực tiếp tài liệu đã truy xuất ở mục Nguồn tham khảo bên dưới."
                        )
                    else:
                        answer = "Tôi không tìm thấy tài liệu phù hợp trong hệ thống để trả lời câu hỏi này."
                    rewritten_query = search_query if search_query != query else None
                else:
                    response = generate_with_citation(query, top_k=top_k, history=history)
                    answer = response.get("answer", "Chưa thể khởi tạo câu trả lời.")
                    sources = response.get("sources", [])
                    search_query = response.get("search_query", query)
                    rewritten_query = search_query if search_query != query else None
            except Exception as e:
                answer = f"❌ **Lỗi khi chạy RAG pipeline:** `{e}`"
                sources = []

            elapsed = time.time() - started

        st.markdown(answer)
        if rewritten_query:
            st.caption(f"🔎 Câu hỏi dùng để tìm kiếm: _{rewritten_query}_")
        st.caption(f"⏱️ {elapsed:.1f}s · {len(sources)} chunks")
        render_sources(sources, key_prefix="live")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "rewritten_query": rewritten_query,
    })
