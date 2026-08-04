"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"

Gợi ý LLM: OpenRouter có nhiều model gắn hậu tố ":free" không tính phí — xem
https://openrouter.ai/models?max_price=0 — phù hợp nếu chưa có credit trả phí.
Base URL: "https://openrouter.ai/api/v1", dùng chung interface với OpenAI SDK.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

try:
    from src.task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 3 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 3

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.8 vì: trung tính, không quá sáng tạo (0.9-1.0) nhưng cũng không quá an toàn (0.5-0.7)
TOP_P = 0.8

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# Model dùng khi đi qua OpenRouter. Có thể đổi sang model ":free"
# (vd "google/gemini-2.0-flash-exp:free") nếu tài khoản chưa nạp credit.
LLM_MODEL = "openai/gpt-4o-mini"

# Số lượt hội thoại gần nhất được đưa vào prompt. Giữ nhỏ vì:
#   - context dài làm loãng phần evidence vừa retrieve được
#   - câu hỏi follow-up hầu như chỉ tham chiếu tới 1-2 lượt ngay trước
MAX_HISTORY_TURNS = 4


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về dịch vụ và chính sách đại học
(học phí, học bổng, ký túc xá, thư viện, đăng ký học phần).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Mỗi khẳng định phải có trích dẫn ngay sau, ví dụ: [Tuition Fees, 2026]
3. Nếu context không đủ thông tin → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
4. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo đoạn văn
5. Không suy luận hay mở rộng ngoài những gì được nêu trong context"""


# =============================================================================
# LLM PROVIDER — chọn provider khả dụng theo thứ tự ưu tiên
# =============================================================================

def _is_usable(key: str) -> bool:
    """Loại bỏ placeholder còn sót lại từ .env.example."""
    return bool(key) and not key.startswith("your_") and "..." not in key and len(key) > 10


def resolve_llm():
    """
    Chọn LLM provider khả dụng.

    Returns:
        (client, [model_candidates], provider_name) hoặc None nếu không có key nào.

    Thứ tự ưu tiên Gemini → OpenRouter → OpenAI vì free tier của Gemini rộng nhất,
    phù hợp bài lab. Gemini khai báo nhiều model vì Google thường xuyên retire bản cũ
    (gemini-2.0-* / 1.5-* đã hết vòng đời hoặc free-tier quota = 0) — alias "-latest"
    tránh 404 khi điều đó xảy ra.
    """
    from openai import OpenAI

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if _is_usable(gemini_key):
        client = OpenAI(
            api_key=gemini_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        return client, ["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-2.5-flash"], "Gemini"

    if _is_usable(openrouter_key):
        client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
        return client, [LLM_MODEL], "OpenRouter"

    if _is_usable(openai_key):
        return OpenAI(api_key=openai_key), ["gpt-4o-mini"], "OpenAI"

    return None


def _chat(messages: list[dict], temperature: float = TEMPERATURE, top_p: float = TOP_P) -> str:
    """
    Gọi LLM với danh sách model dự phòng. Trả về "" nếu mọi model đều lỗi
    (hết quota, 429, model bị gỡ) — caller tự quyết định hiển thị gì.
    """
    resolved = resolve_llm()
    if not resolved:
        return ""

    client, model_candidates, provider = resolved
    for model_name in model_candidates:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
            )
            answer = (response.choices[0].message.content or "").strip()
            if answer:
                return answer
        except Exception as e:
            err_msg = str(e).encode("ascii", errors="ignore").decode("ascii")
            print(f"  [Warning] {provider}/{model_name} error: {err_msg}")

    return ""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)
    """
    if len(chunks) <= 2:
        return chunks

    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# CONVERSATION MEMORY — hỗ trợ câu hỏi follow-up
# =============================================================================

def condense_question(query: str, history: list[dict] | None) -> str:
    """
    Viết lại câu hỏi follow-up thành câu hỏi ĐỘC LẬP để đưa vào retrieval.

    Vì sao cần bước này: retrieval không có trí nhớ. Câu "Thế còn học bổng thì sao?"
    khi embed một mình sẽ không match được tài liệu nào có ý nghĩa — phải ghép ngữ
    cảnh từ lượt trước thành "Điều kiện nhận học bổng của RMIT là gì?" rồi mới search.

    Không có API key → fallback ghép câu hỏi gần nhất của user với câu hỏi hiện tại.
    Thô nhưng vẫn tốt hơn nhiều so với embed mỗi đại từ trỏ ngược.
    """
    if not history:
        return query

    recent = history[-MAX_HISTORY_TURNS * 2:]
    transcript = "\n".join(
        f"{'Người dùng' if m['role'] == 'user' else 'Trợ lý'}: {m['content'][:400]}"
        for m in recent
    )

    rewritten = _chat(
        [
            {
                "role": "system",
                "content": (
                    "Viết lại câu hỏi cuối cùng của người dùng thành một câu hỏi ĐỘC LẬP, "
                    "tự hiểu được mà không cần đọc lịch sử hội thoại. Thay mọi đại từ/tham "
                    "chiếu ngược bằng danh từ cụ thể. CHỈ trả về câu hỏi đã viết lại, "
                    "không giải thích, không thêm gì khác."
                ),
            },
            {
                "role": "user",
                "content": f"Lịch sử hội thoại:\n{transcript}\n\nCâu hỏi cần viết lại: {query}",
            },
        ],
        temperature=0.0,
    )

    if rewritten and len(rewritten) < 500:
        return rewritten

    # Fallback không cần LLM: ghép câu hỏi user gần nhất làm ngữ cảnh.
    # `history` không chứa câu hỏi hiện tại, nên message 'user' cuối cùng trong đó
    # chính là câu hỏi ở lượt trước.
    last_user = next(
        (m["content"] for m in reversed(history) if m.get("role") == "user"),
        "",
    )
    return f"{last_user} {query}".strip() if last_user else query


def _build_messages(query: str, context: str, history: list[dict] | None) -> list[dict]:
    """Ghép system prompt + lịch sử hội thoại rút gọn + context của lượt hiện tại."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        for m in history[-MAX_HISTORY_TURNS * 2:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                messages.append({"role": m["role"], "content": m["content"][:1500]})

    messages.append(
        {"role": "user", "content": f"Context:\n{context}\n\n---\n\nQuestion: {query}"}
    )
    return messages


# =============================================================================
# GENERATION
# =============================================================================

def _dump_chunks(chunks: list[dict], banner: str) -> str:
    """Hiển thị nguyên văn chunks khi không gọi được LLM — vẫn có ích để demo retrieval."""
    lines = [banner + "\n"]
    for i, c in enumerate(chunks, 1):
        source = c.get("metadata", {}).get("source", "")
        lines.append(f"**[Tài liệu {i} - {source}]:**\n{c['content']}\n")
    return "\n".join(lines)


def generate_answer_from_chunks(
    query: str,
    chunks: list[dict],
    history: list[dict] | None = None,
) -> str:
    """
    Sinh câu trả lời có citation từ một tập chunks CÓ SẴN.

    Tách riêng khỏi `generate_with_citation()` để các nhánh retrieval khác
    (vd `src/supervisor.py`, app.py ở chế độ Supervisor) dùng lại đúng
    SYSTEM_PROMPT / reorder / tham số sinh, thay vì tự chế prompt riêng.

    Trả về "" nếu không gọi được LLM — caller tự quyết định hiển thị gì.
    """
    if not chunks:
        return ""
    context = format_context(reorder_for_llm(chunks))
    return _chat(_build_messages(query, context, history))


def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    history: list[dict] | None = None,
) -> dict:
    """
    End-to-end RAG generation có citation, hỗ trợ hội thoại nhiều lượt.

    Args:
        query: Câu hỏi của người dùng ở lượt hiện tại
        top_k: Số chunks đưa vào context
        history: Lịch sử hội thoại [{'role': 'user'|'assistant', 'content': str}, ...]
                 KHÔNG bao gồm câu hỏi hiện tại. None = hội thoại 1 lượt.

    Returns:
        {
            'answer': str,               # câu trả lời có citation
            'sources': list[dict],       # chunks đã dùng làm evidence
            'retrieval_source': str,     # 'hybrid' | 'pageindex' | 'none'
            'search_query': str,         # câu hỏi thực sự dùng để retrieval
        }
    """
    # Bước 0: viết lại câu hỏi follow-up thành câu hỏi độc lập trước khi retrieval
    search_query = condense_question(query, history)

    chunks = retrieve(search_query, top_k=top_k)
    if not chunks:
        return {
            "answer": "Tôi không tìm thấy tài liệu phù hợp trong hệ thống để trả lời câu hỏi này.",
            "sources": [],
            "retrieval_source": "none",
            "search_query": search_query,
        }

    # Bước 1-4: reorder chống lost-in-the-middle → format context → prompt → LLM
    answer = generate_answer_from_chunks(query, chunks, history)

    # Bước 5: không gọi được LLM → vẫn trả về evidence đã retrieve, nói rõ lý do
    if not answer:
        banner = (
            "⚠️ **[Chưa cấu hình GEMINI_API_KEY / OPENROUTER_API_KEY trong .env — "
            "hiển thị tài liệu trích xuất từ RAG]:**"
            if resolve_llm() is None
            else "⚠️ **[LLM không phản hồi (rate limit / hết quota) — "
                 "hiển thị tài liệu trích xuất từ RAG]:**"
        )
        answer = _dump_chunks(chunks, banner)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
        "search_query": search_query,
    }


if __name__ == "__main__":
    test_queries = [
        "Học phí tại RMIT Vietnam là bao nhiêu?",
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Sinh viên quốc tế có những học bổng nào?",
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")

    # Demo hội thoại nhiều lượt: câu 2 chỉ hiểu được nhờ ngữ cảnh câu 1
    print(f"\n{'=' * 70}")
    print("DEMO FOLLOW-UP (conversation memory)")
    print("=" * 70)
    first = generate_with_citation("Trả sách trễ ở thư viện bị phạt bao nhiêu?")
    print(f"\nQ1: Trả sách trễ ở thư viện bị phạt bao nhiêu?\nA1: {first['answer'][:300]}...")

    history = [
        {"role": "user", "content": "Trả sách trễ ở thư viện bị phạt bao nhiêu?"},
        {"role": "assistant", "content": first["answer"]},
    ]
    second = generate_with_citation("Thế còn phí in ấn thì sao?", history=history)
    print(f"\nQ2: Thế còn phí in ấn thì sao?")
    print(f"    -> câu hỏi dùng để retrieval: {second['search_query']}")
    print(f"A2: {second['answer'][:300]}...")
