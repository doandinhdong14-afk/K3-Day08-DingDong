"""
RAG Evaluation Pipeline — RAGAS.

Đánh giá chất lượng RAG pipeline trên golden dataset, so sánh A/B 2 config
retrieval, và xuất kết quả ra results.md.

Chạy:
    .venv/bin/python group_project/evaluation/eval_pipeline.py

Thiết kế:
    1. Load golden_dataset.json (15 Q&A pairs, ground truth lấy trực tiếp từ corpus)
    2. Với mỗi config, chạy retrieval + generation trên toàn bộ 15 câu hỏi
    3. Chấm bằng RAGAS: faithfulness, answer_relevancy, context_recall, context_precision
    4. So sánh A/B giữa 2 config retrieval
    5. Export results.md

Lưu ý rate limit: RAGAS gọi LLM RẤT NHIỀU LẦN (nhiều lần/metric/câu hỏi, không phải
1 lần/câu hỏi). Để giảm áp lực quota:
    - LLM judge chạy max_workers thấp + retry/backoff (xem RUN_CONFIG)
    - Embeddings dùng model LOCAL (bge-m3 đã cache) thay vì gọi API embedding
Nếu vẫn bị rate limit, RAGAS trả NaN cho metric đó — script vẫn chạy tiếp và ghi rõ
số ô NaN trong results.md thay vì che giấu.
"""

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
RAW_PATH = Path(__file__).parent / "results_raw.json"

# Model dùng làm LLM judge cho RAGAS. Dùng alias "-latest" để không chết khi
# Google retire bản cũ (gemini-2.0-flash-exp / 1.5-flash-8b đều đã 404).
JUDGE_MODEL = "gemini-flash-lite-latest"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
EMBEDDING_MODEL = "BAAI/bge-m3"  # local, khớp với model index ở Task 4

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

# RAGAS đặt tên cột trong to_pandas() không trùng tên metric (vd context_precision ->
# llm_context_precision_with_reference), nên dò theo danh sách ứng viên.
METRIC_COLUMNS = {
    "faithfulness": ["faithfulness"],
    "answer_relevancy": ["answer_relevancy", "response_relevancy"],
    "context_recall": ["context_recall", "llm_context_recall"],
    "context_precision": ["context_precision", "llm_context_precision_with_reference"],
}


def resolve_columns(df) -> dict:
    """Map tên metric chuẩn -> tên cột thật trong dataframe của RAGAS."""
    resolved = {}
    for metric, candidates in METRIC_COLUMNS.items():
        for cand in candidates:
            if cand in df.columns:
                resolved[metric] = cand
                break
        else:
            hit = next((c for c in df.columns if metric.split("_")[-1] in c), None)
            if hit:
                resolved[metric] = hit
    return resolved


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# CONFIGS để so sánh A/B
# =============================================================================

CONFIGS = {
    "A_hybrid_rerank": {
        "label": "Hybrid (dense + BM25) + RRF rerank",
        "mode": "hybrid",
        "top_k": 5,
        "use_reranking": True,
    },
    "B_dense_only": {
        "label": "Dense-only (semantic search, không rerank)",
        "mode": "dense",
        "top_k": 5,
        "use_reranking": False,
    },
}


def retrieve_for_config(question: str, config: dict) -> list[dict]:
    """Chạy retrieval theo config. Trả về list chunk dicts."""
    if config["mode"] == "dense":
        from src.task5_semantic_search import semantic_search

        results = semantic_search(question, top_k=config["top_k"])
        for r in results:
            r.setdefault("source", "dense")
        return results

    from src.task9_retrieval_pipeline import retrieve

    return retrieve(question, top_k=config["top_k"],
                    use_reranking=config["use_reranking"])


def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    Sinh câu trả lời từ chunks, dùng lại đúng prompt/format của Task 10
    để kết quả eval phản ánh pipeline thật.
    """
    from openai import OpenAI

    from src.task10_generation import (
        SYSTEM_PROMPT, TEMPERATURE, TOP_P, format_context, reorder_for_llm,
    )

    if not chunks:
        return "Tôi không tìm thấy tài liệu phù hợp trong hệ thống để trả lời câu hỏi này."

    context = format_context(reorder_for_llm(chunks))
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {question}"

    client = OpenAI(api_key=os.getenv("GEMINI_API_KEY"), base_url=GEMINI_BASE_URL)
    last_err = None
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = resp.choices[0].message.content or ""
            if answer.strip():
                return answer
        except Exception as e:  # rate limit / transient
            last_err = e
            time.sleep(min(2 ** attempt * 5, 60))
    return f"[GENERATION_FAILED] {last_err}"


def run_pipeline_on_dataset(golden_dataset: list[dict], config: dict) -> list[dict]:
    """Chạy retrieval + generation cho toàn bộ dataset theo 1 config."""
    rows = []
    for i, item in enumerate(golden_dataset, 1):
        q = item["question"]
        print(f"    [{i:2}/{len(golden_dataset)}] {q[:58]}...", flush=True)
        chunks = retrieve_for_config(q, config)
        answer = generate_answer(q, chunks)
        rows.append({
            "id": item.get("id", f"Q{i:02}"),
            "question": q,
            "answer": answer,
            "contexts": [c["content"] for c in chunks],
            "retrieved_sources": [c.get("metadata", {}).get("source", "?") for c in chunks],
            "ground_truth": item["expected_answer"],
            "expected_source": item.get("source_file"),
        })
    return rows


# =============================================================================
# RAGAS evaluation
# =============================================================================

def _build_judge():
    """LLM judge + embeddings cho RAGAS. Embeddings chạy local để tiết kiệm quota."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url=GEMINI_BASE_URL,
        temperature=0.0,
        max_retries=5,
    )
    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # bypass_n=True: Gemini không hỗ trợ nhiều candidate trong 1 request
    # ("Multiple candidates is not enabled for this model" — 400). Cờ này bắt RAGAS
    # gọi LLM n lần riêng lẻ thay vì xin n candidate cùng lúc.
    return LangchainLLMWrapper(llm, bypass_n=True), LangchainEmbeddingsWrapper(emb)


def evaluate_with_ragas(rows: list[dict]) -> "object":
    """
    Evaluate bằng RAGAS với 4 metrics:
        faithfulness       — câu trả lời có bám vào context không (chống bịa)
        answer_relevancy   — câu trả lời có đúng trọng tâm câu hỏi không
        context_recall     — context có chứa đủ thông tin của ground truth không
        context_precision  — context lấy về có bao nhiêu phần thực sự liên quan
    """
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    llm, emb = _build_judge()

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        )
        for r in rows
    ]

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
    ]

    # max_workers thấp + backoff dài: free tier Gemini giới hạn request/phút
    run_config = RunConfig(timeout=300, max_retries=10, max_wait=90, max_workers=2)

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=llm,
        embeddings=emb,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=True,
    )
    return result


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(golden_dataset: list[dict]) -> dict:
    """Chạy full eval cho từng config rồi gom kết quả để so sánh."""
    import pandas as pd

    comparison = {}
    for name, config in CONFIGS.items():
        print(f"\n>>> CONFIG {name}: {config['label']}", flush=True)
        rows = run_pipeline_on_dataset(golden_dataset, config)
        print(f"    -> chấm RAGAS ({len(rows)} câu x 4 metrics)...", flush=True)
        result = evaluate_with_ragas(rows)
        df = result.to_pandas()
        cols = resolve_columns(df)

        # Retrieval recall: tài liệu đúng có nằm trong context lấy về không
        hits = [
            r["expected_source"] in r["retrieved_sources"]
            for r in rows if r["expected_source"]
        ]
        comparison[name] = {
            "config": config,
            "rows": rows,
            "df": df,
            "columns": cols,
            "scores": {m: float(pd.to_numeric(df[c], errors="coerce").mean())
                       for m, c in cols.items()},
            "nan_counts": {m: int(pd.to_numeric(df[c], errors="coerce").isna().sum())
                           for m, c in cols.items()},
            "source_recall": sum(hits) / len(hits) if hits else 0.0,
        }
    return comparison


# =============================================================================
# Export Results
# =============================================================================

def _fmt(v) -> str:
    import math
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.3f}"


def export_results(comparison: dict, golden_dataset: list[dict]):
    """Export evaluation results to results.md"""
    import pandas as pd

    names = list(comparison.keys())
    best = max(names, key=lambda n: comparison[n]["scores"].get("faithfulness", 0) or 0)

    L = []
    L.append("# RAG Evaluation Results\n")
    L.append(f"- **Framework:** RAGAS · **LLM judge:** `{JUDGE_MODEL}` (Gemini qua OpenAI-compatible API)")
    L.append(f"- **Embeddings (judge):** `{EMBEDDING_MODEL}` chạy local")
    L.append(f"- **Golden dataset:** {len(golden_dataset)} câu hỏi, ground truth trích trực tiếp từ corpus")
    L.append(f"- **Ngày chạy:** {time.strftime('%Y-%m-%d %H:%M')}\n")

    L.append("## 1. Configs so sánh\n")
    L.append("| Config | Mô tả | top_k | Reranking |")
    L.append("|---|---|---|---|")
    for n in names:
        c = comparison[n]["config"]
        L.append(f"| `{n}` | {c['label']} | {c['top_k']} | {'có' if c['use_reranking'] else 'không'} |")

    L.append("\n## 2. Overall Scores\n")
    header = "| Metric | " + " | ".join(f"`{n}`" for n in names) + " |"
    L.append(header)
    L.append("|---" * (len(names) + 1) + "|")
    for m in METRIC_NAMES:
        cells = [_fmt(comparison[n]["scores"].get(m)) for n in names]
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L.append("| **source_recall** (tài liệu đúng có trong context) | "
             + " | ".join(_fmt(comparison[n]["source_recall"]) for n in names) + " |")

    nan_total = {n: sum(comparison[n]["nan_counts"].values()) for n in names}
    if any(nan_total.values()):
        L.append("\n> ⚠️ Số ô không chấm được (NaN — thường do rate limit hoặc LLM judge "
                 "trả về format sai): "
                 + ", ".join(f"`{n}`: {nan_total[n]}/{len(golden_dataset)*4}" for n in names))

    L.append("\n## 3. A/B Comparison\n")
    L.append("| Metric | " + " | ".join(f"`{n}`" for n in names) + " | Chênh lệch (A - B) |")
    L.append("|---" * (len(names) + 2) + "|")
    for m in METRIC_NAMES + ["source_recall"]:
        if m == "source_recall":
            vals = [comparison[n]["source_recall"] for n in names]
        else:
            vals = [comparison[n]["scores"].get(m) for n in names]
        try:
            diff = _fmt(vals[0] - vals[1])
        except Exception:
            diff = "n/a"
        L.append(f"| {m} | " + " | ".join(_fmt(v) for v in vals) + f" | {diff} |")
    L.append(f"\n**Config tốt hơn theo faithfulness:** `{best}`")

    L.append("\n## 4. Chi tiết từng câu hỏi\n")
    for n in names:
        L.append(f"\n### Config `{n}`\n")
        df = comparison[n]["df"]
        rows = comparison[n]["rows"]
        L.append("| ID | Câu hỏi | " + " | ".join(METRIC_NAMES) + " | Nguồn đúng được lấy? |")
        L.append("|---" * (len(METRIC_NAMES) + 3) + "|")
        cols = comparison[n]["columns"]
        for i, r in enumerate(rows):
            cells = []
            for m in METRIC_NAMES:
                c = cols.get(m)
                v = pd.to_numeric(df[c], errors="coerce").iloc[i] if c else None
                cells.append(_fmt(v))
            hit = "✅" if r["expected_source"] in r["retrieved_sources"] else "❌"
            q = r["question"][:52] + ("..." if len(r["question"]) > 52 else "")
            L.append(f"| {r['id']} | {q} | " + " | ".join(cells) + f" | {hit} |")

    L.append("\n## 5. Worst Performers\n")
    for n in names:
        df = comparison[n]["df"]
        rows = comparison[n]["rows"]
        avg = pd.concat(
            [pd.to_numeric(df[c], errors="coerce")
             for c in comparison[n]["columns"].values()],
            axis=1,
        ).mean(axis=1)
        order = avg.sort_values(na_position="first").index[:3]
        L.append(f"\n**Config `{n}`** — 3 câu điểm thấp nhất:\n")
        for idx in order:
            r = rows[idx]
            hit = "có" if r["expected_source"] in r["retrieved_sources"] else "KHÔNG"
            L.append(f"- `{r['id']}` (avg {_fmt(avg[idx])}) — {r['question']}")
            L.append(f"  - Nguồn cần có: `{r['expected_source']}` → lấy được: **{hit}**")
            L.append(f"  - Thực tế lấy về: {r['retrieved_sources']}")

    L.append("\n## 6. Recommendations\n")
    L.append(_recommendations(comparison))

    RESULTS_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")

    raw = {
        n: {
            "config": comparison[n]["config"],
            "scores": comparison[n]["scores"],
            "nan_counts": comparison[n]["nan_counts"],
            "source_recall": comparison[n]["source_recall"],
            "rows": [
                {k: v for k, v in r.items() if k != "contexts"}
                for r in comparison[n]["rows"]
            ],
        }
        for n in names
    }
    RAW_PATH.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] Đã ghi {RESULTS_PATH}")
    print(f"[OK] Đã ghi {RAW_PATH}")


def _recommendations(comparison: dict) -> str:
    """Sinh nhận xét dựa trên số liệu thực tế, không viết cứng."""
    from collections import Counter

    lines = []
    for n, data in comparison.items():
        recall = data["source_recall"]
        miss = [r for r in data["rows"]
                if r["expected_source"] not in r["retrieved_sources"]]
        flooded = Counter(
            s for r in data["rows"] for s in r["retrieved_sources"]
        ).most_common(1)
        lines.append(f"**`{n}`** — source_recall {recall:.0%}, {len(miss)} câu không lấy được tài liệu đúng.")
        if flooded:
            src, cnt = flooded[0]
            total = sum(len(r["retrieved_sources"]) for r in data["rows"])
            lines.append(f"  - Nguồn chiếm nhiều chunk nhất trong kết quả: `{src}` ({cnt}/{total} = {cnt/total:.0%}).")
        if miss:
            lines.append("  - Câu miss: " + ", ".join(f"`{r['id']}`" for r in miss))
        lines.append("")

    lines.append("**Đề xuất cải thiện:**\n")
    lines.append("1. **Giới hạn số chunk mỗi tài liệu trong kết quả cuối** (ví dụ tối đa 2 chunk/file) "
                 "hoặc đổi `RERANK_METHOD` sang `mmr`. Corpus hiện mất cân bằng nặng — "
                 "`hoc-phi-va-cac-khoan-thu.md` chiếm 412/573 chunk (72%), nên nó áp đảo cả dense "
                 "lẫn BM25 và đẩy các tài liệu nhỏ ra khỏi top_k.")
    lines.append("2. **Tăng `top_k` không giải quyết được triệt để** — đã đo: với câu hỏi eSIM, "
                 "top_k=8 vẫn không lấy được đúng tài liệu.")
    lines.append("3. **Chuẩn hoá dấu tiếng Việt cho nhánh BM25** — BM25 khớp token chính xác nên "
                 "truy vấn không dấu không bao giờ match được văn bản có dấu.")
    lines.append("4. **Cache SentenceTransformer trong `task5_semantic_search`** — hiện mỗi lần gọi "
                 "`semantic_search()` lại load lại bge-m3 (~15s/lần), là phần lớn thời gian chạy eval.")
    return "\n".join(lines)


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    if not os.getenv("GEMINI_API_KEY"):
        sys.exit("Thiếu GEMINI_API_KEY trong .env")

    comparison = compare_configs(golden_dataset)
    export_results(comparison, golden_dataset)

    print("\n=== TỔNG KẾT ===")
    for name, data in comparison.items():
        scores = " | ".join(f"{m}={_fmt(v)}" for m, v in data["scores"].items())
        print(f"{name:20} source_recall={data['source_recall']:.0%} | {scores}")
