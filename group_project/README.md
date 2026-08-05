# Bài Tập Nhóm — University Services RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1: Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về dịch vụ và chính sách đại học liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

Xem code mẫu (DeepEval/RAGAS/TruLens) chi tiết trong `README.md` gốc mục "Yêu cầu 2".

### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```
   Streamlit  app.py
        │  câu hỏi + lịch sử chat
        ▼
   Task 10  condense_question()          ← biến câu follow-up thành câu độc lập
        │
        ▼
   Task 9  retrieve()  ──────────────────────────────────────────┐
        │                                                        │
        ├─ Task 5  Semantic search  (bge-m3 → ChromaDB, cosine)  │
        ├─ Task 6  Lexical search   (BM25 Okapi)                 │  song song ở
        ├─ Task 5  HyDE             (tài liệu giả định)          │  supervisor.py
        └─ Query Expansion          (n biến thể câu hỏi)         │
                             │                                   │
                             ▼                                   │
        Task 7  RRF fusion  Σ 1/(60+rank)  →  Rerank             │
                             │                                   │
              cosine gốc < 0.48 ? ──► Task 8  PageIndex vectorless
                             │                                   │
                             ▼                                   │
                     top_k chunks  ◄──────────────────────────────┘
                             │
                             ▼
   Task 10  reorder_for_llm() [1,3,5,4,2]  →  prompt + citation  →  LLM
                             │
                             ▼
   Streamlit: câu trả lời + expander "Nguồn tham khảo" (file, score, nhánh retrieval)
```

Chi tiết đầy đủ (ingestion + evaluation) xem `README.md` ở thư mục gốc, mục
**Kiến Trúc Hệ Thống**.

---

## Phân Công Công Việc

> Suy ra từ lịch sử commit. **Cột MSSV cần các thành viên tự điền trước khi nộp.**

| Thành viên (GitHub) | MSSV | Vai trò | Nhiệm vụ | Trạng thái |
|---|---|---|---|---|
| Trần Hoài Nam | 2A202601751 | Role 2 — Data & Pipeline | Task 1, 4, 7, 9 + tích hợp Streamlit | ✅ |
| Bùi Tiến Phát | 2A202601861 | Role 3 — Retrieval & Fallback | Task 2, 5 (HyDE), 7 (RRF), 8 (PageIndex) | ✅ |
| `Dienamyte` | _(điền)_ | Role 4 — Evaluation & QA | `golden_dataset.json`, `eval_pipeline.py`, `results.md` | ✅ |
| `doandinhdong14-afk` | _(điền)_ | Role 1 — Team Leader & Architect | `supervisor.py`, Task 10, ghép code, rà soát | ✅ |

---

## Hướng Dẫn Chạy

```bash
# Cài đặt dependencies (Python 3.11)
pip install -r requirements.txt

# Build vector store — BẮT BUỘC trước khi mở app
python -m src.task4_chunking_indexing

# Chạy chatbot
streamlit run app.py

# Chạy evaluation RAGAS (cần GEMINI_API_KEY trong .env)
python -m group_project.evaluation.eval_pipeline
```

---

## Lưu ý

Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
