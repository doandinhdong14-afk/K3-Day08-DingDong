# RAG Evaluation Results

- **Framework:** RAGAS · **LLM judge:** `gemini-flash-lite-latest` (Gemini qua OpenAI-compatible API)
- **Embeddings (judge):** `BAAI/bge-m3` chạy local
- **Golden dataset:** 15 câu hỏi, ground truth trích trực tiếp từ corpus
- **Ngày chạy:** 2026-08-04 21:01

## 1. Configs so sánh

| Config | Mô tả | top_k | Reranking |
|---|---|---|---|
| `A_hybrid_rerank` | Hybrid (dense + BM25) + RRF rerank | 5 | có |
| `B_dense_only` | Dense-only (semantic search, không rerank) | 5 | không |

## 2. Overall Scores

| Metric | `A_hybrid_rerank` | `B_dense_only` |
|---|---|---|
| faithfulness | 0.661 | 0.643 |
| answer_relevancy | 0.656 | 0.678 |
| context_recall | 0.679 | 0.667 |
| context_precision | 0.458 | 0.479 |
| **source_recall** (tài liệu đúng có trong context) | 0.933 | 1.000 |

> ⚠️ Số ô không chấm được (NaN — thường do rate limit hoặc LLM judge trả về format sai): `A_hybrid_rerank`: 5/60, `B_dense_only`: 2/60

## 3. A/B Comparison

| Metric | `A_hybrid_rerank` | `B_dense_only` | Chênh lệch (A - B) |
|---|---|---|---|
| faithfulness | 0.661 | 0.643 | 0.018 |
| answer_relevancy | 0.656 | 0.678 | -0.022 |
| context_recall | 0.679 | 0.667 | 0.012 |
| context_precision | 0.458 | 0.479 | -0.020 |
| source_recall | 0.933 | 1.000 | -0.067 |

**Config tốt hơn theo faithfulness:** `A_hybrid_rerank`

## 4. Chi tiết từng câu hỏi


### Config `A_hybrid_rerank`

| ID | Câu hỏi | faithfulness | answer_relevancy | context_recall | context_precision | Nguồn đúng được lấy? |
|---|---|---|---|---|---|---|
| Q01 | Sinh viên RMIT có thể thanh toán học phí trực tuyến ... | 1.000 | n/a | 1.000 | 0.500 | ✅ |
| Q02 | Khi chuyển khoản đóng học phí thì cần ghi những thôn... | 1.000 | 0.972 | 1.000 | 1.000 | ✅ |
| Q03 | Trả sách trễ hạn ở thư viện RMIT bị phạt bao nhiêu t... | 1.000 | 0.881 | 1.000 | 1.000 | ✅ |
| Q04 | Nếu sinh viên rút môn học sau tuần thứ tư của học kỳ... | 1.000 | 0.966 | 1.000 | 1.000 | ✅ |
| Q05 | Sinh viên có thể thanh toán học phí bằng thẻ tín dụn... | 1.000 | 0.972 | 1.000 | 0.333 | ✅ |
| Q06 | Sinh viên nhận học bổng RMIT phải duy trì GPA tối th... | n/a | n/a | n/a | n/a | ✅ |
| Q07 | Người nhận học bổng phải đăng ký tối thiểu bao nhiêu... | 1.000 | 0.933 | 1.000 | 0.333 | ✅ |
| Q08 | Tiền trợ cấp hàng tháng của học bổng được chuyển vào... | 0.250 | 0.000 | 0.500 | 0.000 | ✅ |
| Q09 | Học bổng RMIT có được quy đổi thành tiền mặt hoặc ch... | 0.000 | 0.000 | 0.000 | 0.000 | ❌ |
| Q10 | Giờ hoạt động của khuôn viên cơ sở Nam Sài Gòn là mấ... | 0.000 | 0.000 | 0.000 | 0.000 | ✅ |
| Q11 | Active Hub ở tòa nhà 10 mở cửa vào những khung giờ n... | 1.000 | 0.963 | 1.000 | 1.000 | ✅ |
| Q12 | Sinh viên quốc tế có thể mua eSIM từ những nhà cung ... | 0.000 | 0.934 | 0.000 | 0.000 | ✅ |
| Q13 | Sau khi mua eSIM online và nhận được mã QR thì có nê... | 1.000 | 0.914 | 0.500 | 0.250 | ✅ |
| Q14 | Mỗi học kỳ sinh viên phải đăng ký tối thiểu và tối đ... | 1.000 | 0.996 | 1.000 | 1.000 | ✅ |
| Q15 | Điều kiện để được rút bớt học phần khi lớp đã ở trạn... | 0.000 | 0.000 | 0.500 | 0.000 | ✅ |

### Config `B_dense_only`

| ID | Câu hỏi | faithfulness | answer_relevancy | context_recall | context_precision | Nguồn đúng được lấy? |
|---|---|---|---|---|---|---|
| Q01 | Sinh viên RMIT có thể thanh toán học phí trực tuyến ... | 1.000 | 1.000 | 1.000 | 0.500 | ✅ |
| Q02 | Khi chuyển khoản đóng học phí thì cần ghi những thôn... | 1.000 | 0.821 | 1.000 | 1.000 | ✅ |
| Q03 | Trả sách trễ hạn ở thư viện RMIT bị phạt bao nhiêu t... | 1.000 | 0.868 | 1.000 | 1.000 | ✅ |
| Q04 | Nếu sinh viên rút môn học sau tuần thứ tư của học kỳ... | 1.000 | 0.974 | 1.000 | n/a | ✅ |
| Q05 | Sinh viên có thể thanh toán học phí bằng thẻ tín dụn... | 0.000 | 0.000 | 0.000 | 0.000 | ✅ |
| Q06 | Sinh viên nhận học bổng RMIT phải duy trì GPA tối th... | 0.000 | 0.000 | 0.000 | 0.000 | ✅ |
| Q07 | Người nhận học bổng phải đăng ký tối thiểu bao nhiêu... | 1.000 | 0.861 | 1.000 | 0.500 | ✅ |
| Q08 | Tiền trợ cấp hàng tháng của học bổng được chuyển vào... | 0.000 | 0.970 | 0.500 | 0.200 | ✅ |
| Q09 | Học bổng RMIT có được quy đổi thành tiền mặt hoặc ch... | 0.000 | 0.000 | 0.000 | 0.000 | ✅ |
| Q10 | Giờ hoạt động của khuôn viên cơ sở Nam Sài Gòn là mấ... | 1.000 | 0.834 | 1.000 | 0.250 | ✅ |
| Q11 | Active Hub ở tòa nhà 10 mở cửa vào những khung giờ n... | 1.000 | 0.958 | 1.000 | 1.000 | ✅ |
| Q12 | Sinh viên quốc tế có thể mua eSIM từ những nhà cung ... | n/a | 0.929 | 0.000 | 0.000 | ✅ |
| Q13 | Sau khi mua eSIM online và nhận được mã QR thì có nê... | 0.000 | 0.000 | 1.000 | 0.500 | ✅ |
| Q14 | Mỗi học kỳ sinh viên phải đăng ký tối thiểu và tối đ... | 1.000 | 0.980 | 1.000 | 1.000 | ✅ |
| Q15 | Điều kiện để được rút bớt học phần khi lớp đã ở trạn... | 1.000 | 0.981 | 0.500 | 0.750 | ✅ |

## 5. Worst Performers


**Config `A_hybrid_rerank`** — 3 câu điểm thấp nhất:

- `Q06` (avg n/a) — Sinh viên nhận học bổng RMIT phải duy trì GPA tối thiểu bao nhiêu để không bị rút học bổng?
  - Nguồn cần có: `dieu-khoan-hoc-bong.md` → lấy được: **có**
  - Thực tế lấy về: ['hoc-phi-va-cac-khoan-thu.md', 'hoc-phi-va-cac-khoan-thu.md', 'dieu-khoan-hoc-bong.md', 'Dangki hoc phan.md', 'hoc-phi-va-cac-khoan-thu.md']
- `Q09` (avg 0.000) — Học bổng RMIT có được quy đổi thành tiền mặt hoặc chuyển nhượng cho người khác không?
  - Nguồn cần có: `dieu-khoan-hoc-bong.md` → lấy được: **KHÔNG**
  - Thực tế lấy về: ['hoc-phi-va-cac-khoan-thu.md', 'hoc-phi-va-cac-khoan-thu.md', 'hoc-phi-va-cac-khoan-thu.md', 'hoc-phi-va-cac-khoan-thu.md', 'hoc-phi-va-cac-khoan-thu.md']
- `Q10` (avg 0.000) — Giờ hoạt động của khuôn viên cơ sở Nam Sài Gòn là mấy giờ?
  - Nguồn cần có: `gio-mo-cua-co-so-sgs.md` → lấy được: **có**
  - Thực tế lấy về: ['hoc-phi-va-cac-khoan-thu.md', 'hoc-phi-va-cac-khoan-thu.md', 'gio-mo-cua-co-so-sgs.md', 'hoc-phi-va-cac-khoan-thu.md', 'gio-mo-cua-co-so-sgs.md']

**Config `B_dense_only`** — 3 câu điểm thấp nhất:

- `Q05` (avg 0.000) — Sinh viên có thể thanh toán học phí bằng thẻ tín dụng ở đâu?
  - Nguồn cần có: `hoc-phi-va-cac-khoan-thu.md` → lấy được: **có**
  - Thực tế lấy về: ['hoc-phi-va-cac-khoan-thu.md', 'hoc-phi-va-cac-khoan-thu.md', 'hoc-phi-va-cac-khoan-thu.md', 'hoc-phi-va-cac-khoan-thu.md', 'hoc-phi-va-cac-khoan-thu.md']
- `Q06` (avg 0.000) — Sinh viên nhận học bổng RMIT phải duy trì GPA tối thiểu bao nhiêu để không bị rút học bổng?
  - Nguồn cần có: `dieu-khoan-hoc-bong.md` → lấy được: **có**
  - Thực tế lấy về: ['dieu-khoan-hoc-bong.md', 'hoc-phi-va-cac-khoan-thu.md', 'hoc-phi-va-cac-khoan-thu.md', 'hoc-phi-va-cac-khoan-thu.md', 'hoc-phi-va-cac-khoan-thu.md']
- `Q09` (avg 0.000) — Học bổng RMIT có được quy đổi thành tiền mặt hoặc chuyển nhượng cho người khác không?
  - Nguồn cần có: `dieu-khoan-hoc-bong.md` → lấy được: **có**
  - Thực tế lấy về: ['hoc-phi-va-cac-khoan-thu.md', 'hoc-phi-va-cac-khoan-thu.md', 'hoc-phi-va-cac-khoan-thu.md', 'dieu-khoan-hoc-bong.md', 'hoc-phi-va-cac-khoan-thu.md']

## 6. Recommendations

**`A_hybrid_rerank`** — source_recall 93%, 1 câu không lấy được tài liệu đúng.
  - Nguồn chiếm nhiều chunk nhất trong kết quả: `hoc-phi-va-cac-khoan-thu.md` (45/75 = 60%).
  - Câu miss: `Q09`

**`B_dense_only`** — source_recall 100%, 0 câu không lấy được tài liệu đúng.
  - Nguồn chiếm nhiều chunk nhất trong kết quả: `hoc-phi-va-cac-khoan-thu.md` (42/75 = 56%).

**Đề xuất cải thiện (còn mở):**

1. **Giới hạn số chunk mỗi tài liệu trong kết quả cuối** (ví dụ tối đa 2 chunk/file) hoặc đổi `RERANK_METHOD` sang `mmr`. Corpus hiện mất cân bằng nặng — `hoc-phi-va-cac-khoan-thu.md` chiếm 412/573 chunk (72%), nên nó áp đảo cả dense lẫn BM25 và đẩy các tài liệu nhỏ ra khỏi top_k.
2. **Tăng `top_k` không giải quyết được triệt để** — đã đo: với câu hỏi eSIM, top_k=8 vẫn không lấy được đúng tài liệu.
3. **Chuẩn hoá dấu tiếng Việt cho nhánh BM25** — BM25 khớp token chính xác nên truy vấn không dấu không bao giờ match được văn bản có dấu.

**Đã triển khai sau lần đo này:**

4. ✅ **Cache SentenceTransformer trong `task5_semantic_search`** — trước đây mỗi lần gọi `semantic_search()` lại load bge-m3 (~15s/lần), chiếm phần lớn thời gian chạy eval. Nay model và ChromaDB collection được cache ở module level. Đây là tối ưu thuần hiệu năng: cùng model, cùng vector, **không làm đổi thứ hạng** nên các số liệu ở trên vẫn còn hiệu lực.
5. ✅ **Supervisor + workers song song** (`src/supervisor.py`) — chạy đồng thời dense / sparse / HyDE / query-expansion rồi gộp bằng RRF. Đây là nhánh retrieval MỚI, chưa được đo trong bảng A/B trên; cần thêm làm config C rồi chạy lại eval.
