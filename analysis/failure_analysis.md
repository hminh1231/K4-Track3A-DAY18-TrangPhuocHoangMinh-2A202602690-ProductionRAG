# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Trang Phước Hoàng Minh — 2A202602690  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

Kết quả từ `python main.py` (20 câu hỏi trong `test_set.json`, LLM sinh câu trả lời và LLM chấm điểm đều là `gpt-4o-mini`):

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8167 | 0.8500 | +0.0333 |
| Answer Relevancy | 0.7147 | 0.8403 | +0.1256 |
| Context Precision | 0.9250 | 0.9750 | +0.0500 |
| Context Recall | 0.9250 | 0.9500 | +0.0250 |

- **Naive Baseline:** paragraph chunking (`chunk_basic`) + dense search (bge-m3), top-3, không rerank, không enrichment.
- **Production:** hierarchical chunking (parent 2048 / child 256) → enrichment 1 call/chunk (contextual prepend + HyQA + metadata) → hybrid BM25 (underthesea) + dense, gộp bằng RRF → rerank bằng `bge-reranker-v2-m3` lấy top-3 child → trả về **parent chunk** tương ứng cho LLM.

**Ghi chú về lần chạy đầu tiên.** Lần chạy đầu, Production còn *thua* Baseline (faithfulness 0.64, answer_relevancy 0.65). Có ba nguyên nhân, đều đã sửa trước khi chạy lại:

1. Pipeline chỉ đưa các child chunk 256 ký tự vào LLM mà không lấy lại parent, nên mất ngữ cảnh. Giờ `_expand_to_parents()` trong `src/pipeline.py` xếp hạng bằng child rồi trả parent cho LLM.
2. Prompt sinh câu trả lời không xử lý được tài liệu có nhiều phiên bản mâu thuẫn (v2023 với v2024, mật khẩu 90 với 120 ngày), nên LLM trả "Không tìm thấy". Prompt mới yêu cầu dùng phiên bản mới nhất, trả lời trực tiếp, và dùng `temperature=0`.
3. Khi LLM chấm điểm của RAGAS bị lỗi ở một câu, kết quả là `NaN`, nhưng code cũ đổi `NaN` thành 0. Hậu quả là các câu trả lời đúng bị chẩn đoán nhầm là "hallucination". Giờ `NaN` được giữ là "chưa chấm" và bị bỏ qua khi tính trung bình. Mình cũng giảm số luồng chạy song song của RAGAS (`max_workers=4`) để bớt timeout.

## Bottom-5 Failures

Các câu dưới đây được xếp theo điểm trung bình 4 chỉ số, từ thấp nhất (lấy từ `reports/ragas_report.json`).

### #1
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 + 3 = 18 ngày phép (chính sách v2024); lương Senior (P3-P4) 20–35 triệu VNĐ/tháng.
- **Got:** "18 ngày phép năm (15 + 3). Lương không được đề cập trong context."
- **Worst metric:** faithfulness = 0.33 (context_recall = 0.5, avg 0.67)
- **Error Tree:** Output sai một nửa → Context đúng? **Không đủ**: chỉ có đoạn nghỉ phép, thiếu `bang_luong_2024.md` → Query OK? Câu hỏi multi-hop gộp 2 chủ đề vào 1 query → **lỗi ở Retrieval**.
- **Root cause:** Câu hỏi cần thông tin từ 2 tài liệu khác nhau (nghỉ phép và bảng lương). Top-3 sau rerank bị chiếm hết bởi các đoạn về nghỉ phép (cả v2023 lẫn v2024), vì phần "nghỉ phép" của câu hỏi khớp mạnh hơn. Đoạn bảng lương không lọt vào top-3. LLM đã trả lời đúng là "không có trong context", nhưng RAGAS vẫn trừ điểm faithfulness cho câu đó.
- **Suggested fix:** Query decomposition, tức tách câu hỏi thành các câu con ("số ngày phép" và "lương Senior") rồi retrieve riêng và gộp context. Hoặc tăng `RERANK_TOP_K` cho câu hỏi dài hoặc có chữ "và". Thêm một bước lọc metadata để bỏ chính sách đã hết hiệu lực (v2023) cũng giúp giải phóng chỗ trong top-3.

### #2
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Hạn 15 ngày → quá hạn 5 ngày; phí 2%/tháng × 15 triệu = 300.000 VNĐ/tháng → pro-rata ≈ 50.000 VNĐ.
- **Got:** "15.000.000 × 2% × (20/30) = 200.000 VNĐ."
- **Worst metric:** faithfulness = 0.17 (context_precision và context_recall đều = 1.0)
- **Error Tree:** Output sai → Context đúng? **Đúng**, có câu "chưa thanh toán sau 15 ngày sẽ bị tính phí 2%/tháng" → Query OK? **OK** → **lỗi ở Generation (suy luận)**.
- **Root cause:** Retrieval hoàn hảo, nhưng LLM tính phí trên toàn bộ 20 ngày thay vì chỉ 5 ngày quá hạn. Đây là lỗi suy luận số học nhiều bước của `gpt-4o-mini`, không phải do thiếu tài liệu.
- **Suggested fix:** Ở prompt, yêu cầu LLM liệt kê dữ kiện trước khi tính (thời hạn, số ngày quá hạn, mức phí), tức chain-of-thought có cấu trúc. Hoặc dùng model suy luận mạnh hơn cho các câu có phép tính. Có thể thêm few-shot ví dụ về tính phí pro-rata.

### #3
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Cam kết làm việc tối thiểu 1 năm; nghỉ sau 8 tháng → hoàn trả 100% = 25.000.000 VNĐ.
- **Got:** "Phải hoàn trả 100% chi phí đào tạo, tức là 25.000.000 VNĐ."
- **Worst metric:** faithfulness = 0.5 (context_recall = 1.0)
- **Error Tree:** Output **đúng** → Context đúng? Đúng → Query OK? OK → **không phải lỗi pipeline, là hạn chế của metric**.
- **Root cause:** Đáp án đúng, nhưng con số "25.000.000 VNĐ" lấy từ câu hỏi chứ không có trong context. Vì vậy RAGAS coi phát biểu đó là "không được context hỗ trợ". Ngoài ra, câu trả lời không nêu lý do (cam kết 1 năm), nên chuỗi suy luận khó kiểm chứng.
- **Suggested fix:** Ở prompt, yêu cầu trích dẫn điều khoản làm căn cứ ("vì cam kết làm việc ≥ 1 năm sau khóa học…") để mọi phát biểu đều bám vào context. Khi đánh giá, nên đọc lại thủ công các câu có faithfulness 0.5 mà recall bằng 1.0 trước khi kết luận là hallucination.

### #4
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** 85% × 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** "17.000.000 VNĐ/tháng (85% của 20.000.000 VNĐ)."
- **Worst metric:** faithfulness = 0.5 (context_precision và context_recall đều = 1.0)
- **Error Tree:** Output **đúng** → Context đúng? Đúng (`bang_luong_2024.md` có 85% và mức trần Junior) → Query OK? OK → **hạn chế của metric**.
- **Root cause:** Giống #3: con số 17.000.000 là kết quả tính toán, không xuất hiện nguyên văn trong context, nên LLM chấm điểm của RAGAS chỉ chấp nhận một nửa số phát biểu.
- **Suggested fix:** Cho LLM ghi rõ "Theo bảng lương, Junior tối đa 20.000.000; lương thử việc = 85%…". Với bộ test nhiều câu tính toán, nên bổ sung một metric so khớp đáp số (exact-match) bên cạnh faithfulness.

### #5
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16–30 ngày cần Giám đốc điều hành (CEO) phê duyệt; nghỉ trên 14 ngày phải tự đóng phần bảo hiểm.
- **Got:** "Nghỉ phép không lương 20 ngày cần phê duyệt của Giám đốc điều hành (CEO)."
- **Worst metric:** faithfulness = 0.5 (context_recall = 1.0)
- **Error Tree:** Output đúng nhưng thiếu ý → Context đúng? Đúng (có đoạn "Nghỉ từ 16-30 ngày: … CEO") → Query OK? OK → **Generation chưa đầy đủ**.
- **Root cause:** Câu trả lời đúng người phê duyệt nhưng không nêu căn cứ (khoảng 16–30 ngày) và bỏ qua lưu ý về bảo hiểm có trong ground truth. RAGAS tách câu trả lời thành nhiều phát biểu, trong đó "20 ngày → CEO" là suy luận chứ không phải trích dẫn trực tiếp.
- **Suggested fix:** Thêm vào prompt yêu cầu "nêu điều khoản áp dụng và các lưu ý liên quan trong context". Prompt hiện tại ưu tiên ngắn gọn, nên LLM bỏ qua thông tin phụ.

**Nhận xét chung.** Sau khi sửa pipeline, 4/5 câu yếu nhất có `context_recall = 1.0`, tức retrieval đã đúng. Điểm nghẽn còn lại nằm ở **generation** (suy luận số học, thiếu trích dẫn căn cứ) và ở **giới hạn của faithfulness với câu hỏi cần tính toán**. Chỉ có #1 (multi-hop) là lỗi retrieval thật.

## Case Study (cho presentation)

**Question chọn phân tích:** "Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?"

**Error Tree walkthrough:**
1. Output đúng? → **Sai**: 200.000 VNĐ (tính 20 ngày), đáp án ≈ 50.000 VNĐ (5 ngày quá hạn, pro-rata của 300.000/tháng).
2. Context đúng? → **Đúng**: context_precision = context_recall = 1.0, đoạn `tam_ung.md` "chưa thanh toán sau 15 ngày sẽ bị tính phí 2%/tháng" đứng đầu top-3 sau rerank.
3. Query rewrite OK? → **OK**: query được tách từ đúng ("tạm ứng", "thanh toán", "phạt"), BM25 và dense đều tìm được đoạn này.
4. Fix ở bước: **Generation**. Thêm vào prompt bước "xác định dữ kiện → số ngày quá hạn = 20 − 15 → phí = số tiền × 2% × số ngày quá hạn / 30", hoặc định tuyến các câu có phép tính sang model suy luận mạnh hơn.

Bài học: điểm faithfulness thấp không phải lúc nào cũng do LLM bịa thông tin. Cần đi theo Error Tree từ Output → Context → Query để biết lỗi nằm ở retrieval hay generation, tránh sửa nhầm chỗ (ví dụ đi tune chunking trong khi retrieval đã hoàn hảo).

**Nếu có thêm 1 giờ, sẽ optimize:**
- **Query decomposition** cho câu hỏi multi-hop (#1): tách câu hỏi thành các câu con, retrieve từng câu, rồi gộp context.
- **Lọc theo phiên bản/ngày hiệu lực**: đưa `version` và `effective_date` vào metadata khi làm giàu, rồi lọc bỏ chính sách đã bị thay thế trước khi rerank, để top-3 không bị v2023 chiếm chỗ.
- **Prompt có cấu trúc cho câu tính toán**: liệt kê dữ kiện → phép tính → kết luận, kèm trích dẫn điều khoản, để vừa đúng vừa tăng faithfulness.
- **Giảm độ trễ rerank** (~275ms trên CPU cho 3 đoạn, vượt mục tiêu 150ms): thử fp16 trên GPU hoặc `FlashrankReranker` cho đường online.
