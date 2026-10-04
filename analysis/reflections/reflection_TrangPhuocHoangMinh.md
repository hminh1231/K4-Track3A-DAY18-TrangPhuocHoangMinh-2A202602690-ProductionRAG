# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Trang Phước Hoàng Minh — 2A202602690  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 tạo **208 chunks** (avg 99 ký tự, min 6), trong khi basic chỉ có **51 chunks** (avg 410). Chunk bị vụn vì `all-MiniLM-L6-v2` là model tiếng Anh, nên độ tương đồng giữa các câu tiếng Việt thấp và gần như câu nào cũng bị tách. Muốn dùng thật thì phải hạ threshold về khoảng 0.5–0.6 hoặc dùng embedding đa ngôn ngữ (bge-m3). |
| Hierarchical (parent-child) chunking | M1 + pipeline | `chunk_hierarchical()`, `_expand_to_parents()` | 26 tài liệu → 104 child (≤256 ký tự) / parent ≤2048 ký tự. Bài học lớn nhất của lab: chỉ cắt child mà không **trả parent về cho LLM** thì Production còn thua Baseline (faithfulness 0.64 so với 0.77). Sau khi thêm bước small-to-big (rank bằng child, đưa parent vào prompt), faithfulness lên **0.85**. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Tách theo header `#`/`##`/`###`, ghi tên mục vào `metadata["section"]`, được 106 chunks. Tiêu đề không có nội dung riêng thì không tạo chunk rỗng. |
| BM25 + Dense fusion | M2 | `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | BM25 cần tách từ tiếng Việt bằng underthesea rồi `replace("_", " ")`, nếu không thì "nghỉ_phép" (1 token) sẽ không khớp với query "nghỉ phép" (2 token). RRF (k=60) cộng điểm theo thứ hạng nên không phải chuẩn hóa thang điểm BM25 với cosine. Dense dùng `query_points()` (qdrant-client 1.19), và tạo collection bằng `collection_exists` → `delete` → `create` vì `recreate_collection` đã deprecated. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `bge-reranker-v2-m3` tách biệt rất rõ: đoạn đúng được 0.99, các đoạn nhiễu 0.02 và 0.0007. Nhưng latency khoảng **275ms** cho 3 đoạn trên CPU, vượt mục tiêu 150ms. Production cần GPU/fp16 hoặc reranker nhẹ hơn (Flashrank). Context precision của Production đạt **0.975**. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Production: faithfulness 0.85, relevancy 0.84, precision 0.975, recall 0.95, cả 4 đều cao hơn Baseline. Phát hiện quan trọng: RAGAS trả `NaN` khi LLM chấm điểm bị lỗi, và nếu ép `NaN` thành 0 thì câu trả lời đúng bị chẩn đoán là hallucination. Ngoài ra faithfulness phạt các câu trả lời có phép tính đúng (vì kết quả không có nguyên văn trong context). |
| Contextual embeddings / Enrichment | M5 | `_enrich_single_call()`, `contextual_prepend()` | 1 call `gpt-4o-mini` (JSON mode) cho mỗi chunk trả về summary, 3 câu HyQA, câu context và metadata. Mất khoảng 6 phút cho 104 chunks, thay vì 416 call nếu gọi riêng. Câu context ("Đoạn văn nằm trong tài liệu về chính sách nghỉ phép năm 2024…") giúp phân biệt các chunk giống nhau giữa v2023 và v2024. Khi không có API key, mỗi trường đều có fallback trích từ văn bản gốc. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  1. Production RAG thua Baseline ở lần chạy đầu: `faithfulness 0.7738 → 0.6417 (-0.1321)`, `answer_relevancy 0.7592 → 0.6486 (-0.1106)`.
  2. Nhiều câu có `faithfulness: 0.0, answer_relevancy: 0.0` nhưng `context_precision: 1.0, context_recall: 1.0`. Bảng chẩn đoán gán nhãn "LLM tự bịa câu trả lời" cho cả những câu trả lời đúng (ví dụ "15 ngày phép năm", "CEO phê duyệt").
  3. Câu "Bao lâu phải đổi mật khẩu một lần?" được trả lời: *"Có hai quy định… mỗi 90 ngày và một là mỗi 120 ngày. Không tìm thấy."*
  4. Trên Windows: `UnicodeEncodeError: 'charmap' codec can't encode characters` khi in tiếng Việt ra terminal.

- **Nguyên nhân gốc rễ & Cách debug:**
  - Mình viết một script debug in ra **câu hỏi, ground truth, câu trả lời và 3 context** cho từng câu điểm thấp, thay vì chỉ nhìn điểm tổng. Kết quả đọc được:
    - Với (1): context đưa vào LLM chỉ là các child chunk 256 ký tự kèm câu context của M5. `pipeline.py` lưu `parent_id` nhưng không bao giờ lấy parent ra, nên kỹ thuật hierarchical chưa thực sự được dùng. Cách sửa: tạo `parent_store` với key `source::parent_id` (vì `parent_0` lặp lại ở mọi file), rồi `_expand_to_parents()` thay child bằng parent và bỏ trùng.
    - Với (2): đọc lại code M4 thì thấy hàm `_safe_float` đổi `NaN` thành 0. Đó là các câu mà LLM chấm điểm bị timeout hoặc lỗi parse, không phải câu sai. Cách sửa: giữ `NaN`, bỏ qua khi tính trung bình và khi tìm `worst_metric`, ghi `null` vào report, và giảm `max_workers` của RAGAS xuống 4 để bớt timeout.
    - Với (3): dữ liệu có cả chính sách cũ lẫn mới (v1.0 với v2.0, 2023 với 2024), mà prompt cũ không nói phải chọn bản nào. Cách sửa: prompt mới yêu cầu dùng phiên bản mới nhất, nói rõ bản cũ đã bị thay thế, trả lời trực tiếp, và đặt `temperature=0`.
    - Với (4): đặt `PYTHONIOENCODING=utf-8` (các module đã có sẵn `sys.stdout.reconfigure(encoding="utf-8")`).
  - Kết quả sau khi sửa: faithfulness **0.85**, relevancy **0.84**, precision **0.975**, recall **0.95**, cả 4 chỉ số đều ≥ 0.75 và đều cao hơn Baseline.

- **Kiến thức còn thiếu & Cách khắc phục:**
  - Trước lab mình nghĩ "chunk nhỏ hơn thì chính xác hơn". Thực tế chunk nhỏ chỉ tốt cho bước *tìm*, còn bước *trả lời* cần ngữ cảnh rộng. Đó chính là lý do có parent-child, và mình hiểu ra điều này qua việc debug chứ không phải đọc slide.
  - Chưa hiểu cách RAGAS tính faithfulness: nó tách câu trả lời thành các phát biểu và kiểm tra từng phát biểu với context. Vì vậy câu trả lời có phép tính đúng vẫn bị trừ điểm. Mình đã đọc lại tài liệu RAGAS và đọc thủ công các câu có faithfulness 0.5 nhưng recall 1.0 trước khi kết luận.
  - Cần tìm hiểu thêm về query decomposition cho câu hỏi multi-hop và cách đánh giá câu hỏi tính toán (exact-match đáp số).

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: EV RangeMind — DEV-06 EV Trip Planner (repo `P-118`)

Trợ lý lập lộ trình cho xe điện VinFast: đăng nhập → đăng ký xe + % pin → chọn điểm đến → agent tính pin tới nơi, nếu không đủ thì chọn trạm sạc dọc đường (tổng thời gian lái + sạc nhỏ nhất, mọi điểm dừng còn trên ngưỡng pin). Có "Trợ lý EV" chat/giọng nói tiếng Việt.

#### 1. Hiện trạng
- **Pipeline hiện tại:** Chưa có RAG. `app/assistant.py` dùng LLM tool calling (`gpt-4o-mini`) với 9 tool: `plan_route`, `search_place`, `get_directions`, `find_stations`, `add_break`, `find_food`, `clear_breaks`, `where_at_time`, `update_battery`. Dữ liệu là CSV có cấu trúc: `vehicle_specs.csv` (thông số + cột `notes` trích từ brochure) và khoảng 6.440 trạm VinFast trong `charging_stations_vinfast.csv` (có địa chỉ, số cổng, giờ mở cửa, ghi chú gửi xe). Khi mất key hoặc LLM lỗi thì chuyển sang bộ phân tích câu theo luật (chế độ offline).
- **Vấn đề / Bottlenecks đang gặp:**
  - Trợ lý chỉ trả lời được những gì 9 tool trả về. Các câu hỏi kiến thức như *"VF 8 sạc 10–70% mất bao lâu?"*, *"trạm này có mất phí gửi xe không?"*, *"pin LFP của VF 3 có nên sạc đầy 100%?"* không có tool nào phục vụ, nên LLM dễ trả lời chung chung hoặc bịa. Điều này đi ngược yêu cầu F3/G3 trong PRD: *"câu trả lời bám tool/cache, không bịa trạm"*.
  - Thông tin chữ tự do (cột `notes` của xe và trạm, brochure PDF) hiện không tìm kiếm được, chỉ được đọc như trường dữ liệu.
  - Dữ liệu có nhiều phiên bản: brochure VF 8 "The All New" khác bản cũ, quãng đường ghi "480–500 km", và cột `collected_on` / "Cập nhật nguồn" của từng trạm. Đây đúng là kiểu xung đột phiên bản đã làm LLM trả "Không tìm thấy" trong lab này.
  - Ràng buộc của đề: phải chạy được **offline / on-device** (C1), dữ liệu vị trí **không gửi lên cloud** (C3), và chấp nhận độ trễ (C5).

#### 2. Kế hoạch cải tiến
Thêm một tool mới `search_knowledge(query, vehicle_id?, station_id?)` vào danh sách tool của trợ lý. Tool này chạy pipeline RAG local trên 3 nguồn: (a) brochure + `notes` của xe, (b) mô tả chữ của từng trạm sạc, (c) bộ FAQ về sạc/chăm sóc pin do nhóm tự viết.

1. **Chunking strategy:** Mỗi nguồn dùng một cách cắt khác nhau.
   - **Trạm sạc và xe: mỗi dòng CSV là 1 chunk** (dạng structure-aware), sinh câu mô tả bằng template từ các cột (tên, địa chỉ, kW, số cổng, giờ mở cửa, ghi chú gửi xe). Các cột số và tỉnh giữ trong metadata để **lọc** (ví dụ `max_power_kw ≥ 100`, cùng tỉnh), không để LLM tự so sánh số.
   - **Brochure PDF: hierarchical** (parent theo mục của brochure, child khoảng 256 ký tự), có bước **trả parent cho LLM**. Lab cho thấy thiếu bước này là Production thua Baseline.
   - **FAQ: structure-aware theo header.** Không dùng semantic chunking với MiniLM vì chunk tiếng Việt bị vụn (208 chunk, min 6 ký tự).
2. **Search retrieval:** **Hybrid BM25 + dense, gộp bằng RRF.** BM25 (tách từ bằng underthesea) rất quan trọng ở đây vì người dùng gõ chính xác mã xe ("VF 8", "VF3"), công suất ("180 kW"), tên đường/tỉnh và tên trạm. Dense xử lý câu hỏi diễn đạt tự do ("sạc ở đâu nhanh nhất gần Phủ Lý"). Vì phải chạy on-device, dense dùng model embedding đa ngôn ngữ nhỏ (bge-m3 bản quantize ONNX hoặc multilingual-e5-small), và index lưu local (Qdrant `:memory:`/file hoặc ChromaDB có sẵn trong template) để vẫn chạy khi mất mạng.
3. **Reranking:** **Có, nhưng phải cân với độ trễ trên thiết bị.** Trong lab, `bge-reranker-v2-m3` mất khoảng 275ms cho 3 đoạn trên CPU, quá chậm cho trợ lý giọng nói. Kế hoạch: Flashrank (model nhỏ) cho câu hỏi thời gian thực; cross-encoder lớn chỉ dùng khi online hoặc khi câu hỏi phức tạp. Lọc metadata (loại xe, tỉnh, kW) chạy **trước** rerank để giảm số ứng viên.
4. **Evaluation:**
   - Bộ test khoảng 40 câu hỏi kiến thức (thông số xe, thời gian sạc, ghi chú trạm, chăm sóc pin, câu so sánh giữa 2 xe), chấm bằng **RAGAS 4 metrics** và so với baseline (dense-only, không rerank).
   - Thêm metric riêng cho F3: **tỉ lệ ID/tên trạm trong câu trả lời có xuất hiện trong context** (0 trạm bịa), và **exact-match con số** (kW, phút sạc, km) vì faithfulness phạt oan các câu có tính toán.
   - Ghép vào `eval/run_eval.py` hiện có (đang 7/8 case đạt) để chạy chung một lệnh. Đọc bottom-5 theo Error Tree.
5. **Enrichment:**
   - **Contextual prepend** cho chunk brochure (*"Trích từ brochure VF 8 The All New, mục Sạc pin…"*) để phân biệt các xe có thông số gần giống nhau.
   - **Metadata extraction** thêm `vehicle_id`, `variant`, `collected_on` để ưu tiên dữ liệu mới nhất khi có xung đột.
   - **HyQA** cho FAQ.
   - Mô tả trạm sạc sinh bằng template, không gọi LLM, vì có 6.440 dòng. Enrichment chạy offline một lần lúc build gói dữ liệu và cache lại, đi kèm gói OTA dữ liệu trạm của admin.

#### 3. Timeline triển khai
- **Tuần 1:** Viết bộ test khoảng 40 câu + FAQ sạc/pin, dựng baseline (dense-only) và đo RAGAS. Sinh chunk cho trạm/xe từ CSV, cắt hierarchical cho brochure.
- **Tuần 2:** Thêm BM25 tiếng Việt + RRF và bộ lọc metadata, viết tool `search_knowledge` rồi nối vào `assistant.py`. Thêm vào system prompt quy tắc phiên bản mới nhất và yêu cầu chỉ nêu trạm có trong context. Đo lại RAGAS và metric "trạm bịa".
- **Tuần 3:** Enrichment (contextual prepend + metadata) có cache, thêm reranker nhẹ, đo độ trễ end-to-end của trợ lý giọng nói, kiểm tra chế độ offline (không key, không mạng), và đưa kết quả vào `eval/RESULTS.md`.
