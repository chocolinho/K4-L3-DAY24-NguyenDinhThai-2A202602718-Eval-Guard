# Blueprint CI/CD cho RAG và guardrails

Nhóm: K4-L3-DAY24-NguyenDinhThai-EvalGuard

Người phụ trách: Nguyễn Đình Thái - 2A202602718

Đây là **đề xuất CI/CD**, không phải workflow đã được triển khai/chạy.
Nguồn kết quả: artifacts/ragas_50q.json, judge_results.json, guard_results.json.
Run timestamp: 2026-10-12T13:32:19.642471+00:00; guard models: `gpt-4o-mini`.
Models generation/evaluator/judge từ A/B: Generation: `openai/gpt-4o-mini`; Evaluator: `openai/gpt-4o-mini`, `BAAI/bge-m3`; Judge: `gpt-4o-mini`.

## Kiến trúc và failure action

Input → Presidio → NeMo input → RAG → NeMo output → response.
Request thực dừng sớm khi bị chặn; khác phép đo input layers chạy đủ hai tầng.

Mô tả luồng dữ liệu và cơ chế xử lý:
1. **Data Flow:** Khi có truy vấn từ người dùng, dữ liệu đi qua Presidio PII Engine đầu tiên để quét CCCD/CMND, số điện thoại và email. Tiếp đó, dữ liệu đi qua NeMo Input Rails để phát hiện prompt injection, jailbreak và câu hỏi sai phạm vi nghiệp vụ. Nếu hợp lệ, RAG Pipeline xử lý truy vấn (Hybrid BM25 + Qdrant Dense + BGE Reranker + LLM Generation). Cuối cùng, câu trả lời đi qua NeMo Output Rails để ngăn chặn rò rỉ thông tin cá nhân và dữ liệu bảo mật trước khi gửi về client.
2. **Failure Action:** Cơ chế fail-fast (short-circuit). Nếu vi phạm ở bất kỳ tầng nào, request bị hủy ngay lập tức và trả về thông điệp từ chối tiêu chuẩn, không chuyển tiếp xuống các tầng sau nhằm tiết kiệm tài nguyên và bảo vệ hệ thống.
3. **Log & Redact:** Mọi trường PII phát hiện bởi Presidio đều được che mờ (redacted) thành `<VN_CCCD>`, `<VN_PHONE>`, `<EMAIL_ADDRESS>` trước khi ghi vào log audit. Tuyệt đối không log thông tin người dùng thô hoặc nội dung payload độc hại.
4. **Quản lý Secrets & Fallback:** Các API keys được quản lý trong Secret Manager / GitHub Secrets, nạp qua biến môi trường `.env`. Nếu LLM Guardrail gặp sự cố mạng hoặc timeout, hệ thống fallback về rule-based regex and keyword checks nội bộ để đảm bảo không bị gián đoạn dịch vụ.

## Phạm vi và số đo latency

Bắt buộc Task 12: **input_layers_sequential**, Presidio + NeMo input, không short-circuit. Engines tạo trước đo; có hai warm-up thật không tính vào mẫu. Percentile nội suy tuyến tính từ raw samples; tổng từ tổng từng sample.

Số mẫu timed: 20; warm-up: 2; phương pháp: linear interpolation.

| Layer / scope | P50 ms | P95 ms | P99 ms | Tình trạng |
|---|---|---|---|---|
| Presidio | 9.21 ms | 10.62 ms | 10.75 ms | Hoàn thành đo Task 12 |
| NeMo input | 1009.22 ms | 1995.59 ms | 2039.42 ms | Hoàn thành đo Task 12 |
| Input guard sequential | 1018.08 ms | 2004.08 ms | 2048.23 ms | Hoàn thành đo Task 12 |
| RAG | chưa đo | chưa đo | chưa đo | Ngoài phạm vi bắt buộc |
| NeMo output | chưa đo | chưa đo | chưa đo | Ngoài phạm vi bắt buộc |
| Full guard / short-circuit request | chưa đo | chưa đo | chưa đo | Không suy từ input_guard_ms |

Input-layer budget target: 500ms; kết quả thực tế: 2004.08 ms; budget_ok: False.

Phân tích điểm nghẽn (Bottleneck):
- Tầng Presidio chạy cục bộ bằng regex và spaCy xử lý cực kỳ nhanh chóng (P95 chỉ 10.62 ms), hoàn toàn nằm trong ngân sách cho phép.
- Điểm nghẽn duy nhất nằm ở NeMo input rail (P95 lên tới 1995.59 ms) do mỗi lần kiểm tra phải gửi request qua Internet tới OpenRouter cloud LLM (`gpt-4o-mini`).
- Đối với các phần ngoài phạm vi bắt buộc (RAG, NeMo output, full short-circuit request), báo cáo duy trì trạng thái "chưa đo". Để đo lường các phần này trong tương lai, cần xây dựng kịch bản benchmark độc lập với ít nhất 20 mẫu, ghi nhận timestamp chi tiết ở từng chặng.

## CI gates đề xuất

- RAGAS faithfulness ≥0.75, avg_score ≥0.65 trên 50 câu.
- Attacks ≥18/20 cho CI đề xuất; clean 8/8 allowed; output cases đúng.
- Full guard P95 <500ms chỉ áp dụng sau khi thực sự đo full guard.

Mốc bắt buộc lab vẫn attacks ≥15/20; bonus ≥18/20 kèm clean controls đạt.

Đề xuất kịch bản tích hợp CI:
- Lệnh chạy kiểm thử: `python scripts/check_lab.py --with-tests`
- Trong quy trình CI/CD, script kiểm thử trả về mã lỗi nonzero khi bất kỳ ngưỡng nào không đạt, từ đó tự động chặn quy trình merge pull request hoặc deploy container.
- Lưu ý: Đây là workflow đề xuất dựa trên kiến trúc thực tế của lab, chưa được kích hoạt tự động trên runner của GitHub Actions nhằm tránh rò rỉ API key và phát sinh chi phí tự động.

## Monitoring và cải tiến

Chiến lược giám sát vận hành:
- Giám sát độ trễ P95, tỷ lệ chặn (block rate), tỷ lệ dương tính giả (false positive rate) và độ lệch phân bố (data drift) của câu hỏi người dùng.
- Thiết lập cảnh báo tự động khi P95 vượt ngưỡng 2500ms hoặc khi tỷ lệ cảnh báo PII/Jailbreak tăng đột biến trên 15% trong vòng 10 phút.
- Đội ngũ phụ trách phản ứng sự cố: Kỹ sư AI Platform phối hợp cùng Security Operations Center (SOC).

Số liệu thực nghiệm từ bộ kiểm thử:
RAGAS mean/worst distribution: mean 0.7451, worst distribution: multi_hop (0.6449).
Cohen’s κ: 1.0.
Attacks passed: 20/20; clean false_positive_rate: 0.0 (8/8 clean controls allowed).

Các đề xuất cải tiến có thể kiểm chứng:
1. Áp dụng Sub-query Decomposition để phân rã câu hỏi multi-hop thành các truy vấn đơn lẻ, giúp tăng faithfulness của nhóm multi-hop từ 0.37 lên trên 0.70.
2. Thêm bộ lọc metadata thời gian và phiên bản văn bản để giải quyết triệt để lỗi xung đột phiên bản ở nhóm câu hỏi adversarial.
3. Thay thế cloud LLM trong NeMo Guardrails bằng mô hình phân loại nhỏ chạy cục bộ (local SLM / ONNX classifier) để cắt giảm độ trễ P95 từ 2004ms xuống dưới 100ms.
4. Mở rộng tập dữ liệu đánh giá lên 200 câu hỏi nghiệp vụ thực tế nhằm tăng độ tin cậy thống kê cho các chỉ số đánh giá.

