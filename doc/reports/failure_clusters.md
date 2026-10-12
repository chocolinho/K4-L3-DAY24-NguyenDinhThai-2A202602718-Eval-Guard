# Phân tích lỗi Phase A

Nhóm: K4-L3-DAY24-NguyenDinhThai-EvalGuard

Người phụ trách: Nguyễn Đình Thái - 2A202602718

Nguồn: `artifacts/ragas_50q.json`; run timestamp: 2026-10-12T13:21:17.016839+00:00.
Generation mode/fallback và điều kiện chạy: Chạy trực tiếp `--live` với OpenRouter API (`openai/gpt-4o-mini`), 100% LLM generation thành công không rơi vào extractive fallback.
Models generation/evaluator: Generation: `openai/gpt-4o-mini`; Evaluator: `openai/gpt-4o-mini`, `BAAI/bge-m3`.

## Điểm theo distribution

| Metric | factual (20) | multi_hop (20) | adversarial (10) |
|---|---|---|---|
| faithfulness | 0.9417 | 0.3763 | 0.6500 |
| answer_relevancy | 0.7744 | 0.5493 | 0.5411 |
| context_precision | 0.9375 | 0.9208 | 0.9417 |
| context_recall | 0.9250 | 0.7333 | 0.6500 |
| avg_score | 0.8947 | 0.6449 | 0.6957 |

## Bottom 10 và nguyên nhân

| Rank | Question ID | Câu hỏi | Distribution | Avg Score | Worst Metric | Diagnosis | Suggested Fix |
|---|---|---|---|---|---|---|---|
| 1 | 21 | Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào? | multi_hop | 0.2083 | faithfulness | LLM hallucinating | Tighten system prompt, lower temperature |
| 2 | 50 | Nhân viên Manager có thể dùng VPN cá nhân (như NordVPN) khi WFH để tăng bảo mật thêm không? | adversarial | 0.2917 | faithfulness | LLM hallucinating | Tighten system prompt, lower temperature |
| 3 | 33 | Nhân viên Manager có thâm niên 12 năm: tổng phụ cấp hàng tháng và số ngày phép năm theo v2024 là bao nhiêu? | multi_hop | 0.3750 | faithfulness | LLM hallucinating | Tighten system prompt, lower temperature |
| 4 | 39 | So sánh yêu cầu mật khẩu giữa policy v1.0 và v2.0 về độ dài tối thiểu, thời hạn đổi và MFA. | multi_hop | 0.3750 | faithfulness | LLM hallucinating | Tighten system prompt, lower temperature |
| 5 | 44 | Bao lâu phải đổi mật khẩu một lần? | adversarial | 0.3958 | faithfulness | LLM hallucinating | Tighten system prompt, lower temperature |
| 6 | 46 | Nhân viên thử việc có được nghỉ phép năm không? | adversarial | 0.4167 | faithfulness | LLM hallucinating | Tighten system prompt, lower temperature |
| 7 | 32 | Nhân viên mới cấp Manager có thời hạn thử việc 90 ngày, đang trong tháng thứ 3, có BẮT BUỘC tham dự buổi đào tạo nội bộ của phòng ban không? Tại sao? | multi_hop | 0.4583 | faithfulness | LLM hallucinating | Tighten system prompt, lower temperature |
| 8 | 30 | So sánh quyền lợi bảo hiểm giữa nhân viên thử việc và nhân viên chính thức. | multi_hop | 0.5000 | faithfulness | LLM hallucinating | Tighten system prompt, lower temperature |
| 9 | 29 | Nhân viên tạm ứng 8 triệu, chưa thanh toán sau 30 ngày (quá hạn 15 ngày). Ai phê duyệt khoản này và phí phạt là bao nhiêu? | multi_hop | 0.5000 | faithfulness | LLM hallucinating | Tighten system prompt, lower temperature |
| 10 | 40 | Nhân viên cấp Manager có thời hạn thử việc 90 ngày, đang trong tháng thứ 3, phát hiện vi phạm bảo mật. Họ nên và không nên làm gì theo chính sách? | multi_hop | 0.5410 | context_precision | Too many irrelevant chunks | Add reranking or metadata filter |

Đối chiếu chi tiết các câu đại diện:
- **Câu 21 (multi_hop - rank 1):** RAG retrieved context chỉ chứa một phần quy định thâm niên nhưng thiếu bảng khung lương Senior P3-P4, khiến mô hình sinh "Không tìm thấy" hoặc phỏng đoán sai dữ liệu ngoài context, kéo faithfulness xuống 0.0.
- **Câu 50 & 44 (adversarial - rank 2 & 5):** Câu hỏi gài bẫy giả định sai ("dùng NordVPN cá nhân để tăng bảo mật") hoặc bẫy phiên bản cũ ("bao lâu phải đổi mật khẩu một lần"). RAG pipeline đôi khi trích nhầm tài liệu cũ hoặc mô hình cố giải thích theo giả định của người hỏi thay vì bác bỏ dứt khoát rằng chính sách cấm hoàn toàn VPN bên ngoài, dẫn tới faithfulness thấp.
- **Câu 40 (multi_hop - rank 10):** Do kết hợp nhiều ý (thử việc 90 ngày + vi phạm bảo mật), top-3 reranker đưa vào các chunk về thủ tục thử việc và nhân sự thay vì tập trung vào incident response của chính sách bảo mật, làm context_precision bị tụt.

## Matrix và nhóm yếu nhất

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | 1 | 15 | 4 |
| context_recall | 3 | 2 | 5 |
| context_precision | 3 | 1 | 0 |
| answer_relevancy | 13 | 2 | 1 |

Nhóm yếu theo mean thấp nhất: multi_hop; mean: 0.6449.
Worst metric xuất hiện nhiều nhất: faithfulness; số câu: 20 câu.

Nhóm multi_hop có điểm trung bình thấp nhất (0.6449) và là điểm yếu chủ đạo do câu hỏi yêu cầu tổng hợp thông tin từ nhiều tài liệu độc lập (chính sách thâm niên + bảng lương + quy chế tạm ứng). Khi retrieval chỉ lấy top-3 chunks sau rerank, các chunk bổ sung cần thiết cho bước lập luận thứ hai thường bị rớt ngoài top-3, khiến LLM rơi vào tình trạng thiếu dữ kiện và tự suy diễn (hallucination), làm suy giảm nghiêm trọng faithfulness.

Đề xuất sửa cụ thể:
1. Áp dụng Sub-query Decomposition / Multi-Query Retrieval trong bước retrieval để truy vấn riêng từng mệnh đề trong câu hỏi multi-hop.
2. Tăng số chunk sau rerank lên top-5 cho nhóm multi-hop, hoặc thêm metadata tagging (role, policy_version) để liên kết chéo các văn bản.
3. Siết chặt prompt thế hệ (system prompt) yêu cầu LLM nêu rõ phần nào có trong context và từ chối suy đoán khi thiếu dữ liệu hỗ trợ. Chạy lại `python scripts/run_experiments.py --phase a --live` để đối chiếu avg_score tăng trưởng.

## Bẫy adversarial và giới hạn

So sánh với factual (avg_score: 0.8947), nhóm adversarial có avg_score thấp hơn đáng kể (0.6957) do gặp phải các bẫy:
1. **Xung đột phiên bản (Version Conflict):** Câu 41 và 44 chứa mâu thuẫn giữa chính sách bảo mật / nghỉ phép v1.0 và v2.0. Retrieval lấy cả chunk cũ lẫn mới, khiến mô hình nhầm lẫn thời hạn đổi mật khẩu hoặc số ngày nghỉ tiêu chuẩn.
2. **Tiền đề phủ định / giả định sai (False Premise / Negation):** Câu 50 đưa giả định sai rằng nhân viên có thể dùng VPN cá nhân. Điểm số của nhóm adversarial thấp không đồng nghĩa với việc pipeline đã nhận ra bẫy, mà thực tế phản ánh mô hình đã bị lừa khi cố chấp nhận giả định của người dùng.

Giới hạn thực nghiệm: Kích thước mẫu 50 câu (20/20/10) đủ để phát hiện các cụm lỗi chính nhưng chưa bao quát toàn bộ các kịch bản edge case; việc chạy qua OpenRouter với `gpt-4o-mini` cho kết quả nhất quán nhưng phụ thuộc vào độ trễ mạng và token generation format.
