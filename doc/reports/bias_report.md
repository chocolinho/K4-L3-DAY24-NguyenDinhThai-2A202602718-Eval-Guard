# Đánh giá LLM Judge Phase B

Nhóm: K4-L3-DAY24-NguyenDinhThai-EvalGuard

Người phụ trách: Nguyễn Đình Thái - 2A202602718

Nguồn: `artifacts/judge_results.json`; run timestamp: 2026-10-12T13:22:20.968599+00:00.
Models generation/judge: Generation: `openai/gpt-4o-mini`; Judge: `gpt-4o-mini`.

## Năm cặp A/B và hai lượt

Nguồn cố định: A từ artifacts/answers_50q.json:answer; B từ fixtures/human_labels_10q.json:model_answer; reference từ fixtures/test_set_50q.json:ground_truth.
Ids: 1, 5, 12, 21, 41. Không đổi nguồn sau khi xem kết quả judge.

| ID | Candidate A | Candidate B | Winner Pass 1 | Winner Pass 2 | Final Winner | Position Consistent | Scores Pass 1 (A/B) | Scores Pass 2 (A/B) | Reasoning Pass 1 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Nhân viên được nghỉ 3 ngày làm việc khi kết hôn. | Nhân viên được nghỉ 3 ngày làm việc có lương khi kết hôn. | B | B | B | True | 0.8 / 1.0 | 0.8 / 1.0 | Answer B cung cấp thông tin đầy đủ về việc nghỉ có lương, điều mà Answer A không đề cập đến. |
| 5 | Cần Tổng Giám đốc (CEO) phê duyệt. | Cần Giám đốc phòng ban phê duyệt. | A | A | A | True | 1.0 / 0.0 | 1.0 / 0.0 | Câu trả lời A chính xác với yêu cầu phê duyệt từ chính sách tham chiếu, trong khi câu B không đúng. |
| 12 | Thưởng Tết tối thiểu cho nhân viên chính thức có từ 6 tháng trở lên là 1 tháng lương. | Nhân viên được thưởng Tết tối thiểu 1 tháng lương. | A | A | A | True | 1.0 / 0.7 | 1.0 / 0.8 | Answer A đưa ra thông tin đầy đủ và chính xác về mức thưởng Tết cho nhân viên có thời gian làm việc từ 6 tháng trở lên. |
| 21 | Không tìm thấy. | Được nghỉ 18 ngày phép (15 cơ bản + 3 thâm niên). Lương Senior từ 20 đến 35 triệu. | B | B | B | True | 0.0 / 1.0 | 0.0 / 1.0 | Answer B cung cấp thông tin chính xác về số ngày phép và khoảng lương, trong khi Answer A không có thông tin nào. |
| 41 | Nhân viên được nghỉ 12 ngày phép năm. | Nhân viên được nghỉ 12 ngày phép năm. | tie | tie | tie | True | 0.0 / 0.0 | 0.0 / 0.0 | Cả hai câu trả lời đều không chính xác vì đều đưa ra thông tin về 12 ngày phép, không cập nhật theo chính sách v2024. |

Position bias count/rate: 0 câu và 0.0 (0.0%).
Verbosity win rate: 1.0 (100.0%); decisive cases: 4 câu.

Giải thích kết quả:
- Về position bias: Cả 5/5 cặp đều đạt position consistency 100% khi đảo vị trí giữa Pass 1 và Pass 2 (A vs B và B vs A). Judge không bị ảnh hưởng bởi thứ tự xuất hiện của ứng viên.
- Về verbosity bias: Ở 4 ca phân định thắng thua rõ ràng (decisive cases: 1, 5, 12, 21), ứng viên dài hơn đều giành chiến thắng (tỷ lệ 100%). Tuy nhiên, khi phân tích nội dung, lý do thắng là do ứng viên dài hơn chứa đầy đủ các dữ kiện quan trọng theo ground truth (câu 1 có "có lương", câu 5 chỉ đúng cấp CEO, câu 12 có điều kiện "từ 6 tháng trở lên", câu 21 có thông tin thay vì từ chối). Sự vượt trội về độ dài phản ánh tính đầy đủ về mặt thông tin chứ không phải sự thiên vị độ dài hình thức.

## Calibration đúng model_answer

Helper đánh giá đúng question/model_answer đang có trong fixtures/human_labels_10q.json; không lấy answer mới của pipeline. Prompt chỉ nhận question, answer, reference. Không gửi human_label, human_note hoặc phân bố nhãn human vào prompt.

| Question ID | Model Answer Gốc | Nhãn Judge | Reasoning | Human Label | Agree |
|---|---|---|---|---|---|
| 1 | Nhân viên được nghỉ 3 ngày làm việc có lương khi kết hôn. | 1 | Câu trả lời chính xác và đầy đủ theo thông tin trong tài liệu tham khảo, không thiếu ý thiết yếu. | 1 | True |
| 5 | Cần Giám đốc phòng ban phê duyệt. | 0 | Câu trả lời sai vì không nêu đúng người phê duyệt theo chính sách, cần Tổng Giám đốc chứ không phải Giám đốc phòng ban. | 0 | True |
| 12 | Nhân viên được thưởng Tết tối thiểu 1 tháng lương. | 1 | Câu trả lời chính xác theo thông tin trong reference, nêu rõ điều kiện và mức thưởng cho nhân viên chính thức có từ 6 tháng trở lên. | 1 | True |
| 21 | Được nghỉ 18 ngày phép (15 cơ bản + 3 thâm niên). Lương Senior từ 20 đến 35 triệu. | 1 | Câu trả lời chính xác theo reference, nêu rõ số ngày phép và khoảng lương cho nhân viên Senior với 9 năm thâm niên. | 1 | True |
| 23 | Hoàn trả 25 triệu vì chưa đủ 1 năm cam kết. | 1 | Câu trả lời chính xác theo thông tin trong reference, nêu rõ rằng nhân viên phải hoàn trả 100% chi phí vì chưa đủ 1 năm cam kết. | 1 | True |
| 29 | Trưởng phòng phê duyệt. Phạt 2% tháng trên 8 triệu. | 0 | Câu trả lời thiếu thông tin về sự phê duyệt của Kế toán trưởng, và phí phạt được tính cho quá hạn 15 ngày chứ không phải là 2% trên 8 triệu cho toàn bộ tháng. | 0 | True |
| 33 | Phép: 19 ngày. Phụ cấp: 1.500.000 VNĐ/tháng. | 1 | Câu trả lời cung cấp đúng số ngày phép và tổng phụ cấp hàng tháng theo thông tin trong tài liệu tham khảo. | 1 | True |
| 41 | Nhân viên được nghỉ 12 ngày phép năm. | 0 | Câu trả lời sai vì nó dựa trên chính sách cũ (v2023) thay vì chính sách hiện hành (v2024) là 15 ngày phép năm. | 0 | True |
| 46 | Nhân viên thử việc không được nghỉ phép năm và phải xin nghỉ không lương nếu cần. | 1 | Câu trả lời chính xác theo reference, nêu rõ rằng nhân viên thử việc không được nghỉ phép năm và phải xin nghỉ không lương nếu cần. | 1 | True |
| 50 | Được, miễn là đảm bảo kết nối an toàn. | 0 | Câu trả lời sai vì cho phép sử dụng VPN cá nhân trong khi chính sách rõ ràng cấm điều này. | 0 | True |

Cohen’s κ: 1.0.
Observed/chance agreement và diễn giải:
- Observed agreement $P_o = 10 / 10 = 1.0$.
- Chance agreement $P_e = (6/10) \times (6/10) + (4/10) \times (4/10) = 0.36 + 0.16 = 0.52$.
- Cohen's $\kappa = \frac{P_o - P_e}{1 - P_e} = \frac{1.0 - 0.52}{1.0 - 0.52} = 1.0$.
Kết quả cho thấy sự tương đồng tuyệt đối giữa nhãn của LLM Judge và chuyên gia con người trên 10 câu hỏi calibration.

## Giới hạn và cách dùng judge

- **Model và prompt:** Sử dụng `gpt-4o-mini` làm LLM Judge với cấu trúc prompt yêu cầu định dạng JSON nghiêm ngặt, tách biệt rõ ràng giữa câu hỏi, câu trả lời cần chấm và ground truth chuẩn.
- **Kích thước mẫu:** Mặc dù đạt $\kappa = 1.0$, mẫu 10 câu calibration và 5 cặp A/B là tương đối nhỏ. Trong môi trường production quy mô lớn, $\kappa$ thực tế thường dao động trong khoảng 0.7 - 0.85 do các sắc thái ngôn từ phức tạp.
- **Khuyến nghị vận hành:** Không coi LLM Judge là chân lý tuyệt đối (ground truth tối hậu). Cần thiết lập cơ chế human-in-the-loop: tự động lấy mẫu ngẫu nhiên 5% - 10% các phán quyết của Judge để con người phúc tra định kỳ, đặc biệt là các ca có độ tương đồng điểm số suýt soát hoặc các câu trả lời phủ định.

