# Nhóm thực hiện K4 Track 3A

Tên nhóm: K4-L3-DAY24-NguyenDinhThai-EvalGuard

Repo chung: https://github.com/chocolinho/K4-L3-DAY24-NguyenDinhThai-2A202602718-Eval-Guard

Trưởng nhóm/người nộp: Nguyễn Đình Thái - 2A202602718

Ngày lab/deadline áp dụng: 12/10/2026 23:59 — Asia/Ho_Chi_Minh (UTC+07:00)

Commit dùng để chấm: 0d771e8b2efa0175a9e0fe9880d56941b51c8e42

Mỗi nhóm nộp một repo. Điền đủ thành viên và một mục INDIVIDUAL cho từng người.
Thêm/xóa rows để khớp nhóm thật; mẫu ba rows không ép nhóm phải có ba người.

## Thành viên và phân công

| STT | Họ và tên | MSSV | Vai trò / công việc thật | Bằng chứng |
|---|---|---|---|---|
| 1 | Nguyễn Đình Thái | 2A202602718 | Toàn bộ Phase A, Phase B, Phase C, tích hợp RAG pipeline và hoàn thiện báo cáo | Commits trên GitHub, artifacts/*.json, doc/reports/*.md |

Người tích hợp/chạy batch/nộp link: Nguyễn Đình Thái, thực hiện chạy batch script setup 50 câu, chạy toàn bộ runner Phase A/B/C và commit artifacts lên repo.

## Môi trường và chuẩn bị trước lớp

OS/Python/compiler: Windows 11 64-bit, Python 3.11.9, annoy 1.17.3 prebuilt wheel (bỏ qua MSVC requirement), pyarrow 18.1.0, spacy 3.8.4 với model en_core_web_lg.
Virtualenv/dependencies/imports: .venv, pip install -r requirements/requirements.txt, cài đặt thành công, tất cả imports ragas, nemoguardrails, presidio_analyzer, spacy hoạt động bình thường.
Models/Qdrant/50 answers đã chuẩn bị: Docker Qdrant (v1.9.7) port 6333; BAAI/bge-m3; BAAI/bge-reranker-v2-m3; OpenRouter openai/gpt-4o-mini; sinh đủ 50 câu trong artifacts/answers_50q.json ở chế độ llm.
NeMo Guardrails 0.10.0 cấu hình thành công với fastembed và OpenRouter, vượt qua 20/20 attacks và 8/8 clean controls.

## INDIVIDUAL

Sao chép mẫu cho mỗi thành viên; tiêu đề họ tên không dấu-MSSV.

### NguyenDinhThai-2A202602718

- Phần được giao và phần thực tế đã làm: Thực hiện độc lập toàn bộ các phần: Copy và chuẩn hóa 5 module Day 18 (m1–m5), sinh 50 câu trả lời RAG qua OpenRouter, cài đặt Tasks 1–4 trong src/phase_a_ragas.py, Tasks 5–8 trong src/phase_b_judge.py, Tasks 9a, 9b, 10, 11, 12 trong src/phase_c_guard.py, thực hiện live experiments và viết hoàn chỉnh 3 báo cáo.
- Bằng chứng: Các commit trên repo GitHub, 4 file artifact (answers_50q.json, ragas_50q.json, judge_results.json, guard_results.json) và 3 báo cáo (failure_clusters.md, bias_report.md, blueprint.md).
- Cách kiểm tra: Đã chạy offline pytest đạt 86/86 passed (python -m pytest tests/ -m "not integration"), chạy thực nghiệm live bằng scripts/run_experiments.py cho cả 3 phase, và kiểm tra hợp quy bằng python scripts/check_lab.py --with-tests.
- Điều đã học: Hiểu sâu về 4 tiêu chí đánh giá RAGAS (faithfulness, answer_relevancy, context_precision, context_recall), kỹ thuật LLM-as-a-Judge với position swapping và tính toán Cohen's kappa, kiến trúc bảo vệ đa tầng kết hợp Presidio PII Engine và NeMo Guardrails, cùng phương pháp đo lường percentile latency P50/P95/P99 theo chuẩn CI/CD.
- Giới hạn/cách xử lý: Gặp lỗi xung đột DLL c10.dll trên Windows khi load Torch sau các thư viện khác (đã giải quyết bằng cách preload Torch ở đầu entrypoint); giải quyết false positive của Presidio trên tiếng Việt bằng cách lọc đúng tập entity mục tiêu (VN_CCCD, VN_PHONE, EMAIL); tinh chỉnh prompt của NeMo Input Rails để tránh chặn nhầm câu hỏi chính sách mật khẩu sạch.
- Nguồn/AI hỗ trợ nếu có: Sử dụng Antigravity coding assistant hỗ trợ debug môi trường Windows và đối chiếu schema bài lab, toàn bộ kết quả thực nghiệm và số liệu đều được tự chạy kiểm chứng thực tế 100%.

## Cập nhật sau khi nộp

Bản nộp đầu tiên đúng hạn trước deadline; không có cập nhật sau deadline.

