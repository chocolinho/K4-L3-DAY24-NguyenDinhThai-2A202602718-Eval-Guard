"""Small live compatibility check and source-grounded lab analysis output."""
import json
from pathlib import Path


def print_status():
    """Read progress without loading models or making network calls."""
    for name in ("naive_baseline_report", "ragas_report"):
        path = Path("reports") / f"{name}.json"
        if not path.exists():
            print(f"{name}: chưa có báo cáo")
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        aggregate = report.get("aggregate", {})
        counts = aggregate.get("valid_counts", {})
        complete = min(counts.values()) if counts else 0
        print(f"{name}: {aggregate.get('status', 'unknown')}; "
              f"generated={len(report.get('generation_samples', []))}; fully_scored={complete}; "
              f"rate_limit_kind={aggregate.get('rate_limit_kind') or 'none'}")


def smoke_test(*, batch_llm: bool = False):
    from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL
    from openai import OpenAI
    from src.llm_runtime import sync_client
    from src.m5_enrichment import _request_json
    from src.m4_eval import evaluate_ragas, save_report, failure_analysis
    context = "Nhân viên chính thức được nghỉ phép năm 15 ngày làm việc mỗi năm."
    question = "Nhân viên chính thức được nghỉ phép năm bao nhiêu ngày?"
    enrichment = _request_json('Trả về JSON gồm summary (string), questions (list), '
        'context (string) và metadata (object). Không thêm dữ kiện.', context)
    if not (isinstance(enrichment.get("summary"), str) and enrichment["summary"].strip()
            and isinstance(enrichment.get("questions"), list)
            and isinstance(enrichment.get("context"), str)
            and isinstance(enrichment.get("metadata"), dict)):
        raise ValueError("Model did not return the expected enrichment JSON")
    with OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL,
                max_retries=0, timeout=60, http_client=sync_client()) as client:
        response = client.chat.completions.create(model=OPENROUTER_MODEL, temperature=0,
            max_tokens=768, messages=[
                {"role": "system", "content": "Trả lời CHỈ dựa trên context."},
                {"role": "user", "content": f"Context:\n{context}\n\nCâu hỏi: {question}"}])
    answer = response.choices[0].message.content
    if not answer or "15" not in answer or response.choices[0].finish_reason != "stop":
        raise ValueError("Model did not answer the Vietnamese lookup question correctly")
    scores = evaluate_ragas([question], [answer], [[context]], ["15 ngày làm việc mỗi năm."],
                            resume=True, batch_judge=batch_llm)
    save_report(scores, failure_analysis(scores.get("per_question", [])), "reports/smoke_test.json")
    print(f"Enrichment JSON: OK; Vietnamese answer: {answer}")
    print(f"RAGAS compatibility: {scores['status']}")
    if scores["status"] != "ok":
        raise RuntimeError(scores.get("error", "Smoke evaluation incomplete"))


def write_analysis():
    """Generate comparison and bottom-5 only from fully evaluated reports."""
    paths = [Path("reports/naive_baseline_report.json"), Path("reports/ragas_report.json")]
    if not all(path.exists() for path in paths):
        return False
    baseline, production = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(report.get("aggregate", {}).get("status") != "ok" for report in (baseline, production)):
        return False
    if (not baseline["aggregate"].get("model")
            or baseline["aggregate"].get("model") != production["aggregate"].get("model")):
        return False  # Do not compare experiments with different judge models.
    from src.m4_eval import load_test_set
    if any(report.get("num_questions") != len(load_test_set()) for report in (baseline, production)):
        return False
    from src.m4_eval import METRICS
    lines = ["# Failure Analysis — Lab 18: Production RAG", "",
        "**Học viên:** Nguyễn Đình Thái — K4 Track 3A", "",
        f"Model: `{production['aggregate'].get('model', 'unknown')}`. "
        f"Baseline: {baseline['num_questions']} câu; production: {production['num_questions']} câu.", "",
        "## So sánh RAGAS", "", "| Metric | Naive | Production | Δ |",
        "|---|---:|---:|---:|"]
    for metric in METRICS:
        n, p = baseline["aggregate"][metric], production["aggregate"][metric]
        lines.append(f"| {metric} | {n:.4f} | {p:.4f} | {p-n:+.4f} |")
    lines += ["", "## Bottom-5", "",
        "Các chẩn đoán dưới đây là giả thuyết theo metric thấp nhất, cần đối chiếu bằng chứng nguồn."]
    for i, row in enumerate(production.get("failures", []), 1):
        lines += ["", f"### #{i}: {row['question']}", "",
            f"- Expected: {row['ground_truth']}", f"- Answer: {row['answer']}",
            f"- Worst metric: {row['worst_metric']} = {row['worst_score']:.4f}",
            f"- Mean score: {row['score']:.4f}", f"- Error Tree: {row['error_tree']}",
            f"- Diagnosis: {row['diagnosis']}", f"- Suggested fix: {row['suggested_fix']}", "",
            "Bằng chứng được đưa vào generation:"]
        for context in row["contexts"]:
            lines += ["", *["> " + line for line in context.splitlines()]]
    Path("analysis/failure_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    reflection = Path("analysis/reflections/reflection_NguyenDinhThai.md")
    if reflection.exists():
        text = reflection.read_text(encoding="utf-8").split("\n## Kết quả đo thực tế")[0]
        text += "\n## Kết quả đo thực tế\n\n" + "\n".join(lines[8:15]) + "\n"
        reflection.write_text(text, encoding="utf-8")
    return True
