from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import torch
import os, sys, json, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, asdict
from copy import deepcopy
from functools import lru_cache
from src.llm_runtime import sync_client, async_client, fingerprint, read_cache, write_cache, rate_limit_kind

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TEST_SET_PATH, OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
                    OPENROUTER_MODEL, RAGAS_EMBEDDING_MODEL)

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _metric_value(value):
    """Invalid or failed individual scores remain missing, rather than becoming zero."""
    try:
        number = float(value)
        return number if math.isfinite(number) and 0 <= number <= 1 else None
    except (TypeError, ValueError):
        return None


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _get_eval_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=RAGAS_EMBEDDING_MODEL,
                                 encode_kwargs={"normalize_embeddings": True})


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str], *, resume: bool = False,
                   batch_judge: bool = False) -> dict:
    """Run RAGAS evaluation."""
    fallback = {name: 0.0 for name in METRICS}
    fallback.update(per_question=[], valid_counts={name: 0 for name in METRICS},
                    metadata={"evaluator_models": [OPENROUTER_MODEL, RAGAS_EMBEDDING_MODEL]})
    try:
        if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
            raise ValueError("All evaluation inputs must have the same length")
        if not questions:
            return {**fallback, "status": "empty"}
        if not OPENROUTER_API_KEY:
            return {**fallback, "status": "skipped", "error": "OPENROUTER_API_KEY is not configured"}
        if resume:
            if batch_judge:
                return _evaluate_batch_resumable(questions, answers, contexts, ground_truths)
            return _evaluate_resumable(questions, answers, contexts, ground_truths)
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.run_config import RunConfig
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        dataset = Dataset.from_dict({"question": questions, "answer": answers,
                                     "contexts": contexts, "ground_truth": ground_truths})
        metrics = deepcopy([faithfulness, answer_relevancy, context_precision, context_recall])
        # Use one sample to reduce free-tier usage; keep baseline and production identical.
        metrics[1].strictness = 1
        judge = None
        if batch_judge:
            from src.batch_llm import BatchRagasLLM
            judge = BatchRagasLLM(batch_size=2)
        llm = judge if judge is not None else ChatOpenAI(model=OPENROUTER_MODEL, temperature=0, api_key=OPENROUTER_API_KEY,
                          base_url=OPENROUTER_BASE_URL, timeout=60, max_retries=0,
                          max_tokens=768, http_client=sync_client(), http_async_client=async_client())
        result = evaluate(dataset, metrics=metrics, llm=llm,
            embeddings=_get_eval_embeddings(),
            # A free-provider cooldown can outlast 60 seconds across the four
            # sequential RAGAS metrics for one question. The transport already
            # retries boundedly; let the full sample finish before timing it out.
            run_config=RunConfig(timeout=600, max_retries=0,
                                 max_workers=16 if batch_judge else 1), raise_exceptions=not batch_judge)
        rows = result.to_pandas().to_dict(orient="records")
        if len(rows) != len(questions):
            raise ValueError("RAGAS returned an unexpected number of rows")
        per_question = [EvalResult(question=questions[i], answer=answers[i], contexts=contexts[i],
            ground_truth=ground_truths[i], **{name: _metric_value(row.get(name)) for name in METRICS})
            for i, row in enumerate(rows)]
        aggregate, counts = {}, {}
        for name in METRICS:
            values = [getattr(row, name) for row in per_question if getattr(row, name) is not None]
            counts[name] = len(values)
            aggregate[name] = sum(values) / len(values) if values else 0.0
        status = "ok" if all(count == len(questions) for count in counts.values()) else "partial"
        if not any(counts.values()):
            status = "failed"
        output = {**aggregate, "per_question": per_question, "valid_counts": counts, "status": status,
                  "metadata": {"evaluator_models": [OPENROUTER_MODEL, RAGAS_EMBEDDING_MODEL]}}
        if judge is not None and judge.error is not None:
            output.update(status="rate_limited" if getattr(judge.error, "status_code", None) == 429 else "failed",
                          error=str(judge.error), rate_limit_kind=rate_limit_kind(judge.error))
        return output
    except Exception as exc:
        print(f"RAGAS evaluation failed: {exc}")
        status = "rate_limited" if getattr(exc, "status_code", None) == 429 else "failed"
        return {**fallback, "status": status, "error": str(exc), "rate_limit_kind": rate_limit_kind(exc)}


def _evaluate_resumable(questions, answers, contexts, ground_truths):
    """Checkpoint each scored question; transport also saves partial LLM work."""
    rows = []
    error, stopped_status, stopped_kind = None, None, None
    for i, (question, answer, context, truth) in enumerate(zip(questions, answers, contexts, ground_truths)):
        key = fingerprint({"version": 1, "model": OPENROUTER_MODEL,
            "endpoint": OPENROUTER_BASE_URL, "embedding": RAGAS_EMBEDDING_MODEL,
            "strictness": 1, "question": question, "answer": answer,
            "contexts": context, "ground_truth": truth})
        cached = read_cache("evaluations", key)
        if cached is not None:
            row = EvalResult(**cached)
            rows.append(row)
            print(f"  RAGAS [{i+1}/{len(questions)}]: cached", flush=True)
            continue
        result = evaluate_ragas([question], [answer], [context], [truth])
        if result["status"] in ("partial", "failed"):
            # Some chat-compatible providers return malformed judge output. A
            # parse/validation exception makes the primary result "failed" and
            # has no per-question row, so retry the sample through the independent
            # batch adapter (with its own prompt cache key) as well.
            alternate = evaluate_ragas([question], [answer], [context], [truth],
                                       resume=True, batch_judge=True)
            if alternate.get("status") == "rate_limited":
                result = alternate
            elif alternate.get("per_question"):
                alternate_row = alternate["per_question"][0]
                other = asdict(alternate_row) if isinstance(alternate_row, EvalResult) else dict(alternate_row)
                if result.get("per_question"):
                    primary_row = result["per_question"][0]
                    primary = asdict(primary_row) if isinstance(primary_row, EvalResult) else dict(primary_row)
                    for metric in METRICS:
                        if _metric_value(primary.get(metric)) is None and _metric_value(other.get(metric)) is not None:
                            primary[metric] = other[metric]
                    merged = EvalResult(**primary)
                else:
                    merged = EvalResult(**other)
                complete = all(getattr(merged, metric) is not None for metric in METRICS)
                result = {**{name: getattr(merged, name) if getattr(merged, name) is not None else 0.0 for name in METRICS},
                          "per_question": [merged],
                          "valid_counts": {name: int(getattr(merged, name) is not None) for name in METRICS},
                          "status": "ok" if complete else "partial"}
        if result["status"] != "ok":
            stopped_status = result["status"]
            error = result.get("error", "One or more metrics returned missing values")
            stopped_kind = result.get("rate_limit_kind")
            # Keep partial metrics visible, but only cache fully scored rows.
            rows.extend(result.get("per_question", []))
            if result["status"] != "partial":
                break
            continue
        row = result["per_question"][0]
        write_cache("evaluations", key, asdict(row))
        rows.append(row)
        print(f"  RAGAS [{i+1}/{len(questions)}]: saved", flush=True)
    counts, aggregate = {}, {}
    for name in METRICS:
        values = [getattr(row, name) for row in rows if getattr(row, name) is not None]
        counts[name] = len(values)
        aggregate[name] = sum(values) / len(values) if values else 0.0
    result = {**aggregate, "valid_counts": counts, "per_question": rows,
              "status": stopped_status or "ok", "num_expected": len(questions),
              "model": OPENROUTER_MODEL, "embedding_model": RAGAS_EMBEDDING_MODEL,
              "metadata": {"evaluator_models": [OPENROUTER_MODEL, RAGAS_EMBEDDING_MODEL]}}
    if error:
        result["error"] = error
        result["rate_limit_kind"] = stopped_kind
    return result


def _evaluate_batch_resumable(questions, answers, contexts, ground_truths):
    rows, missing, keys = {}, [], {}
    for i, (q, a, c, g) in enumerate(zip(questions, answers, contexts, ground_truths)):
        key = fingerprint({"policy": "independent-batch-v1", "model": OPENROUTER_MODEL,
            "endpoint": OPENROUTER_BASE_URL, "embedding": RAGAS_EMBEDDING_MODEL,
            "strictness": 1, "question": q, "answer": a, "contexts": c, "ground_truth": g})
        keys[i] = key
        cached = read_cache("evaluations", key)
        if cached:
            rows[i] = EvalResult(**cached)
        else:
            missing.append(i)
    error, kind, status = None, None, "ok"
    if missing:
        result = evaluate_ragas([questions[i] for i in missing], [answers[i] for i in missing],
            [contexts[i] for i in missing], [ground_truths[i] for i in missing], batch_judge=True)
        status, error, kind = result["status"], result.get("error"), result.get("rate_limit_kind")
        for i, row in zip(missing, result.get("per_question", [])):
            rows[i] = row
            if all(getattr(row, metric) is not None for metric in METRICS):
                write_cache("evaluations", keys[i], asdict(row))
    ordered = [rows[i] for i in sorted(rows)]
    counts, aggregate = {}, {}
    for metric in METRICS:
        values = [getattr(row, metric) for row in ordered if getattr(row, metric) is not None]
        counts[metric] = len(values)
        aggregate[metric] = sum(values) / len(values) if values else 0.0
    output = {**aggregate, "per_question": ordered, "valid_counts": counts,
        "status": status if all(count == len(questions) for count in counts.values()) or status != "ok" else "partial",
        "num_expected": len(questions), "model": OPENROUTER_MODEL, "embedding_model": RAGAS_EMBEDDING_MODEL,
        "judge_policy": "independent-batch-v1", "judge_batch_size": 2,
        "metadata": {"evaluator_models": [OPENROUTER_MODEL, RAGAS_EMBEDDING_MODEL]}}
    if error:
        output.update(error=error, rate_limit_kind=kind)
    return output


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 5) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []
    tree = {
        "faithfulness": ("Có thể có thông tin không được context hỗ trợ", "Siết prompt chỉ trả lời từ context; kiểm tra từng phát biểu", "Output sai → phát biểu có căn cứ trong context? → kiểm tra generation"),
        "context_recall": ("Có thể thiếu đoạn chứa thông tin cần thiết", "Kiểm tra chunking, hybrid retrieval và top-k", "Output sai → context đủ thông tin? → kiểm tra retrieval và chunking"),
        "context_precision": ("Có thể có nhiều đoạn không liên quan", "Kiểm tra reranking và lọc metadata/phiên bản", "Output sai → context có nhiều nhiễu? → kiểm tra reranking và filter"),
        "answer_relevancy": ("Câu trả lời có thể lệch yêu cầu câu hỏi", "Làm rõ prompt và kiểm tra câu hỏi nhiều ý", "Output sai → context phù hợp? → câu trả lời có đúng yêu cầu? → kiểm tra prompt"),
    }
    failures = []
    for result in eval_results:
        row = asdict(result) if isinstance(result, EvalResult) else dict(result)
        scores = {name: _metric_value(row.get(name)) for name in METRICS}
        if any(score is None for score in scores.values()):
            continue  # Incomplete evaluations must not be ranked as genuine failures.
        worst = min(scores, key=scores.get)
        diagnosis, fix, error_tree = tree[worst]
        failures.append({"question": row["question"], "answer": row["answer"],
            "ground_truth": row["ground_truth"], "contexts": row["contexts"],
            "score": sum(scores.values()) / len(METRICS), "worst_metric": worst,
            "worst_score": scores[worst], "metrics": scores, "diagnosis": diagnosis,
            "suggested_fix": fix, "error_tree": error_tree})
    return sorted(failures, key=lambda row: row["score"])[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {"model": OPENROUTER_MODEL, "embedding_model": RAGAS_EMBEDDING_MODEL,
                      **{k: v for k, v in results.items() if k not in ("per_question", "generation_samples")}},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
        "per_question": [asdict(row) if isinstance(row, EvalResult) else row
                         for row in results.get("per_question", [])],
        "generation_samples": results.get("generation_samples", []),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
