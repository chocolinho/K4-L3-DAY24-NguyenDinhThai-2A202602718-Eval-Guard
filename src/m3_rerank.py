from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents or not query.strip() or top_k <= 0:
            return []
        candidates = [doc for doc in documents if doc["text"].strip()]
        if not candidates:
            return []
        import numpy as np
        pairs = [(query, doc["text"]) for doc in candidates]
        scores = np.asarray(self._load_model().predict(pairs), dtype=float).reshape(-1)
        if len(scores) != len(candidates):
            raise ValueError("CrossEncoder must return one score per document")
        if not np.isfinite(scores).all():
            raise ValueError("CrossEncoder returned non-finite scores")
        scored = sorted(zip(scores, candidates), key=lambda item: item[0], reverse=True)
        return [RerankResult(
            text=doc["text"], original_score=float(doc.get("score", 0.0)),
            rerank_score=float(score), metadata=dict(doc.get("metadata", {})), rank=rank,
        ) for rank, (score, doc) in enumerate(scored[:top_k], start=1)]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not query.strip() or not documents or top_k <= 0:
            return []
        from flashrank import Ranker, RerankRequest
        candidates = [doc for doc in documents if doc["text"].strip()]
        if not candidates:
            return []
        if self._model is None:
            self._model = Ranker()
        passages = [{"id": i, "text": doc["text"]} for i, doc in enumerate(candidates)]
        results = self._model.rerank(RerankRequest(query=query, passages=passages))
        results = sorted(results, key=lambda item: float(item["score"]), reverse=True)
        return [RerankResult(text=candidates[item["id"]]["text"],
            original_score=float(candidates[item["id"]].get("score", 0)),
            rerank_score=float(item["score"]),
            metadata=dict(candidates[item["id"]].get("metadata", {})), rank=rank)
            for rank, item in enumerate(results[:top_k], start=1)]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark end-to-end latency; the first run may include model loading."""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
