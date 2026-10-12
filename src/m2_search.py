from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys, re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    if not text.strip():
        return ""
    from underthesea import word_tokenize
    return word_tokenize(text, format="text").replace("_", " ")


def _bm25_tokens(text: str) -> list[str]:
    """Use the same casing and punctuation rules for documents and queries."""
    return re.findall(r'[^\W_]+', segment_vietnamese(text).casefold())


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi
        documents, tokens = [], []
        for chunk in chunks:
            tokenized = _bm25_tokens(chunk["text"])
            if tokenized:
                documents.append({"text": chunk["text"], "metadata": dict(chunk.get("metadata", {}))})
                tokens.append(tokenized)
        self.documents, self.corpus_tokens = documents, tokens
        self.bm25 = BM25Okapi(tokens) if tokens else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0 or not query.strip():
            return []
        tokens = _bm25_tokens(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [SearchResult(text=self.documents[i]["text"], score=float(scores[i]),
                             metadata=dict(self.documents[i]["metadata"]), method="bm25")
                for i in indices if scores[i] > 0][:top_k]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception as exc:
            print(f"Qdrant unavailable; using temporary in-memory storage: {exc}")
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct
        documents = [c for c in chunks if c["text"].strip()]
        # Encode before replacing an existing index, so a model failure leaves it intact.
        vectors = self._get_encoder().encode(
            [c["text"] for c in documents], show_progress_bar=True,
        ) if documents else []
        if any(len(vector) != EMBEDDING_DIM for vector in vectors):
            raise ValueError(f"Embedding dimension must be {EMBEDDING_DIM}")
        if len(vectors) != len(documents):
            raise ValueError("Expected one embedding per document")
        if self.client.collection_exists(collection_name=collection):
            self.client.delete_collection(collection_name=collection)
        self.client.create_collection(collection_name=collection, vectors_config=VectorParams(
            size=EMBEDDING_DIM, distance=Distance.COSINE,
        ))
        for start in range(0, len(documents), 64):
            points = [PointStruct(id=i, vector=list(map(float, vectors[i])), payload={
                **documents[i].get("metadata", {}), "text": documents[i]["text"],
            }) for i in range(start, min(start + 64, len(documents)))]
            self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0 or not query.strip():
            return []
        if not self.client.collection_exists(collection_name=collection):
            return []
        if self.client.count(collection_name=collection, exact=True).count == 0:
            return []
        vector = self._get_encoder().encode(query)
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(collection_name=collection,
                query=list(map(float, vector)), limit=top_k, with_payload=True)
            points = response.points
        else:
            points = self.client.search(collection_name=collection,
                query_vector=list(map(float, vector)), limit=top_k, with_payload=True)
        return [SearchResult(text=point.payload["text"], score=float(point.score),
                             metadata={key: value for key, value in point.payload.items() if key != "text"},
                             method="dense")
                for point in points if point.payload and "text" in point.payload]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if top_k <= 0:
        return []
    scores, originals = {}, {}
    for results in results_list:
        seen = set()
        for rank, result in enumerate(results, start=1):
            if result.text in seen:
                continue
            seen.add(result.text)
            scores[result.text] = scores.get(result.text, 0.0) + 1.0 / (k + rank)
            originals.setdefault(result.text, result)
    texts = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [SearchResult(text=text, score=scores[text],
                         metadata=dict(originals[text].metadata), method="hybrid") for text in texts]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
