from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K
from src.llm_runtime import sync_client, fingerprint, read_cache, write_cache, rate_limit_kind


def build_pipeline(*, enrichment_backend: str = "llm"):
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    parent_chunks = {}
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        parent_chunks.update({parent.metadata["parent_id"]: parent.text for parent in parents})
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks, backend=enrichment_backend)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": {
            **e.auto_metadata, "original_text": e.original_text,
        }} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    search.parent_chunks = parent_chunks
    from config import EMBEDDING_MODEL
    search.corpus_fingerprint = fingerprint({"chunks": all_chunks,
        "parents": parent_chunks, "embedding": EMBEDDING_MODEL})
    search.enrichment_backend = enrichment_backend
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def retrieve_contexts(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> list[str]:
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = []
    parents = getattr(search, "parent_chunks", {})
    for result in reranked or results[:RERANK_TOP_K]:
        # Retrieval aids help search, but only source text grounds the answer.
        context = parents.get(result.metadata.get("parent_id"),
                              result.metadata.get("original_text", result.text))
        if context not in contexts:
            contexts.append(context)
    return contexts


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    contexts = retrieve_contexts(query, search, reranker)

    from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL
    if OPENROUTER_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL,
                            timeout=60, max_retries=0, http_client=sync_client())
            context_str = "\n\n".join(contexts)
            with client:
                resp = client.chat.completions.create(model=OPENROUTER_MODEL,
                    temperature=0, max_tokens=768, messages=[
                    {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
                ])
            answer = resp.choices[0].message.content
            if not answer or not answer.strip() or resp.choices[0].finish_reason != "stop":
                message = resp.choices[0].message
                reasoning = getattr(message, "reasoning", None) or ""
                raise ValueError(
                    "LLM returned an empty or incomplete answer "
                    f"(finish_reason={resp.choices[0].finish_reason}, "
                    f"content_chars={len(answer or '')}, reasoning_chars={len(reasoning)})"
                )
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            raise  # Failed generation must not become a cached/evaluated extract.
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker, *, batch_llm: bool = False):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []
    generation_error = None

    if batch_llm:
        from config import OPENROUTER_MODEL, OPENROUTER_BASE_URL
        from src.batch_llm import answer_batch
        try:
            identity = search.corpus_fingerprint
            batch_keys = [fingerprint({"stage": "production-batch-v1", "corpus": identity,
                "item": item, "model": OPENROUTER_MODEL, "endpoint": OPENROUTER_BASE_URL,
                "top_k": RERANK_TOP_K}) for item in test_set]
            missing = [i for i, key in enumerate(batch_keys) if read_cache("samples", key) is None]
            contexts_to_answer = [retrieve_contexts(test_set[i]["question"], search, reranker) for i in missing]
            batch_answers = answer_batch([test_set[i]["question"] for i in missing], contexts_to_answer)
            for i, answer, context in zip(missing, batch_answers, contexts_to_answer):
                write_cache("samples", batch_keys[i], {"answer": answer, "contexts": context})
        except Exception as exc:
            generation_error = ("rate_limited" if getattr(exc, "status_code", None) == 429 else "failed",
                                str(exc), rate_limit_kind(exc))

    for i, item in enumerate(test_set):
        if generation_error:
            break
        try:
            from config import OPENROUTER_MODEL, OPENROUTER_BASE_URL, OPENROUTER_API_KEY
            identity = getattr(search, "corpus_fingerprint", None)
            key = batch_keys[i] if batch_llm else fingerprint({"stage": "production-v2", "corpus": identity,
                "item": item, "model": OPENROUTER_MODEL, "endpoint": OPENROUTER_BASE_URL,
                "top_k": RERANK_TOP_K}) if identity and OPENROUTER_API_KEY else None
            sample = read_cache("samples", key) if key else None
            if sample is None:
                answer, contexts = run_query(item["question"], search, reranker)
                sample = {"answer": answer, "contexts": contexts}
                if key:
                    write_cache("samples", key, sample)
            answer, contexts = sample["answer"], sample["contexts"]
        except Exception as exc:
            generation_error = ("rate_limited" if getattr(exc, "status_code", None) == 429 else "failed",
                                str(exc), rate_limit_kind(exc))
            break
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    if generation_error:
        results = evaluate_ragas([], [], [], [])
        results.update(status=generation_error[0], error=generation_error[1])
        results.update(stage="generation", num_expected=len(test_set))
        results["rate_limit_kind"] = generation_error[2]
    else:
        results = evaluate_ragas(questions, answers, all_contexts, ground_truths, resume=True, batch_judge=batch_llm)
    results.update(generation_policy="independent-batch-v1" if batch_llm else "single-v2",
                   enrichment_backend=getattr(search, "enrichment_backend", "unknown"))
    results["generation_samples"] = [dict(question=q, answer=a, contexts=c, ground_truth=g)
        for q, a, c, g in zip(questions, answers, all_contexts, ground_truths)]
    print(f"  RAGAS status: {results.get('status', 'unknown')} ({time.time()-t0:.1f}s)", flush=True)
    if results.get("status") != "ok":
        print("  Evaluation incomplete: inspect status, error and valid_counts in the report.", flush=True)
        if results.get("error"):
            print(f"  Reason: {results['error']}", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if results.get("valid_counts", {}).get(m, 0) == 0:
            print(f"  {m}: N/A (chưa có điểm đánh giá hợp lệ)")
            continue
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
