"""Batch independent LLM tasks; RAGAS still computes its own four metrics.

Each job retains its original prompt and response. Only API transport is batched,
not the metric formulas. The batch policy is recorded in evaluation reports.
"""
from __future__ import annotations

import asyncio
import json

from openai import AsyncOpenAI, OpenAI
from langchain_core.outputs import Generation, LLMResult
from ragas.llms.base import BaseRagasLLM

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL
from src.llm_runtime import async_client, sync_client, fingerprint, read_cache, write_cache


BATCH_INSTRUCTION = (
    "Process each independent job separately. Each job contains its own complete instructions. "
    "Use only that job's inputs; never mix evidence or outputs between jobs. "
    "Return one JSON object: {\"results\": [{\"id\": 0, \"output\": \"the exact assistant response requested by job 0, as a string\"}, ...]}. "
    "Include every job ID exactly once. Preserve each job's requested JSON schema inside its output string. "
    "Do not assign RAGAS metric scores yourself; answer each supplied prompt exactly."
)


def validate_outputs(value: dict, count: int) -> list[str]:
    rows = value.get("results")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("Batch response omitted jobs")
    outputs = {}
    for row in rows:
        if (not isinstance(row, dict) or type(row.get("id")) is not int
                or row["id"] in outputs or not isinstance(row.get("output"), str)
                or not row["output"].strip()):
            raise ValueError("Invalid or duplicated batch output")
        outputs[row["id"]] = row["output"]
    if set(outputs) != set(range(count)):
        raise ValueError("Unexpected batch output IDs")
    return [outputs[i] for i in range(count)]


def parse_json_object(content: str) -> dict:
    """Parse a JSON object even when the model surrounds it with prose/fences."""
    try:
        value = json.loads(content)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for offset, char in enumerate(content):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(content[offset:])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                continue
    raise ValueError("Model response did not contain a JSON object")


class BatchRagasLLM(BaseRagasLLM):
    def __init__(self, batch_size: int = 2):
        super().__init__()
        self.batch_size = batch_size
        self.pending = []
        self.worker = None
        self.client = None
        self.error = None

    def generate_text(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None):
        raise NotImplementedError("BatchRagasLLM requires asynchronous RAGAS evaluation")

    async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        if n != 1:
            raise ValueError("Batch judge requires answer_relevancy.strictness=1")
        text = prompt.to_string()
        key = fingerprint({"policy": "independent-batch-v1", "model": OPENROUTER_MODEL,
            "endpoint": OPENROUTER_BASE_URL, "prompt": text, "stop": stop})
        cached = read_cache("ragas_prompts", key)
        if cached is not None:
            return LLMResult(generations=[[Generation(text=cached["output"])]])
        if self.error is not None:
            raise self.error
        future = asyncio.get_running_loop().create_future()
        self.pending.append((text, key, future))
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self._drain())
        output = await future
        return LLMResult(generations=[[Generation(text=output)]])

    async def _drain(self):
        try:
            self.client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL,
                timeout=240, max_retries=0, http_client=async_client())
            async with self.client:
                while self.pending:
                    await asyncio.sleep(0.1)  # Gather jobs scheduled in this event-loop turn.
                    batch, self.pending = self.pending[:self.batch_size], self.pending[self.batch_size:]
                    # Stable ordering gives the same request/cache key after resuming.
                    batch.sort(key=lambda item: item[1])
                    try:
                        response = await self.client.chat.completions.create(model=OPENROUTER_MODEL,
                            temperature=0, max_tokens=768,
                            messages=[{"role": "system", "content": BATCH_INSTRUCTION},
                                {"role": "user", "content": json.dumps({"jobs": [
                                    {"id": i, "prompt": task[0]} for i, task in enumerate(batch)]}, ensure_ascii=False)}])
                        if response.choices[0].finish_reason != "stop":
                            raise ValueError("Batch judge response was truncated")
                        outputs = validate_outputs(parse_json_object(response.choices[0].message.content or ""), len(batch))
                        for (_, key, future), output in zip(batch, outputs):
                            write_cache("ragas_prompts", key, {"output": output})
                            if not future.done():
                                future.set_result(output)
                    except Exception as exc:
                        self.error = exc
                        for _, _, future in batch + self.pending:
                            if not future.done():
                                future.set_exception(exc)
                        self.pending.clear()
                        return
        except Exception as exc:
            self.error = exc
            for _, _, future in self.pending:
                if not future.done():
                    future.set_exception(exc)
            self.pending.clear()
        finally:
            self.worker = None
            if self.pending and self.error is None:
                self.worker = asyncio.create_task(self._drain())


def answer_batch(questions: list[str], contexts: list[list[str]]) -> list[str]:
    """Generate all missing answers in one request, preserving per-query evidence."""
    if not questions:
        return []
    jobs = [{"id": i, "prompt": "Trả lời CHỈ từ context của câu hỏi này. "
        "Nếu thiếu bằng chứng, nói không tìm thấy thông tin. Trả lời ngắn gọn bằng tiếng Việt.\n"
        + "Context:\n" + "\n\n".join(contexts[i]) + "\nCâu hỏi: " + question}
        for i, question in enumerate(questions)]
    with OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL, timeout=240,
                max_retries=0, http_client=sync_client()) as client:
        response = client.chat.completions.create(model=OPENROUTER_MODEL, temperature=0,
            max_tokens=8192, messages=[
                {"role": "system", "content": BATCH_INSTRUCTION},
                {"role": "user", "content": json.dumps({"jobs": jobs}, ensure_ascii=False)}])
    if response.choices[0].finish_reason != "stop":
        raise ValueError("Batch answer response was truncated")
    return validate_outputs(parse_json_object(response.choices[0].message.content or ""), len(questions))
