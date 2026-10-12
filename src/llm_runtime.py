"""Disk caching and one shared request gate for SDK and async RAGAS calls.

Cache keys include endpoint and the complete request body, never API keys.
Only complete successful completions are cached. Rate limits with a reset hint
are retried after waiting, with a bounded attempt count.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

import httpx

from config import CACHE_DIR, OPENROUTER_REQUEST_INTERVAL

MAX_RATE_LIMIT_RETRIES = 5
MAX_AUTOMATIC_RATE_LIMIT_WAIT = 300  # Do not hold a terminal for long account/day resets.


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def read_cache(namespace: str, key: str):
    try:
        return json.loads((Path(CACHE_DIR) / namespace / f"{key}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_cache(namespace: str, key: str, value):
    path = Path(CACHE_DIR) / namespace / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class RequestGate:
    def __init__(self, interval: float = OPENROUTER_REQUEST_INTERVAL):
        self.interval = interval
        self.lock = threading.Lock()

    def wait(self):
        # Persist the last attempt so restarting a process does not reset its pacing.
        # One pipeline process at a time; this lock covers sync and async clients.
        with self.lock:
            previous = read_cache("runtime", "last_request") or {}
            delay = max(0, float(previous.get("time", 0)) + self.interval - time.time())
            if delay:
                time.sleep(delay)
            write_cache("runtime", "last_request", {"time": time.time()})


GATE = RequestGate()
QUOTA_LOCK = threading.Lock()


def _scope(host, credential):
    return fingerprint({"endpoint": str(host), "credential": credential})


def _reserve_quota(request, scope, body):
    if not (body.get("model", "").endswith(":free") or body.get("model") == "openrouter/free"):
        return None
    with QUOTA_LOCK:
        state = read_cache("quota", scope)
        day = time.strftime("%Y-%m-%d", time.gmtime())
        if not state or state.get("day") != day:
            return None
        if state["remaining"] <= 0:
            return httpx.Response(429, request=request, json={"error": {"code": 429,
                "message": "Daily free quota exhausted; stopped locally before sending another model request.",
                "metadata": {"limit_source": "openrouter_free_tier_daily"}}})
        state["remaining"] -= 1
        write_cache("quota", scope, state)
    return None


def _prepare(request: httpx.Request):
    if request.method != "POST" or not request.url.path.endswith("/chat/completions"):
        return request, None, None
    body = json.loads(request.content)
    model = body.get("model", "")
    if (request.url.host == "openrouter.ai"
            and (model.startswith("qwen/") or model.startswith("nvidia/nemotron-3.5-lightning"))):
        # Keep the free-model output budget for the answer instead of hidden reasoning.
        body.setdefault("reasoning", {"enabled": False})
        request = httpx.Request(request.method, request.url,
            headers={k: v for k, v in request.headers.items() if k != "content-length"},
            json=body, extensions=request.extensions)
    # Bump after changing retry/cooldown handling so bad cached judge output from
    # an earlier provider run cannot make a partial metric permanent.
    key = fingerprint({"version": 2, "endpoint": str(request.url), "body": body})
    return request, key, body


def _blocked_response(request):
    # Account-scoped cooldown; hash the credential only for scope, never save it.
    scope = _scope(request.url.host, request.headers.get("authorization", ""))
    state = read_cache("cooldown", scope) or {}
    if state.get("until", 0) > time.time():
        return scope, httpx.Response(429, request=request, json={"error": {
            "code": 429, "message": "OpenRouter cooldown active; cached progress is preserved.",
            "metadata": {"limit_source": state.get("kind", "unknown"),
                "headers": {"X-RateLimit-Reset": str(int(state["until"] * 1000))}}}})
    return scope, None


def _retry_delay_seconds(response):
    """Read provider backoff hints such as “try again in 18.36s”."""
    try:
        message = response.json().get("error", {}).get("message", "")
    except (ValueError, TypeError, AttributeError):
        message = ""
    if "request too large" in str(message).lower():
        return None  # This is a permanent request-size error, not a cooldown.
    for header in ("retry-after", "x-ratelimit-reset-tokens"):
        value = response.headers.get(header)
        if not value:
            continue
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(ms|s|sec|seconds|m|min)?", value, re.I)
        if match:
            amount, unit = float(match.group(1)), (match.group(2) or "s").lower()
            return amount / 1000 if unit == "ms" else amount * (60 if unit in ("m", "min") else 1)
    reset = response.headers.get("x-ratelimit-reset")
    if reset:
        try:
            value = float(reset)
            return max(0.0, value / 1000 - time.time()) if value > 1_000_000_000_000 else max(0.0, value)
        except ValueError:
            pass
    match = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*(ms|s|sec|seconds|m|min)?", message, re.I)
    if not match:
        return None
    amount, unit = float(match.group(1)), (match.group(2) or "s").lower()
    return amount / 1000 if unit == "ms" else amount * (60 if unit in ("m", "min") else 1)


def _record_response(response, key, body, scope, retry_delay=None):
    if response.status_code == 429:
        until = time.time() + (retry_delay + 1 if retry_delay is not None else 60)
        kind = "unknown"
        try:
            error = response.json().get("error", {})
            kind = error.get("metadata", {}).get("limit_source", "unknown")
            headers = error.get("metadata", {}).get("headers", {})
            reset = headers.get("X-RateLimit-Reset")
            if reset:
                until = max(until, float(reset) / 1000)
            elif response.headers.get("retry-after"):
                until = time.time() + float(response.headers["retry-after"])
        except (ValueError, TypeError, AttributeError):
            pass
        write_cache("cooldown", scope, {"until": until, "kind": kind})
        return
    if response.status_code != 200:
        return
    try:
        value = response.json()
        choices = value.get("choices", [])
        if not choices or any(c.get("finish_reason") != "stop" or not
                (c.get("message", {}).get("content") or "").strip() for c in choices):
            return
        if body.get("response_format", {}).get("type") in ("json_object", "json_schema"):
            for choice in choices:
                if not isinstance(json.loads(choice["message"]["content"]), dict):
                    return
        write_cache("completions", key, value)
    except (ValueError, TypeError, AttributeError):
        return


class CachedTransport(httpx.BaseTransport):
    def __init__(self, inner=None):
        self.inner = inner if inner is not None else httpx.HTTPTransport()

    def handle_request(self, request):
        request, key, body = _prepare(request)
        if key is None:
            return self.inner.handle_request(request)
        cached = read_cache("completions", key)
        if cached is not None:
            return httpx.Response(200, json=cached, request=request)
        scope, blocked = _blocked_response(request)
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            if blocked is not None:
                response = blocked
                blocked = None
            else:
                GATE.wait()
                exhausted = _reserve_quota(request, scope, body)
                if exhausted is not None:
                    return exhausted
                response = self.inner.handle_request(request)
                response.read()
            delay = _retry_delay_seconds(response) if response.status_code == 429 else None
            _record_response(response, key, body, scope, delay)
            if response.status_code != 429 or delay is None or attempt >= MAX_RATE_LIMIT_RETRIES:
                return response
            if delay > MAX_AUTOMATIC_RATE_LIMIT_WAIT:
                print(f"OpenRouter cooldown is {delay / 3600:.1f} hours; stopping now. "
                      "Progress is cached; rerun after reset or with another valid key.", flush=True)
                return response
            print(f"OpenRouter rate limit; đợi {delay + 1:.1f}s rồi thử lại ({attempt + 1}/{MAX_RATE_LIMIT_RETRIES}).", flush=True)
            response.close()
            time.sleep(delay + 1)
            request = httpx.Request(request.method, request.url, headers=request.headers,
                                    content=request.content, extensions=request.extensions)
        return response

    def close(self):
        self.inner.close()


class AsyncCachedTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner=None):
        self.inner = inner if inner is not None else httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request):
        request, key, body = _prepare(request)
        if key is None:
            return await self.inner.handle_async_request(request)
        cached = read_cache("completions", key)
        if cached is not None:
            return httpx.Response(200, json=cached, request=request)
        scope, blocked = _blocked_response(request)
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            if blocked is not None:
                response = blocked
                blocked = None
            else:
                await asyncio.to_thread(GATE.wait)
                exhausted = _reserve_quota(request, scope, body)
                if exhausted is not None:
                    return exhausted
                response = await self.inner.handle_async_request(request)
                await response.aread()
            delay = _retry_delay_seconds(response) if response.status_code == 429 else None
            _record_response(response, key, body, scope, delay)
            if response.status_code != 429 or delay is None or attempt >= MAX_RATE_LIMIT_RETRIES:
                return response
            if delay > MAX_AUTOMATIC_RATE_LIMIT_WAIT:
                print(f"OpenRouter cooldown is {delay / 3600:.1f} hours; stopping now. "
                      "Progress is cached; rerun after reset or with another valid key.", flush=True)
                return response
            print(f"OpenRouter rate limit; đợi {delay + 1:.1f}s rồi thử lại ({attempt + 1}/{MAX_RATE_LIMIT_RETRIES}).", flush=True)
            await response.aclose()
            await asyncio.sleep(delay + 1)
            request = httpx.Request(request.method, request.url, headers=request.headers,
                                    content=request.content, extensions=request.extensions)
        return response

    async def aclose(self):
        await self.inner.aclose()


def sync_client():
    return httpx.Client(transport=CachedTransport(), timeout=60)


def async_client():
    return httpx.AsyncClient(transport=AsyncCachedTransport(), timeout=60)


def check_api():
    """Validate the OpenRouter API key and configured model without a completion."""
    from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not configured")
    with httpx.Client(timeout=30) as client:
        response = client.get(f"{OPENROUTER_BASE_URL.rstrip('/')}/models",
                              headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"})
        response.raise_for_status()
        models = response.json().get("data", [])
    available = {entry.get("id") for entry in models if isinstance(entry, dict)}
    if available and OPENROUTER_MODEL not in available:
        raise ValueError(f"Configured model {OPENROUTER_MODEL!r} is not available through this OpenRouter key")
    return {"provider": "OpenRouter", "model": OPENROUTER_MODEL,
            "base_url": OPENROUTER_BASE_URL, "model_visible": OPENROUTER_MODEL in available}


def rate_limit_kind(exc) -> str:
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return "unknown"
    error = body.get("error", body)
    source = error.get("metadata", {}).get("limit_source")
    if source:
        return source
    message = str(error.get("message", "")).lower()
    if error.get("code") == "rate_limit_exceeded":
        if "free-models-per-day" in message or "daily free quota" in message:
            return "openrouter_free_tier_daily"
        if "output tokens per minute" in message or "otpm" in message:
            return "openrouter_output_tokens_per_minute"
        if "tokens per minute" in message or error.get("type") == "tokens":
            return "openrouter_token_limit"
        return "openrouter_request_limit"
    if "free-models-per-day" in message or "daily free quota" in message:
        return "openrouter_free_tier_daily"
    return "unknown"
