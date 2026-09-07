# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chat-completion proxying for Model Playground sources.

A playground *source* is one of:

* ``external_openai`` — an OpenAI-compatible endpoint (base URL + optional API
  key + model id). Proxied server-side so tester browsers never see the key.
* a hosted vLLM instance — resolved through ``core.inference.vllm`` by
  instance id, or by finding the running instance of a local source.

All requests go out as standard OpenAI ``/chat/completions`` calls with SSE
streaming, which vLLM, llama.cpp's server, OpenAI, and every compatible
endpoint speak.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import httpx

from loggers import get_logger

from storage import playground_db
from storage.credential_secrets import get_secret

logger = get_logger(__name__)

PLAYGROUND_API_KEY_KIND = "playground_api_key"

_PROXY_TIMEOUT_S = 300.0
_CONNECT_TIMEOUT_S = 15.0


class PlaygroundProxyError(RuntimeError):
    """Base error for playground proxy failures."""


class SourceDownError(PlaygroundProxyError):
    """The source endpoint refused the connection or returned 5xx."""


class SourceNotReadyError(PlaygroundProxyError):
    """A hosted instance exists but is not running."""


@dataclass
class ProxyTarget:
    """Where to send a completion request."""

    base_url: str
    model: str
    api_key: Optional[str] = None
    source_kind: str = "external_openai"
    instance_id: Optional[str] = None
    model_identity: str = ""  # Human-readable real identity for storage.


def resolve_source_api_key(source: dict) -> Optional[str]:
    """Fetch the stored API key for a source, if any."""
    if not source.get("api_key_set"):
        return None
    return get_secret(PLAYGROUND_API_KEY_KIND, source["id"])


def resolve_source_target(source: dict) -> ProxyTarget:
    """Resolve a source row into a concrete proxy target.

    For hosted sources this prefers the live vLLM instance registered for the
    source. External sources resolve directly.
    """
    kind = source["kind"]
    if kind == "external_openai":
        base_url = (source.get("ref") or "").rstrip("/")
        if not base_url:
            raise PlaygroundProxyError(f"Source '{source['name']}' has no base URL")
        return ProxyTarget(
            base_url = base_url,
            model = source.get("external_model") or "",
            api_key = resolve_source_api_key(source),
            source_kind = kind,
            model_identity = f"{source['name']} · {source.get('external_model') or base_url}",
        )

    # local_dir / hf_model: proxy through a hosted vLLM instance of the source.
    from core.inference.vllm import get_vllm_backend

    backend = get_vllm_backend()
    for instance in playground_db.list_running_instances():
        if instance["source_id"] == source["id"] and instance["status"] == "running":
            info = backend.proxy_info(instance["id"])
            if info is None:
                continue
            return ProxyTarget(
                base_url = info["base_url"],
                model = instance["model_slug"],
                api_key = info["api_key"],
                source_kind = "hosted_vllm",
                instance_id = instance["id"],
                model_identity = f"{source['name']} · vLLM",
            )
    raise SourceNotReadyError(
        f"Source '{source['name']}' has no running hosted instance. "
        "Start hosting it first (Models tab)."
    )


def _headers(target: ProxyTarget) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if target.api_key:
        headers["Authorization"] = f"Bearer {target.api_key}"
    return headers


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(_PROXY_TIMEOUT_S, connect = _CONNECT_TIMEOUT_S)


@dataclass
class _OpenStream:
    """An open httpx response plus the client that owns its connection pool."""

    client: httpx.AsyncClient
    response: httpx.Response

    async def close(self) -> None:
        try:
            await self.response.aclose()
        finally:
            await self.client.aclose()


async def _open_request(
    target: ProxyTarget,
    messages: list[dict],
    params: dict,
    *,
    stream: bool,
    response_format: Optional[dict] = None,
) -> _OpenStream:
    url = f"{target.base_url}/chat/completions"
    payload: dict = {
        "model": target.model,
        "messages": messages,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    for key in (
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "seed",
        "top_k",
        "presence_penalty",
        "frequency_penalty",
    ):
        if key in params and params[key] is not None:
            payload[key] = params[key]
    if response_format is not None:
        payload["response_format"] = response_format
    client = httpx.AsyncClient(timeout = _timeout(), trust_env = False)
    request = client.build_request("POST", url, headers = _headers(target), json = payload)
    try:
        response = await client.send(request, stream = True)
    except httpx.HTTPError as exc:
        await client.aclose()
        raise SourceDownError(f"Source unreachable: {exc}") from exc
    if response.status_code >= 400:
        body = (await response.aread()).decode("utf-8", errors = "replace")[:400]
        await response.aclose()
        await client.aclose()
        if response.status_code >= 500:
            raise SourceDownError(f"Source error {response.status_code}: {body}")
        raise PlaygroundProxyError(f"Source rejected request ({response.status_code}): {body}")
    return _OpenStream(client = client, response = response)


@dataclass
class CompletionResult:
    content: str = ""
    reasoning: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    raw_finish: Optional[str] = None
    error: Optional[str] = None
    meta: dict = field(default_factory = dict)


def _delta_text(delta: dict) -> tuple[str, str]:
    """Extract (content, reasoning) from an OpenAI stream delta."""
    content = ""
    reasoning = ""
    if delta.get("content"):
        content += delta["content"]
    # vLLM/OpenAI reasoning content; some servers nest under reasoning_content.
    if delta.get("reasoning_content"):
        reasoning += delta["reasoning_content"]
    if delta.get("reasoning"):
        reason = delta["reasoning"]
        if isinstance(reason, str):
            reasoning += reason
        elif isinstance(reason, dict) and reason.get("content"):
            reasoning += str(reason["content"])
    return content, reasoning


async def _consume_stream(opened: _OpenStream) -> CompletionResult:
    """Read an SSE completion stream to completion, accumulating text."""
    started = time.perf_counter()
    result = CompletionResult()
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    response = opened.response
    try:
        async for line in response.aiter_lines():
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content, reasoning = _delta_text(delta)
                if content:
                    content_parts.append(content)
                if reasoning:
                    reasoning_parts.append(reasoning)
                finish = choices[0].get("finish_reason")
                if finish:
                    result.raw_finish = finish
            usage = chunk.get("usage")
            if isinstance(usage, dict):
                result.prompt_tokens = usage.get("prompt_tokens")
                result.completion_tokens = usage.get("completion_tokens")
    finally:
        await opened.close()
    result.content = "".join(content_parts)
    result.reasoning = "".join(reasoning_parts) or None
    result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
    return result


async def _consume_json(opened: _OpenStream) -> CompletionResult:
    started = time.perf_counter()
    try:
        body = await opened.response.aread()
        data = json.loads(body)
    except (json.JSONDecodeError, httpx.HTTPError) as exc:
        raise PlaygroundProxyError(f"Invalid response from source: {exc}") from exc
    finally:
        await opened.close()
    result = CompletionResult(latency_ms = round((time.perf_counter() - started) * 1000, 1))
    choices = data.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        result.content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if isinstance(reasoning, dict):
            reasoning = reasoning.get("content")
        result.reasoning = reasoning if isinstance(reasoning, str) and reasoning else None
        result.raw_finish = choices[0].get("finish_reason")
    usage = data.get("usage")
    if isinstance(usage, dict):
        result.prompt_tokens = usage.get("prompt_tokens")
        result.completion_tokens = usage.get("completion_tokens")
    return result


async def chat_completion_aggregate(
    target: ProxyTarget,
    messages: list[dict],
    params: Optional[dict] = None,
) -> CompletionResult:
    """Non-streaming completion: aggregate the full result server-side."""
    opened = await _open_request(target, messages, params or {}, stream = False)
    return await _consume_json(opened)


async def chat_completion(
    source: dict,
    messages: list[dict],
    params: Optional[dict] = None,
) -> CompletionResult:
    """Convenience: resolve a source row, then aggregate a completion."""
    target = resolve_source_target(source)
    return await chat_completion_aggregate(target, messages, params)


async def stream_chat_completion(
    target: ProxyTarget,
    messages: list[dict],
    params: Optional[dict] = None,
) -> AsyncIterator[dict]:
    """Yield normalized stream events: {type: delta|done, ...}.

    ``delta`` events carry ``text`` (or ``reasoning``); ``done`` carries the
    final CompletionResult fields. Errors raise SourceDownError /
    PlaygroundProxyError.
    """
    opened = await _open_request(target, messages, params or {}, stream = True)
    started = time.perf_counter()
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    prompt_tokens = completion_tokens = None
    finish = None
    try:
        async for line in opened.response.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content, reasoning = _delta_text(delta)
                if content:
                    content_parts.append(content)
                    yield {"type": "delta", "text": content}
                if reasoning:
                    reasoning_parts.append(reasoning)
                    yield {"type": "delta", "reasoning": reasoning}
                if choices[0].get("finish_reason"):
                    finish = choices[0].get("finish_reason")
            usage = chunk.get("usage")
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
    finally:
        await opened.close()
    yield {
        "type": "done",
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts) or None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "finish_reason": finish,
    }


async def test_source_connection(source: dict) -> dict:
    """Probe a source endpoint: list models / check reachability."""
    kind = source["kind"]
    if kind == "external_openai":
        base_url = (source.get("ref") or "").rstrip("/")
        headers = {}
        api_key = resolve_source_api_key(source)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            async with httpx.AsyncClient(
                timeout = httpx.Timeout(20.0, connect = _CONNECT_TIMEOUT_S),
                trust_env = False,
                headers = headers,
            ) as client:
                response = await client.get(f"{base_url}/models")
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"Unreachable: {exc}", "models": []}
        if response.status_code >= 400:
            return {
                "ok": False,
                "error": f"HTTP {response.status_code}",
                "models": [],
            }
        models: list[str] = []
        try:
            data = response.json()
            for item in data.get("data") or []:
                model_id = item.get("id")
                if model_id:
                    models.append(str(model_id))
        except (json.JSONDecodeError, AttributeError):
            pass
        return {"ok": True, "models": models}

    # Hosted sources: check whether any instance of the source is running.
    from core.inference.vllm import get_vllm_backend

    backend = get_vllm_backend()
    running = [
        instance
        for instance in playground_db.list_running_instances()
        if instance["source_id"] == source["id"]
    ]
    if not running:
        return {
            "ok": False,
            "error": "Not hosted. Start hosting this model first.",
            "models": [],
        }
    info = backend.proxy_info(running[0]["id"])
    if info is None:
        return {"ok": False, "error": "Hosted server is not responding.", "models": []}
    return {"ok": True, "models": [running[0]["model_slug"]]}
