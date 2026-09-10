#!/usr/bin/env python
"""Mock OpenAI-compatible server for Model Battleground end-to-end testing.

Stands in for external LLM endpoints (and vLLM) on machines without a GPU:
- GET  /v1/models            -> model catalog
- POST /v1/chat/completions  -> SSE streaming completion

Behavior: normal chats get an echo-style deterministic response that mentions
the model id; requests carrying the judge system prompt (contains "impartial
judge") get a valid judge JSON verdict whose winner is derived from a stable
hash of the prompt so reports have interesting, reproducible data.

Usage: python battleground_mock_openai.py [--port 8899] [--key sk-mock]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

app = FastAPI(title = "Mock OpenAI server")

API_KEY: Optional[str] = None


def _usage(messages: list[dict]) -> tuple[int, int]:
    prompt_tokens = sum(len(str(m.get("content", "")).split()) + 4 for m in messages)
    return prompt_tokens, 24


def _normal_reply(model: str, messages: list[dict]) -> str:
    last_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
    )
    words = last_user.split()
    topic = " ".join(words[:8]) if words else "your message"
    return (
        f"[{model}] Here is my response to: \"{topic}\". "
        f"I produce a clear, structured answer with {(len(words) % 5) + 2} key points. "
        "This is a deterministic mock response for battleground end-to-end testing."
    )


def _judge_reply(messages: list[dict]) -> str:
    """Deterministic judge verdict from a stable hash of the pair text."""
    payload = json.dumps(messages, sort_keys = True)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    pick = int(digest[:2], 16)
    winner = "a" if pick % 3 == 0 else ("b" if pick % 3 == 1 else "tie")
    confidence = round(0.6 + (pick % 40) / 100, 2)
    return json.dumps(
        {
            "winner": winner,
            "reason": f"Mock judge verdict (hash {digest[:8]}): response quality determined the winner.",
            "confidence": confidence,
        }
    )


def _rubric_reply(messages: list[dict]) -> str:
    payload = json.dumps(messages, sort_keys = True)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    base = 5 + int(digest[:2], 16) % 6  # 5..10
    return json.dumps(
        {
            "scores": {
                "correctness": base,
                "instruction_following": max(1, base - 1),
                "helpfulness": base,
                "clarity": max(1, base - 2),
                "safety": 9,
            },
            "reason": f"Mock rubric scores (hash {digest[:8]}).",
            "confidence": 0.8,
        }
    )


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "mock-model-a", "object": "model", "owned_by": "mock"},
            {"id": "mock-model-b", "object": "model", "owned_by": "mock"},
            {"id": "mock-model-judge", "object": "model", "owned_by": "mock"},
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if API_KEY:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {API_KEY}":
            raise HTTPException(status_code = 401, detail = "bad key")
    body = await request.json()
    messages = body.get("messages") or []
    model = body.get("model", "mock-model-a")
    stream = bool(body.get("stream"))
    system = " ".join(
        str(m.get("content", "")) for m in messages if m.get("role") == "system"
    )
    if "impartial judge" in system.lower():
        text = (
            _rubric_reply(messages) if "score it 1-10" in system.lower() else _judge_reply(messages)
        )
    else:
        text = _normal_reply(model, messages)
    prompt_tokens, completion_tokens = _usage(messages)

    if not stream:
        return {
            "id": f"chatcmpl-mock-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def gen():
        chunk_size = max(1, len(text) // 6)
        for i in range(0, len(text), chunk_size):
            piece = text[i : i + chunk_size]
            chunk = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.01)
        final = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type = "text/event-stream")


def main() -> None:
    global API_KEY
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type = int, default = 8899)
    parser.add_argument("--key", default = None)
    args = parser.parse_args()
    API_KEY = args.key
    uvicorn.run(app, host = "127.0.0.1", port = args.port, log_level = "warning")


if __name__ == "__main__":
    main()
