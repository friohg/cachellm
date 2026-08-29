"""Minimal OpenAI-compatible upstream for trying CacheLLM without a real provider.

    python examples/mock_upstream.py --port 8099

Then point CacheLLM at it:

    UPSTREAM_BASE_URL=http://127.0.0.1:8099/v1 cachellm start

Every call is logged with an incrementing counter and the reply embeds that
counter, so a repeated answer proves the response came from the cache.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

CALLS = {"n": 0}


def _reply_text(body: dict) -> str:
    messages = body.get("messages") or []
    question = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            question = content if isinstance(content, str) else str(content)
            break
    if not question and isinstance(body.get("input"), str):
        question = body["input"]
    return f"[upstream call #{CALLS['n']}] you asked: {question}"


async def chat(request: Request):
    body = await request.json()
    CALLS["n"] += 1
    print(f"upstream call #{CALLS['n']} model={body.get('model')} stream={bool(body.get('stream'))}")
    text = _reply_text(body)

    if body.get("stream"):
        async def stream():
            base = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": body.get("model", "mock-model"),
            }
            for index, word in enumerate(text.split(" ")):
                piece = word if index == 0 else " " + word
                frame = {**base, "choices": [{"index": 0, "delta": {"content": piece},
                                              "finish_reason": None}]}
                yield f"data: {json.dumps(frame)}\n\n"
            yield "data: " + json.dumps(
                {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            ) + "\n\n"
            yield "data: " + json.dumps(
                {**base, "choices": [],
                 "usage": {"prompt_tokens": 25, "completion_tokens": 18, "total_tokens": 43}}
            ) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return JSONResponse(
        content={
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "mock-model"),
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": text},
                 "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 25, "completion_tokens": 18, "total_tokens": 43},
        }
    )


async def models(request: Request):
    return JSONResponse(
        content={"object": "list", "data": [{"id": "mock-model", "object": "model"}]}
    )


app = Starlette(
    routes=[
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/v1/responses", chat, methods=["POST"]),
        Route("/v1/models", models, methods=["GET"]),
    ]
)


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"mock upstream on http://{args.host}:{args.port}/v1")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
