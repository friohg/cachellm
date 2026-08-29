"""Shared test fixtures.

Tests run the *real* proxy against a *real* in-process fake upstream: both are
ASGI apps wired together with ``httpx.ASGITransport``, so requests traverse the
actual HTTP layer, streaming code and cache engine - no monkeypatched internals.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from cachellm.app import AppState
from cachellm.config import Config, ModelPricing
from cachellm.providers.openai_compatible import OpenAICompatibleProvider
from cachellm.server import create_app


# ---------------------------------------------------------------------------
# fake upstream
# ---------------------------------------------------------------------------


@dataclass
class FakeUpstream:
    """Counts calls and can be told to fail, stall or truncate streams."""

    calls: int = 0
    stream_calls: int = 0
    seen_authorization: list[str] = field(default_factory=list)
    seen_payloads: list[dict[str, Any]] = field(default_factory=list)
    fail_status: int | None = None
    fail_body: dict[str, Any] | None = None
    delay: float = 0.0
    truncate_stream: bool = False
    stream_error: bool = False
    reply_text: str = "hello from upstream"
    include_usage: bool = True
    counter_suffix: bool = False
    """When true the reply text includes the call number, making cache hits obvious."""

    def reset(self) -> None:
        self.calls = 0
        self.stream_calls = 0
        self.seen_authorization.clear()
        self.seen_payloads.clear()

    def _text(self) -> str:
        return f"{self.reply_text} #{self.calls}" if self.counter_suffix else self.reply_text

    def build_app(self) -> Starlette:
        async def chat(request: Request) -> Any:
            body = await request.json()
            self.calls += 1
            self.seen_payloads.append(body)
            auth = request.headers.get("authorization")
            if auth:
                self.seen_authorization.append(auth)
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.fail_status:
                return JSONResponse(
                    status_code=self.fail_status,
                    content=self.fail_body
                    or {"error": {"message": "upstream exploded", "type": "server_error"}},
                )
            if body.get("stream"):
                self.stream_calls += 1
                return StreamingResponse(
                    self._stream(body), media_type="text/event-stream"
                )
            text = self._text()
            payload: dict[str, Any] = {
                "id": f"chatcmpl-{self.calls}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "unknown"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
            }
            if self.include_usage:
                payload["usage"] = {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                }
            return JSONResponse(content=payload)

        async def _stream_body(body: dict[str, Any]) -> AsyncIterator[bytes]:
            base = {
                "id": f"chatcmpl-stream-{self.stream_calls}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": body.get("model", "unknown"),
            }
            words = self._text().split(" ")
            for index, word in enumerate(words):
                piece = word if index == 0 else " " + word
                frame = {
                    **base,
                    "choices": [
                        {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(frame)}\n\n".encode()
                if self.truncate_stream and index == 0:
                    return  # stop mid-stream without [DONE]
                if self.stream_error and index == 0:
                    yield (
                        "data: "
                        + json.dumps({"error": {"message": "stream blew up"}})
                        + "\n\n"
                    ).encode()
                    return
            yield (
                "data: "
                + json.dumps(
                    {
                        **base,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                )
                + "\n\n"
            ).encode()
            if self.include_usage:
                yield (
                    "data: "
                    + json.dumps(
                        {
                            **base,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": 11,
                                "completion_tokens": 7,
                                "total_tokens": 18,
                            },
                        }
                    )
                    + "\n\n"
                ).encode()
            yield b"data: [DONE]\n\n"

        self._stream = _stream_body  # type: ignore[attr-defined]

        async def models(request: Request) -> Any:
            return JSONResponse(
                content={"object": "list", "data": [{"id": "test-model", "object": "model"}]}
            )

        async def embeddings(request: Request) -> Any:
            await request.json()
            self.calls += 1
            return JSONResponse(
                content={"data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]}
            )

        return Starlette(
            routes=[
                Route("/v1/chat/completions", chat, methods=["POST"]),
                Route("/v1/responses", chat, methods=["POST"]),
                Route("/v1/models", models, methods=["GET"]),
                Route("/v1/embeddings", embeddings, methods=["POST"]),
            ]
        )


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def upstream() -> FakeUpstream:
    return FakeUpstream()


def make_config(tmp_path, **overrides: Any) -> Config:
    config = Config()
    config.server.host = "127.0.0.1"
    config.server.port = 4999
    config.server.log_level = "WARNING"
    config.server.dashboard_enabled = True
    config.upstream.base_url = "http://upstream.test/v1"
    config.upstream.api_key = "sk-test-secret-key-value"
    config.cache.backend = "sqlite"
    config.cache.sqlite_path = str(tmp_path / "cache.db")
    config.cache.default_ttl = 60
    config.pricing.models["test-model"] = ModelPricing(
        input_per_1m=1.0, output_per_1m=2.0
    )
    config.pricing.currency = "USD"
    for dotted, value in overrides.items():
        section, _, attr = dotted.partition(".")
        target = getattr(config, section)
        setattr(target, attr, value)
    return config


@pytest.fixture
def config(tmp_path) -> Config:
    return make_config(tmp_path)


def build_state(config: Config, upstream: FakeUpstream) -> AppState:
    state = AppState(config)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=upstream.build_app()),
        base_url="http://upstream.test",
        timeout=30.0,
    )
    state._provider = OpenAICompatibleProvider(config.upstream, client=client)
    return state


@pytest.fixture
async def state(config: Config, upstream: FakeUpstream) -> AsyncIterator[AppState]:
    app_state = build_state(config, upstream)
    yield app_state
    await app_state.shutdown()


@pytest.fixture
async def client(state: AppState) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(state=state)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://proxy.test", timeout=30.0
    ) as http_client:
        # Trigger lifespan startup/shutdown manually so background tasks exist.
        async with app.router.lifespan_context(app):
            yield http_client


def chat_body(
    *,
    model: str = "test-model",
    system: str | None = "You are a helpful assistant.",
    user: str = "What is 2 + 2?",
    **extra: Any,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    body: dict[str, Any] = {"model": model, "messages": messages}
    body.update(extra)
    return body


def content_of(payload: dict[str, Any]) -> str:
    return payload["choices"][0]["message"]["content"]


def sse_content(text: str) -> str:
    """Reassemble assistant text from an SSE response body."""
    out: list[str] = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if isinstance(piece, str):
                out.append(piece)
    return "".join(out)
