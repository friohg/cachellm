"""Streaming, concurrency (single-flight) and failure handling."""

from __future__ import annotations

import asyncio
import json

import pytest

from cachellm.singleflight import SingleFlight
from tests.conftest import chat_body, content_of, sse_content


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------


async def test_streaming_miss_then_cached_replay(client, upstream):
    upstream.reply_text = "streamed answer in several words"
    body = chat_body(user="stream please", stream=True)

    first = await client.post("/v1/chat/completions", json=body)
    assert first.status_code == 200
    assert first.headers["content-type"].startswith("text/event-stream")
    assert sse_content(first.text) == "streamed answer in several words"
    assert first.text.rstrip().endswith("data: [DONE]")
    assert upstream.stream_calls == 1

    second = await client.post("/v1/chat/completions", json=body)
    assert second.headers["x-cachellm-cache"] == "exact"
    assert second.headers["content-type"].startswith("text/event-stream")
    assert sse_content(second.text) == "streamed answer in several words"
    assert second.text.rstrip().endswith("data: [DONE]")
    assert upstream.stream_calls == 1, "replay must not hit upstream"


async def test_streamed_response_is_reusable_by_non_streaming_request(client, upstream):
    """A stream is cached in normalized (non-streaming) form."""
    upstream.reply_text = "shared answer"
    await client.post("/v1/chat/completions", json=chat_body(user="q", stream=True))
    response = await client.post("/v1/chat/completions", json=chat_body(user="q"))
    assert response.headers["x-cachellm-cache"] == "exact"
    assert content_of(response.json()) == "shared answer"
    assert upstream.calls == 1


async def test_interrupted_stream_is_not_cached(client, upstream):
    upstream.truncate_stream = True
    body = chat_body(user="truncate me", stream=True)

    first = await client.post("/v1/chat/completions", json=body)
    assert first.status_code == 200  # client still receives what arrived

    upstream.truncate_stream = False
    second = await client.post("/v1/chat/completions", json=body)
    assert second.headers["x-cachellm-cache"] == "miss", "incomplete stream must not be cached"
    assert upstream.stream_calls == 2


async def test_stream_with_inline_error_is_not_cached(client, upstream):
    upstream.stream_error = True
    body = chat_body(user="error mid stream", stream=True)
    first = await client.post("/v1/chat/completions", json=body)
    assert "stream blew up" in first.text

    upstream.stream_error = False
    second = await client.post("/v1/chat/completions", json=body)
    assert second.headers["x-cachellm-cache"] == "miss"


async def test_stream_usage_is_replayed_when_requested(client, upstream):
    body = chat_body(user="usage please", stream=True, stream_options={"include_usage": True})
    await client.post("/v1/chat/completions", json=body)
    replay = await client.post("/v1/chat/completions", json=body)
    assert replay.headers["x-cachellm-cache"] == "exact"
    frames = [
        json.loads(line[5:].strip())
        for line in replay.text.splitlines()
        if line.startswith("data:") and line[5:].strip() != "[DONE]"
    ]
    assert any(frame.get("usage") for frame in frames)


async def test_stream_flag_does_not_fragment_the_key(client, upstream):
    """stream=true/false must map to the same cache key."""
    await client.post("/v1/chat/completions", json=chat_body(user="same", stream=False))
    streamed = await client.post("/v1/chat/completions", json=chat_body(user="same", stream=True))
    assert streamed.headers["x-cachellm-cache"] == "exact"
    assert upstream.calls == 1


async def test_upstream_error_during_stream_is_reported_not_cached(client, upstream):
    upstream.fail_status = 500
    body = chat_body(user="fail stream", stream=True)
    response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200  # SSE already started; error is in-band
    assert "upstream" in response.text.lower() or "error" in response.text.lower()

    upstream.fail_status = None
    retry = await client.post("/v1/chat/completions", json=body)
    assert retry.headers["x-cachellm-cache"] == "miss"


# ---------------------------------------------------------------------------
# single-flight / coalescing
# ---------------------------------------------------------------------------


async def test_concurrent_identical_requests_are_coalesced(client, upstream):
    upstream.delay = 0.25
    upstream.counter_suffix = True
    body = chat_body(user="expensive question")

    responses = await asyncio.gather(
        *[client.post("/v1/chat/completions", json=body) for _ in range(10)]
    )

    assert all(r.status_code == 200 for r in responses)
    assert upstream.calls == 1, f"expected 1 upstream call, got {upstream.calls}"
    texts = {content_of(r.json()) for r in responses}
    assert len(texts) == 1, "all waiters must receive the identical result"
    outcomes = [r.headers["x-cachellm-cache"] for r in responses]
    assert outcomes.count("miss") == 1
    assert outcomes.count("coalesced") == 9


async def test_concurrent_different_requests_are_not_coalesced(client, upstream):
    upstream.delay = 0.1
    bodies = [chat_body(user=f"question {i}") for i in range(5)]
    await asyncio.gather(*[client.post("/v1/chat/completions", json=b) for b in bodies])
    assert upstream.calls == 5


async def test_coalescing_propagates_failures_and_caches_nothing(client, upstream):
    upstream.delay = 0.15
    upstream.fail_status = 503
    body = chat_body(user="always failing")

    responses = await asyncio.gather(
        *[client.post("/v1/chat/completions", json=body) for _ in range(5)]
    )
    assert all(r.status_code == 503 for r in responses)
    assert upstream.calls == 1, "the failure was shared, not retried five times"

    upstream.fail_status = None
    retry = await client.post("/v1/chat/completions", json=body)
    assert retry.headers["x-cachellm-cache"] == "miss", "failures must not be cached"


async def test_singleflight_releases_slot_after_exception():
    flight: SingleFlight[int] = SingleFlight()

    async def boom() -> int:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await flight.run("k", boom)
    assert flight.inflight == 0, "no stale locks may remain"

    async def fine() -> int:
        return 7

    result = await flight.run("k", fine)
    assert result.value == 7
    assert flight.inflight == 0


async def test_singleflight_leader_and_waiters():
    flight: SingleFlight[int] = SingleFlight()
    started = asyncio.Event()
    calls = 0

    async def slow() -> int:
        nonlocal calls
        calls += 1
        started.set()
        await asyncio.sleep(0.1)
        return 42

    results = await asyncio.gather(*[flight.run("same", slow) for _ in range(4)])
    assert calls == 1
    assert [r.value for r in results] == [42] * 4
    assert sum(1 for r in results if r.leader) == 1


# ---------------------------------------------------------------------------
# failure handling
# ---------------------------------------------------------------------------


async def test_upstream_failure_is_not_cached_and_returns_useful_error(client, upstream):
    upstream.fail_status = 429
    upstream.fail_body = {"error": {"message": "rate limited", "type": "rate_limit_error"}}
    body = chat_body(user="rate limited question")

    failure = await client.post("/v1/chat/completions", json=body)
    assert failure.status_code == 429
    error = failure.json()["error"]
    assert error["message"] == "rate limited"
    assert error["source"] == "cachellm.upstream"
    assert "sk-test-secret-key-value" not in failure.text

    upstream.fail_status = None
    upstream.fail_body = None
    retry = await client.post("/v1/chat/completions", json=body)
    assert retry.headers["x-cachellm-cache"] == "miss"
    ok = await client.post("/v1/chat/completions", json=body)
    assert ok.headers["x-cachellm-cache"] == "exact"


async def test_missing_upstream_configuration_returns_500(tmp_path, upstream):
    from cachellm.app import AppState
    from cachellm.server import create_app
    from tests.conftest import make_config
    import httpx

    config = make_config(tmp_path)
    config.upstream.base_url = ""
    state = AppState(config)
    app = create_app(state=state)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            response = await client.post("/v1/chat/completions", json=chat_body())
            assert response.status_code == 500
            assert response.json()["error"]["type"] == "configuration_error"
    finally:
        await state.shutdown()


async def test_cache_backend_failure_falls_back_to_upstream(client, upstream, state):
    """fail_open=True: a broken cache must not break the agent."""
    from cachellm.cache.backends import CacheBackendError

    class BrokenBackend:
        name = "broken"

        async def get(self, key):
            raise CacheBackendError("disk on fire")

        async def set(self, entry, ttl):
            raise CacheBackendError("disk on fire")

        async def delete(self, key):
            return 0

        async def delete_by_model(self, model):
            return 0

        async def delete_by_namespace(self, namespace):
            return 0

        async def clear(self):
            return 0

        async def purge_expired(self):
            return 0

        async def size(self):
            return 0

        async def health(self):
            return {"backend": "broken", "ok": False}

        async def close(self):
            return None

    state.engine.exact.backend = BrokenBackend()
    state.engine.exact.l1 = None
    state.engine.exact.fail_open = True

    response = await client.post("/v1/chat/completions", json=chat_body(user="broken cache"))
    assert response.status_code == 200
    assert upstream.calls == 1
    assert state.engine.exact.errors >= 1


async def test_cache_backend_failure_with_fail_open_disabled_surfaces_error(client, state):
    from cachellm.cache.backends import CacheBackendError

    class BrokenBackend:
        name = "broken"

        async def get(self, key):
            raise CacheBackendError("nope")

        async def set(self, entry, ttl):
            raise CacheBackendError("nope")

        async def delete(self, key):
            return 0

        async def delete_by_model(self, model):
            return 0

        async def delete_by_namespace(self, namespace):
            return 0

        async def clear(self):
            return 0

        async def purge_expired(self):
            return 0

        async def size(self):
            return 0

        async def health(self):
            return {"ok": False}

        async def close(self):
            return None

    state.engine.exact.backend = BrokenBackend()
    state.engine.exact.l1 = None
    state.engine.exact.fail_open = False

    response = await client.post("/v1/chat/completions", json=chat_body(user="hard fail"))
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "cache_backend_error"
