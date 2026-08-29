"""Exact cache behaviour: hits, misses, and the correctness boundaries."""

from __future__ import annotations

import pytest

from tests.conftest import chat_body, content_of


async def test_exact_cache_miss_then_hit(client, upstream):
    body = chat_body()

    first = await client.post("/v1/chat/completions", json=body)
    assert first.status_code == 200
    assert first.headers["x-cachellm-cache"] == "miss"
    assert upstream.calls == 1

    second = await client.post("/v1/chat/completions", json=body)
    assert second.status_code == 200
    assert second.headers["x-cachellm-cache"] == "exact"
    assert upstream.calls == 1, "cache hit must not touch upstream"
    assert content_of(second.json()) == content_of(first.json())


async def test_key_order_and_whitespace_do_not_fragment_cache(client, upstream):
    await client.post("/v1/chat/completions", json=chat_body())
    assert upstream.calls == 1

    # Same request, keys in a different order and trailing whitespace in content.
    reordered = {
        "messages": [
            {"content": "You are a helpful assistant.", "role": "system"},
            {"content": "What is 2 + 2?   \n", "role": "user"},
        ],
        "model": "test-model",
    }
    response = await client.post("/v1/chat/completions", json=reordered)
    assert response.headers["x-cachellm-cache"] == "exact"
    assert upstream.calls == 1


async def test_different_system_prompt_never_shares_cache(client, upstream):
    """The headline correctness test from the spec."""
    question = "Summarise the architecture."
    a = await client.post(
        "/v1/chat/completions",
        json=chat_body(system="You are a terse assistant.", user=question),
    )
    b = await client.post(
        "/v1/chat/completions",
        json=chat_body(system="You are a verbose pirate.", user=question),
    )
    assert a.headers["x-cachellm-cache"] == "miss"
    assert b.headers["x-cachellm-cache"] == "miss"
    assert a.headers["x-cachellm-key"] != b.headers["x-cachellm-key"]
    assert upstream.calls == 2


async def test_developer_role_counts_as_context(client, upstream):
    base = chat_body(system=None, user="ping")
    with_developer = {
        "model": "test-model",
        "messages": [
            {"role": "developer", "content": "Always answer in JSON."},
            {"role": "user", "content": "ping"},
        ],
    }
    await client.post("/v1/chat/completions", json=base)
    response = await client.post("/v1/chat/completions", json=with_developer)
    assert response.headers["x-cachellm-cache"] == "miss"
    assert upstream.calls == 2


async def test_different_models_do_not_share_cache(client, upstream):
    await client.post("/v1/chat/completions", json=chat_body(model="model-a"))
    response = await client.post("/v1/chat/completions", json=chat_body(model="model-b"))
    assert response.headers["x-cachellm-cache"] == "miss"
    assert upstream.calls == 2


async def test_generation_parameters_change_the_key(client, upstream):
    await client.post("/v1/chat/completions", json=chat_body(temperature=0.2))
    hit = await client.post("/v1/chat/completions", json=chat_body(temperature=0.2))
    assert hit.headers["x-cachellm-cache"] == "exact"

    different = await client.post("/v1/chat/completions", json=chat_body(temperature=0.7))
    assert different.headers["x-cachellm-cache"] == "miss"

    top_p = await client.post("/v1/chat/completions", json=chat_body(temperature=0.2, top_p=0.5))
    assert top_p.headers["x-cachellm-cache"] == "miss"


async def test_response_format_changes_the_key(client, upstream):
    await client.post("/v1/chat/completions", json=chat_body())
    response = await client.post(
        "/v1/chat/completions", json=chat_body(response_format={"type": "json_object"})
    )
    assert response.headers["x-cachellm-cache"] == "miss"


async def test_conversation_history_is_part_of_the_key(client, upstream):
    first = chat_body(user="and then?")
    second = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "and then?"},
        ],
    }
    await client.post("/v1/chat/completions", json=first)
    response = await client.post("/v1/chat/completions", json=second)
    assert response.headers["x-cachellm-cache"] == "miss"
    assert upstream.calls == 2


async def test_namespace_isolation(client, upstream):
    body = chat_body()
    await client.post("/v1/chat/completions", json=body)
    other = await client.post(
        "/v1/chat/completions", json=body, headers={"X-CacheLLM-Namespace": "tenant-b"}
    )
    assert other.headers["x-cachellm-cache"] == "miss"
    again = await client.post(
        "/v1/chat/completions", json=body, headers={"X-CacheLLM-Namespace": "tenant-b"}
    )
    assert again.headers["x-cachellm-cache"] == "exact"
    assert upstream.calls == 2


async def test_no_store_header_bypasses_cache(client, upstream):
    body = chat_body()
    await client.post("/v1/chat/completions", json=body)
    bypass = await client.post(
        "/v1/chat/completions", json=body, headers={"Cache-Control": "no-store"}
    )
    assert bypass.headers["x-cachellm-cache"] == "miss"
    assert upstream.calls == 2


async def test_upstream_api_key_never_leaks_to_client(client, upstream):
    response = await client.post("/v1/chat/completions", json=chat_body())
    blob = response.text + str(dict(response.headers))
    assert "sk-test-secret-key-value" not in blob
    # The upstream did receive the configured key.
    assert upstream.seen_authorization == ["Bearer sk-test-secret-key-value"]


async def test_client_authorization_is_not_forwarded_by_default(client, upstream):
    await client.post(
        "/v1/chat/completions",
        json=chat_body(),
        headers={"Authorization": "Bearer client-key-should-be-dropped"},
    )
    assert upstream.seen_authorization == ["Bearer sk-test-secret-key-value"]


@pytest.mark.parametrize(
    "body,expected_fragment",
    [
        ({"messages": [{"role": "user", "content": "hi"}]}, "model"),
        ({"model": "test-model"}, "messages"),
        ({"model": "test-model", "messages": []}, "messages"),
    ],
)
async def test_malformed_requests_are_rejected(client, upstream, body, expected_fragment):
    response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 400
    assert expected_fragment in response.json()["error"]["message"]
    assert upstream.calls == 0


async def test_malformed_json_body(client, upstream):
    response = await client.post(
        "/v1/chat/completions",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert upstream.calls == 0


async def test_responses_endpoint_caches(client, upstream):
    body = {"model": "test-model", "input": "Explain caching in one line."}
    first = await client.post("/v1/responses", json=body)
    second = await client.post("/v1/responses", json=body)
    assert first.headers["x-cachellm-cache"] == "miss"
    assert second.headers["x-cachellm-cache"] == "exact"
    assert upstream.calls == 1


async def test_chat_and_responses_endpoints_have_separate_keys(client, upstream):
    await client.post("/v1/responses", json={"model": "test-model", "input": "hello"})
    response = await client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.headers["x-cachellm-cache"] == "miss"
    assert upstream.calls == 2
