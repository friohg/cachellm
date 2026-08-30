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


# ---------------------------------------------------------------------------
# The proxy is a cache, not a context manager. On a miss the provider must
# receive the request byte-for-byte; nothing is stripped, summarised or held
# back. These tests exist because "does caching lose my context?" is the first
# thing anyone sensibly worries about.
# ---------------------------------------------------------------------------


async def test_full_conversation_is_forwarded_on_a_miss(client, upstream):
    conversation = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are a careful assistant. Rule one: be brief."},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer"},
            {"role": "user", "content": "third question"},
        ],
        "temperature": 0.3,
        "max_tokens": 512,
    }
    await client.post("/v1/chat/completions", json=conversation)

    assert upstream.calls == 1
    sent = upstream.seen_payloads[0]
    # Every message, in order, unmodified - including the system prompt.
    assert sent["messages"] == conversation["messages"]
    assert sent["messages"][0]["role"] == "system"
    assert "Rule one: be brief." in sent["messages"][0]["content"]
    assert sent["temperature"] == 0.3
    assert sent["max_tokens"] == 512


async def test_second_turn_of_a_conversation_still_sends_everything(client, upstream):
    """A cached first turn must not stop the second turn carrying its history."""
    first = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "system rules here"},
            {"role": "user", "content": "turn one"},
        ],
    }
    await client.post("/v1/chat/completions", json=first)
    hit = await client.post("/v1/chat/completions", json=first)
    assert hit.headers["x-cachellm-cache"] == "exact"
    assert upstream.calls == 1

    second = {
        "model": "test-model",
        "messages": [
            *first["messages"],
            {"role": "assistant", "content": "reply to turn one"},
            {"role": "user", "content": "turn two"},
        ],
    }
    await client.post("/v1/chat/completions", json=second)

    assert upstream.calls == 2, "a new turn is a miss and must reach the provider"
    sent = upstream.seen_payloads[-1]
    assert len(sent["messages"]) == 4
    assert sent["messages"][0]["content"] == "system rules here"
    assert sent["messages"][-1]["content"] == "turn two"


async def test_tool_schemas_are_forwarded_intact(client, upstream):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_repositories",
                "description": "List the repos",
                "parameters": {
                    "type": "object",
                    "properties": {"page": {"type": "integer"}},
                    "required": ["page"],
                },
            },
        }
    ]
    await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "list them"}],
            "tools": tools,
            "tool_choice": "auto",
        },
    )
    sent = upstream.seen_payloads[0]
    assert sent["tools"] == tools, "tool schemas must arrive complete"
    assert sent["tool_choice"] == "auto"


async def test_cache_hit_returns_the_whole_response_not_a_fragment(client, upstream):
    """A hit replays the entire response object, so the client's state is intact."""
    body = chat_body(user="give me the whole thing")
    first = await client.post("/v1/chat/completions", json=body)
    second = await client.post("/v1/chat/completions", json=body)

    assert second.json() == first.json()
    payload = second.json()
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"]["prompt_tokens"] == 11
    assert payload["model"] == "test-model"


async def test_only_non_content_fields_are_omitted_from_the_key(client, upstream):
    """`stream` and `user` don't change the answer, so they don't split the cache -
    but they are still forwarded upstream when we do call it."""
    await client.post(
        "/v1/chat/completions",
        json=chat_body(user="who am i", stream=False, user_id_unused=None),
    )
    sent = upstream.seen_payloads[0]
    assert sent["messages"][-1]["content"] == "who am i"
    # The control fields we add for ourselves never leak upstream.
    assert not any(key.startswith("cachellm_") for key in sent)


async def test_cachellm_control_fields_are_stripped_before_forwarding(client, upstream):
    await client.post(
        "/v1/chat/completions",
        json={
            **chat_body(user="strip my knobs"),
            "cachellm_ttl": 60,
            "cachellm_namespace": "team-a",
        },
    )
    sent = upstream.seen_payloads[0]
    assert "cachellm_ttl" not in sent
    assert "cachellm_namespace" not in sent
