"""Unit tests for normalization, key building and the policy engine."""

from __future__ import annotations

import pytest

from cachellm.cache.keys import (
    KeyBuilder,
    extract_query_text,
    extract_system_text,
    extract_tool_names,
)
from cachellm.config import PolicyConfig, ToolPolicy
from cachellm.logging_utils import mask_secret, redact_headers, redact_text, redact_value
from cachellm.normalize import canonical_json, default_normalizer
from cachellm.policy import (
    CATEGORY_CURRENT_INFO,
    CATEGORY_MUTATION,
    CATEGORY_READ_ONLY_TOOL,
    CATEGORY_SEARCH,
    CATEGORY_STATIC,
    PolicyEngine,
)


# ---------------------------------------------------------------------------
# canonical JSON / normalization
# ---------------------------------------------------------------------------


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 1, "a": [3, {"z": 1, "y": 2}]}) == canonical_json(
        {"a": [3, {"y": 2, "z": 1}], "b": 1}
    )


def test_canonical_json_normalizes_integral_floats():
    assert canonical_json({"n": 1.0}) == canonical_json({"n": 1})


def test_normalizer_drops_volatile_fields_only():
    normalizer = default_normalizer()
    out = normalizer.normalize(
        {
            "model": "m",
            "messages": [{"role": "user", "content": " hi "}],
            "stream": True,
            "user": "user-42",
            "temperature": 0.3,
        }
    )
    assert "stream" not in out and "user" not in out
    assert out["temperature"] == 0.3
    assert out["messages"][0]["content"] == "hi"


def test_normalizer_canonicalizes_tool_call_arguments():
    normalizer = default_normalizer()
    a = normalizer.canonical(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"b":2,"a":1}'},
                        }
                    ],
                }
            ],
        }
    )
    b = normalizer.canonical(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"a": 1, "b": 2}'},
                        }
                    ],
                }
            ],
        }
    )
    assert a == b


def test_normalizer_preserves_meaningful_content():
    normalizer = default_normalizer()
    a = normalizer.canonical({"model": "m", "messages": [{"role": "user", "content": "Hi there"}]})
    b = normalizer.canonical({"model": "m", "messages": [{"role": "user", "content": "hi there"}]})
    c = normalizer.canonical({"model": "m", "messages": [{"role": "user", "content": "Hi  there"}]})
    assert a != b, "casing must not be normalized away"
    assert a != c, "interior whitespace must not be normalized away"


# ---------------------------------------------------------------------------
# key builder
# ---------------------------------------------------------------------------


def test_key_is_sha256_hex():
    material = KeyBuilder().build(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        endpoint="/v1/chat/completions",
    )
    assert len(material.key) == 64
    assert all(ch in "0123456789abcdef" for ch in material.key)


def test_tool_schema_change_changes_key():
    builder = KeyBuilder()
    schema_a = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    schema_b = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
            },
        }
    ]
    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    key_a = builder.build({**base, "tools": schema_a}, endpoint="/v1/chat/completions").key
    key_b = builder.build({**base, "tools": schema_b}, endpoint="/v1/chat/completions").key
    assert key_a != key_b


def test_context_hash_isolates_system_prompts():
    builder = KeyBuilder()
    question = {"role": "user", "content": "the same question"}
    a = builder.build(
        {"model": "m", "messages": [{"role": "system", "content": "A"}, question]},
        endpoint="/v1/chat/completions",
    )
    b = builder.build(
        {"model": "m", "messages": [{"role": "system", "content": "B"}, question]},
        endpoint="/v1/chat/completions",
    )
    assert a.context_hash != b.context_hash
    assert a.query_text == b.query_text == "the same question"


def test_extraction_helpers():
    payload = {
        "model": "m",
        "instructions": "top-level instruction",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": [{"type": "text", "text": "last"}]},
        ],
        "tools": [
            {"type": "function", "function": {"name": "b_tool"}},
            {"type": "function", "function": {"name": "a_tool"}},
        ],
    }
    assert "top-level instruction" in extract_system_text(payload)
    assert "sys" in extract_system_text(payload)
    assert extract_query_text(payload) == "last"
    assert extract_tool_names(payload) == ("a_tool", "b_tool")


# ---------------------------------------------------------------------------
# policy engine
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> PolicyEngine:
    return PolicyEngine(PolicyConfig(), default_ttl=3600)


@pytest.mark.parametrize(
    "tool",
    ["get_user", "list_repositories", "search_code", "read_file", "fetch_url", "describe_table"],
)
def test_read_only_tools_are_cacheable(engine, tool):
    decision = engine.decide_tool(tool)
    assert decision.cacheable is True
    assert decision.ttl > 0


@pytest.mark.parametrize(
    "tool",
    [
        "create_repository",
        "delete_repository",
        "update_issue",
        "send_email",
        "purchase_item",
        "execute_sql",
        "transfer_funds",
        "deploy_service",
    ],
)
def test_mutation_tools_are_never_cacheable(engine, tool):
    decision = engine.decide_tool(tool)
    assert decision.cacheable is False
    assert decision.ttl == 0
    assert decision.category == CATEGORY_MUTATION


def test_unknown_tools_are_treated_as_unsafe(engine):
    assert engine.decide_tool("frobnicate_widget").cacheable is False


def test_search_tools_get_short_ttl(engine):
    decision = engine.decide_tool("web_search")
    assert decision.cacheable is True
    assert decision.category == CATEGORY_SEARCH
    assert decision.ttl == 60


def test_custom_tool_policies_and_ttls():
    policy = PolicyConfig(
        tool_policies={
            "weather": ToolPolicy(cacheable=True, ttl=300),
            "list_repositories": ToolPolicy(cacheable=True, ttl=30),
            "delete_repository": ToolPolicy(cacheable=False, ttl=0),
            "internal_*": ToolPolicy(cacheable=True, ttl=15),
        }
    )
    engine = PolicyEngine(policy)
    assert engine.decide_tool("weather").ttl == 300
    assert engine.decide_tool("list_repositories").ttl == 30
    assert engine.decide_tool("delete_repository").cacheable is False
    assert engine.decide_tool("internal_metrics").ttl == 15


def test_deny_list_beats_everything():
    policy = PolicyConfig(
        tool_policies={"get_payment_status": ToolPolicy(cacheable=True, ttl=600)},
        unsafe_tool_deny_list=["*payment*"],
    )
    engine = PolicyEngine(policy)
    decision = engine.decide_tool("get_payment_status")
    assert decision.cacheable is False
    assert decision.reason == "deny_list"


def test_request_with_mutation_tool_is_not_cacheable(engine):
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "delete the repo"}],
        "tools": [
            {"type": "function", "function": {"name": "delete_repository"}},
            {"type": "function", "function": {"name": "list_repositories"}},
        ],
    }
    decision = engine.decide_request(payload)
    assert decision.cacheable is False
    assert decision.category == CATEGORY_MUTATION


def test_request_with_only_read_only_tools_is_cacheable(engine):
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "list my repos"}],
        "tools": [{"type": "function", "function": {"name": "list_repositories"}}],
    }
    decision = engine.decide_request(payload)
    assert decision.cacheable is True
    assert decision.category == CATEGORY_READ_ONLY_TOOL
    assert decision.semantic_allowed is False, "tool requests must not semantic-match"


def test_dynamic_information_requests_are_not_cached(engine):
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "what is the current price of gold right now"}],
    }
    decision = engine.decide_request(payload)
    assert decision.category == CATEGORY_CURRENT_INFO
    assert decision.cacheable is False


def test_zero_temperature_is_static_and_aggressively_cached(engine):
    decision = engine.decide_request(
        {"model": "m", "messages": [{"role": "user", "content": "2+2"}], "temperature": 0}
    )
    assert decision.category == CATEGORY_STATIC
    assert decision.ttl == 86400


def test_zero_temperature_only_mode(engine):
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.9}
    decision = engine.decide_request(payload, zero_temperature_only=True)
    assert decision.cacheable is False
    assert decision.reason == "non_deterministic_temperature"


def test_semantic_only_enabled_for_allowed_categories(engine):
    payload = {"model": "m", "messages": [{"role": "user", "content": "explain caching"}]}
    assert engine.decide_request(payload, semantic_enabled=True).semantic_allowed is True
    assert engine.decide_request(payload, semantic_enabled=False).semantic_allowed is False


def test_word_boundary_matching_avoids_false_mutations(engine):
    # 'get_updates' contains 'update' as a substring but is a read verb.
    assert engine.decide_tool("get_updates").cacheable is True
    assert engine.decide_tool("update_settings").cacheable is False


# ---------------------------------------------------------------------------
# redaction
# ---------------------------------------------------------------------------


def test_headers_are_redacted():
    safe = redact_headers({"Authorization": "Bearer sk-abcdef123456", "X-Trace": "ok"})
    assert safe["Authorization"] == "***REDACTED***"
    assert safe["X-Trace"] == "ok"


def test_inline_secrets_are_redacted():
    assert "sk-abcdef123456" not in redact_text("key is sk-abcdef123456 ok")
    assert "abcdef123456" not in redact_text("Bearer abcdef123456")


def test_nested_values_are_redacted():
    safe = redact_value({"upstream": {"api_key": "sk-xyz1234567", "base_url": "http://x"}})
    assert safe["upstream"]["api_key"] == "***REDACTED***"
    assert safe["upstream"]["base_url"] == "http://x"


def test_mask_secret():
    assert mask_secret("sk-1234567890abcdef") == "sk-1...cdef"
    assert mask_secret("short") == "***REDACTED***"
    assert mask_secret("") == ""
