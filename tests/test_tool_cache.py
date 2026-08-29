"""Tool-result cache: keys, per-tool TTLs and unsafe-tool refusal."""

from __future__ import annotations

import time

import pytest

from cachellm.cache.tools import ToolCache, tool_cache_key
from cachellm.config import PolicyConfig, ToolPolicy
from cachellm.policy import PolicyEngine


@pytest.fixture
def tool_cache() -> ToolCache:
    policy = PolicyConfig(
        tool_policies={
            "weather": ToolPolicy(cacheable=True, ttl=300),
            "list_repositories": ToolPolicy(cacheable=True, ttl=30),
            "delete_repository": ToolPolicy(cacheable=False, ttl=0),
            "get_file": ToolPolicy(cacheable=True, ttl=60, include_context_keys=["repo"]),
        }
    )
    return ToolCache(None, PolicyEngine(policy), namespace="test")


async def test_store_and_lookup(tool_cache):
    await tool_cache.store("weather", {"city": "Delhi"}, {"temp_c": 41})
    lookup = await tool_cache.lookup("weather", {"city": "Delhi"})
    assert lookup.hit is True
    assert lookup.result == {"temp_c": 41}
    assert lookup.decision.ttl == 300


async def test_different_arguments_are_different_entries(tool_cache):
    await tool_cache.store("weather", {"city": "Delhi"}, {"temp_c": 41})
    other = await tool_cache.lookup("weather", {"city": "Mumbai"})
    assert other.hit is False


async def test_argument_key_order_does_not_matter(tool_cache):
    await tool_cache.store("weather", {"city": "Delhi", "units": "c"}, {"temp_c": 41})
    lookup = await tool_cache.lookup("weather", {"units": "c", "city": "Delhi"})
    assert lookup.hit is True


async def test_mutation_tools_are_refused(tool_cache):
    stored = await tool_cache.store("delete_repository", {"name": "x"}, {"deleted": True})
    assert stored.hit is False
    assert "not_cacheable" in (stored.error or "")
    lookup = await tool_cache.lookup("delete_repository", {"name": "x"})
    assert lookup.hit is False
    assert lookup.decision.cacheable is False


async def test_per_tool_ttl_expiry(tool_cache):
    tool_cache.policy.policy.tool_policies["quick"] = ToolPolicy(cacheable=True, ttl=1)
    await tool_cache.store("quick", {}, "value")
    assert (await tool_cache.lookup("quick", {})).hit is True
    time.sleep(1.1)
    assert (await tool_cache.lookup("quick", {})).hit is False


async def test_context_keys_participate_in_the_key(tool_cache):
    await tool_cache.store(
        "get_file", {"path": "README.md"}, "contents-a", context={"repo": "repo-a"}
    )
    same = await tool_cache.lookup(
        "get_file", {"path": "README.md"}, context={"repo": "repo-a"}
    )
    other = await tool_cache.lookup(
        "get_file", {"path": "README.md"}, context={"repo": "repo-b"}
    )
    assert same.hit is True
    assert other.hit is False, "different environment/context must not share a tool result"


async def test_unlisted_context_is_ignored(tool_cache):
    """Only declared context keys matter, so noise cannot fragment the cache."""
    await tool_cache.store(
        "get_file", {"path": "a"}, "c", context={"repo": "r", "trace_id": "1"}
    )
    lookup = await tool_cache.lookup(
        "get_file", {"path": "a"}, context={"repo": "r", "trace_id": "2"}
    )
    assert lookup.hit is True


async def test_invalidate_by_tool(tool_cache):
    await tool_cache.store("weather", {"city": "A"}, 1)
    await tool_cache.store("weather", {"city": "B"}, 2)
    await tool_cache.store("list_repositories", {}, [])
    assert await tool_cache.invalidate_tool("weather") == 2
    assert (await tool_cache.lookup("weather", {"city": "A"})).hit is False
    assert (await tool_cache.lookup("list_repositories", {})).hit is True


async def test_clear(tool_cache):
    await tool_cache.store("weather", {"city": "A"}, 1)
    assert await tool_cache.clear() >= 1
    assert await tool_cache.size() == 0


def test_key_is_stable_and_namespaced():
    a = tool_cache_key(tool_name="t", arguments={"x": 1}, namespace="ns")
    b = tool_cache_key(tool_name="t", arguments={"x": 1}, namespace="ns")
    c = tool_cache_key(tool_name="t", arguments={"x": 1}, namespace="other")
    assert a == b != c
    assert len(a) == 64


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


async def test_tool_cache_http_roundtrip(client):
    lookup = await client.post(
        "/v1/tools/cache/lookup", json={"tool": "list_repositories", "arguments": {}}
    )
    assert lookup.json()["hit"] is False
    assert lookup.json()["cacheable"] is True

    stored = await client.post(
        "/v1/tools/cache/store",
        json={"tool": "list_repositories", "arguments": {}, "result": ["repo-1", "repo-2"]},
    )
    assert stored.json()["stored"] is True

    hit = await client.post(
        "/v1/tools/cache/lookup", json={"tool": "list_repositories", "arguments": {}}
    )
    body = hit.json()
    assert body["hit"] is True
    assert body["result"] == ["repo-1", "repo-2"]
    assert body["age_seconds"] is not None


async def test_tool_cache_http_refuses_mutation(client):
    stored = await client.post(
        "/v1/tools/cache/store",
        json={"tool": "delete_repository", "arguments": {"n": 1}, "result": "gone"},
    )
    assert stored.json()["stored"] is False
    assert stored.json()["cacheable"] is False


async def test_tool_policy_endpoint(client):
    response = await client.post("/v1/tools/cache/policy", json={"tool": "search_issues"})
    body = response.json()
    assert body["cacheable"] is True
    assert body["category"] == "search"


async def test_tool_cache_requires_tool_name(client):
    response = await client.post("/v1/tools/cache/lookup", json={"arguments": {}})
    assert response.status_code == 400


async def test_tool_cache_ttl_override(client):
    stored = await client.post(
        "/v1/tools/cache/store",
        json={"tool": "get_config", "arguments": {}, "result": {"a": 1}, "ttl": 900},
    )
    assert stored.json()["ttl"] == 900


async def test_tool_hits_appear_in_stats(client):
    await client.post(
        "/v1/tools/cache/store",
        json={"tool": "list_repositories", "arguments": {}, "result": []},
    )
    await client.post(
        "/v1/tools/cache/lookup", json={"tool": "list_repositories", "arguments": {}}
    )
    stats = (await client.get("/api/stats")).json()
    assert stats["counters"]["tool_cache_hits"] == 1


async def test_tool_cache_browser_lists_entries(client):
    await client.post(
        "/v1/tools/cache/store",
        json={"tool": "list_repositories", "arguments": {"page": 1}, "result": []},
    )
    listing = await client.get("/api/tools/cache")
    body = listing.json()
    assert body["total"] == 1
    assert body["entries"][0]["tool_name"] == "list_repositories"
