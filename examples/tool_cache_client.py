"""Agent tool-result caching against CacheLLM's tool cache API.

    python examples/tool_cache_client.py

Pattern: before executing a tool, ask CacheLLM whether it has a fresh result.
Afterwards, offer the result back. CacheLLM's policy engine decides whether the
tool may be cached at all - mutations are refused automatically.

Naming matters: the policy engine classifies by verb, so ``get_weather`` and
``list_repositories`` are recognised as read-only, while a bare ``weather`` is
an unknown verb and refused unless you give it an explicit policy in
cachellm.config.json:

    "policy": { "tool_policies": { "weather": { "cacheable": true, "ttl": 300 } } }
"""

from __future__ import annotations

import time
from typing import Any

import httpx

BASE = "http://localhost:4000"
http = httpx.Client(base_url=BASE, timeout=30)

CALLS: dict[str, int] = {}


# --- the "real" tools ------------------------------------------------------


def get_weather(city: str) -> dict[str, Any]:
    CALLS["get_weather"] = CALLS.get("get_weather", 0) + 1
    time.sleep(0.3)  # pretend this is a slow HTTP call
    return {"city": city, "temp_c": 41, "fetched_call": CALLS["get_weather"]}


def list_repositories() -> list[str]:
    CALLS["list_repositories"] = CALLS.get("list_repositories", 0) + 1
    time.sleep(0.3)
    return ["repo-a", "repo-b"]


def delete_repository(name: str) -> dict[str, Any]:
    CALLS["delete_repository"] = CALLS.get("delete_repository", 0) + 1
    return {"deleted": name}


TOOLS = {
    "get_weather": lambda args: get_weather(**args),
    "list_repositories": lambda args: list_repositories(),
    "delete_repository": lambda args: delete_repository(**args),
}


# --- cache-aware invocation ----------------------------------------------


def call_tool(name: str, arguments: dict[str, Any], context: dict[str, Any] | None = None) -> Any:
    lookup = http.post(
        "/v1/tools/cache/lookup",
        json={"tool": name, "arguments": arguments, "context": context or {}},
    ).json()

    if lookup["hit"]:
        print(f"  TOOL CACHE HIT  {name} (age {lookup['age_seconds']:.1f}s)")
        return lookup["result"]

    if not lookup["cacheable"]:
        print(f"  not cacheable   {name} ({lookup['reason']}) - executing directly")
        return TOOLS[name](arguments)

    print(f"  TOOL CACHE MISS {name} - executing (ttl will be {lookup['ttl']}s)")
    result = TOOLS[name](arguments)
    http.post(
        "/v1/tools/cache/store",
        json={
            "tool": name,
            "arguments": arguments,
            "result": result,
            "context": context or {},
        },
    )
    return result


if __name__ == "__main__":
    print("get_weather('Delhi') twice:")
    print("   ", call_tool("get_weather", {"city": "Delhi"}))
    print("   ", call_tool("get_weather", {"city": "Delhi"}))

    print("\nget_weather('Mumbai') - different arguments, different entry:")
    print("   ", call_tool("get_weather", {"city": "Mumbai"}))

    print("\nlist_repositories() twice:")
    print("   ", call_tool("list_repositories", {}))
    print("   ", call_tool("list_repositories", {}))

    print("\ndelete_repository() - a mutation, never cached:")
    print("   ", call_tool("delete_repository", {"name": "repo-a"}))
    print("   ", call_tool("delete_repository", {"name": "repo-a"}))

    print(f"\nactual tool executions: {CALLS}")
    stats = http.get("/api/stats").json()
    print(f"tool cache hits recorded by CacheLLM: {stats['counters']['tool_cache_hits']}")
