"""TTL expiry and cache invalidation."""

from __future__ import annotations

import time

from tests.conftest import chat_body, make_config


async def test_ttl_expiration(tmp_path, upstream):
    """A 1-second TTL entry must be a miss after it expires."""
    import httpx

    from cachellm.server import create_app
    from tests.conftest import build_state

    config = make_config(tmp_path)
    config.cache.default_ttl = 1
    config.policy.category_ttl["general"] = 1
    state = build_state(config, upstream)
    app = create_app(state=state)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            body = chat_body(user="short lived")
            await client.post("/v1/chat/completions", json=body)
            hit = await client.post("/v1/chat/completions", json=body)
            assert hit.headers["x-cachellm-cache"] == "exact"

            time.sleep(1.2)
            expired = await client.post("/v1/chat/completions", json=body)
            assert expired.headers["x-cachellm-cache"] == "miss"
            assert upstream.calls == 2
    finally:
        await state.shutdown()


async def test_explicit_ttl_header(client, upstream):
    body = chat_body(user="ttl header")
    response = await client.post(
        "/v1/chat/completions", json=body, headers={"X-CacheLLM-TTL": "1"}
    )
    assert response.headers["x-cachellm-cache"] == "miss"
    time.sleep(1.2)
    after = await client.post("/v1/chat/completions", json=body, headers={"X-CacheLLM-TTL": "1"})
    assert after.headers["x-cachellm-cache"] == "miss"


async def test_zero_ttl_header_disables_caching(client, upstream):
    body = chat_body(user="no caching please")
    await client.post("/v1/chat/completions", json=body, headers={"X-CacheLLM-TTL": "0"})
    again = await client.post("/v1/chat/completions", json=body, headers={"X-CacheLLM-TTL": "0"})
    assert again.headers["x-cachellm-cache"] == "miss"
    assert upstream.calls == 2


async def test_invalidate_by_key(client, upstream):
    body = chat_body(user="invalidate by key")
    first = await client.post("/v1/chat/completions", json=body)
    key_prefix = first.headers["x-cachellm-key"]

    listing = await client.get("/api/cache")
    full_key = next(
        e["key"] for e in listing.json()["entries"] if e["key"].startswith(key_prefix)
    )

    removed = await client.request(
        "POST", "/api/cache/invalidate", json={"key": full_key}
    )
    assert removed.json()["removed"]["cache_entries"] == 1

    after = await client.post("/v1/chat/completions", json=body)
    assert after.headers["x-cachellm-cache"] == "miss"


async def test_invalidate_by_model(client, upstream):
    await client.post("/v1/chat/completions", json=chat_body(model="model-x", user="a"))
    await client.post("/v1/chat/completions", json=chat_body(model="model-y", user="a"))

    result = await client.post("/api/cache/invalidate", json={"model": "model-x"})
    assert result.json()["removed"]["cache_entries"] == 1

    x_again = await client.post("/v1/chat/completions", json=chat_body(model="model-x", user="a"))
    y_again = await client.post("/v1/chat/completions", json=chat_body(model="model-y", user="a"))
    assert x_again.headers["x-cachellm-cache"] == "miss"
    assert y_again.headers["x-cachellm-cache"] == "exact"


async def test_invalidate_by_namespace(client, upstream):
    body = chat_body(user="namespaced")
    await client.post("/v1/chat/completions", json=body, headers={"X-CacheLLM-Namespace": "ns1"})
    await client.post("/v1/chat/completions", json=body, headers={"X-CacheLLM-Namespace": "ns2"})

    result = await client.post("/api/cache/invalidate", json={"namespace": "ns1"})
    assert result.json()["removed"]["cache_entries"] == 1

    ns1 = await client.post(
        "/v1/chat/completions", json=body, headers={"X-CacheLLM-Namespace": "ns1"}
    )
    ns2 = await client.post(
        "/v1/chat/completions", json=body, headers={"X-CacheLLM-Namespace": "ns2"}
    )
    assert ns1.headers["x-cachellm-cache"] == "miss"
    assert ns2.headers["x-cachellm-cache"] == "exact"


async def test_clear_all(client, upstream):
    await client.post("/v1/chat/completions", json=chat_body(user="one"))
    await client.post("/v1/chat/completions", json=chat_body(user="two"))
    await client.post(
        "/v1/tools/cache/store",
        json={"tool": "list_repositories", "arguments": {}, "result": ["a"]},
    )

    result = await client.post("/api/cache/invalidate", json={"all": True})
    removed = result.json()["removed"]
    assert removed["cache_entries"] == 2
    assert removed["tool_entries"] == 1

    listing = await client.get("/api/cache")
    assert listing.json()["total"] == 0


async def test_delete_single_entry_via_rest(client, upstream):
    await client.post("/v1/chat/completions", json=chat_body(user="delete me"))
    listing = await client.get("/api/cache")
    key = listing.json()["entries"][0]["key"]

    response = await client.delete(f"/api/cache/{key}")
    assert response.json()["removed"]["cache_entries"] == 1
    assert (await client.get("/api/cache")).json()["total"] == 0


async def test_purge_expired_maintenance_endpoint(client, upstream):
    await client.post(
        "/v1/chat/completions", json=chat_body(user="expiring"), headers={"X-CacheLLM-TTL": "1"}
    )
    time.sleep(1.1)
    result = await client.post("/api/maintenance/purge")
    assert result.json()["purged"] >= 1
