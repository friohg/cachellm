"""Statistics, pricing, dashboard API, privacy controls and configuration."""

from __future__ import annotations

import json

import httpx
import pytest

from cachellm.config import Config, ModelPricing, load_config, validate_config
from cachellm.pricing import CostCalculator, TokenCounter, TokenUsage
from cachellm.server import create_app
from tests.conftest import build_state, chat_body, make_config


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


async def test_stats_track_hits_misses_and_savings(client, upstream):
    body = chat_body(user="stats please")
    await client.post("/v1/chat/completions", json=body)
    await client.post("/v1/chat/completions", json=body)
    await client.post("/v1/chat/completions", json=body)

    stats = (await client.get("/api/stats")).json()
    counters = stats["counters"]
    assert counters["total_requests"] == 3
    assert counters["cache_misses"] == 1
    assert counters["exact_hits"] == 2
    assert counters["upstream_requests"] == 1
    assert stats["cache_hit_rate_pct"] == pytest.approx(66.67, abs=0.1)
    # 11 prompt + 7 completion tokens saved twice.
    assert stats["estimated_tokens_saved"] == 36
    assert stats["cost"]["savings"] > 0
    assert stats["cost"]["cost_without_cache"] > stats["cost"]["actual_cost"]


async def test_real_provider_usage_is_used_when_available(client, upstream):
    await client.post("/v1/chat/completions", json=chat_body(user="usage test"))
    requests = (await client.get("/api/requests")).json()["requests"]
    assert requests[0]["prompt_tokens"] == 11
    assert requests[0]["completion_tokens"] == 7
    assert requests[0]["tokens_estimated"] == 0


async def test_tokens_are_estimated_when_provider_omits_usage(client, upstream):
    upstream.include_usage = False
    await client.post("/v1/chat/completions", json=chat_body(user="no usage from provider"))
    requests = (await client.get("/api/requests")).json()["requests"]
    assert requests[0]["tokens_estimated"] == 1
    assert requests[0]["prompt_tokens"] > 0


async def test_latency_is_recorded(client, upstream):
    upstream.delay = 0.05
    await client.post("/v1/chat/completions", json=chat_body(user="latency"))
    stats = (await client.get("/api/stats")).json()
    assert stats["latency"]["avg_total_ms"] > 0
    assert stats["latency"]["avg_upstream_ms"] > 0


async def test_recent_requests_filterable(client, upstream):
    body = chat_body(user="filter me")
    await client.post("/v1/chat/completions", json=body)
    await client.post("/v1/chat/completions", json=body)
    hits = (await client.get("/api/requests?outcome=exact_hit")).json()["requests"]
    assert len(hits) == 1
    assert hits[0]["outcome"] == "exact_hit"


# ---------------------------------------------------------------------------
# pricing / cost calculator
# ---------------------------------------------------------------------------


def test_cost_calculation():
    config = Config()
    config.pricing.models["m"] = ModelPricing(input_per_1m=1.0, output_per_1m=3.0)
    calc = CostCalculator(config.pricing)
    usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert calc.cost("m", usage) == pytest.approx(4.0)
    breakdown = calc.breakdown("m", usage, cached=True)
    assert breakdown.actual_cost == 0.0
    assert breakdown.cost_without_cache == pytest.approx(4.0)
    assert breakdown.savings == pytest.approx(4.0)
    assert breakdown.savings_pct == pytest.approx(100.0)


def test_pricing_prefix_match():
    config = Config()
    config.pricing.models["gpt-4o-mini"] = ModelPricing(input_per_1m=0.15, output_per_1m=0.6)
    calc = CostCalculator(config.pricing)
    assert calc.pricing_for("gpt-4o-mini-2024-07-18").input_per_1m == 0.15


def test_no_hardcoded_prices():
    """A fresh install must not assume any provider's prices."""
    config = Config()
    assert config.pricing.models == {}
    assert config.pricing.default.input_per_1m == 0.0
    assert CostCalculator(config.pricing).cost("anything", TokenUsage(1000, 1000)) == 0.0


def test_token_estimator_heuristic():
    config = Config()
    counter = TokenCounter(config.pricing)
    assert counter.count_text("a" * 400) == 100
    usage = counter.usage_for(
        request={"model": "m", "messages": [{"role": "user", "content": "hello world"}]},
        response=None,
        model="m",
    )
    assert usage.estimated is True
    assert usage.prompt_tokens > 0


def test_usage_from_responses_api_shape():
    usage = TokenCounter(Config().pricing).usage_from_response(
        {"usage": {"input_tokens": 5, "output_tokens": 9}}
    )
    assert usage is not None
    assert (usage.prompt_tokens, usage.completion_tokens) == (5, 9)


async def test_pricing_api_roundtrip(client):
    response = await client.post(
        "/api/pricing", json={"model": "priced-model", "input_per_1m": 2.5, "output_per_1m": 10.0}
    )
    assert response.status_code == 200
    table = (await client.get("/api/pricing")).json()["models"]
    assert any(m["model"] == "priced-model" and m["input_per_1m"] == 2.5 for m in table)

    await client.delete("/api/pricing/priced-model")
    table = (await client.get("/api/pricing")).json()["models"]
    assert not any(m["model"] == "priced-model" for m in table)


async def test_pricing_persists_to_database(tmp_path, upstream):
    config = make_config(tmp_path)
    state = build_state(config, upstream)
    app = create_app(state=state)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            await client.post(
                "/api/pricing",
                json={"model": "persisted", "input_per_1m": 1.0, "output_per_1m": 2.0},
            )
    finally:
        await state.shutdown()

    reopened = build_state(make_config(tmp_path), upstream)
    try:
        assert reopened.costs.pricing_for("persisted").input_per_1m == 1.0
    finally:
        await reopened.shutdown()


# ---------------------------------------------------------------------------
# dashboard / admin API
# ---------------------------------------------------------------------------


async def test_health_endpoint(client):
    response = await client.get("/health")
    body = response.json()
    assert body["status"] == "ok"
    assert body["cache"]["exact"]["backend"] == "sqlite"
    assert body["upstream"]["ok"] is True


async def test_dashboard_html_served(client):
    response = await client.get("/dashboard")
    assert response.status_code == 200
    assert "CacheLLM" in response.text
    # Behaviour lives in a separate script now, so the page links to it.
    assert "/static/dashboard.js" in response.text


async def test_root_redirects_to_dashboard(client):
    response = await client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/dashboard"


async def test_cache_browser_search_and_inspect(client, upstream):
    await client.post("/v1/chat/completions", json=chat_body(user="findable question"))
    listing = await client.get("/api/cache?search=findable")
    entries = listing.json()["entries"]
    assert len(entries) == 1

    detail = await client.get(f"/api/cache/{entries[0]['key']}")
    body = detail.json()
    assert body["payload"]["choices"][0]["message"]["content"]
    assert body["request_json"]["messages"][-1]["content"] == "findable question"


async def test_inspect_missing_entry_returns_404(client):
    response = await client.get("/api/cache/" + "0" * 64)
    assert response.status_code == 404


async def test_config_api_redacts_secrets(client):
    body = (await client.get("/api/config")).json()
    assert "sk-test-secret-key-value" not in json.dumps(body)
    assert body["config"]["upstream"]["api_key"].startswith("sk-t")
    assert "..." in body["config"]["upstream"]["api_key"]
    assert body["providers"]
    assert body["env_docs"]


async def test_runtime_config_updates(client, upstream):
    response = await client.post(
        "/api/config", json={"cache.default_ttl": 120, "semantic.threshold": 0.85}
    )
    body = response.json()
    assert body["applied"]["cache.default_ttl"] == 120
    assert body["applied"]["semantic.threshold"] == 0.85


async def test_runtime_config_rejects_unknown_and_secret_paths(client):
    response = await client.post(
        "/api/config", json={"upstream.api_key": "leaked", "nonsense": 1}
    )
    rejected = response.json()["rejected"]
    assert "upstream.api_key" in rejected
    assert "nonsense" in rejected


async def test_policy_endpoint(client):
    body = (await client.get("/api/policy")).json()
    assert body["category_ttl"]["mutation"] == 0
    assert "delete" in body["mutation_tool_prefixes"]


async def test_models_passthrough(client, upstream):
    response = await client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "test-model"


async def test_unknown_route_returns_json_error(client):
    response = await client.get("/nope")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found"


# ---------------------------------------------------------------------------
# privacy
# ---------------------------------------------------------------------------


async def test_privacy_disables_prompt_storage(tmp_path, upstream):
    config = make_config(tmp_path)
    config.privacy.store_prompts = False
    config.privacy.store_responses = False
    state = build_state(config, upstream)
    app = create_app(state=state)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            await client.post("/v1/chat/completions", json=chat_body(user="secret prompt text"))
            listing = await client.get("/api/cache")
            entry = listing.json()["entries"][0]
            assert entry["prompt_excerpt"] is None

            detail = await client.get(f"/api/cache/{entry['key']}")
            assert detail.json()["request_json"] is None

            requests = (await client.get("/api/requests")).json()["requests"]
            assert requests[0]["prompt_excerpt"] is None
            # Aggregate stats still work.
            stats = (await client.get("/api/stats")).json()
            assert stats["counters"]["total_requests"] == 1
    finally:
        await state.shutdown()


async def test_privacy_disables_tool_argument_storage(tmp_path, upstream):
    config = make_config(tmp_path)
    config.privacy.store_tool_arguments = False
    state = build_state(config, upstream)
    app = create_app(state=state)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            await client.post(
                "/v1/tools/cache/store",
                json={
                    "tool": "list_repositories",
                    "arguments": {"secret_token": "abc"},
                    "result": [],
                },
            )
            entries = (await client.get("/api/tools/cache")).json()["entries"]
            assert entries[0]["arguments"] is None
    finally:
        await state.shutdown()


def test_database_refuses_secret_config_keys(tmp_path):
    from cachellm.db import Database

    db = Database(str(tmp_path / "secrets.db"))
    with pytest.raises(ValueError):
        db.set_config_value("upstream.api_key", "sk-should-not-persist")
    db.set_config_value("cache.default_ttl", 60)
    assert db.get_config_value("cache.default_ttl") == 60
    db.close()


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("PORT", "4321")
    monkeypatch.setenv("CACHE_BACKEND", "memory")
    monkeypatch.setenv("DEFAULT_TTL", "77")
    monkeypatch.setenv("SEMANTIC_CACHE", "true")
    monkeypatch.setenv("SEMANTIC_THRESHOLD", "0.77")
    monkeypatch.setenv("UPSTREAM_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("UPSTREAM_API_KEY", "sk-from-env")
    monkeypatch.setenv("STORE_PROMPTS", "false")

    config = load_config(env=dict(__import__("os").environ))
    assert config.server.port == 4321
    assert config.cache.backend == "memory"
    assert config.cache.default_ttl == 77
    assert config.semantic.enabled is True
    assert config.semantic.threshold == 0.77
    assert config.upstream.base_url == "https://example.test/v1"
    assert config.upstream.api_key == "sk-from-env"
    assert config.privacy.store_prompts is False


def test_config_file_is_merged(tmp_path):
    path = tmp_path / "cachellm.json"
    path.write_text(
        json.dumps(
            {
                "server": {"port": 5005},
                "cache": {"default_ttl": 42, "namespace": "team-a"},
                "pricing": {"models": {"m": {"input_per_1m": 1.5, "output_per_1m": 4.5}}},
                "policy": {"tool_policies": {"weather": {"cacheable": True, "ttl": 300}}},
            }
        ),
        encoding="utf-8",
    )
    config = load_config(path, env={})
    assert config.server.port == 5005
    assert config.cache.default_ttl == 42
    assert config.cache.namespace == "team-a"
    assert config.pricing.models["m"].output_per_1m == 4.5
    assert config.policy.tool_policies["weather"].ttl == 300
    assert config.source_path == str(path)


def test_env_beats_config_file(tmp_path):
    path = tmp_path / "cachellm.json"
    path.write_text(json.dumps({"server": {"port": 5005}}), encoding="utf-8")
    config = load_config(path, env={"PORT": "6006"})
    assert config.server.port == 6006


def test_invalid_config_is_rejected():
    config = Config()
    config.cache.backend = "cassandra"
    with pytest.raises(ValueError):
        validate_config(config)
    config = Config()
    config.semantic.threshold = 1.5
    with pytest.raises(ValueError):
        validate_config(config)


def test_redacted_config_hides_keys():
    config = Config()
    config.upstream.api_key = "sk-abcdef1234567890"
    redacted = config.redacted()
    assert "abcdef1234567890" not in json.dumps(redacted)


def test_model_routing():
    config = Config()
    config.routing.model_map = {"fast": "upstream-small", "smart": "upstream-large"}
    assert config.resolve_model("fast") == "upstream-small"
    assert config.resolve_model("unlisted") == "unlisted"


async def test_model_routing_is_applied(client, upstream, state):
    state.config.routing.model_map = {"alias": "real-upstream-model"}
    await client.post("/v1/chat/completions", json=chat_body(model="alias", user="routed"))
    assert upstream.seen_payloads[0]["model"] == "real-upstream-model"


async def test_cache_can_be_disabled_entirely(tmp_path, upstream):
    config = make_config(tmp_path)
    config.cache.enabled = False
    state = build_state(config, upstream)
    app = create_app(state=state)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            body = chat_body(user="passthrough")
            await client.post("/v1/chat/completions", json=body)
            second = await client.post("/v1/chat/completions", json=body)
            assert second.headers["x-cachellm-cache"] == "miss"
            assert upstream.calls == 2
    finally:
        await state.shutdown()


async def test_memory_backend_works(tmp_path, upstream):
    config = make_config(tmp_path)
    config.cache.backend = "memory"
    state = build_state(config, upstream)
    app = create_app(state=state)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            body = chat_body(user="memory backend")
            await client.post("/v1/chat/completions", json=body)
            hit = await client.post("/v1/chat/completions", json=body)
            assert hit.headers["x-cachellm-cache"] == "exact"
            assert upstream.calls == 1
    finally:
        await state.shutdown()
