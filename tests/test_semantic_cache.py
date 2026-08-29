"""Semantic cache: paraphrase matching plus its safety boundaries."""

from __future__ import annotations

import httpx
import pytest

from cachellm.cache.keys import KeyBuilder
from cachellm.cache.semantic import SemanticCache
from cachellm.config import SemanticConfig
from cachellm.db import Database
from cachellm.embeddings import HashingEmbedder, cosine_similarity
from cachellm.server import create_app
from tests.conftest import build_state, chat_body, content_of, make_config


@pytest.fixture
async def semantic_client(tmp_path, upstream):
    config = make_config(tmp_path)
    config.semantic.enabled = True
    config.semantic.backend = "hash"
    config.semantic.threshold = 0.80
    config.semantic.dimensions = 512
    state = build_state(config, upstream)
    app = create_app(state=state)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test", timeout=30.0
        ) as client:
            yield client, state
    finally:
        await state.shutdown()


async def test_semantic_hit_for_near_identical_question(semantic_client, upstream):
    client, _ = semantic_client
    upstream.reply_text = "Caching stores results to avoid repeat work."

    first = await client.post(
        "/v1/chat/completions",
        json=chat_body(user="Please explain what caching means in software systems."),
    )
    assert first.headers["x-cachellm-cache"] == "miss"

    second = await client.post(
        "/v1/chat/completions",
        json=chat_body(user="Please explain what caching means in software system."),
    )
    assert second.headers["x-cachellm-cache"] == "semantic"
    assert float(second.headers["x-cachellm-similarity"]) >= 0.80
    assert content_of(second.json()) == "Caching stores results to avoid repeat work."
    assert upstream.calls == 1


async def test_semantic_cache_never_crosses_system_prompts(semantic_client, upstream):
    """Same question, different system instruction -> must not share an answer."""
    client, _ = semantic_client
    question = "Please explain what caching means in software systems."

    await client.post(
        "/v1/chat/completions",
        json=chat_body(system="You are a database expert.", user=question),
    )
    other = await client.post(
        "/v1/chat/completions",
        json=chat_body(system="You are a poet who answers in verse.", user=question),
    )
    assert other.headers["x-cachellm-cache"] == "miss"
    assert upstream.calls == 2


async def test_semantic_cache_never_crosses_models(semantic_client, upstream):
    client, _ = semantic_client
    question = "Explain semantic caching in a sentence or two please."
    await client.post("/v1/chat/completions", json=chat_body(model="model-a", user=question))
    other = await client.post(
        "/v1/chat/completions", json=chat_body(model="model-b", user=question + " ")
    )
    assert other.headers["x-cachellm-cache"] == "miss"


async def test_semantic_cache_ignores_unrelated_questions(semantic_client, upstream):
    client, _ = semantic_client
    await client.post(
        "/v1/chat/completions", json=chat_body(user="Explain how database indexes work.")
    )
    other = await client.post(
        "/v1/chat/completions", json=chat_body(user="What is the capital city of France?")
    )
    assert other.headers["x-cachellm-cache"] == "miss"
    assert upstream.calls == 2


async def test_semantic_cache_skips_tool_requests(semantic_client, upstream):
    client, state = semantic_client
    payload = chat_body(
        user="List every repository that I own on the platform please.",
        tools=[{"type": "function", "function": {"name": "list_repositories"}}],
    )
    await client.post("/v1/chat/completions", json=payload)
    variant = dict(payload)
    variant["messages"] = [
        payload["messages"][0],
        {"role": "user", "content": "List every repository that I own on the platform pleas."},
    ]
    response = await client.post("/v1/chat/completions", json=variant)
    assert response.headers["x-cachellm-cache"] == "miss"


async def test_semantic_disabled_by_default(client, upstream):
    await client.post(
        "/v1/chat/completions", json=chat_body(user="Explain caching in software systems please.")
    )
    response = await client.post(
        "/v1/chat/completions", json=chat_body(user="Explain caching in software system please.")
    )
    assert response.headers["x-cachellm-cache"] == "miss", "semantic cache must be opt-in"


async def test_multi_turn_conversations_are_not_semantically_matched(semantic_client, upstream):
    client, _ = semantic_client
    convo = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Tell me about caching strategies in detail."},
            {"role": "assistant", "content": "Sure."},
            {"role": "user", "content": "Now explain semantic caching in more detail please."},
        ],
    }
    await client.post("/v1/chat/completions", json=convo)
    variant = {**convo, "messages": list(convo["messages"])}
    variant["messages"][-1] = {
        "role": "user",
        "content": "Now explain semantic caching in more detail pleas.",
    }
    response = await client.post("/v1/chat/completions", json=variant)
    assert response.headers["x-cachellm-cache"] == "miss"


# ---------------------------------------------------------------------------
# unit level
# ---------------------------------------------------------------------------


def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = HashingEmbedder(256)
    a = embedder.embed_sync("hello world")
    b = embedder.embed_sync("hello world")
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6


def test_hashing_embedder_similarity_ordering():
    embedder = HashingEmbedder(512)
    base = embedder.embed_sync("how do I configure the cache time to live")
    near = embedder.embed_sync("how do I configure the cache time to live?")
    far = embedder.embed_sync("what is the weather in Delhi tomorrow")
    assert cosine_similarity(base, near) > cosine_similarity(base, far)
    assert cosine_similarity(base, near) > 0.9


async def test_semantic_eligibility_rules(tmp_path):
    db = Database(str(tmp_path / "sem.db"))
    cfg = SemanticConfig(enabled=True, backend="hash", min_chars=12)
    cache = SemanticCache(cfg, db, embedder=HashingEmbedder(128))
    builder = KeyBuilder()

    short = builder.build(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        endpoint="/v1/chat/completions",
    )
    ok, reason = cache.eligible(short, semantic_allowed=True)
    assert (ok, reason) == (False, "query_too_short")

    long_enough = builder.build(
        {"model": "m", "messages": [{"role": "user", "content": "a reasonably long question"}]},
        endpoint="/v1/chat/completions",
    )
    assert cache.eligible(long_enough, semantic_allowed=True)[0] is True
    assert cache.eligible(long_enough, semantic_allowed=False) == (False, "policy")

    with_tools = builder.build(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "a reasonably long question"}],
            "tools": [{"type": "function", "function": {"name": "list_files"}}],
        },
        endpoint="/v1/chat/completions",
    )
    assert cache.eligible(with_tools, semantic_allowed=True) == (False, "tools_present")
    db.close()


async def test_semantic_store_and_lookup_roundtrip(tmp_path):
    db = Database(str(tmp_path / "sem2.db"))
    cfg = SemanticConfig(enabled=True, backend="hash", threshold=0.9)
    cache = SemanticCache(cfg, db, embedder=HashingEmbedder(512))
    builder = KeyBuilder()

    material = builder.build(
        {"model": "m", "messages": [{"role": "user", "content": "what is the cache ttl setting"}]},
        endpoint="/v1/chat/completions",
    )
    # The exact entry must exist for the join in semantic_candidates to match.
    db.put_cache_entry(
        key=material.key, namespace=material.namespace, payload={"ok": True}, model="m", ttl=600
    )
    assert await cache.store(material, ttl=600) is True

    similar = builder.build(
        {"model": "m", "messages": [{"role": "user", "content": "what is the cache ttl setting?"}]},
        endpoint="/v1/chat/completions",
    )
    match = await cache.lookup(similar)
    assert match is not None
    assert match.cache_key == material.key
    assert match.similarity >= 0.9

    unrelated = builder.build(
        {"model": "m", "messages": [{"role": "user", "content": "unrelated pizza recipe request"}]},
        endpoint="/v1/chat/completions",
    )
    assert await cache.lookup(unrelated) is None
    db.close()
