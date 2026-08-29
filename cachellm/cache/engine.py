"""Cache engine - orchestrates policy, exact cache, semantic cache and coalescing.

The engine is deliberately transport-agnostic: it takes a request payload and a
callable that performs the upstream work, and returns a :class:`CacheOutcome`.
The HTTP layer (``cachellm.server``) handles SSE and status codes; everything
about *what may be cached and for how long* lives here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

from ..config import Config
from ..db import Database
from ..extract import excerpt
from ..logging_utils import EVENT_CACHE_SKIP, EVENT_SEMANTIC_HIT, get_logger, log_event
from ..policy import CacheDecision, PolicyEngine
from ..singleflight import SingleFlight
from .backends import CacheBackend, CachedEntry, MemoryBackend, build_backend
from .exact import ExactCache
from .keys import KeyBuilder, KeyMaterial
from .semantic import SemanticCache
from .tools import ToolCache

log = get_logger("cachellm.engine")

SOURCE_EXACT = "exact"
SOURCE_SEMANTIC = "semantic"
SOURCE_UPSTREAM = "upstream"
SOURCE_COALESCED = "coalesced"
SOURCE_BYPASS = "bypass"


@dataclass
class CacheOutcome:
    """Result of an engine lookup/execution cycle."""

    source: str
    payload: dict[str, Any] | None = None
    material: KeyMaterial | None = None
    decision: CacheDecision | None = None
    similarity: float | None = None
    cache_latency_ms: float = 0.0
    upstream_latency_ms: float = 0.0
    waiters: int = 1
    cached_age: float | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_hit(self) -> bool:
        return self.source in {SOURCE_EXACT, SOURCE_SEMANTIC, SOURCE_COALESCED}


class CacheEngine:
    """Facade over every cache layer."""

    def __init__(
        self,
        config: Config,
        *,
        db: Database | None = None,
        backend: CacheBackend | None = None,
        key_builder: KeyBuilder | None = None,
    ) -> None:
        self.config = config
        self.db = db
        if backend is None:
            backend = build_backend(config.cache, db)
        self.backend = backend
        self.keys = key_builder or KeyBuilder()
        self.policy = PolicyEngine(config.policy, default_ttl=config.cache.default_ttl)
        self.exact = ExactCache(
            backend,
            l1=MemoryBackend(config.cache.memory_max_entries)
            if backend.name != "memory"
            else None,
            fail_open=config.cache.fail_open,
            max_entry_bytes=config.cache.max_entry_bytes,
        )
        self.semantic = SemanticCache(config.semantic, db)
        self.tools = ToolCache(
            db,
            self.policy,
            namespace=config.cache.namespace,
            fail_open=config.cache.fail_open,
            store_arguments=config.privacy.store_tool_arguments,
            store_results=config.privacy.store_tool_results,
        )
        self.flight: SingleFlight[dict[str, Any]] = SingleFlight(
            wait_timeout=config.concurrency.coalesce_wait,
            max_inflight=config.concurrency.max_inflight,
        )

    # -- key/decision helpers --------------------------------------------
    def prepare(
        self,
        payload: Mapping[str, Any],
        *,
        endpoint: str,
        namespace: str | None = None,
        resolved_model: str | None = None,
        no_store: bool = False,
        explicit_ttl: int | None = None,
    ) -> tuple[KeyMaterial, CacheDecision]:
        data = dict(payload)
        ns = namespace or data.get("cachellm_namespace") or self.config.cache.namespace
        material = self.keys.build(
            data, endpoint=endpoint, namespace=str(ns), resolved_model=resolved_model
        )
        ttl_override = explicit_ttl
        if ttl_override is None and data.get("cachellm_ttl") is not None:
            try:
                ttl_override = int(data["cachellm_ttl"])
            except (TypeError, ValueError):
                ttl_override = None
        decision = self.policy.decide_request(
            data,
            semantic_enabled=self.config.semantic.enabled,
            max_temperature=self.config.cache.max_temperature,
            zero_temperature_only=self.config.cache.cache_zero_temperature_only,
            explicit_ttl=ttl_override,
            no_store=no_store or not self.config.cache.enabled,
        )
        return material, decision

    # -- lookup -----------------------------------------------------------
    async def lookup(
        self,
        material: KeyMaterial,
        decision: CacheDecision,
        *,
        request_id: str | None = None,
    ) -> CacheOutcome:
        if not decision.cacheable:
            log_event(
                EVENT_CACHE_SKIP,
                rid=request_id,
                reason=decision.reason,
                category=decision.category,
                model=material.model,
            )
            return CacheOutcome(SOURCE_BYPASS, material=material, decision=decision)

        started = time.perf_counter()
        result = await self.exact.get(material.key, request_id=request_id)
        if result.hit and result.entry is not None:
            if self.db is not None:
                try:
                    await self.db.run(self.db.touch_cache_entry, material.key)
                except Exception:
                    pass
            return CacheOutcome(
                SOURCE_EXACT,
                payload=result.entry.payload,
                material=material,
                decision=decision,
                cache_latency_ms=result.latency_ms,
                cached_age=result.entry.age,
            )

        eligible, reason = self.semantic.eligible(
            material, semantic_allowed=decision.semantic_allowed
        )
        if eligible:
            match = await self.semantic.lookup(material, request_id=request_id)
            if match is not None:
                entry_result = await self.exact.get(
                    match.cache_key, request_id=request_id, quiet=True
                )
                if entry_result.hit and entry_result.entry is not None:
                    log_event(
                        EVENT_SEMANTIC_HIT,
                        rid=request_id,
                        key=match.cache_key[:16],
                        similarity=round(match.similarity, 4),
                        threshold=self.config.semantic.threshold,
                    )
                    return CacheOutcome(
                        SOURCE_SEMANTIC,
                        payload=entry_result.entry.payload,
                        material=material,
                        decision=decision,
                        similarity=match.similarity,
                        cache_latency_ms=(time.perf_counter() - started) * 1000.0,
                        cached_age=entry_result.entry.age,
                    )
                # Vector points at a vanished entry - clean up.
                await self.semantic.delete_by_cache_key(match.cache_key)

        return CacheOutcome(
            SOURCE_UPSTREAM,
            material=material,
            decision=decision,
            cache_latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    # -- store ------------------------------------------------------------
    async def store(
        self,
        material: KeyMaterial,
        decision: CacheDecision,
        response: Mapping[str, Any],
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        request_id: str | None = None,
    ) -> bool:
        if not decision.cacheable or not decision.ttl:
            return False
        privacy = self.config.privacy
        entry = CachedEntry(
            key=material.key,
            namespace=material.namespace,
            payload=dict(response),
            created_at=time.time(),
            model=material.model,
            endpoint=material.endpoint,
            category=decision.category,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            meta={
                "request_json": material.normalized if privacy.store_prompts else None,
                "prompt_excerpt": (
                    excerpt(material.query_text) if privacy.store_prompts else None
                ),
                "response_excerpt": (
                    excerpt(_response_text(response)) if privacy.store_responses else None
                ),
            },
        )
        ok = await self.exact.set(entry, decision.ttl, request_id=request_id)
        if ok:
            eligible, _ = self.semantic.eligible(
                material, semantic_allowed=decision.semantic_allowed
            )
            if eligible:
                await self.semantic.store(
                    material,
                    ttl=decision.ttl,
                    store_text=privacy.store_prompts,
                    request_id=request_id,
                )
        return ok

    # -- coalesced execution ---------------------------------------------
    async def execute(
        self,
        material: KeyMaterial,
        decision: CacheDecision,
        factory: Callable[[], Awaitable[dict[str, Any]]],
        *,
        request_id: str | None = None,
    ) -> CacheOutcome:
        """Run ``factory`` upstream, coalescing identical concurrent requests."""
        if not (self.config.concurrency.single_flight and decision.cacheable):
            started = time.perf_counter()
            payload = await factory()
            return CacheOutcome(
                SOURCE_UPSTREAM if decision.cacheable else SOURCE_BYPASS,
                payload=payload,
                material=material,
                decision=decision,
                upstream_latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        started = time.perf_counter()
        flight = await self.flight.run(material.key, factory, request_id=request_id)
        elapsed = (time.perf_counter() - started) * 1000.0
        return CacheOutcome(
            SOURCE_UPSTREAM if flight.leader else SOURCE_COALESCED,
            payload=flight.value,
            material=material,
            decision=decision,
            upstream_latency_ms=elapsed if flight.leader else 0.0,
            waiters=flight.waiters,
        )

    # -- invalidation -----------------------------------------------------
    async def invalidate(
        self,
        *,
        key: str | None = None,
        model: str | None = None,
        namespace: str | None = None,
        tool: str | None = None,
        all_entries: bool = False,
    ) -> dict[str, int]:
        removed = {"cache_entries": 0, "tool_entries": 0, "semantic_entries": 0}
        if all_entries:
            removed["cache_entries"] = await self.exact.clear()
            removed["tool_entries"] = await self.tools.clear()
            return removed
        if key:
            removed["semantic_entries"] += await self.semantic.delete_by_cache_key(key)
            removed["cache_entries"] += await self.exact.delete(key)
            removed["tool_entries"] += await self.tools.invalidate_key(key)
        if model:
            removed["cache_entries"] += await self.exact.delete_by_model(model)
        if namespace:
            removed["cache_entries"] += await self.exact.delete_by_namespace(namespace)
        if tool:
            removed["tool_entries"] += await self.tools.invalidate_tool(tool)
        return removed

    async def purge_expired(self) -> int:
        return await self.exact.purge_expired()

    # -- introspection ----------------------------------------------------
    async def health(self) -> dict[str, Any]:
        return {
            "enabled": self.config.cache.enabled,
            "namespace": self.config.cache.namespace,
            "default_ttl": self.config.cache.default_ttl,
            "exact": await self.exact.health(),
            "semantic": await self.semantic.health(),
            "tools": await self.tools.health(),
            "singleflight": self.flight.stats(),
        }

    async def close(self) -> None:
        await self.semantic.close()
        await self.exact.close()


def _response_text(response: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for choice in response.get("choices") or []:
        if not isinstance(choice, Mapping):
            continue
        message = choice.get("message") or {}
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            for tool_call in message.get("tool_calls") or []:
                if isinstance(tool_call, Mapping):
                    function = tool_call.get("function") or {}
                    parts.append(
                        f"{function.get('name', '')}({function.get('arguments', '')})"
                    )
    if isinstance(response.get("output_text"), str):
        parts.append(response["output_text"])
    for item in response.get("output") or []:
        if isinstance(item, Mapping):
            for content in item.get("content") or []:
                if isinstance(content, Mapping) and isinstance(content.get("text"), str):
                    parts.append(content["text"])
    return "\n".join(parts)
