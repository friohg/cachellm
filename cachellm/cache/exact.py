"""Exact response cache.

Two-tier: an in-process LRU (L1) in front of the configured durable backend
(L2).  Lookups and writes are guarded so a backend failure degrades to a miss
(``fail_open``) instead of breaking the agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..logging_utils import (
    EVENT_CACHE_ERROR,
    EVENT_CACHE_HIT,
    EVENT_CACHE_MISS,
    EVENT_CACHE_WRITE,
    get_logger,
    log_event,
)
from .backends import CacheBackend, CacheBackendError, CachedEntry, MemoryBackend

log = get_logger("cachellm.cache.exact")


@dataclass(slots=True)
class LookupResult:
    entry: CachedEntry | None
    latency_ms: float
    error: str | None = None

    @property
    def hit(self) -> bool:
        return self.entry is not None


class ExactCache:
    """Deterministic key -> response store."""

    def __init__(
        self,
        backend: CacheBackend,
        *,
        l1: MemoryBackend | None = None,
        fail_open: bool = True,
        max_entry_bytes: int = 1_000_000,
    ) -> None:
        self.backend = backend
        self.l1 = l1 if l1 is not None or backend.name == "memory" else MemoryBackend(512)
        if backend.name == "memory":
            self.l1 = None  # backend already is memory
        self.fail_open = fail_open
        self.max_entry_bytes = max_entry_bytes
        self.errors = 0

    async def get(
        self, key: str, *, request_id: str | None = None, quiet: bool = False
    ) -> LookupResult:
        """Look up ``key``.  ``quiet`` suppresses hit/miss logging for internal
        re-checks (e.g. the single-flight double-check) so one client request
        produces exactly one CACHE HIT/MISS line."""
        started = time.perf_counter()
        try:
            if self.l1 is not None:
                entry = await self.l1.get(key)
                if entry is not None:
                    latency = (time.perf_counter() - started) * 1000.0
                    if not quiet:
                        log_event(
                            EVENT_CACHE_HIT,
                            rid=request_id,
                            key=key[:16],
                            tier="l1",
                            age_s=round(entry.age, 1),
                            lookup_ms=round(latency, 2),
                        )
                    return LookupResult(entry, latency)
            entry = await self.backend.get(key)
        except CacheBackendError as exc:
            self.errors += 1
            latency = (time.perf_counter() - started) * 1000.0
            log_event(EVENT_CACHE_ERROR, rid=request_id, op="get", error=str(exc))
            if not self.fail_open:
                raise
            return LookupResult(None, latency, error=str(exc))
        except Exception as exc:  # unexpected backend blow-up
            self.errors += 1
            latency = (time.perf_counter() - started) * 1000.0
            log_event(EVENT_CACHE_ERROR, rid=request_id, op="get", error=repr(exc))
            if not self.fail_open:
                raise CacheBackendError(str(exc)) from exc
            return LookupResult(None, latency, error=str(exc))

        latency = (time.perf_counter() - started) * 1000.0
        if entry is None:
            if not quiet:
                log_event(
                    EVENT_CACHE_MISS, rid=request_id, key=key[:16], lookup_ms=round(latency, 2)
                )
            return LookupResult(None, latency)

        if self.l1 is not None:
            remaining = None
            if entry.expires_at is not None:
                remaining = max(1, int(entry.expires_at - time.time()))
            await self.l1.set(entry, remaining)
        if not quiet:
            log_event(
                EVENT_CACHE_HIT,
                rid=request_id,
                key=key[:16],
                tier=self.backend.name,
                age_s=round(entry.age, 1),
                lookup_ms=round(latency, 2),
            )
        return LookupResult(entry, latency)

    async def set(
        self,
        entry: CachedEntry,
        ttl: int | None,
        *,
        request_id: str | None = None,
    ) -> bool:
        size = len(str(entry.payload))
        if self.max_entry_bytes and size > self.max_entry_bytes:
            log_event(
                "CACHE SKIP",
                rid=request_id,
                key=entry.key[:16],
                reason="entry_too_large",
                size=size,
            )
            return False
        try:
            await self.backend.set(entry, ttl)
            if self.l1 is not None:
                await self.l1.set(entry, ttl)
        except CacheBackendError as exc:
            self.errors += 1
            log_event(EVENT_CACHE_ERROR, rid=request_id, op="set", error=str(exc))
            if not self.fail_open:
                raise
            return False
        except Exception as exc:
            self.errors += 1
            log_event(EVENT_CACHE_ERROR, rid=request_id, op="set", error=repr(exc))
            if not self.fail_open:
                raise CacheBackendError(str(exc)) from exc
            return False
        log_event(
            EVENT_CACHE_WRITE,
            rid=request_id,
            key=entry.key[:16],
            model=entry.model,
            ttl=ttl,
            category=entry.category,
            bytes=size,
        )
        return True

    # -- invalidation -----------------------------------------------------
    async def delete(self, key: str) -> int:
        count = await self.backend.delete(key)
        if self.l1 is not None:
            count = max(count, await self.l1.delete(key))
        return count

    async def delete_by_model(self, model: str) -> int:
        count = await self.backend.delete_by_model(model)
        if self.l1 is not None:
            await self.l1.delete_by_model(model)
        return count

    async def delete_by_namespace(self, namespace: str) -> int:
        count = await self.backend.delete_by_namespace(namespace)
        if self.l1 is not None:
            await self.l1.delete_by_namespace(namespace)
        return count

    async def clear(self) -> int:
        count = await self.backend.clear()
        if self.l1 is not None:
            await self.l1.clear()
        return count

    async def purge_expired(self) -> int:
        count = await self.backend.purge_expired()
        if self.l1 is not None:
            count += await self.l1.purge_expired()
        return count

    async def health(self) -> dict[str, Any]:
        info = await self.backend.health()
        info["errors"] = self.errors
        if self.l1 is not None:
            info["l1"] = await self.l1.health()
        return info

    async def close(self) -> None:
        if self.l1 is not None:
            await self.l1.close()
        await self.backend.close()
