"""Pluggable cache backends: in-memory LRU, SQLite, Redis.

All backends implement the small async :class:`CacheBackend` protocol so the
cache engine never cares which one is configured.  A backend failure raises
:class:`CacheBackendError`; the engine decides whether to fail open.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..db import Database
from ..logging_utils import get_logger

log = get_logger("cachellm.cache.backend")


class CacheBackendError(RuntimeError):
    """Raised when the cache backend cannot serve a request."""


@dataclass(slots=True)
class CachedEntry:
    """A stored upstream response plus the metadata needed for stats/TTL."""

    key: str
    namespace: str
    payload: dict[str, Any]
    created_at: float
    expires_at: float | None = None
    model: str | None = None
    endpoint: str | None = None
    category: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    hits: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= time.time()

    @property
    def age(self) -> float:
        return max(0.0, time.time() - self.created_at)

    def to_json(self) -> str:
        return json.dumps(
            {
                "key": self.key,
                "namespace": self.namespace,
                "payload": self.payload,
                "created_at": self.created_at,
                "expires_at": self.expires_at,
                "model": self.model,
                "endpoint": self.endpoint,
                "category": self.category,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "meta": self.meta,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, text: str) -> "CachedEntry":
        data = json.loads(text)
        return cls(
            key=data["key"],
            namespace=data.get("namespace", "default"),
            payload=data["payload"],
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at"),
            model=data.get("model"),
            endpoint=data.get("endpoint"),
            category=data.get("category"),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            meta=data.get("meta") or {},
        )


@runtime_checkable
class CacheBackend(Protocol):
    name: str

    async def get(self, key: str) -> CachedEntry | None: ...
    async def set(self, entry: CachedEntry, ttl: int | None) -> None: ...
    async def delete(self, key: str) -> int: ...
    async def delete_by_model(self, model: str) -> int: ...
    async def delete_by_namespace(self, namespace: str) -> int: ...
    async def clear(self) -> int: ...
    async def purge_expired(self) -> int: ...
    async def size(self) -> int: ...
    async def health(self) -> dict[str, Any]: ...
    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------


class MemoryBackend:
    """Bounded LRU cache.  Also used as the L1 in front of persistent backends."""

    name = "memory"

    def __init__(self, max_entries: int = 2048) -> None:
        self.max_entries = max(1, int(max_entries))
        self._data: OrderedDict[str, CachedEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self.evictions = 0

    async def get(self, key: str) -> CachedEntry | None:
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry.expired:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return entry

    async def set(self, entry: CachedEntry, ttl: int | None) -> None:
        async with self._lock:
            if ttl:
                entry.expires_at = time.time() + float(ttl)
            self._data[entry.key] = entry
            self._data.move_to_end(entry.key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
                self.evictions += 1

    async def delete(self, key: str) -> int:
        async with self._lock:
            return 1 if self._data.pop(key, None) is not None else 0

    async def delete_by_model(self, model: str) -> int:
        async with self._lock:
            keys = [k for k, v in self._data.items() if v.model == model]
            for key in keys:
                self._data.pop(key, None)
            return len(keys)

    async def delete_by_namespace(self, namespace: str) -> int:
        async with self._lock:
            keys = [k for k, v in self._data.items() if v.namespace == namespace]
            for key in keys:
                self._data.pop(key, None)
            return len(keys)

    async def clear(self) -> int:
        async with self._lock:
            count = len(self._data)
            self._data.clear()
            return count

    async def purge_expired(self) -> int:
        async with self._lock:
            keys = [k for k, v in self._data.items() if v.expired]
            for key in keys:
                self._data.pop(key, None)
            return len(keys)

    async def size(self) -> int:
        return len(self._data)

    async def health(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "ok": True,
            "entries": len(self._data),
            "max_entries": self.max_entries,
            "evictions": self.evictions,
        }

    async def close(self) -> None:
        await self.clear()


# ---------------------------------------------------------------------------
# sqlite (default)
# ---------------------------------------------------------------------------


class SQLiteBackend:
    """Durable local backend; the default because CacheLLM runs on your machine."""

    name = "sqlite"

    def __init__(self, db: Database) -> None:
        self.db = db

    async def get(self, key: str) -> CachedEntry | None:
        try:
            row = await self.db.run(self.db.get_cache_entry, key)
        except Exception as exc:  # sqlite3.Error and friends
            raise CacheBackendError(f"sqlite get failed: {exc}") from exc
        if row is None:
            return None
        if row.expired:
            await self.db.run(self.db.delete_cache_entry, key)
            return None
        return CachedEntry(
            key=row.key,
            namespace=row.namespace,
            payload=row.payload,
            created_at=row.created_at,
            expires_at=row.expires_at,
            model=row.model,
            endpoint=row.endpoint,
            category=row.category,
            prompt_tokens=row.prompt_tokens,
            completion_tokens=row.completion_tokens,
            hits=row.hits,
        )

    async def set(self, entry: CachedEntry, ttl: int | None) -> None:
        try:
            await self.db.run(
                lambda: self.db.put_cache_entry(
                    key=entry.key,
                    namespace=entry.namespace,
                    payload=entry.payload,
                    model=entry.model,
                    endpoint=entry.endpoint,
                    category=entry.category,
                    request_json=entry.meta.get("request_json"),
                    prompt_excerpt=entry.meta.get("prompt_excerpt"),
                    response_excerpt=entry.meta.get("response_excerpt"),
                    prompt_tokens=entry.prompt_tokens,
                    completion_tokens=entry.completion_tokens,
                    ttl=ttl,
                )
            )
        except Exception as exc:
            raise CacheBackendError(f"sqlite set failed: {exc}") from exc

    async def delete(self, key: str) -> int:
        return await self.db.run(self.db.delete_cache_entry, key)

    async def delete_by_model(self, model: str) -> int:
        return await self.db.run(self.db.delete_cache_by_model, model)

    async def delete_by_namespace(self, namespace: str) -> int:
        return await self.db.run(self.db.delete_cache_by_namespace, namespace)

    async def clear(self) -> int:
        return await self.db.run(self.db.clear_cache)

    async def purge_expired(self) -> int:
        return await self.db.run(self.db.purge_expired)

    async def size(self) -> int:
        return await self.db.run(self.db.count_cache_entries)

    async def health(self) -> dict[str, Any]:
        try:
            entries = await self.size()
            return {"backend": self.name, "ok": True, "path": self.db.path, "entries": entries}
        except Exception as exc:
            return {"backend": self.name, "ok": False, "error": str(exc)}

    async def close(self) -> None:
        await self.db.run(self.db.close)


# ---------------------------------------------------------------------------
# redis (optional)
# ---------------------------------------------------------------------------


class RedisBackend:
    """Optional shared backend.  Requires ``pip install cachellm[redis]``."""

    name = "redis"

    def __init__(self, url: str, *, prefix: str = "cachellm") -> None:
        try:
            import redis.asyncio as aioredis  # type: ignore import-not-found
        except ImportError as exc:  # pragma: no cover - optional dep
            raise CacheBackendError(
                "redis backend requested but the 'redis' package is not installed "
                "(pip install 'cachellm[redis]')"
            ) from exc
        self._client = aioredis.from_url(url, encoding="utf-8", decode_responses=True)
        self.prefix = prefix
        self.url = url

    def _k(self, key: str) -> str:
        return f"{self.prefix}:entry:{key}"

    async def get(self, key: str) -> CachedEntry | None:
        try:
            raw = await self._client.get(self._k(key))
        except Exception as exc:
            raise CacheBackendError(f"redis get failed: {exc}") from exc
        if not raw:
            return None
        try:
            entry = CachedEntry.from_json(raw)
        except (json.JSONDecodeError, KeyError):
            await self.delete(key)
            return None
        if entry.expired:
            await self.delete(key)
            return None
        return entry

    async def set(self, entry: CachedEntry, ttl: int | None) -> None:
        if ttl:
            entry.expires_at = time.time() + float(ttl)
        try:
            await self._client.set(self._k(entry.key), entry.to_json(), ex=ttl or None)
            if entry.model:
                await self._client.sadd(f"{self.prefix}:model:{entry.model}", entry.key)
            await self._client.sadd(f"{self.prefix}:ns:{entry.namespace}", entry.key)
        except Exception as exc:
            raise CacheBackendError(f"redis set failed: {exc}") from exc

    async def delete(self, key: str) -> int:
        try:
            return int(await self._client.delete(self._k(key)))
        except Exception as exc:
            raise CacheBackendError(f"redis delete failed: {exc}") from exc

    async def _delete_set(self, set_key: str) -> int:
        members = await self._client.smembers(set_key)
        count = 0
        for member in members:
            count += int(await self._client.delete(self._k(member)))
        await self._client.delete(set_key)
        return count

    async def delete_by_model(self, model: str) -> int:
        return await self._delete_set(f"{self.prefix}:model:{model}")

    async def delete_by_namespace(self, namespace: str) -> int:
        return await self._delete_set(f"{self.prefix}:ns:{namespace}")

    async def clear(self) -> int:
        count = 0
        async for key in self._client.scan_iter(match=f"{self.prefix}:*"):
            count += int(await self._client.delete(key))
        return count

    async def purge_expired(self) -> int:
        return 0  # Redis expires keys itself.

    async def size(self) -> int:
        count = 0
        async for _ in self._client.scan_iter(match=f"{self.prefix}:entry:*"):
            count += 1
        return count

    async def health(self) -> dict[str, Any]:
        try:
            await self._client.ping()
            return {"backend": self.name, "ok": True, "url": self.url}
        except Exception as exc:
            return {"backend": self.name, "ok": False, "error": str(exc)}

    async def close(self) -> None:
        try:
            await self._client.close()
        except Exception:  # pragma: no cover
            pass


def build_backend(cache_cfg: Any, db: Database | None) -> CacheBackend:
    """Instantiate the configured backend."""
    backend = (cache_cfg.backend or "sqlite").lower()
    if backend == "memory":
        return MemoryBackend(cache_cfg.memory_max_entries)
    if backend == "sqlite":
        if db is None:  # pragma: no cover - defensive
            raise CacheBackendError("sqlite backend requires a Database instance")
        return SQLiteBackend(db)
    if backend == "redis":
        return RedisBackend(cache_cfg.redis_url)
    raise CacheBackendError(f"unknown cache backend: {backend!r}")
