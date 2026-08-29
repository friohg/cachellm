"""Tool-result cache.

Separate subsystem from the LLM response cache: agents call tools far more often
than they call the model, and each tool has its own freshness requirements.

Key = SHA-256(namespace | tool name | canonical arguments | selected context).
Only the context keys a tool's policy declares are folded in, so an unrelated
context field cannot fragment the cache.

Usage from an agent (HTTP):

    POST /v1/tools/cache/lookup   {"tool": "weather", "arguments": {...}}
    POST /v1/tools/cache/store    {"tool": "weather", "arguments": {...}, "result": ...}
    POST /v1/tools/cache/execute  {"tool": ..., "arguments": ..., "result": ...}
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Mapping

from ..db import Database
from ..extract import sha256_hex
from ..logging_utils import (
    EVENT_CACHE_ERROR,
    EVENT_TOOL_CACHE_HIT,
    EVENT_TOOL_CACHE_MISS,
    EVENT_TOOL_CACHE_WRITE,
    get_logger,
    log_event,
)
from ..normalize import canonical_json
from ..policy import PolicyEngine, ToolDecision

log = get_logger("cachellm.cache.tool")

TOOL_KEY_VERSION = "v1"


@dataclass(slots=True)
class ToolLookup:
    hit: bool
    key: str
    decision: ToolDecision
    result: Any = None
    created_at: float | None = None
    expires_at: float | None = None
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def age(self) -> float | None:
        return None if self.created_at is None else max(0.0, time.time() - self.created_at)


def tool_cache_key(
    *,
    tool_name: str,
    arguments: Any,
    namespace: str,
    context: Mapping[str, Any] | None = None,
) -> str:
    document = {
        "v": TOOL_KEY_VERSION,
        "namespace": namespace,
        "tool": tool_name,
        "arguments": arguments,
        "context": dict(context or {}),
    }
    return sha256_hex(canonical_json(document))


class ToolCache:
    """Policy-driven cache for tool invocations."""

    def __init__(
        self,
        db: Database | None,
        policy: PolicyEngine,
        *,
        namespace: str = "default",
        fail_open: bool = True,
        store_arguments: bool = True,
        store_results: bool = True,
    ) -> None:
        self.db = db
        self.policy = policy
        self.namespace = namespace
        self.fail_open = fail_open
        self.store_arguments = store_arguments
        self.store_results = store_results
        self.errors = 0
        self._memory: dict[str, dict[str, Any]] = {}
        """Fallback store used when no database is configured."""

    # -- helpers ----------------------------------------------------------
    def _select_context(
        self, decision: ToolDecision, context: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        if not context:
            return {}
        if not decision.include_context_keys:
            return {}
        return {
            key: context[key] for key in decision.include_context_keys if key in context
        }

    def key_for(
        self,
        tool_name: str,
        arguments: Any,
        *,
        namespace: str | None = None,
        context: Mapping[str, Any] | None = None,
        decision: ToolDecision | None = None,
    ) -> tuple[str, ToolDecision]:
        decision = decision or self.policy.decide_tool(tool_name)
        ns = namespace or decision.namespace or self.namespace
        key = tool_cache_key(
            tool_name=tool_name,
            arguments=arguments,
            namespace=ns,
            context=self._select_context(decision, context),
        )
        return key, decision

    # -- lookup / store ---------------------------------------------------
    async def lookup(
        self,
        tool_name: str,
        arguments: Any,
        *,
        namespace: str | None = None,
        context: Mapping[str, Any] | None = None,
        ttl: int | None = None,
        request_id: str | None = None,
    ) -> ToolLookup:
        decision = self.policy.decide_tool(tool_name, explicit_ttl=ttl)
        key, decision = self.key_for(
            tool_name, arguments, namespace=namespace, context=context, decision=decision
        )
        if not decision.cacheable:
            log_event(
                EVENT_TOOL_CACHE_MISS,
                rid=request_id,
                tool=tool_name,
                reason=decision.reason,
                cacheable=False,
            )
            return ToolLookup(False, key, decision)

        started = time.perf_counter()
        try:
            row = await self._get_row(key)
        except Exception as exc:
            self.errors += 1
            log_event(EVENT_CACHE_ERROR, rid=request_id, op="tool_lookup", error=str(exc))
            if not self.fail_open:
                raise
            return ToolLookup(False, key, decision, error=str(exc))

        latency = (time.perf_counter() - started) * 1000.0
        if row is None:
            log_event(
                EVENT_TOOL_CACHE_MISS,
                rid=request_id,
                tool=tool_name,
                key=key[:16],
                lookup_ms=round(latency, 2),
            )
            return ToolLookup(False, key, decision, latency_ms=latency)

        expires_at = row.get("expires_at")
        if expires_at is not None and float(expires_at) <= time.time():
            await self.invalidate_key(key)
            log_event(
                EVENT_TOOL_CACHE_MISS,
                rid=request_id,
                tool=tool_name,
                key=key[:16],
                reason="expired",
            )
            return ToolLookup(False, key, decision, latency_ms=latency)

        try:
            envelope = json.loads(row["result"]) if isinstance(row["result"], str) else row["result"]
            result = envelope.get("result") if isinstance(envelope, dict) else None
        except json.JSONDecodeError:
            await self.invalidate_key(key)
            return ToolLookup(False, key, decision, latency_ms=latency, error="corrupt_entry")

        await self._touch(key)
        log_event(
            EVENT_TOOL_CACHE_HIT,
            rid=request_id,
            tool=tool_name,
            key=key[:16],
            age_s=round(max(0.0, time.time() - float(row["created_at"])), 1),
            lookup_ms=round(latency, 2),
        )
        return ToolLookup(
            True,
            key,
            decision,
            result=result,
            created_at=float(row["created_at"]),
            expires_at=None if expires_at is None else float(expires_at),
            latency_ms=latency,
        )

    async def store(
        self,
        tool_name: str,
        arguments: Any,
        result: Any,
        *,
        namespace: str | None = None,
        context: Mapping[str, Any] | None = None,
        ttl: int | None = None,
        request_id: str | None = None,
    ) -> ToolLookup:
        decision = self.policy.decide_tool(tool_name, explicit_ttl=ttl)
        key, decision = self.key_for(
            tool_name, arguments, namespace=namespace, context=context, decision=decision
        )
        if not decision.cacheable:
            return ToolLookup(False, key, decision, error=f"not_cacheable:{decision.reason}")

        ns = namespace or decision.namespace or self.namespace
        try:
            await self._put_row(
                key=key,
                namespace=ns,
                tool_name=tool_name,
                arguments=arguments if self.store_arguments else None,
                context=self._select_context(decision, context),
                result=result,
                ttl=decision.ttl,
            )
        except Exception as exc:
            self.errors += 1
            log_event(EVENT_CACHE_ERROR, rid=request_id, op="tool_store", error=str(exc))
            if not self.fail_open:
                raise
            return ToolLookup(False, key, decision, error=str(exc))

        log_event(
            EVENT_TOOL_CACHE_WRITE,
            rid=request_id,
            tool=tool_name,
            key=key[:16],
            ttl=decision.ttl,
            category=decision.category,
        )
        return ToolLookup(
            True, key, decision, result=result, created_at=time.time(),
            expires_at=time.time() + decision.ttl,
        )

    # -- storage primitives ----------------------------------------------
    async def _get_row(self, key: str) -> dict[str, Any] | None:
        if self.db is None:
            return self._memory.get(key)
        row = await self.db.run(self.db.get_tool_entry, key)
        return dict(row) if row is not None else None

    async def _put_row(
        self,
        *,
        key: str,
        namespace: str,
        tool_name: str,
        arguments: Any,
        context: Mapping[str, Any],
        result: Any,
        ttl: int,
    ) -> None:
        stored_result = result if self.store_results else {"__redacted__": True}
        if self.db is None:
            self._memory[key] = {
                "key": key,
                "namespace": namespace,
                "tool_name": tool_name,
                "arguments": json.dumps(arguments) if arguments is not None else None,
                "result": json.dumps({"result": result}),
                "created_at": time.time(),
                "expires_at": time.time() + ttl if ttl else None,
                "hits": 0,
            }
            return
        await self.db.run(
            lambda: self.db.put_tool_entry(  # type: ignore[union-attr]
                key=key,
                namespace=namespace,
                tool_name=tool_name,
                arguments=arguments,
                context=dict(context),
                result=stored_result,
                ttl=ttl,
            )
        )

    async def _touch(self, key: str) -> None:
        if self.db is None:
            entry = self._memory.get(key)
            if entry is not None:
                entry["hits"] = int(entry.get("hits", 0)) + 1
            return
        try:
            await self.db.run(self.db.touch_tool_entry, key)
        except Exception:
            pass

    # -- invalidation -----------------------------------------------------
    async def invalidate_key(self, key: str) -> int:
        if self.db is None:
            return 1 if self._memory.pop(key, None) is not None else 0
        return await self.db.run(self.db.delete_tool_entry, key)

    async def invalidate_tool(self, tool_name: str) -> int:
        if self.db is None:
            keys = [k for k, v in self._memory.items() if v["tool_name"] == tool_name]
            for key in keys:
                self._memory.pop(key, None)
            return len(keys)
        return await self.db.run(self.db.delete_tool_entries_by_tool, tool_name)

    async def clear(self) -> int:
        if self.db is None:
            count = len(self._memory)
            self._memory.clear()
            return count
        return await self.db.run(self.db.clear_tool_cache)

    async def size(self) -> int:
        if self.db is None:
            return len(self._memory)
        return await self.db.run(self.db.count_tool_entries)

    async def list_entries(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        if self.db is None:
            rows = list(self._memory.values())[offset : offset + limit]
            return [dict(r) for r in rows]
        rows = await self.db.run(lambda: self.db.list_tool_entries(limit=limit, offset=offset))  # type: ignore[union-attr]
        out = []
        for row in rows:
            data = dict(row)
            if not self.store_arguments:
                data["arguments"] = None
            out.append(data)
        return out

    async def health(self) -> dict[str, Any]:
        return {
            "entries": await self.size(),
            "errors": self.errors,
            "storage": "sqlite" if self.db is not None else "memory",
        }
