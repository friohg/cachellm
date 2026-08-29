"""Statistics collector.

In-process counters (cheap, always available) plus durable aggregates written to
SQLite so numbers survive restarts.  Aggregate stats keep working even when
privacy settings disable prompt/response storage.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Mapping

from .db import Database
from .pricing import CostCalculator

OUTCOME_EXACT_HIT = "exact_hit"
OUTCOME_SEMANTIC_HIT = "semantic_hit"
OUTCOME_TOOL_HIT = "tool_hit"
OUTCOME_MISS = "miss"
OUTCOME_COALESCED = "coalesced"
OUTCOME_BYPASS = "bypass"
OUTCOME_ERROR = "error"

HIT_OUTCOMES = frozenset({OUTCOME_EXACT_HIT, OUTCOME_SEMANTIC_HIT, OUTCOME_TOOL_HIT})


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


@dataclass
class RequestRecord:
    """One proxied request, as stored in the ``requests`` table."""

    request_id: str
    endpoint: str
    outcome: str
    model: str = ""
    upstream_model: str = ""
    namespace: str = "default"
    cache_key: str | None = None
    category: str | None = None
    stream: bool = False
    status_code: int | None = None
    similarity: float | None = None
    total_latency_ms: float = 0.0
    cache_latency_ms: float = 0.0
    upstream_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_estimated: bool = False
    cost: float = 0.0
    cost_without_cache: float = 0.0
    error: str | None = None
    prompt_excerpt: str | None = None
    response_excerpt: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_row(self) -> dict[str, Any]:
        data = asdict(self)
        data["stream"] = 1 if self.stream else 0
        data["tokens_estimated"] = 1 if self.tokens_estimated else 0
        return data

    @property
    def is_hit(self) -> bool:
        return self.outcome in HIT_OUTCOMES


@dataclass
class Counters:
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    tool_cache_hits: int = 0
    tool_cache_misses: int = 0
    upstream_requests: int = 0
    upstream_errors: int = 0
    coalesced_requests: int = 0
    bypassed_requests: int = 0
    cache_errors: int = 0
    cache_writes: int = 0
    streamed_requests: int = 0
    aborted_streams: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    saved_input_tokens: int = 0
    saved_output_tokens: int = 0
    actual_cost: float = 0.0
    cost_without_cache: float = 0.0


class StatsCollector:
    """Thread-safe counters + rolling latency/recent-request windows."""

    def __init__(
        self,
        db: Database | None,
        cost_calculator: CostCalculator,
        *,
        store_prompts: bool = True,
        store_responses: bool = True,
        recent_window: int = 100,
    ) -> None:
        self.db = db
        self.costs = cost_calculator
        self.store_prompts = store_prompts
        self.store_responses = store_responses
        self._lock = threading.Lock()
        self.counters = Counters()
        self.started_at = time.time()
        self._total_latency: Deque[float] = deque(maxlen=500)
        self._cache_latency: Deque[float] = deque(maxlen=500)
        self._upstream_latency: Deque[float] = deque(maxlen=500)
        self._recent: Deque[dict[str, Any]] = deque(maxlen=recent_window)

    # -- recording --------------------------------------------------------
    def record(self, record: RequestRecord) -> None:
        with self._lock:
            c = self.counters
            c.total_requests += 1
            if record.outcome == OUTCOME_EXACT_HIT:
                c.exact_hits += 1
                c.cache_hits += 1
            elif record.outcome == OUTCOME_SEMANTIC_HIT:
                c.semantic_hits += 1
                c.cache_hits += 1
            elif record.outcome == OUTCOME_TOOL_HIT:
                c.tool_cache_hits += 1
                c.cache_hits += 1
            elif record.outcome == OUTCOME_MISS:
                c.cache_misses += 1
            elif record.outcome == OUTCOME_COALESCED:
                c.coalesced_requests += 1
                c.cache_hits += 1
            elif record.outcome == OUTCOME_BYPASS:
                c.bypassed_requests += 1
            elif record.outcome == OUTCOME_ERROR:
                c.upstream_errors += 1

            if record.is_hit or record.outcome == OUTCOME_COALESCED:
                c.saved_input_tokens += record.prompt_tokens
                c.saved_output_tokens += record.completion_tokens
            else:
                c.input_tokens += record.prompt_tokens
                c.output_tokens += record.completion_tokens

            c.actual_cost += record.cost
            c.cost_without_cache += record.cost_without_cache
            if record.stream:
                c.streamed_requests += 1

            if record.total_latency_ms:
                self._total_latency.append(record.total_latency_ms)
            if record.cache_latency_ms:
                self._cache_latency.append(record.cache_latency_ms)
            if record.upstream_latency_ms:
                self._upstream_latency.append(record.upstream_latency_ms)

            self._recent.appendleft(self._recent_view(record))

        if self.db is not None:
            self._persist(record)

    def _recent_view(self, record: RequestRecord) -> dict[str, Any]:
        return {
            "request_id": record.request_id,
            "created_at": record.created_at,
            "endpoint": record.endpoint,
            "model": record.model,
            "outcome": record.outcome,
            "category": record.category,
            "stream": record.stream,
            "status_code": record.status_code,
            "similarity": record.similarity,
            "total_latency_ms": round(record.total_latency_ms, 2),
            "cache_latency_ms": round(record.cache_latency_ms, 2),
            "upstream_latency_ms": round(record.upstream_latency_ms, 2),
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": record.completion_tokens,
            "cost": round(record.cost, 8),
            "cost_without_cache": round(record.cost_without_cache, 8),
            "cache_key": (record.cache_key or "")[:16] or None,
            "error": record.error,
            "prompt_excerpt": record.prompt_excerpt if self.store_prompts else None,
            "response_excerpt": record.response_excerpt if self.store_responses else None,
        }

    def _persist(self, record: RequestRecord) -> None:
        assert self.db is not None
        row = record.to_row()
        if not self.store_prompts:
            row["prompt_excerpt"] = None
        if not self.store_responses:
            row["response_excerpt"] = None
        try:
            self.db.insert_request(row)
            self.db.bump_usage(
                day=today_utc(),
                model=record.model or "unknown",
                hit=record.is_hit or record.outcome == OUTCOME_COALESCED,
                prompt_tokens=record.prompt_tokens,
                completion_tokens=record.completion_tokens,
                cost=record.cost,
                cost_without_cache=record.cost_without_cache,
            )
        except Exception:  # stats must never break the proxy
            with self._lock:
                self.counters.cache_errors += 1

    # -- individual event counters ---------------------------------------
    def note_upstream_request(self) -> None:
        with self._lock:
            self.counters.upstream_requests += 1

    def note_cache_write(self) -> None:
        with self._lock:
            self.counters.cache_writes += 1

    def note_cache_error(self) -> None:
        with self._lock:
            self.counters.cache_errors += 1

    def note_tool_miss(self) -> None:
        with self._lock:
            self.counters.tool_cache_misses += 1

    def note_stream_abort(self) -> None:
        with self._lock:
            self.counters.aborted_streams += 1

    # -- reporting --------------------------------------------------------
    @staticmethod
    def _avg(values: Deque[float]) -> float:
        return round(sum(values) / len(values), 2) if values else 0.0

    def snapshot(self, *, include_db: bool = True) -> dict[str, Any]:
        with self._lock:
            counters = asdict(self.counters)
            latency = {
                "avg_total_ms": self._avg(self._total_latency),
                "avg_cache_lookup_ms": self._avg(self._cache_latency),
                "avg_upstream_ms": self._avg(self._upstream_latency),
                "samples": len(self._total_latency),
            }
            recent = list(self._recent)

        total = counters["total_requests"]
        hits = counters["cache_hits"]
        hit_rate = round(100.0 * hits / total, 2) if total else 0.0
        saved_tokens = counters["saved_input_tokens"] + counters["saved_output_tokens"]
        savings = max(0.0, counters["cost_without_cache"] - counters["actual_cost"])

        out: dict[str, Any] = {
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "counters": counters,
            "cache_hit_rate_pct": hit_rate,
            "estimated_tokens_saved": saved_tokens,
            "latency": latency,
            "cost": {
                "currency": self.costs.currency,
                "actual_cost": round(counters["actual_cost"], 6),
                "cost_without_cache": round(counters["cost_without_cache"], 6),
                "savings": round(savings, 6),
                "savings_pct": (
                    round(100.0 * savings / counters["cost_without_cache"], 2)
                    if counters["cost_without_cache"] > 0
                    else 0.0
                ),
            },
            "recent_requests": recent[:25],
        }

        if include_db and self.db is not None:
            try:
                out["lifetime"] = self.db.usage_totals()
                out["by_model"] = [dict(r) for r in self.db.usage_by_model()]
                out["storage"] = self.db.stats_snapshot()
                out["lifetime_cost"] = self.costs.totals(out["by_model"])
            except Exception as exc:
                out["db_error"] = str(exc)
        return out

    def brief(self) -> dict[str, Any]:
        """Compact form used by ``cachellm stats`` and /health."""
        snap = self.snapshot(include_db=False)
        c = snap["counters"]
        return {
            "requests": c["total_requests"],
            "hits": c["cache_hits"],
            "misses": c["cache_misses"],
            "exact_hits": c["exact_hits"],
            "semantic_hits": c["semantic_hits"],
            "tool_hits": c["tool_cache_hits"],
            "coalesced": c["coalesced_requests"],
            "upstream_requests": c["upstream_requests"],
            "upstream_errors": c["upstream_errors"],
            "hit_rate_pct": snap["cache_hit_rate_pct"],
            "tokens_saved": snap["estimated_tokens_saved"],
            "savings": snap["cost"]["savings"],
            "currency": snap["cost"]["currency"],
            "avg_latency_ms": snap["latency"]["avg_total_ms"],
        }

    def recent(self, limit: int = 50, outcome: str | None = None) -> list[dict[str, Any]]:
        if self.db is not None:
            try:
                rows = self.db.recent_requests(limit=limit, outcome=outcome)
                return [self._scrub_row(dict(r)) for r in rows]
            except Exception:
                pass
        with self._lock:
            items = list(self._recent)
        if outcome:
            items = [i for i in items if i.get("outcome") == outcome]
        return items[:limit]

    def _scrub_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        data = dict(row)
        if not self.store_prompts:
            data["prompt_excerpt"] = None
        if not self.store_responses:
            data["response_excerpt"] = None
        return data
