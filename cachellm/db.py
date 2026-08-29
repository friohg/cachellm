"""SQLite persistence layer.

Holds cache entries, tool-cache entries, request history, usage aggregates,
model pricing and dashboard-editable configuration overrides.

Design notes
------------
* One connection per thread (``threading.local``) with WAL enabled - safe for
  uvicorn's threadpool and for the CLI touching the same file concurrently.
* Every statement is parameterised.
* Secrets are never inserted: the ``configuration`` table rejects keys that
  look like credentials (see :data:`_SECRET_KEY_RE`).
* All timestamps are Unix epoch seconds (REAL) in UTC.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypeVar

from .logging_utils import get_logger

log = get_logger("cachellm.db")

T = TypeVar("T")

SCHEMA_VERSION = 1

_SECRET_KEY_RE = re.compile(r"(api[_-]?key|secret|password|token|credential)", re.I)


SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cache_entries (
    key           TEXT PRIMARY KEY,
    namespace     TEXT NOT NULL DEFAULT 'default',
    model         TEXT,
    endpoint      TEXT,
    category      TEXT,
    payload       TEXT NOT NULL,          -- JSON: cached upstream response
    request_json  TEXT,                   -- JSON: normalized request (privacy-gated)
    prompt_excerpt TEXT,                  -- short preview for the cache browser
    response_excerpt TEXT,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    created_at    REAL NOT NULL,
    expires_at    REAL,                   -- NULL = never expires
    hits          INTEGER NOT NULL DEFAULT 0,
    last_hit_at   REAL,
    size_bytes    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_cache_ns      ON cache_entries(namespace);
CREATE INDEX IF NOT EXISTS idx_cache_model   ON cache_entries(model);
CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at);
CREATE INDEX IF NOT EXISTS idx_cache_created ON cache_entries(created_at);

CREATE TABLE IF NOT EXISTS semantic_entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key   TEXT NOT NULL,
    namespace   TEXT NOT NULL DEFAULT 'default',
    context_hash TEXT NOT NULL,           -- model + system + tools + params
    model       TEXT,
    text        TEXT,                     -- normalized semantic text (privacy-gated)
    embedding   BLOB NOT NULL,            -- float32 little-endian
    dims        INTEGER NOT NULL,
    created_at  REAL NOT NULL,
    expires_at  REAL,
    FOREIGN KEY (cache_key) REFERENCES cache_entries(key) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_sem_ctx  ON semantic_entries(context_hash);
CREATE INDEX IF NOT EXISTS idx_sem_key  ON semantic_entries(cache_key);

CREATE TABLE IF NOT EXISTS tool_cache_entries (
    key         TEXT PRIMARY KEY,
    namespace   TEXT NOT NULL DEFAULT 'default',
    tool_name   TEXT NOT NULL,
    arguments   TEXT,                     -- JSON (privacy-gated)
    context     TEXT,                     -- JSON of context fields
    result      TEXT NOT NULL,            -- JSON envelope {"result": ...}
    created_at  REAL NOT NULL,
    expires_at  REAL,
    hits        INTEGER NOT NULL DEFAULT 0,
    last_hit_at REAL,
    size_bytes  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_tool_name    ON tool_cache_entries(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_ns      ON tool_cache_entries(namespace);
CREATE INDEX IF NOT EXISTS idx_tool_expires ON tool_cache_entries(expires_at);

CREATE TABLE IF NOT EXISTS requests (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT NOT NULL,
    created_at      REAL NOT NULL,
    endpoint        TEXT NOT NULL,
    model           TEXT,
    upstream_model  TEXT,
    namespace       TEXT,
    cache_key       TEXT,
    outcome         TEXT NOT NULL,        -- exact_hit|semantic_hit|miss|coalesced|bypass|error|tool_hit
    category        TEXT,
    stream          INTEGER NOT NULL DEFAULT 0,
    status_code     INTEGER,
    similarity      REAL,
    total_latency_ms   REAL,
    cache_latency_ms   REAL,
    upstream_latency_ms REAL,
    prompt_tokens   INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    tokens_estimated INTEGER NOT NULL DEFAULT 0,
    cost            REAL NOT NULL DEFAULT 0.0,
    cost_without_cache REAL NOT NULL DEFAULT 0.0,
    error           TEXT,
    prompt_excerpt  TEXT,
    response_excerpt TEXT
);
CREATE INDEX IF NOT EXISTS idx_req_created ON requests(created_at);
CREATE INDEX IF NOT EXISTS idx_req_outcome ON requests(outcome);
CREATE INDEX IF NOT EXISTS idx_req_model   ON requests(model);

CREATE TABLE IF NOT EXISTS usage (
    day               TEXT NOT NULL,
    model             TEXT NOT NULL,
    requests          INTEGER NOT NULL DEFAULT 0,
    hits              INTEGER NOT NULL DEFAULT 0,
    misses            INTEGER NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    saved_prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    saved_completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost              REAL NOT NULL DEFAULT 0.0,
    cost_without_cache REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (day, model)
);

CREATE TABLE IF NOT EXISTS model_pricing (
    model               TEXT PRIMARY KEY,
    input_per_1m        REAL NOT NULL DEFAULT 0.0,
    output_per_1m       REAL NOT NULL DEFAULT 0.0,
    cached_input_per_1m REAL,
    updated_at          REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS configuration (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,             -- JSON encoded
    updated_at REAL NOT NULL
);
"""


@dataclass(slots=True)
class CacheRow:
    key: str
    namespace: str
    model: str | None
    endpoint: str | None
    category: str | None
    payload: dict[str, Any]
    created_at: float
    expires_at: float | None
    hits: int
    size_bytes: int
    prompt_excerpt: str | None = None
    response_excerpt: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= time.time()


class Database:
    """Thread-safe SQLite wrapper."""

    def __init__(self, path: str | Path, *, timeout: float = 15.0) -> None:
        self.path = str(path)
        self._timeout = timeout
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._is_memory = self.path in {":memory:", "file::memory:"}
        self._shared_conn: sqlite3.Connection | None = None
        if not self._is_memory:
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self.path = str(Path(self.path).expanduser())
        self._ensure_schema()

    # -- connection handling ---------------------------------------------
    def connect(self) -> sqlite3.Connection:
        if self._is_memory:
            # A single shared connection keeps an in-memory DB alive/consistent.
            with self._init_lock:
                if self._shared_conn is None:
                    self._shared_conn = self._new_connection()
                return self._shared_conn
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._new_connection()
            self._local.conn = conn
        return conn

    def _new_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path,
            timeout=self._timeout,
            check_same_thread=False,
            isolation_level=None,  # autocommit; explicit BEGIN when needed
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=15000")
        return conn

    def _ensure_schema(self) -> None:
        conn = self.connect()
        with self._init_lock:
            conn.executescript(SCHEMA)
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
        if self._shared_conn is not None:
            self._shared_conn.close()
            self._shared_conn = None

    # -- primitives -------------------------------------------------------
    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self.connect().execute(sql, tuple(params))

    def executemany(self, sql: str, seq: Iterable[Sequence[Any]]) -> sqlite3.Cursor:
        return self.connect().executemany(sql, [tuple(p) for p in seq])

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self.execute(sql, params).fetchall())

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        return self.execute(sql, params).fetchone()

    async def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run a blocking DB function off the event loop."""
        return await asyncio.to_thread(fn, *args, **kwargs)

    # -- cache entries ----------------------------------------------------
    def get_cache_entry(self, key: str) -> CacheRow | None:
        row = self.query_one("SELECT * FROM cache_entries WHERE key = ?", (key,))
        if row is None:
            return None
        try:
            payload = json.loads(row["payload"])
        except json.JSONDecodeError:
            log.warning("corrupt cache payload for key=%s; dropping", key)
            self.delete_cache_entry(key)
            return None
        return CacheRow(
            key=row["key"],
            namespace=row["namespace"],
            model=row["model"],
            endpoint=row["endpoint"],
            category=row["category"],
            payload=payload,
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            hits=row["hits"],
            size_bytes=row["size_bytes"],
            prompt_excerpt=row["prompt_excerpt"],
            response_excerpt=row["response_excerpt"],
            prompt_tokens=row["prompt_tokens"] or 0,
            completion_tokens=row["completion_tokens"] or 0,
        )

    def put_cache_entry(
        self,
        *,
        key: str,
        namespace: str,
        payload: dict[str, Any],
        model: str | None = None,
        endpoint: str | None = None,
        category: str | None = None,
        request_json: dict[str, Any] | None = None,
        prompt_excerpt: str | None = None,
        response_excerpt: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        ttl: int | None = None,
    ) -> None:
        now = time.time()
        payload_text = json.dumps(payload, ensure_ascii=False)
        expires_at = None if not ttl else now + float(ttl)
        self.execute(
            """
            INSERT INTO cache_entries
                (key, namespace, model, endpoint, category, payload, request_json,
                 prompt_excerpt, response_excerpt, prompt_tokens, completion_tokens,
                 created_at, expires_at, hits, size_bytes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)
            ON CONFLICT(key) DO UPDATE SET
                payload=excluded.payload,
                request_json=excluded.request_json,
                prompt_excerpt=excluded.prompt_excerpt,
                response_excerpt=excluded.response_excerpt,
                prompt_tokens=excluded.prompt_tokens,
                completion_tokens=excluded.completion_tokens,
                created_at=excluded.created_at,
                expires_at=excluded.expires_at,
                category=excluded.category,
                size_bytes=excluded.size_bytes
            """,
            (
                key,
                namespace,
                model,
                endpoint,
                category,
                payload_text,
                json.dumps(request_json, ensure_ascii=False) if request_json else None,
                prompt_excerpt,
                response_excerpt,
                int(prompt_tokens or 0),
                int(completion_tokens or 0),
                now,
                expires_at,
                len(payload_text.encode("utf-8")),
            ),
        )

    def touch_cache_entry(self, key: str) -> None:
        self.execute(
            "UPDATE cache_entries SET hits = hits + 1, last_hit_at = ? WHERE key = ?",
            (time.time(), key),
        )

    def delete_cache_entry(self, key: str) -> int:
        cur = self.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
        return cur.rowcount or 0

    def delete_cache_by_model(self, model: str) -> int:
        cur = self.execute("DELETE FROM cache_entries WHERE model = ?", (model,))
        return cur.rowcount or 0

    def delete_cache_by_namespace(self, namespace: str) -> int:
        cur = self.execute("DELETE FROM cache_entries WHERE namespace = ?", (namespace,))
        return cur.rowcount or 0

    def clear_cache(self) -> int:
        cur = self.execute("DELETE FROM cache_entries")
        self.execute("DELETE FROM semantic_entries")
        return cur.rowcount or 0

    def purge_expired(self, now: float | None = None) -> int:
        now = now if now is not None else time.time()
        total = 0
        for table in ("cache_entries", "tool_cache_entries", "semantic_entries"):
            cur = self.execute(
                f"DELETE FROM {table} WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            total += cur.rowcount or 0
        return total

    def list_cache_entries(
        self,
        *,
        search: str | None = None,
        namespace: str | None = None,
        model: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[sqlite3.Row]:
        clauses: list[str] = []
        params: list[Any] = []
        if search:
            clauses.append("(key LIKE ? OR prompt_excerpt LIKE ? OR response_excerpt LIKE ?)")
            like = f"%{search}%"
            params += [like, like, like]
        if namespace:
            clauses.append("namespace = ?")
            params.append(namespace)
        if model:
            clauses.append("model = ?")
            params.append(model)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params += [int(limit), int(offset)]
        return self.query(
            f"""
            SELECT key, namespace, model, endpoint, category, created_at, expires_at,
                   hits, size_bytes, prompt_excerpt, response_excerpt,
                   prompt_tokens, completion_tokens
            FROM cache_entries {where}
            ORDER BY created_at DESC LIMIT ? OFFSET ?
            """,
            params,
        )

    def count_cache_entries(self) -> int:
        row = self.query_one("SELECT COUNT(*) AS n FROM cache_entries")
        return int(row["n"]) if row else 0

    def cache_entry_detail(self, key: str) -> dict[str, Any] | None:
        row = self.query_one("SELECT * FROM cache_entries WHERE key = ?", (key,))
        if row is None:
            return None
        data = dict(row)
        for field_name in ("payload", "request_json"):
            if data.get(field_name):
                try:
                    data[field_name] = json.loads(data[field_name])
                except json.JSONDecodeError:
                    data[field_name] = None
        return data

    # -- semantic entries -------------------------------------------------
    def add_semantic_entry(
        self,
        *,
        cache_key: str,
        namespace: str,
        context_hash: str,
        model: str | None,
        text: str | None,
        embedding: bytes,
        dims: int,
        ttl: int | None,
    ) -> None:
        now = time.time()
        self.execute(
            """
            INSERT INTO semantic_entries
                (cache_key, namespace, context_hash, model, text, embedding, dims,
                 created_at, expires_at)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                cache_key,
                namespace,
                context_hash,
                model,
                text,
                embedding,
                dims,
                now,
                None if not ttl else now + float(ttl),
            ),
        )

    def semantic_candidates(
        self, *, namespace: str, context_hash: str, limit: int
    ) -> list[sqlite3.Row]:
        return self.query(
            """
            SELECT s.id, s.cache_key, s.embedding, s.dims, s.text, s.expires_at
            FROM semantic_entries s
            JOIN cache_entries c ON c.key = s.cache_key
            WHERE s.namespace = ? AND s.context_hash = ?
              AND (s.expires_at IS NULL OR s.expires_at > ?)
              AND (c.expires_at IS NULL OR c.expires_at > ?)
            ORDER BY s.created_at DESC LIMIT ?
            """,
            (namespace, context_hash, time.time(), time.time(), int(limit)),
        )

    def delete_semantic_by_key(self, cache_key: str) -> int:
        cur = self.execute("DELETE FROM semantic_entries WHERE cache_key = ?", (cache_key,))
        return cur.rowcount or 0

    def count_semantic_entries(self) -> int:
        row = self.query_one("SELECT COUNT(*) AS n FROM semantic_entries")
        return int(row["n"]) if row else 0

    # -- tool cache -------------------------------------------------------
    def get_tool_entry(self, key: str) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM tool_cache_entries WHERE key = ?", (key,))

    def put_tool_entry(
        self,
        *,
        key: str,
        namespace: str,
        tool_name: str,
        arguments: Any,
        context: Any,
        result: Any,
        ttl: int | None,
    ) -> None:
        now = time.time()
        result_text = json.dumps({"result": result}, ensure_ascii=False)
        self.execute(
            """
            INSERT INTO tool_cache_entries
                (key, namespace, tool_name, arguments, context, result,
                 created_at, expires_at, hits, size_bytes)
            VALUES (?,?,?,?,?,?,?,?,0,?)
            ON CONFLICT(key) DO UPDATE SET
                result=excluded.result,
                arguments=excluded.arguments,
                context=excluded.context,
                created_at=excluded.created_at,
                expires_at=excluded.expires_at,
                size_bytes=excluded.size_bytes
            """,
            (
                key,
                namespace,
                tool_name,
                json.dumps(arguments, ensure_ascii=False) if arguments is not None else None,
                json.dumps(context, ensure_ascii=False) if context else None,
                result_text,
                now,
                None if not ttl else now + float(ttl),
                len(result_text.encode("utf-8")),
            ),
        )

    def touch_tool_entry(self, key: str) -> None:
        self.execute(
            "UPDATE tool_cache_entries SET hits = hits + 1, last_hit_at = ? WHERE key = ?",
            (time.time(), key),
        )

    def delete_tool_entry(self, key: str) -> int:
        cur = self.execute("DELETE FROM tool_cache_entries WHERE key = ?", (key,))
        return cur.rowcount or 0

    def delete_tool_entries_by_tool(self, tool_name: str) -> int:
        cur = self.execute("DELETE FROM tool_cache_entries WHERE tool_name = ?", (tool_name,))
        return cur.rowcount or 0

    def clear_tool_cache(self) -> int:
        cur = self.execute("DELETE FROM tool_cache_entries")
        return cur.rowcount or 0

    def list_tool_entries(self, *, limit: int = 50, offset: int = 0) -> list[sqlite3.Row]:
        return self.query(
            """
            SELECT key, namespace, tool_name, arguments, created_at, expires_at, hits, size_bytes
            FROM tool_cache_entries ORDER BY created_at DESC LIMIT ? OFFSET ?
            """,
            (int(limit), int(offset)),
        )

    def count_tool_entries(self) -> int:
        row = self.query_one("SELECT COUNT(*) AS n FROM tool_cache_entries")
        return int(row["n"]) if row else 0

    # -- request log / usage ---------------------------------------------
    def insert_request(self, record: dict[str, Any]) -> None:
        columns = (
            "request_id", "created_at", "endpoint", "model", "upstream_model",
            "namespace", "cache_key", "outcome", "category", "stream", "status_code",
            "similarity", "total_latency_ms", "cache_latency_ms", "upstream_latency_ms",
            "prompt_tokens", "completion_tokens", "tokens_estimated", "cost",
            "cost_without_cache", "error", "prompt_excerpt", "response_excerpt",
        )
        placeholders = ",".join("?" for _ in columns)
        self.execute(
            f"INSERT INTO requests ({','.join(columns)}) VALUES ({placeholders})",
            [record.get(c) for c in columns],
        )

    def recent_requests(self, limit: int = 50, outcome: str | None = None) -> list[sqlite3.Row]:
        if outcome:
            return self.query(
                "SELECT * FROM requests WHERE outcome = ? ORDER BY id DESC LIMIT ?",
                (outcome, int(limit)),
            )
        return self.query("SELECT * FROM requests ORDER BY id DESC LIMIT ?", (int(limit),))

    def bump_usage(
        self,
        *,
        day: str,
        model: str,
        hit: bool,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        cost_without_cache: float,
    ) -> None:
        self.execute(
            """
            INSERT INTO usage (day, model, requests, hits, misses, prompt_tokens,
                completion_tokens, saved_prompt_tokens, saved_completion_tokens,
                cost, cost_without_cache)
            VALUES (?,?,1,?,?,?,?,?,?,?,?)
            ON CONFLICT(day, model) DO UPDATE SET
                requests = requests + 1,
                hits = hits + excluded.hits,
                misses = misses + excluded.misses,
                prompt_tokens = prompt_tokens + excluded.prompt_tokens,
                completion_tokens = completion_tokens + excluded.completion_tokens,
                saved_prompt_tokens = saved_prompt_tokens + excluded.saved_prompt_tokens,
                saved_completion_tokens = saved_completion_tokens + excluded.saved_completion_tokens,
                cost = cost + excluded.cost,
                cost_without_cache = cost_without_cache + excluded.cost_without_cache
            """,
            (
                day,
                model,
                1 if hit else 0,
                0 if hit else 1,
                0 if hit else int(prompt_tokens),
                0 if hit else int(completion_tokens),
                int(prompt_tokens) if hit else 0,
                int(completion_tokens) if hit else 0,
                float(cost),
                float(cost_without_cache),
            ),
        )

    def usage_totals(self) -> dict[str, Any]:
        row = self.query_one(
            """
            SELECT COALESCE(SUM(requests),0) AS requests,
                   COALESCE(SUM(hits),0) AS hits,
                   COALESCE(SUM(misses),0) AS misses,
                   COALESCE(SUM(prompt_tokens),0) AS prompt_tokens,
                   COALESCE(SUM(completion_tokens),0) AS completion_tokens,
                   COALESCE(SUM(saved_prompt_tokens),0) AS saved_prompt_tokens,
                   COALESCE(SUM(saved_completion_tokens),0) AS saved_completion_tokens,
                   COALESCE(SUM(cost),0.0) AS cost,
                   COALESCE(SUM(cost_without_cache),0.0) AS cost_without_cache
            FROM usage
            """
        )
        return dict(row) if row else {}

    def usage_by_model(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.query(
            """
            SELECT model,
                   SUM(requests) AS requests,
                   SUM(hits) AS hits,
                   SUM(prompt_tokens) AS prompt_tokens,
                   SUM(completion_tokens) AS completion_tokens,
                   SUM(saved_prompt_tokens) AS saved_prompt_tokens,
                   SUM(saved_completion_tokens) AS saved_completion_tokens,
                   SUM(cost) AS cost,
                   SUM(cost_without_cache) AS cost_without_cache
            FROM usage GROUP BY model ORDER BY requests DESC LIMIT ?
            """,
            (int(limit),),
        )

    def prune_history(self, retention_days: int) -> int:
        if retention_days <= 0:
            return 0
        cutoff = time.time() - retention_days * 86400
        cur = self.execute("DELETE FROM requests WHERE created_at < ?", (cutoff,))
        return cur.rowcount or 0

    # -- pricing ----------------------------------------------------------
    def upsert_pricing(
        self, model: str, input_per_1m: float, output_per_1m: float,
        cached_input_per_1m: float | None = None,
    ) -> None:
        self.execute(
            """
            INSERT INTO model_pricing (model, input_per_1m, output_per_1m,
                                       cached_input_per_1m, updated_at)
            VALUES (?,?,?,?,?)
            ON CONFLICT(model) DO UPDATE SET
                input_per_1m=excluded.input_per_1m,
                output_per_1m=excluded.output_per_1m,
                cached_input_per_1m=excluded.cached_input_per_1m,
                updated_at=excluded.updated_at
            """,
            (model, float(input_per_1m), float(output_per_1m), cached_input_per_1m, time.time()),
        )

    def delete_pricing(self, model: str) -> int:
        cur = self.execute("DELETE FROM model_pricing WHERE model = ?", (model,))
        return cur.rowcount or 0

    def all_pricing(self) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM model_pricing ORDER BY model")

    # -- configuration overrides -----------------------------------------
    def set_config_value(self, key: str, value: Any) -> None:
        if _SECRET_KEY_RE.search(key):
            raise ValueError(f"refusing to persist secret-like config key: {key!r}")
        self.execute(
            """
            INSERT INTO configuration (key, value, updated_at) VALUES (?,?,?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (key, json.dumps(value, ensure_ascii=False), time.time()),
        )

    def get_config_value(self, key: str, default: Any = None) -> Any:
        row = self.query_one("SELECT value FROM configuration WHERE key = ?", (key,))
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return default

    def all_config_values(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for row in self.query("SELECT key, value FROM configuration"):
            try:
                out[row["key"]] = json.loads(row["value"])
            except json.JSONDecodeError:
                continue
        return out

    def delete_config_value(self, key: str) -> int:
        cur = self.execute("DELETE FROM configuration WHERE key = ?", (key,))
        return cur.rowcount or 0

    # -- maintenance ------------------------------------------------------
    def vacuum(self) -> None:
        self.connect().execute("VACUUM")

    def stats_snapshot(self) -> dict[str, int]:
        return {
            "cache_entries": self.count_cache_entries(),
            "semantic_entries": self.count_semantic_entries(),
            "tool_cache_entries": self.count_tool_entries(),
        }
