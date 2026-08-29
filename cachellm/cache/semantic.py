"""Semantic cache (opt-in).

Safety model
------------
A semantic hit is only allowed inside an identical *context fingerprint*: same
model, same system/developer instructions, same tool schemas, same generation
parameters and same prior conversation.  Different system prompts therefore
produce different fingerprints and can never share an answer, no matter how
similar the user question is.

Storage: vectors live in the ``semantic_entries`` SQLite table alongside the
exact cache entry they point at; deleting the exact entry cascades.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..config import SemanticConfig
from ..db import Database
from ..embeddings import (
    EmbeddingBackend,
    build_embedder,
    cosine_similarity,
    pack_vector,
    unpack_vector,
)
from ..logging_utils import EVENT_CACHE_ERROR, get_logger, log_event
from .keys import KeyMaterial

log = get_logger("cachellm.cache.semantic")


@dataclass(slots=True)
class SemanticMatch:
    cache_key: str
    similarity: float
    text: str | None
    latency_ms: float


class SemanticCache:
    """Vector-similarity lookup scoped by context fingerprint."""

    def __init__(
        self,
        cfg: SemanticConfig,
        db: Database | None,
        *,
        embedder: EmbeddingBackend | None = None,
    ) -> None:
        self.cfg = cfg
        self.db = db
        self._embedder = embedder
        self._embedder_failed = False
        self.errors = 0

    # -- availability -----------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled and self.db is not None and not self._embedder_failed)

    def embedder(self) -> EmbeddingBackend | None:
        if self._embedder is not None:
            return self._embedder
        if self._embedder_failed:
            return None
        try:
            self._embedder = build_embedder(self.cfg)
        except Exception as exc:
            self._embedder_failed = True
            log.warning("semantic cache disabled: %s", exc)
            return None
        return self._embedder

    def eligible(self, material: KeyMaterial, *, semantic_allowed: bool) -> tuple[bool, str]:
        """Decide whether this request may participate in semantic caching."""
        if not self.enabled:
            return False, "disabled"
        if not semantic_allowed:
            return False, "policy"
        if material.tool_names and not self.cfg.allow_tools:
            return False, "tools_present"
        text = material.query_text
        if len(text) < self.cfg.min_chars:
            return False, "query_too_short"
        if len(text) > self.cfg.max_chars:
            return False, "query_too_long"
        if self.cfg.require_single_turn:
            # 1 user turn (+ optional system/developer messages) only.
            non_context = material.message_count
            if non_context > 3:
                return False, "multi_turn_conversation"
        return True, "eligible"

    # -- lookup / store ---------------------------------------------------
    async def lookup(
        self, material: KeyMaterial, *, request_id: str | None = None
    ) -> SemanticMatch | None:
        embedder = self.embedder()
        if embedder is None or self.db is None:
            return None
        started = time.perf_counter()
        try:
            vector = await embedder.embed(material.query_text)
            rows = await self.db.run(
                lambda: self.db.semantic_candidates(  # type: ignore[union-attr]
                    namespace=material.namespace,
                    context_hash=material.context_hash,
                    limit=self.cfg.max_candidates,
                )
            )
        except Exception as exc:
            self.errors += 1
            log_event(EVENT_CACHE_ERROR, rid=request_id, op="semantic_lookup", error=str(exc))
            return None

        best_key: str | None = None
        best_score = 0.0
        best_text: str | None = None
        for row in rows:
            candidate = unpack_vector(row["embedding"])
            if len(candidate) != len(vector):
                continue
            score = cosine_similarity(vector, candidate)
            if score > best_score:
                best_score, best_key, best_text = score, row["cache_key"], row["text"]

        latency = (time.perf_counter() - started) * 1000.0
        if best_key is None or best_score < self.cfg.threshold:
            return None
        return SemanticMatch(best_key, best_score, best_text, latency)

    async def store(
        self,
        material: KeyMaterial,
        *,
        ttl: int | None,
        store_text: bool = True,
        request_id: str | None = None,
    ) -> bool:
        embedder = self.embedder()
        if embedder is None or self.db is None:
            return False
        try:
            vector = await embedder.embed(material.query_text)
            await self.db.run(
                lambda: self.db.add_semantic_entry(  # type: ignore[union-attr]
                    cache_key=material.key,
                    namespace=material.namespace,
                    context_hash=material.context_hash,
                    model=material.model,
                    text=material.query_text if store_text else None,
                    embedding=pack_vector(vector),
                    dims=len(vector),
                    ttl=ttl,
                )
            )
        except Exception as exc:
            self.errors += 1
            log_event(EVENT_CACHE_ERROR, rid=request_id, op="semantic_store", error=str(exc))
            return False
        return True

    async def delete_by_cache_key(self, cache_key: str) -> int:
        if self.db is None:
            return 0
        return await self.db.run(self.db.delete_semantic_by_key, cache_key)

    async def health(self) -> dict[str, Any]:
        embedder = self._embedder
        return {
            "enabled": self.enabled,
            "configured": self.cfg.enabled,
            "backend": self.cfg.backend,
            "model": self.cfg.model,
            "threshold": self.cfg.threshold,
            "dimensions": getattr(embedder, "dimensions", self.cfg.dimensions),
            "errors": self.errors,
            "entries": (
                await self.db.run(self.db.count_semantic_entries) if self.db is not None else 0
            ),
        }

    async def close(self) -> None:
        if self._embedder is not None:
            await self._embedder.close()
