"""Application container: wires configuration, database, cache engine, provider,
stats and pricing into a single object the HTTP layer and CLI both use.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from .cache.engine import CacheEngine
from .config import Config, ModelPricing, load_config
from .db import Database
from .logging_utils import configure_logging, get_logger, log_event
from .pricing import CostCalculator, TokenCounter
from .providers import ProviderAdapter, ProviderError, build_provider
from .stats import StatsCollector

log = get_logger("cachellm.app")


class AppState:
    """Owns every long-lived component.  One instance per process."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()
        configure_logging(self.config.server.log_level)

        self.db: Database | None = None
        if self.config.cache.backend == "sqlite" or self.config.semantic.enabled:
            self.db = Database(self.config.cache.sqlite_path)
        elif self.config.privacy.retention_days != 0:
            # History/stats still want durable storage even with a memory cache.
            self.db = Database(self.config.cache.sqlite_path)

        self._load_pricing_from_db()
        self.token_counter = TokenCounter(self.config.pricing)
        self.costs = CostCalculator(self.config.pricing)
        self.stats = StatsCollector(
            self.db,
            self.costs,
            store_prompts=self.config.privacy.store_prompts,
            store_responses=self.config.privacy.store_responses,
        )
        self.engine = CacheEngine(self.config, db=self.db)

        self._provider: ProviderAdapter | None = None
        self._provider_error: str | None = None
        self._maintenance_task: asyncio.Task[None] | None = None
        self.started_at = time.time()

    # -- provider ---------------------------------------------------------
    @property
    def provider(self) -> ProviderAdapter:
        if self._provider is None:
            self._provider = build_provider(
                self.config.upstream,
                request_timeout=self.config.server.request_timeout,
                connect_timeout=self.config.server.connect_timeout,
            )
        return self._provider

    def try_provider(self) -> ProviderAdapter | None:
        try:
            return self.provider
        except ProviderError as exc:
            self._provider_error = str(exc)
            return None

    @property
    def provider_error(self) -> str | None:
        return self._provider_error

    def resolve_model(self, model: str) -> str:
        return self.config.resolve_model(model)

    # -- pricing ----------------------------------------------------------
    def _load_pricing_from_db(self) -> None:
        """Dashboard-edited prices live in SQLite and override the config file."""
        if self.db is None:
            return
        try:
            for row in self.db.all_pricing():
                self.config.pricing.models[row["model"]] = ModelPricing(
                    input_per_1m=row["input_per_1m"],
                    output_per_1m=row["output_per_1m"],
                    cached_input_per_1m=row["cached_input_per_1m"],
                )
        except Exception as exc:  # pragma: no cover
            log.warning("could not load pricing overrides: %s", exc)

    def set_pricing(self, model: str, pricing: ModelPricing) -> None:
        self.costs.upsert(model, pricing)
        if self.db is not None:
            self.db.upsert_pricing(
                model, pricing.input_per_1m, pricing.output_per_1m, pricing.cached_input_per_1m
            )

    def delete_pricing(self, model: str) -> bool:
        removed = self.costs.remove(model)
        if self.db is not None:
            self.db.delete_pricing(model)
        return removed

    # -- lifecycle --------------------------------------------------------
    async def startup(self) -> None:
        log_event(
            "STARTUP",
            backend=self.config.cache.backend,
            namespace=self.config.cache.namespace,
            ttl=self.config.cache.default_ttl,
            semantic=self.config.semantic.enabled,
            upstream=self.config.upstream.base_url or "<unset>",
            port=self.config.server.port,
        )
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())

    async def shutdown(self) -> None:
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._provider is not None:
            await self._provider.close()
        await self.engine.close()
        if self.db is not None:
            self.db.close()

    async def _maintenance_loop(self, interval: float = 300.0) -> None:
        """Purge expired entries and apply the retention policy."""
        while True:
            try:
                await asyncio.sleep(interval)
                purged = await self.engine.purge_expired()
                pruned = 0
                if self.db is not None and self.config.privacy.retention_days > 0:
                    pruned = await self.db.run(
                        self.db.prune_history, self.config.privacy.retention_days
                    )
                if purged or pruned:
                    log_event("MAINTENANCE", purged=purged, pruned=pruned)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # never kill the loop
                log.warning("maintenance cycle failed: %s", exc)

    # -- misc -------------------------------------------------------------
    @staticmethod
    def new_request_id() -> str:
        return uuid.uuid4().hex[:12]

    async def health(self) -> dict[str, Any]:
        provider = self.try_provider()
        upstream = (
            await provider.health()
            if provider is not None
            else {"ok": False, "error": self._provider_error or "provider unavailable"}
        )
        return {
            "status": "ok",
            "version": _version(),
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "cache": await self.engine.health(),
            "upstream": upstream,
            "stats": self.stats.brief(),
            "config_file": self.config.source_path,
        }


def _version() -> str:
    from . import __version__

    return __version__
