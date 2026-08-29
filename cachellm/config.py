"""Configuration: layered defaults <- config file <- environment variables.

Config file format is JSON (stdlib, no extra dependency) and is searched in:

    $CACHELLM_CONFIG
    ./cachellm.config.json
    ./config/cachellm.json
    ~/.cachellm/config.json

Secrets (upstream API keys) are read from the environment or the config file,
kept in memory only, and never written to the database.  ``Config.redacted()``
returns a copy safe to render in the dashboard or logs.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .logging_utils import mask_secret

DEFAULT_PORT = 4000
CONFIG_ENV_VAR = "CACHELLM_CONFIG"

_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n", ""}


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    log_level: str = "INFO"
    request_timeout: float = 600.0
    """Total seconds allowed for one upstream call (streaming included)."""
    connect_timeout: float = 10.0
    dashboard_enabled: bool = True


@dataclass
class UpstreamConfig:
    """One OpenAI-compatible upstream provider."""

    base_url: str = ""
    api_key: str = ""
    provider: str = "openai_compatible"
    """Adapter name from ``cachellm.providers`` - keeps other providers pluggable."""
    extra_headers: dict[str, str] = field(default_factory=dict)
    forward_client_key: bool = False
    """If true, a client-supplied Authorization header is forwarded as-is
    (useful when each agent brings its own key).  Otherwise the configured
    upstream key is used and the client key is discarded."""


@dataclass
class CacheConfig:
    backend: str = "sqlite"  # memory | sqlite | redis
    sqlite_path: str = "./data/cache.db"
    redis_url: str = "redis://127.0.0.1:6379/0"
    memory_max_entries: int = 2048
    """L1 in-process LRU size; always on in front of the persistent backend."""
    default_ttl: int = 3600
    namespace: str = "default"
    enabled: bool = True
    fail_open: bool = True
    """If the cache backend errors, serve from upstream instead of failing."""
    cache_zero_temperature_only: bool = False
    """When true, only deterministic requests (temperature==0) are cached."""
    max_temperature: float = 1.0
    """Requests above this temperature are treated as non-deterministic."""
    cache_streaming: bool = True
    max_entry_bytes: int = 1_000_000


@dataclass
class SemanticConfig:
    enabled: bool = False  # opt-in, as requested
    threshold: float = 0.92
    backend: str = "hash"  # hash | sentence_transformers | openai
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: int = 256  # used by the dependency-free 'hash' backend
    max_candidates: int = 500
    """Vectors compared per lookup (scoped to the same context fingerprint)."""
    min_chars: int = 12
    max_chars: int = 8000
    embed_base_url: str = ""
    embed_api_key: str = ""
    require_single_turn: bool = True
    """Only semantic-match short/simple exchanges; long agent transcripts stay exact-only."""
    allow_tools: bool = False
    """Never semantic-match requests carrying tool schemas unless explicitly allowed."""


@dataclass
class ToolPolicy:
    """Per-tool caching policy."""

    cacheable: bool = False
    ttl: int = 0
    namespace: str | None = None
    include_context_keys: list[str] = field(default_factory=list)
    """Extra context fields (e.g. ``repo``, ``user``) folded into the tool key."""


@dataclass
class PolicyConfig:
    """Policy engine defaults.  Categories map to TTLs; tools map to policies."""

    default_category: str = "general"
    category_ttl: dict[str, int] = field(
        default_factory=lambda: {
            "static": 86400,
            "general": 3600,
            "read_only_tool": 300,
            "search": 60,
            "current_information": 0,
            "mutation": 0,
        }
    )
    semantic_categories: list[str] = field(default_factory=lambda: ["general", "static"])
    read_only_tool_prefixes: list[str] = field(
        default_factory=lambda: [
            "get", "read", "list", "search", "find", "fetch", "query", "lookup",
            "describe", "show", "view", "inspect", "count", "stat", "head", "diff",
            "grep", "browse", "check", "resolve", "explain",
        ]
    )
    mutation_tool_prefixes: list[str] = field(
        default_factory=lambda: [
            "create", "delete", "remove", "update", "put", "patch", "post", "send",
            "write", "insert", "drop", "purchase", "buy", "pay", "charge", "execute",
            "run", "exec", "deploy", "publish", "merge", "push", "reset", "kill",
            "terminate", "destroy", "revoke", "rotate", "grant", "transfer",
            "upload", "move", "rename", "cancel", "approve", "invite", "email",
            "notify", "sms", "call", "order", "refund", "shell", "sudo", "install",
        ]
    )
    dynamic_markers: list[str] = field(
        default_factory=lambda: [
            "today", "right now", "current time", "current price", "latest news",
            "breaking news", "as of now", "this morning", "this afternoon",
            "stock price", "weather right now", "what time is it",
        ]
    )
    tool_policies: dict[str, ToolPolicy] = field(default_factory=dict)
    unsafe_tool_deny_list: list[str] = field(default_factory=list)
    """Tool names (or ``prefix*`` globs) that must never be cached."""


@dataclass
class ModelPricing:
    """Price per 1M tokens; nothing is hardcoded per provider."""

    input_per_1m: float = 0.0
    output_per_1m: float = 0.0
    cached_input_per_1m: float | None = None


@dataclass
class PricingConfig:
    currency: str = "USD"
    models: dict[str, ModelPricing] = field(default_factory=dict)
    default: ModelPricing = field(default_factory=ModelPricing)
    token_estimator: str = "heuristic"  # heuristic | tiktoken
    chars_per_token: float = 4.0


@dataclass
class PrivacyConfig:
    store_prompts: bool = True
    store_responses: bool = True
    store_tool_arguments: bool = True
    store_tool_results: bool = True
    retention_days: int = 30
    """0 disables automatic pruning of request/usage history."""
    log_request_bodies: bool = False


@dataclass
class ConcurrencyConfig:
    single_flight: bool = True
    coalesce_wait: float = 600.0
    max_inflight: int = 256


@dataclass
class RoutingConfig:
    """Model routing: agent-facing name -> upstream model name."""

    model_map: dict[str, str] = field(default_factory=dict)
    allow_unlisted_models: bool = True
    default_model: str = ""


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    upstream: UpstreamConfig = field(default_factory=UpstreamConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    source_path: str | None = None

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("source_path", None)
        return data

    def redacted(self) -> dict[str, Any]:
        data = self.to_dict()
        data["upstream"]["api_key"] = mask_secret(self.upstream.api_key)
        data["semantic"]["embed_api_key"] = mask_secret(self.semantic.embed_api_key)
        data["upstream"]["extra_headers"] = {
            k: ("***" if k.lower() in {"authorization", "api-key", "x-api-key"} else v)
            for k, v in self.upstream.extra_headers.items()
        }
        return data

    def resolve_model(self, model: str) -> str:
        return self.routing.model_map.get(model, model or self.routing.default_model)

    def pricing_for(self, model: str) -> ModelPricing:
        return self.pricing.models.get(model, self.pricing.default)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _merge_dataclass(instance: Any, data: Mapping[str, Any]) -> None:
    """Recursively apply ``data`` onto a dataclass instance in place."""
    valid = {f.name: f for f in fields(instance)}
    for key, value in data.items():
        if key not in valid:
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            _merge_dataclass(current, value)
        elif isinstance(current, dict) and isinstance(value, Mapping):
            merged = dict(current)
            merged.update(value)
            setattr(instance, key, merged)
        else:
            setattr(instance, key, copy.deepcopy(value))


def _coerce_nested(cfg: Config) -> None:
    """Turn plain dicts from JSON into their dataclass counterparts."""
    cfg.pricing.models = {
        name: value if isinstance(value, ModelPricing) else ModelPricing(**_pricing_keys(value))
        for name, value in cfg.pricing.models.items()
    }
    if isinstance(cfg.pricing.default, Mapping):
        cfg.pricing.default = ModelPricing(**_pricing_keys(cfg.pricing.default))
    cfg.policy.tool_policies = {
        name: value if isinstance(value, ToolPolicy) else ToolPolicy(**_tool_keys(value))
        for name, value in cfg.policy.tool_policies.items()
    }


def _pricing_keys(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {f.name for f in fields(ModelPricing)}
    aliases = {
        "input": "input_per_1m",
        "output": "output_per_1m",
        "input_price_per_1m": "input_per_1m",
        "output_price_per_1m": "output_per_1m",
        "cached_input": "cached_input_per_1m",
    }
    out: dict[str, Any] = {}
    for key, item in value.items():
        key = aliases.get(key, key)
        if key in allowed:
            out[key] = item
    return out


def _tool_keys(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {f.name for f in fields(ToolPolicy)}
    aliases = {"ttl_seconds": "ttl", "cache": "cacheable", "enabled": "cacheable"}
    out: dict[str, Any] = {}
    for key, item in value.items():
        key = aliases.get(key, key)
        if key in allowed:
            out[key] = item
    return out


CANDIDATE_PATHS: tuple[str, ...] = (
    "./cachellm.config.json",
    "./config/cachellm.json",
    "~/.cachellm/config.json",
)


def find_config_file(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        path = Path(env_path).expanduser()
        if path.is_file():
            return path
    for candidate in CANDIDATE_PATHS:
        path = Path(candidate).expanduser()
        if path.is_file():
            return path
    return None


def load_config(
    path: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> Config:
    """Build a :class:`Config` from defaults, an optional file, and env vars."""
    env = os.environ if env is None else env
    cfg = Config()

    config_path = find_config_file(path)
    if config_path is not None:
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in config file {config_path}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ValueError(f"config file {config_path} must contain a JSON object")
        _merge_dataclass(cfg, raw)
        cfg.source_path = str(config_path)

    _apply_env(cfg, env)
    _coerce_nested(cfg)
    validate_config(cfg)
    return cfg


ENV_DOCS: tuple[tuple[str, str], ...] = (
    ("PORT", "Proxy listen port (default 4000)"),
    ("HOST", "Proxy bind address (default 127.0.0.1)"),
    ("LOG_LEVEL", "DEBUG|INFO|WARNING|ERROR"),
    ("UPSTREAM_BASE_URL", "OpenAI-compatible base URL, e.g. https://api.example.com/v1"),
    ("UPSTREAM_API_KEY", "Upstream key; never returned to clients or stored in the DB"),
    ("UPSTREAM_PROVIDER", "Provider adapter name (default openai_compatible)"),
    ("FORWARD_CLIENT_KEY", "true = forward the client's Authorization header upstream"),
    ("CACHE_BACKEND", "memory | sqlite | redis (default sqlite)"),
    ("SQLITE_PATH", "SQLite file path (default ./data/cache.db)"),
    ("REDIS_URL", "Redis URL when CACHE_BACKEND=redis"),
    ("DEFAULT_TTL", "Default cache TTL in seconds (default 3600)"),
    ("CACHE_NAMESPACE", "Namespace prefix isolating cache entries (default 'default')"),
    ("CACHE_ENABLED", "false disables all caching (pure passthrough proxy)"),
    ("CACHE_FAIL_OPEN", "true = on cache backend failure, still call upstream"),
    ("CACHE_STREAMING", "false disables caching of streamed responses"),
    ("SEMANTIC_CACHE", "true enables the semantic cache (default false)"),
    ("SEMANTIC_THRESHOLD", "Cosine similarity threshold, 0-1 (default 0.92)"),
    ("SEMANTIC_BACKEND", "hash | sentence_transformers | openai"),
    ("SEMANTIC_MODEL", "Embedding model name for the chosen backend"),
    ("EMBED_BASE_URL", "Base URL for the 'openai' embedding backend"),
    ("EMBED_API_KEY", "API key for the 'openai' embedding backend"),
    ("SINGLE_FLIGHT", "false disables request coalescing (not recommended)"),
    ("STORE_PROMPTS", "false stops persisting prompt text"),
    ("STORE_RESPONSES", "false stops persisting response text"),
    ("STORE_TOOL_ARGUMENTS", "false stops persisting tool arguments"),
    ("STORE_TOOL_RESULTS", "false stops persisting tool results"),
    ("RETENTION_DAYS", "History retention in days; 0 = keep forever"),
    ("DASHBOARD_ENABLED", "false disables the /dashboard UI"),
    ("TOKEN_ESTIMATOR", "heuristic | tiktoken"),
    ("CACHELLM_CONFIG", "Path to the JSON config file"),
)


def _apply_env(cfg: Config, env: Mapping[str, str]) -> None:
    g = env.get

    if v := g("HOST"):
        cfg.server.host = v
    if v := g("PORT"):
        cfg.server.port = _as_int(v, cfg.server.port)
    if v := g("LOG_LEVEL"):
        cfg.server.log_level = v.upper()
    if v := g("REQUEST_TIMEOUT"):
        cfg.server.request_timeout = _as_float(v, cfg.server.request_timeout)
    if (v := g("DASHBOARD_ENABLED")) is not None:
        cfg.server.dashboard_enabled = _as_bool(v, cfg.server.dashboard_enabled)

    if v := g("UPSTREAM_BASE_URL"):
        cfg.upstream.base_url = v.rstrip("/")
    if v := g("UPSTREAM_API_KEY"):
        cfg.upstream.api_key = v
    if v := g("UPSTREAM_PROVIDER"):
        cfg.upstream.provider = v
    if (v := g("FORWARD_CLIENT_KEY")) is not None:
        cfg.upstream.forward_client_key = _as_bool(v, cfg.upstream.forward_client_key)

    if v := g("CACHE_BACKEND"):
        cfg.cache.backend = v.lower()
    if v := g("SQLITE_PATH"):
        cfg.cache.sqlite_path = v
    if v := g("REDIS_URL"):
        cfg.cache.redis_url = v
    if v := g("DEFAULT_TTL"):
        cfg.cache.default_ttl = _as_int(v, cfg.cache.default_ttl)
    if v := g("CACHE_NAMESPACE"):
        cfg.cache.namespace = v
    if (v := g("CACHE_ENABLED")) is not None:
        cfg.cache.enabled = _as_bool(v, cfg.cache.enabled)
    if (v := g("CACHE_FAIL_OPEN")) is not None:
        cfg.cache.fail_open = _as_bool(v, cfg.cache.fail_open)
    if (v := g("CACHE_STREAMING")) is not None:
        cfg.cache.cache_streaming = _as_bool(v, cfg.cache.cache_streaming)
    if v := g("MEMORY_MAX_ENTRIES"):
        cfg.cache.memory_max_entries = _as_int(v, cfg.cache.memory_max_entries)

    if (v := g("SEMANTIC_CACHE")) is not None:
        cfg.semantic.enabled = _as_bool(v, cfg.semantic.enabled)
    if v := g("SEMANTIC_THRESHOLD"):
        cfg.semantic.threshold = _as_float(v, cfg.semantic.threshold)
    if v := g("SEMANTIC_BACKEND"):
        cfg.semantic.backend = v.lower()
    if v := g("SEMANTIC_MODEL"):
        cfg.semantic.model = v
    if v := g("EMBED_BASE_URL"):
        cfg.semantic.embed_base_url = v.rstrip("/")
    if v := g("EMBED_API_KEY"):
        cfg.semantic.embed_api_key = v

    if (v := g("SINGLE_FLIGHT")) is not None:
        cfg.concurrency.single_flight = _as_bool(v, cfg.concurrency.single_flight)

    if (v := g("STORE_PROMPTS")) is not None:
        cfg.privacy.store_prompts = _as_bool(v, cfg.privacy.store_prompts)
    if (v := g("STORE_RESPONSES")) is not None:
        cfg.privacy.store_responses = _as_bool(v, cfg.privacy.store_responses)
    if (v := g("STORE_TOOL_ARGUMENTS")) is not None:
        cfg.privacy.store_tool_arguments = _as_bool(v, cfg.privacy.store_tool_arguments)
    if (v := g("STORE_TOOL_RESULTS")) is not None:
        cfg.privacy.store_tool_results = _as_bool(v, cfg.privacy.store_tool_results)
    if v := g("RETENTION_DAYS"):
        cfg.privacy.retention_days = _as_int(v, cfg.privacy.retention_days)

    if v := g("TOKEN_ESTIMATOR"):
        cfg.pricing.token_estimator = v.lower()


def validate_config(cfg: Config) -> None:
    if cfg.cache.backend not in {"memory", "sqlite", "redis"}:
        raise ValueError(f"unknown CACHE_BACKEND: {cfg.cache.backend!r}")
    if not 0.0 < cfg.semantic.threshold <= 1.0:
        raise ValueError("SEMANTIC_THRESHOLD must be in (0, 1]")
    if cfg.cache.default_ttl < 0:
        raise ValueError("DEFAULT_TTL must be >= 0")
    if not 1 <= cfg.server.port <= 65535:
        raise ValueError("PORT must be between 1 and 65535")


def env_docs() -> Iterable[tuple[str, str]]:
    return ENV_DOCS
