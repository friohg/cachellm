"""Cache subsystem: keys, backends, exact/semantic/tool caches, engine."""

from .backends import (
    CacheBackend,
    CacheBackendError,
    CachedEntry,
    MemoryBackend,
    RedisBackend,
    SQLiteBackend,
    build_backend,
)
from .engine import (
    SOURCE_BYPASS,
    SOURCE_COALESCED,
    SOURCE_EXACT,
    SOURCE_SEMANTIC,
    SOURCE_UPSTREAM,
    CacheEngine,
    CacheOutcome,
)
from .exact import ExactCache, LookupResult
from .keys import KeyBuilder, KeyMaterial, excerpt, sha256_hex
from .semantic import SemanticCache, SemanticMatch
from .tools import ToolCache, ToolLookup, tool_cache_key

__all__ = [
    "CacheBackend",
    "CacheBackendError",
    "CachedEntry",
    "MemoryBackend",
    "SQLiteBackend",
    "RedisBackend",
    "build_backend",
    "CacheEngine",
    "CacheOutcome",
    "SOURCE_EXACT",
    "SOURCE_SEMANTIC",
    "SOURCE_UPSTREAM",
    "SOURCE_COALESCED",
    "SOURCE_BYPASS",
    "ExactCache",
    "LookupResult",
    "KeyBuilder",
    "KeyMaterial",
    "sha256_hex",
    "excerpt",
    "SemanticCache",
    "SemanticMatch",
    "ToolCache",
    "ToolLookup",
    "tool_cache_key",
]
