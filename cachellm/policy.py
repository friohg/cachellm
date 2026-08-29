"""Policy engine: decides *whether* and *how long* something may be cached.

Two entry points:

``decide_request(payload, ...)``  -> :class:`CacheDecision` for an LLM call
``decide_tool(name, ...)``        -> :class:`ToolDecision` for a tool invocation

Categories
----------
static              deterministic / templated work        aggressive TTL
general             normal Q&A                            default TTL, semantic ok
read_only_tool      request whose tools are all read-only short TTL
search              web/search style work                 short TTL
current_information "today", "right now", live data       disabled (TTL 0)
mutation            create/delete/send/execute ...        never cached

Safety rules that override everything else:
* any mutation-capable tool present  -> not cacheable
* any tool on the deny list          -> not cacheable
* non-deterministic sampling above the configured limit (optional) -> not cacheable
* explicit ``cachellm_no_cache``/``Cache-Control: no-store`` -> not cacheable
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .config import PolicyConfig, ToolPolicy
from .extract import extract_query_text, extract_system_text, extract_tool_names

CATEGORY_STATIC = "static"
CATEGORY_GENERAL = "general"
CATEGORY_READ_ONLY_TOOL = "read_only_tool"
CATEGORY_SEARCH = "search"
CATEGORY_CURRENT_INFO = "current_information"
CATEGORY_MUTATION = "mutation"

ALL_CATEGORIES = (
    CATEGORY_STATIC,
    CATEGORY_GENERAL,
    CATEGORY_READ_ONLY_TOOL,
    CATEGORY_SEARCH,
    CATEGORY_CURRENT_INFO,
    CATEGORY_MUTATION,
)

SEARCH_TOOL_HINTS = ("search", "browse", "web", "google", "bing", "crawl", "scrape", "news")


@dataclass(slots=True)
class CacheDecision:
    cacheable: bool
    ttl: int
    category: str
    semantic_allowed: bool = False
    reason: str = ""

    @property
    def skipped(self) -> bool:
        return not self.cacheable


@dataclass(slots=True)
class ToolDecision:
    cacheable: bool
    ttl: int
    category: str
    reason: str = ""
    namespace: str | None = None
    include_context_keys: tuple[str, ...] = field(default_factory=tuple)


class PolicyEngine:
    """Classifies requests/tools and applies user overrides."""

    def __init__(self, policy: PolicyConfig, *, default_ttl: int = 3600) -> None:
        self.policy = policy
        self.default_ttl = default_ttl

    # -- helpers ----------------------------------------------------------
    def _ttl_for(self, category: str) -> int:
        ttl = self.policy.category_ttl.get(category)
        if ttl is None:
            ttl = self.default_ttl if category != CATEGORY_MUTATION else 0
        return max(0, int(ttl))

    @staticmethod
    def _normalize_tool_name(name: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in (name or "").lower()).strip("_")

    def _matches_prefixes(self, name: str, prefixes: Iterable[str]) -> bool:
        normalized = self._normalize_tool_name(name)
        segments = [seg for seg in normalized.split("_") if seg]
        if not segments:
            return False
        for prefix in prefixes:
            prefix = self._normalize_tool_name(prefix)
            if not prefix:
                continue
            # match a whole leading/embedded word, not a substring:
            # 'update_repo' matches 'update'; 'get_updates' does not match 'update'
            if segments[0] == prefix or prefix in segments:
                return True
            if normalized.startswith(prefix + "_"):
                return True
        return False

    def is_mutation_tool(self, name: str) -> bool:
        return self._matches_prefixes(name, self.policy.mutation_tool_prefixes)

    def is_read_only_tool(self, name: str) -> bool:
        if self.is_mutation_tool(name):
            return False
        return self._matches_prefixes(name, self.policy.read_only_tool_prefixes)

    def is_denied_tool(self, name: str) -> bool:
        normalized = (name or "").lower()
        for pattern in self.policy.unsafe_tool_deny_list:
            pattern = pattern.lower()
            if normalized == pattern or fnmatch.fnmatch(normalized, pattern):
                return True
        return False

    def tool_policy(self, name: str) -> ToolPolicy | None:
        policies = self.policy.tool_policies
        if name in policies:
            return policies[name]
        for pattern, policy in policies.items():
            if any(ch in pattern for ch in "*?[") and fnmatch.fnmatch(name, pattern):
                return policy
        return None

    def classify_tool(self, name: str) -> str:
        if self.is_denied_tool(name) or self.is_mutation_tool(name):
            return CATEGORY_MUTATION
        lowered = (name or "").lower()
        if any(hint in lowered for hint in SEARCH_TOOL_HINTS):
            return CATEGORY_SEARCH
        if self.is_read_only_tool(name):
            return CATEGORY_READ_ONLY_TOOL
        return CATEGORY_MUTATION  # unknown tools are treated as unsafe

    # -- request classification ------------------------------------------
    def classify_request(self, payload: Mapping[str, Any]) -> str:
        data = dict(payload)
        tool_names = extract_tool_names(data)
        if tool_names:
            if any(
                self.is_denied_tool(name) or self.is_mutation_tool(name) for name in tool_names
            ):
                return CATEGORY_MUTATION
            if any(hint in name.lower() for name in tool_names for hint in SEARCH_TOOL_HINTS):
                return CATEGORY_SEARCH
            if all(self.is_read_only_tool(name) for name in tool_names):
                return CATEGORY_READ_ONLY_TOOL
            return CATEGORY_MUTATION

        text = f"{extract_system_text(data)}\n{extract_query_text(data)}".lower()
        if any(marker in text for marker in self.policy.dynamic_markers):
            return CATEGORY_CURRENT_INFO

        temperature = data.get("temperature")
        seeded = data.get("seed") is not None
        if (temperature is not None and float(temperature) == 0.0) or seeded:
            return CATEGORY_STATIC

        return self.policy.default_category or CATEGORY_GENERAL

    def decide_request(
        self,
        payload: Mapping[str, Any],
        *,
        semantic_enabled: bool = False,
        max_temperature: float = 1.0,
        zero_temperature_only: bool = False,
        explicit_ttl: int | None = None,
        no_store: bool = False,
    ) -> CacheDecision:
        data = dict(payload)

        if no_store or data.get("cachellm_no_cache"):
            return CacheDecision(False, 0, CATEGORY_GENERAL, reason="no_store_requested")

        category = self.classify_request(data)

        if category == CATEGORY_MUTATION:
            return CacheDecision(False, 0, category, reason="mutation_or_unsafe_tool")

        # n>1 samples multiple completions; replaying one cached set is fine, but
        # streaming logprobs/audio replay is not, so refuse those.
        if data.get("logprobs") and data.get("stream"):
            return CacheDecision(False, 0, category, reason="stream_logprobs_unsupported")
        if data.get("modalities") and "audio" in (data.get("modalities") or []):
            return CacheDecision(False, 0, category, reason="audio_modality")

        temperature = data.get("temperature")
        temperature_value = float(temperature) if temperature is not None else 1.0
        if zero_temperature_only and temperature_value != 0.0 and data.get("seed") is None:
            return CacheDecision(False, 0, category, reason="non_deterministic_temperature")
        if temperature_value > max_temperature and data.get("seed") is None:
            return CacheDecision(False, 0, category, reason="temperature_above_limit")

        ttl = self._ttl_for(category) if explicit_ttl is None else max(0, int(explicit_ttl))
        if ttl == 0:
            return CacheDecision(False, 0, category, reason=f"ttl_zero_for_{category}")

        semantic_allowed = (
            semantic_enabled
            and category in set(self.policy.semantic_categories)
            and not extract_tool_names(data)
        )
        return CacheDecision(True, ttl, category, semantic_allowed, reason="allowed")

    def decide_tool(
        self,
        tool_name: str,
        *,
        explicit_ttl: int | None = None,
    ) -> ToolDecision:
        if self.is_denied_tool(tool_name):
            return ToolDecision(False, 0, CATEGORY_MUTATION, reason="deny_list")

        override = self.tool_policy(tool_name)
        if override is not None:
            if not override.cacheable:
                return ToolDecision(
                    False, 0, self.classify_tool(tool_name), reason="policy_disabled"
                )
            if self.is_mutation_tool(tool_name) and override.ttl > 0:
                # An explicit opt-in wins, but we record why it was risky.
                reason = "explicit_override_on_mutation_tool"
            else:
                reason = "explicit_policy"
            ttl = int(explicit_ttl if explicit_ttl is not None else override.ttl)
            if ttl <= 0:
                return ToolDecision(False, 0, self.classify_tool(tool_name), reason="ttl_zero")
            return ToolDecision(
                True,
                ttl,
                self.classify_tool(tool_name),
                reason=reason,
                namespace=override.namespace,
                include_context_keys=tuple(override.include_context_keys),
            )

        category = self.classify_tool(tool_name)
        if category == CATEGORY_MUTATION:
            return ToolDecision(False, 0, category, reason="unsafe_or_unknown_tool")
        ttl = self._ttl_for(category) if explicit_ttl is None else max(0, int(explicit_ttl))
        if ttl <= 0:
            return ToolDecision(False, 0, category, reason="ttl_zero")
        return ToolDecision(True, ttl, category, reason="category_default")

    # -- introspection for the dashboard ---------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "default_category": self.policy.default_category,
            "category_ttl": dict(self.policy.category_ttl),
            "semantic_categories": list(self.policy.semantic_categories),
            "read_only_tool_prefixes": list(self.policy.read_only_tool_prefixes),
            "mutation_tool_prefixes": list(self.policy.mutation_tool_prefixes),
            "dynamic_markers": list(self.policy.dynamic_markers),
            "unsafe_tool_deny_list": list(self.policy.unsafe_tool_deny_list),
            "tool_policies": {
                name: {
                    "cacheable": p.cacheable,
                    "ttl": p.ttl,
                    "namespace": p.namespace,
                    "include_context_keys": list(p.include_context_keys),
                }
                for name, p in self.policy.tool_policies.items()
            },
        }
