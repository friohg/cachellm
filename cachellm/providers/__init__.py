"""Provider adapters.

A provider adapter knows how to talk to one upstream API shape.  The registry
keeps CacheLLM open to new providers without touching the proxy layer: register
a class and select it with ``UPSTREAM_PROVIDER=<name>``.
"""

from __future__ import annotations

from typing import Callable

from ..config import UpstreamConfig
from .base import (
    ProviderAdapter,
    ProviderError,
    ProviderResponse,
    StreamChunk,
)
from .openai_compatible import OpenAICompatibleProvider

_REGISTRY: dict[str, Callable[[UpstreamConfig, float, float], ProviderAdapter]] = {}


def register_provider(
    name: str, factory: Callable[[UpstreamConfig, float, float], ProviderAdapter]
) -> None:
    _REGISTRY[name.lower()] = factory


def available_providers() -> list[str]:
    return sorted(_REGISTRY)


def build_provider(
    cfg: UpstreamConfig, *, request_timeout: float = 600.0, connect_timeout: float = 10.0
) -> ProviderAdapter:
    name = (cfg.provider or "openai_compatible").lower()
    factory = _REGISTRY.get(name)
    if factory is None:
        raise ProviderError(
            f"unknown upstream provider {name!r}; available: {', '.join(available_providers())}"
        )
    return factory(cfg, request_timeout, connect_timeout)


register_provider(
    "openai_compatible",
    lambda cfg, rt, ct: OpenAICompatibleProvider(
        cfg, request_timeout=rt, connect_timeout=ct
    ),
)
# Convenience aliases - all speak the same wire format.
for _alias in ("openai", "azure_openai", "openrouter", "together", "groq", "vllm",
               "ollama", "lmstudio", "llamacpp", "deepseek", "mistral", "xai",
               "fireworks", "anyscale", "local"):
    register_provider(
        _alias,
        lambda cfg, rt, ct: OpenAICompatibleProvider(
            cfg, request_timeout=rt, connect_timeout=ct
        ),
    )

__all__ = [
    "ProviderAdapter",
    "ProviderError",
    "ProviderResponse",
    "StreamChunk",
    "OpenAICompatibleProvider",
    "build_provider",
    "register_provider",
    "available_providers",
]
