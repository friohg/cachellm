"""Cache key construction.

The key is a SHA-256 over canonical JSON of everything that can materially
change the answer:

    namespace | endpoint | resolved model | messages/input | system+developer
    instructions | tools + tool schemas | tool_choice | response_format |
    generation parameters (temperature, top_p, seed, penalties, ...)

Two requests sharing only the last user message but differing in system prompt,
tools, model or sampling settings therefore get *different* keys - that is the
core correctness property.  A separate "context fingerprint" (everything except
the final user turn) is used to scope semantic lookups so a semantically similar
question can only match inside an identical context.

Content-extraction helpers live in ``cachellm.extract`` and are re-exported here
for convenience.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..extract import (
    CONTEXT_ROLES,
    excerpt,
    extract_query_text,
    extract_system_text,
    extract_tool_names,
    history_fingerprint,
    message_text,
    messages_of,
    sha256_hex,
)
from ..normalize import Normalizer, canonical_json, default_normalizer

KEY_VERSION = "v1"

__all__ = [
    "KEY_VERSION",
    "CONTEXT_ROLES",
    "GENERATION_PARAMS",
    "KeyBuilder",
    "KeyMaterial",
    "excerpt",
    "extract_query_text",
    "extract_system_text",
    "extract_tool_names",
    "message_text",
    "sha256_hex",
]


@dataclass(slots=True)
class KeyMaterial:
    """Everything derived from a request that the cache layers need."""

    key: str
    context_hash: str
    canonical_request: str
    normalized: dict[str, Any]
    model: str
    endpoint: str
    namespace: str
    system_text: str
    query_text: str
    tool_names: tuple[str, ...]
    message_count: int


GENERATION_PARAMS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "n",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "reasoning_effort",
)


class KeyBuilder:
    """Builds cache keys and context fingerprints from OpenAI-style payloads."""

    def __init__(self, normalizer: Normalizer | None = None) -> None:
        self.normalizer = normalizer or default_normalizer()

    def build(
        self,
        payload: dict[str, Any],
        *,
        endpoint: str,
        namespace: str = "default",
        resolved_model: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> KeyMaterial:
        normalized = self.normalizer.normalize(payload)
        model = str(resolved_model or normalized.get("model") or "")
        normalized_for_key = dict(normalized)
        normalized_for_key["model"] = model

        key_document: dict[str, Any] = {
            "v": KEY_VERSION,
            "namespace": namespace,
            "endpoint": endpoint,
            "request": normalized_for_key,
        }
        if extra:
            key_document["extra"] = extra
        canonical_request = canonical_json(key_document)
        key = sha256_hex(canonical_request)

        messages = messages_of(normalized)
        system_text = extract_system_text(normalized)
        query_text = extract_query_text(normalized)
        tool_names = extract_tool_names(normalized)

        context_document = {
            "v": KEY_VERSION,
            "namespace": namespace,
            "endpoint": endpoint,
            "model": model,
            "system": system_text,
            "tools": list(tool_names),
            "tool_schemas": canonical_json(normalized.get("tools") or []),
            "tool_choice": normalized.get("tool_choice"),
            "response_format": normalized.get("response_format") or normalized.get("text"),
            "params": {k: normalized.get(k) for k in GENERATION_PARAMS if k in normalized},
            "history": history_fingerprint(messages),
        }
        if extra:
            context_document["extra"] = extra
        context_hash = sha256_hex(canonical_json(context_document))

        return KeyMaterial(
            key=key,
            context_hash=context_hash,
            canonical_request=canonical_request,
            normalized=normalized,
            model=model,
            endpoint=endpoint,
            namespace=namespace,
            system_text=system_text,
            query_text=query_text,
            tool_names=tool_names,
            message_count=len(messages),
        )
