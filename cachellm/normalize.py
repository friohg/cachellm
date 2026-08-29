"""Request normalization + canonical JSON serialization.

Normalization is deliberately conservative: it only removes differences that
cannot change the model's output (key ordering, JSON formatting, trailing
whitespace).  Anything that *could* change semantics - casing, punctuation,
interior whitespace of prose, message ordering - is preserved.

Every step is a small pure function so callers can compose their own pipeline
(``Normalizer(steps=[...])``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

# ---------------------------------------------------------------------------
# canonical JSON
# ---------------------------------------------------------------------------


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, UTF-8 safe.

    Two payloads that differ only in key order or formatting produce byte
    identical output, which is what makes cache keys stable.
    """
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, float) and value.is_integer():
        # 1.0 and 1 mean the same thing to every provider we target.
        return int(value)
    return value


def parse_json_object(text: str) -> Any:
    """Best-effort JSON parse used for tool arguments (which arrive as strings)."""
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# individual normalization steps
# ---------------------------------------------------------------------------


def strip_edge_whitespace(payload: dict[str, Any]) -> dict[str, Any]:
    """Trim leading/trailing whitespace of message text (safe, never interior)."""
    messages = payload.get("messages")
    if isinstance(messages, list):
        payload["messages"] = [_strip_message(m) for m in messages]
    return payload


def _strip_message(message: Any) -> Any:
    if not isinstance(message, dict):
        return message
    out = dict(message)
    content = out.get("content")
    if isinstance(content, str):
        out["content"] = content.strip()
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                part = {**part, "text": part["text"].strip()}
            parts.append(part)
        out["content"] = parts
    return out


def normalize_tool_arguments(payload: dict[str, Any]) -> dict[str, Any]:
    """Re-serialize assistant tool-call arguments canonically.

    Providers emit ``{"b":2,"a":1}`` and ``{"a": 1, "b": 2}`` interchangeably;
    both describe the same call, so they must hash identically.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload
    out_messages = []
    for message in messages:
        if isinstance(message, dict) and isinstance(message.get("tool_calls"), list):
            message = dict(message)
            message["tool_calls"] = [_normalize_tool_call(tc) for tc in message["tool_calls"]]
        out_messages.append(message)
    payload["messages"] = out_messages
    return payload


def _normalize_tool_call(tool_call: Any) -> Any:
    if not isinstance(tool_call, dict):
        return tool_call
    out = dict(tool_call)
    function = out.get("function")
    if isinstance(function, dict) and isinstance(function.get("arguments"), str):
        parsed = parse_json_object(function["arguments"])
        if parsed is not None:
            function = dict(function)
            function["arguments"] = canonical_json(parsed)
            out["function"] = function
    return out


def drop_volatile_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove fields that never change the generated content."""
    for key in VOLATILE_FIELDS:
        payload.pop(key, None)
    return payload


VOLATILE_FIELDS: frozenset[str] = frozenset(
    {
        "stream",  # handled separately: a hit can be replayed as SSE
        "stream_options",
        "user",  # identity, not content (namespaces handle isolation)
        "metadata",
        "store",
        "cachellm_namespace",
        "cachellm_ttl",
        "cachellm_no_cache",
    }
)

# Fields that materially affect generation and therefore belong in the key.
SIGNIFICANT_FIELDS: tuple[str, ...] = (
    "model",
    "messages",
    "input",
    "instructions",
    "prompt",
    "system",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "functions",
    "function_call",
    "response_format",
    "text",
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
    "reasoning",
    "reasoning_effort",
    "modalities",
    "audio",
    "prediction",
    "truncation",
    "service_tier",
)


def project_significant_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields that can influence the response.

    Unknown/extra fields are kept too (conservative: an unrecognised knob might
    matter), except the explicit volatile list above.
    """
    return {k: v for k, v in payload.items() if k not in VOLATILE_FIELDS}


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

Step = Callable[[dict[str, Any]], dict[str, Any]]

DEFAULT_STEPS: tuple[Step, ...] = (
    drop_volatile_fields,
    strip_edge_whitespace,
    normalize_tool_arguments,
    project_significant_fields,
)


@dataclass(slots=True)
class Normalizer:
    """Composable normalization pipeline."""

    steps: Sequence[Step] = field(default=DEFAULT_STEPS)

    def normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        out = json.loads(json.dumps(payload))  # deep copy, JSON-safe
        if not isinstance(out, dict):  # pragma: no cover - defensive
            raise TypeError("request payload must be a JSON object")
        for step in self.steps:
            out = step(out)
        return out

    def canonical(self, payload: dict[str, Any]) -> str:
        return canonical_json(self.normalize(payload))


def default_normalizer(extra_steps: Iterable[Step] = ()) -> Normalizer:
    return Normalizer(steps=tuple(DEFAULT_STEPS) + tuple(extra_steps))
