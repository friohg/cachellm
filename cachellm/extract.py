"""Content extraction from OpenAI-style payloads.

Lives at the top level (not inside ``cachellm.cache``) so both the cache key
builder and the policy engine can use it without an import cycle.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from .normalize import canonical_json

# Roles whose content is instruction context rather than the current query.
CONTEXT_ROLES = ("system", "developer")


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def messages_of(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a message list for both /chat/completions and /responses shapes."""
    messages = payload.get("messages")
    if isinstance(messages, list):
        return [m for m in messages if isinstance(m, dict)]
    raw_input = payload.get("input")
    if isinstance(raw_input, list):
        return [m for m in raw_input if isinstance(m, dict)]
    if isinstance(raw_input, str):
        return [{"role": "user", "content": raw_input}]
    if isinstance(payload.get("prompt"), str):
        return [{"role": "user", "content": payload["prompt"]}]
    return []


def message_text(message: Any) -> str:
    """Flatten OpenAI content (string, or list of typed parts) into text."""
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if content is None:
        content = message.get("text")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif part.get("type") in {"image_url", "input_image"}:
                    url = part.get("image_url")
                    if isinstance(url, dict):
                        url = url.get("url")
                    parts.append(f"[image:{sha256_hex(str(url))[:16]}]")
        return "\n".join(parts)
    return ""


def extract_system_text(payload: dict[str, Any]) -> str:
    """Concatenate every system/developer instruction, including top-level ones."""
    chunks: list[str] = []
    for field_name in ("instructions", "system"):
        value = payload.get(field_name)
        if isinstance(value, str) and value.strip():
            chunks.append(value.strip())
    for message in messages_of(payload):
        if message.get("role") in CONTEXT_ROLES:
            text = message_text(message).strip()
            if text:
                chunks.append(text)
    return "\n\n".join(chunks)


def extract_query_text(payload: dict[str, Any]) -> str:
    """The final user turn - the semantic 'question' being asked."""
    for message in reversed(messages_of(payload)):
        if message.get("role") == "user":
            return message_text(message).strip()
    return ""


def extract_tool_names(payload: dict[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.append(function["name"])
        elif isinstance(tool.get("name"), str):
            names.append(tool["name"])
    for function in payload.get("functions") or []:
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.append(function["name"])
    return tuple(sorted(set(names)))


def history_fingerprint(messages: Iterable[dict[str, Any]]) -> str:
    """Hash of the conversation *excluding* the trailing user turn.

    Included in the semantic context so a paraphrase can only match within the
    same prior conversation, never across sessions.
    """
    items = list(messages)
    while items and items[-1].get("role") == "user":
        items.pop()
    payload = [
        {
            "role": m.get("role"),
            "content": message_text(m),
            "name": m.get("name"),
            "tool_call_id": m.get("tool_call_id"),
            "tool_calls": m.get("tool_calls"),
        }
        for m in items
        if m.get("role") not in CONTEXT_ROLES
    ]
    return sha256_hex(canonical_json(payload))


def excerpt(text: str, limit: int = 240) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"
