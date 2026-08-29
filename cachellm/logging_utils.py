"""Structured logging with hard redaction of secrets.

Every log line is emitted as ``EVENT key=value ...`` which is greppable in a
terminal and trivially parsable.  Authorization headers, API keys and anything
that looks like a bearer token are redacted before formatting - the logger is
the only place allowed to touch header dictionaries, so keys cannot leak by
accident from another module.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any, Iterable, Mapping

REDACTED = "***REDACTED***"

# Header names that must never be printed.
_SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "api-key",
        "x-api-key",
        "x-goog-api-key",
        "openai-api-key",
        "anthropic-api-key",
        "cookie",
        "set-cookie",
    }
)

# Config/JSON keys that must never be printed.
_SENSITIVE_KEYS = re.compile(
    r"(api[_-]?key|secret|password|passwd|token|authorization|credential)", re.I
)

# Inline secret shapes (bearer tokens, sk-..., long hex blobs).
_INLINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9._\-]{8,}"),
    re.compile(r"\b(?:gsk|xai|hf|ghp|gho|glpat)[-_][A-Za-z0-9._\-]{12,}"),
)


def redact_text(value: str) -> str:
    """Redact inline secret shapes from a free-form string."""
    out = value
    for pattern in _INLINE_PATTERNS:
        out = pattern.sub(REDACTED, out)
    return out


def redact_headers(headers: Mapping[str, str] | Iterable[tuple[str, str]]) -> dict[str, str]:
    """Return a copy of ``headers`` with all sensitive values replaced."""
    items = headers.items() if isinstance(headers, Mapping) else headers
    safe: dict[str, str] = {}
    for key, value in items:
        if key.lower() in _SENSITIVE_HEADERS or _SENSITIVE_KEYS.search(key):
            safe[key] = REDACTED
        else:
            safe[key] = redact_text(str(value))
    return safe


def redact_value(value: Any) -> Any:
    """Recursively redact secrets from arbitrary JSON-ish data."""
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and (
                key.lower() in _SENSITIVE_HEADERS or _SENSITIVE_KEYS.search(key)
            ):
                out[key] = REDACTED
            else:
                out[key] = redact_value(item)
        return out
    if isinstance(value, (list, tuple)):
        return [redact_value(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def mask_secret(value: str | None, keep: int = 4) -> str:
    """Mask a secret for display: ``sk-abcd...wxyz`` -> ``sk-a...wxyz``."""
    if not value:
        return ""
    if len(value) <= keep * 2:
        return REDACTED
    return f"{value[:keep]}...{value[-keep:]}"


class _KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        return redact_text(base)


_CONFIGURED = False


def configure_logging(level: str = "INFO", stream: Any = None) -> None:
    """Install the CacheLLM log handler once per process."""
    global _CONFIGURED
    logger = logging.getLogger("cachellm")
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(
        _KeyValueFormatter(
            fmt="%(asctime)s %(levelname)-5s %(name)s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED = True


def get_logger(name: str = "cachellm") -> logging.Logger:
    return logging.getLogger(name)


def _fmt_fields(fields: Mapping[str, Any]) -> str:
    parts = []
    for key, value in fields.items():
        if value is None:
            continue
        safe = redact_value(value)
        if isinstance(safe, float):
            safe = f"{safe:.3f}"
        text = str(safe)
        if any(ch in text for ch in " \t\""):
            text = '"' + text.replace('"', "'") + '"'
        parts.append(f"{key}={text}")
    return " ".join(parts)


def log_event(
    event: str,
    *,
    level: int = logging.INFO,
    logger: logging.Logger | None = None,
    **fields: Any,
) -> None:
    """Emit a structured event line, e.g. ``CACHE HIT rid=.. key=.. ttl=..``."""
    log = logger or get_logger()
    log.log(level, "%s %s", event, _fmt_fields(fields))


# Canonical event names used across the codebase (also documented in README).
EVENT_CACHE_HIT = "CACHE HIT"
EVENT_CACHE_MISS = "CACHE MISS"
EVENT_SEMANTIC_HIT = "SEMANTIC HIT"
EVENT_TOOL_CACHE_HIT = "TOOL CACHE HIT"
EVENT_TOOL_CACHE_MISS = "TOOL CACHE MISS"
EVENT_TOOL_CACHE_WRITE = "TOOL CACHE WRITE"
EVENT_UPSTREAM_REQUEST = "UPSTREAM REQUEST"
EVENT_UPSTREAM_ERROR = "UPSTREAM ERROR"
EVENT_CACHE_WRITE = "CACHE WRITE"
EVENT_CACHE_SKIP = "CACHE SKIP"
EVENT_CACHE_ERROR = "CACHE ERROR"
EVENT_CACHE_INVALIDATE = "CACHE INVALIDATE"
EVENT_COALESCED = "COALESCED"
EVENT_STREAM_ABORT = "STREAM ABORT"
