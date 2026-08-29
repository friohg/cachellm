"""Provider adapter interface.

Adapters translate a normalized CacheLLM request into an upstream call and
return either a full JSON response or an async iterator of raw SSE lines.  The
proxy layer never imports httpx directly, so a new provider only has to satisfy
this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping, Protocol, runtime_checkable


class ProviderError(RuntimeError):
    """Upstream failure.  Carries a status code and a safe client-facing body."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        upstream_body: Any = None,
        error_type: str = "upstream_error",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.upstream_body = upstream_body
        self.error_type = error_type

    def to_client_payload(self) -> dict[str, Any]:
        """Client-safe error body.  Never includes credentials or raw headers."""
        detail: Any = self.upstream_body
        if isinstance(detail, Mapping) and "error" in detail:
            inner = detail["error"]
            message = (
                inner.get("message")
                if isinstance(inner, Mapping)
                else str(inner)
            ) or str(self)
            error_type = (
                inner.get("type") if isinstance(inner, Mapping) else None
            ) or self.error_type
            return {
                "error": {
                    "message": message,
                    "type": error_type,
                    "code": (inner.get("code") if isinstance(inner, Mapping) else None),
                    "source": "cachellm.upstream",
                }
            }
        return {
            "error": {
                "message": str(self),
                "type": self.error_type,
                "source": "cachellm.upstream",
            }
        }


@dataclass(slots=True)
class ProviderResponse:
    """A non-streaming upstream response."""

    status_code: int
    payload: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


@dataclass(slots=True)
class StreamChunk:
    """One raw SSE ``data:`` payload from the upstream stream."""

    data: str
    """Raw text after ``data: `` - either JSON or the literal ``[DONE]``."""

    @property
    def is_done(self) -> bool:
        return self.data.strip() == "[DONE]"


@runtime_checkable
class ProviderAdapter(Protocol):
    name: str
    base_url: str

    async def chat_completion(
        self, payload: Mapping[str, Any], *, headers: Mapping[str, str] | None = None
    ) -> ProviderResponse: ...

    def stream_chat_completion(
        self, payload: Mapping[str, Any], *, headers: Mapping[str, str] | None = None
    ) -> AsyncIterator[StreamChunk]: ...

    async def post_json(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> ProviderResponse: ...

    def stream_json(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[StreamChunk]: ...

    async def list_models(self) -> ProviderResponse: ...

    async def health(self) -> dict[str, Any]: ...

    async def close(self) -> None: ...
