"""Adapter for any OpenAI-compatible HTTP API.

Handles auth injection, timeouts, error translation and SSE streaming.  The
upstream API key lives only in this object's headers; it is never echoed into a
client response, log line or database row.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, Mapping

import httpx

from ..config import UpstreamConfig
from ..logging_utils import get_logger
from .base import ProviderError, ProviderResponse, StreamChunk

log = get_logger("cachellm.provider")

# Client headers that must never be forwarded upstream verbatim.
_HOP_BY_HOP = frozenset(
    {
        "host", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
        "te", "trailer", "transfer-encoding", "upgrade", "content-length",
        "accept-encoding", "authorization", "api-key", "x-api-key", "cookie",
    }
)

# Client headers that are useful to pass through.
_FORWARDABLE_PREFIXES = ("x-request-id", "x-title", "http-referer", "openai-beta", "user-agent")


class OpenAICompatibleProvider:
    """Talks to ``{base_url}/chat/completions`` and friends."""

    name = "openai_compatible"

    def __init__(
        self,
        cfg: UpstreamConfig,
        *,
        request_timeout: float = 600.0,
        connect_timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not cfg.base_url and client is None:
            raise ProviderError(
                "UPSTREAM_BASE_URL is not configured; set it in the environment or config file",
                status_code=500,
                error_type="configuration_error",
            )
        self.cfg = cfg
        self.base_url = (cfg.base_url or "").rstrip("/")
        self._own_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout, connect=connect_timeout),
            follow_redirects=True,
        )

    # -- helpers ----------------------------------------------------------
    def _url(self, path: str) -> str:
        path = path if path.startswith("/") else f"/{path}"
        return f"{self.base_url}{path}"

    def _headers(self, client_headers: Mapping[str, str] | None) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        for key, value in (client_headers or {}).items():
            lowered = key.lower()
            if lowered in _HOP_BY_HOP:
                continue
            if lowered.startswith("x-cachellm"):
                continue
            if lowered.startswith(_FORWARDABLE_PREFIXES) or lowered.startswith("x-stainless"):
                headers[key] = value

        auth = None
        if self.cfg.forward_client_key:
            for key, value in (client_headers or {}).items():
                if key.lower() == "authorization" and value:
                    auth = value
                    break
        if auth is None and self.cfg.api_key:
            auth = f"Bearer {self.cfg.api_key}"
        if auth:
            headers["Authorization"] = auth
        headers.update(self.cfg.extra_headers or {})
        return headers

    @staticmethod
    def _safe_body(response: httpx.Response) -> Any:
        try:
            return response.json()
        except (json.JSONDecodeError, ValueError):
            text = response.text
            return {"error": {"message": text[:2000] or response.reason_phrase}}

    def _raise_for_status(self, response: httpx.Response) -> None:
        if 200 <= response.status_code < 300:
            return
        body = self._safe_body(response)
        raise ProviderError(
            f"upstream returned HTTP {response.status_code}",
            status_code=response.status_code,
            upstream_body=body,
            error_type="upstream_http_error",
        )

    # -- non-streaming ----------------------------------------------------
    async def post_json(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> ProviderResponse:
        started = time.perf_counter()
        try:
            response = await self._client.post(
                self._url(path), json=dict(payload), headers=self._headers(headers)
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"upstream timed out: {type(exc).__name__}",
                status_code=504,
                error_type="upstream_timeout",
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"upstream connection failed: {type(exc).__name__}: {exc}",
                status_code=502,
                error_type="upstream_unavailable",
            ) from exc
        self._raise_for_status(response)
        body = self._safe_body(response)
        if not isinstance(body, dict):
            raise ProviderError(
                "upstream returned a non-object JSON body",
                status_code=502,
                error_type="upstream_bad_response",
            )
        return ProviderResponse(
            status_code=response.status_code,
            payload=body,
            headers={
                k: v
                for k, v in response.headers.items()
                if k.lower() in {"content-type", "x-request-id", "openai-version"}
            },
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def chat_completion(
        self, payload: Mapping[str, Any], *, headers: Mapping[str, str] | None = None
    ) -> ProviderResponse:
        return await self.post_json("/chat/completions", payload, headers=headers)

    async def responses(
        self, payload: Mapping[str, Any], *, headers: Mapping[str, str] | None = None
    ) -> ProviderResponse:
        return await self.post_json("/responses", payload, headers=headers)

    # -- streaming --------------------------------------------------------
    async def stream_json(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        request_headers = self._headers(headers)
        request_headers["Accept"] = "text/event-stream"
        body = dict(payload)
        body["stream"] = True
        try:
            async with self._client.stream(
                "POST", self._url(path), json=body, headers=request_headers
            ) as response:
                if response.status_code >= 300:
                    await response.aread()
                    self._raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith(":"):  # SSE comment / keep-alive
                        continue
                    if line.startswith("data:"):
                        yield StreamChunk(line[5:].lstrip())
                    # Other SSE fields (event:, id:) are not used by OpenAI-compatible APIs.
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"upstream stream timed out: {type(exc).__name__}",
                status_code=504,
                error_type="upstream_timeout",
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"upstream stream failed: {type(exc).__name__}: {exc}",
                status_code=502,
                error_type="upstream_unavailable",
            ) from exc

    def stream_chat_completion(
        self, payload: Mapping[str, Any], *, headers: Mapping[str, str] | None = None
    ) -> AsyncIterator[StreamChunk]:
        return self.stream_json("/chat/completions", payload, headers=headers)

    # -- misc -------------------------------------------------------------
    async def list_models(self) -> ProviderResponse:
        started = time.perf_counter()
        try:
            response = await self._client.get(
                self._url("/models"), headers=self._headers(None)
            )
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"upstream connection failed: {exc}",
                status_code=502,
                error_type="upstream_unavailable",
            ) from exc
        self._raise_for_status(response)
        body = self._safe_body(response)
        return ProviderResponse(
            status_code=response.status_code,
            payload=body if isinstance(body, dict) else {"data": body},
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def health(self) -> dict[str, Any]:
        if not self.base_url:
            return {"ok": False, "error": "UPSTREAM_BASE_URL not configured"}
        try:
            result = await self.list_models()
            count = len(result.payload.get("data") or [])
            return {
                "ok": True,
                "base_url": self.base_url,
                "models": count,
                "latency_ms": round(result.latency_ms, 2),
            }
        except ProviderError as exc:
            return {"ok": False, "base_url": self.base_url, "error": str(exc)}

    async def close(self) -> None:
        if self._own_client:
            await self._client.aclose()
