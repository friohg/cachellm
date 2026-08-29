"""OpenAI-compatible proxy routes.

Endpoints
---------
POST /v1/chat/completions   full cache pipeline (exact -> semantic -> upstream)
POST /v1/responses          same pipeline, /v1/responses shape
GET  /v1/models             passthrough
POST /v1/embeddings         passthrough (no caching by default)
POST /v1/tools/cache/*      tool-result cache API

Streaming: a hit is replayed as SSE; a miss streams upstream through to the
client while collecting chunks, and only a cleanly completed stream is cached.
"""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..cache.backends import CacheBackendError
from ..cache.engine import (
    SOURCE_BYPASS,
    SOURCE_COALESCED,
    SOURCE_EXACT,
    SOURCE_SEMANTIC,
)
from ..cache.keys import KeyMaterial
from ..extract import excerpt
from ..logging_utils import (
    EVENT_STREAM_ABORT,
    EVENT_UPSTREAM_ERROR,
    EVENT_UPSTREAM_REQUEST,
    get_logger,
    log_event,
)
from ..providers import ProviderError
from ..stats import (
    OUTCOME_BYPASS,
    OUTCOME_COALESCED,
    OUTCOME_ERROR,
    OUTCOME_EXACT_HIT,
    OUTCOME_MISS,
    OUTCOME_SEMANTIC_HIT,
    OUTCOME_TOOL_HIT,
    RequestRecord,
)
from ..streaming import StreamCollector, chunks_from_response, error_sse
from .deps import get_state

router = APIRouter()
log = get_logger("cachellm.proxy")

SSE_HEADERS = {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

_OUTCOME_BY_SOURCE = {
    SOURCE_EXACT: OUTCOME_EXACT_HIT,
    SOURCE_SEMANTIC: OUTCOME_SEMANTIC_HIT,
    SOURCE_COALESCED: OUTCOME_COALESCED,
    SOURCE_BYPASS: OUTCOME_BYPASS,
}


def _client_headers(request: Request) -> dict[str, str]:
    return {k: v for k, v in request.headers.items()}


def _no_store_requested(request: Request, payload: Mapping[str, Any]) -> bool:
    cache_control = (request.headers.get("cache-control") or "").lower()
    if "no-store" in cache_control:
        return True
    if (request.headers.get("x-cachellm-cache") or "").lower() in {"off", "no", "false"}:
        return True
    return bool(payload.get("cachellm_no_cache"))


def _namespace(request: Request, payload: Mapping[str, Any]) -> str | None:
    return (
        request.headers.get("x-cachellm-namespace")
        or payload.get("cachellm_namespace")
        or None
    )


def _explicit_ttl(request: Request) -> int | None:
    raw = request.headers.get("x-cachellm-ttl")
    if not raw:
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        return None


def _hit_headers(outcome: str, *, key: str, similarity: float | None, age: float | None) -> dict[str, str]:
    headers = {"X-CacheLLM-Cache": outcome, "X-CacheLLM-Key": key[:32]}
    if similarity is not None:
        headers["X-CacheLLM-Similarity"] = f"{similarity:.4f}"
    if age is not None:
        headers["X-CacheLLM-Age"] = f"{age:.1f}"
    return headers


async def _read_json(request: Request) -> dict[str, Any]:
    body = await request.body()
    if not body:
        raise ValueError("request body is empty")
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("request body must be a JSON object")
    return data


def _bad_request(message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "source": "cachellm",
            }
        },
        headers={"X-CacheLLM-Request-Id": request_id},
    )


def _strip_control_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove CacheLLM-only knobs before forwarding upstream."""
    return {k: v for k, v in payload.items() if not k.startswith("cachellm_")}


# ---------------------------------------------------------------------------
# main handler shared by /chat/completions and /responses
# ---------------------------------------------------------------------------


async def _handle_completion(request: Request, *, endpoint: str, upstream_path: str) -> Any:
    state = get_state(request)
    request_id = state.new_request_id()
    started = time.perf_counter()

    try:
        payload = await _read_json(request)
    except (ValueError, json.JSONDecodeError) as exc:
        return _bad_request(f"malformed JSON body: {exc}", request_id)

    client_model = str(payload.get("model") or "")
    if not client_model and endpoint == "/v1/chat/completions":
        return _bad_request("'model' is required", request_id)
    if endpoint == "/v1/chat/completions":
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            return _bad_request("'messages' must be a non-empty array", request_id)
    upstream_model = state.resolve_model(client_model)
    if (
        upstream_model
        and not state.config.routing.allow_unlisted_models
        and client_model not in state.config.routing.model_map
    ):
        return _bad_request(f"model {client_model!r} is not in the routing table", request_id)

    wants_stream = bool(payload.get("stream"))
    stream_options = payload.get("stream_options") or {}
    include_usage = bool(isinstance(stream_options, dict) and stream_options.get("include_usage"))

    material, decision = state.engine.prepare(
        payload,
        endpoint=endpoint,
        namespace=_namespace(request, payload),
        resolved_model=upstream_model,
        no_store=_no_store_requested(request, payload),
        explicit_ttl=_explicit_ttl(request),
    )

    try:
        outcome = await state.engine.lookup(material, decision, request_id=request_id)
    except CacheBackendError as exc:
        # Only reachable when fail_open is disabled: report clearly instead of
        # leaking a stack trace, and never serve a possibly-wrong response.
        state.stats.note_cache_error()
        state.stats.record(
            RequestRecord(
                request_id=request_id,
                endpoint=endpoint,
                outcome=OUTCOME_ERROR,
                model=client_model,
                upstream_model=upstream_model,
                namespace=material.namespace,
                cache_key=material.key,
                category=decision.category,
                status_code=503,
                error=f"cache backend unavailable: {exc}",
                total_latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": f"cache backend unavailable: {exc}",
                    "type": "cache_backend_error",
                    "source": "cachellm",
                    "hint": "set cache.fail_open=true to fall back to upstream instead",
                }
            },
            headers={"X-CacheLLM-Request-Id": request_id, "X-CacheLLM-Cache": "error"},
        )

    # ---------------- cache hit ----------------
    if outcome.is_hit and outcome.payload is not None:
        usage = state.token_counter.usage_for(
            request=material.normalized, response=outcome.payload, model=upstream_model
        )
        breakdown = state.costs.breakdown(upstream_model, usage, cached=True)
        record = RequestRecord(
            request_id=request_id,
            endpoint=endpoint,
            outcome=_OUTCOME_BY_SOURCE.get(outcome.source, OUTCOME_EXACT_HIT),
            model=client_model,
            upstream_model=upstream_model,
            namespace=material.namespace,
            cache_key=material.key,
            category=decision.category,
            stream=wants_stream,
            status_code=200,
            similarity=outcome.similarity,
            total_latency_ms=(time.perf_counter() - started) * 1000.0,
            cache_latency_ms=outcome.cache_latency_ms,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            tokens_estimated=usage.estimated,
            cost=0.0,
            cost_without_cache=breakdown.cost_without_cache,
            prompt_excerpt=excerpt(material.query_text),
            response_excerpt=None,
        )
        state.stats.record(record)

        headers = _hit_headers(
            outcome.source, key=material.key, similarity=outcome.similarity, age=outcome.cached_age
        )
        headers["X-CacheLLM-Request-Id"] = request_id
        if wants_stream:
            frames = list(
                chunks_from_response(outcome.payload, include_usage=include_usage)
            )

            async def replay() -> Any:
                for frame in frames:
                    yield frame

            return StreamingResponse(replay(), headers={**SSE_HEADERS, **headers})
        return JSONResponse(content=outcome.payload, headers=headers)

    # ---------------- miss: go upstream ----------------
    provider = state.try_provider()
    if provider is None:
        state.stats.record(
            RequestRecord(
                request_id=request_id,
                endpoint=endpoint,
                outcome=OUTCOME_ERROR,
                model=client_model,
                upstream_model=upstream_model,
                status_code=500,
                error=state.provider_error,
                total_latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": state.provider_error
                    or "upstream provider is not configured; set UPSTREAM_BASE_URL",
                    "type": "configuration_error",
                    "source": "cachellm",
                }
            },
            headers={"X-CacheLLM-Request-Id": request_id},
        )

    upstream_payload = _strip_control_fields(dict(payload))
    upstream_payload["model"] = upstream_model
    headers = _client_headers(request)

    if wants_stream:
        return await _proxy_stream(
            request=request,
            state=state,
            provider=provider,
            upstream_path=upstream_path,
            upstream_payload=upstream_payload,
            client_headers=headers,
            material=material,
            decision=decision,
            request_id=request_id,
            started=started,
            endpoint=endpoint,
            client_model=client_model,
            upstream_model=upstream_model,
            include_usage=include_usage,
        )

    served_after_wait = False

    async def call_upstream() -> dict[str, Any]:
        nonlocal served_after_wait
        # Re-check the cache inside the flight slot.  A request whose lookup
        # missed while a previous leader was in flight would otherwise become a
        # second leader and duplicate the upstream call.
        if decision.cacheable:
            recheck = await state.engine.exact.get(
                material.key, request_id=request_id, quiet=True
            )
            if recheck.hit and recheck.entry is not None:
                served_after_wait = True
                return recheck.entry.payload

        log_event(
            EVENT_UPSTREAM_REQUEST,
            rid=request_id,
            model=upstream_model,
            endpoint=upstream_path,
            stream=False,
        )
        state.stats.note_upstream_request()
        result = await provider.post_json(
            upstream_path, upstream_payload, headers=headers
        )
        # Store *inside* the single-flight slot: releasing the slot before the
        # entry is durable would let a request arriving in that window make a
        # second upstream call.
        if decision.cacheable:
            usage_now = state.token_counter.usage_for(
                request=material.normalized, response=result.payload, model=upstream_model
            )
            stored = await state.engine.store(
                material,
                decision,
                result.payload,
                prompt_tokens=usage_now.prompt_tokens,
                completion_tokens=usage_now.completion_tokens,
                request_id=request_id,
            )
            if stored:
                state.stats.note_cache_write()
        return result.payload

    try:
        exec_outcome = await state.engine.execute(
            material, decision, call_upstream, request_id=request_id
        )
    except ProviderError as exc:
        log_event(
            EVENT_UPSTREAM_ERROR,
            rid=request_id,
            status=exc.status_code,
            error=str(exc),
            model=upstream_model,
        )
        state.stats.record(
            RequestRecord(
                request_id=request_id,
                endpoint=endpoint,
                outcome=OUTCOME_ERROR,
                model=client_model,
                upstream_model=upstream_model,
                namespace=material.namespace,
                cache_key=material.key,
                category=decision.category,
                status_code=exc.status_code,
                error=str(exc),
                total_latency_ms=(time.perf_counter() - started) * 1000.0,
                cache_latency_ms=outcome.cache_latency_ms,
            )
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_client_payload(),
            headers={"X-CacheLLM-Request-Id": request_id, "X-CacheLLM-Cache": "error"},
        )

    response_payload = exec_outcome.payload or {}
    coalesced = exec_outcome.source == SOURCE_COALESCED or served_after_wait

    usage = state.token_counter.usage_for(
        request=material.normalized, response=response_payload, model=upstream_model
    )
    breakdown = state.costs.breakdown(upstream_model, usage, cached=coalesced)

    state.stats.record(
        RequestRecord(
            request_id=request_id,
            endpoint=endpoint,
            outcome=(
                OUTCOME_COALESCED
                if coalesced
                else (OUTCOME_MISS if decision.cacheable else OUTCOME_BYPASS)
            ),
            model=client_model,
            upstream_model=upstream_model,
            namespace=material.namespace,
            cache_key=material.key,
            category=decision.category,
            stream=False,
            status_code=200,
            total_latency_ms=(time.perf_counter() - started) * 1000.0,
            cache_latency_ms=outcome.cache_latency_ms,
            upstream_latency_ms=exec_outcome.upstream_latency_ms,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            tokens_estimated=usage.estimated,
            cost=breakdown.actual_cost,
            cost_without_cache=breakdown.cost_without_cache,
            prompt_excerpt=excerpt(material.query_text),
            response_excerpt=None,
        )
    )

    return JSONResponse(
        content=response_payload,
        headers={
            "X-CacheLLM-Cache": "coalesced" if coalesced else "miss",
            "X-CacheLLM-Key": material.key[:32],
            "X-CacheLLM-Request-Id": request_id,
        },
    )


async def _proxy_stream(
    *,
    request: Request,
    state: Any,
    provider: Any,
    upstream_path: str,
    upstream_payload: dict[str, Any],
    client_headers: dict[str, str],
    material: KeyMaterial,
    decision: Any,
    request_id: str,
    started: float,
    endpoint: str,
    client_model: str,
    upstream_model: str,
    include_usage: bool,
) -> StreamingResponse:
    """Stream upstream to the client while collecting the response for the cache."""
    collector = StreamCollector()
    cache_streaming = state.config.cache.cache_streaming and decision.cacheable

    async def generator() -> Any:
        upstream_started = time.perf_counter()
        error: str | None = None
        status_code = 200
        log_event(
            EVENT_UPSTREAM_REQUEST,
            rid=request_id,
            model=upstream_model,
            endpoint=upstream_path,
            stream=True,
        )
        state.stats.note_upstream_request()
        try:
            async for chunk in provider.stream_json(
                upstream_path, upstream_payload, headers=client_headers
            ):
                collector.feed(chunk.data)
                yield f"data: {chunk.data}\n\n"
        except ProviderError as exc:
            error = str(exc)
            status_code = exc.status_code
            collector.failed = True
            log_event(
                EVENT_UPSTREAM_ERROR,
                rid=request_id,
                status=exc.status_code,
                error=error,
                stream=True,
            )
            yield error_sse(exc.to_client_payload())
        except Exception as exc:  # client disconnect, transport error, ...
            error = f"{type(exc).__name__}: {exc}"
            collector.failed = True
            state.stats.note_stream_abort()
            log_event(EVENT_STREAM_ABORT, rid=request_id, error=error)
        finally:
            upstream_latency = (time.perf_counter() - upstream_started) * 1000.0
            await _finalize_stream(
                state=state,
                collector=collector,
                cache_streaming=cache_streaming,
                material=material,
                decision=decision,
                request_id=request_id,
                started=started,
                upstream_latency=upstream_latency,
                endpoint=endpoint,
                client_model=client_model,
                upstream_model=upstream_model,
                error=error,
                status_code=status_code,
            )

    return StreamingResponse(
        generator(),
        headers={
            **SSE_HEADERS,
            "X-CacheLLM-Cache": "miss",
            "X-CacheLLM-Key": material.key[:32],
            "X-CacheLLM-Request-Id": request_id,
        },
    )


async def _finalize_stream(
    *,
    state: Any,
    collector: StreamCollector,
    cache_streaming: bool,
    material: Any,
    decision: Any,
    request_id: str,
    started: float,
    upstream_latency: float,
    endpoint: str,
    client_model: str,
    upstream_model: str,
    error: str | None,
    status_code: int,
) -> None:
    reconstructed: dict[str, Any] | None = None
    if collector.cacheable:
        reconstructed = collector.to_response(model_hint=upstream_model)
        if cache_streaming:
            stored = await state.engine.store(
                material, decision, reconstructed, request_id=request_id
            )
            if stored:
                state.stats.note_cache_write()
    elif error is None and not collector.completed:
        log_event(
            EVENT_STREAM_ABORT,
            rid=request_id,
            reason="incomplete_stream_not_cached",
            chunks=collector.chunk_count,
        )

    usage = state.token_counter.usage_for(
        request=material.normalized, response=reconstructed, model=upstream_model
    )
    breakdown = state.costs.breakdown(upstream_model, usage, cached=False)
    state.stats.record(
        RequestRecord(
            request_id=request_id,
            endpoint=endpoint,
            outcome=OUTCOME_ERROR if error else OUTCOME_MISS,
            model=client_model,
            upstream_model=upstream_model,
            namespace=material.namespace,
            cache_key=material.key,
            category=decision.category,
            stream=True,
            status_code=status_code,
            total_latency_ms=(time.perf_counter() - started) * 1000.0,
            upstream_latency_ms=upstream_latency,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            tokens_estimated=usage.estimated,
            cost=breakdown.actual_cost if not error else 0.0,
            cost_without_cache=breakdown.cost_without_cache,
            error=error,
            prompt_excerpt=excerpt(material.query_text),
        )
    )


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    return await _handle_completion(
        request, endpoint="/v1/chat/completions", upstream_path="/chat/completions"
    )


@router.post("/v1/responses")
async def responses(request: Request) -> Any:
    return await _handle_completion(
        request, endpoint="/v1/responses", upstream_path="/responses"
    )


@router.get("/v1/models")
async def list_models(request: Request) -> Any:
    state = get_state(request)
    provider = state.try_provider()
    if provider is None:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": state.provider_error or "upstream not configured",
                    "type": "configuration_error",
                    "source": "cachellm",
                }
            },
        )
    try:
        result = await provider.list_models()
    except ProviderError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.to_client_payload())
    return JSONResponse(content=result.payload)


@router.post("/v1/embeddings")
async def embeddings(request: Request) -> Any:
    """Passthrough - embeddings are cheap and usually not worth caching."""
    state = get_state(request)
    provider = state.try_provider()
    if provider is None:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": state.provider_error or "upstream not configured",
                    "type": "configuration_error",
                    "source": "cachellm",
                }
            },
        )
    try:
        payload = await _read_json(request)
    except (ValueError, json.JSONDecodeError) as exc:
        return _bad_request(str(exc), state.new_request_id())
    try:
        result = await provider.post_json(
            "/embeddings", payload, headers=_client_headers(request)
        )
    except ProviderError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.to_client_payload())
    return JSONResponse(content=result.payload)


# ---------------------------------------------------------------------------
# tool-result cache API
# ---------------------------------------------------------------------------


@router.post("/v1/tools/cache/lookup")
async def tool_cache_lookup(request: Request) -> Any:
    state = get_state(request)
    request_id = state.new_request_id()
    try:
        body = await _read_json(request)
    except (ValueError, json.JSONDecodeError) as exc:
        return _bad_request(str(exc), request_id)

    tool = body.get("tool") or body.get("name")
    if not isinstance(tool, str) or not tool:
        return _bad_request("'tool' is required", request_id)

    lookup = await state.engine.tools.lookup(
        tool,
        body.get("arguments"),
        namespace=body.get("namespace"),
        context=body.get("context"),
        ttl=body.get("ttl"),
        request_id=request_id,
    )
    if lookup.hit:
        state.stats.record(
            RequestRecord(
                request_id=request_id,
                endpoint="/v1/tools/cache/lookup",
                outcome=OUTCOME_TOOL_HIT,
                model=f"tool:{tool}",
                cache_key=lookup.key,
                category=lookup.decision.category,
                status_code=200,
                cache_latency_ms=lookup.latency_ms,
                total_latency_ms=lookup.latency_ms,
            )
        )
    else:
        state.stats.note_tool_miss()
    return JSONResponse(
        content={
            "hit": lookup.hit,
            "key": lookup.key,
            "cacheable": lookup.decision.cacheable,
            "ttl": lookup.decision.ttl,
            "category": lookup.decision.category,
            "reason": lookup.decision.reason,
            "result": lookup.result,
            "age_seconds": lookup.age,
            "expires_at": lookup.expires_at,
            "lookup_ms": round(lookup.latency_ms, 3),
        }
    )


@router.post("/v1/tools/cache/store")
async def tool_cache_store(request: Request) -> Any:
    state = get_state(request)
    request_id = state.new_request_id()
    try:
        body = await _read_json(request)
    except (ValueError, json.JSONDecodeError) as exc:
        return _bad_request(str(exc), request_id)

    tool = body.get("tool") or body.get("name")
    if not isinstance(tool, str) or not tool:
        return _bad_request("'tool' is required", request_id)
    if "result" not in body:
        return _bad_request("'result' is required", request_id)

    stored = await state.engine.tools.store(
        tool,
        body.get("arguments"),
        body["result"],
        namespace=body.get("namespace"),
        context=body.get("context"),
        ttl=body.get("ttl"),
        request_id=request_id,
    )
    return JSONResponse(
        content={
            "stored": stored.hit,
            "key": stored.key,
            "cacheable": stored.decision.cacheable,
            "ttl": stored.decision.ttl,
            "category": stored.decision.category,
            "reason": stored.decision.reason,
            "error": stored.error,
        }
    )


@router.post("/v1/tools/cache/policy")
async def tool_cache_policy(request: Request) -> Any:
    """Ask what the policy engine would do with a tool, without touching the cache."""
    state = get_state(request)
    try:
        body = await _read_json(request)
    except (ValueError, json.JSONDecodeError) as exc:
        return _bad_request(str(exc), state.new_request_id())
    tool = body.get("tool") or body.get("name")
    if not isinstance(tool, str) or not tool:
        return _bad_request("'tool' is required", state.new_request_id())
    decision = state.engine.policy.decide_tool(tool, explicit_ttl=body.get("ttl"))
    return JSONResponse(
        content={
            "tool": tool,
            "cacheable": decision.cacheable,
            "ttl": decision.ttl,
            "category": decision.category,
            "reason": decision.reason,
        }
    )
