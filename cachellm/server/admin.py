"""Admin / dashboard API.

Everything the dashboard needs, all JSON:

GET    /health                      liveness + cache + upstream status
GET    /api/stats                   full statistics snapshot
GET    /api/requests                recent requests (filterable by outcome)
GET    /api/cache                   cache browser (search/paginate)
GET    /api/cache/{key}             inspect one entry
DELETE /api/cache/{key}             delete one entry
POST   /api/cache/invalidate        by key / model / namespace / tool / all
GET    /api/tools/cache             tool cache browser
GET    /api/config                  redacted effective config
POST   /api/config                  update safe runtime settings
GET    /api/pricing  POST /api/pricing  DELETE /api/pricing/{model}
GET    /api/policy                  policy engine description
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from ..config import ModelPricing
from ..logging_utils import EVENT_CACHE_INVALIDATE, log_event
from .deps import get_state

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> Any:
    state = get_state(request)
    payload = await state.health()
    code = 200 if payload["cache"]["exact"].get("ok", True) else 503
    return JSONResponse(status_code=code, content=payload)


@router.get("/api/stats")
async def stats(request: Request) -> Any:
    state = get_state(request)
    return JSONResponse(content=state.stats.snapshot())


@router.get("/api/requests")
async def requests_log(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    outcome: str | None = None,
) -> Any:
    state = get_state(request)
    return JSONResponse(content={"requests": state.stats.recent(limit=limit, outcome=outcome)})


@router.get("/api/cache")
async def cache_browser(
    request: Request,
    search: str | None = None,
    namespace: str | None = None,
    model: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Any:
    state = get_state(request)
    if state.db is None:
        return JSONResponse(
            content={"entries": [], "total": 0, "note": "cache browser requires the sqlite backend"}
        )
    rows = await state.db.run(
        lambda: state.db.list_cache_entries(  # type: ignore[union-attr]
            search=search, namespace=namespace, model=model, limit=limit, offset=offset
        )
    )
    total = await state.db.run(state.db.count_cache_entries)
    now = time.time()
    entries = []
    for row in rows:
        data = dict(row)
        if not state.config.privacy.store_prompts:
            data["prompt_excerpt"] = None
        if not state.config.privacy.store_responses:
            data["response_excerpt"] = None
        data["expired"] = bool(data["expires_at"] and data["expires_at"] <= now)
        data["ttl_remaining"] = (
            max(0, int(data["expires_at"] - now)) if data["expires_at"] else None
        )
        entries.append(data)
    return JSONResponse(content={"entries": entries, "total": total})


@router.get("/api/cache/{key}")
async def cache_inspect(request: Request, key: str) -> Any:
    state = get_state(request)
    if state.db is None:
        return JSONResponse(status_code=404, content={"error": "sqlite backend required"})
    detail = await state.db.run(state.db.cache_entry_detail, key)
    if detail is None:
        return JSONResponse(status_code=404, content={"error": "entry not found"})
    if not state.config.privacy.store_prompts:
        detail["request_json"] = None
        detail["prompt_excerpt"] = None
    if not state.config.privacy.store_responses:
        detail["response_excerpt"] = None
    return JSONResponse(content=detail)


@router.delete("/api/cache/{key}")
async def cache_delete(request: Request, key: str) -> Any:
    state = get_state(request)
    removed = await state.engine.invalidate(key=key)
    log_event(EVENT_CACHE_INVALIDATE, scope="key", key=key[:16], removed=removed)
    return JSONResponse(content={"removed": removed})


@router.post("/api/cache/invalidate")
async def cache_invalidate(request: Request) -> Any:
    state = get_state(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    removed = await state.engine.invalidate(
        key=body.get("key"),
        model=body.get("model"),
        namespace=body.get("namespace"),
        tool=body.get("tool"),
        all_entries=bool(body.get("all")),
    )
    log_event(
        EVENT_CACHE_INVALIDATE,
        scope=(
            "all"
            if body.get("all")
            else ",".join(k for k in ("key", "model", "namespace", "tool") if body.get(k))
        ),
        removed=removed,
    )
    return JSONResponse(content={"removed": removed})


@router.get("/api/tools/cache")
async def tool_cache_browser(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Any:
    state = get_state(request)
    entries = await state.engine.tools.list_entries(limit=limit, offset=offset)
    total = await state.engine.tools.size()
    now = time.time()
    for entry in entries:
        entry["expired"] = bool(entry.get("expires_at") and entry["expires_at"] <= now)
    return JSONResponse(content={"entries": entries, "total": total})


@router.delete("/api/tools/cache/{key}")
async def tool_cache_delete(request: Request, key: str) -> Any:
    state = get_state(request)
    removed = await state.engine.tools.invalidate_key(key)
    return JSONResponse(content={"removed": removed})


@router.get("/api/config")
async def get_config(request: Request) -> Any:
    state = get_state(request)
    return JSONResponse(
        content={
            "config": state.config.redacted(),
            "env_docs": [{"name": n, "description": d} for n, d in _env_docs()],
            "providers": _providers(),
            "config_file": state.config.source_path,
        }
    )


# Runtime-editable settings.  Secrets are intentionally NOT editable here so the
# dashboard can never write an API key into the database.
_EDITABLE: dict[str, tuple[str, str, type]] = {
    "cache.enabled": ("cache", "enabled", bool),
    "cache.default_ttl": ("cache", "default_ttl", int),
    "cache.namespace": ("cache", "namespace", str),
    "cache.fail_open": ("cache", "fail_open", bool),
    "cache.cache_streaming": ("cache", "cache_streaming", bool),
    "cache.max_temperature": ("cache", "max_temperature", float),
    "cache.cache_zero_temperature_only": ("cache", "cache_zero_temperature_only", bool),
    "semantic.enabled": ("semantic", "enabled", bool),
    "semantic.threshold": ("semantic", "threshold", float),
    "semantic.require_single_turn": ("semantic", "require_single_turn", bool),
    "semantic.allow_tools": ("semantic", "allow_tools", bool),
    "privacy.store_prompts": ("privacy", "store_prompts", bool),
    "privacy.store_responses": ("privacy", "store_responses", bool),
    "privacy.store_tool_arguments": ("privacy", "store_tool_arguments", bool),
    "privacy.store_tool_results": ("privacy", "store_tool_results", bool),
    "privacy.retention_days": ("privacy", "retention_days", int),
    "concurrency.single_flight": ("concurrency", "single_flight", bool),
    "pricing.token_estimator": ("pricing", "token_estimator", str),
    "pricing.chars_per_token": ("pricing", "chars_per_token", float),
    "pricing.currency": ("pricing", "currency", str),
}


@router.post("/api/config")
async def update_config(request: Request) -> Any:
    state = get_state(request)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid JSON body"})
    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"error": "body must be an object"})

    applied: dict[str, Any] = {}
    rejected: dict[str, str] = {}
    for path, value in body.items():
        target = _EDITABLE.get(path)
        if target is None:
            rejected[path] = "not editable at runtime"
            continue
        section, attr, kind = target
        try:
            coerced = _coerce(value, kind)
        except (TypeError, ValueError) as exc:
            rejected[path] = f"invalid value: {exc}"
            continue
        setattr(getattr(state.config, section), attr, coerced)
        applied[path] = coerced
        if state.db is not None:
            try:
                state.db.set_config_value(path, coerced)
            except ValueError as exc:
                rejected[path] = str(exc)

    # Propagate the settings that other components cached at construction time.
    state.stats.store_prompts = state.config.privacy.store_prompts
    state.stats.store_responses = state.config.privacy.store_responses
    state.engine.tools.store_arguments = state.config.privacy.store_tool_arguments
    state.engine.tools.store_results = state.config.privacy.store_tool_results
    state.engine.exact.fail_open = state.config.cache.fail_open
    state.engine.tools.namespace = state.config.cache.namespace

    return JSONResponse(
        content={"applied": applied, "rejected": rejected, "config": state.config.redacted()}
    )


@router.get("/api/pricing")
async def get_pricing(request: Request) -> Any:
    state = get_state(request)
    return JSONResponse(
        content={
            "currency": state.config.pricing.currency,
            "default": {
                "input_per_1m": state.config.pricing.default.input_per_1m,
                "output_per_1m": state.config.pricing.default.output_per_1m,
            },
            "models": state.costs.table(),
            "token_estimator": state.config.pricing.token_estimator,
        }
    )


@router.post("/api/pricing")
async def set_pricing(request: Request) -> Any:
    state = get_state(request)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid JSON body"})
    model = body.get("model")
    if not isinstance(model, str) or not model:
        return JSONResponse(status_code=400, content={"error": "'model' is required"})
    try:
        pricing = ModelPricing(
            input_per_1m=float(body.get("input_per_1m") or 0.0),
            output_per_1m=float(body.get("output_per_1m") or 0.0),
            cached_input_per_1m=(
                float(body["cached_input_per_1m"])
                if body.get("cached_input_per_1m") is not None
                else None
            ),
        )
    except (TypeError, ValueError) as exc:
        return JSONResponse(status_code=400, content={"error": f"invalid price: {exc}"})
    state.set_pricing(model, pricing)
    return JSONResponse(content={"model": model, "pricing": state.costs.table()})


@router.delete("/api/pricing/{model:path}")
async def delete_pricing(request: Request, model: str) -> Any:
    state = get_state(request)
    removed = state.delete_pricing(model)
    return JSONResponse(content={"removed": removed, "models": state.costs.table()})


@router.get("/api/policy")
async def get_policy(request: Request) -> Any:
    state = get_state(request)
    return JSONResponse(content=state.engine.policy.describe())


@router.post("/api/maintenance/purge")
async def purge(request: Request) -> Any:
    state = get_state(request)
    purged = await state.engine.purge_expired()
    pruned = 0
    if state.db is not None and state.config.privacy.retention_days > 0:
        pruned = await state.db.run(
            state.db.prune_history, state.config.privacy.retention_days
        )
    return JSONResponse(content={"purged": purged, "pruned_requests": pruned})


def _coerce(value: Any, kind: type) -> Any:
    if kind is bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{value!r} is not a boolean")
    if kind is int:
        return int(value)
    if kind is float:
        return float(value)
    return str(value)


def _env_docs() -> Any:
    from ..config import env_docs

    return env_docs()


def _providers() -> list[str]:
    from ..providers import available_providers

    return available_providers()
