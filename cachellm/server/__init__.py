"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ..app import AppState
from ..config import Config
from ..logging_utils import get_logger
from . import admin, proxy

log = get_logger("cachellm.server")

STATIC_DIR = Path(__file__).parent / "static"


def create_app(config: Config | None = None, *, state: AppState | None = None) -> FastAPI:
    app_state = state or AppState(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await app_state.startup()
        try:
            yield
        finally:
            await app_state.shutdown()

    app = FastAPI(
        title="CacheLLM",
        description="Local-first caching proxy for OpenAI-compatible LLM APIs",
        version=_version(),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.cachellm = app_state

    app.include_router(proxy.router)
    app.include_router(admin.router)

    if app_state.config.server.dashboard_enabled:
        _mount_dashboard(app)

    @app.get("/")
    async def root() -> Any:
        if app_state.config.server.dashboard_enabled:
            return RedirectResponse(url="/dashboard")
        return JSONResponse(
            content={
                "name": "CacheLLM",
                "version": _version(),
                "endpoints": ["/v1/chat/completions", "/v1/responses", "/health", "/api/stats"],
            }
        )

    @app.exception_handler(404)
    async def not_found(request: Request, exc: Any) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"no route for {request.method} {request.url.path}",
                    "type": "not_found",
                    "source": "cachellm",
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"internal proxy error: {type(exc).__name__}",
                    "type": "internal_error",
                    "source": "cachellm",
                }
            },
        )

    return app


def _mount_dashboard(app: FastAPI) -> None:
    index = STATIC_DIR / "index.html"

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        if not index.is_file():  # pragma: no cover
            return HTMLResponse("<h1>CacheLLM</h1><p>dashboard assets missing</p>", status_code=500)
        return HTMLResponse(index.read_text(encoding="utf-8"))


def _version() -> str:
    from .. import __version__

    return __version__


__all__ = ["create_app"]
