"""Request-scoped access to the application state."""

from __future__ import annotations

from fastapi import Request

from ..app import AppState


def get_state(request: Request) -> AppState:
    state: AppState | None = getattr(request.app.state, "cachellm", None)
    if state is None:  # pragma: no cover - app factory always sets this
        raise RuntimeError("CacheLLM application state is not initialised")
    return state
