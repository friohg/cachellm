"""Single-flight / request coalescing.

If N identical requests arrive before the first one finishes, only the first
goes upstream; the rest await the same future and receive the same result.
Failures propagate to every waiter (nothing is cached), and the slot is always
released - a crashed leader can never leave a stale lock behind.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generic, TypeVar

from .logging_utils import EVENT_COALESCED, get_logger, log_event

T = TypeVar("T")
log = get_logger("cachellm.singleflight")


@dataclass
class _Slot(Generic[T]):
    future: "asyncio.Future[T]"
    waiters: int = 1
    leader: bool = True


@dataclass
class FlightResult(Generic[T]):
    value: T
    leader: bool
    """True if this caller actually executed the work."""
    waiters: int = 1


class SingleFlight(Generic[T]):
    """Coalesces concurrent calls that share a key."""

    def __init__(self, *, wait_timeout: float = 600.0, max_inflight: int = 256) -> None:
        self.wait_timeout = wait_timeout
        self.max_inflight = max_inflight
        self._slots: dict[str, _Slot[T]] = {}
        self._lock = asyncio.Lock()
        self.coalesced_count = 0

    @property
    def inflight(self) -> int:
        return len(self._slots)

    async def run(
        self,
        key: str,
        factory: Callable[[], Awaitable[T]],
        *,
        request_id: str | None = None,
    ) -> FlightResult[T]:
        async with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                if len(self._slots) >= self.max_inflight:
                    # Backpressure: run without coalescing rather than queueing forever.
                    value = await factory()
                    return FlightResult(value, leader=True, waiters=1)
                loop = asyncio.get_running_loop()
                slot = _Slot(future=loop.create_future())
                self._slots[key] = slot
                is_leader = True
            else:
                slot.waiters += 1
                is_leader = False
                self.coalesced_count += 1
                log_event(
                    EVENT_COALESCED,
                    rid=request_id,
                    key=key[:16],
                    waiters=slot.waiters,
                )

        if is_leader:
            try:
                value = await factory()
            except BaseException as exc:  # noqa: BLE001 - propagate to all waiters
                await self._finish(key, exc=exc)
                raise
            waiters = await self._finish(key, value=value)
            return FlightResult(value, leader=True, waiters=waiters)

        try:
            value = await asyncio.wait_for(
                asyncio.shield(slot.future), timeout=self.wait_timeout
            )
        except asyncio.TimeoutError:
            # Never hang an agent forever: run independently instead.
            value = await factory()
            return FlightResult(value, leader=True, waiters=slot.waiters)
        return FlightResult(value, leader=False, waiters=slot.waiters)

    async def _finish(
        self, key: str, *, value: Any = None, exc: BaseException | None = None
    ) -> int:
        async with self._lock:
            slot = self._slots.pop(key, None)
        if slot is None:
            return 1
        if not slot.future.done():
            if exc is not None:
                slot.future.set_exception(exc)
            else:
                slot.future.set_result(value)
        # Ensure an unobserved exception never spams the event loop.
        if exc is not None:
            slot.future.exception()
        return slot.waiters

    def stats(self) -> dict[str, int]:
        return {"inflight": self.inflight, "coalesced": self.coalesced_count}
