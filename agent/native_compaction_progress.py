"""Request-owned progress for native compression, never agent-global state.

The compression owner installs this scope before dispatch. Context propagation
carries its exact fence into the Codex stream worker; a later request cannot
replace it. Non-native calls have no scope and keep their existing watchdogs.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Iterator


@dataclass
class NativeCompactionRequest:
    fence: Any
    last_event_ts: float | None = None
    _closed: threading.Event = field(default_factory=threading.Event)

    @property
    def cancelled(self) -> bool:
        """Work with both source and v0.21.1 fence cancellation contracts."""
        if self._closed.is_set() or self.fence is None:
            return self._closed.is_set()
        cancellation_requested = getattr(self.fence, "cancellation_requested", None)
        if cancellation_requested is not None:
            return bool(cancellation_requested() if callable(cancellation_requested) else cancellation_requested)
        is_cancelled = getattr(self.fence, "is_cancelled", None)
        return bool(is_cancelled() if callable(is_cancelled) else is_cancelled)

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise InterruptedError("Native compression request was cancelled")

    def on_event(self) -> None:
        self.check_cancelled()
        self.last_event_ts = time.time()
        # A native checkpoint may be opaque until its terminal frame. All valid
        # SSE events establish liveness; the existing hard total ceiling still
        # bounds a provider that sends keepalives forever.
        if self.fence is not None:
            self.fence.touch_progress()

    def close(self) -> None:
        self._closed.set()


_current_request: ContextVar[NativeCompactionRequest | None] = ContextVar(
    "native_compaction_request", default=None
)


def current_native_compaction_request() -> NativeCompactionRequest | None:
    return _current_request.get()


@contextmanager
def native_compaction_request(fence: Any) -> Iterator[NativeCompactionRequest]:
    request = NativeCompactionRequest(fence)
    token = _current_request.set(request)
    try:
        request.check_cancelled()
        yield request
    finally:
        request.close()
        _current_request.reset(token)
