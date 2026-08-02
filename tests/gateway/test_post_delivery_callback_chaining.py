"""Tests for ``BasePlatformAdapter.register_post_delivery_callback`` chaining.

When two features want to run after the final response lands on the same
session (e.g. background-review release + temporary-progress cleanup), the
registration API chains them rather than clobbering. Per-callback
exceptions are swallowed so one bad callback can't sabotage the others.
Stale-generation registrations are rejected.

The chained wrapper is ``async`` so it transparently supports sync or async
callbacks — the outer invoker in ``_handle_message`` awaits awaitable
callbacks, and a sync wrapper would silently drop coroutine results from
async callbacks chained behind it.
"""
import asyncio
import inspect

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource


class _MinAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.fixture
def adapter():
    return _MinAdapter(PlatformConfig(enabled=True), Platform.TELEGRAM)


def _invoke(cb):
    """Invoke a popped callback, awaiting if it returns a coroutine.

    Single-registration callbacks are returned as the raw user callable
    (sync). Chained callbacks (two or more registrations on the same
    session) are wrapped in an async helper. Tests use this helper so
    they don't have to care which case they're exercising.
    """
    result = cb()
    if inspect.isawaitable(result):
        asyncio.run(result)


class TestPostDeliveryCallbackChaining:
    def test_single_callback_fires(self, adapter):
        fired = []
        adapter.register_post_delivery_callback("s", lambda: fired.append("A"))
        cb = adapter.pop_post_delivery_callback("s")
        _invoke(cb)
        assert fired == ["A"]

    def test_two_callbacks_chain_in_order(self, adapter):
        fired = []
        adapter.register_post_delivery_callback("s", lambda: fired.append("A"))
        adapter.register_post_delivery_callback("s", lambda: fired.append("B"))
        cb = adapter.pop_post_delivery_callback("s")
        _invoke(cb)
        assert fired == ["A", "B"]

    def test_three_callbacks_chain_in_order(self, adapter):
        """Chain composes over an already-chained callback."""
        fired = []
        for label in ("A", "B", "C"):
            adapter.register_post_delivery_callback(
                "s", lambda x=label: fired.append(x)
            )
        cb = adapter.pop_post_delivery_callback("s")
        _invoke(cb)
        assert fired == ["A", "B", "C"]


class TestPostDeliveryCallbackAsyncChaining:
    """When an async callback is chained, the wrapper must await it.

    Regression test for a bug where the sync ``_chained`` wrapper called
    async callbacks without awaiting, silently dropping the returned
    coroutine. This broke ``/goal`` continuations (Discord etc.) where
    the continuation injection is an async ``_deliver()`` coroutine.
    """

    def test_async_callback_in_chain_is_awaited(self, adapter):
        fired = []

        async def async_cb():
            await asyncio.sleep(0)
            fired.append("async")

        adapter.register_post_delivery_callback("s", lambda: fired.append("sync"))
        adapter.register_post_delivery_callback("s", async_cb)
        cb = adapter.pop_post_delivery_callback("s")
        _invoke(cb)
        assert fired == ["sync", "async"]


class TestFinalDeliveryState:
    @pytest.mark.asyncio
    async def test_callback_observes_successful_final_delivery(self, adapter):
        session_key = "agent:main:telegram:private:c1"
        observed = []

        async def handler(_event):
            adapter.register_post_delivery_callback(
                session_key,
                lambda: observed.append(
                    getattr(
                        adapter._active_sessions[session_key],
                        "_hermes_final_delivery_succeeded",
                        None,
                    )
                ),
            )
            return "final answer"

        adapter.set_message_handler(handler)
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="c1",
            user_id="u1",
            chat_type="private",
        )
        await adapter._process_message_background(
            MessageEvent(text="hello", message_type=MessageType.TEXT, source=source),
            session_key,
        )

        assert observed == [True]

    @pytest.mark.asyncio
    async def test_callback_observes_failed_final_delivery(self, adapter):
        session_key = "agent:main:telegram:private:c1"
        observed = []

        async def failed_send(*_args, **_kwargs):
            return SendResult(success=False, error="simulated delivery failure")

        adapter.send = failed_send

        async def handler(_event):
            adapter.register_post_delivery_callback(
                session_key,
                lambda: observed.append(
                    getattr(
                        adapter._active_sessions[session_key],
                        "_hermes_final_delivery_succeeded",
                        None,
                    )
                ),
            )
            return "final answer"

        adapter.set_message_handler(handler)
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="c1",
            user_id="u1",
            chat_type="private",
        )
        await adapter._process_message_background(
            MessageEvent(text="hello", message_type=MessageType.TEXT, source=source),
            session_key,
        )

        assert observed == [False]

    @pytest.mark.asyncio
    async def test_queued_turn_cannot_steal_prior_delivery_callback(self, adapter):
        """Queued turns start only after the prior callback has completed."""
        session_key = "agent:main:telegram:private:c1"
        observed = []
        second_registered = asyncio.Event()
        stop_calls = 0

        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="c1",
            user_id="u1",
            chat_type="private",
        )
        second_event = MessageEvent(
            text="second", message_type=MessageType.TEXT, source=source
        )

        async def send(chat_id, content, **_kwargs):
            if content != "reply-1":
                return SendResult(
                    success=False, error="simulated queued delivery failure"
                )
            return SendResult(success=True, message_id="first-final")

        async def stop_typing_refresh(*_args, **_kwargs):
            nonlocal stop_calls
            stop_calls += 1
            if stop_calls == 2:
                try:
                    await asyncio.wait_for(second_registered.wait(), timeout=0.05)
                except asyncio.TimeoutError:
                    pass

        adapter.send = send
        adapter._stop_typing_refresh = stop_typing_refresh

        async def handler(event):
            label = event.text
            adapter.register_post_delivery_callback(
                session_key,
                lambda label=label: observed.append(
                    (
                        label,
                        getattr(
                            adapter._active_sessions[session_key],
                            "_hermes_final_delivery_succeeded",
                            None,
                        ),
                    )
                ),
            )
            if label == "first":
                adapter._pending_messages[session_key] = second_event
                return "reply-1"
            second_registered.set()
            return "reply-2"

        adapter.set_message_handler(handler)
        await adapter._process_message_background(
            MessageEvent(
                text="first", message_type=MessageType.TEXT, source=source
            ),
            session_key,
        )

        for _ in range(100):
            if len(observed) == 2 and session_key not in adapter._active_sessions:
                break
            await asyncio.sleep(0.01)

        assert observed == [("first", True), ("second", False)]
        assert adapter._post_delivery_callbacks == {}

