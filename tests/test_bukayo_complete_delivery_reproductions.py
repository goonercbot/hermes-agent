"""Parent boundary reproductions: a successful send need not deliver the whole final."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from gateway.run import GatewayRunner
from gateway.platforms.base import SendResult
from tests.gateway.test_run_progress_topics import (
    ProgressCaptureAdapter, SessionSource, MessageEvent, MessageType, Platform,
)


def setup_turn():
    adapter = ProgressCaptureAdapter()
    adapter.config.typing_indicator = False
    adapter.delete_message = AsyncMock(return_value=True)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm")
    event = MessageEvent(text="canary", message_type=MessageType.TEXT, source=source, message_id="incoming")
    key = "complete-delivery-review"
    interrupt = asyncio.Event()
    interrupt._hermes_run_generation = 1
    adapter._active_sessions[key] = interrupt
    ctx = SimpleNamespace(_cleanup_progress=True, _cleanup_msg_ids=["progress"],
                          _commentary_messages=[("progress", "Working")], session_key=key,
                          source=source, run_generation=1)
    runner = object.__new__(GatewayRunner)
    return adapter, source, event, key, ctx, runner


@pytest.mark.asyncio
async def test_successful_truncated_fallback_cannot_authorize_cleanup():
    adapter, source, event, key, ctx, runner = setup_turn()
    final = "a" * 4000 + " TAIL_UNDELIVERED"
    delivered = []

    async def send(chat_id, content, **kwargs):
        if content == final:
            return SendResult(success=False, error="Bad Request: can't parse entities")
        delivered.append(content)
        return SendResult(success=True, message_id="sent")

    adapter.send = send
    await adapter.send(source.chat_id, "Working")

    async def handler(_event):
        runner._run_agent_schedule_bubble_cleanup({"failed": False, "final_response": final}, adapter, ctx)
        return final

    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, key)
    for _ in range(5):
        await asyncio.sleep(0)
    assert len(delivered) >= 2 and any("a" * 3500 in s for s in delivered)
    assert all("TAIL_UNDELIVERED" not in s for s in delivered)
    adapter.delete_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_transformed_stream_edit_cannot_authorize_cleanup():
    adapter, source, event, key, ctx, runner = setup_turn()
    original = "original answer"
    final = original + " TRANSFORMED_TAIL"
    delivered = []

    async def send(chat_id, content, **kwargs):
        if content in ("Working", original):
            delivered.append(content)
            return SendResult(success=True, message_id="stream")
        return SendResult(success=False, error="injected final failure")

    adapter.send = send
    adapter.edit_message = AsyncMock(return_value=SendResult(success=False, error="injected edit failure"))
    await adapter.send(source.chat_id, "Working")
    initial = await adapter.send(source.chat_id, original)
    ctx.stream_consumer_holder = [SimpleNamespace(
        adapter=adapter, message_id=initial.message_id, final_content_delivered=initial.success,
        final_response_sent=initial.success, delivered_final_matches=lambda text: text == original,
        stream_deltas_enabled=True, _turn_split_delivery=False,
    )]
    runner._delivery_adapter_for = lambda _source: adapter
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._deliver_media_from_response = AsyncMock()

    async def handler(_event):
        result = {"failed": False, "final_response": final, "response_transformed": True,
                  "response_previewed": True}
        await runner._run_agent_mark_streamed_delivery(result, ctx)
        runner._run_agent_schedule_bubble_cleanup(result, adapter, ctx)
        return await runner._hmwa_deliver_turn_response(
            _event, source, SimpleNamespace(session_id="test"), key, 1,
            result, [], final, None, False,
        )

    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, key)
    for _ in range(5):
        await asyncio.sleep(0)
    adapter.edit_message.assert_awaited_once()
    assert all("TRANSFORMED_TAIL" not in text for text in delivered)
    adapter.delete_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_complete_short_plain_fallback_authorizes_cleanup():
    adapter, source, event, key, ctx, runner = setup_turn()
    final = "short complete answer"
    delivered = []

    async def send(chat_id, content, **kwargs):
        if content == final:
            return SendResult(success=False, error="Bad Request: can't parse entities")
        delivered.append(content)
        return SendResult(success=True, message_id="sent")

    adapter.send = send
    await adapter.send(source.chat_id, "Working")

    async def handler(_event):
        runner._run_agent_schedule_bubble_cleanup({"failed": False, "final_response": final}, adapter, ctx)
        return final

    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, key)
    for _ in range(5):
        await asyncio.sleep(0)
    assert any(final in text for text in delivered)
    adapter.delete_message.assert_awaited_once_with(source.chat_id, "progress")


@pytest.mark.asyncio
async def test_successful_transformed_stream_edit_authorizes_cleanup():
    adapter, source, event, key, ctx, runner = setup_turn()
    original = "original answer"
    final = original + " TRANSFORMED_TAIL"

    async def send(chat_id, content, **kwargs):
        return SendResult(success=True, message_id="stream")

    adapter.send = send
    adapter.edit_message = AsyncMock(return_value=SendResult(success=True, message_id="stream"))
    initial = await adapter.send(source.chat_id, original)
    ctx.stream_consumer_holder = [SimpleNamespace(
        adapter=adapter, message_id=initial.message_id, final_content_delivered=initial.success,
        final_response_sent=initial.success, delivered_final_matches=lambda text: text == original,
        stream_deltas_enabled=True, _turn_split_delivery=False,
    )]
    runner._delivery_adapter_for = lambda _source: adapter
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._deliver_media_from_response = AsyncMock()

    async def handler(_event):
        result = {"failed": False, "final_response": final, "response_transformed": True,
                  "response_previewed": True}
        await runner._run_agent_mark_streamed_delivery(result, ctx)
        runner._run_agent_schedule_bubble_cleanup(result, adapter, ctx)
        return await runner._hmwa_deliver_turn_response(
            _event, source, SimpleNamespace(session_id="test"), key, 1,
            result, [], final, None, False,
        )

    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, key)
    for _ in range(5):
        await asyncio.sleep(0)
    adapter.edit_message.assert_awaited_once_with(
        chat_id=source.chat_id, message_id="stream", content=final, finalize=True,
    )
    adapter.delete_message.assert_awaited_once_with(source.chat_id, "progress")
