"""Preserve direct and split commentary cleanup across the gateway refactor."""

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter


def _context():
    return SimpleNamespace(
        _cleanup_progress=True, _cleanup_msg_ids=[], _commentary_messages=[],
        _direct_commentary_futures=[], _run_still_current=lambda: True,
        streaming_tts_consumer_holder=[None], stream_consumer_holder=[None],
        user_config={}, resolve_display_setting=lambda *_args: False,
        interim_assistant_messages_enabled=True,
        source=SimpleNamespace(chat_id="chat"), session_key=None,
        _status_chat_id="chat", _status_thread_metadata=None,
        mute_notification_reply=False, scheduled_heartbeat=False,
    )


def _split_result(success=True):
    return SendResult(
        success=success, message_id="last", continuation_message_ids=("first", "middle"),
        raw_response={"message_ids": ["first", "middle", "last"]},
    )


def test_split_commentary_tracks_all_unique_ids_and_final_text():
    ctx = _context()
    turn = TurnRunner(None, ctx)
    turn._track_commentary_cleanup(_split_result(), "answer")
    assert ctx._cleanup_msg_ids == ["last", "first", "middle"]
    assert ctx._commentary_messages == [(mid, "answer") for mid in ctx._cleanup_msg_ids]


@pytest.mark.parametrize("outcome", ["success", "failed", "exception"])
def test_direct_commentary_fallback_tracks_completion(monkeypatch, outcome):
    ctx = _context()

    async def send(*_args, **_kwargs):
        raise AssertionError("the scheduling fake must not execute the send")

    ctx._status_adapter = SimpleNamespace(send=send)
    runner = SimpleNamespace(
        config=SimpleNamespace(streaming=SimpleNamespace(enabled=False, transport="off")),
        _adapter_for_source=lambda _source: ctx._status_adapter,
        _build_stream_consumer_config=lambda *_args: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    turn = TurnRunner(runner, ctx)
    future = Future()

    def schedule(coro, *_args, **_kwargs):
        coro.close()
        return future

    monkeypatch.setattr(turn, "_schedule", schedule)
    consumer, _, callback, _ = turn._setup_stream_consumer("telegram")
    assert consumer is None
    callback("answer")
    assert ctx._direct_commentary_futures == [future]
    if outcome == "exception":
        future.set_exception(RuntimeError("send failed"))
    else:
        future.set_result(_split_result(success=outcome == "success"))
    assert ctx._cleanup_msg_ids == (["last", "first", "middle"] if outcome == "success" else [])


@pytest.mark.asyncio
async def test_cleanup_waits_for_late_direct_send_before_releasing_turn():
    ctx = _context()
    future = Future()
    ctx._direct_commentary_futures.append(future)
    turn = TurnRunner(None, ctx)
    future.add_done_callback(lambda done: turn._track_commentary_cleanup(done.result(), "answer"))
    tracking = asyncio.create_task(asyncio.Event().wait())
    runner = object.__new__(GatewayRunner)
    runner._draining = False
    cleanup = asyncio.create_task(runner._run_agent_cleanup_turn_tasks(
        ctx, progress_task=None, log_task=None, interrupt_monitor=None,
        _notify_task=None, tracking_task=tracking, stream_task=None,
    ))
    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not tracking.cancelled()
        future.set_result(_split_result())
        await asyncio.wait_for(cleanup, timeout=1)
        assert ctx._cleanup_msg_ids == ["last", "first", "middle"]
        assert tracking.cancelled()
    finally:
        if not future.done():
            future.set_result(SendResult(success=False))
        cleanup.cancel()
        tracking.cancel()
        await asyncio.gather(cleanup, tracking, return_exceptions=True)


async def _deliver_with_cleanup(*, final_result, streamed=False, generation=1, commentary=None,
                                extra_callback=None):
    adapter = ProgressCaptureAdapter()
    adapter.config.typing_indicator = False
    adapter.send = AsyncMock(return_value=final_result)
    adapter.delete_message = AsyncMock(return_value=True)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm")
    event = MessageEvent(text="question", message_type=MessageType.TEXT, source=source, message_id="incoming")
    session_key = "cleanup-delivery"
    interrupt = asyncio.Event()
    interrupt._hermes_run_generation = generation
    adapter._active_sessions[session_key] = interrupt
    runner = object.__new__(GatewayRunner)
    ctx = SimpleNamespace(
        _cleanup_progress=True,
        _cleanup_msg_ids=[message_id for message_id, _text in commentary or [("progress", "Working")]],
        _commentary_messages=commentary or [("progress", "Working")],
        session_key=session_key,
        source=source,
        run_generation=1,
    )
    if extra_callback is not None:
        adapter.register_post_delivery_callback(session_key, extra_callback, generation=generation)

    async def handler(_event):
        runner._run_agent_schedule_bubble_cleanup(
            {"failed": False, "final_response": "Complete answer"}, adapter, ctx,
        )
        if streamed:
            # This marker is produced by GatewayTurnMixin only after it reconciles
            # the consumer's durable final payload with the completed final response.
            event._streamed_final_response = "Complete answer"
            return None
        return "Complete answer"

    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, session_key)
    for _ in range(5):
        await asyncio.sleep(0)
    return adapter


@pytest.mark.asyncio
async def test_cleanup_follows_successful_final_and_keeps_matching_interim_bubble():
    adapter = await _deliver_with_cleanup(
        final_result=SendResult(success=True, message_id="final"),
        commentary=[("matching", "Complete answer"), ("progress", "Working")],
    )
    adapter.delete_message.assert_awaited_once_with("chat", "progress")


@pytest.mark.asyncio
async def test_legacy_streamed_marker_does_not_authorize_cleanup():
    adapter = await _deliver_with_cleanup(
        final_result=SendResult(success=False), streamed=True,
    )
    adapter.send.assert_not_awaited()
    adapter.delete_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_keeps_partial_stream_when_fallback_final_fails():
    adapter = await _deliver_with_cleanup(final_result=SendResult(success=False))
    assert adapter.send.await_count >= 1
    adapter.delete_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_does_not_block_an_unrelated_post_delivery_callback():
    fired = []
    adapter = await _deliver_with_cleanup(
        final_result=SendResult(success=True, message_id="final"),
        extra_callback=lambda: fired.append("unrelated"),
    )
    assert fired == ["unrelated"]
    adapter.delete_message.assert_awaited_once_with("chat", "progress")


@pytest.mark.asyncio
async def test_stale_generation_cannot_delete_a_newer_turns_commentary():
    adapter = await _deliver_with_cleanup(
        final_result=SendResult(success=True, message_id="final"), generation=2,
    )
    adapter.delete_message.assert_not_awaited()
    assert adapter.pop_post_delivery_callback("cleanup-delivery", generation=1) is not None
