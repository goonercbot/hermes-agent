"""Preserve direct and split commentary cleanup across the gateway refactor."""

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner


def _context():
    return SimpleNamespace(
        _cleanup_progress=True, _cleanup_msg_ids=[], _commentary_messages=[],
        _direct_commentary_futures=[], _run_still_current=lambda: True,
        streaming_tts_consumer_holder=[None], stream_consumer_holder=[None],
        user_config={}, resolve_display_setting=lambda *_args: False,
        interim_assistant_messages_enabled=True,
        source=SimpleNamespace(chat_id="chat"), session_key=None,
        _status_chat_id="chat", _status_thread_metadata=None,
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

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("stream consumer setup unavailable")

    runner = SimpleNamespace(
        config=SimpleNamespace(streaming=SimpleNamespace(enabled=False, transport="off")),
        _adapter_for_source=lambda _source: ctx._status_adapter,
        _build_stream_consumer_config=unavailable,
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
    assert not ctx._cleanup_msg_ids
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
        assert not tracking.cancelled(), "turn released before commentary send settled"
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
