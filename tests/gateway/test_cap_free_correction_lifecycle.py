"""Lifecycle regressions for request-owned cleanup and queued finals."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.commentary_delivery import edit_streamed_final, track_progress
from gateway.platforms.base import SendResult
from gateway.stream_consumer import GatewayStreamConsumer
from tests.gateway.test_run_cleanup_progress import _make_runner
from tests.gateway.test_waiting_review_safety import owned, delivered, other_turn


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['reconcile', 'fresh', 'fallback', 'empty', 'first', 'edit', 'chunk'])
async def test_failed_final_receipts_protect_commentary(owned, path):
    _, _, _, adapter, guard, ctx = owned
    track_progress(ctx, SendResult(success=True, message_id='alias', fresh_message_ids=('alias',)))
    other_guard, other_ctx = other_turn(adapter, ctx)
    consumer = GatewayStreamConsumer(adapter, 'chat')
    consumer._message_id = 'preview'
    other_ctx.stream_consumer_holder[0] = consumer
    failed = SendResult(success=False, message_id='alias', continuation_message_ids=('alias',),
                        fresh_message_ids=('alias',), raw_response={'partial_overflow': True})
    adapter.edit_message = AsyncMock(return_value=failed)
    adapter.send = AsyncMock(return_value=failed)
    adapter.prefers_fresh_final_streaming = lambda _: False
    if path == 'reconcile':
        await edit_streamed_final(consumer, other_ctx.source, {}, 'Answer', _sk='other',
                                  ok=('ok',), fail_result='failed %s %s', fail_exc='exception %s %s')
    elif path == 'fresh':
        await consumer._try_fresh_final('Answer')
    elif path == 'fallback':
        await consumer._send_fallback_final('Answer')
    elif path == 'empty':
        await consumer._send_empty_fallback_final('Answer')
    elif path == 'first':
        await consumer._first_send('Answer', finalize=True)
    elif path == 'edit':
        await consumer._edit_existing('Answer', finalize=True, is_turn_final=True)
    else:
        await consumer._send_new_chunk('Answer', None, final=True)
    assert 'alias' not in consumer._final_fresh_message_ids
    await delivered(adapter, other_guard, other_ctx, 'Fallback.', ['fallback'])
    await delivered(adapter, guard, ctx, 'Local answer.', ['local'])
    assert 'alias' not in [row['message_id'] for row in adapter.deleted]


@pytest.mark.asyncio
async def test_cancelled_queued_delivery_retires_barrier_and_preserves_overlap(owned, monkeypatch):
    _, _, _, adapter, guard, ctx = owned
    track_progress(ctx, SendResult(success=True, message_id='alias', fresh_message_ids=('alias',)))
    queued_guard, queued_ctx = other_turn(adapter, ctx)
    runner = _make_runner(adapter)
    monkeypatch.setattr(runner, '_record_room_silence_decision', AsyncMock())
    monkeypatch.setattr(runner, '_run_agent_stream_confirmed_final_delivery', lambda *a, **k: False)
    monkeypatch.setattr(runner, '_is_intentional_silence', lambda *a: False)
    started = asyncio.Event()
    async def send(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()
    adapter.send_final_ledgered = send
    task = asyncio.create_task(runner._run_agent_deliver_first_response(
        queued_ctx, adapter, {'final_response': 'Answer'}, {}, None))
    await asyncio.wait_for(started.wait(), 2)
    assert adapter._waiting_pending_finals
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not adapter._waiting_pending_finals
    assert not queued_guard._hermes_delivery_in_flight
    await delivered(adapter, guard, ctx, 'Local answer.', ['local'])
    assert 'alias' not in [row['message_id'] for row in adapter.deleted]


@pytest.mark.asyncio
async def test_queued_completion_preserves_newer_callback_slot(owned, monkeypatch):
    _, _, _, adapter, _, ctx = owned
    queued_guard, queued_ctx = other_turn(adapter, ctx)
    runner = _make_runner(adapter)
    monkeypatch.setattr(runner, '_record_room_silence_decision', AsyncMock())
    monkeypatch.setattr(runner, '_run_agent_stream_confirmed_final_delivery', lambda *a, **k: False)
    monkeypatch.setattr(runner, '_is_intentional_silence', lambda *a: False)
    monkeypatch.setattr(runner, '_deliver_media_from_response', AsyncMock())
    started, release = asyncio.Event(), asyncio.Event()
    async def send(*args, **kwargs):
        started.set()
        await release.wait()
        return SendResult(success=True, message_id='answer', full_payload_delivered=True), None
    adapter.send_final_ledgered = send
    task = asyncio.create_task(runner._run_agent_deliver_first_response(
        queued_ctx, adapter, {'final_response': 'Answer'}, {}, None))
    await asyncio.wait_for(started.wait(), 2)
    newer = AsyncMock()
    queued_guard._hermes_run_generation = 2
    adapter.register_post_delivery_callback(queued_ctx.session_key, newer, generation=2)
    release.set()
    await asyncio.wait_for(task, 2)
    assert not adapter._waiting_pending_finals
    assert not queued_guard._hermes_delivery_in_flight
    assert adapter.pop_post_delivery_callback(queued_ctx.session_key, generation=2) is newer
    newer.assert_not_called()
