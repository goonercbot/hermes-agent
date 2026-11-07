"""Reviewer-owned acceptance probes; no network or implementation mutations."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from gateway.commentary_delivery import schedule_cleanup, track_progress, edit_streamed_final
from gateway.platforms.base import SendResult
from gateway.stream_consumer import GatewayStreamConsumer
from tests.gateway.test_waiting_review_safety import owned, delivered, resume, other_turn
from tests.gateway.test_waiting_reply_delivery import classify
from tests.gateway.test_run_cleanup_progress import _make_runner


@pytest.mark.asyncio
@pytest.mark.parametrize('final_before_schedule', [False, True])
async def test_commentary_creation_before_concurrent_final_keeps_final(owned, final_before_schedule):
    agent, _, _, adapter, guard, ctx = owned
    # Positive fresh evidence is genuine: this ID was first a temporary send.
    track_progress(ctx, SendResult(success=True, message_id='alias', fresh_message_ids=('alias',)))
    if not final_before_schedule:
        schedule_cleanup({'final_response': 'Local answer.'}, adapter, ctx)
    other_guard, other_ctx = other_turn(adapter, ctx)
    # The other request now reuses/edits the former temporary message as a final.
    adapter._test_seen_ids = {'alias'}
    await delivered(adapter, other_guard, other_ctx, 'Genuine answer.', ['alias'])
    if final_before_schedule:
        schedule_cleanup({'final_response': 'Local answer.'}, adapter, ctx)
    guard._hermes_final_delivery_ids = {'local-final'}
    adapter._confirm_final_delivery('session', 1, True)
    await adapter._fire_post_delivery_callback('session', guard)
    deleted = [item['message_id'] for item in adapter.deleted]
    print('late commentary capture', final_before_schedule, 'deleted', deleted)
    assert 'alias' not in deleted


@pytest.mark.asyncio
async def test_streamed_wait_created_before_concurrent_final_cannot_be_readmitted(owned):
    agent, _, _, adapter, guard, ctx = owned
    classify(agent, 'unit')
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id='alias',
        fresh_message_ids=('alias',), full_payload_delivered=True))
    consumer = GatewayStreamConsumer(adapter, 'chat')
    assert await consumer._try_fresh_final('Waiting.')
    ctx.stream_consumer_holder[0] = consumer
    other_guard, other_ctx = other_turn(adapter, ctx)
    adapter._test_seen_ids = {'alias'}
    await delivered(adapter, other_guard, other_ctx, 'Genuine answer.', ['alias'])
    # This is the production order: _run_agent schedules after stream completion.
    schedule_cleanup({'final_response': 'Waiting.', '_complete_final_body': True}, adapter, ctx)
    await adapter._fire_post_delivery_callback('session', guard)
    ctx.stream_consumer_holder[0] = None
    resume(owned)
    await delivered(adapter, guard, ctx, 'Resolved.', ['resolution'])
    deleted = [item['message_id'] for item in adapter.deleted]
    print('late waiting stream capture deleted', deleted)
    assert 'alias' not in deleted


@pytest.mark.asyncio
@pytest.mark.parametrize('cleanup_kind', ['waiting', 'commentary'])
@pytest.mark.parametrize('delivery_phase', ['inflight', 'completed'])
async def test_real_queued_send_inflight_blocks_other_root_cleanup(owned, monkeypatch, cleanup_kind, delivery_phase):
    agent, _, _, adapter, guard, ctx = owned
    if cleanup_kind == 'waiting':
        classify(agent, 'unit')
        await delivered(adapter, guard, ctx, 'Waiting.', ['alias'])
    else:
        track_progress(ctx, SendResult(success=True, message_id='alias', fresh_message_ids=('alias',)))
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
        return SendResult(success=True, message_id='alias', full_payload_delivered=True), None
    adapter.send_final_ledgered = send
    # No pre-registered cleanup: _run_agent's queued return precedes schedule_cleanup.
    task = asyncio.create_task(runner._run_agent_deliver_first_response(
        queued_ctx, adapter, {'final_response': 'Queued genuine answer.'}, {}, None))
    try:
        await asyncio.wait_for(started.wait(), 2)
        assert queued_guard._hermes_delivery_in_flight
        if delivery_phase == 'completed':
            release.set()
            await asyncio.wait_for(task, 2)
        if cleanup_kind == 'waiting':
            resume(owned)
        await delivered(adapter, guard, ctx, 'Resolved.', ['resolution'])
    finally:
        release.set()
        await asyncio.wait_for(task, 2)
    deleted = [item['message_id'] for item in adapter.deleted]
    print('queued inflight final deleted', deleted)
    assert 'alias' not in deleted


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['reconcile', 'fresh-final', 'fallback-final', 'empty-fallback', 'first-send', 'edit-existing', 'new-final-chunk'])
async def test_failed_final_receipt_ids_protect_final_aliases(owned, path):
    agent, _, _, adapter, guard, ctx = owned
    classify(agent, 'unit')
    await delivered(adapter, guard, ctx, 'Waiting.', ['alias'])
    other_guard, other_ctx = other_turn(adapter, ctx)
    consumer = GatewayStreamConsumer(adapter, 'chat')
    consumer._message_id = 'preview'
    other_ctx.stream_consumer_holder[0] = consumer
    failed = SendResult(success=False, message_id='alias', continuation_message_ids=('alias',),
        raw_response={'partial_overflow': True}, error='partial')
    adapter.edit_message = AsyncMock(return_value=failed)
    adapter.send = AsyncMock(return_value=failed)
    adapter.prefers_fresh_final_streaming = lambda _: False
    if path == 'reconcile':
        await edit_streamed_final(consumer, other_ctx.source, {}, 'Genuine answer.',
            _sk='other-session', ok=('ok',), fail_result='failed %s %s', fail_exc='exception %s %s')
    elif path == 'fresh-final':
        await consumer._try_fresh_final('Genuine answer.')
    elif path == 'fallback-final':
        await consumer._send_fallback_final('Genuine answer.')
    elif path == 'empty-fallback':
        await consumer._send_empty_fallback_final('Genuine answer.')
    elif path == 'first-send':
        await consumer._first_send('Genuine answer.', finalize=True)
    elif path == 'edit-existing':
        await consumer._edit_existing('Genuine answer.', finalize=True, is_turn_final=True)
    else:
        await consumer._send_new_chunk('Genuine answer.', None, final=True)
    await delivered(adapter, other_guard, other_ctx, 'Genuine fallback answer.', ['fallback'])
    resume(owned)
    await delivered(adapter, guard, ctx, 'Resolved.', ['resolution'])
    deleted = [item['message_id'] for item in adapter.deleted]
    print('failed reconciliation alias deleted', deleted)
    assert 'alias' not in deleted
