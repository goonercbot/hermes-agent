"""Consumer lifetimes and native callback ownership use isolated transports."""
import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import session_ingress
from gateway.commentary_delivery import schedule_cleanup, track_progress, track_commentary
from gateway.config import Platform
from gateway.platforms.base import SendResult
from tests.gateway.test_waiting_reply_delivery import setup
from tests.gateway.test_run_cleanup_progress import _make_runner


@pytest.mark.asyncio
async def test_2001_ordinary_commentaries_release_each_lifecycle():
    from gateway.turn_context import TurnContext
    agent = SimpleNamespace(_current_turn_id='turn')
    adapter, guard, original = setup(agent)
    for index in range(2001):
        ctx = TurnContext(source=original.source, session_key='session', run_generation=1,
                          _cleanup_progress=True, _run_still_current=lambda: True)
        ctx.agent_holder[0] = agent
        mid = f'progress-{index}'
        track_commentary(ctx, SendResult(success=True, message_id=mid, fresh_message_ids=(mid,)), 'working')
        schedule_cleanup({'final_response': 'answer'}, adapter, ctx)
        guard._hermes_final_delivery_ids = {f'answer-{index}'}
        adapter._confirm_final_delivery('session', 1, True)
        await adapter._fire_post_delivery_callback('session', guard)
        assert not adapter._waiting_pending_finals
        assert not guard._waiting_live_captures
    assert len(adapter.deleted) == 2001
    assert adapter.deleted[-1]['message_id'] == 'progress-2000'
    assert not adapter._waiting_reply_receipts


@pytest.mark.asyncio
async def test_non_telegram_ordinary_receipt_but_telegram_unknown_not_deleted():
    adapter, guard, ctx = setup(SimpleNamespace(_current_turn_id='turn'))
    track_progress(ctx, SendResult(success=True, message_id='unknown'))
    assert not ctx._cleanup_msg_ids
    ctx.source.platform = Platform.DISCORD
    track_progress(ctx, SendResult(success=True, message_id='ordinary'))
    schedule_cleanup({'final_response': 'answer'}, adapter, ctx)
    guard._hermes_final_delivery_ids = {'answer'}
    adapter._confirm_final_delivery('session', 1, True)
    await adapter._fire_post_delivery_callback('session', guard)
    assert [row['message_id'] for row in adapter.deleted] == ['ordinary']


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['cancel', 'replace'])
async def test_authority_native_taken_callback_disposes_own_generation(monkeypatch, mode):
    adapter, guard, ctx = setup(SimpleNamespace(_current_turn_id='turn'))
    async def cancel():
        raise asyncio.CancelledError()
    adapter.register_post_delivery_callback('session', cancel, generation=1)
    ctx._cleanup_msg_ids.append('progress')
    schedule_cleanup({'final_response': 'answer'}, adapter, ctx)
    authority = SimpleNamespace(runner=SimpleNamespace(_adapter_for_source=lambda _: adapter), native_waiters=set())
    session_ingress._take_post_delivery(authority, 'admission', ctx.source, 'session')
    if mode == 'replace':
        adapter.register_post_delivery_callback('session', lambda: None, generation=2)
    async def scope(*_):
        return nullcontext()
    monkeypatch.setattr(session_ingress, '_execution_scope', scope)
    with pytest.raises(asyncio.CancelledError):
        await session_ingress.fire_post_delivery(authority, 'admission')
    assert not adapter._waiting_pending_finals
    assert not guard._waiting_live_captures
    if mode == 'replace':
        assert adapter.pop_post_delivery_callback('session', generation=2)


@pytest.mark.asyncio
async def test_queued_early_pop_cancellation_disposes_pending(monkeypatch):
    adapter, guard, ctx = setup(SimpleNamespace(_current_turn_id='turn'))
    async def cancel():
        raise asyncio.CancelledError()
    adapter.register_post_delivery_callback('session', cancel, generation=1)
    ctx._cleanup_msg_ids.append('progress')
    schedule_cleanup({'final_response': 'answer'}, adapter, ctx)
    runner = _make_runner(adapter)
    monkeypatch.setattr(runner, '_record_room_silence_decision', AsyncMock())
    monkeypatch.setattr(runner, '_run_agent_stream_confirmed_final_delivery', lambda *a, **k: False)
    monkeypatch.setattr(runner, '_is_intentional_silence', lambda *a: False)
    adapter.send_final_ledgered = AsyncMock(return_value=(SendResult(
        success=True, message_id='queued-final', full_payload_delivered=True), None))
    monkeypatch.setattr(runner, '_deliver_media_from_response', AsyncMock())
    with pytest.raises(asyncio.CancelledError):
        await runner._run_agent_deliver_first_response(ctx, adapter, {'final_response': 'answer'}, {}, None)
    assert adapter.send_final_ledgered.await_count == 1
    assert not adapter._waiting_pending_finals
    assert not guard._waiting_live_captures
    assert not guard._hermes_final_receipt_uncertain


@pytest.mark.asyncio
async def test_authority_real_delivery_captures_fresh_waiting_receipt(monkeypatch):
    from gateway.platforms.event import MessageEvent, MessageType
    from tests.gateway.test_waiting_reply_delivery import classify
    from tools import async_delegation
    from tools.delegate_waiting_reply import dispatch_identity
    agent = SimpleNamespace(_current_turn_id='turn', session_id='sid')
    identity = dispatch_identity(agent)
    monkeypatch.setattr(async_delegation, '_records', {'unit': {
        'status': 'running', 'waiting_identity': identity,
        'parent_session_id': 'sid', 'session_key': 'session'}})
    classify(agent, 'unit')
    adapter, guard, ctx = setup(agent)
    adapter._send_with_retry = AsyncMock(return_value=SendResult(
        success=True, message_id='wait', fresh_message_ids=('wait',), full_payload_delivered=True))
    schedule_cleanup({'final_response': 'Waiting.'}, adapter, ctx)
    event = MessageEvent(text='question', message_type=MessageType.TEXT, source=ctx.source, message_id='in')
    authority = SimpleNamespace(runner=SimpleNamespace(_adapter_for_source=lambda _: adapter),
                                native_waiters=set(), pending_deliveries={})
    session_ingress._take_post_delivery(authority, 'admission', ctx.source, 'session')
    authority.pending_deliveries['admission'] = (adapter, event, 'session', 'Waiting.', None)
    async def scope(*_):
        return nullcontext()
    monkeypatch.setattr(session_ingress, '_execution_scope', scope)
    monkeypatch.setattr(session_ingress, '_release_turn_marker', AsyncMock())
    monkeypatch.setattr(session_ingress, 'close_producer_turn', AsyncMock())
    await session_ingress.deliver_settled(authority, 'admission')
    await session_ingress.fire_post_delivery(authority, 'admission')
    assert next(iter(adapter._waiting_reply_receipts.values()))['message_ids'] == {'wait'}
    assert not guard._waiting_live_captures
