"""Waiting receipts cross native classification, delivery, and callback boundaries."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.commentary_delivery import schedule_cleanup
from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer
from gateway.turn_context import TurnContext
from tests.gateway.test_run_cleanup_progress import CleanupCaptureAdapter
from tools import async_delegation
from tools.delegate_waiting_reply import dispatch_identity, waiting_reply
from tools.registry import registry
import tools.delegate_tool  # native handler registration


@pytest.fixture
def turn():
    agent = SimpleNamespace(_current_turn_id='turn')
    identity = dispatch_identity(agent)
    unit = 'waiting-delivery-test'
    with async_delegation._records_lock:
        async_delegation._records[unit] = {'status': 'running', 'waiting_identity': identity}
    yield agent, unit
    with async_delegation._records_lock:
        async_delegation._records.pop(unit, None)


def classify(agent, unit):
    result = json.loads(registry.dispatch('delegate_task', {
        'action': 'classify_waiting_reply', 'delegation_ids': [unit],
    }, parent_agent=agent))
    assert result['status'] == 'classified'


def setup(agent):
    adapter = CleanupCaptureAdapter()
    guard = asyncio.Event()
    guard._hermes_run_generation = 1
    guard._hermes_final_delivery_ids = set()
    adapter._active_sessions['session'] = guard
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='chat', chat_type='dm', thread_id='thread')
    ctx = TurnContext(source=source, session_key='session', run_generation=1,
                      _cleanup_progress=True, _run_still_current=lambda: True)
    ctx.agent_holder[0] = agent
    from gateway.waiting_reply_delivery import begin_request_capture
    begin_request_capture(adapter, ctx)
    return adapter, guard, ctx


def receipt(complete=True):
    return SendResult(success=True, message_id='a', raw_response={'message_ids': ['a', 'b']},
                      continuation_message_ids=('c',), full_payload_delivered=complete,
                      fresh_message_ids=('a', 'b', 'c'))


def assert_record(adapter, agent, unit):
    key = ('session', 'chat', 'thread', waiting_reply(agent).root_request_id)
    assert adapter._waiting_reply_receipts == {
        key: {'message_ids': frozenset({'a', 'b', 'c'}), 'delegation_ids': (unit,)}}
    assert not adapter.deleted


@pytest.mark.asyncio
async def test_normal_real_delivery_records_split_receipts(turn):
    agent, unit = turn
    classify(agent, unit)
    adapter, guard, ctx = setup(agent)
    adapter._send_with_retry = AsyncMock(return_value=receipt())
    event = MessageEvent(text='question', message_type=MessageType.TEXT, source=ctx.source, message_id='incoming')
    async def handler(_event):
        schedule_cleanup({'final_response': 'Still waiting.'}, adapter, ctx)
        return 'Still waiting.'
    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, 'session')
    assert_record(adapter, agent, unit)


@pytest.mark.asyncio
async def test_streamed_real_split_receipts(turn):
    agent, unit = turn
    classify(agent, unit)
    adapter, guard, ctx = setup(agent)
    adapter.send = AsyncMock(return_value=receipt())
    consumer = GatewayStreamConsumer(adapter, 'chat')
    assert await consumer._try_fresh_final('Still waiting.')
    ctx.stream_consumer_holder[0] = consumer
    complete = consumer._confirmed_complete_body and consumer.delivered_final_matches('Still waiting.')
    assert complete
    schedule_cleanup({'final_response': 'Still waiting.', '_complete_final_body': complete}, adapter, ctx)
    await adapter._fire_post_delivery_callback('session', guard)
    assert_record(adapter, agent, unit)


@pytest.mark.asyncio
async def test_nontext_attempt_invalidates_normal_final(turn):
    agent, unit = turn
    classify(agent, unit)
    adapter, guard, ctx = setup(agent)
    adapter._send_with_retry = AsyncMock(return_value=receipt())
    async def attachment_attempt(event, extracted, metadata, *, anything_sent, record_delivery):
        record_delivery(SendResult(success=False, error='attachment failed'))
    adapter._deliver_attachments = attachment_attempt
    event = MessageEvent(text='question', message_type=MessageType.TEXT, source=ctx.source, message_id='incoming')
    async def handler(_event):
        schedule_cleanup({'final_response': 'Still waiting.'}, adapter, ctx)
        return 'Still waiting.'
    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, 'session')
    assert not getattr(adapter, '_waiting_reply_receipts', {})
    assert not adapter.deleted


@pytest.mark.asyncio
async def test_multiple_receipts_are_retained_without_eviction(turn):
    from gateway.platforms.post_delivery import record_final_ids
    agent, unit = turn
    classify(agent, unit)
    adapter, guard, ctx = setup(agent)
    for thread in ('one', 'two', 'three'):
        ctx.source.thread_id = thread
        schedule_cleanup({'final_response': 'Still waiting.'}, adapter, ctx)
        record_final_ids(guard, SendResult(success=True, message_id=thread,
                                            fresh_message_ids=(thread,)))
        guard._hermes_final_fresh_message_ids = {thread}
        adapter._confirm_final_delivery('session', 1, True)
        await adapter._fire_post_delivery_callback('session', guard)
    assert [key[2] for key in adapter._waiting_reply_receipts] == ['one', 'two', 'three']
    assert not adapter.deleted


@pytest.mark.asyncio
@pytest.mark.parametrize('complete', [False, None])
async def test_real_incomplete_normal_delivery_is_not_recorded(turn, complete):
    agent, unit = turn
    classify(agent, unit)
    adapter, guard, ctx = setup(agent)
    adapter._send_with_retry = AsyncMock(return_value=receipt(complete))
    event = MessageEvent(text='question', message_type=MessageType.TEXT, source=ctx.source, message_id='incoming')
    async def handler(_event):
        schedule_cleanup({'final_response': 'Still waiting.'}, adapter, ctx)
        return 'Still waiting.'
    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, 'session')
    assert not getattr(adapter, '_waiting_reply_receipts', {})
    assert not adapter.deleted


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['unclassified', 'disabled', 'failed', 'partial', 'interrupted',
                                  'incomplete', 'no_receipt', 'no_confirmation', 'stale_guard',
                                  'stale_generation', 'stale_turn', 'media', 'image', 'interrupted_guard'])
async def test_negative_controls(turn, case):
    agent, unit = turn
    if case != 'unclassified':
        classify(agent, unit)
    adapter, guard, ctx = setup(agent)
    response = {'final_response': 'Still waiting.'}
    if case == 'disabled':
        ctx._cleanup_progress = False
    if case in ('failed', 'partial', 'interrupted'):
        response[case] = True
    if case == 'incomplete':
        response['completed'] = False
    if case == 'media':
        response['final_response'] += ' MEDIA:/tmp/image.png'
    if case == 'image':
        response['final_response'] += ' ![image](https://example.org/img.png)'
    schedule_cleanup(response, adapter, ctx)
    if case != 'no_receipt':
        from gateway.platforms.post_delivery import record_final_ids
        record_final_ids(guard, receipt())
    if case != 'no_confirmation':
        adapter._confirm_final_delivery('session', 1, True)
    if case == 'stale_guard':
        adapter._active_sessions['session'] = asyncio.Event()
    if case == 'stale_generation':
        guard._hermes_run_generation = 2
    if case == 'stale_turn':
        agent._current_turn_id = 'another'
    if case == 'interrupted_guard':
        guard.set()
    await adapter._fire_post_delivery_callback('session', guard)
    assert not getattr(adapter, '_waiting_reply_receipts', {})
    assert not adapter.deleted
