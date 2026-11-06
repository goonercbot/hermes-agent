"""Independent correction probes; synthetic transport only."""
import asyncio
import threading

import pytest

from gateway.config import Platform, StreamingConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from tests.gateway.test_commentary_cleanup_integration import CommentaryAgent
from tests.gateway.test_run_cleanup_progress import CleanupCaptureAdapter, _install_fakes, _make_runner


@pytest.mark.asyncio
@pytest.mark.parametrize("reuse", [True, False])
async def test_fallback_split_final_receipts_survive(monkeypatch, tmp_path, reuse):
    await _fallback_split_final_reused_ids_survive(monkeypatch, tmp_path, reuse)


async def _fallback_split_final_reused_ids_survive(monkeypatch, tmp_path, reuse=True):
    commentary_seen, preview_seen = threading.Event(), threading.Event()
    final = 'the start of the final answer'

    class Adapter(CleanupCaptureAdapter):
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            if content.startswith('working on it'):
                mid, raw, continuation = 'comment-1', {'message_ids': ['comment-1', 'comment-2']}, ('comment-3',)
                commentary_seen.set()
            elif content.startswith('the start'):
                mid, raw, continuation = 'final-preview', None, ()
                preview_seen.set()
            else:
                mid = 'final-tail'
                raw = {'message_ids': ['final-tail', 'comment-2' if reuse else 'final-2']}
                continuation = ('comment-3' if reuse else 'final-3',)
            self.sent.append({'content': content, 'message_id': mid, 'raw': raw, 'continuation': continuation})
            return SendResult(success=True, message_id=mid, raw_response=raw,
                              continuation_message_ids=continuation, full_payload_delivered=True)

        async def edit_message(self, chat_id, message_id, content, **kwargs):
            self.edits.append({'content': content, 'message_id': message_id})
            if message_id == 'final-preview':
                return SendResult(success=False, message_id=message_id, error='editing unsupported')
            return SendResult(success=True, message_id=message_id, full_payload_delivered=True)

    class Agent(CommentaryAgent):
        def run_conversation(self, *args, **kwargs):
            self.stream_delta_callback('working on it')
            assert commentary_seen.wait(5)
            self.interim_assistant_callback('working on it', already_streamed=True)
            self.stream_delta_callback('the start ')
            assert preview_seen.wait(5)
            self.stream_delta_callback('of the final answer')
            return {'final_response': final, 'messages': [], 'api_calls': 1}

    adapter = Adapter()
    runner = _make_runner(adapter)
    runner.config.streaming = StreamingConfig(enabled=True, buffer_threshold=1)
    gateway_run = _install_fakes(monkeypatch, Agent, cleanup_on=True)
    monkeypatch.setattr(gateway_run, '_hermes_home', tmp_path)
    monkeypatch.setattr(gateway_run, '_load_gateway_config', lambda: {
        'display': {'tool_progress': 'off', 'interim_assistant_messages': True,
                    'platforms': {'telegram': {'streaming': True, 'cleanup_progress': True}}}})
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='4242', chat_type='dm')
    event = MessageEvent(text='question', message_type=MessageType.TEXT, source=source, message_id='incoming')
    key, generation = 'fallback-final-recheck', 1
    guard = asyncio.Event()
    guard._hermes_run_generation = generation
    adapter._active_sessions[key] = guard
    runner._session_run_generation[key] = generation
    results = []

    async def handler(_event):
        result = await runner._run_agent(message='question', context_prompt='', history=[], source=source,
                                         session_id=key, session_key=key, run_generation=generation)
        results.append(result)
        return None if result.get('already_sent') else result['final_response']

    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, key)
    assert any(row['message_id'] == 'final-tail' for row in adapter.sent), 'fallback final not exercised'
    assert results[0].get('_complete_final_body') is True
    protected = {'comment-2', 'comment-3'} if reuse else {'final-2', 'final-3'}
    deleted = {row['message_id'] for row in adapter.deleted}
    assert not (protected & deleted), 'deleted final split receipts'
    if not reuse:
        assert deleted == {'comment-1', 'comment-2', 'comment-3'}


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["empty_fallback", "fresh_final", "reconciliation"])
async def test_final_receipt_siblings_protect_split_ids(lane):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from gateway.commentary_delivery import edit_streamed_final, schedule_cleanup
    from gateway.stream_consumer import GatewayStreamConsumer
    from gateway.turn_context import TurnContext

    adapter = CleanupCaptureAdapter()
    receipt = SendResult(success=True, message_id='final',
                         raw_response={'message_ids': ['final', 'comment-2']},
                         continuation_message_ids=('comment-3',), full_payload_delivered=True)
    adapter.send = AsyncMock(return_value=receipt)
    adapter.edit_message = AsyncMock(return_value=receipt)
    consumer = GatewayStreamConsumer(adapter, '4242')
    consumer._message_id = 'preview'
    consumer._segment_preview_message_ids = {'preview', 'comment-2', 'comment-3'}
    consumer._preview_message_ids = {'preview', 'comment-2', 'comment-3'}
    if lane == 'empty_fallback':
        assert await consumer._send_empty_fallback_final('the answer') == 'delivered'
    elif lane == 'fresh_final':
        assert await consumer._try_fresh_final('the answer')
    else:
        response = {}
        assert await edit_streamed_final(consumer, SimpleNamespace(chat_id='4242'), response,
            'the answer', _sk='test', ok=('edited',), fail_result='%s %s', fail_exc='%s %s')
        assert response['_complete_final_body'] is True

    assert not ({'comment-2', 'comment-3'} & {row['message_id'] for row in adapter.deleted})
    adapter.deleted.clear()
    key = 'sibling'
    guard = asyncio.Event()
    guard._hermes_run_generation = 1
    guard._hermes_final_delivery_ids = set()
    adapter._active_sessions[key] = guard
    ctx = TurnContext(_cleanup_progress=True, _cleanup_msg_ids=['comment-1', 'comment-2', 'comment-3'],
        session_key=key, source=SimpleNamespace(chat_id='4242'), run_generation=1)
    ctx.stream_consumer_holder[0] = consumer
    schedule_cleanup({'final_response': 'the answer', '_complete_final_body': True}, adapter, ctx)
    await adapter._fire_post_delivery_callback(key, guard)
    assert {row['message_id'] for row in adapter.deleted} == {'comment-1'}


@pytest.mark.asyncio
async def test_each_fallback_chunk_receipt_survives():
    from unittest.mock import AsyncMock
    from gateway.stream_consumer import GatewayStreamConsumer

    adapter = CleanupCaptureAdapter()
    adapter.send = AsyncMock(side_effect=[
        SendResult(success=True, message_id='first', raw_response={'message_ids': ['first', 'reused-1']},
                   continuation_message_ids=('reused-2',), full_payload_delivered=True),
        SendResult(success=True, message_id='last', raw_response={'message_ids': ['last', 'reused-3']},
                   continuation_message_ids=('reused-4',), full_payload_delivered=True),
    ])
    consumer = GatewayStreamConsumer(adapter, '4242')
    consumer._split_text_chunks = lambda *args, **kwargs: ['chunk one', 'chunk two']
    await consumer._send_fallback_final('complete answer')
    assert adapter.send.await_count == 2
    assert consumer._confirmed_complete_body is True
    assert consumer._final_receipt_ids == {'first', 'last', 'reused-1', 'reused-2', 'reused-3', 'reused-4'}


@pytest.mark.asyncio
@pytest.mark.parametrize('lane', ['first_send', 'final_edit', 'split_head'])
async def test_consumer_final_send_and_edit_receipts(lane):
    from unittest.mock import AsyncMock
    from gateway.stream_consumer import GatewayStreamConsumer

    adapter = CleanupCaptureAdapter()
    receipt = SendResult(success=True, message_id='primary',
                         raw_response={'message_ids': ['primary', 'raw']},
                         continuation_message_ids=('continuation',), full_payload_delivered=True)
    adapter.send = AsyncMock(return_value=receipt)
    adapter.edit_message = AsyncMock(return_value=receipt)
    consumer = GatewayStreamConsumer(adapter, '4242')
    if lane == 'first_send':
        assert await consumer._first_send('the answer', finalize=True)
    elif lane == 'final_edit':
        consumer._message_id = 'preview'
        assert await consumer._edit_existing('the answer', finalize=True, is_turn_final=True)
    else:
        assert await consumer._send_new_chunk('the answer', None, final=True) == 'primary'
    assert consumer._final_receipt_ids == {'primary', 'raw', 'continuation'}
