"""Positive creation evidence survives Telegram and stream receipt boundaries."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.stream_consumer import GatewayStreamConsumer
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.fixture
def adapter(monkeypatch):
    obj = TelegramAdapter(PlatformConfig(enabled=True, token='fake'))
    obj._bot = MagicMock()
    obj._rich_send_disabled = True
    obj._retrigger_typing = AsyncMock()
    obj._chat_outbound_slot_remaining = lambda *_: 0
    obj._hold_chat_outbound_slot = lambda *_: None
    monkeypatch.setattr('plugins.platforms.telegram.adapter.asyncio.sleep', AsyncMock())
    return obj


@pytest.mark.asyncio
async def test_send_edit_and_overflow_receipts(adapter):
    adapter._bot.send_message = AsyncMock(side_effect=[SimpleNamespace(message_id=11),
                                                       SimpleNamespace(message_id=12),
                                                       SimpleNamespace(message_id=13)])
    sent = await adapter.send('1', 'hello')
    assert sent.message_id == '11' and sent.fresh_message_ids == ('11',)
    adapter._bot.edit_message_text = AsyncMock(return_value=True)
    edited = await adapter.edit_message('1', '11', 'edited', finalize=True)
    assert edited.success and edited.fresh_message_ids == ()
    adapter.MAX_MESSAGE_LENGTH = 160
    result = await adapter._edit_overflow_split('1', '11', 'word ' * 70, finalize=False)
    assert result.success and result.fresh_message_ids == ('12', '13')
    assert '11' not in result.fresh_message_ids


@pytest.mark.asyncio
async def test_partial_and_resumed_send_preserve_only_positive_evidence(adapter):
    adapter.MAX_MESSAGE_LENGTH = 160
    calls = 0

    async def send(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('failure')
        return SimpleNamespace(message_id=100 + calls)

    adapter._bot.send_message = AsyncMock(side_effect=send)
    partial = await adapter.send('1', 'word ' * 70)
    assert not partial.success and partial.fresh_message_ids == ('101',)
    assert partial.raw_response['undelivered_chunks']
    result = await adapter._resume_partial_send('1', partial, reply_to=None, metadata=None)
    assert result.success
    assert result.fresh_message_ids == tuple(result.raw_response['message_ids'])
    assert result.fresh_message_ids[0] == '101'
    # Raw legacy IDs are not proof, even when resuming a legacy partial receipt.
    partial.fresh_message_ids = ()
    again = await adapter._resume_partial_send('1', partial, reply_to=None, metadata=None)
    assert '101' not in again.fresh_message_ids


@pytest.mark.asyncio
async def test_partial_overflow_retains_only_new_continuations(adapter):
    adapter.MAX_MESSAGE_LENGTH = 160
    adapter._bot.edit_message_text = AsyncMock(return_value=True)
    adapter._send_overflow_continuation = AsyncMock(side_effect=[SimpleNamespace(message_id=22), None])
    result = await adapter._edit_overflow_split('1', '21', 'word ' * 100, finalize=False)
    assert not result.success and result.fresh_message_ids == ('22',)
    assert '21' not in result.fresh_message_ids


@pytest.mark.asyncio
@pytest.mark.parametrize('response,expected', [({'message_id': 31}, ('31',)), ({'ok': True}, ())])
async def test_rich_send_only_reports_returned_id(adapter, response, expected):
    adapter._bot.do_api_request = AsyncMock(return_value=response)
    adapter._record_rich_sent = AsyncMock()
    result = await adapter._try_send_rich('1', 'hello', None, None)
    assert result.success and result.fresh_message_ids == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('prior_boundary', [False, True])
@pytest.mark.parametrize('evidence', [('41',), ()])
async def test_streamed_commentary_propagates_receipt_without_inference(prior_boundary, evidence):
    received = []
    transport = SimpleNamespace(
        send=AsyncMock(return_value=SendResult(success=True, message_id='41', fresh_message_ids=evidence)),
        edit_message=AsyncMock(return_value=SendResult(success=True, message_id='41')),
    )
    consumer = GatewayStreamConsumer(transport, '1', on_commentary_delivered=lambda r, t: received.append((r, t)))
    assert await consumer._send_or_edit('working', finalize=False)
    assert await consumer._send_or_edit('working now', finalize=False)
    consumer._stream_ledger = 'working now'
    if prior_boundary:
        consumer._reset_segment_state()
    consumer._register_streamed_commentary('working now')
    result, text = received[0]
    assert text == 'working now' and result.continuation_message_ids == ('41',)
    assert result.fresh_message_ids == evidence


def test_default_does_not_infer_creation():
    assert SendResult(success=True, message_id='old').fresh_message_ids == ()


@pytest.mark.asyncio
async def test_telegram_send_and_edit_reach_streamed_commentary_callback(adapter):
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=51))
    adapter._bot.edit_message_text = AsyncMock(return_value=True)
    receipts = []
    consumer = GatewayStreamConsumer(adapter, '1', on_commentary_delivered=lambda r, t: receipts.append(r))
    assert await consumer._send_or_edit('working', finalize=False)
    assert await consumer._send_or_edit('working now', finalize=False)
    consumer._stream_ledger = 'working now'
    consumer._reset_segment_state()
    consumer._register_streamed_commentary('working now')
    assert receipts[0].continuation_message_ids == ('51',)
    assert receipts[0].fresh_message_ids == ('51',)


@pytest.mark.asyncio
async def test_partial_edit_receipt_reaches_stream_identity(adapter):
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=61))
    consumer = GatewayStreamConsumer(adapter, '1')
    assert await consumer._send_or_edit('working', finalize=False)
    adapter.edit_message = AsyncMock(return_value=SendResult(
        success=False, message_id='62', continuation_message_ids=('62',),
        fresh_message_ids=('62',), raw_response={'partial_overflow': True, 'last_message_id': '62'}))
    assert not await consumer._send_or_edit('working longer', finalize=False)
    assert consumer._segment_preview_message_ids == {'61', '62'}
    assert consumer._segment_fresh_message_ids == {'61', '62'}


@pytest.mark.asyncio
async def test_real_stream_queue_carries_telegram_receipt_to_commentary(adapter):
    sent = asyncio.Event()

    async def send(**kwargs):
        sent.set()
        return SimpleNamespace(message_id=71)

    adapter._bot.send_message = AsyncMock(side_effect=send)
    adapter._bot.edit_message_text = AsyncMock(return_value=True)
    receipts = []
    consumer = GatewayStreamConsumer(adapter, '1', on_commentary_delivered=lambda r, t: receipts.append((r, t)))
    consumer.cfg.buffer_threshold = 1
    consumer.on_delta('working')
    task = asyncio.create_task(consumer.run())
    await asyncio.wait_for(sent.wait(), 5)
    consumer.on_streamed_commentary('working')
    consumer.finish()
    await asyncio.wait_for(task, 5)
    assert len(receipts) == 1
    assert receipts[0][1] == 'working'
    assert receipts[0][0].fresh_message_ids == ('71',)
    assert receipts[0][0].continuation_message_ids == ('71',)
