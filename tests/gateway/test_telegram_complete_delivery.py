"""Complete-body receipts come from the real Telegram transport paths."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.config import Platform
from gateway.commentary_delivery import schedule_cleanup
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.post_delivery import complete_body
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr('plugins.platforms.telegram.adapter.asyncio.sleep', AsyncMock())
    instance = TelegramAdapter(PlatformConfig(enabled=True, token='synthetic', typing_indicator=False))
    instance._rich_send_disabled = True
    instance._bot = MagicMock()
    instance._bot.send_message = AsyncMock(return_value=MagicMock(message_id=101))
    instance._bot.edit_message_text = AsyncMock(return_value=MagicMock(message_id=100))
    return instance


@pytest.mark.asyncio
async def test_final_overflow_and_partial_continuation(adapter):
    text = 'word ' * 2000
    result = await adapter.edit_message('4242', '100', text, finalize=True)
    assert complete_body(result)
    assert adapter._bot.send_message.await_count >= 1
    adapter._bot.send_message = AsyncMock(side_effect=Exception('refused'))
    partial = await adapter.edit_message('4242', '100', text, finalize=True)
    assert not complete_body(partial)
    assert partial.raw_response['partial_overflow']


@pytest.mark.asyncio
async def test_skipped_clipped_and_unknown_are_not_complete(adapter):
    assert not complete_body(await adapter.send('4242', '  '))
    adapter._chat_outbound_slot_remaining = lambda _: 1
    skipped = await adapter.edit_message('4242', '100', 'new', finalize=False)
    assert skipped.raw_response['skipped'] and not complete_body(skipped)
    adapter._chat_outbound_slot_remaining = lambda _: 0
    clipped = await adapter.edit_message('4242', '100', 'word ' * 2000, finalize=False)
    assert not complete_body(clipped)
    fallback = await adapter._send_plain_fallback('4242', 'word ' * 1000, reply_to=None, metadata=None)
    assert fallback.success and not complete_body(fallback)


@pytest.mark.asyncio
async def test_rich_send_and_final_edit_real_api_boundary(adapter):
    adapter._rich_send_disabled = False
    adapter._rich_messages_enabled = True
    adapter._record_rich_sent = AsyncMock()
    adapter._bot.do_api_request = AsyncMock(return_value={'message_id': 101})
    content = '| a | b |\n|---|---|\n| 1 | 2 |'
    sent = await adapter.send('4242', content)
    assert complete_body(sent)
    edited = await adapter.edit_message('4242', '100', content, finalize=True)
    assert complete_body(edited)
    assert adapter._bot.do_api_request.await_count == 2


@pytest.mark.asyncio
async def test_plain_edit_fallback_and_not_modified(adapter):
    adapter._bot.edit_message_text = AsyncMock(side_effect=[Exception('bad markup'), MagicMock()])
    assert complete_body(await adapter.edit_message('4242', '100', 'answer', finalize=True))
    adapter._bot.edit_message_text = AsyncMock(side_effect=Exception('message is not modified'))
    assert complete_body(await adapter.edit_message('4242', '100', 'answer', finalize=True))


@pytest.mark.asyncio
async def test_split_retry_receipt_only_after_tail_lands(adapter, monkeypatch):
    class Flood(Exception):
        retry_after = 7.0

    calls = []
    async def send(text, **kwargs):
        calls.append(text)
        if len(calls) == 2:
            raise Flood('flood retry after 7')
        return MagicMock(message_id=100 + len(calls))

    adapter._bot.send_message = AsyncMock(side_effect=send)
    async def elapsed(_delay):
        adapter._telegram_send_cooldown_until.clear()
    monkeypatch.setattr('gateway.platforms.base.asyncio.sleep', elapsed)
    result = await adapter._send_with_retry(chat_id='4242', content='word ' * 2000)
    assert complete_body(result)
    assert calls.count(calls[0]) == 1


@pytest.mark.asyncio
async def test_partial_final_preserves_progress_in_background(adapter):
    class Flood(Exception):
        retry_after = 120.0
    adapter._bot.send_message = AsyncMock(side_effect=[MagicMock(message_id=100), Flood('flood retry after 120')])
    adapter.delete_message = AsyncMock(return_value=True)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='4242', chat_type='dm')
    event = MessageEvent(text='question', message_type=MessageType.TEXT, source=source, message_id='incoming')
    key = 'partial-cleanup'
    interrupt = asyncio.Event()
    interrupt._hermes_run_generation = 1
    adapter._active_sessions[key] = interrupt
    ctx = TurnContext(_cleanup_progress=True, _cleanup_msg_ids=['progress'], session_key=key, source=source, run_generation=1)
    async def handler(_event):
        text = 'word ' * 2000
        schedule_cleanup({'failed': False, 'final_response': text}, adapter, ctx)
        return text
    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, key)
    assert adapter._bot.send_message.await_count == 2
    adapter.delete_message.assert_not_awaited()
