"""Actual final-delivery callback gate: unknown/clipped/fallback never erase evidence.

Owner audit replay, with an explicit complete-body receipt on successful controls.
The transport is synthetic; persistence/runner/formatter/callback paths are real.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer
from gateway.turn_context import TurnContext
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter


async def deliver(outcome, *, text='Complete answer', generation=1, fallback=False):
    adapter = ProgressCaptureAdapter()
    adapter.config.typing_indicator = False
    attempted = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        attempted.append(content)
        if fallback and len(attempted) == 1:
            return SendResult(success=False, error="can't parse entities", error_kind='bad_format')
        return outcome

    adapter.send = AsyncMock(side_effect=send)
    adapter.delete_message = AsyncMock(return_value=True)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='chat', chat_type='dm')
    event = MessageEvent(text='question', message_type=MessageType.TEXT, source=source, message_id='incoming')
    key = 'cleanup-delivery'
    interrupt = asyncio.Event()
    interrupt._hermes_run_generation = generation
    adapter._active_sessions[key] = interrupt
    runner = object.__new__(GatewayRunner)
    ctx = TurnContext(_cleanup_progress=True, _cleanup_msg_ids=['progress'],
                      session_key=key, source=source, run_generation=1)

    async def handler(_event):
        runner._run_agent_schedule_bubble_cleanup({'failed': False, 'final_response': text}, adapter, ctx)
        return None if generation != ctx.run_generation else text

    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, key)
    return adapter, attempted


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', [SendResult(success=False, error='rejected'),
    SendResult(success=True, message_id='unknown'),
    SendResult(success=True, message_id='clipped', full_payload_delivered=False),
    SendResult(success=True, message_id='partial', full_payload_delivered=True, raw_response={'partial_overflow': True})])
async def test_incomplete_or_unknown_send_retains_commentary(outcome):
    adapter, _ = await deliver(outcome)
    adapter.delete_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('split', [False, True])
async def test_complete_body_receipt_authorizes_cleanup(split):
    adapter, _ = await deliver(SendResult(success=True, message_id='final', full_payload_delivered=True,
        continuation_message_ids=('head',) if split else ()))
    adapter.delete_message.assert_awaited_once_with('chat', 'progress')


@pytest.mark.asyncio
async def test_actual_truncated_fallback_cannot_reuse_success_receipt():
    body = 'complete-current-body ' * 300
    adapter, attempted = await deliver(SendResult(success=True, message_id='fallback', full_payload_delivered=True),
                                       text=body, fallback=True)
    assert len(attempted) == 2 and body not in attempted[1]
    adapter.delete_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_stale_generation_cannot_delete():
    adapter, _ = await deliver(SendResult(success=True, full_payload_delivered=True), generation=2)
    adapter.delete_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('full,raw,expected', [(True, None, True), (False, None, False),
    (None, None, False), (True, {'partial_overflow': True}, False), (True, {'skipped': True}, False)])
async def test_transformed_edit_needs_complete_current_body(full, raw, expected):
    runner = object.__new__(GatewayRunner)
    edit = AsyncMock(return_value=SendResult(success=True, full_payload_delivered=full, raw_response=raw))
    consumer = GatewayStreamConsumer(SimpleNamespace(edit_message=edit), 'chat')
    consumer._message_id = 'streamed'
    response = {}
    await runner._run_agent_edit_streamed_message(consumer, SimpleNamespace(chat_id='chat'), response,
        'transformed body', _sk='synthetic', ok=('edited',), fail_result='failed %s %s', fail_exc='exception %s %s')
    assert bool(response.get('already_sent')) is (expected or full is None)
    assert bool(response.get('_complete_final_body')) is expected
