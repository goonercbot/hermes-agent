"""Authority-delivered replies gate cleanup on a complete synthetic transport receipt."""
from contextlib import nullcontext
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import session_ingress
from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from gateway.commentary_delivery import schedule_cleanup
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize('result,deleted', [
    (SendResult(success=True, message_id='final', full_payload_delivered=True), True),
    (SendResult(success=True, message_id='partial', full_payload_delivered=False), False),
    (SendResult(success=False, error='rejected'), False),
])
async def test_authority_delivers_then_fires_proof_gated_callback(monkeypatch, result, deleted):
    adapter = ProgressCaptureAdapter()
    adapter.send = AsyncMock(return_value=result)
    adapter.delete_message = AsyncMock(return_value=True)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='chat', chat_type='dm')
    event = MessageEvent(text='question', message_type=MessageType.TEXT, source=source, message_id='in')
    route = 'authority-route'
    guard = asyncio.Event()
    guard._hermes_run_generation = 9
    adapter._active_sessions[route] = guard
    ctx = TurnContext(_cleanup_progress=True, _cleanup_msg_ids=['progress'], session_key=route,
                      source=source, run_generation=9)
    schedule_cleanup({'final_response': 'answer'}, adapter, ctx)
    authority = SimpleNamespace(runner=SimpleNamespace(_adapter_for_source=lambda _: adapter),
                                native_waiters=set(), pending_deliveries={})
    session_ingress._take_post_delivery(authority, 'admission', source, route)
    authority.pending_deliveries['admission'] = (adapter, event, route, 'answer', None)
    async def scope(*args):
        return nullcontext()
    monkeypatch.setattr(session_ingress, '_execution_scope', scope)
    monkeypatch.setattr(session_ingress, '_release_turn_marker', AsyncMock())
    monkeypatch.setattr(session_ingress, 'close_producer_turn', AsyncMock())
    await session_ingress.deliver_settled(authority, 'admission')
    await session_ingress.fire_post_delivery(authority, 'admission')
    assert adapter.delete_message.await_count == int(deleted)
    assert not getattr(adapter, '_confirmed_final_deliveries', {})
