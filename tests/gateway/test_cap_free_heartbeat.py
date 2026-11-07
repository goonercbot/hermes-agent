"""Heartbeat send and edit receipts flow through the native turn consumer."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import SendResult
from tests.gateway.test_waiting_reply_delivery import setup
from tests.gateway.test_run_cleanup_progress import _make_runner


@pytest.mark.asyncio
async def test_heartbeat_tracks_only_fresh_send_and_edit_overflow(monkeypatch):
    from gateway.commentary_delivery import schedule_cleanup
    adapter, guard, ctx = setup(SimpleNamespace(_current_turn_id='turn'))
    runner = _make_runner(adapter)
    monkeypatch.setattr(runner, '_delivery_adapter_for', lambda _: adapter)
    decisions = iter([True, True, True, False])
    monkeypatch.setattr(runner, '_should_emit_long_running_notification', lambda *a: next(decisions))
    monkeypatch.setattr(runner, '_agent_activity_summary', lambda _: None)
    import gateway.run as run
    monkeypatch.setattr(run, '_float_env', lambda *a: 1)
    async def no_wait(_):
        return None
    monkeypatch.setattr(asyncio, 'sleep', no_wait)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id='heartbeat', fresh_message_ids=('heartbeat',)))
    adapter.edit_message = AsyncMock(return_value=SendResult(success=True, message_id='heartbeat',
                                                             fresh_message_ids=('overflow',)))
    disp = SimpleNamespace(_display_surface_mode=lambda *a, **k: 'generic',
                           resolve_display_setting=lambda *a: False, user_config={}, platform_key='telegram',
                           _generic_status_phrase=lambda _: 'working')
    await runner._run_agent_notify_long_running(disp, ctx, [None])
    assert adapter.send.await_count == 1
    assert adapter.edit_message.await_count == 1
    assert ctx._cleanup_msg_ids == ['heartbeat', 'overflow']
    schedule_cleanup({'final_response': 'answer'}, adapter, ctx)
    guard._hermes_final_delivery_ids = {'answer'}
    adapter._confirm_final_delivery('session', 1, True)
    await adapter._fire_post_delivery_callback('session', guard)
    assert [row['message_id'] for row in adapter.deleted] == ['heartbeat', 'overflow']
