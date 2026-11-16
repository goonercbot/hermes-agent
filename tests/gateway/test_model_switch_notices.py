"""Receipt, ownership and open-stream boundaries for persistent model announcements."""
import asyncio
from concurrent.futures import Future
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock
import pytest
from agent.model_switch import bind_model_switch_state, emit_model_switch, flush_model_switch_notices
from gateway.model_switch_notices import bind_compression_notice_sink
from gateway.platforms.base import SendResult
from gateway.run_turn_progress import GatewayTurnProgressMixin
from gateway.session import Platform, SessionSource
from tests.waiting import async_wait_until


def foreground(adapter, source, muted=False):
    ctx = NS(mute_notification_reply=muted, _status_adapter=adapter,
        _run_still_current=lambda: True, source=source, _status_chat_id=source.chat_id,
        _status_thread_metadata={"thread_ts": "1700.42"}, _cleanup_progress=True,
        user_config={"display": {"suppress_warning_notifications": True}})
    return NS(_ctx=ctx, _approval_owner=None, _status_live=lambda: ctx._run_still_current(),
        _track_future_cleanup_id=Mock(), _schedule=lambda coro, label: asyncio.create_task(coro))


@pytest.mark.asyncio
async def test_diagnostic_mute_does_not_consume_required_notice_and_late_stale_send_is_refused():
    adapter = NS(send=AsyncMock(return_value=SendResult(success=True)))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="human")
    owner = foreground(adapter, source, muted=True)
    assert await GatewayTurnProgressMixin._status_callback_sync(owner, "model_switch", "notice")
    adapter.send.assert_awaited_once_with("chat", "notice", metadata={"thread_ts": "1700.42", "_interim_send": True})
    owner._track_future_cleanup_id.assert_not_called()
    delayed = GatewayTurnProgressMixin._status_callback_sync(owner, "model_switch", "stale")
    owner._ctx._run_still_current = lambda: False
    assert await delayed is False
    assert adapter.send.await_count == 1


@pytest.mark.asyncio
async def test_bot_loop_guard_survives_mute_exception():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="bot", is_bot=True)
    adapter = NS(send=AsyncMock())
    assert not GatewayTurnProgressMixin._status_callback_sync(foreground(adapter, source), "model_switch", "no loop")
    adapter.send.assert_not_awaited()


def test_missing_failed_and_inflight_receipts_are_retained_then_retried_once():
    owner, agent = NS(), NS(platform="gateway")
    bind_model_switch_state(agent, owner)
    emit_model_switch(agent, "main", "p", "backup")
    assert len(agent._model_switch_state.pending) == 1
    future = Future()
    agent._model_switch_callback = Mock(return_value=future)
    flush_model_switch_notices(agent)
    flush_model_switch_notices(agent)
    agent._model_switch_callback.assert_called_once()
    future.set_result(SendResult(success=False))
    assert len(agent._model_switch_state.pending) == 1
    replacement = NS(platform="gateway", _model_switch_callback=Mock(return_value=True))
    bind_model_switch_state(replacement, owner)
    assert not emit_model_switch(replacement, "main", "p", "backup")
    replacement._model_switch_callback.assert_called_once()
    assert not replacement._model_switch_state.pending
    emit_model_switch(replacement, "main", "p", "backup")
    replacement._model_switch_callback.assert_called_once()


def test_initial_notice_survives_first_owner_binding():
    agent = NS(platform="gateway")
    emit_model_switch(agent, "main", "p", "backup")
    pending_state = agent._model_switch_state
    bind_model_switch_state(agent, NS())
    assert agent._model_switch_state is pending_state
    agent._model_switch_callback = Mock(return_value=True)
    flush_model_switch_notices(agent)
    assert not pending_state.pending


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["foreground", "compression"])
async def test_notice_through_real_relay_does_not_seal_open_stream(lane):
    from tests.gateway.relay.test_relay_live_cards import _connected_adapter
    adapter, _ = _connected_adapter(supported_ops=("send", "edit", "typing", "draft"))
    ops = []
    async def send_outbound(payload, platform=None):
        ops.append(dict(payload))
        return {"success": True, "message_id": "111.222"}
    adapter._transport = NS(send_outbound=send_outbound)
    metadata = {"thread_ts": "1700.42"}
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="human")
    await adapter.send_draft("C1", 5, "answer in progress", metadata=metadata)
    key = adapter._draft_key("C1", metadata)
    if lane == "foreground":
        assert await GatewayTurnProgressMixin._status_callback_sync(foreground(adapter, source), "model_switch", "notice")
    else:
        entry, agent = NS(session_key="key"), NS()
        gateway = NS(session_store=NS(_entries={"key": entry}), _delivery_adapter_for=lambda s: adapter)
        bind_compression_notice_sink(gateway, agent, source, entry, metadata)
        emit_model_switch(agent, "compression", "p", "backup")
        await async_wait_until(lambda: not any(p.in_flight for p in agent._model_switch_state.pending), what="model-switch delivery settled")
        assert not agent._model_switch_state.pending
    assert adapter._open_draft_by_chat.get(key) == 5
    assert not any(op.get("op") == "draft_seal" for op in ops)
    result = await adapter.send("C1", "actual final answer", metadata=metadata)
    assert result.success and adapter._open_draft_by_chat.get(key) is None


@pytest.mark.asyncio
async def test_compression_sink_rechecks_entry_and_failed_receipt_then_retries():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="origin", user_id="human")
    adapter = NS(send=AsyncMock(side_effect=[SendResult(success=False), SendResult(success=True)]))
    entry, agent = NS(session_key="key"), NS()
    entries = {"key": entry}
    gateway = NS(session_store=NS(_entries=entries), _delivery_adapter_for=lambda s: adapter)
    bind_compression_notice_sink(gateway, agent, source, entry, {"thread_id": "topic"})
    emit_model_switch(agent, "compression", "p", "backup")
    await async_wait_until(lambda: not any(p.in_flight for p in agent._model_switch_state.pending), what="model-switch delivery settled")
    assert len(agent._model_switch_state.pending) == 1
    entries["key"] = NS()
    flush_model_switch_notices(agent)
    await async_wait_until(lambda: not any(p.in_flight for p in agent._model_switch_state.pending), what="model-switch delivery settled")
    assert adapter.send.await_count == 1
    entries["key"] = entry
    flush_model_switch_notices(agent)
    await async_wait_until(lambda: not any(p.in_flight for p in agent._model_switch_state.pending), what="model-switch delivery settled")
    assert not agent._model_switch_state.pending and adapter.send.await_count == 2


@pytest.mark.asyncio
async def test_manual_compression_real_caller_announces_fallback_and_recovery(monkeypatch):
    from gateway.slash_commands_session import GatewaySessionCommandsMixin
    from agent import conversation_compression as cc
    from agent.auxiliary_client import _record_route_info
    from agent.auxiliary_task_config import session_auxiliary_scope
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="origin", user_id="human")
    entry = NS(session_key="key", session_id="sid")
    agent = NS(platform="gateway_hygiene", context_compressor=NS())
    adapter = NS(send=AsyncMock(return_value=SendResult(success=True)))
    gateway = NS(session_store=NS(_entries={"key": entry}),
        _session_key_for_source=lambda s: "key", _resolve_session_agent_runtime=lambda **kw: ("main", {"api_key": "fixture"}),
        _resolve_session_reasoning_config=lambda **kw: None,
        _build_manual_compression_agent=AsyncMock(return_value=agent),
        _run_in_executor_with_context=lambda fn: asyncio.to_thread(fn),
        _delivery_adapter_for=lambda s: adapter, _evict_cached_agent=Mock(),
        _cleanup_agent_resources_off_loop=AsyncMock())
    def compress_now(actual_agent, rows, request, **kwargs):
        assert actual_agent is agent
        def dispatch(messages):
            _record_route_info({}, "p", "backup")
            _record_route_info({}, "p", "preferred")
            return messages
        cc._run_summary_dispatch(agent, rows, dispatch, {}, commit_fence=None,
            attempt_generation=0, hard_cancel_event=None)
        return NS(status="nothing_to_do")
    monkeypatch.setattr("agent.conversation_compression_manual.compress_now", compress_now)
    with session_auxiliary_scope({"compression": {"provider": "p", "model": "preferred"}}):
        await GatewaySessionCommandsMixin._run_manual_compression(gateway, source, entry,
            [{"role": "user", "content": "retain"}], NS(), notice_metadata={"thread_id": "topic"})
    await async_wait_until(lambda: not any(p.in_flight for p in agent._model_switch_state.pending), what="model-switch delivery settled")
    notices = [call.args[1] for call in adapter.send.await_args_list]
    assert len(notices) == 2
    assert "preferred via p → backup via p" in notices[0]
    assert "backup via p → preferred via p" in notices[1]
    assert all(call.kwargs["metadata"] == {"thread_id": "topic", "_interim_send": True}
               for call in adapter.send.await_args_list)
    assert not agent._model_switch_state.pending
