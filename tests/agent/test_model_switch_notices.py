"""Model switches are immediate, per-conversation events, not disposable retry progress."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.model_switch import bind_model_switch_state, emit_model_switch
from agent.status_output import StatusOutputMixin


class RecordingAgent(StatusOutputMixin):
    def __init__(self):
        self.model = "backup"
        self.provider = "backup-provider"
        self.events = []
        self.status_callback = lambda kind, text: self.events.append((kind, str(text)))
        self._vprint = lambda *a, **kw: None
        self.log_prefix = ""


def test_fallback_is_announced_before_final_response_and_not_replayed_on_failure():
    from agent.chat_completion_helpers import _buffer_fallback_notice
    agent = RecordingAgent()
    _buffer_fallback_notice(agent, "Main switched from preferred to backup")
    assert agent.events == [("model_switch", "Main switched from preferred to backup")]
    agent._clear_status_buffer()
    agent._emit_pending_fallback_notice()
    agent._flush_status_buffer()
    assert len(agent.events) == 1


def test_pre_agent_notice_is_consumed_once_on_callback_wiring():
    agent = RecordingAgent()
    agent._pending_fallback_notice = "Pre-agent switch"
    agent._emit_pending_fallback_notice()
    agent._emit_pending_fallback_notice()
    assert agent.events == [("model_switch", "Pre-agent switch")]


def test_recreated_main_and_hygiene_agents_share_routes_without_cross_chat_leak():
    owner = SimpleNamespace()
    a, b, other = RecordingAgent(), RecordingAgent(), RecordingAgent()
    bind_model_switch_state(a, owner)
    bind_model_switch_state(b, owner)
    bind_model_switch_state(other, SimpleNamespace())
    assert emit_model_switch(a, "compression", "p", "backup", previous=("p", "preferred"))
    assert not emit_model_switch(b, "compression", "p", "backup", previous=("p", "preferred"))
    assert emit_model_switch(b, "compression", "p", "preferred")
    assert emit_model_switch(b, "compression", "p", "backup")
    assert emit_model_switch(other, "compression", "p", "backup")
    # Main and compression remain independent even when their current pair happens to match.
    assert emit_model_switch(b, "main", "p", "backup")
    assert len(a.events) == 1 and len(b.events) == 3 and len(other.events) == 1


def test_initial_preferred_route_is_silent_but_provider_only_change_is_not():
    agent = RecordingAgent()
    assert not emit_model_switch(agent, "compression", "p", "m", previous=("p", "m"))
    assert emit_model_switch(agent, "compression", "other", "m")
    assert "m via p → m via other" in agent.events[0][1]


def test_route_observer_restores_outer_scope_and_cannot_break_provider_resolution():
    from agent.auxiliary_client import _record_route_info, aux_route_hook
    outer, inner = Mock(), Mock()
    route = {}
    with aux_route_hook(outer):
        _record_route_info(route, "p", "a")
        with aux_route_hook(inner):
            _record_route_info(route, "p", "b")
        _record_route_info(route, "p", "c")
    _record_route_info(route, "p", "d")
    assert outer.call_count == 2 and inner.call_count == 1
    assert route == {"provider": "p", "model": "d"}
    with aux_route_hook(Mock(side_effect=RuntimeError("fixture failure"))):
        _record_route_info(route, "p", "e")
    assert route["model"] == "e"


def test_real_summary_dispatch_observes_fallback_then_return_without_touching_messages(monkeypatch):
    from agent import conversation_compression as cc
    from agent.auxiliary_client import _record_route_info
    from agent.auxiliary_task_config import session_auxiliary_scope
    agent = RecordingAgent()
    agent.context_compressor = SimpleNamespace()
    messages = [{"role": "user", "content": "preserve me"}]
    def compress(rows):
        assert rows is messages
        _record_route_info({}, "p", "backup")
        _record_route_info({}, "p", "backup")
        _record_route_info({}, "p", "preferred")
        return rows
    with session_auxiliary_scope({"compression": {"provider": "p", "model": "preferred"}}):
        result = cc._run_summary_dispatch(agent, messages, compress, {}, commit_fence=None,
                                          attempt_generation=0, hard_cancel_event=None)
    assert result is messages and messages[0]["content"] == "preserve me"
    assert [kind for kind, _ in agent.events] == ["model_switch", "model_switch"]
    assert "preferred via p → backup via p" in agent.events[0][1]
    assert "backup via p → preferred via p" in agent.events[1][1]


def test_cancelled_summary_never_announces_a_route():
    from agent import conversation_compression as cc
    from agent.auxiliary_client import _record_route_info
    agent = RecordingAgent()
    agent.session_id = "fixture-session"
    agent.context_compressor = SimpleNamespace()
    fence = cc.CompressionCommitFence()
    fence.try_cancel_before_commit()
    call = Mock(side_effect=lambda rows: _record_route_info({}, "p", "m"))
    rows = [{"role": "user", "content": "keep"}]
    assert cc._run_summary_dispatch(agent, rows, call, {}, commit_fence=fence,
                                    attempt_generation=0, hard_cancel_event=None) is rows
    call.assert_not_called()
    assert not agent.events
