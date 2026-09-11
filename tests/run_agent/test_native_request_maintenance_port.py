"""Regression definitions for the split v0.21.1 native maintenance request seam.

These tests deliberately target the request-loop seam rather than re-testing native-note
cryptographic/durable helpers owned by their respective modules.
"""
from types import SimpleNamespace
from typing import Any, cast

import pytest


def test_authenticated_native_replay_uses_projection_source_not_wire_copy(monkeypatch):
    from agent import turn_api_request
    import agent.native_compaction as native_compaction
    import agent.native_incremental_handoff as handoff

    source = [{"role": "user", "content": "durable authenticated source"}]
    wire_projection = [{"role": "user", "content": "mutable wire projection"}]
    agent = SimpleNamespace(
        api_mode="codex_responses", native_incremental_handoff_enabled=True,
        provider="openai-codex", base_url="https://chatgpt.com/backend-api/codex",
        model="gpt-5.6",
    )
    monkeypatch.setattr(handoff, "native_incremental_continuity_capable", lambda *_a, **_k: True)
    monkeypatch.setattr(handoff, "_projection_source_for_messages", lambda *_a: (source, 1))
    monkeypatch.setattr(handoff, "native_incremental_note_from_history", lambda _source: {"source_cursor": 1})
    monkeypatch.setattr(native_compaction, "validate_persisted_native_compaction_history", lambda _source: {})

    kwargs = turn_api_request._authenticated_native_continuity_kwargs(agent, wire_projection)

    assert kwargs["native_continuity_replay"] is True
    assert kwargs["native_continuity_source_messages"] is source
    assert kwargs["native_continuity_source_messages"] is not wire_projection


def test_unauthenticated_projection_cannot_enable_native_replay(monkeypatch):
    from agent import turn_api_request
    import agent.native_incremental_handoff as handoff

    agent = SimpleNamespace(
        api_mode="codex_responses", native_incremental_handoff_enabled=True,
        provider="openai-codex", base_url="https://chatgpt.com/backend-api/codex",
        model="gpt-5.6",
    )
    monkeypatch.setattr(handoff, "native_incremental_continuity_capable", lambda *_a, **_k: True)
    monkeypatch.setattr(handoff, "_projection_source_for_messages", lambda *_a: (None, None))

    assert turn_api_request._authenticated_native_continuity_kwargs(agent, [{"role": "user"}]) == {}


def test_preflight_defers_to_one_native_note_refresh_instead_of_generic_compression(monkeypatch):
    from agent import turn_preflight

    agent = SimpleNamespace(
        api_mode="codex_responses", native_incremental_handoff_enabled=True,
        compression_enabled=True,
    )
    verdict = SimpleNamespace(
        messages=[{"role": "user", "content": "stale protected tail"}], action=None, result=None,
    )
    monkeypatch.setattr(turn_preflight, "_native_note_refresh_pending", lambda *_a: True)

    result = turn_preflight.run_preflight_compression(
        agent, cast(Any, verdict), compressor=SimpleNamespace(), request_pressure_tokens=999999,
        provider_overflow_preflight=False, defer_preflight=lambda _n: False,
        moa_prepared_request=None, system_message=None, user_message="next", max_compression_attempts=3,
        effective_task_id="task",
    )

    assert result.action == "fallthrough"


def test_successful_maintenance_bypasses_normal_tool_round_and_rebuilds_next_request(monkeypatch):
    from agent import conversation_loop
    import agent.native_note_refresh as native_note_refresh

    capability = SimpleNamespace(check_request=lambda *_a: None)
    state = SimpleNamespace(
        retry_count=0, max_retries=1, native_note_refresh_request=False,
        native_note_refresh_completed=False, response=None, messages=[], conversation_history=[],
        api_call_count=1, effective_task_id="task", api_kwargs={},
    )
    agent = SimpleNamespace(_persist_session=lambda *_a: None)
    executed = []
    closed = []

    def fake_phase(phase, _agent, s, **_extra):
        if phase is conversation_loop.nous_rate_limit_guard:
            return SimpleNamespace(action="fallthrough")
        if phase is conversation_loop.build_api_request:
            s.native_note_refresh_request = capability
            return SimpleNamespace(action="fallthrough")
        if phase is conversation_loop.perform_api_call:
            s.response = "raw-maintenance-response"
            return SimpleNamespace(action="fallthrough")
        if phase is conversation_loop.check_api_response:
            assert s.response == "validated-maintenance-response"
            return SimpleNamespace(action="break")
        raise AssertionError(f"unexpected phase {phase}")

    monkeypatch.setattr(conversation_loop, "_run_phase", fake_phase)
    monkeypatch.setattr(native_note_refresh, "execute_native_note_refresh", lambda *_a: executed.append(True) or "validated-maintenance-response")
    monkeypatch.setattr(native_note_refresh, "close_native_note_refresh", lambda *_a: closed.append(True))

    assert conversation_loop._run_api_retry_loop(agent, cast(Any, state)) is None
    assert executed == [True]
    assert state.native_note_refresh_completed is True
    assert closed  # no single-request authority remains for an ordinary follow-up


@pytest.mark.parametrize("failure", ["malformed", "blocked", "interrupted"])
def test_maintenance_failure_returns_closed_partial_result(monkeypatch, failure):
    from agent import conversation_loop
    import agent.native_note_refresh as native_note_refresh

    state = SimpleNamespace(
        retry_count=0, max_retries=1,
        native_note_refresh_request=SimpleNamespace(check_request=lambda *_a: None),
        native_note_refresh_completed=False, response="provider-response", messages=[],
        conversation_history=[], api_call_count=1, effective_task_id="task", api_kwargs={},
    )
    agent = SimpleNamespace(_persist_session=lambda *_a: None)
    monkeypatch.setattr(conversation_loop, "_run_phase", lambda phase, _agent, s, **_extra: (
        SimpleNamespace(action="fallthrough") if phase is conversation_loop.nous_rate_limit_guard
        else SimpleNamespace(action="fallthrough") if phase is conversation_loop.perform_api_call
        else SimpleNamespace(action="fallthrough") if phase is conversation_loop.build_api_request
        else (_ for _ in ()).throw(AssertionError("response normalization must not run"))
    ))
    monkeypatch.setattr(
        native_note_refresh, "execute_native_note_refresh",
        lambda *_a: (_ for _ in ()).throw(native_note_refresh.NativeNoteRefreshFailure(failure)),
    )
    monkeypatch.setattr(native_note_refresh, "account_failed_native_note_refresh", lambda *_a: None)
    monkeypatch.setattr(native_note_refresh, "close_native_note_refresh", lambda *_a: None)

    result = conversation_loop._run_api_retry_loop(agent, cast(Any, state))

    assert result is not None
    assert result["failed"] is True
    assert result["error"] == "native_note_refresh_failed"
    assert result["maintenance_failure"] == failure
