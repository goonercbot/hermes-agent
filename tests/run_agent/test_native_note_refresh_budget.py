"""End-to-end budget recovery for host-owned native continuity notes."""
from copy import deepcopy
import json
import logging
import sqlite3
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from agent.native_incremental_handoff import (
    NATIVE_INCREMENTAL_NOTE_MAX_SERIALIZED_CHARS,
    create_native_incremental_note,
    native_incremental_note_budget_breakdown,
    native_incremental_note_budget_correction_guidance,
    native_incremental_note_serialized_size,
    record_native_incremental_note_from_tool_call,
    restore_native_incremental_note,
)
from tests.run_agent.test_native_note_refresh import ARGS, blocking_router, reply


def _exact_budget_args(session_id, source):
    """Make an exact-size valid note with escaped and non-ASCII text."""
    args = {**ARGS, "current_plan": 'quote " slash \\ newline\n界'}
    note = create_native_incremental_note(
        session_id=session_id, source_messages=source, **args,
    )
    args["current_plan"] += "x" * (
        NATIVE_INCREMENTAL_NOTE_MAX_SERIALIZED_CHARS
        - native_incremental_note_serialized_size(note)
    )
    note = create_native_incremental_note(
        session_id=session_id, source_messages=source, **args,
    )
    assert native_incremental_note_serialized_size(note) == NATIVE_INCREMENTAL_NOTE_MAX_SERIALIZED_CHARS
    return args


def test_serialized_note_budget_is_exact_and_includes_escaping_and_unicode():
    from tools.continuity_note_tool import CONTINUITY_NOTE_SCHEMA

    source = [{"role": "user", "content": "preserve current task"}]
    args = _exact_budget_args("budget-boundary", source)
    note = create_native_incremental_note(
        session_id="budget-boundary", source_messages=source, **args,
    )
    assert native_incremental_note_serialized_size(note) == NATIVE_INCREMENTAL_NOTE_MAX_SERIALIZED_CHARS
    assert str(NATIVE_INCREMENTAL_NOTE_MAX_SERIALIZED_CHARS) in CONTINUITY_NOTE_SCHEMA["description"]
    with pytest.raises(ValueError, match="host serialized size budget"):
        create_native_incremental_note(
            session_id="budget-boundary", source_messages=source,
            **{**args, "current_plan": args["current_plan"] + "x"},
        )


def test_budget_feedback_accounts_for_fields_without_echoing_rejected_contents():
    source = [{"role": "user", "content": "preserve current task"}]
    args = _exact_budget_args("budget-feedback", source)
    sensitive = "SENSITIVE_IDENTIFIER_NOT_FOR_CORRECTION_PROMPT"
    oversized = {**args, "current_plan": args["current_plan"] + sensitive}
    baseline = create_native_incremental_note(
        session_id="budget-feedback", source_messages=source,
        objective="baseline objective", current_plan="baseline plan",
        next_action="baseline next action", blockers=[],
    )
    candidate = {
        "version": 1,
        "session_id": "budget-feedback",
        "source_cursor": len(source),
        "source_prefix_fence": baseline["source_prefix_fence"],
        "claim_kind": "agent_authored",
        "instruction_precedence": (
            "Current and later user instructions supersede this historical "
            "agent-authored continuity note."
        ),
        **oversized,
    }
    breakdown = native_incremental_note_budget_breakdown(candidate)
    guidance = native_incremental_note_budget_correction_guidance(breakdown)
    assert breakdown["serialized_chars"] == native_incremental_note_serialized_size(candidate)
    assert breakdown["over_budget_chars"] == len(sensitive)
    assert sum(breakdown["field_value_chars"].values()) + breakdown["non_field_value_chars"] == breakdown["serialized_chars"]
    assert guidance is not None
    assert f"measured {breakdown['serialized_chars']} serialized characters" in guidance
    assert "current_plan=" in guidance
    assert sensitive not in guidance


@pytest.mark.parametrize(("reason", "phase"), [
    ("maintenance cancelled", "interrupted"),
    ("forged maintenance capability", "authorization_or_source_validation"),
    ("invalid, wrapped or reused maintenance call", "response_validation"),
    ("blocked, failed or unchanged authenticated note", "middleware_dispatch"),
    ("maintenance pair persistence failed", "persistence"),
    ("durable note readback failed authentication", "post_flush_verification"),
])
def test_nonretryable_failure_classes_have_safe_host_phases(reason, phase):
    from agent.native_note_refresh import NativeNoteRefreshFailure

    assert NativeNoteRefreshFailure(reason).phase == phase


@pytest.fixture
def stale_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "compression:\n  enabled: true\n  codex_responses_native: true\n"
        "  native_incremental_handoff: true\n"
    )
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "native-note-budget-recovery"
    db.create_session(sid, "cli", model="gpt-6-astra")
    with patch("agent.turn_context._maybe_title_session_at_turn_start", return_value=None):
        agent = AIAgent(
            api_key="fixture", base_url="https://chatgpt.com/backend-api/codex",
            api_mode="codex_responses", model="gpt-6-astra", provider="openai-codex",
            session_db=db, session_id=sid, quiet_mode=True, skip_memory=True,
            skip_context_files=True, skip_background_review=True,
            enabled_toolsets=["continuity", "terminal", "fleet_task_router"],
        )
        agent._disable_streaming = True
        agent._compression_feasibility_checked = True
        agent._emit_status = lambda *args, **kwargs: None
        agent.commit_memory_session = lambda *args, **kwargs: None
        agent.context_compressor.threshold_tokens = 1_000_000
        db.append_message(sid, "user", "Original exact objective")
        db.append_message(sid, "assistant", "", tool_calls=[{
            "id": "old-note", "type": "function", "function": {
                "name": "continuity_note", "arguments": json.dumps(ARGS),
            },
        }])
        initial = db.get_messages_as_conversation(sid)
        db.append_message(
            sid, "tool", record_native_incremental_note_from_tool_call(agent, ARGS, initial),
            tool_call_id="old-note", tool_name="continuity_note",
        )
        db.append_message(sid, "assistant", "verified durable evidence " * 20000)
        source = db.get_messages_as_conversation(sid)
        assert restore_native_incremental_note(agent, source) is not None
        try:
            yield agent, db, sid, source, tmp_path
        finally:
            agent.close()
            db.close()


def _oversized_args(session_id, source):
    args = _exact_budget_args(session_id, source)
    return {**args, "current_plan": args["current_plan"] + "x"}


def test_oversized_note_gets_one_fresh_corrective_request_then_real_tools_resume(
    stale_agent, blocking_router,
):
    agent, db, sid, source, tmp_path = stale_agent
    ordinary_tools = deepcopy(agent.tools)
    calls = []
    capabilities = []
    rejected_marker = "REJECTED_IDENTIFIER_MUST_NOT_BE_ECHOED"
    oversized_base = _oversized_args(sid, source)
    oversized = {
        **oversized_base,
        "current_plan": oversized_base["current_plan"] + rejected_marker,
    }

    def provider(request):
        calls.append(deepcopy(request))
        capabilities.append(agent._native_note_refresh_capability)
        if len(calls) == 1:
            assert [tool["name"] for tool in request["tools"]] == ["continuity_note"]
            assert str(NATIVE_INCREMENTAL_NOTE_MAX_SERIALIZED_CHARS) in request["instructions"]
            return reply(NS(
                type="function_call", id="oversized", call_id="oversized-note",
                name="continuity_note", arguments=json.dumps(oversized),
            ))
        if len(calls) == 2:
            assert [tool["name"] for tool in request["tools"]] == ["continuity_note"]
            assert "rejected before dispatch: its complete canonical envelope measured" in request["instructions"]
            assert "field-value costs were objective=" in request["instructions"]
            assert rejected_marker not in request["instructions"]
            assert agent._current_api_request_id.endswith(":api:2")
            return reply(NS(
                type="function_call", id="corrected", call_id="corrected-note",
                name="continuity_note", arguments=json.dumps(ARGS),
            ))
        assert agent._native_note_refresh_capability is None
        assert agent.tools == ordinary_tools
        assert "terminal" in {tool["name"] for tool in request["tools"]}
        if len(calls) == 3:
            return reply(NS(
                type="function_call", id="route", call_id="route-call", name="tool_call",
                arguments=json.dumps({"name": "fleet_route_task", "arguments": {
                    "work_shape": "direct", "consequence": "routine",
                    "reason_codes": ["known_short_path"], "proof_target": "focused_test",
                }}),
            ), tokens=600)
        if len(calls) == 4:
            return reply(NS(
                type="function_call", id="terminal", call_id="ordinary-call", name="terminal",
                arguments=json.dumps({"command": "printf BUDGET_CORRECTION_OK", "workdir": str(tmp_path)}),
            ), tokens=600)
        assert len(calls) == 5
        return reply(NS(
            type="message", role="assistant", content=[
                NS(type="output_text", text="BUDGET_CORRECTION_OK"),
            ],
        ), tokens=600)

    agent._interruptible_api_call = provider
    result = agent.run_conversation("Continue with the active task.", conversation_history=source)
    assert result["completed"] and result["final_response"] == "BUDGET_CORRECTION_OK"
    assert len(calls) == 5
    assert capabilities[:2][0] is not capabilities[:2][1]
    stored = db.get_messages_as_conversation(sid, repair_alternation=False)
    assert "oversized-note" not in json.dumps(stored)
    assert sum(row.get("tool_call_id") == "corrected-note" for row in stored) == 1
    assert sum(row.get("tool_call_id") == "ordinary-call" for row in stored) == 1
    assert restore_native_incremental_note(NS(session_id=sid), stored) == agent._native_incremental_handoff_note

    # The published correction remains authenticated after a new agent opens
    # the durable history; the next request has the normal tool inventory.
    from hermes_state import SessionDB
    from run_agent import AIAgent
    resumed_db = SessionDB(db_path=tmp_path / "state.db")
    resumed = AIAgent(
        api_key="fixture", base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses", model="gpt-6-astra", provider="openai-codex",
        session_db=resumed_db, session_id=sid, quiet_mode=True, skip_memory=True,
        skip_context_files=True, skip_background_review=True,
        enabled_toolsets=["continuity", "terminal", "fleet_task_router"],
    )
    resumed._disable_streaming = True
    resumed._compression_feasibility_checked = True
    resumed._emit_status = lambda *args, **kwargs: None
    resumed.commit_memory_session = lambda *args, **kwargs: None
    resumed.context_compressor.threshold_tokens = 1_000_000
    resumed_requests = []
    resumed._interruptible_api_call = lambda request: (
        resumed_requests.append(deepcopy(request)) or reply(NS(
            type="message", role="assistant", content=[
                NS(type="output_text", text="RESUMED_CORRECTED_NOTE_OK"),
            ],
        ), tokens=600)
    )
    try:
        result = resumed.run_conversation(
            "Continue after restart.", conversation_history=resumed_db.get_messages_as_conversation(sid),
        )
        assert result["completed"] and result["final_response"] == "RESUMED_CORRECTED_NOTE_OK"
        assert len(resumed_requests) == 1
        assert "terminal" in {tool["name"] for tool in resumed_requests[0]["tools"]}
        assert "prior valid continuity_note exceeded" not in resumed_requests[0]["instructions"]
    finally:
        resumed.close()
        resumed_db.close()


def test_second_oversized_note_fails_closed_without_ordinary_work(stale_agent):
    agent, db, sid, source, _tmp_path = stale_agent
    oversized = _oversized_args(sid, source)
    calls = []

    def provider(request):
        calls.append(deepcopy(request))
        return reply(NS(
            type="function_call", id=f"oversized-{len(calls)}", call_id=f"oversized-{len(calls)}",
            name="continuity_note", arguments=json.dumps(oversized),
        ))

    agent._interruptible_api_call = provider
    result = agent.run_conversation("Continue with the active task.", conversation_history=source)
    assert result["failed"] is True
    assert result["error"] == "native_note_refresh_failed"
    assert result["maintenance_failure"] == "continuity note correction did not reduce host serialized size"
    assert result["maintenance_phase"] == "pre_publication_validation"
    assert len(calls) == 2
    stored = db.get_messages_as_conversation(sid, repair_alternation=False)
    assert "oversized-" not in json.dumps(stored)


def test_smaller_but_still_oversized_correction_reports_safe_exhaustion(stale_agent):
    agent, db, sid, source, _tmp_path = stale_agent
    smaller = _oversized_args(sid, source)
    larger = {**smaller, "current_plan": smaller["current_plan"] + "x" * 100}
    calls = []

    def provider(request):
        calls.append(deepcopy(request))
        args = larger if len(calls) == 1 else smaller
        return reply(NS(
            type="function_call", id=f"oversized-{len(calls)}", call_id=f"oversized-{len(calls)}",
            name="continuity_note", arguments=json.dumps(args),
        ))

    agent._interruptible_api_call = provider
    result = agent.run_conversation("Continue with the active task.", conversation_history=source)
    assert result["failed"] is True
    assert result["maintenance_failure"] == "continuity note remains over host serialized size budget after correction"
    assert result["maintenance_phase"] == "pre_publication_validation"
    assert len(calls) == 2
    stored = db.get_messages_as_conversation(sid, repair_alternation=False)
    assert "oversized-" not in json.dumps(stored)


def test_two_independent_refresh_events_each_receive_one_correction(stale_agent, blocking_router):
    """A corrected event must not consume the next stale tail's allowance."""
    agent, db, sid, source, tmp_path = stale_agent
    maintenance = []
    ordinary = []
    capabilities = []

    def provider(request):
        if request.get("tool_choice") == {"type": "function", "name": "continuity_note"}:
            maintenance.append(deepcopy(request))
            capabilities.append(agent._native_note_refresh_capability)
            args = _oversized_args(sid, source) if len(maintenance) % 2 else ARGS
            return reply(NS(
                type="function_call", id=f"note-{len(maintenance)}", call_id=f"note-{len(maintenance)}",
                name="continuity_note", arguments=json.dumps(args),
            ))
        ordinary.append(deepcopy(request))
        if len(ordinary) == 1:
            # This real-loop fixture deliberately grows the authenticated tail,
            # creating a separate maintenance event without changing thresholds.
            return reply(
                NS(type="message", role="assistant", content=[
                    NS(type="output_text", text="additional verified public evidence " * 20000),
                ]),
                NS(type="function_call", id="route", call_id="route", name="tool_call", arguments=json.dumps({
                    "name": "fleet_route_task", "arguments": {
                        "work_shape": "direct", "consequence": "routine",
                        "reason_codes": ["known_short_path"], "proof_target": "focused_test",
                    },
                })),
                tokens=220000,
            )
        return reply(NS(type="message", role="assistant", content=[
            NS(type="output_text", text="TWO_CORRECTIONS_DONE"),
        ]), tokens=600)

    agent._interruptible_api_call = provider
    result = agent.run_conversation("Continue the approved exact task.", conversation_history=source)
    assert result.get("completed") and result.get("final_response") == "TWO_CORRECTIONS_DONE"
    assert len(maintenance) == 4
    assert capabilities[0] is not capabilities[1]
    assert capabilities[2] is not capabilities[3]
    stored = db.get_messages_as_conversation(sid, repair_alternation=False)
    assert sum(row.get("tool_call_id") == call_id for row in stored for call_id in ("note-2", "note-4")) == 2
    assert all(call_id not in json.dumps(stored) for call_id in ("note-1", "note-3"))


def test_canonical_database_failure_is_safe_in_executor_and_loop(stale_agent, caplog):
    """Database and history classes must survive the generic exception boundary."""
    from agent.native_note_refresh import (
        NativeNoteRefreshFailure, execute_native_note_refresh, issue_native_note_refresh,
    )
    from agent.native_incremental_handoff import bind_native_incremental_replay_projection

    agent, db, sid, source, _tmp_path = stale_agent
    agent._current_api_request_id = "database-failure:api:1"
    assert bind_native_incremental_replay_projection(agent, source_messages=source, replay_messages=source)
    capability = issue_native_note_refresh(agent, source)
    response = reply(NS(
        type="function_call", id="database-failure", call_id="database-failure",
        name="continuity_note", arguments=json.dumps(ARGS),
    ))
    with patch.object(db, "get_messages_as_conversation", side_effect=sqlite3.OperationalError("private synthetic failure")):
        with pytest.raises(NativeNoteRefreshFailure) as caught:
            execute_native_note_refresh(agent, capability, response, source, sid)
    assert caught.value.reason == "canonical maintenance database read failed"
    assert caught.value.phase == "canonical_database_read"

    # Recreate the normal real-loop path and assert the returned/logged class;
    # raw database text is never surfaced in the result or warning.
    source = db.get_messages_as_conversation(sid, repair_alternation=False)
    agent._interruptible_api_call = lambda _request: reply(NS(
        type="function_call", id="database-loop", call_id="database-loop",
        name="continuity_note", arguments=json.dumps(ARGS),
    ))
    with patch.object(db, "get_messages_as_conversation", side_effect=sqlite3.OperationalError("private synthetic failure")):
        with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
            result = agent.run_conversation("Continue safely.", conversation_history=source)
    assert result["failed"] and result["maintenance_failure"] == "canonical maintenance database read failed"
    assert result["maintenance_phase"] == "canonical_database_read"
    assert "private synthetic failure" not in caplog.text
