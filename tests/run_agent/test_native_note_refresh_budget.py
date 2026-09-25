"""End-to-end budget recovery for host-owned native continuity notes."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from agent.native_incremental_handoff import (
    NATIVE_INCREMENTAL_NOTE_MAX_SERIALIZED_CHARS,
    create_native_incremental_note,
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
    oversized = _oversized_args(sid, source)

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
            assert "prior valid continuity_note exceeded the host serialized-note budget" in request["instructions"]
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
    assert result["maintenance_failure"] == "continuity note remains over host serialized size budget after correction"
    assert result["maintenance_phase"] == "pre_publication_validation"
    assert len(calls) == 2
    stored = db.get_messages_as_conversation(sid, repair_alternation=False)
    assert "oversized-" not in json.dumps(stored)
