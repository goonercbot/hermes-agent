"""Synthetic long-state lifecycle evidence for native note refresh reliability."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS
from unittest.mock import patch

from agent.native_incremental_handoff import (
    NATIVE_INCREMENTAL_NOTE_MAX_SERIALIZED_CHARS,
    create_native_incremental_note,
    native_incremental_note_serialized_size,
    record_native_incremental_note_from_tool_call,
    restore_native_incremental_note,
)
from agent.native_note_refresh import execute_native_note_refresh, issue_native_note_refresh
from tests.run_agent.test_native_note_refresh import ARGS, blocking_router, reply


_UNRELATED = "UNRELATED-OPS-2026.09.27-RC7"
_REPLACED = "REPLACED-OBJECTIVE-2026.09.27"


def _dense_state(index):
    atom = (
        f" ticket=INC-{index:03d}-界 route=/native/{index}/v1 "
        f"version=rel-{index}.2 quoted=\"keep\\exact\" "
        f"path=C:\\state\\{index} marker={_UNRELATED};"
    )
    return atom * 55


def _make_agent(db, session_id):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="fixture", base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses", model="gpt-6-astra", provider="openai-codex",
        session_db=db, session_id=session_id, quiet_mode=True, skip_memory=True,
        skip_context_files=True, skip_background_review=True,
        enabled_toolsets=["continuity", "terminal", "fleet_task_router"],
    )
    agent._disable_streaming = True
    agent._compression_feasibility_checked = True
    agent._emit_status = lambda *args, **kwargs: None
    agent.commit_memory_session = lambda *args, **kwargs: None
    agent.context_compressor.threshold_tokens = 1_000_000
    return agent


def _append_long_stale_history(db, agent, session_id):
    """Create 165 synthetic canonical rows with a note cursor at row 114."""
    for index in range(111):
        db.append_message(
            session_id, "user" if index % 2 == 0 else "assistant", _dense_state(index),
        )
    db.append_message(session_id, "assistant", "", tool_calls=[{
        "id": "prior-a", "type": "function",
        "function": {"name": "terminal", "arguments": json.dumps({"command": "true"})},
    }, {
        "id": "prior-b", "type": "function",
        "function": {"name": "terminal", "arguments": json.dumps({"command": "true"})},
    }])
    db.append_message(session_id, "tool", "prior-a-ok", tool_call_id="prior-a", tool_name="terminal")
    db.append_message(session_id, "tool", "prior-b-ok", tool_call_id="prior-b", tool_name="terminal")
    note_source = db.get_messages_as_conversation(session_id, repair_alternation=False)
    assert len(note_source) == 114
    initial = {
        "objective": f"Preserve {_UNRELATED}",
        "current_plan": "Gather verified state",
        "next_action": "Refresh continuity note",
        "blockers": [],
    }
    db.append_message(session_id, "assistant", "", tool_calls=[{
        "id": "old-note", "type": "function",
        "function": {"name": "continuity_note", "arguments": json.dumps(initial)},
    }])
    db.append_message(
        session_id, "tool",
        record_native_incremental_note_from_tool_call(agent, initial, note_source),
        tool_call_id="old-note", tool_name="continuity_note",
    )
    for index in range(49):
        db.append_message(
            session_id, "user" if index % 2 == 0 else "assistant", _dense_state(200 + index),
        )
    history = db.get_messages_as_conversation(session_id, repair_alternation=False)
    assert len(history) == 165
    assert restore_native_incremental_note(agent, history)["source_cursor"] == 114
    return history


def _sized_args(session_id, source, target):
    args = {
        "objective": f"{_REPLACED}; preserve {_UNRELATED}; identifier=界-77",
        "current_plan": "Validate escaped path state and preserve identifiers.",
        # "current_plan": "Validate quoted \\" path C:\\native\\state and preserve exact identifiers.",
        "next_action": "Run harmless terminal verification after publication.",
        "blockers": ["No deployment; provider acceptance remains parent-owned."],
    }
    note = create_native_incremental_note(
        session_id=session_id, source_messages=source, **args,
    )
    args["current_plan"] += "x" * (target - native_incremental_note_serialized_size(note))
    note = create_native_incremental_note(
        session_id=session_id, source_messages=source, **args,
    )
    assert native_incremental_note_serialized_size(note) == target
    return args


def test_long_state_7714_correction_preserves_prefix_supersession_and_tools(
    tmp_path, monkeypatch, blocking_router,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "compression:\n  enabled: true\n  codex_responses_native: true\n"
        "  native_incremental_handoff: true\n"
    )
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "long-state-recovery"
    db.create_session(session_id, "cli", model="gpt-6-astra")
    with patch("agent.turn_context._maybe_title_session_at_turn_start", return_value=None):
        agent = _make_agent(db, session_id)
    history = _append_long_stale_history(db, agent, session_id)
    original_prefix = deepcopy(history)
    calls = []
    capabilities = []

    def provider(request):
        calls.append(deepcopy(request))
        if request.get("tool_choice") == {"type": "function", "name": "continuity_note"}:
            capabilities.append(agent._native_note_refresh_capability)
            maintenance_count = sum(
                call.get("tool_choice") == {"type": "function", "name": "continuity_note"}
                for call in calls
            )
            if maintenance_count % 2:
                assert "prepared authenticated source" in request["instructions"]
                assert "not per-field caps" in request["instructions"]
                too_large = {**ARGS, "current_plan": _dense_state(999) * 2}
                return reply(NS(
                    type="function_call", id=f"rejected-{maintenance_count}",
                    call_id=f"rejected-long-note-{maintenance_count}",
                    name="continuity_note", arguments=json.dumps(too_large),
                ))
            current = db.get_messages_as_conversation(session_id, repair_alternation=False)
            args = _sized_args(session_id, current, 7714)
            return reply(NS(
                type="function_call", id=f"corrected-{maintenance_count}",
                call_id=f"corrected-7714-note-{maintenance_count}",
                name="continuity_note", arguments=json.dumps(args),
            ))
        if len(calls) == 3:
            return reply(
                NS(type="message", role="assistant", content=[
                    NS(type="output_text", text=_dense_state(500) * 20),
                ]),
                NS(type="function_call", id="route", call_id="route-long", name="tool_call",
                   arguments=json.dumps({"name": "fleet_route_task", "arguments": {
                       "work_shape": "direct", "consequence": "routine",
                       "reason_codes": ["known_short_path"], "proof_target": "focused_test",
                   }})),
                tokens=220000,
            )
        if len(calls) == 6:
            return reply(NS(
                type="function_call", id="terminal", call_id="terminal-long", name="terminal",
                arguments=json.dumps({"command": "printf LONG_STATE_OK", "workdir": str(tmp_path)}),
            ), tokens=600)
        return reply(NS(
            type="message", role="assistant", content=[
                NS(type="output_text", text="LONG_STATE_OK"),
            ], tokens=600,
        ))

    agent._interruptible_api_call = provider
    try:
        result = agent.run_conversation(
            f"Supersede the old goal with {_REPLACED}; retain {_UNRELATED}; do not deploy.",
            conversation_history=history,
        )
        assert result["completed"] and result["final_response"] == "LONG_STATE_OK"
        assert len(calls) == 7
        assert len(capabilities) == 4
        assert capabilities[0] is not capabilities[1]
        assert capabilities[2] is not capabilities[3]
        stored = db.get_messages_as_conversation(session_id, repair_alternation=False)
        assert stored[:len(original_prefix)] == original_prefix
        assert all(f"rejected-long-note-{index}" not in json.dumps(stored) for index in (1, 3))
        assert sum(
            row.get("tool_call_id") == f"corrected-7714-note-{index}"
            for row in stored for index in (2, 4)
        ) == 2
        restored = restore_native_incremental_note(NS(session_id=session_id), stored)
        assert restored is not None
        assert _REPLACED in restored["objective"] and _UNRELATED in restored["objective"]
        assert native_incremental_note_serialized_size(restored) == 7714
        terminal = next(row for row in stored if row.get("tool_call_id") == "terminal-long")
        terminal_result = json.loads(terminal["content"])
        assert terminal_result["exit_code"] == 0 and terminal_result["output"] == "LONG_STATE_OK"
    finally:
        agent.close()
        db.close()


def test_interleaved_session_capabilities_publish_once_without_cross_binding(
    tmp_path, monkeypatch, blocking_router,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "compression:\n  enabled: true\n  codex_responses_native: true\n"
        "  native_incremental_handoff: true\n"
    )
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    agents = []
    try:
        for session_id in ("interleave-a", "interleave-b"):
            db.create_session(session_id, "cli", model="gpt-6-astra")
            agent = _make_agent(db, session_id)
            db.append_message(session_id, "user", f"objective {session_id}")
            db.append_message(session_id, "assistant", "evidence")
            db.append_message(session_id, "user", f"latest {session_id}")
            history = db.get_messages_as_conversation(session_id, repair_alternation=False)
            agent._current_api_request_id = f"{session_id}:api:1"
            capability = issue_native_note_refresh(agent, history)
            capability.bind_request({})
            agents.append((session_id, agent, history, capability))
        # Both capabilities are live before either publication. Publish B then A.
        for session_id, agent, history, capability in reversed(agents):
            response = reply(NS(
                type="function_call", id=f"fc-{session_id}", call_id=f"note-{session_id}",
                name="continuity_note", arguments=json.dumps({
                    **ARGS, "objective": f"objective {session_id}",
                }),
            ))
            execute_native_note_refresh(agent, capability, response, history, session_id)
        for session_id, agent, _history, _capability in agents:
            stored = db.get_messages_as_conversation(session_id, repair_alternation=False)
            assert sum(row.get("tool_call_id") == f"note-{session_id}" for row in stored) == 1
            assert all(f"note-{other}" not in json.dumps(stored) for other in ("interleave-a", "interleave-b") if other != session_id)
            assert restore_native_incremental_note(NS(session_id=session_id), stored)["session_id"] == session_id
    finally:
        for _session_id, agent, _history, _capability in agents:
            agent.close()
        db.close()


def test_failed_refresh_then_new_user_turn_restores_with_fresh_agent(
    tmp_path, monkeypatch, blocking_router,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "compression:\n  enabled: true\n  codex_responses_native: true\n"
        "  native_incremental_handoff: true\n"
    )
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "failed-then-restored"
    db.create_session(session_id, "cli", model="gpt-6-astra")
    agent = _make_agent(db, session_id)
    history = _append_long_stale_history(db, agent, session_id)
    rejected = {**ARGS, "current_plan": _dense_state(700) * 2}
    failed_calls = []
    agent._interruptible_api_call = lambda request: (
        failed_calls.append(deepcopy(request)) or reply(NS(
            type="function_call", id=f"failed-{len(failed_calls)}",
            call_id=f"failed-refresh-{len(failed_calls)}", name="continuity_note",
            arguments=json.dumps(rejected),
        ))
    )
    try:
        failed = agent.run_conversation("First user turn must stop safely.", conversation_history=history)
        assert failed["failed"] is True
        assert len(failed_calls) == 2
        stored_after_failure = db.get_messages_as_conversation(session_id, repair_alternation=False)
        assert "failed-refresh-" not in json.dumps(stored_after_failure)
    finally:
        agent.close()

    fresh = _make_agent(db, session_id)
    fresh_calls = []

    def recovered_provider(request):
        fresh_calls.append(deepcopy(request))
        if request.get("tool_choice") == {"type": "function", "name": "continuity_note"}:
            current = db.get_messages_as_conversation(session_id, repair_alternation=False)
            return reply(NS(
                type="function_call", id="recovered-note", call_id="recovered-note",
                name="continuity_note", arguments=json.dumps(_sized_args(session_id, current, 7600)),
            ))
        if len(fresh_calls) == 2:
            return reply(NS(
                type="function_call", id="route", call_id="route-restored", name="tool_call",
                arguments=json.dumps({"name": "fleet_route_task", "arguments": {
                    "work_shape": "direct", "consequence": "routine",
                    "reason_codes": ["known_short_path"], "proof_target": "focused_test",
                }}),
            ), tokens=600)
        if len(fresh_calls) == 3:
            return reply(NS(
                type="function_call", id="terminal", call_id="terminal-restored", name="terminal",
                arguments=json.dumps({"command": "printf RESTORED_OK", "workdir": str(tmp_path)}),
            ), tokens=600)
        return reply(NS(
            type="message", role="assistant", content=[
                NS(type="output_text", text="RESTORED_OK"),
            ], tokens=600,
        ))

    fresh._interruptible_api_call = recovered_provider
    try:
        assert restore_native_incremental_note(fresh, stored_after_failure)["source_cursor"] == 114
        result = fresh.run_conversation(
            "Second user turn supersedes no unrelated identifiers; continue safely.",
            conversation_history=stored_after_failure,
        )
        assert result["completed"] and result["final_response"] == "RESTORED_OK"
        assert len(fresh_calls) == 4
        stored = db.get_messages_as_conversation(session_id, repair_alternation=False)
        assert sum(row.get("tool_call_id") == "recovered-note" for row in stored) == 1
        terminal = next(row for row in stored if row.get("tool_call_id") == "terminal-restored")
        assert json.loads(terminal["content"])["exit_code"] == 0
        assert restore_native_incremental_note(NS(session_id=session_id), stored) is not None
    finally:
        fresh.close()
        db.close()
