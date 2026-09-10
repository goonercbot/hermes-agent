"""Real budget mutation and SQLite note authentication (no provider calls)."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest

from agent import tool_executor
from agent.native_incremental_handoff import (
    _projection_source_for_messages,
    bind_native_incremental_replay_projection,
    record_native_incremental_note_from_tool_call,
    restore_native_incremental_note,
)
from hermes_state import SessionDB
from tools.budget_config import BudgetConfig

ARGS = dict(objective="Keep evidence", current_plan="Verify locally",
            next_action="Report", blockers=[])


def budget(agent, messages, count, config=BudgetConfig()):
    # Also exercises the original implementation when reproducing the defect.
    wrapper = getattr(tool_executor, "_enforce_turn_budget_with_native_projection", None)
    if wrapper is None:
        return tool_executor.enforce_turn_budget(messages[-count:], config=config)
    return wrapper(agent, messages, count, config=config)


def append_note(db, agent, messages, call_id):
    call = dict(id=call_id, type="function", function=dict(
        name="continuity_note", arguments=json.dumps(ARGS)))
    db.append_message(agent.session_id, "assistant", "", tool_calls=[call])
    messages.append(db.get_messages_as_conversation(agent.session_id)[-1])
    result = record_native_incremental_note_from_tool_call(agent, ARGS, messages)
    assert json.loads(result).get("ok"), result
    db.append_message(agent.session_id, "tool", result,
                      tool_call_id=call_id, tool_name="continuity_note")
    messages.append(db.get_messages_as_conversation(agent.session_id)[-1])
    return json.loads(result)["note"]


def test_budget_note_persistence_and_repeated_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = NS(session_id="budget", native_incremental_handoff_enabled=True)
    db.create_session(agent.session_id, "cli")
    db.append_message(agent.session_id, "user", "Keep evidence")
    messages = db.get_messages_as_conversation(agent.session_id)
    try:
        old_note = append_note(db, agent, messages, "old")
        for turn in range(2):
            calls = [dict(id=f"read-{turn}-{i}", type="function", function=dict(
                name="read_file", arguments='{"path":"fixture"}')) for i in range(4)]
            db.append_message(agent.session_id, "assistant", "", tool_calls=calls)
            messages.append(db.get_messages_as_conversation(agent.session_id)[-1])
            for call in calls:
                db.append_message(agent.session_id, "tool", "x" * 61132,
                                  tool_call_id=call["id"], tool_name="read_file")
                messages.append(db.get_messages_as_conversation(agent.session_id)[-1])
            source = deepcopy(db.get_messages_as_conversation(agent.session_id))
            budget(agent, messages, 4)
            assert len(messages[-4]["content"]) < 61132
            assert db.get_messages_as_conversation(agent.session_id) == source
            assert restore_native_incremental_note(agent, messages) == old_note
            old_note = append_note(db, agent, messages, f"note-{turn}")
            fresh = NS(session_id=agent.session_id)
            stored = db.get_messages_as_conversation(agent.session_id)
            assert restore_native_incremental_note(fresh, stored) == old_note
            tampered = deepcopy(stored)
            tampered[0]["content"] = "tampered"
            assert restore_native_incremental_note(fresh, tampered) is None
    finally:
        db.close()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("size", [10, 61132])
def test_non_native_and_noop(native, size, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = NS(native_incremental_handoff_enabled=native)
    messages = [dict(role="tool", content="x" * size, tool_call_id=str(i)) for i in range(4)]
    before = deepcopy(messages)
    budget(agent, messages, 4)
    if not native or size == 10:
        assert not hasattr(agent, "_native_incremental_replay_projection")
    if size == 10:
        assert messages == before
    else:
        assert messages != before


def test_invalid_projection_is_not_reauthenticated(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = NS(native_incremental_handoff_enabled=True)
    messages = [dict(role="tool", content="x" * 61132, tool_call_id=str(i)) for i in range(4)]
    assert bind_native_incremental_replay_projection(agent, source_messages=messages, replay_messages=messages)
    previous = deepcopy(agent._native_incremental_replay_projection)
    messages[0]["content"] += "tamper"
    budget(agent, messages, 4)
    assert agent._native_incremental_replay_projection == previous
    assert _projection_source_for_messages(agent, messages)[0] is None


def test_post_budget_steer_is_not_laundered(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = NS(native_incremental_handoff_enabled=True)
    messages = [dict(role="tool", content="x" * 61132, tool_call_id=str(i)) for i in range(4)]
    budget(agent, messages, 4)
    messages[-1]["content"] += "\nnew user steer"
    budget(agent, messages, 4)
    assert messages[-1]["content"].endswith("new user steer")
    assert _projection_source_for_messages(agent, messages)[0] is None
