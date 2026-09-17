"""Production-shaped unequal replay projection refresh regression."""
from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace as NS

from agent.native_incremental_handoff import (
    _projection_cursor_for_source_cursor,
    _projection_source_for_messages,
    _staged_note,
    bind_native_incremental_replay_projection,
    create_native_incremental_note,
    record_native_incremental_note,
    restore_native_incremental_note,
)
from agent.native_note_refresh import (
    close_native_note_refresh,
    execute_native_note_refresh,
)
from tests.run_agent.test_native_continuity_lifecycle import ARGS
from tests.run_agent.test_native_note_refresh import blocking_router, reply


def _refresh_reply(call_id):
    return reply(NS(
        type="function_call", id=f"fc-{call_id}", call_id=call_id,
        name="continuity_note", arguments=json.dumps(ARGS),
    ))


def _new_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "compression:\n  enabled: true\n  codex_responses_native: true\n"
        "  native_incremental_handoff: true\n"
    )
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "unequal-projection-boundary"
    db.create_session(sid, "subagent", model="gpt-5.6-terra")
    agent = AIAgent(
        api_key="fixture", base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses", model="gpt-5.6-terra", provider="openai-codex",
        session_db=db, session_id=sid, platform="subagent", quiet_mode=True,
        skip_memory=True, skip_context_files=True, skip_background_review=True,
        enabled_toolsets=["continuity"],
    )
    agent._end_session_on_close = False  # type: ignore[attr-defined]
    return agent, db, sid


def _append_shared_tail(db, sid, replay, marker):
    db.append_message(sid, "user", f"{marker} user correction")
    db.append_message(sid, "assistant", f"{marker} durable evidence " * 9000)
    durable = db.get_messages_as_conversation(sid, repair_alternation=False)
    replay.extend(deepcopy(durable[-2:]))


def test_note_refresh_keeps_pre_maintenance_unequal_projection_boundary(
    tmp_path, monkeypatch, blocking_router,
):
    """Alternation repair may shrink replay without moving a published note interior."""
    from agent.native_incremental_handoff import prepare_native_note_refresh_request

    agent, db, sid = _new_agent(tmp_path, monkeypatch)
    try:
        # SessionDB's production restore repair merges these two durable user
        # rows only in the replay. The canonical transcript is never rewritten.
        db.append_message(sid, "user", "durable objective")
        db.append_message(sid, "user", "durable correction")
        db.append_message(sid, "assistant", "initial verified state")
        canonical_prefix = db.get_messages_as_conversation(sid, repair_alternation=False)
        repaired_prefix = db.get_messages_as_conversation(sid, repair_alternation=True)
        assert len(canonical_prefix) == 3 and len(repaired_prefix) == 2
        assert canonical_prefix[:2] != repaired_prefix[:2]

        initial_note = create_native_incremental_note(
            session_id=sid, source_messages=canonical_prefix, **ARGS,
        )
        record_native_incremental_note(agent, initial_note, canonical_prefix)
        assert bind_native_incremental_replay_projection(
            agent, source_messages=canonical_prefix, replay_messages=repaired_prefix,
        )
        assert _staged_note(agent, repaired_prefix) == initial_note
        assert _projection_cursor_for_source_cursor(
            initial_note["source_cursor"], source_base_count=3, replay_base_count=2,
        ) == 2

        _append_shared_tail(db, sid, repaired_prefix, "first")
        canonical_before_first = db.get_messages_as_conversation(sid, repair_alternation=False)
        assert len(canonical_before_first) == 5 and len(repaired_prefix) == 4
        agent._current_api_request_id = "unequal:api:1"  # type: ignore[attr-defined]
        first_request = {"instructions": "ordinary", "tools": []}
        first = prepare_native_note_refresh_request(agent, repaired_prefix, first_request)
        assert first is not False
        first.bind_request(first_request)
        execute_native_note_refresh(agent, first, _refresh_reply("first-note"), repaired_prefix, sid)

        first_note = getattr(agent, "_native_incremental_handoff_note", None)
        assert first_note is not None and first_note["source_cursor"] == 5
        assert _staged_note(agent, repaired_prefix) == first_note
        assert db.get_messages_as_conversation(sid, repair_alternation=False)[:5] == canonical_before_first

        # The immediate same-agent ordinary request must continue normally, not
        # lose its note and re-enter the maintenance guard.
        agent._current_api_request_id = "unequal:api:2"  # type: ignore[attr-defined]
        assert prepare_native_note_refresh_request(
            agent, repaired_prefix, {"instructions": "ordinary", "tools": []},
        ) is False

        _append_shared_tail(db, sid, repaired_prefix, "second")
        agent._current_api_request_id = "unequal:api:3"  # type: ignore[attr-defined]
        second_request = {"instructions": "ordinary", "tools": []}
        second = prepare_native_note_refresh_request(agent, repaired_prefix, second_request)
        assert second is not False
        second.bind_request(second_request)
        execute_native_note_refresh(agent, second, _refresh_reply("second-note"), repaired_prefix, sid)
        second_note = getattr(agent, "_native_incremental_handoff_note", None)
        assert second_note is not None and second_note["source_cursor"] == 9
        assert _staged_note(agent, repaired_prefix) == second_note

        stored = db.get_messages_as_conversation(sid, repair_alternation=False)
        assert restore_native_incremental_note(NS(session_id=sid), stored) == second_note
        db.close()
        db = type(db)(db_path=tmp_path / "state.db")
        assert restore_native_incremental_note(
            NS(session_id=sid), db.get_messages_as_conversation(sid, repair_alternation=False),
        ) == second_note

        tampered_replay = deepcopy(repaired_prefix)
        tampered_replay[0]["content"] = "forged replay"
        assert _projection_source_for_messages(agent, tampered_replay) == (None, None)
        tampered_source = deepcopy(stored)
        tampered_source[0]["content"] = "forged source"
        assert restore_native_incremental_note(NS(session_id=sid), tampered_source) is None
        assert _projection_cursor_for_source_cursor(
            8, source_base_count=9, replay_base_count=8,
        ) is None

        equal = NS(session_id=sid)
        assert bind_native_incremental_replay_projection(
            equal, source_messages=stored, replay_messages=stored,
        )
        assert restore_native_incremental_note(equal, stored) == second_note
    finally:
        close_native_note_refresh(agent)
        agent.close()
        db.close()
