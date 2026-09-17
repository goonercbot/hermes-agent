"""Native-continuity lifecycle regressions without private production history."""
from __future__ import annotations

import json
import logging
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from agent.native_incremental_handoff import (
    bind_native_incremental_replay_projection,
    create_native_incremental_note,
    _native_incremental_operation_messages,
    _protected_tail_since_note,
    native_note_refresh_required,
    record_native_incremental_note_from_tool_call,
    restore_native_incremental_note,
)
from agent.native_compaction import NATIVE_COMPACTION_METADATA_KEY, native_continuity_boundary_fence, validate_persisted_native_compaction_history
from agent.native_note_refresh import (
    NativeNoteRefreshFailure,
    execute_native_note_refresh,
    issue_native_note_refresh,
)
from tests.run_agent.test_native_note_refresh import blocking_router, reply

ARGS = {
    "objective": "Keep the latest user correction",
    "current_plan": "Preserve queued completion evidence",
    "next_action": "Continue without replaying external work",
    "blockers": [],
}
SID = "synthetic-native-lifecycle"


def _record(source):
    return record_native_incremental_note_from_tool_call(
        NS(session_id=SID, native_incremental_handoff_enabled=True), ARGS, source
    )


def _legacy_poisoned_history():
    """Faithful shape: old valid note + async user/result + same-length bad source."""
    history = [
        {"role": "user", "content": "initial objective"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "old", "type": "function",
            "function": {"name": "continuity_note", "arguments": json.dumps(ARGS)},
        }]},
    ]
    history.append({"role": "tool", "content": _record(history), "tool_call_id": "old", "name": "continuity_note"})
    history.extend([
        {"role": "assistant", "content": "ordinary completion"},
        {"role": "user", "content": "queued follow-up"},
        {"role": "assistant", "content": "async completion"},
    ])
    projected = deepcopy(history)
    projected[-1]["content"] = "cleaned async completion"
    history.append({"role": "assistant", "content": "", "tool_calls": [{
        "id": "poison", "type": "function",
        "function": {"name": "continuity_note", "arguments": json.dumps(ARGS)},
    }]})
    history.append({"role": "tool", "content": _record(projected), "tool_call_id": "poison", "name": "continuity_note"})
    return history


def test_restart_recovers_only_legacy_projection_pair_without_history_loss():
    history = _legacy_poisoned_history()
    before = deepcopy(history)

    restored = restore_native_incremental_note(NS(session_id=SID), history)

    assert restored is not None
    assert restored["source_cursor"] == 2
    assert history == before  # the failed pair and every user/task/tool row remain auditable


@pytest.mark.parametrize("tamper", ["payload_field", "malformed_payload", "call_arguments", "[]", "null", "0", "true", "\"text\""])
def test_terminal_malformed_or_tampered_note_fails_closed(tamper):
    history = _legacy_poisoned_history()
    if tamper == "payload_field":
        payload = json.loads(history[-1]["content"])
        payload["note"]["objective"] = "tampered"
        history[-1]["content"] = json.dumps(payload)
    elif tamper == "malformed_payload":
        history[-1]["content"] = "not-json"
    elif tamper in {"[]", "null", "0", "true", "\"text\""}:
        history[-1]["content"] = tamper
    else:
        history[-2]["tool_calls"][0]["function"]["arguments"] = json.dumps({**ARGS, "next_action": "tampered"})

    assert restore_native_incremental_note(NS(session_id=SID), history) is None


def _synthetic_checkpoint_child_with_historical_pair(*, pair_in_tail):
    """Public-data shape of the copied child: carrier, handoff, old pair."""
    carried_note = create_native_incremental_note(
        session_id=SID,
        source_messages=[{"role": "user", "content": "public original objective"}],
        **ARGS,
    )
    historical_note = create_native_incremental_note(
        session_id=SID,
        source_messages=[{"role": "user", "content": "older public objective"}],
        **ARGS,
    )
    handoff = "NATIVE_INCREMENTAL_NOTE\n" + json.dumps(
        carried_note, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    historical_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "historical-carried-note",
            "type": "function",
            "function": {"name": "continuity_note", "arguments": json.dumps(ARGS)},
        }],
    }
    historical_result = {
        "role": "tool",
        "name": "continuity_note",
        "tool_call_id": "historical-carried-note",
        "content": json.dumps({
            "authenticated_by": "hermes.continuity_note.v1",
            "note": historical_note,
        }),
    }
    stale_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "historical-note",
            "type": "function",
            "function": {"name": "continuity_note", "arguments": json.dumps({
                **ARGS, "next_action": "Stale historical action",
            })},
        }],
    }
    stale_result = {
        "role": "tool",
        "name": "continuity_note",
        "tool_call_id": "historical-note",
        "content": json.dumps({
            "authenticated_by": "hermes.continuity_note.v1",
            "note": carried_note,
        }),
    }
    tail = [{"role": "assistant", "content": "public retained completion"}]
    if pair_in_tail:
        tail.extend([historical_call, historical_result, stale_call, stale_result])
    metadata = {
        "version": 2,
        "identity": "synthetic-public-checkpoint",
        "handoff": handoff,
        "tail_count": len(tail),
        "tail_fence": native_continuity_boundary_fence(tail),
    }
    history = [
        {"role": "assistant", "content": "", "codex_reasoning_items": [{
            "type": "compaction",
            "encrypted_content": "synthetic-public-ciphertext",
            NATIVE_COMPACTION_METADATA_KEY: metadata,
        }]},
        {"role": "user", "content": handoff},
        *tail,
    ]
    if not pair_in_tail:
        history.extend([stale_call, stale_result])
    return history


def test_checkpoint_bound_historical_maintenance_pair_restores_carrier_without_mutation():
    history = _synthetic_checkpoint_child_with_historical_pair(pair_in_tail=True)
    before = deepcopy(history)

    assert validate_persisted_native_compaction_history(history)
    restored = restore_native_incremental_note(NS(session_id=SID), history)

    assert restored is not None
    assert restored["objective"] == ARGS["objective"]
    assert restored["source_cursor"] == 2
    assert history == before


@pytest.mark.parametrize("ordinary_row", [
    {"role": "user", "content": "Continue the retained task."},
    {"role": "assistant", "content": "ordinary resumed completion"},
])
def test_checkpoint_boundary_excludes_historical_pairs_after_ordinary_append(ordinary_row):
    """The sealed boundary survives the first resumed user or assistant row."""
    history = _synthetic_checkpoint_child_with_historical_pair(pair_in_tail=True)
    history.append(ordinary_row)
    before = deepcopy(history)

    restored = restore_native_incremental_note(NS(session_id=SID), history)

    assert restored is not None
    assert restored["source_cursor"] == 2
    assert history == before


def test_new_valid_direct_note_after_sealed_boundary_supersedes_carrier():
    history = _synthetic_checkpoint_child_with_historical_pair(pair_in_tail=True)
    history.append({"role": "user", "content": "new correction after restart"})
    refreshed = {**ARGS, "objective": "New post-checkpoint objective"}
    history.append({"role": "assistant", "content": "", "tool_calls": [{
        "id": "post-boundary-note", "type": "function",
        "function": {"name": "continuity_note", "arguments": json.dumps(refreshed)},
    }]})
    result = record_native_incremental_note_from_tool_call(
        NS(session_id=SID, native_incremental_handoff_enabled=True), refreshed, history,
    )
    history.append({
        "role": "tool", "name": "continuity_note", "tool_call_id": "post-boundary-note",
        "content": result,
    })
    before = deepcopy(history)

    restored = restore_native_incremental_note(NS(session_id=SID), history)

    assert restored is not None
    assert restored["objective"] == refreshed["objective"]
    assert restored["source_cursor"] == len(history) - 1
    assert history == before


def test_unbound_terminal_maintenance_pair_still_fails_closed():
    history = _synthetic_checkpoint_child_with_historical_pair(pair_in_tail=False)

    assert validate_persisted_native_compaction_history(history)
    assert restore_native_incremental_note(NS(session_id=SID), history) is None


def test_divergent_canonical_source_rolls_back_before_pair_flush(tmp_path):
    from hermes_state import SessionDB

    messages = [{"role": "user", "content": "durable queued input"}]
    replay = deepcopy(messages)
    source = deepcopy(messages)
    source[0]["content"] = "stale cleaned input"
    flushed = []
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(SID, "subagent", model="gpt-5.6-terra")
    db.append_message(SID, "user", messages[0]["content"])
    before = [row["content"] for row in db.get_messages(SID)]

    agent = NS(
        session_id=SID,
        _current_api_request_id="turn:api:1",
        _interrupt_requested=False,
        _session_db=db,
        _persist_disabled=False,
        _native_incremental_replay_projection={
            "source": source,
            "source_fence": __import__("agent.native_incremental_handoff", fromlist=["_note_fence"])._note_fence(source),
            "replay": replay,
            "replay_fence": __import__("agent.native_incremental_handoff", fromlist=["_note_fence"])._note_fence(replay),
        },
        _flush_messages_to_session_db=lambda rows: flushed.append(deepcopy(rows)) or True,
    )
    try:
        capability = issue_native_note_refresh(agent, messages)
        response = {
            "status": "completed",
            "output": [{"type": "function_call", "name": "continuity_note", "call_id": "new-note", "arguments": json.dumps(ARGS)}],
        }

        with pytest.raises(NativeNoteRefreshFailure, match="canonical maintenance source diverged"):
            execute_native_note_refresh(agent, capability, response, messages, SID)

        assert messages == replay
        assert flushed == []
        assert [row["content"] for row in db.get_messages(SID)] == before
    finally:
        db.close()


@pytest.mark.parametrize("raw_history", [
    [
        {"role": "user", "content": "durable context"},
        {"role": "assistant", "content": "  completed work\n"},
    ],
    [
        {"role": "user", "content": "durable context"},
        {"role": "assistant", "content": "completed work"},
        {"role": "user", "content": "\nqueued follow-up  "},
    ],
], ids=["assistant-edge-whitespace", "user-edge-whitespace"])
def test_native_note_refresh_uses_sqlite_canonical_source_without_rewriting_rows(
    tmp_path, monkeypatch, blocking_router, raw_history,
):
    """Only the loader's top-level user/assistant text cleanup may bridge views."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "compression:\n  enabled: true\n  codex_responses_native: true\n"
        "  native_incremental_handoff: true\n"
    )
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "canonical-whitespace-source"
    db.create_session(sid, "subagent", model="gpt-5.6-terra")
    agent = AIAgent(
        api_key="fixture", base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses", model="gpt-5.6-terra", provider="openai-codex",
        session_db=db, session_id=sid, platform="subagent", quiet_mode=True,
        skip_memory=True, skip_context_files=True, skip_background_review=True,
        enabled_toolsets=["continuity"],
    )
    agent._end_session_on_close = False  # type: ignore[attr-defined]
    try:
        for row in raw_history:
            db.append_message(sid, row["role"], row["content"])
        persisted_before = deepcopy(db.get_messages(sid))
        history = deepcopy(db.get_messages_as_conversation(sid, repair_alternation=False))
        for replay_row, raw_row in zip(history, raw_history):
            replay_row["content"] = raw_row["content"]

        agent._current_api_request_id = "canonical:maintenance:1"  # type: ignore[attr-defined]
        capability = issue_native_note_refresh(agent, history)
        capability.bind_request({})
        response = reply(NS(
            type="function_call", id="fc-note", call_id="new-note",
            name="continuity_note", arguments=json.dumps(ARGS),
        ))
        execute_native_note_refresh(agent, capability, response, history, sid)

        stored = db.get_messages_as_conversation(sid, repair_alternation=False)
        assert restore_native_incremental_note(NS(session_id=sid), stored) is not None
        assert db.get_messages(sid)[:len(raw_history)] == persisted_before
    finally:
        agent.close()
        db.close()


def test_untrimmed_replay_refresh_advances_next_request_and_compacts_canonical_source(
    tmp_path, monkeypatch, blocking_router,
):
    """A published refresh must carry its DB source into both next consumers."""
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from agent.native_incremental_handoff import (
        _projection_source_for_messages,
        bind_native_incremental_replay_projection,
        native_incremental_compact_context,
        prepare_native_note_refresh_request,
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "compression:\n  enabled: true\n  codex_responses_native: true\n"
        "  native_incremental_handoff: true\n"
    )
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "canonical-refresh-next-consumer"
    db.create_session(sid, "subagent", model="gpt-5.6-terra")
    agent = AIAgent(
        api_key="fixture", base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses", model="gpt-5.6-terra", provider="openai-codex",
        session_db=db, session_id=sid, platform="subagent", quiet_mode=True,
        skip_memory=True, skip_context_files=True, skip_background_review=True,
        enabled_toolsets=["continuity"],
    )
    agent._end_session_on_close = False  # type: ignore[attr-defined]
    try:
        db.append_message(sid, "user", "durable objective")
        db.append_message(sid, "assistant", "initial verified state")
        db.append_message(sid, "assistant", "", tool_calls=[{
            "id": "old-note", "type": "function",
            "function": {"name": "continuity_note", "arguments": json.dumps(ARGS)},
        }])
        old_source = db.get_messages_as_conversation(sid, repair_alternation=False)
        old_result = record_native_incremental_note_from_tool_call(agent, ARGS, old_source)
        db.append_message(sid, "tool", old_result, tool_call_id="old-note", tool_name="continuity_note")
        # The durable loader strips these edge bytes. The provider replay keeps
        # them until publication, which is the real divergence boundary.
        stale_tail = "verified stale evidence " * 7000 + " \n"
        db.append_message(sid, "assistant", stale_tail)
        durable_before_refresh = db.get_messages_as_conversation(sid, repair_alternation=False)
        replay = deepcopy(durable_before_refresh)
        replay[-1]["content"] = stale_tail
        assert replay[-1]["content"] != durable_before_refresh[-1]["content"]
        assert bind_native_incremental_replay_projection(
            agent, source_messages=durable_before_refresh, replay_messages=replay,
        )
        assert restore_native_incremental_note(agent, replay) is not None

        agent._current_api_request_id = "canonical:api:1"  # type: ignore[attr-defined]
        first_request = {"instructions": "ordinary", "tools": []}
        first = prepare_native_note_refresh_request(agent, replay, first_request)
        assert first is not False
        first.bind_request(first_request)
        execute_native_note_refresh(
            agent, first,
            reply(NS(type="function_call", id="fc-note", call_id="fresh-note",
                     name="continuity_note", arguments=json.dumps(ARGS))),
            replay, sid,
        )

        source_after_refresh, cursor = _projection_source_for_messages(agent, replay)
        assert source_after_refresh == db.get_messages_as_conversation(sid, repair_alternation=False)
        assert cursor == len(replay)
        assert replay[-3]["content"] == stale_tail  # no historical rewrite
        assert not native_note_refresh_required(agent, replay)

        # The following physical request uses the same turn identity. Before
        # this repair it saw the raw replay as a stale/no-note identity and the
        # guard stopped it rather than issuing the required refresh.
        next_tail = "new durable evidence " * 7000 + " \n"
        db.append_message(sid, "assistant", next_tail)
        replay.append({"role": "assistant", "content": next_tail})
        agent._current_api_request_id = "canonical:api:2"  # type: ignore[attr-defined]
        second_request = {"instructions": "ordinary", "tools": []}
        second = prepare_native_note_refresh_request(agent, replay, second_request)
        assert second is not False
        assert second.canonical_source_fence != first.canonical_source_fence
        second.close()

        # A separate fresh canonical projection proves the automatic native
        # compaction request and result use DB text rather than raw replay text.
        compact_replay = replay[:-1]
        agent._current_api_request_id = "canonical:compact:1"  # type: ignore[attr-defined]
        provider_requests = []
        agent._interruptible_api_call = lambda request: provider_requests.append(deepcopy(request)) or reply(
            NS(type="compaction", id="checkpoint", encrypted_content="public-checkpoint"),
            model="gpt-5.6-luna",
        )
        compacted = native_incremental_compact_context(agent, compact_replay)
        assert compacted != compact_replay
        assert len(provider_requests) == 1
        wire = json.dumps(provider_requests[0]["input"], ensure_ascii=False)
        assert stale_tail.strip() in wire
        assert stale_tail not in wire
        assert all(row.get("content") != stale_tail for row in compacted)
        validate_persisted_native_compaction_history(compacted)
    finally:
        agent.close()
        db.close()


def test_native_note_transition_diagnostics_are_correlated_private_and_neutral(
    tmp_path, monkeypatch, caplog, blocking_router,
):
    """Real prepare/execute boundaries expose state without logging note contents."""
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from agent.native_incremental_handoff import prepare_native_note_refresh_request
    import agent.native_incremental_handoff as handoff
    import agent.native_note_refresh as refresh

    private_session = "PRIVATE_SESSION_SENTINEL"
    private_request = "PRIVATE_REQUEST_SENTINEL"
    private_message = "PRIVATE_MESSAGE_SENTINEL"
    private_note = {
        "objective": "PRIVATE_OBJECTIVE_SENTINEL",
        "current_plan": "PRIVATE_PLAN_SENTINEL",
        "next_action": "PRIVATE_ACTION_SENTINEL",
        "blockers": ["PRIVATE_BLOCKER_SENTINEL"],
    }
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "compression:\n  enabled: true\n  codex_responses_native: true\n"
        "  native_incremental_handoff: true\n"
    )
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(private_session, "subagent", model="gpt-5.6-terra")
    agent = AIAgent(
        api_key="fixture", base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses", model="gpt-5.6-terra", provider="openai-codex",
        session_db=db, session_id=private_session, platform="subagent", quiet_mode=True,
        skip_memory=True, skip_context_files=True, skip_background_review=True,
        enabled_toolsets=["continuity"],
    )
    agent._end_session_on_close = False  # type: ignore[attr-defined]
    try:
        db.append_message(private_session, "user", private_message)
        db.append_message(private_session, "assistant", "public initial evidence")
        db.append_message(private_session, "assistant", "", tool_calls=[{
            "id": "old-note", "type": "function",
            "function": {"name": "continuity_note", "arguments": json.dumps(private_note)},
        }])
        source = db.get_messages_as_conversation(private_session, repair_alternation=False)
        db.append_message(
            private_session, "tool",
            record_native_incremental_note_from_tool_call(agent, private_note, source),
            tool_call_id="old-note", tool_name="continuity_note",
        )
        db.append_message(private_session, "assistant", "public stale evidence " * 9000)
        replay = db.get_messages_as_conversation(private_session, repair_alternation=False)
        assert restore_native_incremental_note(agent, replay) is not None

        with caplog.at_level(logging.INFO, logger="agent.native_continuity.diagnostics"):
            for number in (1, 2):
                setattr(agent, "_current_api_request_id", f"{private_request}:api:{number}")
                request = {"instructions": "ordinary", "tools": []}
                capability = prepare_native_note_refresh_request(agent, replay, request)
                assert capability is not False
                capability.bind_request(request)
                execute_native_note_refresh(
                    agent, capability,
                    reply(NS(type="function_call", id=f"fc-{number}", call_id=f"note-{number}",
                             name="continuity_note", arguments=json.dumps(private_note))),
                    replay, private_session,
                )
                if number == 1:
                    db.append_message(private_session, "assistant", "public next evidence " * 9000)
                    replay.append({"role": "assistant", "content": "public next evidence " * 9000})

            # This second physical request shares an authenticated cursor with
            # the first and must preserve the existing guard rejection.
            db.append_message(private_session, "assistant", "public guard evidence " * 9000)
            replay.append({"role": "assistant", "content": "public guard evidence " * 9000})
            setattr(agent, "_current_api_request_id", f"{private_request}-guard:api:1")
            assert prepare_native_note_refresh_request(agent, replay, {"instructions": "ordinary", "tools": []})
            setattr(agent, "_current_api_request_id", f"{private_request}-guard:api:2")
            with pytest.raises(NativeNoteRefreshFailure, match="did not advance"):
                prepare_native_note_refresh_request(agent, replay, {"instructions": "ordinary", "tools": []})

        events = [
            json.loads(record.getMessage().split(" ", 1)[1])
            for record in caplog.records
            if record.name == "agent.native_continuity.diagnostics"
        ]
        pairs = {(event["event"], event["disposition"]) for event in events}
        assert ("native_note_issue", "issued") in pairs
        assert ("native_note_execute", "entered") in pairs
        assert ("native_note_publication", "flush_accepted") in pairs
        assert ("native_note_publication", "projection_bound") in pairs
        assert ("native_note_execute", "succeeded") in pairs
        assert ("native_note_prepare", "guard_rejected") in pairs
        assert all(set(event) <= {
            "event", "disposition", "message_count", "source_cursor", "capability_present",
            "projection_present", "request_digest", "session_digest", "prefix_digest",
            "flush_accepted", "dispatched", "turn_digest", "staged_note_present",
            "staged_source_cursor", "staged_prefix_digest",
        } for event in events)
        assert [event["source_cursor"] for event in events if event["event"] == "native_note_execute"
                and event["disposition"] == "succeeded"] == sorted(
                    event["source_cursor"] for event in events if event["event"] == "native_note_execute"
                    and event["disposition"] == "succeeded"
                )
        diagnostic_text = "\n".join(record.getMessage() for record in caplog.records)
        for sentinel in (*private_note.values(), private_session, private_request, private_message):
            if isinstance(sentinel, list):
                for item in sentinel:
                    assert item not in diagnostic_text
            else:
                assert sentinel not in diagnostic_text

        # A durable save followed by readback invalidation must emit the save
        # and fixed failure disposition, without exposing the raised detail.
        db.append_message(private_session, "assistant", "public failure evidence " * 9000)
        replay.append({"role": "assistant", "content": "public failure evidence " * 9000})
        setattr(agent, "_current_api_request_id", f"{private_request}-failure:api:1")
        failed = prepare_native_note_refresh_request(agent, replay, {"instructions": "ordinary", "tools": []})
        assert failed is not False
        failed.bind_request({})
        with monkeypatch.context() as patched:
            patched.setattr(refresh, "restore_native_incremental_note", lambda *_a, **_k: None)
            with pytest.raises(NativeNoteRefreshFailure, match="readback failed"):
                execute_native_note_refresh(
                    agent, failed,
                    reply(NS(type="function_call", id="fc-failure", call_id="note-failure",
                             name="continuity_note", arguments=json.dumps(private_note))),
                    replay, private_session,
                )
        failure_events = [
            json.loads(record.getMessage().split(" ", 1)[1])
            for record in caplog.records
            if record.name == "agent.native_continuity.diagnostics"
        ]
        assert ("native_note_publication", "flush_accepted") in {
            (event["event"], event["disposition"]) for event in failure_events
        }
        assert failure_events[-1]["disposition"] == "failed_after_flush_accepted"
        failed_request = handoff._native_note_diagnostic_digest(f"{private_request}-failure:api:1")
        failed_dispositions = [event["disposition"] for event in failure_events
                               if event.get("request_digest") == failed_request]
        assert "flush_accepted" in failed_dispositions
        assert "readback_validated" not in failed_dispositions
        assert "succeeded" not in failed_dispositions
        assert all("persisted" not in event for event in failure_events)
        failure_text = "\n".join(record.getMessage() for record in caplog.records)
        for sentinel in (*private_note.values(), private_session, private_request, private_message):
            for value in sentinel if isinstance(sentinel, list) else [sentinel]:
                assert value not in failure_text

        # Diagnostics failures are inert even on the production preparation path.
        from tests.run_agent.test_native_incremental_handoff import _agent, _note
        neutral_agent, _ = _agent([])
        neutral_agent._current_api_request_id = "neutral:api:1"
        neutral_messages = [{"role": "user", "content": "fixed synthetic input"}]
        _note(neutral_agent, neutral_messages)
        neutral_messages.append({"role": "assistant", "content": "fixed evidence " * 30000})
        with monkeypatch.context() as patched:
            patched.setattr(handoff._diagnostic_logger, "info", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("private log failure")))
            neutral = prepare_native_note_refresh_request(
                neutral_agent, neutral_messages, {"instructions": "ordinary", "tools": []},
            )
        assert neutral is not False
        neutral.close()
    finally:
        agent.close()
        db.close()


def test_native_note_diagnostics_do_not_authenticate_disabled_route(monkeypatch, caplog):
    import agent.native_incremental_handoff as handoff
    from tests.run_agent.test_native_incremental_handoff import _agent

    agent, _ = _agent([])
    agent._current_api_request_id = "private-turn:api:1"
    agent._native_incremental_handoff_projection_cursor = 77
    agent._native_incremental_handoff_note = {
        "source_cursor": 12, "source_prefix_fence": "private-prefix",
    }
    monkeypatch.setattr(handoff, "native_incremental_continuity_capable", lambda *_a, **_k: False)
    def forbidden(*_a, **_k):
        pytest.fail("diagnostics must not invoke stateful authentication")
    monkeypatch.setattr(handoff, "_staged_note", forbidden)
    with caplog.at_level(logging.INFO, logger="agent.native_continuity.diagnostics"):
        assert handoff.prepare_native_note_refresh_request(agent, [], {}) is False
    assert agent._native_incremental_handoff_projection_cursor == 77
    events = [json.loads(r.getMessage().split(" ", 1)[1]) for r in caplog.records
              if r.name == "agent.native_continuity.diagnostics"]
    assert [e["disposition"] for e in events] == ["attempted", "not_required"]
    assert all(e["staged_source_cursor"] == 12 for e in events)
    assert all("source_cursor" not in e for e in events)
    assert all(e["turn_digest"] == handoff._native_note_diagnostic_digest("private-turn") for e in events)
    assert "private-turn" not in caplog.text and "private-prefix" not in caplog.text


def test_native_canonical_source_keeps_sqlite_structured_content_exact(tmp_path):
    from hermes_state import SessionDB
    from agent.native_incremental_handoff import _note_fence
    from agent.native_note_refresh import _native_canonical_source_messages

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(SID, "subagent", model="gpt-5.6-terra")
        structured = [{"type": "input_text", "text": "  structured edge text\n"}]
        db.append_message(SID, "user", structured)  # type: ignore[arg-type]
        persisted = db.get_messages_as_conversation(SID, repair_alternation=False)

        assert _native_canonical_source_messages(deepcopy(persisted), persisted) == persisted
        assert _note_fence(persisted) == _note_fence(
            _native_canonical_source_messages(deepcopy(persisted), persisted)
        )
    finally:
        db.close()


def test_persistence_disabled_fork_is_never_maintenance_eligible():
    fork = NS(_persist_disabled=True, _session_db=None)
    assert not native_note_refresh_required(fork, [{"role": "user", "content": "x" * 200000}])


def _agent_with_authenticated_prefix_note(messages, *, cursor=2):
    agent = NS(session_id=SID, native_incremental_handoff_enabled=True)
    recorded = json.loads(record_native_incremental_note_from_tool_call(agent, ARGS, messages[:cursor]))
    assert recorded["ok"] is True
    assert agent._native_incremental_handoff_note["source_cursor"] == cursor
    return agent


def test_checkpoint_carrier_latest_user_is_protected_but_not_refreshed_again():
    """Public shape of the retained child before repeated forced maintenance."""
    history = _synthetic_checkpoint_child_with_historical_pair(pair_in_tail=True)
    history.append({"role": "user", "content": "latest required correction " * 6000})
    agent = NS(session_id=SID)

    note = restore_native_incremental_note(agent, history)

    assert note is not None and note["source_cursor"] == 2
    assert not native_note_refresh_required(agent, history)
    protected = _protected_tail_since_note(history, note)
    assert protected[-1] == history[-1]


def test_authenticated_note_freshness_keeps_older_and_post_user_evidence():
    initial = [
        {"role": "user", "content": "initial objective"},
        {"role": "assistant", "content": "initial completion"},
    ]
    older_large = [
        *initial,
        {"role": "user", "content": "older retained evidence " * 6000},
        {"role": "assistant", "content": "ordinary completion"},
        {"role": "user", "content": "small latest correction"},
    ]
    post_user_tool_output = [
        *initial,
        {"role": "user", "content": "small latest correction"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "large-result", "type": "function",
            "function": {"name": "terminal", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "large-result", "name": "terminal",
         "content": "post-user operational result " * 6000},
    ]
    non_user_large = [
        *initial,
        {"role": "user", "content": "small latest correction"},
        {"role": "assistant", "content": "non-user evidence " * 8000},
    ]

    assert native_note_refresh_required(_agent_with_authenticated_prefix_note(older_large), older_large)
    assert native_note_refresh_required(
        _agent_with_authenticated_prefix_note(post_user_tool_output), post_user_tool_output,
    )
    assert native_note_refresh_required(_agent_with_authenticated_prefix_note(non_user_large), non_user_large)


def test_latest_user_exclusion_uses_direct_and_noncanonical_projection_cursors():
    direct = [
        {"role": "user", "content": "initial objective"},
        {"role": "assistant", "content": "initial completion"},
        {"role": "user", "content": "latest required correction " * 6000},
    ]
    direct_agent = _agent_with_authenticated_prefix_note(direct)
    _source, direct_is_canonical = _native_incremental_operation_messages(direct_agent, direct)
    assert direct_is_canonical
    assert not native_note_refresh_required(direct_agent, direct)

    source = [
        {"role": "user", "content": "initial objective"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "projection-tool", "type": "function",
            "function": {"name": "terminal", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "projection-tool", "name": "terminal",
         "content": "canonical tool result"},
        {"role": "user", "content": "latest required correction " * 6000},
    ]
    replay = deepcopy(source)
    replay[2]["content"] = "replay-only tool result"
    projection_agent = _agent_with_authenticated_prefix_note(source)
    assert bind_native_incremental_replay_projection(
        projection_agent, source_messages=source, replay_messages=replay,
    )
    _source, projection_is_canonical = _native_incremental_operation_messages(projection_agent, replay)
    assert not projection_is_canonical
    assert not native_note_refresh_required(projection_agent, replay)


def test_initial_no_note_still_uses_full_size_accounting():
    messages = [{"role": "user", "content": "initial required input " * 6000}]
    assert native_note_refresh_required(NS(), messages)


def test_projection_binding_keeps_valid_source_for_fresh_reload():
    history = _legacy_poisoned_history()[:-2]
    agent = NS(session_id=SID)
    assert bind_native_incremental_replay_projection(agent, source_messages=history, replay_messages=history)
    assert restore_native_incremental_note(agent, history) is not None


@pytest.mark.parametrize("event_pending", [True, False])
def test_queued_followup_survives_failed_canonical_reload(event_pending):
    import asyncio
    from gateway.run_turn import GatewayTurnMixin

    queued = []
    event = NS(text="retained queued correction")
    pending_store = {"key": event} if event_pending else {}
    pending_event = pending_store.pop("key", None)  # actual destructive dequeue
    adapter = NS(_active_sessions={}, _pending_messages=pending_store,
                 queue_message=lambda *args: queued.append(args))

    def failed_read(*args, **kwargs):
        raise OSError("isolated canonical read failure")

    ctx = NS(source=NS(chat_id="isolated"), session_id=SID, session_key="key",
             run_generation=1, _interrupt_depth=0, history=[], _status_thread_metadata=None,
             agent_holder=[NS(native_incremental_handoff_enabled=True,
                              _session_db=NS(get_messages_as_conversation=failed_read),
                              session_id=SID)], result_holder=[None])
    result = {"interrupted": True, "messages": []}
    returned = asyncio.run(GatewayTurnMixin._run_agent_queued_followup(
        NS(_MAX_INTERRUPT_DEPTH=9), ctx, adapter, event.text, pending_event,
        result, result, None,
    ))
    assert returned is result
    if event_pending:
        assert adapter._pending_messages["key"] is event
        assert not queued
    else:
        assert queued == [("key", event.text)]
