"""Native-continuity lifecycle regressions without private production history."""
from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from agent.native_incremental_handoff import (
    bind_native_incremental_replay_projection,
    create_native_incremental_note,
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
            "note": carried_note,
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


def test_unbound_terminal_maintenance_pair_still_fails_closed():
    history = _synthetic_checkpoint_child_with_historical_pair(pair_in_tail=False)

    assert validate_persisted_native_compaction_history(history)
    assert restore_native_incremental_note(NS(session_id=SID), history) is None


def test_divergent_canonical_source_rolls_back_before_pair_flush():
    messages = [{"role": "user", "content": "durable queued input"}]
    replay = deepcopy(messages)
    source = deepcopy(messages)
    source[0]["content"] = "stale cleaned input"
    flushed = []

    class DB:
        def get_messages_as_conversation(self, *_args, **_kwargs):
            return deepcopy(messages)

    agent = NS(
        session_id=SID,
        _current_api_request_id="turn:api:1",
        _interrupt_requested=False,
        _session_db=DB(),
        _persist_disabled=False,
        _native_incremental_replay_projection={
            "source": source,
            "source_fence": __import__("agent.native_incremental_handoff", fromlist=["_note_fence"])._note_fence(source),
            "replay": replay,
            "replay_fence": __import__("agent.native_incremental_handoff", fromlist=["_note_fence"])._note_fence(replay),
        },
        _flush_messages_to_session_db=lambda rows: flushed.append(deepcopy(rows)) or True,
    )
    capability = issue_native_note_refresh(agent, messages)
    response = {
        "status": "completed",
        "output": [{"type": "function_call", "name": "continuity_note", "call_id": "new-note", "arguments": json.dumps(ARGS)}],
    }

    with pytest.raises(NativeNoteRefreshFailure, match="canonical maintenance source diverged"):
        execute_native_note_refresh(agent, capability, response, messages, SID)

    assert messages == replay
    assert flushed == []


def test_persistence_disabled_fork_is_never_maintenance_eligible():
    fork = NS(_persist_disabled=True, _session_db=None)
    assert not native_note_refresh_required(fork, [{"role": "user", "content": "x" * 200000}])


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
