"""Regression coverage for native preflight replay-projection rebinding."""
from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from agent.native_compaction import native_continuity_boundary_fence
from agent.native_incremental_handoff import (
    _projection_source_for_messages,
    bind_native_incremental_replay_projection,
    record_native_incremental_note_from_tool_call,
    restore_native_incremental_note,
)


NOTE = {
    "objective": "Verify native continuity",
    "current_plan": "Run isolated tests",
    "next_action": "Report results",
    "blockers": ["No live effects"],
}


def _response(*items, model="gpt-6-astra"):
    return NS(
        output=list(items),
        usage=NS(input_tokens=600, output_tokens=4, total_tokens=604),
        status="completed",
        model=model,
    )


def _message(text):
    return NS(type="message", role="assistant", content=[NS(type="output_text", text=text)])


def _agent_with_authenticated_history(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "compression:\n  enabled: true\n  codex_responses_native: true\n"
        "  native_incremental_handoff: true\n"
        "  native_incremental_model: gpt-5.6-luna\n"
        "  native_incremental_compact_threshold: 1000\n"
        "  context_timeout_seconds: 0\n"
    )
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "native-preflight-projection"
    db.create_session(session_id, "cli", model="gpt-6-astra")
    agent = AIAgent(
        api_key="fixture",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        model="gpt-6-astra",
        provider="openai-codex",
        session_db=db,
        session_id=session_id,
        quiet_mode=True,
        skip_memory=True,
        skip_context_files=True,
        skip_background_review=True,
        enabled_toolsets=[],
    )
    agent._disable_streaming = True
    agent.compression_in_place = False
    agent._compression_feasibility_checked = True
    agent._emit_status = lambda *args, **kwargs: None
    agent._emit_warning = lambda *args, **kwargs: None
    agent.commit_memory_session = lambda *args, **kwargs: None
    agent.context_compressor.threshold_tokens = 100
    agent.context_compressor.should_compress_preflight = lambda _: True
    agent.context_compressor.should_compress = lambda _: False

    db.append_message(session_id, "user", "old objective " * 2000)
    db.append_message(session_id, "assistant", "old evidence " * 2000)
    db.append_message(
        session_id,
        "assistant",
        "",
        tool_calls=[{
            "id": "note-1",
            "type": "function",
            "function": {"name": "continuity_note", "arguments": json.dumps(NOTE)},
        }],
    )
    source = db.get_messages_as_conversation(session_id, repair_alternation=False)
    db.append_message(
        session_id,
        "tool",
        record_native_incremental_note_from_tool_call(agent, NOTE, source),
        tool_call_id="note-1",
        tool_name="continuity_note",
    )
    source = db.get_messages_as_conversation(session_id, repair_alternation=False)
    assert bind_native_incremental_replay_projection(
        agent, source_messages=source, replay_messages=source
    )
    assert restore_native_incremental_note(agent, source) is not None
    return agent, db, source


def _fixture_replies():
    return [
        _response(
            NS(type="compaction", id="fixture-cp", encrypted_content="fixture-checkpoint"),
            NS(type="reasoning", id="fixture-reasoning", encrypted_content="fixture", summary=[]),
            _message("fixture suffix"),
            model="gpt-5.6-luna",
        ),
        _response(_message("FIXTURE_OK")),
    ]


def _run_preflight(agent, source, *, plugin_context, replies):
    calls = []
    agent._interruptible_api_call = lambda request: (
        calls.append(request["model"]) or replies.pop(0)
    )
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]), patch(
        "hermes_cli.lifecycle.invoke_hook",
        return_value=([{"context": plugin_context}] if plugin_context else []),
    ), patch("agent.turn_context._maybe_title_session_at_turn_start", return_value=None):
        result = agent.run_conversation(
            "Controlled fixture follow-up.", conversation_history=source
        )
    return result, calls


@pytest.mark.parametrize("plugin_context", ["PLUGIN_CONTEXT", ""])
def test_native_preflight_rebinds_only_to_persisted_source_and_rejects_tampering(
    tmp_path, monkeypatch, plugin_context
):
    """A real loop/DB replay stays authenticated after the published tail changes."""
    agent, db, source = _agent_with_authenticated_history(tmp_path, monkeypatch)
    try:
        result, calls = _run_preflight(
            agent, source, plugin_context=plugin_context, replies=_fixture_replies()
        )
        assert result["completed"] and result["final_response"] == "FIXTURE_OK"
        assert calls == ["gpt-5.6-luna", "gpt-6-astra"]

        persisted = db.get_messages_as_conversation(
            agent.session_id, repair_alternation=False
        )
        current_user = next(
            row for row in persisted if row.get("content") == "Controlled fixture follow-up."
        )
        if plugin_context:
            assert current_user["api_content"].endswith(plugin_context)
        else:
            assert "api_content" not in current_user

        state = agent._native_incremental_replay_projection
        cursor = len(state["replay"])
        assert state["source_fence"] == native_continuity_boundary_fence(persisted[:-1])
        assert state["replay_fence"] == native_continuity_boundary_fence(
            result["messages"][:cursor]
        )
        authenticated_source, projection_cursor = _projection_source_for_messages(
            agent, result["messages"]
        )
        assert authenticated_source is not None
        assert projection_cursor == cursor

        tampered = deepcopy(result["messages"])
        tampered[0]["codex_reasoning_items"][0]["encrypted_content"] = "tampered"
        assert _projection_source_for_messages(agent, tampered) == (None, None)
    finally:
        agent.close()
        db.close()


def test_native_preflight_failed_sidecar_transaction_does_not_rebind_projection(
    tmp_path, monkeypatch
):
    """The post-publish copy is unavailable when its authorized transaction fails."""
    agent, db, source = _agent_with_authenticated_history(tmp_path, monkeypatch)
    import agent.native_incremental_handoff as incremental

    rebinds = []
    original_bind = incremental.bind_native_incremental_replay_projection

    def watched_bind(*args, **kwargs):
        rebinds.append((len(kwargs["source_messages"]), len(kwargs["replay_messages"])))
        return original_bind(*args, **kwargs)

    def fail_transaction(*args, **kwargs):
        raise RuntimeError("fixture transaction failure")

    monkeypatch.setattr(
        db, "set_in_place_native_compaction_api_content", fail_transaction
    )
    try:
        replies = _fixture_replies()
        agent._interruptible_api_call = lambda request: replies.pop(0)
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]), patch(
            "hermes_cli.lifecycle.invoke_hook", return_value=[{"context": "PLUGIN_CONTEXT"}]
        ), patch("agent.turn_context._maybe_title_session_at_turn_start", return_value=None), patch.object(
            incremental, "bind_native_incremental_replay_projection", watched_bind
        ):
            with pytest.raises(
                ValueError, match="native compaction api_content atomic rebind failed"
            ):
                agent.run_conversation(
                    "Controlled fixture follow-up.", conversation_history=source
                )

        # Only native candidate construction ran; turn_context did not copy the
        # projection from the persisted source after the rejected transaction.
        assert len(rebinds) == 1
    finally:
        agent.close()
        db.close()
