"""Host seams for opt-in native incremental continuity."""
from types import SimpleNamespace as NS

import pytest

from agent.agent_init import _parse_compression_config
from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
from hermes_state import SessionDB


def test_native_incremental_settings_are_opt_in_and_default_safe():
    disabled = _parse_compression_config(NS(api_mode="chat_completions"), {"compression": {}})
    assert disabled.native_incremental_handoff_enabled is False
    assert disabled.native_incremental_handoff_model == "gpt-5.6-luna"
    assert disabled.native_incremental_compact_threshold == 128000

    enabled = _parse_compression_config(
        NS(api_mode="codex_responses"),
        {"compression": {"native_incremental_handoff": True, "native_incremental_compact_threshold": "123"}},
    )
    assert enabled.native_incremental_handoff_enabled is True
    assert enabled.native_incremental_compact_threshold == 123


def test_native_steer_is_a_durable_append_and_generic_steer_is_unchanged():
    generic_history = [{"role": "tool", "content": "result", "tool_call_id": "call"}]
    generic = NS(native_incremental_handoff_enabled=False, _drain_pending_steer=lambda: "ordinary")
    apply_pending_steer_to_tool_results(generic, generic_history, 1)
    assert len(generic_history) == 1
    assert "ordinary" in generic_history[0]["content"]

    persisted = []
    native_history = [{"role": "tool", "content": "result", "tool_call_id": "call"}]
    native = NS(
        native_incremental_handoff_enabled=True,
        _drain_pending_steer=lambda: "keep the durable tail",
        _flush_messages_to_session_db=lambda rows: persisted.append(list(rows)) or True,
    )
    apply_pending_steer_to_tool_results(native, native_history, 1)
    assert native_history[0]["content"] == "result"
    assert native_history[-1]["role"] == "user"
    assert "keep the durable tail" in native_history[-1]["content"]
    assert persisted and persisted[-1] == native_history


def test_native_api_sidecar_update_is_atomic(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("native-host", "cli", model="gpt-5.6-luna")
        db.append_message("native-host", "user", "user content")
        old_metadata = {"version": 2, "identity": "id", "handoff": "handoff", "tail_count": 0, "tail_fence": ""}
        db.append_message(
            "native-host", "assistant", "",
            codex_reasoning_items=[{"type": "compaction", "encrypted_content": "cipher", "_hermes_native_compaction": old_metadata}],
        )
        new_metadata = dict(old_metadata, tail_fence="updated")
        db.set_in_place_native_compaction_api_content(
            "native-host", user_content="user content", api_content="wire user content",
            identity="id", encrypted_content="cipher", old_metadata=old_metadata, new_metadata=new_metadata,
        )
        rows = db.get_messages_as_conversation("native-host", repair_alternation=False)
        assert rows[0]["api_content"] == "wire user content"
        assert rows[1]["codex_reasoning_items"][0]["_hermes_native_compaction"] == new_metadata
        with pytest.raises(ValueError, match="metadata mismatch"):
            db.set_in_place_native_compaction_api_content(
                "native-host", user_content="user content", api_content="bad",
                identity="id", encrypted_content="cipher", old_metadata=old_metadata, new_metadata=old_metadata,
            )
        assert db.get_messages_as_conversation("native-host", repair_alternation=False)[0]["api_content"] == "wire user content"
    finally:
        db.close()
