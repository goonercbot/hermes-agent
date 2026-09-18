"""Native checkpoint rows must survive SessionDB restore byte-for-byte."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.native_compaction import validate_persisted_native_compaction_history
from agent.native_incremental_handoff import (
    create_native_incremental_note,
    native_incremental_compact_context,
    record_native_incremental_note,
)
from hermes_state import SessionDB


def _response(*items):
    return SimpleNamespace(output=list(items), status="completed", usage=None)


def _produced_checkpoint_rows():
    """Produce current checkpoint metadata, then persist its real SQLite shape."""
    responses = [_response(
        {"type": "compaction", "id": "checkpoint", "encrypted_content": "sealed-checkpoint"},
        {"type": "message", "id": "maintenance", "role": "assistant", "content": [
            {"type": "output_text", "text": "maintenance output \n"},
        ]},
    )]

    def call(_request):
        return responses.pop(0)

    source = [
        {"role": "user", "content": "old instruction " * 400},
        {"role": "assistant", "content": "old work " * 400},
        {"role": "user", "content": "Review the conversation above and update the skill library\n"},
        {"role": "assistant", "content": "[memory] \n", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "terminal", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "name": "terminal", "content": "result"},
    ]
    agent = SimpleNamespace(
        api_mode="codex_responses",
        provider="openai-codex",
        model="gpt-6-astra",
        base_url="https://chatgpt.com/backend-api/codex",
        session_id="native-checkpoint-sessiondb",
        native_incremental_handoff_enabled=True,
        native_incremental_handoff_model="gpt-5.6-luna",
        native_incremental_compact_threshold=1_000,
        compression_enabled=True,
        _codex_reasoning_replay_enabled=True,
        context_compressor=SimpleNamespace(compression_count=0),
        _interruptible_api_call=call,
        _native_compaction_attempt=None,
    )
    note = create_native_incremental_note(
        session_id=agent.session_id,
        source_messages=source,
        objective="preserve checkpoint bytes",
        current_plan="restore from SQLite",
        next_action="validate authenticated tail",
        blockers=[],
    )
    record_native_incremental_note(agent, note, source)
    rows = native_incremental_compact_context(agent, source)
    validate_persisted_native_compaction_history(rows)
    return rows


def _persist(db: SessionDB, session_id: str, rows) -> None:
    db.create_session(session_id, source="cli")
    for row in rows:
        db.append_message(
            session_id, row["role"], row.get("content"), tool_name=row.get("name"),
            tool_calls=row.get("tool_calls"), tool_call_id=row.get("tool_call_id"),
            codex_reasoning_items=row.get("codex_reasoning_items"),
            codex_message_items=row.get("codex_message_items"),
        )


@pytest.mark.parametrize("repair_alternation", [False, True])
def test_current_checkpoint_metadata_roundtrips_sealed_tail_without_cleanup(
    tmp_path, repair_alternation,
):
    """Current producer metadata protects whitespace and cleanup lookalikes.

    The rows include an assistant reasoning/message suffix with a trailing byte,
    followed by ordinary-looking background-review and stale-tool-marker content.
    Those rows are authenticated checkpoint tail, not candidates for load repair.
    """
    rows = _produced_checkpoint_rows()
    expected = deepcopy(rows)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _persist(db, "native-checkpoint-sessiondb", rows)
        restored = db.get_messages_as_conversation(
            "native-checkpoint-sessiondb", repair_alternation=repair_alternation,
        )
    finally:
        db.close()

    validate_persisted_native_compaction_history(restored)
    assert [row["content"] for row in restored] == [row["content"] for row in expected]
    assert restored[3]["content"] == expected[3]["content"]
    assert restored[4]["content"] == expected[4]["content"]


def test_checkpoint_metadata_shape_accepts_legacy_and_current_but_fails_closed():
    """The SessionDB pre-sanitize gate shares strict producer-era metadata rules."""
    from agent.native_compaction import NATIVE_COMPACTION_METADATA_KEY, native_compaction_metadata_shape

    rows = _produced_checkpoint_rows()
    metadata = rows[0]["codex_reasoning_items"][0][NATIVE_COMPACTION_METADATA_KEY]
    current, error = native_compaction_metadata_shape(metadata)
    assert current is not None and error == ""

    legacy = deepcopy(metadata)
    del legacy["maintenance_suffix_count"]
    del legacy["maintenance_suffix_fence"]
    old_shape, error = native_compaction_metadata_shape(legacy)
    assert old_shape is not None and old_shape.maintenance_suffix_count == 0 and error == ""

    invalid = []
    for key in ("maintenance_suffix_count", "maintenance_suffix_fence"):
        partial = deepcopy(metadata)
        del partial[key]
        invalid.append(partial)
    unknown = deepcopy(metadata)
    unknown["unrecognized"] = "no"
    invalid.append(unknown)
    for tail_count in (True, -1):
        malformed = deepcopy(metadata)
        malformed["tail_count"] = tail_count
        invalid.append(malformed)
    malformed = deepcopy(metadata)
    malformed["maintenance_suffix_count"] = metadata["tail_count"] + 1
    invalid.append(malformed)

    for candidate in invalid:
        shape, error = native_compaction_metadata_shape(candidate)
        assert shape is None
        assert error in {"metadata", "maintenance suffix"}


def test_checkpoint_validation_rejects_forged_tail_suffix_and_carrier():
    """Shape recognition never replaces complete carrier and content authentication."""
    from agent.native_compaction import NATIVE_COMPACTION_METADATA_KEY

    rows = _produced_checkpoint_rows()
    forged_tail = deepcopy(rows)
    forged_tail[2]["content"] += "forged"
    with pytest.raises(ValueError, match="tail mismatch"):
        validate_persisted_native_compaction_history(forged_tail)

    forged_suffix = deepcopy(rows)
    forged_suffix[0]["codex_reasoning_items"][0][NATIVE_COMPACTION_METADATA_KEY][
        "maintenance_suffix_fence"
    ] = "forged"
    with pytest.raises(ValueError, match="maintenance suffix"):
        validate_persisted_native_compaction_history(forged_suffix)

    forged_carrier = deepcopy(rows)
    forged_carrier[0]["codex_reasoning_items"][0]["encrypted_content"] = ""
    with pytest.raises(ValueError, match="ciphertext or metadata"):
        validate_persisted_native_compaction_history(forged_carrier)
