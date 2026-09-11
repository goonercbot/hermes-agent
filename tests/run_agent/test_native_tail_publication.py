"""Current-attempt tail binding and late sidecar persistence regressions."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from agent.native_compaction import (
    NATIVE_COMPACTION_HANDOFF_ROLE,
    NATIVE_COMPACTION_METADATA_KEY,
    NATIVE_COMPACTION_VERSION,
    NativeCompactionAttempt,
    bind_native_compaction_tail,
    finalize_native_compaction_turn,
    native_continuity_boundary_fence,
    validate_persisted_native_compaction_history,
)
from agent.turn_context import _stamp_api_content_sidecar
from hermes_state import SessionDB


def _current_attempt(*, allow_turn_rebind=True):
    identity = "current-test-attempt"
    metadata = {
        "version": NATIVE_COMPACTION_VERSION,
        "identity": identity,
        "handoff": "retained handoff",
        "tail_count": 0,
        "tail_fence": "",
    }
    checkpoint = {
        "type": "compaction",
        "encrypted_content": "test-ciphertext",
        NATIVE_COMPACTION_METADATA_KEY: metadata,
    }
    carrier = {"role": "assistant", "content": "", "codex_reasoning_items": [checkpoint]}
    handoff = {
        "role": NATIVE_COMPACTION_HANDOFF_ROLE,
        "content": metadata["handoff"],
        "_native_compaction_handoff": identity,
    }
    messages = [carrier, handoff, {"role": "user", "content": "current question"}]
    attempt = NativeCompactionAttempt(identity, carrier, handoff, metadata, allow_turn_rebind)
    return NS(_native_compaction_attempt=attempt), messages, attempt


def test_current_tail_binding_includes_late_context_and_consumes_capability():
    agent, messages, attempt = _current_attempt()
    bind_native_compaction_tail(agent, messages)
    old_fence = attempt.metadata["tail_fence"]
    messages[-1]["api_content"] = "current question with late context"
    finalize_native_compaction_turn(agent, messages)
    assert attempt.metadata["tail_fence"] == native_continuity_boundary_fence(messages[2:])
    assert attempt.metadata["tail_fence"] != old_fence
    assert attempt.metadata["tail_count"] == 1
    assert agent._native_compaction_attempt is None
    sealed = deepcopy(attempt.metadata)
    messages[-1]["api_content"] = "later unauthorised replacement"
    bind_native_compaction_tail(agent, messages)
    assert attempt.metadata == sealed


def test_non_rebindable_attempt_is_consumed_on_initial_binding():
    agent, messages, attempt = _current_attempt(allow_turn_rebind=False)
    bind_native_compaction_tail(agent, messages)
    assert agent._native_compaction_attempt is None
    assert attempt.metadata["tail_fence"] == native_continuity_boundary_fence(messages[2:])


@pytest.mark.parametrize("mutation", ["carrier_copy", "handoff_copy", "identity", "sidecar"])
def test_binding_rejects_lost_current_producer_authority(mutation):
    agent, messages, attempt = _current_attempt()
    if mutation == "carrier_copy":
        messages[0] = deepcopy(messages[0])
    elif mutation == "handoff_copy":
        messages[1] = deepcopy(messages[1])
    elif mutation == "identity":
        attempt.metadata["identity"] = "another-attempt"
    else:
        messages[1]["_native_compaction_handoff"] = "another-attempt"
    before = deepcopy(attempt.metadata)
    with pytest.raises(ValueError, match="capability was lost|sidecar mismatch"):
        bind_native_compaction_tail(agent, messages)
    assert attempt.metadata == before


@pytest.mark.parametrize("in_place", [False, True])
def test_sidecar_stamp_rebinds_persisted_tail_before_fresh_readback(tmp_path, in_place):
    agent, messages, attempt = _current_attempt()
    bind_native_compaction_tail(agent, messages)
    old_fence = attempt.metadata["tail_fence"]
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    try:
        db.create_session("tail-publication", "cli", model="gpt-5.6-luna")
        for row in messages:
            db.append_message(
                "tail-publication", row["role"], row["content"],
                codex_reasoning_items=row.get("codex_reasoning_items"),
            )
        agent._session_db = db
        agent.session_id = "tail-publication"
        agent._last_compaction_in_place = in_place
        _stamp_api_content_sidecar(
            agent, messages, 2, "", "late plugin context", preflight_compressed=True,
        )
        assert attempt.metadata["tail_fence"] != old_fence
        assert attempt.metadata["tail_fence"] == native_continuity_boundary_fence(messages[2:])
    finally:
        db.close()
    fresh = SessionDB(db_path=path)
    try:
        restored = fresh.get_messages_as_conversation("tail-publication", repair_alternation=False)
        assert restored[-1]["content"] == "current question"
        assert restored[-1]["api_content"] == "current question\n\nlate plugin context"
        assert validate_persisted_native_compaction_history(restored)
        assert restored[0]["codex_reasoning_items"][0][NATIVE_COMPACTION_METADATA_KEY] == attempt.metadata
    finally:
        fresh.close()


def test_failed_sidecar_transaction_restores_in_memory_fence():
    agent, messages, attempt = _current_attempt()
    bind_native_compaction_tail(agent, messages)
    old_metadata = deepcopy(attempt.metadata)

    def reject_transaction(**kwargs):
        raise RuntimeError("transaction refused")

    agent.session_id = "tail-publication"
    agent._session_db = NS(
        set_in_place_native_compaction_api_content=lambda *args, **kwargs: reject_transaction(**kwargs)
    )
    with pytest.raises(ValueError, match="atomic rebind failed"):
        _stamp_api_content_sidecar(
            agent, messages, 2, "", "late plugin context", preflight_compressed=True,
        )
    assert attempt.metadata == old_metadata
    assert "api_content" not in messages[-1]
