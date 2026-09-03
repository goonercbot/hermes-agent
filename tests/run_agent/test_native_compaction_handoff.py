"""Backward-compatibility tests for protected checkpoint replay."""

from __future__ import annotations

import json

import pytest

from agent.codex_responses_adapter import (
    _chat_messages_to_responses_input,
    _preflight_codex_input_items,
)
from agent.native_compaction import (
    PROTECTED_HANDOFF_METADATA_KEY,
    attach_protected_handoff,
    parse_protected_handoff,
    protected_handoff_boundary_fence,
)

_LIST_FIELDS = (
    "decision_rationale",
    "agreed_sequence",
    "mechanism_limits",
    "requested_accounting",
    "protected_live_state",
    "verified_completed",
    "started_unverified",
    "verification_required",
    "blockers",
    "approvals",
    "prohibitions",
    "state_distinctions",
    "authoritative_corrections",
    "unrelated_work_to_ignore",
)


def _handoff(fence: str) -> str:
    value: dict[str, object] = {
        "boundary_fence": fence,
        "latest_user_instruction": "Complete the current task.",
        "immediate_resume_cursor": "Run focused verification.",
        "current_task": "Native compaction integration",
        "why_urgent": "The session is over the compression threshold.",
        "next_action": "Verify the committed checkpoint.",
    }
    value.update({field: [f"{field} evidence"] for field in _LIST_FIELDS})
    return json.dumps(value)


def _attached_checkpoint(fence: str, blob: str = "checkpoint"):
    return attach_protected_handoff(
        {"type": "compaction", "encrypted_content": blob, "_issuer_kind": "other"},
        canonical_handoff=_handoff(fence),
        boundary_fence=fence,
    )


def test_handoff_parser_rejects_surrounding_prose_and_duplicate_keys():
    fence = protected_handoff_boundary_fence([{"role": "user", "content": "scope"}])
    clean = _handoff(fence)

    with pytest.raises(ValueError, match="exactly one JSON object"):
        parse_protected_handoff(f"preface\n{clean}", fence)
    duplicate = clean[:-1] + f', "boundary_fence": {json.dumps(fence)}}}'
    with pytest.raises(ValueError, match="duplicate key"):
        parse_protected_handoff(duplicate, fence)


def test_wire_orders_checkpoint_handoff_then_true_tail_without_stale_retention():
    pre_user = {"role": "user", "content": "STALE CLAIM: activation already completed"}
    fence = protected_handoff_boundary_fence([pre_user])
    checkpoint = _attached_checkpoint(fence)
    history = [
        pre_user,
        {"role": "assistant", "content": "", "codex_reasoning_items": [checkpoint]},
        {"role": "user", "content": "tail user"},
    ]
    persisted_before = json.loads(json.dumps(history))

    items = _chat_messages_to_responses_input(
        history,
        current_issuer_kind="other",
        native_compaction_eligible=True,
    )

    assert items[0] == {"type": "compaction", "encrypted_content": "checkpoint"}
    assert items[1]["content"].startswith("PROTECTED PRE-COMPRESSION HANDOFF")
    assert items[2] == {"role": "user", "content": "tail user"}
    assert not any("STALE CLAIM" in str(item.get("content", "")) for item in items)
    assert all(PROTECTED_HANDOFF_METADATA_KEY not in item for item in items)
    assert _preflight_codex_input_items(items)[0] == {
        "type": "compaction",
        "encrypted_content": "checkpoint",
    }
    assert history == persisted_before


def test_legacy_replay_drops_checkpoint_when_durable_prefix_changed():
    pre_user = {"role": "user", "content": "original scope"}
    fence = protected_handoff_boundary_fence([pre_user])
    history = [
        pre_user,
        {
            "role": "assistant",
            "content": "",
            "codex_reasoning_items": [_attached_checkpoint(fence)],
        },
        {"role": "user", "content": "tail user"},
    ]
    pre_user["content"] = "mutated scope"

    with pytest.raises(
        ValueError,
        match="protected native compaction checkpoint failed boundary validation",
    ):
        _chat_messages_to_responses_input(
            history,
            current_issuer_kind="other",
            native_compaction_eligible=True,
        )


def test_newest_checkpoint_owns_the_only_replayed_handoff():
    old_user = {"role": "user", "content": "old"}
    old_fence = protected_handoff_boundary_fence([old_user])
    old_carrier = {
        "role": "assistant",
        "content": "old cp",
        "codex_reasoning_items": [_attached_checkpoint(old_fence, "old")],
    }
    mid_user = {"role": "user", "content": "mid"}
    new_fence = protected_handoff_boundary_fence([old_user, old_carrier, mid_user])
    history = [
        old_user,
        old_carrier,
        mid_user,
        {
            "role": "assistant",
            "content": "new cp",
            "codex_reasoning_items": [_attached_checkpoint(new_fence, "new")],
        },
        {"role": "user", "content": "tail"},
    ]

    items = _chat_messages_to_responses_input(history, native_compaction_eligible=True)

    assert [
        item.get("encrypted_content")
        for item in items
        if item.get("type") == "compaction"
    ] == ["new"]
    handoffs = [
        item["content"]
        for item in items
        if item.get("role") == "user"
        and item["content"].startswith("PROTECTED")
    ]
    assert len(handoffs) == 1


def test_ineligible_wire_keeps_full_history_and_never_injects_handoff():
    pre_user = {"role": "user", "content": "retained user"}
    fence = protected_handoff_boundary_fence([pre_user])
    history = [
        pre_user,
        {
            "role": "assistant",
            "content": "checkpoint carrier",
            "codex_reasoning_items": [_attached_checkpoint(fence)],
        },
        {"role": "user", "content": "tail user"},
    ]

    items = _chat_messages_to_responses_input(
        history, native_compaction_eligible=False
    )

    assert all(item.get("type") != "compaction" for item in items)
    assert all(
        not str(item.get("content", "")).startswith("PROTECTED") for item in items
    )
    assert {"role": "assistant", "content": "checkpoint carrier"} in items
