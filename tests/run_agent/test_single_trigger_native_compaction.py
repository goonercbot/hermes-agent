"""Replay-only compatibility coverage for retired split-path checkpoints.

The split preparation and commit runtime path is deliberately gone. These
fixtures model sessions persisted by that historical implementation.
"""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from agent.codex_responses_adapter import _chat_messages_to_responses_input
from agent.native_compaction import (
    NATIVE_CONTINUITY_METADATA_KEY,
    native_continuity_boundary_fence,
    protected_handoff_wire_item,
)


def _legacy_persisted_history():
    """Handcraft one valid v1 persisted carrier and its replay tail."""
    prefix = [
        {"role": "user", "content": "legacy durable prefix"},
        {"role": "assistant", "content": "legacy accepted result"},
    ]
    triggering_user = {"role": "user", "content": "legacy triggering user"}
    boundary_fence = native_continuity_boundary_fence(prefix)
    canonical_handoff = json.dumps(
        {
            "version": 1,
            "boundary_fence": boundary_fence,
            "excerpts": [
                {
                    "index": index,
                    "role": message["role"],
                    "content": message["content"],
                    "row_fence": native_continuity_boundary_fence([message]),
                }
                for index, message in enumerate(prefix, start=1)
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    checkpoint = {
        "type": "compaction",
        "encrypted_content": "legacy-checkpoint",
        NATIVE_CONTINUITY_METADATA_KEY: {
            "version": 1,
            "boundary_fence": boundary_fence,
            "canonical_handoff": canonical_handoff,
            "triggering_user_fence": native_continuity_boundary_fence(
                [triggering_user]
            ),
        },
    }
    return prefix + [
        {
            "role": "assistant",
            "content": "",
            "display_kind": "hidden",
            "codex_reasoning_items": [checkpoint],
        },
        triggering_user,
    ], canonical_handoff


def test_persisted_legacy_checkpoint_replays_checkpoint_handoff_then_tail():
    history, canonical_handoff = _legacy_persisted_history()
    before = deepcopy(history)

    wire = _chat_messages_to_responses_input(
        history,
        current_issuer_kind="openai_codex",
        native_compaction_eligible=True,
    )

    assert wire == [
        {"type": "compaction", "encrypted_content": "legacy-checkpoint"},
        protected_handoff_wire_item(canonical_handoff),
        {"role": "user", "content": "legacy triggering user"},
    ]
    assert history == before


def test_persisted_legacy_checkpoint_fails_closed_when_prefix_is_tampered():
    history, _canonical_handoff = _legacy_persisted_history()
    history[0]["content"] = "tampered legacy prefix"

    with pytest.raises(
        ValueError,
        match="protected native compaction checkpoint failed boundary validation",
    ):
        _chat_messages_to_responses_input(
            history,
            current_issuer_kind="openai_codex",
            native_compaction_eligible=True,
        )
