"""Regression tests for cached gateway transcript selection."""

from agent.native_compaction import (
    NATIVE_COMPACTION_METADATA_KEY,
    native_continuity_boundary_fence,
    validate_persisted_native_compaction_history,
)
from gateway.run import _build_gateway_agent_history, _select_cached_agent_history


def test_gateway_history_preserves_native_checkpoint_and_exact_handoff():
    handoff = "exact gateway handoff\n"
    tail = [
        {"role": "user", "content": "current user", "timestamp": 1_770_000_000.0},
        {"role": "system", "content": "sealed system tail\n"},
    ]
    checkpoint = {
        "type": "compaction",
        "encrypted_content": "opaque checkpoint",
        NATIVE_COMPACTION_METADATA_KEY: {
            "version": 2,
            "identity": "attempt-1",
            "handoff": handoff,
            "tail_count": len(tail),
            "tail_fence": native_continuity_boundary_fence(tail),
        },
    }
    stored = [
        {
            "role": "assistant",
            "content": "",
            "display_kind": "hidden",
            "codex_reasoning_items": [checkpoint],
        },
        {
            "role": "user",
            "content": handoff,
            "timestamp": 1_770_000_000.0,
        },
        *tail,
    ]

    replay, observed = _build_gateway_agent_history(
        stored,
        inject_timestamps=True,
    )

    assert observed is None
    assert replay[0]["codex_reasoning_items"] == [checkpoint]
    assert replay[1]["content"] == handoff
    validate_persisted_native_compaction_history(replay)


def test_select_cached_history_keeps_expected_replay_cleanup():
    """A durable read-only tool tail removed by stock replay cleanup stays removed."""
    live = [
        {"role": "user", "content": "check status", "_db_persisted": True},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": "{}"},
                }
            ],
            "_db_persisted": True,
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "[Command interrupted]",
            "_db_persisted": True,
        },
    ]

    persisted, observed = _build_gateway_agent_history(live)

    assert observed is None
    assert len(persisted) < len(live)
    assert persisted == [{"role": "user", "content": "check status"}]
    assert _select_cached_agent_history(persisted, live) is persisted


def test_select_cached_history_preserves_live_when_a_real_row_is_unpersisted():
    persisted = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    live = [
        {"role": "user", "content": "hello", "_db_persisted": True},
        {"role": "assistant", "content": "hi", "_db_persisted": True},
        {"role": "user", "content": "not written"},
    ]

    selected = _select_cached_agent_history(persisted, live)

    assert selected is not persisted
    assert selected == live


def test_select_cached_history_ignores_unmarked_ephemeral_scaffolding():
    """Internal retry nudges are deliberately unpersisted, not evidence of DB lag."""
    persisted = [{"role": "user", "content": "ship the fix"}]
    live = [
        {"role": "user", "content": "ship the fix", "_db_persisted": True},
        {
            "role": "user",
            "content": "Verify your work before stopping.",
            "_pre_verify_synthetic": True,
        },
    ]

    assert _select_cached_agent_history(persisted, live) is persisted
