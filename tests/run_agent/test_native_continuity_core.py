"""Core tests for the single-trigger native continuity contract."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.native_compaction import (
    NATIVE_CONTINUITY_METADATA_KEY,
    canonical_native_continuity_handoff,
    capture_native_continuity_seed,
    commit_native_continuity,
    fail_native_continuity,
    native_compaction_context_management,
    native_continuity_capable,
    parse_native_continuity_handoff,
    prepare_native_continuity_request,
    protected_handoff_wire_item,
    release_native_continuity_without_checkpoint,
    resolve_native_compaction_capabilities,
    response_has_valid_native_checkpoint,
)


def _agent(*, threshold: int = 100, **overrides):
    agent = SimpleNamespace(
        api_mode="codex_responses",
        model="gpt-5.6-sol",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        codex_responses_native_compaction=True,
        codex_responses_compact_threshold=1,
        compression_enabled=True,
        compression_checkpoint_required=True,
        _codex_reasoning_replay_enabled=True,
        runtime_capabilities={"native_compaction": True},
        capabilities={"native_compaction": True},
        context_compressor=SimpleNamespace(threshold_tokens=threshold),
        _native_continuity_candidate=None,
        _native_continuity_pending=None,
        _native_continuity_emit_context_management=False,
        _native_continuity_defer_user_persistence=False,
    )
    for key, value in overrides.items():
        setattr(agent, key, value)
    return agent


def _prepared(agent, *, prefix="durable prefix", user="triggering user"):
    messages = [
        {"role": "user", "content": prefix},
        {"role": "assistant", "content": "accepted"},
        {"role": "user", "content": user},
    ]
    assert capture_native_continuity_seed(agent, messages, 2)
    kwargs = {
        "instructions": "stable and ephemeral system instructions",
        "tools": [{"type": "function", "name": "large_tool", "description": "x" * 100}],
        "input": deepcopy(messages),
    }
    assert prepare_native_continuity_request(agent, api_kwargs=kwargs)
    return messages, kwargs


def test_direct_openai_capability_is_explicit():
    assert resolve_native_compaction_capabilities(
        model="gpt-5.6-sol",
        base_url="https://chatgpt.com/backend-api/codex",
        provider="openai-codex",
        is_codex_backend=True,
    ) == {"native_compaction": True}
    assert resolve_native_compaction_capabilities(
        model="gpt-5.6-sol",
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
    ) == {"native_compaction": False}


@pytest.mark.parametrize(
    "field,value",
    [
        ("api_mode", "chat_completions"),
        ("codex_responses_native_compaction", False),
        ("compression_enabled", False),
        ("_codex_reasoning_replay_enabled", False),
    ],
)
def test_capability_fails_closed(field, value):
    assert not native_continuity_capable(
        _agent(**{field: value}), is_codex_backend=True
    )


def test_context_management_exists_only_for_prepared_boundary_and_uses_live_threshold():
    agent = _agent(threshold=204_000)
    assert native_compaction_context_management(agent, is_codex_backend=True) is None
    agent._native_continuity_pending = object()
    agent._native_continuity_emit_context_management = True
    assert native_compaction_context_management(agent, is_codex_backend=True) == [
        {"type": "compaction", "compact_threshold": 204_000}
    ]
    agent._native_continuity_emit_context_management = False
    assert native_compaction_context_management(agent, is_codex_backend=True) is None


def test_legacy_threshold_never_owns_trigger():
    agent = _agent(threshold=204_000, codex_responses_compact_threshold=120_000)
    agent._native_continuity_pending = object()
    agent._native_continuity_emit_context_management = True
    context_management = native_compaction_context_management(
        agent, is_codex_backend=True
    )
    assert context_management is not None
    assert context_management[0]["compact_threshold"] == 204_000


def test_prepare_measures_and_sends_same_handoff_bearing_request():
    agent = _agent(threshold=1)
    original, kwargs = _prepared(agent)
    boundary = agent._native_continuity_pending

    assert original == [
        {"role": "user", "content": "durable prefix"},
        {"role": "assistant", "content": "accepted"},
        {"role": "user", "content": "triggering user"},
    ]
    assert kwargs["context_management"] == [
        {"type": "compaction", "compact_threshold": 1}
    ]
    assert kwargs["input"][-2]["role"] == "user"
    assert kwargs["input"][-2] == protected_handoff_wire_item(
        boundary.canonical_handoff
    )
    assert kwargs["input"][-1]["content"] == "triggering user"
    parsed = parse_native_continuity_handoff(
        boundary.canonical_handoff,
        boundary.boundary_fence,
        expected_snapshot=list(boundary.snapshot),
    )
    assert len(parsed["excerpts"]) == 2


def test_handoff_fills_unused_capacity_after_lane_reservations():
    snapshot = [
        {"role": "user", "content": f"decision {index} " + "x" * 1_000}
        for index in range(20)
    ]

    canonical = canonical_native_continuity_handoff(snapshot, "v2:test")
    parsed = parse_native_continuity_handoff(
        canonical,
        "v2:test",
        expected_snapshot=snapshot,
    )

    assert len(parsed["excerpts"]) > 10


def test_below_threshold_does_not_prepare_or_emit():
    agent = _agent(threshold=1_000_000)
    messages = [{"role": "user", "content": "short"}]
    assert capture_native_continuity_seed(agent, messages, 0)
    kwargs = {"instructions": "system", "tools": [], "input": deepcopy(messages)}
    assert not prepare_native_continuity_request(agent, api_kwargs=kwargs)
    assert agent._native_continuity_pending is None
    assert "context_management" not in kwargs


def test_checkpoint_commit_restores_immutable_boundary_and_orders_carrier_before_user():
    agent = _agent(threshold=1)
    _messages, _kwargs = _prepared(agent)
    assistant = {
        "role": "assistant",
        "content": "continued",
        "codex_reasoning_items": [
            {"type": "reasoning", "encrypted_content": "ordinary", "summary": []},
            {"type": "compaction", "encrypted_content": "checkpoint"},
        ],
    }

    committed = commit_native_continuity(agent, assistant)

    assert committed is not None
    assert committed[-2]["content"] == "triggering user"
    carrier = committed[-3]
    assert carrier["display_kind"] == "hidden"
    checkpoint = carrier["codex_reasoning_items"][0]
    assert checkpoint["encrypted_content"] == "checkpoint"
    assert NATIVE_CONTINUITY_METADATA_KEY in checkpoint
    assert committed[-1]["codex_reasoning_items"] == [
        {"type": "reasoning", "encrypted_content": "ordinary", "summary": []}
    ]
    assert agent._native_continuity_pending is None


def test_missing_checkpoint_releases_the_ordinary_user_turn():
    agent = _agent(threshold=1)
    _messages, _kwargs = _prepared(agent)
    original_snapshot = deepcopy(list(agent._native_continuity_pending.snapshot))

    released = release_native_continuity_without_checkpoint(agent)

    assert released == original_snapshot + [
        {"role": "user", "content": "triggering user"}
    ]
    assert agent._native_continuity_pending is None
    assert agent._native_continuity_defer_user_persistence is False
    assert agent._persist_user_message_idx == len(original_snapshot)


def test_native_checkpoint_response_validation_distinguishes_absence_from_malformed():
    absent = SimpleNamespace(output=[SimpleNamespace(type="message")])
    valid = SimpleNamespace(
        output=[SimpleNamespace(type="compaction", encrypted_content="checkpoint")]
    )
    malformed = SimpleNamespace(
        output=[SimpleNamespace(type="compaction", encrypted_content="")]
    )

    assert response_has_valid_native_checkpoint(absent) is False
    assert response_has_valid_native_checkpoint(valid) is True
    with pytest.raises(ValueError, match="checkpoint is invalid"):
        response_has_valid_native_checkpoint(malformed)
