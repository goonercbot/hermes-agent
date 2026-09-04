"""Focused contract tests for the one native-compaction lifecycle."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.codex_responses_adapter import _chat_messages_to_responses_input
from agent.native_compaction import (
    NATIVE_COMPACTION_METADATA_KEY,
    bind_native_compaction_tail,
    native_compact_context,
    validate_native_compaction_checkpoint,
)


def _response(*, handoff: str = "", checkpoint: object = None, text: str = ""):
    output = []
    if checkpoint is not None:
        output.append(SimpleNamespace(type="compaction", encrypted_content=checkpoint))
    if handoff or text:
        output.append(SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", text=handoff or text)],
        ))
    return SimpleNamespace(output=output)


def _agent(responses):
    calls = []
    def call(request):
        calls.append(deepcopy(request))
        return responses.pop(0)
    agent = SimpleNamespace(
        api_mode="codex_responses",
        model="gpt-5.6-test",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        codex_responses_native_compaction=True,
        compression_enabled=True,
        _codex_reasoning_replay_enabled=True,
        runtime_capabilities={"native_compaction": True},
        capabilities={},
        context_compressor=SimpleNamespace(threshold_tokens=123, compress=lambda m, **_k: m),
        _interruptible_api_call=call,
    )
    return agent, calls


def _history():
    return [
        {"role": "user", "content": "old instruction " * 1000},
        {"role": "assistant", "content": "old answer " * 1000},
        {"role": "user", "content": "current ask", "api_content": "wire ask"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "completed result"},
        {"role": "user", "content": "TODO: verify", "_todo_snapshot_synthetic": True},
    ]


def test_handoff_then_one_context_management_request_and_exact_persistence():
    agent, calls = _agent([_response(handoff="exact handoff\n"), _response(checkpoint="  opaque bytes  ")])
    history = _history()
    compacted = native_compact_context(agent, history, "system")
    bind_native_compaction_tail(agent, compacted)

    assert len(calls) == 2
    assert calls[0]["tools"] == []
    assert "context_management" not in calls[0]
    assert calls[1]["tools"] == []
    assert calls[1]["context_management"] == [{"type": "compaction", "compact_threshold": 123}]
    assert compacted[0]["codex_reasoning_items"][0]["encrypted_content"] == "  opaque bytes  "
    assert compacted[1]["content"] == "exact handoff\n"
    assert compacted[2:] == history[2:]
    meta = compacted[0]["codex_reasoning_items"][0][NATIVE_COMPACTION_METADATA_KEY]
    assert meta["tail_count"] == len(history[2:])

    wire = _chat_messages_to_responses_input(compacted, native_compaction_eligible=True)
    assert wire[0] == {"type": "compaction", "encrypted_content": "  opaque bytes  "}
    assert wire[1] == {"role": "user", "content": "exact handoff\n"}
    assert wire[2:] == _chat_messages_to_responses_input(history[2:])


@pytest.mark.parametrize("checkpoint", ["", " \t", ["bad"]])
def test_malformed_raw_checkpoint_leaves_history_untouched(checkpoint):
    agent, _ = _agent([_response(handoff="handoff"), _response(checkpoint=checkpoint)])
    history = _history()
    before = deepcopy(history)
    with pytest.raises(ValueError, match="checkpoint"):
        native_compact_context(agent, history, "system")
    assert history == before


def test_no_checkpoint_text_declines_without_local_fallback():
    agent, calls = _agent([_response(handoff="handoff"), _response(text="ordinary response")])
    history = _history()
    assert native_compact_context(agent, history, "system") is history
    assert len(calls) == 2


def test_replay_rejects_mutation_before_provider_and_old_checkpoint_never_rebinds():
    agent, _ = _agent([_response(handoff="handoff"), _response(checkpoint="cipher")])
    compacted = native_compact_context(agent, _history(), "system")
    bind_native_compaction_tail(agent, compacted)
    checkpoint = compacted[0]["codex_reasoning_items"][0]
    metadata_before = deepcopy(checkpoint[NATIVE_COMPACTION_METADATA_KEY])

    # A later local/manual compression has no active identity and cannot bind it.
    bind_native_compaction_tail(SimpleNamespace(_native_compaction_attempt=None), compacted)
    assert checkpoint[NATIVE_COMPACTION_METADATA_KEY] == metadata_before
    validate_native_compaction_checkpoint(checkpoint, compacted[1], compacted[2:])

    compacted[2]["content"] = "mutated current ask"
    compacted[2]["api_content"] = "mutated current ask"
    with pytest.raises(ValueError, match="boundary validation"):
        validate_native_compaction_checkpoint(checkpoint, compacted[1], compacted[2:])


def test_duplicate_or_mixed_carriers_fail_closed():
    agent, _ = _agent([_response(handoff="handoff"), _response(checkpoint="cipher")])
    compacted = native_compact_context(agent, _history(), "system")
    bind_native_compaction_tail(agent, compacted)
    checkpoint = compacted[0]["codex_reasoning_items"][0]
    compacted[0]["codex_reasoning_items"].append(deepcopy(checkpoint))
    with pytest.raises(ValueError, match="boundary validation"):
        _chat_messages_to_responses_input(compacted, native_compaction_eligible=True)
