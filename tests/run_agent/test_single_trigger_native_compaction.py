"""End-to-end tests for deterministic native continuity and replay."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.codex_responses_adapter import _chat_messages_to_responses_input
from agent.native_compaction import (
    NATIVE_CONTINUITY_METADATA_KEY,
    canonical_native_continuity_handoff,
    commit_native_continuity,
    parse_native_continuity_handoff,
    protected_handoff_boundary_fence,
)


def _raw_response(*, checkpoint: str | None, text: str = "continued"):
    output = [
        SimpleNamespace(
            type="reasoning",
            encrypted_content="ordinary-reasoning",
            summary=[],
            status="completed",
        )
    ]
    if checkpoint is not None:
        output.append(
            SimpleNamespace(
                type="compaction",
                encrypted_content=checkpoint,
                status="completed",
            )
        )
    output.append(
        SimpleNamespace(
            type="message",
            role="assistant",
            status="completed",
            content=[SimpleNamespace(type="output_text", text=text)],
        )
    )
    return SimpleNamespace(status="completed", output=output, error=None)


def _real_agent(monkeypatch):
    from run_agent import AIAgent

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = AIAgent(
        api_key="test-key",
        provider="openai-codex",
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
        model="gpt-5.6-sol",
        quiet_mode=True,
        enabled_toolsets=[],
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
    )
    agent.codex_responses_native_compaction = True
    agent.compression_enabled = True
    agent.compression_checkpoint_required = True
    agent._codex_reasoning_replay_enabled = True
    agent.runtime_capabilities = {"native_compaction": True}
    agent.capabilities = {"native_compaction": True}
    agent.context_compressor.threshold_tokens = 1
    agent._disable_streaming = True
    return agent


def test_handoff_is_source_verbatim_and_bound_to_complete_prefix():
    snapshot = [
        {"role": "user", "content": "accepted scope"},
        {"role": "assistant", "content": "verified result"},
    ]
    fence = protected_handoff_boundary_fence(snapshot)
    canonical = canonical_native_continuity_handoff(snapshot, fence)
    parsed = parse_native_continuity_handoff(
        canonical,
        fence,
        expected_snapshot=snapshot,
    )

    assert [row["content"] for row in parsed["excerpts"]] == [
        "accepted scope",
        "verified result",
    ]
    assert all(row["row_fence"] for row in parsed["excerpts"])

    tampered = canonical.replace("accepted scope", "invented scope")
    with pytest.raises(ValueError, match="source mismatch"):
        parse_native_continuity_handoff(tampered, fence, expected_snapshot=snapshot)


def test_handoff_skips_secret_shaped_rows_without_redacting_source():
    snapshot = [
        {"role": "user", "content": "API_KEY=super-secret-value"},
        {"role": "assistant", "content": "safe verified decision"},
    ]
    fence = protected_handoff_boundary_fence(snapshot)
    parsed = parse_native_continuity_handoff(
        canonical_native_continuity_handoff(snapshot, fence),
        fence,
        expected_snapshot=snapshot,
    )

    assert [row["content"] for row in parsed["excerpts"]] == ["safe verified decision"]


def _committed_history(messages=None):
    from agent.native_compaction import (
        capture_native_continuity_seed,
        prepare_native_continuity_request,
    )

    agent = SimpleNamespace(
        api_mode="codex_responses",
        model="gpt-5.6-sol",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        codex_responses_native_compaction=True,
        compression_enabled=True,
        compression_checkpoint_required=True,
        _codex_reasoning_replay_enabled=True,
        runtime_capabilities={"native_compaction": True},
        capabilities={"native_compaction": True},
        context_compressor=SimpleNamespace(threshold_tokens=1),
        _native_continuity_candidate=None,
        _native_continuity_pending=None,
        _native_continuity_emit_context_management=False,
        _native_continuity_defer_user_persistence=False,
    )
    if messages is None:
        messages = [
            {"role": "user", "content": "durable prefix"},
            {"role": "assistant", "content": "accepted"},
            {"role": "user", "content": "triggering user"},
        ]
    assert capture_native_continuity_seed(agent, messages, len(messages) - 1)
    kwargs = {"instructions": "system", "tools": [], "input": deepcopy(messages)}
    assert prepare_native_continuity_request(agent, api_kwargs=kwargs)
    committed = commit_native_continuity(
        agent,
        {
            "role": "assistant",
            "content": "continued",
            "codex_reasoning_items": [
                {"type": "compaction", "encrypted_content": "checkpoint"}
            ],
        },
    )
    assert committed is not None
    return committed


def test_replay_orders_checkpoint_handoff_then_triggering_user():
    history = _committed_history()
    before = deepcopy(history)

    items = _chat_messages_to_responses_input(
        history,
        current_issuer_kind="openai_codex",
        native_compaction_eligible=True,
    )

    assert items[0] == {"type": "compaction", "encrypted_content": "checkpoint"}
    assert items[1]["role"] == "user"
    assert items[1]["content"].startswith("PROTECTED PRE-COMPRESSION HANDOFF")
    assert items[2] == {"role": "user", "content": "triggering user"}
    assert history == before
    assert NATIVE_CONTINUITY_METADATA_KEY in history[-3]["codex_reasoning_items"][0]


def test_replay_uses_durable_source_when_api_copy_is_normalized():
    source = _committed_history()
    api_copy = deepcopy(source)
    api_copy[0]["tool_calls"] = []

    items = _chat_messages_to_responses_input(
        api_copy,
        current_issuer_kind="openai_codex",
        native_compaction_eligible=True,
        native_continuity_source_messages=source,
    )

    assert items[0] == {"type": "compaction", "encrypted_content": "checkpoint"}
    assert items[1]["content"].startswith("PROTECTED PRE-COMPRESSION HANDOFF")
    assert items[2] == {"role": "user", "content": "triggering user"}

    tampered_source = deepcopy(source)
    tampered_source[0]["content"] = "tampered durable prefix"
    with pytest.raises(
        ValueError,
        match="protected native compaction checkpoint failed boundary validation",
    ):
        _chat_messages_to_responses_input(
            api_copy,
            current_issuer_kind="openai_codex",
            native_compaction_eligible=True,
            native_continuity_source_messages=tampered_source,
        )


def test_real_replay_validates_untouched_durable_source_before_tool_argument_repair(
    monkeypatch,
):
    truncated_arguments = '{"path": "/tmp/foo'
    history = _committed_history([
        {"role": "user", "content": "durable prefix"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": truncated_arguments,
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "tool result"},
        {"role": "user", "content": "triggering user"},
    ])
    agent = _real_agent(monkeypatch)
    getattr(agent, "context_compressor").threshold_tokens = 1_000_000
    repairs = []
    calls = []
    real_sanitize = agent._sanitize_tool_call_arguments

    def observed_sanitize(candidate_messages, **kwargs):
        repaired = real_sanitize(candidate_messages, **kwargs)
        repairs.append(repaired)
        return repaired

    def fake_call(api_kwargs):
        calls.append(deepcopy(api_kwargs))
        return _raw_response(checkpoint=None)

    setattr(agent, "_sanitize_tool_call_arguments", observed_sanitize)
    agent._interruptible_api_call = fake_call
    result = agent.run_conversation("follow-up user", conversation_history=history)

    assert result["completed"] is True
    assert repairs == [1]
    assert len(calls) == 1
    assert calls[0].get("context_management") is None
    assert calls[0]["input"][0] == {
        "type": "compaction",
        "encrypted_content": "checkpoint",
    }
    assert calls[0]["input"][1]["content"].startswith(
        "PROTECTED PRE-COMPRESSION HANDOFF"
    )
    assert calls[0]["input"][2] == {"role": "user", "content": "triggering user"}
    durable_tool_call = history[1]["tool_calls"][0]
    assert durable_tool_call["function"]["arguments"] == truncated_arguments


@pytest.mark.parametrize(
    "malformation",
    ["non-dict-metadata", "missing-ciphertext", "wrong-item-type"],
)
def test_real_replay_rejects_every_malformed_protected_checkpoint_before_call(
    monkeypatch,
    malformation,
):
    history = _committed_history()
    checkpoint = history[-3]["codex_reasoning_items"][0]
    if malformation == "non-dict-metadata":
        checkpoint[NATIVE_CONTINUITY_METADATA_KEY] = "malformed"
    elif malformation == "missing-ciphertext":
        checkpoint.pop("encrypted_content")
    else:
        checkpoint["type"] = "reasoning"

    agent = _real_agent(monkeypatch)
    getattr(agent, "context_compressor").threshold_tokens = 1_000_000
    calls = []
    compressions = []
    agent._interruptible_api_call = lambda api_kwargs: calls.append(api_kwargs)
    monkeypatch.setattr(
        getattr(agent, "context_compressor"),
        "compress",
        lambda *args, **kwargs: compressions.append((args, kwargs)),
    )

    result = agent.run_conversation("follow-up user", conversation_history=history)

    assert result["completed"] is False
    assert result["error"] == "native_compaction_checkpoint_missing"
    assert result["messages"] == history
    assert calls == []
    assert compressions == []


def test_invalid_persisted_boundary_fails_closed():
    history = _committed_history()
    history[0]["content"] = "mutated durable prefix"

    with pytest.raises(
        ValueError,
        match="protected native compaction checkpoint failed boundary validation",
    ):
        _chat_messages_to_responses_input(
            history,
            current_issuer_kind="openai_codex",
            native_compaction_eligible=True,
        )


def test_real_run_conversation_sends_one_exact_boundary_and_commits_checkpoint(
    monkeypatch,
):
    agent = _real_agent(monkeypatch)
    calls = []

    def fake_call(api_kwargs):
        calls.append(deepcopy(api_kwargs))
        return _raw_response(checkpoint="checkpoint-1")

    agent._interruptible_api_call = fake_call
    history = [{"role": "user", "content": "durable prefix"}]
    result = agent.run_conversation(
        "triggering user",
        conversation_history=history,
    )

    assert result["completed"] is True
    assert len(calls) == 1
    assert calls[0]["context_management"] == [
        {"type": "compaction", "compact_threshold": 1}
    ]
    wire = calls[0]["input"]
    assert wire[-3] == {"role": "user", "content": "durable prefix"}
    assert wire[-2]["content"].startswith("PROTECTED PRE-COMPRESSION HANDOFF")
    assert wire[-1] == {"role": "user", "content": "triggering user"}

    messages = result["messages"]
    assert messages[-3]["display_kind"] == "hidden"
    assert messages[-2]["content"] == "triggering user"
    assert messages[-1]["content"] == "continued"
    assert [item["type"] for item in messages[-1].get("codex_reasoning_items", [])] == [
        "reasoning"
    ]


def test_real_run_missing_checkpoint_is_one_attempt_and_preserves_prefix(monkeypatch):
    agent = _real_agent(monkeypatch)
    calls = []

    def fake_call(kwargs):
        calls.append(deepcopy(kwargs))
        return _raw_response(checkpoint=None)

    agent._interruptible_api_call = fake_call
    history = [{"role": "user", "content": "durable prefix"}]
    result = agent.run_conversation("triggering user", conversation_history=history)

    assert len(calls) == 1
    assert result["completed"] is False
    assert result["error"] == "native_compaction_checkpoint_missing"
    assert result["messages"] == history


def test_real_execution_middleware_cannot_mutate_protected_request(monkeypatch):
    agent = _real_agent(monkeypatch)
    calls = []

    def fake_call(api_kwargs):
        calls.append(deepcopy(api_kwargs))
        return _raw_response(checkpoint="checkpoint-1")

    def mutate(request, executor, **_kwargs):
        changed = deepcopy(request)
        changed["input"][-1]["content"] = "mutated after preflight"
        return executor(changed)

    agent._interruptible_api_call = fake_call
    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_execution_middleware",
        mutate,
    )
    history = [{"role": "user", "content": "durable prefix"}]
    result = agent.run_conversation("triggering user", conversation_history=history)

    assert calls == []
    assert result["completed"] is False
    assert result["messages"] == history


def test_real_execution_middleware_cannot_attempt_protected_request_twice(monkeypatch):
    agent = _real_agent(monkeypatch)
    calls = []

    def fake_call(api_kwargs):
        calls.append(deepcopy(api_kwargs))
        return _raw_response(checkpoint="checkpoint-1")

    def duplicate(request, executor, **_kwargs):
        executor(request)
        return executor(request)

    agent._interruptible_api_call = fake_call
    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_execution_middleware",
        duplicate,
    )
    history = [{"role": "user", "content": "durable prefix"}]
    result = agent.run_conversation("triggering user", conversation_history=history)

    assert len(calls) == 1
    assert result["completed"] is False
    assert result["messages"] == history


@pytest.mark.parametrize(
    "failure_mode", ["crossed-response", "interrupted-error", "provider-error"]
)
def test_real_redirect_crossing_protected_response_fails_without_retry(
    monkeypatch, failure_mode
):
    agent = _real_agent(monkeypatch)
    calls = []

    def fake_call(api_kwargs):
        calls.append(deepcopy(api_kwargs))
        assert agent.redirect("late correction") is True
        if failure_mode == "interrupted-error":
            raise InterruptedError
        if failure_mode == "provider-error":
            raise RuntimeError("provider failed after redirect")
        return _raw_response(checkpoint="checkpoint-1")

    agent._interruptible_api_call = fake_call
    history = [{"role": "user", "content": "durable prefix"}]
    result = agent.run_conversation("triggering user", conversation_history=history)

    assert len(calls) == 1
    assert result["completed"] is False
    assert result["error"] == "native_compaction_checkpoint_missing"
    assert result["messages"] == history
    assert getattr(agent, "_native_continuity_pending") is None
    assert getattr(agent, "_native_continuity_candidate") is None
    assert agent._has_pending_redirect() is False
    assert agent._interrupt_requested is False
    assert all(
        message.get("display_kind") != "hidden" for message in result["messages"]
    )
    assert all(
        message.get("content") != "triggering user" for message in result["messages"]
    )

    getattr(agent, "context_compressor").threshold_tokens = 1_000_000

    def ordinary_call(api_kwargs):
        calls.append(deepcopy(api_kwargs))
        return _raw_response(checkpoint=None, text="fresh response")

    agent._interruptible_api_call = ordinary_call
    fresh = agent.run_conversation("fresh user", conversation_history=history)

    assert len(calls) == 2
    assert fresh["completed"] is True
    assert fresh["final_response"] == "fresh response"


def test_tampered_triggering_user_fails_closed():
    history = _committed_history()
    history[-2]["content"] = "tampered triggering user"
    history[-2]["api_content"] = "tampered triggering user"

    with pytest.raises(
        ValueError,
        match="protected native compaction checkpoint failed boundary validation",
    ):
        _chat_messages_to_responses_input(
            history,
            current_issuer_kind="openai_codex",
            native_compaction_eligible=True,
        )


def test_persist_session_skips_unresolved_native_candidate():
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    setattr(agent, "_native_continuity_candidate", object())
    setattr(agent, "_native_continuity_pending", None)
    setattr(agent, "_native_continuity_defer_user_persistence", True)

    assert agent._persist_session([{"role": "user", "content": "trigger"}], []) is None


def test_real_invalid_replay_fails_without_local_compression_or_provider_call(
    monkeypatch,
):
    agent = _real_agent(monkeypatch)
    history = _committed_history()
    history[-2]["content"] = "tampered triggering user"
    history[-2]["api_content"] = "tampered triggering user"
    calls = []
    compressions = []

    agent._interruptible_api_call = lambda api_kwargs: calls.append(api_kwargs)
    monkeypatch.setattr(
        getattr(agent, "context_compressor"),
        "compress",
        lambda *args, **kwargs: compressions.append((args, kwargs)),
    )
    result = agent.run_conversation("follow-up user", conversation_history=history)

    assert result["completed"] is False
    assert result["error"] == "native_compaction_checkpoint_missing"
    assert result["messages"] == history
    assert calls == []
    assert compressions == []


def test_real_second_request_replays_without_context_management(monkeypatch):
    agent = _real_agent(monkeypatch)
    calls = []
    responses = [
        _raw_response(checkpoint="checkpoint-1", text="first continuation"),
        _raw_response(checkpoint=None, text="second continuation"),
    ]

    def fake_call(kwargs):
        calls.append(deepcopy(kwargs))
        return responses.pop(0)

    agent._interruptible_api_call = fake_call
    first = agent.run_conversation(
        "triggering user",
        conversation_history=[{"role": "user", "content": "durable prefix"}],
    )
    assert first["completed"] is True

    getattr(agent, "context_compressor").threshold_tokens = 1_000_000
    second = agent.run_conversation(
        "follow-up user",
        conversation_history=first["messages"],
    )

    assert second["completed"] is True
    assert len(calls) == 2
    assert "context_management" not in calls[1]
    assert calls[1]["input"][0] == {
        "type": "compaction",
        "encrypted_content": "checkpoint-1",
    }
    assert calls[1]["input"][1]["content"].startswith(
        "PROTECTED PRE-COMPRESSION HANDOFF"
    )
    assert any(
        item.get("role") == "user" and item.get("content") == "triggering user"
        for item in calls[1]["input"]
    )
    assert calls[1]["input"][-1] == {"role": "user", "content": "follow-up user"}
