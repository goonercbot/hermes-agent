"""Core tests for the unified native-compaction lifecycle."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.native_compaction import (
    NATIVE_COMPACTION_HANDOFF_MAX_CHARS,
    NATIVE_COMPACTION_METADATA_KEY,
    bind_native_compaction_tail,
    native_compact_context,
    native_compaction_context_management,
    native_continuity_capable,
    resolve_native_compaction_capabilities,
)


def _response(*items, status="completed", error=None):
    return SimpleNamespace(output=list(items), status=status, error=error)


def _message(text):
    return {"type": "message", "content": [{"type": "output_text", "text": text}]}


def _source():
    return [
        {"role": "user", "content": "durable prefix " * 1000},
        {"role": "assistant", "content": "accepted " * 1000},
        {"role": "user", "content": "triggering user"},
    ]


def _agent(responses, *, threshold=204_000, **overrides):
    calls = []
    transport = SimpleNamespace(preflight_kwargs=lambda kwargs, **_kwargs: kwargs)

    def build(messages, tools_for_api=None):
        return {
            "input": deepcopy(messages),
            "tools": [{"name": "must_not_run"}],
            "tool_choice": "required",
            "parallel_tool_calls": True,
        }

    def call(kwargs):
        calls.append(deepcopy(kwargs))
        return responses.pop(0)

    agent = SimpleNamespace(
        api_mode="codex_responses",
        model="gpt-5.6-sol",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        codex_responses_native_compaction=True,
        codex_responses_compact_threshold=1,
        compression_enabled=True,
        _codex_reasoning_replay_enabled=True,
        runtime_capabilities={"native_compaction": True},
        capabilities={"native_compaction": True},
        context_compressor=SimpleNamespace(
            threshold_tokens=threshold,
            compression_count=0,
        ),
        _build_api_kwargs=build,
        _get_transport=lambda: transport,
        _is_copilot_url=lambda: False,
        _is_codex_backend=lambda: True,
        _interruptible_api_call=call,
        _cached_system_prompt="system",
    )
    for key, value in overrides.items():
        setattr(agent, key, value)
    return agent, calls


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
    agent, _calls = _agent([], **{field: value})
    assert not native_continuity_capable(agent, is_codex_backend=True)


def test_ordinary_request_never_inherits_context_management():
    agent, _calls = _agent([])
    agent._native_continuity_pending = object()
    agent._native_continuity_emit_context_management = True
    assert native_compaction_context_management(agent, is_codex_backend=True) is None


def test_native_request_uses_live_threshold_not_legacy_threshold():
    agent, calls = _agent(
        [
            _response(_message("exact handoff")),
            _response({"type": "compaction", "encrypted_content": "checkpoint"}),
        ],
        threshold=204_000,
    )

    native_compact_context(agent, _source(), "system")

    assert calls[1]["context_management"] == [
        {"type": "compaction", "compact_threshold": 204_000}
    ]


def test_handoff_request_precedes_compaction_and_both_forbid_tools():
    handoff = "resume exact task"
    agent, calls = _agent(
        [
            _response(_message(handoff)),
            _response({"type": "compaction", "encrypted_content": "checkpoint"}),
        ]
    )

    compacted = native_compact_context(agent, _source(), "system")

    assert len(calls) == 2
    assert calls[0]["tools"] == []
    assert "tool_choice" not in calls[0]
    assert "context_management" not in calls[0]
    assert calls[1]["tools"] == []
    assert "tool_choice" not in calls[1]
    assert calls[1]["input"][-1] == {"role": "user", "content": handoff}
    assert compacted[1]["content"] == handoff


def test_model_handoff_bytes_are_preserved_exactly():
    handoff = "  exact handoff bytes\n"
    agent, _calls = _agent(
        [
            _response(_message(handoff)),
            _response({"type": "compaction", "encrypted_content": "checkpoint"}),
        ]
    )

    compacted = native_compact_context(agent, _source(), "system")

    assert compacted[1]["content"] == handoff
    assert compacted[0]["codex_reasoning_items"][0][NATIVE_COMPACTION_METADATA_KEY][
        "handoff"
    ] == handoff


def test_over_bound_handoff_stops_before_compaction_request():
    agent, calls = _agent(
        [_response(_message("x" * (NATIVE_COMPACTION_HANDOFF_MAX_CHARS + 1)))]
    )

    with pytest.raises(ValueError, match="bounded size"):
        native_compact_context(agent, _source(), "system")
    assert len(calls) == 1


def test_disabled_native_path_makes_no_provider_call():
    agent, calls = _agent([], codex_responses_native_compaction=False)
    source = _source()

    assert native_compact_context(agent, source, "system") is source
    assert calls == []


def test_checkpoint_orders_carrier_handoff_then_triggering_tail():
    agent, _calls = _agent(
        [
            _response(_message("exact handoff")),
            _response({"type": "compaction", "encrypted_content": "checkpoint"}),
        ]
    )

    compacted = native_compact_context(agent, _source(), "system")
    bind_native_compaction_tail(agent, compacted)

    checkpoint = compacted[0]["codex_reasoning_items"][0]
    assert checkpoint["encrypted_content"] == "checkpoint"
    assert checkpoint[NATIVE_COMPACTION_METADATA_KEY]["tail_count"] == 1
    assert compacted[1]["content"] == "exact handoff"
    assert compacted[2] == {"role": "user", "content": "triggering user"}


def test_later_native_compaction_replaces_without_mutating_retained_checkpoint():
    agent, calls = _agent(
        [
            _response(_message("first handoff")),
            _response({"type": "compaction", "encrypted_content": "checkpoint-one"}),
            _response(_message("second handoff")),
            _response({"type": "compaction", "encrypted_content": "checkpoint-two"}),
        ]
    )
    first = native_compact_context(agent, _source(), "system")
    bind_native_compaction_tail(agent, first)
    retained_before = deepcopy(first[0])
    later_history = first + [
        {"role": "assistant", "content": "later context " * 1000},
        {"role": "user", "content": "second triggering user"},
    ]

    second = native_compact_context(agent, later_history, "system")
    bind_native_compaction_tail(agent, second)

    assert len(calls) == 4
    assert first[0] == retained_before
    assert second[0]["codex_reasoning_items"][0]["encrypted_content"] == "checkpoint-two"
    assert second[1]["content"] == "second handoff"
    assert second[-1] == {"role": "user", "content": "second triggering user"}


@pytest.mark.parametrize("with_text", [False, True])
def test_checkpoint_only_and_checkpoint_plus_text_use_checkpoint_path(with_text):
    output = [{"type": "compaction", "encrypted_content": "checkpoint"}]
    if with_text:
        output.append(_message("ordinary provider text"))
    agent, _calls = _agent([_response(_message("handoff")), _response(*output)])

    compacted = native_compact_context(agent, _source(), "system")

    assert compacted[0]["codex_reasoning_items"][0]["encrypted_content"] == "checkpoint"


def test_no_checkpoint_ordinary_response_releases_original_history():
    source = _source()
    before = deepcopy(source)
    agent, _calls = _agent(
        [_response(_message("handoff")), _response(_message("ordinary response"))]
    )

    assert native_compact_context(agent, source, "system") is source
    assert source == before


@pytest.mark.parametrize(
    "output,match",
    [
        ([{"type": "compaction", "encrypted_content": ""}], "blank"),
        ([{"type": "compaction", "encrypted_content": 7}], "must be text"),
        (
            [
                {"type": "compaction", "encrypted_content": "one"},
                {"type": "compaction", "encrypted_content": "two"},
            ],
            "duplicate",
        ),
    ],
)
def test_malformed_or_duplicate_provider_checkpoint_fails_closed(output, match):
    agent, _calls = _agent([_response(_message("handoff")), _response(*output)])

    with pytest.raises(ValueError, match=match):
        native_compact_context(agent, _source(), "system")
