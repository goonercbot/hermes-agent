"""Focused regression coverage for the common native compaction engine."""

from __future__ import annotations

import ast
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.codex_responses_adapter import _chat_messages_to_responses_input
from agent.native_compaction import (
    NATIVE_COMPACTION_HANDOFF_MAX_CHARS,
    NATIVE_COMPACTION_METADATA_KEY,
    bind_native_compaction_tail,
    native_compact_context,
    native_continuity_boundary_fence,
    validate_persisted_native_compaction_history,
)


def test_runtime_has_one_native_dispatch_and_no_split_path_preparer():
    root = Path(__file__).resolve().parents[2]
    native_dispatch_calls = []
    split_preparer_calls = []
    for path in [root / "run_agent.py", *(root / "agent").glob("*.py")]:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else None
            )
            if name == "native_compact_context":
                native_dispatch_calls.append(path.relative_to(root).as_posix())
            if name in {
                "capture_native_continuity_seed",
                "prepare_native_continuity_request",
            }:
                split_preparer_calls.append((path.relative_to(root).as_posix(), name))

    assert native_dispatch_calls == ["agent/conversation_compression.py"]
    assert split_preparer_calls == []


def _response(*items, status="completed", error=None):
    return SimpleNamespace(output=list(items), status=status, error=error)


def _message(text):
    return {"type": "message", "content": [{"type": "output_text", "text": text}]}


def _agent(responses):
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
        _base_url_hostname="chatgpt.com",
        _base_url_lower="https://chatgpt.com/backend-api/codex",
        codex_responses_native_compaction=True,
        compression_enabled=True,
        _codex_reasoning_replay_enabled=True,
        runtime_capabilities={"native_compaction": True},
        capabilities={"native_compaction": True},
        context_compressor=SimpleNamespace(
            threshold_tokens=123,
            compression_count=2,
            _last_summary_dropped_count=99,
            _last_summary_fallback_used=True,
            _last_feasibility_skip=True,
            _last_summary_error="stale",
            _last_aux_model_failure_error="stale",
            _last_aux_model_failure_model="stale-model",
            _last_compress_aborted=True,
            _last_compress_refused_would_grow=True,
            _last_compression_made_progress=False,
            _last_compression_savings_pct=0.0,
        ),
        _build_api_kwargs=build,
        _get_transport=lambda: transport,
        _is_copilot_url=lambda: False,
        _is_codex_backend=lambda: True,
        _interruptible_api_call=call,
        _cached_system_prompt="system",
    )
    return agent, calls


@pytest.mark.parametrize("checkpoint", ["ciphertext", "  ciphertext\n"])
def test_common_engine_handoff_then_checkpoint_preserves_ciphertext(checkpoint):
    agent, calls = _agent([
        _response(_message("resume at focused test")),
        _response({"type": "compaction", "encrypted_content": checkpoint}),
    ])
    source = [
        {"role": "user", "content": "old instruction " * 1000},
        {"role": "assistant", "content": "old result " * 1000},
        {"role": "user", "content": "current instruction"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "exact result"},
    ]

    compacted = native_compact_context(agent, source, "system")

    assert len(calls) == 2
    assert "context_management" not in calls[0]
    assert calls[0].get("tools") == []
    assert calls[0].get("tool_choice") is None
    assert calls[0].get("parallel_tool_calls") is None
    assert calls[1]["context_management"] == [{"type": "compaction", "compact_threshold": 123}]
    assert calls[1].get("tools") == []
    assert calls[1].get("tool_choice") is None
    assert calls[1].get("parallel_tool_calls") is None
    assert compacted[2:] == source[2:]
    checkpoint_row = compacted[0]["codex_reasoning_items"][0]
    assert checkpoint_row["encrypted_content"] == checkpoint
    assert checkpoint_row[NATIVE_COMPACTION_METADATA_KEY]["handoff"] == "resume at focused test"
    assert agent.context_compressor.compression_count == 3
    assert agent.context_compressor._last_summary_dropped_count == 2
    assert agent.context_compressor._last_summary_error is None
    assert agent.context_compressor._last_summary_fallback_used is False
    assert agent.context_compressor._last_compress_aborted is False
    assert agent.context_compressor._last_compression_made_progress is True

    wire = _chat_messages_to_responses_input(compacted, native_compaction_eligible=True)
    assert wire[0] == {"type": "compaction", "encrypted_content": checkpoint}
    assert wire[1]["content"].endswith("resume at focused test")
    assert wire[2]["content"] == "current instruction"
    assert wire[-1]["output"] == "exact result"


def test_checkpoint_only_response_commits_without_generic_text():
    agent, _calls = _agent([
        _response(_message("handoff")),
        _response({"type": "compaction", "encrypted_content": "cp"}),
    ])
    source = [
        {"role": "assistant", "content": "old context " * 1000},
        {"role": "user", "content": "current"},
    ]

    compacted = native_compact_context(agent, source, "system")

    assert compacted[0]["codex_reasoning_items"][0]["encrypted_content"] == "cp"
    assert compacted[2:] == source[1:]


def test_no_compressible_prefix_skips_both_provider_requests():
    agent, calls = _agent([])
    source = [{"role": "user", "content": "current"}]

    assert native_compact_context(agent, source, "system") is source
    assert calls == []
    assert agent.context_compressor._last_compression_made_progress is False
    assert agent.context_compressor._last_compression_savings_pct == 0.0


def test_checkpoint_that_would_grow_transcript_is_discarded():
    agent, calls = _agent([
        _response(_message("handoff")),
        _response({"type": "compaction", "encrypted_content": "cp"}),
    ])
    source = [
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "current"},
    ]

    assert native_compact_context(agent, source, "system") is source
    assert len(calls) == 2
    assert agent.context_compressor.compression_count == 2
    assert agent.context_compressor._last_compress_refused_would_grow is True
    assert agent.context_compressor._last_compression_made_progress is False


def test_ordinary_no_checkpoint_discards_handoff_and_keeps_history():
    agent, calls = _agent([
        _response(_message("handoff")),
        _response(_message("ordinary answer")),
    ])
    source = [
        {"role": "assistant", "content": "old context " * 1000},
        {"role": "user", "content": "current"},
    ]

    assert native_compact_context(agent, source, "system") is source
    assert len(calls) == 2
    assert agent.context_compressor.compression_count == 2
    assert agent.context_compressor._last_summary_error is None
    assert agent.context_compressor._last_compress_aborted is False
    assert agent.context_compressor._last_compression_made_progress is False


@pytest.mark.parametrize(
    "response,match",
    [
        (_response({"type": "compaction", "encrypted_content": ""}), "blank"),
        (_response({"type": "compaction", "encrypted_content": 7}), "must be text"),
        (_response({"type": "compaction", "encrypted_content": "a"}, {"type": "compaction", "encrypted_content": "b"}), "duplicate"),
        (_response(status="incomplete"), "failed"),
    ],
)
def test_malformed_or_incomplete_native_result_fails_before_commit(response, match):
    agent, _calls = _agent([_response(_message("handoff")), response])
    source = [
        {"role": "assistant", "content": "old context " * 1000},
        {"role": "user", "content": "current"},
    ]

    with pytest.raises(ValueError, match=match):
        native_compact_context(agent, source, "system")
    assert source[-1] == {"role": "user", "content": "current"}


def test_blank_handoff_never_sends_compaction_request():
    agent, calls = _agent([_response(_message(" "))])

    with pytest.raises(ValueError, match="handoff"):
        native_compact_context(
            agent,
            [
                {"role": "assistant", "content": "old context " * 1000},
                {"role": "user", "content": "current"},
            ],
            "system",
        )
    assert len(calls) == 1


def test_oversized_handoff_never_sends_compaction_request():
    agent, calls = _agent([
        _response(_message("h" * (NATIVE_COMPACTION_HANDOFF_MAX_CHARS + 1)))
    ])
    source = [
        {"role": "assistant", "content": "old context " * 1000},
        {"role": "user", "content": "current"},
    ]

    with pytest.raises(ValueError, match="bounded size"):
        native_compact_context(agent, source, "system")
    assert len(calls) == 1


def test_provider_exception_leaves_source_unchanged():
    agent, calls = _agent([])
    source = [
        {"role": "assistant", "content": "old context " * 1000},
        {"role": "user", "content": "current"},
    ]
    before = deepcopy(source)

    def fail(_kwargs):
        calls.append("attempted")
        raise RuntimeError("provider unavailable")

    agent._interruptible_api_call = fail
    with pytest.raises(RuntimeError, match="provider unavailable"):
        native_compact_context(agent, source, "system")
    assert source == before
    assert calls == ["attempted"]


def test_disabled_native_engine_makes_no_provider_call():
    agent, calls = _agent([])
    agent.codex_responses_native_compaction = False
    source = [
        {"role": "assistant", "content": "old context " * 1000},
        {"role": "user", "content": "current"},
    ]

    assert native_compact_context(agent, source, "system") is source
    assert calls == []


def test_new_checkpoint_replay_rejects_tampered_tail():
    agent, _calls = _agent([
        _response(_message("handoff")),
        _response({"type": "compaction", "encrypted_content": "cp"}),
    ])
    source = [
        {"role": "assistant", "content": "old context " * 1000},
        {"role": "user", "content": "current"},
    ]
    compacted = native_compact_context(agent, source, "system")
    compacted[1]["content"] = "tampered"

    with pytest.raises(
        ValueError,
        match="protected native compaction checkpoint failed boundary validation",
    ):
        _chat_messages_to_responses_input(
            compacted,
            current_issuer_kind="openai_codex",
            native_compaction_eligible=True,
        )


def test_native_request_copy_accepts_canonicalized_tool_arguments_from_valid_source():
    """Request repair may canonicalize JSON without mutating protected history."""
    agent, _calls = _agent([
        _response(_message("handoff")),
        _response({"type": "compaction", "encrypted_content": "cp"}),
    ])
    source = [
        {"role": "assistant", "content": "old context " * 1000},
        {"role": "user", "content": "current"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "tool_call",
                        "arguments": '{"name":"fleet_route_task","arguments":{"work_shape":"direct","consequence":"routine"}}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "fleet_route_task",
            "content": '{"ok":true}',
        },
    ]
    protected = native_compact_context(agent, source, "system")
    bind_native_compaction_tail(agent, protected)
    durable_before = deepcopy(protected)

    request_copy = deepcopy(protected)
    request_copy[3]["tool_calls"][0]["function"]["arguments"] = (
        '{"arguments":{"consequence":"routine","work_shape":"direct"},'
        '"name":"fleet_route_task"}'
    )
    wire = _chat_messages_to_responses_input(
        request_copy,
        current_issuer_kind="openai_codex",
        native_compaction_eligible=True,
        native_continuity_source_messages=protected,
    )

    assert protected == durable_before
    assert wire[0] == {"type": "compaction", "encrypted_content": "cp"}
    assert wire[1] == {"role": "user", "content": "handoff"}
    function_call = next(item for item in wire if item.get("type") == "function_call")
    assert function_call["arguments"] == request_copy[3]["tool_calls"][0]["function"]["arguments"]


def test_native_request_copy_rejects_semantic_tail_mutation():
    agent, _calls = _agent([
        _response(_message("handoff")),
        _response({"type": "compaction", "encrypted_content": "cp"}),
    ])
    source = [
        {"role": "assistant", "content": "old context " * 1000},
        {"role": "user", "content": "current"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path":"/tmp/exact"}',
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "read_file",
            "content": "exact result",
        },
    ]
    protected = native_compact_context(agent, source, "system")
    bind_native_compaction_tail(agent, protected)
    request_copy = deepcopy(protected)
    request_copy[-1]["content"] = "changed result"

    with pytest.raises(ValueError, match="tail mismatch"):
        _chat_messages_to_responses_input(
            request_copy,
            current_issuer_kind="openai_codex",
            native_compaction_eligible=True,
            native_continuity_source_messages=protected,
        )


def test_retained_nv1_checkpoint_without_request_repair_still_replays():
    from agent.native_compaction import _legacy_native_continuity_boundary_fence

    agent, _calls = _agent([
        _response(_message("handoff")),
        _response({"type": "compaction", "encrypted_content": "cp"}),
    ])
    protected = native_compact_context(
        agent,
        [
            {"role": "assistant", "content": "old context " * 1000},
            {"role": "user", "content": "current"},
        ],
        "system",
    )
    bind_native_compaction_tail(agent, protected)
    metadata = protected[0]["codex_reasoning_items"][0][
        NATIVE_COMPACTION_METADATA_KEY
    ]
    metadata["tail_fence"] = _legacy_native_continuity_boundary_fence(
        protected[2:]
    )

    validate_persisted_native_compaction_history(protected)
    wire = _chat_messages_to_responses_input(
        protected,
        current_issuer_kind="openai_codex",
        native_compaction_eligible=True,
    )
    assert wire[:2] == [
        {"type": "compaction", "encrypted_content": "cp"},
        {"role": "user", "content": "handoff"},
    ]


def test_new_checkpoint_replay_rejects_tampered_handoff():
    agent, _calls = _agent([
        _response(_message("handoff")),
        _response({"type": "compaction", "encrypted_content": "cp"}),
    ])
    source = [
        {"role": "assistant", "content": "old context " * 1000},
        {"role": "user", "content": "current"},
    ]
    compacted = native_compact_context(agent, source, "system")
    checkpoint = compacted[0]["codex_reasoning_items"][0]
    checkpoint[NATIVE_COMPACTION_METADATA_KEY]["handoff"] = "tampered"

    with pytest.raises(
        ValueError,
        match="protected native compaction checkpoint failed boundary validation",
    ):
        _chat_messages_to_responses_input(
            compacted,
            current_issuer_kind="openai_codex",
            native_compaction_eligible=True,
        )


@pytest.mark.parametrize(
    "reasoning_items",
    [
        [{"type": "compaction", "encrypted_content": "", NATIVE_COMPACTION_METADATA_KEY: {}}],
        [{"type": "compaction", "encrypted_content": 7, NATIVE_COMPACTION_METADATA_KEY: {}}],
        [
            {
                "type": "compaction",
                "encrypted_content": "ciphertext",
                NATIVE_COMPACTION_METADATA_KEY: "scalar",
            }
        ],
        [
            {
                "type": "compaction",
                "encrypted_content": "ciphertext",
                NATIVE_COMPACTION_METADATA_KEY: [],
            }
        ],
        [
            {
                "type": "compaction",
                "encrypted_content": "ciphertext",
                NATIVE_COMPACTION_METADATA_KEY: None,
            }
        ],
        [
            {"type": "compaction", "encrypted_content": "one", NATIVE_COMPACTION_METADATA_KEY: {}},
            {"type": "compaction", "encrypted_content": "two", NATIVE_COMPACTION_METADATA_KEY: {}},
        ],
        [
            {"type": "compaction", "encrypted_content": "protected", NATIVE_COMPACTION_METADATA_KEY: {}},
            {"type": "compaction", "encrypted_content": "extra"},
        ],
    ],
)
def test_persisted_new_checkpoint_malformed_or_duplicate_fails_before_replay(
    reasoning_items,
):
    history = [
        {
            "role": "assistant",
            "content": "",
            "display_kind": "hidden",
            "codex_reasoning_items": reasoning_items,
        },
        {"role": "user", "content": "current"},
    ]

    with pytest.raises(
        ValueError,
        match="protected native compaction checkpoint failed boundary validation",
    ):
        _chat_messages_to_responses_input(
            history,
            current_issuer_kind="openai_codex",
            native_compaction_eligible=True,
        )


def test_persisted_protected_and_plain_checkpoint_carriers_fail_as_duplicate():
    agent, _calls = _agent(
        [
            _response(_message("exact handoff")),
            _response({"type": "compaction", "encrypted_content": "protected-v2"}),
        ]
    )
    protected = native_compact_context(
        agent,
        [
            {"role": "user", "content": "old " * 1000},
            {"role": "assistant", "content": "answer " * 1000},
            {"role": "user", "content": "current"},
        ],
        "system",
    )
    bind_native_compaction_tail(agent, protected)
    mixed = [
        {
            "role": "assistant",
            "content": "",
            "codex_reasoning_items": [
                {"type": "compaction", "encrypted_content": "retained-plain-v1"}
            ],
        },
        *deepcopy(protected),
    ]

    with pytest.raises(ValueError, match="duplicate carrier"):
        validate_persisted_native_compaction_history(mixed)
    with pytest.raises(ValueError, match="duplicate carrier"):
        _chat_messages_to_responses_input(
            mixed,
            current_issuer_kind="openai_codex",
            native_compaction_eligible=True,
        )


@pytest.mark.parametrize(
    "checkpoint",
    [
        {
            "type": "compaction",
            "encrypted_content": "",
            NATIVE_COMPACTION_METADATA_KEY: {},
        },
        {
            "type": "compaction",
            "encrypted_content": "ciphertext",
            NATIVE_COMPACTION_METADATA_KEY: {},
        },
        {
            "type": "compaction",
            "encrypted_content": "ciphertext",
            NATIVE_COMPACTION_METADATA_KEY: "scalar",
        },
        {
            "type": "compaction",
            "encrypted_content": "ciphertext",
            NATIVE_COMPACTION_METADATA_KEY: [],
        },
        {
            "type": "compaction",
            "encrypted_content": "ciphertext",
            NATIVE_COMPACTION_METADATA_KEY: None,
        },
    ],
)
def test_run_conversation_rejects_invalid_persisted_new_checkpoint_before_provider(
    monkeypatch,
    checkpoint,
):
    """A malformed persisted checkpoint is terminal, not retry/fallback input."""
    from run_agent import AIAgent

    history = [
        {
            "role": "assistant",
            "content": "completed tool call",
            "tool_calls": [
                {
                    "id": "call_before_checkpoint",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        # The normal sanitizer would canonicalize this nested
                        # dict before request construction. The protected
                        # terminal path must restore the exact durable value.
                        "arguments": {"command": "pwd"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_before_checkpoint",
            "content": "completed result",
        },
        {
            "role": "assistant",
            "content": "",
            "display_kind": "hidden",
            "codex_reasoning_items": [deepcopy(checkpoint)],
        }
    ]
    before = deepcopy(history)
    provider_calls = []
    persisted = []

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://chatgpt.com/backend-api/codex",
            api_mode="codex_responses",
            model="gpt-5.6-sol",
            provider="openai-codex",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=[],
        )
    agent.codex_responses_native_compaction = True
    agent.runtime_capabilities = {"native_compaction": True}
    agent._persist_session = lambda messages, history=None: persisted.append(deepcopy(messages))
    monkeypatch.setattr(
        agent,
        "_interruptible_api_call",
        lambda request: provider_calls.append(deepcopy(request)),
    )
    monkeypatch.setattr(
        agent,
        "_try_activate_fallback",
        lambda *args, **kwargs: pytest.fail("protected replay must not activate fallback"),
    )

    result = agent.run_conversation("current", conversation_history=history)

    metadata = checkpoint.get(NATIVE_COMPACTION_METADATA_KEY)
    encrypted = checkpoint.get("encrypted_content")
    if not isinstance(encrypted, str) or not encrypted.strip() or not isinstance(metadata, dict):
        error = "protected native compaction checkpoint failed boundary validation: ciphertext or metadata"
    else:
        error = "protected native compaction checkpoint failed boundary validation: metadata"
    assert provider_calls == []
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["error"] == error
    assert result["final_response"] == error
    assert history == before
    assert result["messages"][:-1] == before
    # The turn prologue durably stages the new user row. The terminal path must
    # not add a retry/fallback persistence write or alter the protected prefix.
    assert len(persisted) == 1
    assert persisted[0][:-1] == before


def test_common_lifecycle_persists_todo_tail_and_replays_after_restart():
    """The real commit path refreshes checkpoint fences after TODO injection."""
    from agent.conversation_compression import compress_context
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools.todo_tool import TodoStore

    sid = "native_todo_restart"
    source = [
        {"role": "user", "content": "old instruction " * 2_000},
        {"role": "assistant", "content": "old result " * 2_000},
        {"role": "user", "content": "exact final user"},
        {
            "role": "assistant",
            "content": "exact tool request",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command":"pwd"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "exact tool result",
        },
    ]
    todo_store = TodoStore()
    todo_store.write([
        {"id": "resume", "content": "resume exact task", "status": "pending"}
    ])
    expected_todo = todo_store.format_for_injection()
    assert expected_todo is not None

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "state.db"
        db = SessionDB(db_path=db_path)
        db.create_session(sid, "cli", model="gpt-5.6-sol")
        for message in source:
            db.append_message(sid, **message)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            agent = AIAgent(
                api_key="test-key",
                base_url="https://chatgpt.com/backend-api/codex",
                api_mode="codex_responses",
                model="gpt-5.6-sol",
                provider="openai-codex",
                quiet_mode=True,
                session_db=db,
                session_id=sid,
                skip_context_files=True,
                skip_memory=True,
                enabled_toolsets=[],
            )
        agent.codex_responses_native_compaction = True
        agent.runtime_capabilities = {"native_compaction": True}
        agent._todo_store = todo_store
        agent._compression_feasibility_checked = True
        agent._emit_status = lambda *_args, **_kwargs: None
        agent._emit_warning = lambda *_args, **_kwargs: None
        agent.commit_memory_session = lambda *_args, **_kwargs: None
        agent.context_compressor.threshold_tokens = 100

        responses = [
            _response(_message("exact model handoff")),
            _response({"type": "compaction", "encrypted_content": "checkpoint"}),
        ]
        requests = []

        def _provider_call(request):
            requests.append(deepcopy(request))
            return responses.pop(0)

        agent._interruptible_api_call = _provider_call
        live_history = db.get_messages_as_conversation(sid)
        compacted, _ = compress_context(
            agent,
            live_history,
            approx_tokens=100_000,
            system_message="system",
            force=True,
        )

        assert len(requests) == 2
        assert "context_management" not in requests[0]
        assert requests[0].get("tools") == []
        assert requests[1]["context_management"] == [
            {"type": "compaction", "compact_threshold": 100}
        ]
        assert requests[1].get("tools") == []
        assert compacted[-1]["content"] == expected_todo
        checkpoint = compacted[0]["codex_reasoning_items"][0]
        # The engine fences three source-tail rows. The commit site must refresh
        # this to include the lifecycle-appended TODO row before persistence.
        assert checkpoint[NATIVE_COMPACTION_METADATA_KEY]["tail_count"] == 4

        # A later forced local/manual compression may retain the protected
        # checkpoint, but without the current native capability it must never
        # rewrite that checkpoint's original boundary.
        checkpoint_before_local = deepcopy(checkpoint)
        agent.codex_responses_native_compaction = False
        agent.runtime_capabilities = {}
        agent.context_compressor.compress = (
            lambda candidate, **_kwargs: list(candidate)
        )
        locally_compacted, _ = compress_context(
            agent,
            compacted,
            approx_tokens=100_000,
            system_message="system",
            force=True,
        )
        assert locally_compacted[0]["codex_reasoning_items"][0] == checkpoint_before_local
        local_wire = _chat_messages_to_responses_input(
            locally_compacted,
            current_issuer_kind="openai_codex",
            native_compaction_eligible=True,
        )
        assert local_wire[0] == {
            "type": "compaction",
            "encrypted_content": "checkpoint",
        }

        db.close()
        restarted_db = SessionDB(db_path=db_path)
        restarted, _display = restarted_db.get_resume_conversations(sid)
        restarted_checkpoint = restarted[0]["codex_reasoning_items"][0]
        assert restarted_checkpoint[NATIVE_COMPACTION_METADATA_KEY]["tail_count"] == 4
        assert restarted_checkpoint[NATIVE_COMPACTION_METADATA_KEY]["tail_fence"] == (
            native_continuity_boundary_fence(restarted[2:])
        )

        wire = _chat_messages_to_responses_input(
            restarted,
            current_issuer_kind="openai_codex",
            native_compaction_eligible=True,
        )
        assert wire == [
            {"type": "compaction", "encrypted_content": "checkpoint"},
            {"role": "user", "content": "exact model handoff"},
            {"role": "user", "content": "exact final user"},
            {"role": "assistant", "content": "exact tool request"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "terminal",
                "arguments": '{"command":"pwd"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "exact tool result",
            },
            {"role": "user", "content": expected_todo},
        ]
        restarted_db.close()


@pytest.mark.parametrize("in_place", [True, False])
def test_turn_prologue_native_checkpoint_refreshes_api_tail_after_restart(in_place):
    """In-place and rotation persist the same final native wire tail."""
    from agent.turn_context import build_turn_context
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools.todo_tool import TodoStore

    sid = f"native_turn_prologue_restart_{in_place}"
    todo_store = TodoStore()
    todo_store.write([
        {"id": "resume", "content": "resume exact task", "status": "pending"}
    ])
    expected_todo = todo_store.format_for_injection()
    assert expected_todo is not None

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "state.db"
        db = SessionDB(db_path=db_path)
        db.create_session(sid, "cli", model="gpt-5.6-sol")
        db.append_message(sid, "user", "old instruction " * 2_000)
        db.append_message(sid, "assistant", "old result " * 2_000)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            agent = AIAgent(
                api_key="test-key",
                base_url="https://chatgpt.com/backend-api/codex",
                api_mode="codex_responses",
                model="gpt-5.6-sol",
                provider="openai-codex",
                quiet_mode=True,
                session_db=db,
                session_id=sid,
                skip_context_files=True,
                skip_memory=True,
                enabled_toolsets=[],
            )
        agent.codex_responses_native_compaction = True
        setattr(agent, "compression_in_place", in_place)
        agent.runtime_capabilities = {"native_compaction": True}
        agent._todo_store = todo_store
        agent._compression_feasibility_checked = True
        agent._emit_status = lambda *_args, **_kwargs: None
        agent._emit_warning = lambda *_args, **_kwargs: None
        agent.commit_memory_session = lambda *_args, **_kwargs: None
        agent.context_compressor.threshold_tokens = 100
        agent.context_compressor.should_compress_preflight = lambda _messages: True

        responses = [
            _response(_message("exact model handoff")),
            _response({"type": "compaction", "encrypted_content": "checkpoint"}),
        ]
        provider_requests = []
        agent._interruptible_api_call = lambda request: (
            provider_requests.append(deepcopy(request)) or responses.pop(0)
        )

        with patch(
            "hermes_cli.plugins.invoke_hook",
            side_effect=lambda hook, **_kwargs: (
                [{"context": "PLUGIN-CTX"}] if hook == "pre_llm_call" else []
            ),
        ), patch(
            "hermes_cli.lifecycle.invoke_hook",
            side_effect=lambda hook, **_kwargs: (
                [{"context": "PLUGIN-CTX"}] if hook == "pre_llm_call" else []
            ),
        ), patch(
            "agent.turn_context._maybe_title_session_at_turn_start",
            return_value=None,
        ):
            context = build_turn_context(
                agent=agent,
                user_message="exact final user",
                system_message=None,
                conversation_history=db.get_messages_as_conversation(sid),
                task_id=None,
                stream_callback=None,
                persist_user_message=None,
                restore_or_build_system_prompt=lambda *_args, **_kwargs: None,
                install_safe_stdio=lambda: None,
                sanitize_surrogates=lambda text: text,
                summarize_user_message_for_log=lambda text: text,
                set_session_context=lambda _sid: None,
                set_current_write_origin=lambda _origin: None,
                ra=lambda: SimpleNamespace(_set_interrupt=lambda *_args, **_kwargs: None),
            )

        assert len(provider_requests) == 2
        assert getattr(agent, "_last_compaction_in_place") is in_place
        resumed_sid = agent.session_id
        assert (resumed_sid == sid) is in_place
        expected_clean_tail = f"exact final user\n\n{expected_todo}"
        expected_api_tail = f"{expected_clean_tail}\n\nPLUGIN-CTX"
        assert context.messages[-1]["content"] == expected_clean_tail
        assert context.messages[-1]["api_content"] == expected_api_tail
        checkpoint = context.messages[0]["codex_reasoning_items"][0]
        assert checkpoint[NATIVE_COMPACTION_METADATA_KEY]["tail_fence"] == (
            native_continuity_boundary_fence(context.messages[2:])
        )

        db.close()
        restarted_db = SessionDB(db_path=db_path)
        restarted, _display = restarted_db.get_resume_conversations(resumed_sid)
        restarted_checkpoint = restarted[0]["codex_reasoning_items"][0]
        assert restarted_checkpoint == checkpoint
        assert restarted[-1]["content"] == expected_clean_tail
        assert restarted[-1]["api_content"] == expected_api_tail
        assert restarted_checkpoint[NATIVE_COMPACTION_METADATA_KEY]["tail_fence"] == (
            native_continuity_boundary_fence(restarted[2:])
        )
        # conversation_loop applies persisted api_content to its provider copy
        # while preserving the untouched durable rows for fence validation.
        api_messages = deepcopy(restarted)
        for message in api_messages:
            api_content = message.pop("api_content", None)
            if isinstance(api_content, str) and api_content:
                message["content"] = api_content
        wire = _chat_messages_to_responses_input(
            api_messages,
            current_issuer_kind="openai_codex",
            native_compaction_eligible=True,
            native_continuity_source_messages=restarted,
        )
        assert wire == [
            {"type": "compaction", "encrypted_content": "checkpoint"},
            {"role": "user", "content": "exact model handoff"},
            {"role": "user", "content": expected_api_tail},
        ]
        restarted_db.close()


@pytest.mark.parametrize("in_place", [True, False])
def test_run_conversation_preserves_new_native_handoff_through_request_repair(in_place):
    """The first live request must not merge the handoff into the active user."""
    from hermes_state import SessionDB
    from run_agent import AIAgent

    sid = f"native_live_request_repair_{in_place}"
    with tempfile.TemporaryDirectory() as tmp:
        db = SessionDB(db_path=Path(tmp) / "state.db")
        db.create_session(sid, "cli", model="gpt-5.6-sol")
        db.append_message(sid, "user", "old instruction " * 2_000)
        db.append_message(sid, "assistant", "old result " * 2_000)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            agent = AIAgent(
                api_key="test-key",
                base_url="https://chatgpt.com/backend-api/codex",
                api_mode="codex_responses",
                model="gpt-5.6-sol",
                provider="openai-codex",
                quiet_mode=True,
                session_db=db,
                session_id=sid,
                skip_context_files=True,
                skip_memory=True,
                enabled_toolsets=[],
            )
        agent.codex_responses_native_compaction = True
        setattr(agent, "compression_in_place", in_place)
        agent.runtime_capabilities = {"native_compaction": True}
        agent.capabilities = {"native_compaction": True}
        agent._compression_feasibility_checked = True
        agent._disable_streaming = True
        agent._emit_status = lambda *_args, **_kwargs: None
        agent._emit_warning = lambda *_args, **_kwargs: None
        agent.commit_memory_session = lambda *_args, **_kwargs: None
        agent.context_compressor.threshold_tokens = 100
        agent.context_compressor.should_compress_preflight = lambda _messages: True

        responses = [
            _response(_message("exact model handoff\n")),
            _response({"type": "compaction", "encrypted_content": "checkpoint"}),
            SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="message",
                        content=[SimpleNamespace(type="output_text", text="START_CANARY_OK")],
                    )
                ],
                usage=SimpleNamespace(input_tokens=600, output_tokens=4, total_tokens=604),
                status="completed",
                model="gpt-5.6-sol",
            ),
            SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="message",
                        content=[SimpleNamespace(type="output_text", text="SECOND_CANARY_OK")],
                    )
                ],
                usage=SimpleNamespace(input_tokens=640, output_tokens=4, total_tokens=644),
                status="completed",
                model="gpt-5.6-sol",
            ),
        ]
        provider_requests = []
        agent._interruptible_api_call = lambda request: (
            provider_requests.append(deepcopy(request)) or responses.pop(0)
        )

        with patch(
            "hermes_cli.plugins.invoke_hook", return_value=[]
        ), patch(
            "hermes_cli.lifecycle.invoke_hook", return_value=[]
        ), patch(
            "agent.turn_context._maybe_title_session_at_turn_start", return_value=None
        ):
            result = agent.run_conversation(
                "Return exactly START_CANARY_OK.",
                conversation_history=db.get_messages_as_conversation(sid),
            )

        assert result["completed"] is True
        assert result["final_response"] == "START_CANARY_OK"
        assert len(provider_requests) == 3
        validate_persisted_native_compaction_history(result["messages"])
        live_checkpoint = result["messages"][0]["codex_reasoning_items"][0]
        assert result["messages"][1]["content"] == (
            live_checkpoint[NATIVE_COMPACTION_METADATA_KEY]["handoff"]
        )
        persisted, _display = db.get_resume_conversations(agent.session_id)
        validate_persisted_native_compaction_history(persisted)
        checkpoint = persisted[0]["codex_reasoning_items"][0]
        metadata = checkpoint[NATIVE_COMPACTION_METADATA_KEY]
        assert persisted[1]["content"] == metadata["handoff"] == "exact model handoff\n"

        getattr(agent, "context_compressor").should_compress_preflight = lambda _messages: False
        with patch(
            "hermes_cli.plugins.invoke_hook", return_value=[]
        ), patch(
            "hermes_cli.lifecycle.invoke_hook", return_value=[]
        ), patch(
            "agent.turn_context._maybe_title_session_at_turn_start", return_value=None
        ):
            second_result = agent.run_conversation(
                "Return exactly SECOND_CANARY_OK.",
                conversation_history=result["messages"],
            )

        assert second_result["completed"] is True
        assert second_result["final_response"] == "SECOND_CANARY_OK"
        assert len(provider_requests) == 4
        validate_persisted_native_compaction_history(second_result["messages"])
        validate_persisted_native_compaction_history(agent._session_messages)
        assert agent._session_messages == second_result["messages"]
        second_checkpoint = second_result["messages"][0]["codex_reasoning_items"][0]
        assert second_result["messages"][1]["content"] == (
            second_checkpoint[NATIVE_COMPACTION_METADATA_KEY]["handoff"]
        )
        db.close()


def test_run_conversation_replays_native_checkpoint_after_real_tool_result():
    """Mid-turn compression survives canonical repair of completed tool input."""
    from hermes_state import SessionDB
    from run_agent import AIAgent

    sid = "native_midturn_request_repair"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "proof.txt").write_text("blue orchid", encoding="utf-8")
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(sid, "cli", model="gpt-5.6-sol")
        db.append_message(sid, "user", "old instruction " * 2_000)
        db.append_message(sid, "assistant", "old result " * 2_000)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            agent = AIAgent(
                api_key="test-key",
                base_url="https://chatgpt.com/backend-api/codex",
                api_mode="codex_responses",
                model="gpt-5.6-sol",
                provider="openai-codex",
                quiet_mode=True,
                session_db=db,
                session_id=sid,
                skip_context_files=True,
                skip_memory=True,
                enabled_toolsets=["file_tools"],
            )
        agent.codex_responses_native_compaction = True
        agent.compression_in_place = False
        agent.runtime_capabilities = {"native_compaction": True}
        agent.capabilities = {"native_compaction": True}
        agent._compression_feasibility_checked = True
        agent._disable_streaming = True
        agent._emit_status = lambda *_args, **_kwargs: None
        agent._emit_warning = lambda *_args, **_kwargs: None
        agent.commit_memory_session = lambda *_args, **_kwargs: None
        agent.context_compressor.threshold_tokens = 100
        agent.context_compressor.should_compress_preflight = lambda _messages: False
        compression_checks = 0

        def _compress_after_tool(_tokens):
            nonlocal compression_checks
            compression_checks += 1
            return bool(provider_requests)

        agent.context_compressor.should_compress = _compress_after_tool

        def _main_response(*items):
            return SimpleNamespace(
                output=list(items),
                usage=SimpleNamespace(
                    input_tokens=600,
                    output_tokens=4,
                    total_tokens=604,
                ),
                status="completed",
                model="gpt-5.6-sol",
            )

        responses = [
            _main_response(
                SimpleNamespace(
                    type="function_call",
                    id="fc_1",
                    call_id="call_1",
                    name="search_files",
                    arguments=(
                        '{"target":"files","path":"'
                        + str(tmp_path)
                        + '","pattern":"*.txt","limit":5}'
                    ),
                )
            ),
            _response(_message("exact model handoff")),
            _response({"type": "compaction", "encrypted_content": "checkpoint"}),
            _main_response(
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(type="output_text", text="MIDTURN_CANARY_OK")
                    ],
                )
            ),
        ]
        provider_requests = []
        agent._interruptible_api_call = lambda request: (
            provider_requests.append(deepcopy(request)) or responses.pop(0)
        )

        with patch(
            "hermes_cli.plugins.invoke_hook", return_value=[]
        ), patch(
            "hermes_cli.lifecycle.invoke_hook", return_value=[]
        ), patch(
            "agent.turn_context._maybe_title_session_at_turn_start", return_value=None
        ):
            result = agent.run_conversation(
                "Find the proof file, then answer.",
                conversation_history=db.get_messages_as_conversation(sid),
            )

        assert result["completed"] is True
        assert result["final_response"] == "MIDTURN_CANARY_OK"
        assert len(provider_requests) == 4
        assert provider_requests[2]["context_management"] == [
            {"type": "compaction", "compact_threshold": 100}
        ]
        final_input = provider_requests[3]["input"]
        assert final_input[0] == {
            "type": "compaction",
            "encrypted_content": "checkpoint",
        }
        assert final_input[1] == {"role": "user", "content": "exact model handoff"}
        assert sum(item.get("type") == "function_call" for item in final_input) == 1

        resumed, _display = db.get_resume_conversations(agent.session_id)
        checkpoint = resumed[0]["codex_reasoning_items"][0]
        metadata = checkpoint[NATIVE_COMPACTION_METADATA_KEY]
        resumed_tail = resumed[2:2 + metadata["tail_count"]]
        live_tail = result["messages"][2:2 + metadata["tail_count"]]
        live_row_fences = [native_continuity_boundary_fence([row]) for row in live_tail]
        resumed_row_fences = [native_continuity_boundary_fence([row]) for row in resumed_tail]
        assert live_row_fences == resumed_row_fences
        assert metadata["tail_fence"] == native_continuity_boundary_fence(resumed_tail)
        validate_persisted_native_compaction_history(resumed)
        resumed_wire = _chat_messages_to_responses_input(
            resumed,
            current_issuer_kind="openai_codex",
            native_compaction_eligible=True,
        )
        assert resumed_wire[0] == {
            "type": "compaction",
            "encrypted_content": "checkpoint",
        }
        assert sum(item.get("type") == "function_call" for item in resumed_wire) == 1
        db.close()
