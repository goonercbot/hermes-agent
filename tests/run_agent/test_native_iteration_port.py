"""Production iteration-prep coverage for authenticated native replay."""
from copy import deepcopy
from types import SimpleNamespace as NS


def _native_checkpoint_with_unsealed_tool_pair():
    from agent.native_incremental_handoff import native_incremental_compact_context
    from tests.run_agent.test_native_incremental_handoff import _agent, _note, _response, _source

    producer, _ = _agent([_response({"type": "compaction", "encrypted_content": "cipher"})])
    source = _source()
    _note(producer, source)
    protected = native_incremental_compact_context(producer, source)
    protected.extend([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "unsealed-call", "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        },
        {
            "role": "tool", "tool_call_id": "unsealed-call", "name": "read_file",
            "content": "unsealed result",
        },
    ])
    return protected


def _iteration_agent(*, steer=None, native=False, flushed=None):
    return NS(
        step_callback=None,
        _skill_nudge_interval=0,
        valid_tool_names=(),
        _drain_pending_steer=lambda: steer,
        _sanitize_args_cursor={},
        _sanitize_tool_call_arguments=lambda *_a, **_k: 0,
        session_id="native-iteration-port",
        _last_flushed_db_idx=1,
        native_incremental_handoff_enabled=native,
        _flush_messages_to_session_db=(
            (lambda messages: flushed.append(deepcopy(messages)) or True)
            if flushed is not None else lambda _messages: True
        ),
    )


def test_prepare_iteration_keeps_sealed_source_and_unsealed_tool_pair_with_stale_cursor():
    """The real pre-request callable repairs a disposable whole native projection."""
    from agent.native_compaction import validate_persisted_native_compaction_history
    from agent.turn_iteration_prep import prepare_iteration

    source = _native_checkpoint_with_unsealed_tool_pair()
    source_before = deepcopy(source)
    agent = _iteration_agent()

    prepared = prepare_iteration(agent, messages=source, api_call_count=1)

    # The durable checkpoint source is never modified by request preparation.
    assert source == source_before
    # The outgoing projection retains the exact sealed span and the entire
    # unsealed assistant/tool pair even though the agent cursor is stale.
    validate_persisted_native_compaction_history(prepared.messages)
    assert prepared.messages == source_before
    assert prepared.messages[-2]["tool_calls"][0]["id"] == "unsealed-call"
    assert prepared.messages[-1]["tool_call_id"] == "unsealed-call"


def test_prepare_iteration_persists_native_mid_api_steer_as_append_only_user_event():
    """Native steering never rewrites the already-durable tool result."""
    from agent.turn_iteration_prep import prepare_iteration

    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call", "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "call", "name": "read_file", "content": "durable result"},
    ]
    flushed = []
    agent = _iteration_agent(steer="Keep MARBLE HERON.", native=True, flushed=flushed)

    prepared = prepare_iteration(agent, messages=messages, api_call_count=1)

    assert messages[1]["content"] == "durable result"
    assert prepared.messages[-1]["role"] == "user"
    assert "MARBLE HERON" in prepared.messages[-1]["content"]
    assert len(flushed) == 1
    assert flushed[0][-1] == prepared.messages[-1]


def test_prepare_iteration_keeps_ordinary_mid_api_steer_on_tool_result():
    """The disabled path retains established cache-safe tool-result injection."""
    from agent.turn_iteration_prep import prepare_iteration

    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call", "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "call", "name": "read_file", "content": "ordinary result"},
    ]
    agent = _iteration_agent(steer="Ordinary instruction", native=False)

    prepared = prepare_iteration(agent, messages=messages, api_call_count=1)

    assert len(prepared.messages) == 2
    assert "Ordinary instruction" in prepared.messages[-1]["content"]
