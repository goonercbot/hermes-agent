"""First-checkpoint native request repairs retain canonical source evidence."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest


def _agent(*, native=True, sanitizer=None):
    return NS(
        step_callback=None,
        _skill_nudge_interval=0,
        valid_tool_names=(),
        _drain_pending_steer=lambda: None,
        _sanitize_args_cursor={},
        _sanitize_tool_call_arguments=sanitizer or (lambda *_a, **_k: 0),
        session_id="first-checkpoint-projection",
        _last_flushed_db_idx=0,
        native_incremental_handoff_enabled=native,
        native_incremental_handoff_model="gpt-5.6-luna",
        compression_enabled=True,
        _codex_reasoning_replay_enabled=True,
        api_mode="codex_responses",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        _flush_messages_to_session_db=lambda _messages: True,
    )


def test_first_checkpoint_user_repair_keeps_canonical_source_and_issues_capability():
    from agent.native_incremental_handoff import _note_fence, _projection_source_for_messages
    from agent.native_incremental_handoff import prepare_native_note_refresh_request
    from agent.native_note_refresh import close_native_note_refresh
    from agent.turn_iteration_prep import prepare_iteration

    source = [
        {"role": "user", "content": "durable instruction"},
        {"role": "user", "content": "latest correction"},
        {"role": "assistant", "content": "evidence " * 30000},
    ]
    canonical = deepcopy(source)
    agent = _agent()
    agent._current_api_request_id = "first:api:1"

    prepared = prepare_iteration(agent, messages=source, api_call_count=1)

    assert source == canonical
    assert prepared.messages[0]["content"] == "durable instruction\n\nlatest correction"
    authenticated_source, cursor = _projection_source_for_messages(agent, prepared.messages)
    assert authenticated_source == canonical
    assert cursor == len(prepared.messages)

    request = {"instructions": "ordinary", "tools": []}
    capability = prepare_native_note_refresh_request(agent, prepared.messages, request)
    try:
        assert capability is not False
        assert capability.canonical_source_fence == _note_fence(canonical)
    finally:
        close_native_note_refresh(agent)


def test_first_checkpoint_sanitizer_and_ghost_repair_bind_only_disposable_replay():
    from agent.conversation_loop import _INTERRUPT_SCAFFOLD_MARKER
    from agent.native_incremental_handoff import _projection_source_for_messages
    from agent.turn_iteration_prep import prepare_iteration

    def sanitize(messages, **_kwargs):
        messages[1]["tool_calls"][0]["function"]["arguments"] = "{}"
        return 1

    source = [
        {"role": "user", "content": "durable instruction"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call", "type": "function",
            "function": {"name": "read_file", "arguments": "not-json"},
        }]},
        {"role": "tool", "tool_call_id": "call", "name": "read_file", "content": "result"},
        {"role": "assistant", "display_kind": "hidden", "content": _INTERRUPT_SCAFFOLD_MARKER},
    ]
    canonical = deepcopy(source)
    agent = _agent(sanitizer=sanitize)

    prepared = prepare_iteration(agent, messages=source, api_call_count=1)

    assert source == canonical
    assert len(prepared.messages) == len(canonical) - 1
    assert prepared.messages[1]["tool_calls"][0]["function"]["arguments"] == "{}"
    authenticated_source, _ = _projection_source_for_messages(agent, prepared.messages)
    assert authenticated_source == canonical

    tampered = deepcopy(prepared.messages)
    tampered[0]["content"] = "forged request source"
    with pytest.raises(ValueError, match="native request source unauthenticated"):
        prepare_iteration(agent, messages=tampered, api_call_count=2)


def test_ordinary_first_checkpoint_shape_keeps_existing_in_place_repair_semantics():
    from agent.turn_iteration_prep import prepare_iteration

    messages = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ]
    agent = _agent(native=False)

    prepared = prepare_iteration(agent, messages=messages, api_call_count=1)

    assert prepared.messages == [{"role": "user", "content": "first\n\nsecond"}]
    # Ordinary mode retains its legacy repair helper (including its mutable-row
    # behavior); only native first-checkpoint requests gain a fenced copy.
    assert messages[0]["content"] == "first\n\nsecond"
    assert len(messages) == 2
    assert not hasattr(agent, "_native_incremental_request_projection")


@pytest.mark.parametrize("gateway_bound", [False, True])
def test_staged_note_uses_selected_request_projection_boundary(gateway_bound):
    from agent.native_incremental_handoff import (
        bind_native_incremental_replay_projection,
        bind_native_incremental_request_projection,
        create_native_incremental_note, record_native_incremental_note,
        _staged_note,
    )
    agent = _agent()
    source = [{"role": "user", "content": "first"},
              {"role": "user", "content": "correction"}]
    replay = [{"role": "user", "content": "first\n\ncorrection"}]
    if gateway_bound:
        assert bind_native_incremental_replay_projection(
            agent, source_messages=source[:1], replay_messages=source[:1])
    assert bind_native_incremental_request_projection(
        agent, source_messages=source, replay_messages=replay)
    note = create_native_incremental_note(
        session_id=agent.session_id, source_messages=source,
        objective="retain", current_plan="verify", next_action="continue", blockers=[])
    record_native_incremental_note(agent, note, source)
    messages = replay + [{"role": "assistant", "content": "continued"}]
    assert _staged_note(agent, messages) == note
    assert agent._native_incremental_handoff_projection_cursor == len(replay)
    # An interior cursor in a length-changing repair remains ambiguous.
    interior = create_native_incremental_note(
        session_id=agent.session_id, source_messages=source[:1],
        objective="retain", current_plan="verify", next_action="continue", blockers=[])
    record_native_incremental_note(agent, interior, source[:1])
    assert _staged_note(agent, messages) is None


def test_unchanged_next_iteration_keeps_first_note_cursor_mapping():
    from agent.native_incremental_handoff import (
        create_native_incremental_note, record_native_incremental_note, _staged_note)
    from agent.turn_iteration_prep import prepare_iteration
    agent = _agent()
    source = [{"role": "user", "content": "first"},
              {"role": "user", "content": "correction"}]
    first = prepare_iteration(agent, messages=source, api_call_count=1)
    note = create_native_incremental_note(
        session_id=agent.session_id, source_messages=source,
        objective="retain", current_plan="verify", next_action="continue", blockers=[])
    record_native_incremental_note(agent, note, source)
    second = prepare_iteration(agent, messages=first.messages + [
        {"role": "assistant", "content": "note recorded"}], api_call_count=2)
    assert _staged_note(agent, second.messages) == note
    assert agent._native_incremental_handoff_projection_cursor == len(first.messages)
