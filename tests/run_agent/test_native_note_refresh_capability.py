"""Capability boundaries and real failover transport with the candidate router."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest

from agent.native_note_refresh import (
    NativeNoteRefreshFailure, close_native_note_refresh,
    consume_native_note_refresh_capability, is_native_note_refresh_authorized,
)
from agent.native_incremental_handoff import prepare_native_note_refresh_request
ARGS = dict(
    objective="Keep evidence", current_plan="Verify locally", next_action="Report", blockers=[]
)


def capability_fixture():
    from tests.run_agent.test_native_incremental_handoff import _agent, _note
    agent, _ = _agent([])
    agent._current_api_request_id = 'turn:api:1'
    messages = [{'role': 'user', 'content': 'original'}]
    _note(agent, messages)
    messages.append({'role': 'assistant', 'content': 'evidence ' * 30000})
    request = {'instructions': 'ordinary', 'tools': []}
    capability = prepare_native_note_refresh_request(agent, messages, request)
    assert capability is not False
    capability.bind_request(request)
    capability.call_id = 'note-call'
    capability.arguments = deepcopy(ARGS)
    return agent, capability


def proof(agent, capability, **changes):
    values = dict(tool_name='continuity_note', args=deepcopy(ARGS),
                  session_id=agent.session_id, tool_call_id='note-call',
                  api_request_id='turn:api:1')
    values.update(changes)
    return is_native_note_refresh_authorized(capability, **values)


def test_capability_requires_exact_live_consumed_dispatch():
    agent, capability = capability_fixture()
    assert not proof(agent, capability)
    consume_native_note_refresh_capability(capability, agent, 'continuity_note', ARGS, 'note-call')
    assert proof(agent, capability)
    assert proof(agent, capability)  # validation is pure; execution is single-use
    with pytest.raises(NativeNoteRefreshFailure):
        consume_native_note_refresh_capability(capability, agent, 'continuity_note', ARGS, 'note-call')
    for changes in (
        {'tool_name': 'terminal'}, {'session_id': 'other'},
        {'tool_call_id': 'other'}, {'api_request_id': 'turn:api:2'},
        {'args': {**ARGS, 'maintenance_context': True}}, {'args': []},
        {'args': {**ARGS, 'objective': 'forged'}},
    ):
        assert not proof(agent, capability, **changes)
    agent._current_api_request_id = 'turn:api:2'
    assert not proof(agent, capability)
    agent._current_api_request_id = 'turn:api:1'
    agent._interrupt_requested = True
    assert not proof(agent, capability)
    agent._interrupt_requested = False
    close_native_note_refresh(agent)
    assert not proof(agent, capability)
    for forged in (None, True, {}, NS(authorized=True), object(), object.__new__(type(capability))):
        assert not proof(agent, forged)

