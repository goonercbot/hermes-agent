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
from tests.run_agent.test_native_note_refresh import ARGS, blocking_router


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


def test_genuine_provider_failure_restores_ordinary_fallback_request(tmp_path, monkeypatch, blocking_router):
    import httpx
    from openai import InternalServerError
    from run_agent import AIAgent
    from hermes_state import SessionDB
    from tests.run_agent.test_native_incremental_integration import chat_response

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path/'config.yaml').write_text('compression:\n  enabled: true\n  codex_responses_native: true\n  native_incremental_handoff: true\n  native_incremental_compact_threshold: 128000\n')
    db = SessionDB(db_path=tmp_path/'state.db')
    sid = 'refresh-provider-failure'
    db.create_session(sid, 'cli', model='gpt-6-astra')
    source = [{'role': 'user', 'content': 'Preserve evidence. ' * 20000},
              {'role': 'assistant', 'content': 'Historical work.'}]
    for row in source:
        db.append_message(sid, row['role'], row['content'])
    agent = AIAgent(api_key='fixture', base_url='https://chatgpt.com/backend-api/codex',
                    api_mode='codex_responses', model='gpt-6-astra', provider='openai-codex',
                    session_db=db, session_id=sid, quiet_mode=True, skip_memory=True,
                    skip_context_files=True, skip_background_review=True,
                    enabled_toolsets=['continuity', 'terminal', 'fleet_task_router'])
    agent._disable_streaming = True
    agent._compression_feasibility_checked = True
    agent.context_compressor.should_compress_preflight = lambda _: False
    agent.context_compressor.should_compress = lambda _: False
    agent.commit_memory_session = lambda *args, **kwargs: None
    agent._fallback_chain = [{'provider':'minimax', 'model':'MiniMax-M3',
                              'base_url':'https://api.minimax.io/v1', 'api_key':'fixture',
                              'api_mode':'chat_completions'}]
    requests = []
    capabilities = []
    ordinary_tools = deepcopy(agent.tools)
    def request(kwargs):
        requests.append(deepcopy(kwargs))
        if kwargs['model'] == 'gpt-6-astra':
            assert kwargs['tool_choice'] == {'type':'function', 'name':'continuity_note'}
            capabilities.append(agent._native_note_refresh_capability)
            raise InternalServerError('provider unavailable', response=httpx.Response(503, request=httpx.Request('POST','https://provider.invalid')), body=None)
        assert agent._native_note_refresh_capability is None
        assert 'messages' in kwargs and 'input' not in kwargs
        assert kwargs.get('tool_choice') != {'type':'function', 'name':'continuity_note'}
        assert 'terminal' in {tool['function']['name'] for tool in kwargs['tools']}
        assert 'Context maintenance: call continuity_note now' not in json.dumps(kwargs)
        assert 'Latest correction: return FALLBACK_OK; never deploy.' in json.dumps(kwargs['messages'])
        return chat_response('FALLBACK_OK')
    agent._interruptible_api_call = request
    monkeypatch.setattr('agent.auxiliary_client.resolve_provider_client', lambda *args, **kwargs: (agent.client, 'MiniMax-M3'))
    monkeypatch.setattr('agent.conversation_loop.jittered_backoff', lambda *args, **kwargs: 0.0)
    monkeypatch.setattr('agent.turn_context._maybe_title_session_at_turn_start', lambda *args, **kwargs: None)
    try:
        result = agent.run_conversation('Latest correction: return FALLBACK_OK; never deploy.', conversation_history=source)
        assert result['completed'] and result['final_response'] == 'FALLBACK_OK'
        assert requests[-1]['model'] == 'MiniMax-M3'
        assert agent.tools == ordinary_tools
        assert capabilities and all(cap.agent is None for cap in capabilities)
        stored = db.get_messages_as_conversation(sid)
        assert 'Preserve evidence.' in json.dumps(stored)
        assert not any(row.get('tool_name') == 'continuity_note' for row in stored)
    finally:
        agent.close()
        db.close()
