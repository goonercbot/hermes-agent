"""Steering must be durable before native note authentication, not a tool edit."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
from tests.run_agent.test_native_note_refresh import blocking_router, ARGS, reply
from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
from agent.native_incremental_handoff import bind_native_incremental_replay_projection, restore_native_incremental_note
from agent.native_note_refresh import issue_native_note_refresh, execute_native_note_refresh


@pytest.mark.parametrize('projection', [False, True])
@pytest.mark.parametrize('long_arguments', [False, True])
def test_persisted_steer_survives_real_note_transaction(tmp_path, monkeypatch, blocking_router, projection, long_arguments):
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    (tmp_path/'config.yaml').write_text('compression:\n  enabled: true\n  codex_responses_native: true\n  native_incremental_handoff: true\n')
    from hermes_state import SessionDB
    from run_agent import AIAgent
    db=SessionDB(db_path=tmp_path/'state.db');sid='steered-subagent'
    db.create_session(sid,'subagent',model='gpt-5.6-terra')
    a=AIAgent(api_key='fixture',base_url='https://chatgpt.com/backend-api/codex',api_mode='codex_responses',model='gpt-5.6-terra',provider='openai-codex',session_db=db,session_id=sid,platform='subagent',quiet_mode=True,skip_memory=True,skip_context_files=True,skip_background_review=True,enabled_toolsets=['continuity'])
    a._end_session_on_close=False
    try:
        db.append_message(sid,'user','Preserve the objective')
        db.append_message(sid,'assistant','',tool_calls=[{'id':'write','type':'function','function':{'name':'write_file','arguments':json.dumps({'path':'fixture.txt','content':'x'*(32000 if long_arguments else 3)})}}])
        db.append_message(sid,'tool','Original durable tool evidence',tool_name='write_file',tool_call_id='write')
        source=db.get_messages_as_conversation(sid,repair_alternation=False)
        history=deepcopy(source)
        if projection:
            history[-1]['content']='Shortened model-only view'
            assert bind_native_incremental_replay_projection(a,source_messages=source,replay_messages=history)
        before=deepcopy(history)
        monkeypatch.setattr(a,'_drain_pending_steer',lambda:'Keep MARBLE HERON. Deployment remains paused.')
        apply_pending_steer_to_tool_results(a,history,1)
        assert history[:len(before)]==before
        assert history[-1]['role']=='user' and 'MARBLE HERON' in history[-1]['content']
        persisted=db.get_messages_as_conversation(sid,repair_alternation=False)
        assert persisted[:len(source)]==source and len(persisted)==len(source)+1
        assert persisted[-1]['content']==history[-1]['content']
        a._current_api_request_id='steer:maintenance:1'
        cap=issue_native_note_refresh(a,history);cap.bind_request({})
        response=reply(NS(type='function_call',id='fc-note',call_id='new-note',name='continuity_note',arguments=json.dumps(ARGS)))
        execute_native_note_refresh(a,cap,response,history,sid)
        stored=db.get_messages_as_conversation(sid,repair_alternation=False)
        note=restore_native_incremental_note(NS(session_id=sid),stored)
        assert note is not None and note==a._native_incremental_handoff_note
        for index in (2,3):
            tampered=deepcopy(stored);tampered[index]['content']+=' tampered'
            assert restore_native_incremental_note(NS(session_id=sid),tampered) is None
        db.close();db=SessionDB(db_path=tmp_path/'state.db')
        assert restore_native_incremental_note(NS(session_id=sid),db.get_messages_as_conversation(sid,repair_alternation=False))==note
    finally:
        a.close();db.close()


def test_native_steer_write_failure_is_not_success():
    history=[{'role':'tool','content':'original','tool_call_id':'call'}]
    agent=NS(native_incremental_handoff_enabled=True,_drain_pending_steer=lambda:'Preserve this instruction',_flush_messages_to_session_db=lambda messages:False)
    with pytest.raises(RuntimeError,match='steering message persistence failed'):
        apply_pending_steer_to_tool_results(agent,history,1)
    assert history[0]['content']=='original'
    assert 'Preserve this instruction' in history[-1]['content']


def test_ordinary_steer_behavior_is_unchanged():
    history=[{'role':'tool','content':'original','tool_call_id':'call'}]
    agent=NS(native_incremental_handoff_enabled=False,_drain_pending_steer=lambda:'Ordinary instruction')
    apply_pending_steer_to_tool_results(agent,history,1)
    assert len(history)==1 and history[0]['content'].startswith('original')
    assert 'Ordinary instruction' in history[0]['content']
