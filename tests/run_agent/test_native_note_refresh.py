"""A stale valid note must trigger maintenance, not repeat useless compaction."""
import json
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
from agent.native_incremental_handoff import (
    record_native_incremental_note_from_tool_call, restore_native_incremental_note,
    native_incremental_compact_context,
)
from agent.native_compaction import validate_persisted_native_compaction_history

ARGS=dict(objective='Keep current user corrections',current_plan='Evidence verified; deployment paused',next_action='Answer the current user',blockers=[])

def reply(*items,model='gpt-6-astra',tokens=200000):
    return NS(output=list(items),usage=NS(input_tokens=tokens,output_tokens=30,total_tokens=tokens+30),status='completed',model=model)

@pytest.mark.parametrize('in_place',[False,True])
def test_stale_note_forces_ordinary_refresh_then_commits_and_continues(tmp_path,monkeypatch,in_place):
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    (tmp_path/'config.yaml').write_text('compression:\n  enabled: true\n  codex_responses_native: true\n  native_incremental_handoff: true\n  native_incremental_model: gpt-5.6-luna\n  native_incremental_compact_threshold: 128000\n')
    from hermes_state import SessionDB
    from run_agent import AIAgent
    db=SessionDB(db_path=tmp_path/'state.db');sid='stale-note-regression'
    db.create_session(sid,'cli',model='gpt-6-astra')
    with patch('hermes_cli.plugins.invoke_hook',return_value=[]),patch('hermes_cli.lifecycle.invoke_hook',return_value=[]),patch('agent.turn_context._maybe_title_session_at_turn_start',return_value=None):
        a=AIAgent(api_key='fixture',base_url='https://chatgpt.com/backend-api/codex',api_mode='codex_responses',model='gpt-6-astra',provider='openai-codex',session_db=db,session_id=sid,quiet_mode=True,skip_memory=True,skip_context_files=True,skip_background_review=True,enabled_toolsets=['continuity'])
        a.compression_in_place=in_place;a._disable_streaming=True;a._compression_feasibility_checked=True
        a._emit_status=lambda *args,**kwargs:None;a.commit_memory_session=lambda *args,**kwargs:None
        warnings=[];a._emit_warning=lambda text,*args,**kwargs:warnings.append(text)
        a.context_compressor.threshold_tokens=100000
        db.append_message(sid,'user','Initial objective')
        db.append_message(sid,'assistant','',tool_calls=[dict(id='old-note',type='function',function=dict(name='continuity_note',arguments=json.dumps(ARGS)))])
        source=db.get_messages_as_conversation(sid)
        result=record_native_incremental_note_from_tool_call(a,ARGS,source)
        db.append_message(sid,'tool',result,tool_call_id='old-note',tool_name='continuity_note')
        db.append_message(sid,'user','Keep new decisions. '*20000)
        db.append_message(sid,'assistant','Verified evidence. '*20000)
        source=db.get_messages_as_conversation(sid);before=deepcopy(source)
        assert restore_native_incremental_note(a,source) is not None

        calls=[]
        def request(req):
            calls.append(deepcopy(req))
            if len(calls)==1:
                assert req['model']=='gpt-6-astra','Must refresh before paying for a futile native request'
                assert req['tool_choice']==dict(type='function',name='continuity_note')
                assert [t['name'] for t in req['tools']]==['continuity_note']
                return reply(NS(type='function_call',id='fc-refresh',call_id='fresh-note',name='continuity_note',arguments=json.dumps(ARGS)))
            if len(calls)==2:
                assert req['model']=='gpt-5.6-luna'
                assert 'Context maintenance: call continuity_note now' not in req['instructions']
                return reply(NS(type='compaction',id='cp',encrypted_content='fixture-checkpoint'),model='gpt-5.6-luna')
            assert len(calls)==3 and req['model']=='gpt-6-astra'
            assert req.get('tool_choice')!=dict(type='function',name='continuity_note')
            return reply(NS(type='message',role='assistant',content=[NS(type='output_text',text='CURRENT_REQUEST_OK')]),tokens=600)
        a._interruptible_api_call=request
        try:
            result=a.run_conversation('Newest instruction: reply CURRENT_REQUEST_OK; no deployment.',conversation_history=source)
            assert result['completed'] and result['final_response']=='CURRENT_REQUEST_OK'
            assert len(calls)==3
            stored=db.get_messages_as_conversation(a.session_id)
            assert validate_persisted_native_compaction_history(stored)
            assert len(stored)<12
            assert 'structural_backoff' not in '\n'.join(warnings)
            assert 'Newest instruction' in json.dumps(calls[-1]['input'])
            if not in_place:
                assert db.get_messages_as_conversation(sid)[:len(before)]==before
        finally:a.close();db.close()


def test_stale_tail_defers_without_provider_and_preserves_source():
    from tests.run_agent.test_native_incremental_handoff import _agent,_note
    a,calls=_agent([])
    messages=[{'role':'user','content':'earlier objective'},{'role':'assistant','content':'earlier state'}]
    _note(a,messages)
    messages.append({'role':'user','content':'new evidence '*20000})
    before=deepcopy(messages)
    assert native_incremental_compact_context(a,messages)==before
    assert calls==[] and messages==before
    assert a._last_native_incremental_compaction['disposition']=='refresh_required'


def test_refresh_is_scoped_and_repeated_failed_maintenance_is_bounded():
    from tests.run_agent.test_native_incremental_handoff import _agent,_note
    from agent.native_incremental_handoff import prepare_native_note_refresh_request
    a,_=_agent([]);a._current_api_request_id='turn1:api:1'
    messages=[{'role':'user','content':'earlier state'}];_note(a,messages)
    messages.append({'role':'assistant','content':'new evidence '*20000})
    request={'instructions':'unchanged base','tools':[{'type':'function','name':'terminal'}]}
    before=deepcopy(messages)
    assert prepare_native_note_refresh_request(a,messages,request)
    assert request['tools'][0]['name']=='continuity_note' and len(request['tools'])==1
    assert messages==before and not hasattr(a,'tools')
    assert prepare_native_note_refresh_request(a,messages,request) # same physical-request retry
    a._current_api_request_id='turn1:api:2'
    with pytest.raises(RuntimeError,match='did not advance'):prepare_native_note_refresh_request(a,messages,{})
    a._current_api_request_id='turn2:api:1'
    assert prepare_native_note_refresh_request(a,messages,{})
    messages.append({'role':'user','content':'latest concise correction'})
    _note(a,messages)
    clean={'instructions':'unchanged base','tools':[]}
    assert not prepare_native_note_refresh_request(a,messages,clean)
    assert clean=={'instructions':'unchanged base','tools':[]}


def test_refresh_does_not_retain_all_work_since_the_latest_user():
    from tests.run_agent.test_native_incremental_handoff import _agent
    from agent.native_incremental_handoff import _protected_tail_since_note
    a,_=_agent([])
    messages=[{'role':'user','content':'Current user instruction; preserve exactly'}]
    messages.extend({'role':'assistant','content':'checkpointed old work '*2000} for _ in range(12))
    messages.append({'role':'assistant','content':'','tool_calls':[{'id':'fresh','type':'function','function':{'name':'continuity_note','arguments':json.dumps(ARGS)}}]})
    raw=record_native_incremental_note_from_tool_call(a,ARGS,messages)
    messages.append({'role':'tool','name':'continuity_note','tool_call_id':'fresh','content':raw})
    original=deepcopy(messages)
    tail=_protected_tail_since_note(messages,a._native_incremental_handoff_note)
    assert tail==[messages[0],*messages[-2:]]
    assert messages==original


def test_large_latest_user_does_not_force_repeated_fresh_notes():
    from tests.run_agent.test_native_incremental_handoff import _agent,_note
    from agent.native_incremental_handoff import native_note_refresh_required
    a,_=_agent([]);messages=[{'role':'user','content':'large required user input '*10000}]
    _note(a,messages)
    assert not native_note_refresh_required(a,messages)


def test_non_native_requests_are_unchanged():
    from tests.run_agent.test_native_incremental_handoff import _agent
    from agent.native_incremental_handoff import prepare_native_note_refresh_request
    a,_=_agent([]);a.api_mode='chat_completions';a.provider='minimax';a.model='MiniMax-M3'
    request={'messages':[],'tools':[]};before=deepcopy(request)
    assert not prepare_native_note_refresh_request(a,[{'role':'user','content':'bulk '*30000}],request)
    assert request==before


def test_fresh_note_rearms_only_readiness_backoff_not_failure_guards():
    from tests.run_agent.test_native_incremental_handoff import _agent,_note
    a,_=_agent([])
    a._last_native_incremental_compaction={'disposition':'refresh_required'}
    a.context_compressor._structural_no_op_backoff_until=999999
    a.context_compressor._native_no_progress_rearm_tokens=999999
    a.context_compressor._summary_failure_cooldown_until=888888
    a.context_compressor._ineffective_compression_count=2
    _note(a,[{'role':'user','content':'valid updated objective'}])
    assert a.context_compressor._structural_no_op_backoff_until==0
    assert a.context_compressor._native_no_progress_rearm_tokens==0
    assert a.context_compressor._summary_failure_cooldown_until==888888
    assert a.context_compressor._ineffective_compression_count==2
