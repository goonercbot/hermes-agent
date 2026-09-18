"""Disposable acceptance probes for the exact worker candidate; no live provider."""
import json
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest

from agent.native_incremental_handoff import record_native_incremental_note_from_tool_call, restore_native_incremental_note

ARGS = dict(objective='Approved task', current_plan='Keep verified work', next_action='Continue', blockers=[])
BLOCKER = 'Only continuity recording is available; terminal tools are unavailable.'


def completed_reply(text):
    return NS(output=[NS(type='message',role='assistant',content=[NS(type='output_text',text=text)])], usage=NS(input_tokens=600, output_tokens=30,total_tokens=630), status='completed', model='gpt-6-astra')


@pytest.mark.parametrize('restart', [False, True], ids=['next-ordinary-turn-cache', 'fresh-agent-restart'])
def test_resume_scope_survives_ordinary_turn_and_restart(tmp_path, monkeypatch, restart):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path/'config.yaml').write_text('compression:\n  enabled: true\n  codex_responses_native: true\n  native_incremental_handoff: true\n  native_incremental_model: gpt-5.6-luna\n  native_incremental_compact_threshold: 128000\n')
    from hermes_state import SessionDB
    from run_agent import AIAgent
    db = SessionDB(db_path=tmp_path/'state.db')
    sid = 'maintenance-resume-acceptance'
    db.create_session(sid, 'cli', model='gpt-6-astra')
    def new_agent():
        a = AIAgent(api_key='fixture', base_url='https://chatgpt.com/backend-api/codex',
                    api_mode='codex_responses', model='gpt-6-astra', provider='openai-codex',
                    session_db=db, session_id=sid, quiet_mode=True, skip_memory=True,
                    skip_context_files=True, skip_background_review=True,
                    enabled_toolsets=['continuity', 'terminal'])
        a._disable_streaming = True
        a._compression_feasibility_checked = True
        a._emit_status = lambda *args, **kwargs: None
        a.commit_memory_session = lambda *args, **kwargs: None
        a.context_compressor.threshold_tokens = 1000000
        return a
    def reply(item):
        return NS(output=[item], usage=NS(input_tokens=600, output_tokens=30,total_tokens=630),
                  status='completed', model='gpt-6-astra')
    with patch('agent.turn_context._maybe_title_session_at_turn_start',return_value=None):
        a=new_agent()
        try:
            db.append_message(sid,'user','Approved task')
            db.append_message(sid,'assistant','',tool_calls=[dict(id='old-note',type='function',function=dict(name='continuity_note',arguments=json.dumps(ARGS)))])
            raw=record_native_incremental_note_from_tool_call(a,ARGS,db.get_messages_as_conversation(sid))
            db.append_message(sid,'tool',raw,tool_call_id='old-note',tool_name='continuity_note')
            db.append_message(sid,'assistant','Verified earlier work. '*20000)
            source=db.get_messages_as_conversation(sid)
            assert restore_native_incremental_note(a,source)
            calls=[]
            def provider(req):
                calls.append(deepcopy(req))
                if len(calls)==1:
                    assert [t['name'] for t in req['tools']]==['continuity_note']
                    return reply(NS(type='function_call',id='fc-new',call_id='new-note',name='continuity_note',arguments=json.dumps({**ARGS,'blockers':[BLOCKER]})))
                assert len(calls)==2
                return reply(NS(type='message',role='assistant',content=[NS(type='output_text',text='Observed state.')]))
            a._interruptible_api_call=provider
            assert a.run_conversation('Continue.',conversation_history=source)['completed']
            assert len(calls)==2
            first_normal=calls[1]
            assert 'single-tool inventory was request-local' in first_normal['instructions']
            if restart:
                a.close()
                db.close()
                db = SessionDB(db_path=tmp_path/'state.db')
                a=new_agent()
            followups=[]
            def followup(req):
                followups.append(deepcopy(req))
                return reply(NS(type='message',role='assistant',content=[NS(type='output_text',text='Follow-up state.')]))
            a._interruptible_api_call=followup
            assert a.run_conversation('Continue the unfinished task.',conversation_history=db.get_messages_as_conversation(sid))['completed']
            assert len(followups)==1
            resumed=followups[0]
            assert 'terminal' in {t['name'] for t in resumed['tools']}
            assert BLOCKER in json.dumps(resumed['input'])
            assert 'Continue the unfinished task.' in json.dumps(resumed['input'])
            if restart:
                assert 'single-tool inventory was request-local' in resumed['instructions'], 'Persisted contaminated note survives but host scope disappears on fresh-agent restart without a checkpoint'
            else:
                assert resumed['instructions']==first_normal['instructions'], 'Ordinary instructions change after one request despite unchanged maintenance evidence (cache churn)'
            db.append_message(sid, 'assistant', 'Later verified work. ' * 20000)
            refreshed=[]
            def refresh_then_work(req):
                refreshed.append(deepcopy(req))
                if len(refreshed)==1:
                    assert [t['name'] for t in req['tools']]==['continuity_note']
                    return reply(NS(type='function_call',id='fc-later',call_id='later-note',name='continuity_note',arguments=json.dumps(ARGS)))
                assert refreshed[1]['instructions']==first_normal['instructions']
                if len(refreshed)==2:
                    return reply(NS(type='function_call',id='fc-terminal',call_id='later-tool',name='terminal',arguments=json.dumps({'command':'printf stable-native-prefix','workdir':str(tmp_path)})))
                assert len(refreshed)==3
                assert refreshed[2]['instructions']==first_normal['instructions']
                return reply(NS(type='message',role='assistant',content=[NS(type='output_text',text='Stable ordinary scope.')]))
            a._interruptible_api_call=refresh_then_work
            result=a.run_conversation('Continue after later evidence.',conversation_history=db.get_messages_as_conversation(sid))
            assert result['completed'] and result['final_response']=='Stable ordinary scope.'
            assert len(refreshed)==3
        finally:
            a.close()
            db.close()


def test_static_eligible_scope_does_not_grant_disabled_tool(tmp_path, monkeypatch):
    """The static scope is a warning, never an authorization or schema mutation."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path/'config.yaml').write_text('compression:\n  enabled: true\n  codex_responses_native: true\n  native_incremental_handoff: true\n  native_incremental_model: gpt-5.6-luna\n')
    from hermes_state import SessionDB
    from run_agent import AIAgent
    db=SessionDB(db_path=tmp_path/'state.db'); sid='native-static-disabled-tool'
    db.create_session(sid,'cli',model='gpt-6-astra')
    with patch('agent.turn_context._maybe_title_session_at_turn_start',return_value=None):
        a=AIAgent(api_key='fixture',base_url='https://chatgpt.com/backend-api/codex',api_mode='codex_responses',model='gpt-6-astra',provider='openai-codex',session_db=db,session_id=sid,quiet_mode=True,skip_memory=True,skip_context_files=True,skip_background_review=True,enabled_toolsets=['continuity','terminal'],disabled_toolsets=['terminal'])
        a._disable_streaming=True; a._compression_feasibility_checked=True
        a._emit_status=lambda *args,**kwargs:None; a.commit_memory_session=lambda *args,**kwargs:None
        requests=[]
        a._interruptible_api_call=lambda req:(requests.append(deepcopy(req)) or completed_reply('No disabled tool granted.'))
        try:
            result=a.run_conversation('Answer without terminal.',conversation_history=[])
            assert result['completed']
            assert len(requests)==1
            request=requests[0]
            assert 'single-tool inventory was request-local' in request['instructions']
            assert 'terminal' not in {tool['name'] for tool in request['tools']}
            assert request.get('tool_choice') != {'type':'function','name':'terminal'}
        finally:
            a.close(); db.close()
