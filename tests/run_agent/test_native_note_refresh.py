"""A stale valid note must trigger maintenance, not repeat useless compaction."""
import json
import importlib.util
from hashlib import sha256
from pathlib import Path
import sys
import uuid
import threading
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


def _router_source(override=None, *, bundle=None):
    if override is not None:
        path = Path(override)
        assert path.is_file(), f'Required integration router missing: {path}'
        return path
    bundle = bundle if bundle is not None else Path(__file__).resolve().parents[1] / 'fixtures/native_maintenance_router'
    path = bundle / '__init__.py'
    manifest = bundle / 'provenance.json'
    assert path.is_file() and manifest.is_file(), 'Required bundled router or provenance missing'
    digest = json.loads(manifest.read_text(encoding='utf-8'))['sha256']
    assert sha256(path.read_bytes()).hexdigest() == digest, 'Bundled router digest mismatch'
    return path


def test_router_source_is_repository_relative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HERMES_TEST_ROUTER_SOURCE', str(tmp_path / 'absent'))
    path = _router_source()
    assert path == Path(__file__).resolve().parents[1] / 'fixtures/native_maintenance_router/__init__.py'


def test_router_source_explicit_selection_never_falls_back(tmp_path):
    source = tmp_path / 'external.py'
    source.write_bytes(_router_source().read_bytes())
    assert _router_source(str(source), bundle=tmp_path / 'missing-bundle') == source
    with pytest.raises(AssertionError, match='Required integration router missing'):
        _router_source(str(tmp_path / 'absent'))


@pytest.mark.parametrize('failure', ['missing_source', 'missing_manifest', 'corrupt_source'])
def test_router_source_bundle_fails_closed(tmp_path, failure):
    source = _router_source()
    data = source.read_bytes()
    if failure != 'missing_source':
        (tmp_path / '__init__.py').write_bytes(data + (b'\n# corrupt' if failure == 'corrupt_source' else b''))
    if failure != 'missing_manifest':
        (tmp_path / 'provenance.json').write_text(json.dumps({'sha256': sha256(data).hexdigest()}))
    with pytest.raises(AssertionError, match='missing|digest mismatch'):
        _router_source(bundle=tmp_path)


@pytest.fixture
def blocking_router(tmp_path, monkeypatch, pytestconfig):
    """Load the independently owned candidate, never a substitute policy."""
    from hermes_cli.plugins import PluginManager
    from tools.registry import registry
    from agent import native_note_refresh
    assert Path(native_note_refresh.__file__).resolve() == Path(__file__).resolve().parents[2]/'agent/native_note_refresh.py'

    path = _router_source(pytestconfig.getoption('--native-router-source'))
    name = f'native_refresh_router_{uuid.uuid4().hex}'
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    setattr(module, '_ROUTER', module.RouteStore(state_root=tmp_path/'router-state'))
    monkeypatch.setattr(registry, '_tools', dict(registry._tools))
    registry.register(name='fleet_route_task', toolset='fleet_task_router', schema=module.ROUTE_SCHEMA, handler=module.fleet_route_task)

    manager = PluginManager(scope_key=str(tmp_path))
    manager._discovered = True
    calls = []

    def require_route(tool_name, args, session_id='', turn_id='', **kwargs):
        calls.append((tool_name, session_id, turn_id))
        return module.pre_tool_call(tool_name, args, session_id=session_id, turn_id=turn_id, **kwargs)

    manager._hooks['pre_tool_call'] = [require_route]
    manager._hooks['post_tool_call'] = [module.post_tool_call]
    monkeypatch.setattr('hermes_cli.plugins.get_plugin_manager', lambda: manager)
    return NS(manager=manager, calls=calls, module=module)


@pytest.mark.parametrize('in_place',[False,True])
def test_stale_note_forces_ordinary_refresh_then_commits_and_continues(tmp_path,monkeypatch,in_place,blocking_router):
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    (tmp_path/'config.yaml').write_text('compression:\n  enabled: true\n  codex_responses_native: true\n  native_incremental_handoff: true\n  native_incremental_model: gpt-5.6-luna\n  native_incremental_compact_threshold: 128000\n')
    from hermes_state import SessionDB
    from run_agent import AIAgent
    db=SessionDB(db_path=tmp_path/'state.db');sid='stale-note-regression'
    db.create_session(sid,'cli',model='gpt-6-astra')
    with patch('agent.turn_context._maybe_title_session_at_turn_start',return_value=None):
        a=AIAgent(api_key='fixture',base_url='https://chatgpt.com/backend-api/codex',api_mode='codex_responses',model='gpt-6-astra',provider='openai-codex',session_db=db,session_id=sid,quiet_mode=True,skip_memory=True,skip_context_files=True,skip_background_review=True,enabled_toolsets=['continuity','terminal','fleet_task_router'])
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
        old_cursor = a._native_incremental_handoff_note['source_cursor']
        ordinary_tools = deepcopy(a.tools)

        calls=[]
        def request(req):
            calls.append(deepcopy(req))
            if len(calls)==1:
                assert req['model']=='gpt-6-astra','Must refresh before paying for a futile native request'
                assert req['tool_choice']==dict(type='function',name='continuity_note')
                assert [t['name'] for t in req['tools']]==['continuity_note']
                return reply(NS(type='function_call',id='fc-refresh',call_id='fresh-note',name='continuity_note',arguments=json.dumps(ARGS)))
            if len(calls)==2:
                assert a._native_incremental_handoff_note['source_cursor'] > old_cursor
                assert a._native_note_refresh_capability is None
                assert req['model']=='gpt-5.6-luna'
                assert 'Context maintenance: call continuity_note now' not in req['instructions']
                return reply(NS(type='compaction',id='cp',encrypted_content='fixture-checkpoint'),model='gpt-5.6-luna')
            assert req['model']=='gpt-6-astra'
            assert req.get('tool_choice')!=dict(type='function',name='continuity_note')
            assert 'terminal' in {tool['name'] for tool in req['tools']}
            assert 'Context maintenance: call continuity_note now' not in req['instructions']
            if len(calls)==3:
                assert blocking_router.module._ROUTER.active_count()==0
                return reply(NS(type='function_call',id='fc-route',call_id='route',name='tool_call',arguments=json.dumps({'name':'fleet_route_task','arguments':{'work_shape':'direct','consequence':'routine','reason_codes':['known_short_path'],'proof_target':'focused_test'}})),tokens=600)
            if len(calls)==4:
                assert '"ok":true' in json.dumps(req['input']).replace('\\"','"')
                return reply(NS(type='function_call',id='fc-terminal',call_id='ordinary',name='terminal',arguments=json.dumps({'command':'printf NATIVE_ORDINARY_OK','workdir':str(tmp_path)})),tokens=600)
            assert len(calls)==5
            assert 'NATIVE_ORDINARY_OK' in json.dumps(req['input'])
            return reply(NS(type='message',role='assistant',content=[NS(type='output_text',text='CURRENT_REQUEST_OK')]),tokens=600)
        a._interruptible_api_call=request
        try:
            result=a.run_conversation('Newest instruction: reply CURRENT_REQUEST_OK; no deployment.',conversation_history=source)
            assert result['completed'] and result['final_response']=='CURRENT_REQUEST_OK'
            assert len(calls)==5
            assert a.tools == ordinary_tools
            assert a.session_api_calls == 4  # refresh, route, terminal, reply; Luna is separate
            assert [c[0] for c in blocking_router.calls] == ['continuity_note','fleet_route_task','terminal']
            assert a._native_note_refresh_capability is None
            stored=db.get_messages_as_conversation(a.session_id)
            terminal_result=next(row for row in stored if row.get('role')=='tool' and (row.get('name') or row.get('tool_name'))=='terminal')
            terminal_result=json.loads(terminal_result['content'])
            assert terminal_result['exit_code']==0 and terminal_result['output']=='NATIVE_ORDINARY_OK'
            assert validate_persisted_native_compaction_history(stored)
            assert len(stored)<12
            assert 'structural_backoff' not in '\n'.join(warnings)
            assert 'Newest instruction' in json.dumps(calls[-1]['input'])
            if not in_place:
                assert db.get_messages_as_conversation(sid)[:len(before)]==before
        finally:a.close();db.close()


@pytest.mark.parametrize('failure_mode',['malformed_response','blocked_dispatch','guardrail_block','security_hook','interrupt','wrapped','mixed','duplicate','late_hook','flush_failed','readback_failed','unchanged'])
def test_failed_refresh_is_atomic_bounded_and_does_not_disable_ordinary_policy(
    tmp_path, monkeypatch, blocking_router, failure_mode,
):
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    (tmp_path/'config.yaml').write_text('compression:\n  enabled: true\n  codex_responses_native: true\n  native_incremental_handoff: true\n  native_incremental_compact_threshold: 128000\n')
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from agent.tool_executor import _run_sequential_tool_execution_middleware

    db=SessionDB(db_path=tmp_path/'state.db');sid=f'refresh-failure-{failure_mode}'
    db.create_session(sid,'cli',model='gpt-6-astra')
    with patch('agent.turn_context._maybe_title_session_at_turn_start',return_value=None):
        a=AIAgent(api_key='fixture',base_url='https://chatgpt.com/backend-api/codex',api_mode='codex_responses',model='gpt-6-astra',provider='openai-codex',session_db=db,session_id=sid,quiet_mode=True,skip_memory=True,skip_context_files=True,skip_background_review=True,enabled_toolsets=['continuity'])
        a.compression_in_place=True;a._disable_streaming=True;a._compression_feasibility_checked=True
        a._emit_status=lambda *args,**kwargs:None;a.commit_memory_session=lambda *args,**kwargs:None
        a.context_compressor.threshold_tokens=100000
        db.append_message(sid,'user','Initial objective')
        db.append_message(sid,'assistant','',tool_calls=[dict(id='old-note',type='function',function=dict(name='continuity_note',arguments=json.dumps(ARGS)))])
        source=db.get_messages_as_conversation(sid)
        old_result=record_native_incremental_note_from_tool_call(a,ARGS,source)
        db.append_message(sid,'tool',old_result,tool_call_id='old-note',tool_name='continuity_note')
        db.append_message(sid,'user','Keep new decisions. '*20000)
        db.append_message(sid,'assistant','Verified evidence. '*20000)
        source=db.get_messages_as_conversation(sid)
        restore_native_incremental_note(a,source)
        old_note=deepcopy(a._native_incremental_handoff_note)
        before=deepcopy(source)
        release=threading.Event();late_done=threading.Event();late_authorized=[]

        provider_calls=[];fallback_calls=[]
        def request(req):
            provider_calls.append(deepcopy(req))
            if failure_mode=='malformed_response':
                return reply(NS(type='message',role='assistant',content=[NS(type='output_text',text='not a note')]))
            if failure_mode=='interrupt':
                a._interrupt_requested=True
            if failure_mode=='wrapped':
                return reply(NS(type='function_call',id='fc-refresh',call_id='failed-note',name='tool_call',arguments=json.dumps({'name':'continuity_note','arguments':ARGS})))
            if failure_mode in ('mixed','duplicate'):
                return reply(NS(type='function_call',id='fc-refresh',call_id='failed-note',name='continuity_note',arguments=json.dumps(ARGS)),NS(type='function_call',id='fc-extra',call_id='extra',name='terminal' if failure_mode=='mixed' else 'continuity_note',arguments=json.dumps(ARGS)))
            return reply(NS(type='function_call',id='fc-refresh',call_id='failed-note',name='continuity_note',arguments=json.dumps(ARGS)))
        a._interruptible_api_call=request
        a._try_activate_fallback=lambda: fallback_calls.append(True) or True
        blocked=NS(result='blocked by test policy',blocked=True,dispatched=False)
        dispatch_patch=(
            patch('agent.tool_executor._run_sequential_tool_execution_middleware',return_value=blocked)
            if failure_mode=='blocked_dispatch' else patch('agent.tool_executor._run_sequential_tool_execution_middleware',wraps=_run_sequential_tool_execution_middleware)
        )
        if failure_mode=='guardrail_block':
            a._tool_guardrails.before_call=lambda *args,**kwargs:NS(allows_execution=False,message='guardrail denied maintenance')
        if failure_mode=='security_hook':
            blocking_router.manager._hooks['pre_tool_call'].append(lambda **kwargs:{'action':'block','message':'independent security denial'})
        if failure_mode=='late_hook':
            from agent.native_note_refresh import is_native_note_refresh_authorized
            def late_hook(tool_name, args, **kwargs):
                release.wait(5)
                late_authorized.append(is_native_note_refresh_authorized(kwargs.get('maintenance_context'),tool_name=tool_name,args=args,session_id=kwargs.get('session_id'),tool_call_id=kwargs.get('tool_call_id'),api_request_id=kwargs.get('api_request_id')))
                late_done.set()
            blocking_router.manager._hooks['pre_tool_call'].append(late_hook)
            monkeypatch.setattr('hermes_cli.plugins._resolve_hook_callback_timeout',lambda:0.02)
        if failure_mode=='flush_failed':
            original_flush=a._flush_messages_to_session_db
            monkeypatch.setattr(a,'_flush_messages_to_session_db',lambda messages,*args:False if any(m.get('tool_call_id')=='failed-note' for m in messages) else original_flush(messages,*args))
        if failure_mode=='readback_failed':
            monkeypatch.setattr('agent.native_note_refresh.restore_native_incremental_note',lambda *args,**kwargs:None)
        if failure_mode=='unchanged':
            monkeypatch.setattr('agent.native_note_refresh.record_native_incremental_note_from_tool_call',lambda *args,**kwargs:old_result)
        try:
            with dispatch_patch:
                result=a.run_conversation('Newest instruction remains pending.',conversation_history=source)
            assert result['failed'] and result['error']=='native_note_refresh_failed'
            assert len(provider_calls)==1 and fallback_calls==[]
            assert a.session_api_calls==1 and a.session_output_tokens==30
            assert a._native_incremental_handoff_note==old_note
            assert a._native_note_refresh_capability is None
            stored=db.get_messages_as_conversation(sid)
            assert stored[:len(before)]==before
            validate_persisted_native_compaction_history(stored)
            assert 'Newest instruction remains pending.' in json.dumps(stored)
            if failure_mode=='readback_failed':
                assert len([m for m in stored if m.get('tool_call_id')=='failed-note'])==1
            else:
                assert 'failed-note' not in json.dumps(stored)
            if failure_mode=='late_hook':
                release.set()
                assert late_done.wait(2) and late_authorized==[False]
                assert db.get_messages_as_conversation(sid)==stored
            a._interrupt_requested=False
            blocking_router.manager._hooks['pre_tool_call'] = blocking_router.manager._hooks['pre_tool_call'][:1]

            # The bypass is capability-bound: an ordinary continuity_note call
            # still reaches the real pre_tool_call policy and is blocked.
            ordinary=_run_sequential_tool_execution_middleware(
                a,function_name='continuity_note',function_args=ARGS,
                effective_task_id=sid,tool_call_id='ordinary-note',execute=lambda args:'unexpected',
            )
            assert ordinary.blocked
            assert 'Record fleet_route_task first' in ordinary.result
            assert blocking_router.calls[-1][0]=='continuity_note'
        finally:
            release.set()
            a.close();db.close()


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
