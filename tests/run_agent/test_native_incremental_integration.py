"""Parent-owned acceptance at the ordinary turn and restart boundary."""
import json
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
from agent.native_incremental_handoff import (bind_native_incremental_replay_projection,create_native_incremental_note,record_native_incremental_note,restore_native_incremental_note,record_native_incremental_note_from_tool_call,native_incremental_compact_context)
from agent.native_compaction import validate_persisted_native_compaction_history

ARGS=dict(objective='Verify native continuity',current_plan='Run isolated tests',next_action='Report results',blockers=['Not live'])


@pytest.mark.parametrize('in_place', [False, True])
def test_native_stream_progress_survives_host_idle_and_continues(tmp_path, monkeypatch, in_place):
    """Actual native dispatch/stream consumption outlives the host idle window."""
    import time
    from hermes_state import SessionDB
    from run_agent import AIAgent
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path/'config.yaml').write_text('compression:\n  enabled: true\n  codex_responses_native: true\n  native_incremental_handoff: true\n  native_incremental_model: gpt-5.6-luna\n  native_incremental_compact_threshold: 1000\n  context_timeout_seconds: 0.4\n  context_total_ceiling_seconds: 5\n')
    db=SessionDB(db_path=tmp_path/'state.db')
    sid='native-progress-regression'
    db.create_session(sid, 'cli', model='gpt-6-astra')
    a=AIAgent(api_key='test-key', base_url='https://chatgpt.com/backend-api/codex', api_mode='codex_responses', model='gpt-6-astra', provider='openai-codex', session_db=db, session_id=sid, quiet_mode=True, skip_memory=True, skip_context_files=True, skip_background_review=True, enabled_toolsets=[])
    calls=[];warnings=[]
    class Stream:
        def __init__(self, model): self.model=model
        def __iter__(self):
            yield {'type':'response.created','response':{'id':'fixture'}}
            if self.model=='gpt-5.6-luna':
                for _ in range(12):
                    time.sleep(.06)
                    yield {'type':'keepalive'}
                item=NS(type='compaction',id='cp-progress',encrypted_content='progress-checkpoint')
            else:
                item=message('CONTINUATION_OK')
                item.id='answer'
            yield NS(type='response.output_item.done', output_index=0, item=item)
            yield NS(type='response.completed', response=response(item, model=self.model))
        def close(self): pass
    def create(**request):
        calls.append(deepcopy(request))
        return Stream(request['model'])
    client=NS(responses=NS(create=create))
    a._create_request_openai_client=lambda *args,**kwargs: client
    a._close_request_openai_client=lambda *args,**kwargs: None
    a._abort_request_openai_client=lambda *args,**kwargs: None
    a._disable_streaming=True
    a.compression_in_place=in_place
    a._compression_feasibility_checked=True
    a._emit_warning=lambda value,*args,**kwargs: warnings.append(value)
    a._emit_status=lambda *args,**kwargs: None
    a.commit_memory_session=lambda *args,**kwargs: None
    a.context_compressor.threshold_tokens=100
    a.context_compressor.should_compress_preflight=lambda _:True
    a.context_compressor.should_compress=lambda _:False
    db.append_message(sid, 'user', 'old objective '*2000)
    db.append_message(sid, 'assistant', 'old evidence '*2000)
    db.append_message(sid, 'assistant', '', tool_calls=[{'id':'note-1','type':'function','function':{'name':'continuity_note','arguments':json.dumps(ARGS)}}])
    # Generate the note against canonical source rows, as the real executor
    # does. The output writer normalizes tool-call fields before this point.
    source=db.get_messages_as_conversation(sid, repair_alternation=False)
    note_result=record_native_incremental_note_from_tool_call(a, ARGS, source)
    db.append_message(sid, 'tool', note_result, tool_call_id='note-1', tool_name='continuity_note')
    source=db.get_messages_as_conversation(sid, repair_alternation=False)
    assert restore_native_incremental_note(a, source) is not None
    try:
        with patch('hermes_cli.plugins.invoke_hook',return_value=[]), patch('hermes_cli.lifecycle.invoke_hook',return_value=[]), patch('agent.turn_context._maybe_title_session_at_turn_start',return_value=None):
            result=a.run_conversation('Reply CONTINUATION_OK; no other action.', conversation_history=source)
        assert not a._last_compression_timed_out, warnings
        assert result['completed'] and result['final_response']=='CONTINUATION_OK'
        assert [call['model'] for call in calls]==['gpt-5.6-luna','gpt-6-astra']
        assert not any('timed out' in warning for warning in warnings)
        assert (a.session_id == sid) is in_place
        validate_persisted_native_compaction_history(result['messages'])
        # Validate canonical stored rows, never an alternation-repaired view.
        persisted=db.get_messages_as_conversation(a.session_id, repair_alternation=False)
        validate_persisted_native_compaction_history(persisted)
        assert any('CONTINUATION_OK' in str(row.get('content')) for row in persisted)
    finally:
        a.close();db.close()


def response(*items,model='gpt-6-astra'):
    return NS(output=list(items),usage=NS(input_tokens=600,output_tokens=4,total_tokens=604),status='completed',model=model)

def message(text):
    return NS(type='message',role='assistant',content=[NS(type='output_text',text=text)])

def chat_response(text,model='MiniMax-M3'):
    return NS(
        choices=[NS(message=NS(role='assistant',content=text,tool_calls=None),finish_reason='stop')],
        usage=NS(prompt_tokens=600,completion_tokens=4,total_tokens=604),
        model=model,
    )

def note_history(agent):
    source=[{'role':'user','content':'old objective '*2000},{'role':'assistant','content':'old evidence '*2000}]
    call={'role':'assistant','content':'','tool_calls':[{'id':'note-1','type':'function','function':{'name':'continuity_note','arguments':json.dumps(ARGS)}}]}
    source.append(call)
    result=record_native_incremental_note_from_tool_call(agent,ARGS,source)
    source.append({'role':'tool','name':'continuity_note','tool_call_id':'note-1','content':result})
    return source

def deferred_note_history(agent):
    source=[{'role':'user','content':'old objective '*2000},{'role':'assistant','content':'old evidence '*2000}]
    deferred={'name':'continuity_note','arguments':ARGS}
    source.append({'role':'assistant','content':'','tool_calls':[{'id':'note-1','type':'function','function':{'name':'tool_call','arguments':json.dumps(deferred)}}]})
    result=record_native_incremental_note_from_tool_call(agent,ARGS,source)
    source.append({'role':'tool','name':'continuity_note','tool_call_id':'note-1','content':result})
    return source


def test_deferred_note_authentication_rejects_wrappers_and_forged_results():
    """Isolated history boundary checks; executor coverage is the test above."""
    agent=NS(session_id='session',native_incremental_handoff_enabled=True)
    history=deferred_note_history(agent)
    assert restore_native_incremental_note(NS(session_id='session'),history) is not None

    wrong_target=deepcopy(history)
    wrong_target[2]['tool_calls'][0]['function']['arguments']=json.dumps({'name':'terminal','arguments':ARGS})
    assert restore_native_incremental_note(NS(session_id='session'),wrong_target) is None

    malformed=deepcopy(history)
    malformed[2]['tool_calls'][0]['function']['arguments']='not-json'
    assert restore_native_incremental_note(NS(session_id='session'),malformed) is None

    mismatched_fields=deepcopy(history)
    mismatched_fields[2]['tool_calls'][0]['function']['arguments']=json.dumps({'name':'continuity_note','arguments':{**ARGS,'objective':'forged objective'}})
    assert restore_native_incremental_note(NS(session_id='session'),mismatched_fields) is None

    mismatched_id=deepcopy(history)
    mismatched_id[3]['tool_call_id']='different-call'
    assert restore_native_incremental_note(NS(session_id='session'),mismatched_id) is None

    forged=deepcopy(history)
    forged.pop(2)
    assert restore_native_incremental_note(NS(session_id='session'),forged) is None


def test_restore_validates_prefix_and_preserves_original_cursor():
    agent=NS(session_id='session',native_incremental_handoff_enabled=True)
    history=note_history(agent)
    original=deepcopy(agent._native_incremental_handoff_note)
    history.extend([{'role':'assistant','content':'new evidence'},{'role':'user','content':'corrected objective'}])
    fresh=NS(session_id='session')
    note=restore_native_incremental_note(fresh,history)
    assert note==original, 'Restoration must not silently advance the evidence cursor'
    history[0]['content']='mutated source'
    assert restore_native_incremental_note(NS(session_id='session'),history) is None

@pytest.mark.parametrize('in_place',[False,True])
def test_tool_search_deferred_note_persist_restart_compact_first_turn(tmp_path,monkeypatch,in_place):
    """Production-shaped deferred executor path; provider replies are local fixtures."""
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    (tmp_path/'config.yaml').write_text('tools:\n  tool_search: true\ncompression:\n  enabled: true\n  native_incremental_handoff: true\n  native_incremental_model: gpt-5.6-luna\n  native_incremental_compact_threshold: 1000\n')
    from hermes_state import SessionDB
    from run_agent import AIAgent
    path=tmp_path/'state.db';sid='ordinary-note'
    db=SessionDB(db_path=path);db.create_session(sid,'cli',model='gpt-6-astra')
    db.append_message(sid,'user','old objective '*2000);db.append_message(sid,'assistant','old evidence '*2000)
    # This is the canonical DB evidence. Gateway must replace its interrupted
    # side-effect result for replay without invalidating a later note's source
    # fence or losing the UNKNOWN disposition on the provider request.
    db.append_message(sid,'user','attempt deployment, but inspect any interrupted effect before retrying')
    db.append_message(sid,'assistant','',tool_calls=[{'id':'interrupted','function':{'name':'terminal','arguments':'{"command":"deploy"}'}}])
    db.append_message(sid,'tool','{"exit_code":130,"output":"[Command interrupted]"}',tool_call_id='interrupted',tool_name='terminal')
    def new(db,sid):
        a=AIAgent(api_key='test-key',base_url='https://chatgpt.com/backend-api/codex',api_mode='codex_responses',model='gpt-6-astra',provider='openai-codex',session_db=db,session_id=sid,quiet_mode=True,skip_memory=True,skip_context_files=True,skip_background_review=True,enabled_toolsets=['continuity'])
        assert a.native_incremental_handoff_enabled
        visible_names={t['function']['name'] for t in a.tools}
        assert 'tool_call' in visible_names and 'continuity_note' not in visible_names, 'Opt-in continuity tool must use production deferred dispatch'
        a.compression_in_place=in_place;a._compression_feasibility_checked=True;a._disable_streaming=True
        a._emit_status=lambda *a,**k:None;a._emit_warning=lambda *a,**k:None
        a.context_compressor.should_compress_preflight=lambda _:False
        a.context_compressor.should_compress=lambda _:False
        a.commit_memory_session=lambda *a,**k:None
        return a
    with patch('hermes_cli.plugins.invoke_hook',return_value=[]),patch('hermes_cli.lifecycle.invoke_hook',return_value=[]),patch('agent.turn_context._maybe_title_session_at_turn_start',return_value=None):
        a=new(db,sid)
        from gateway.run import _build_gateway_agent_history
        canonical,_=db.get_resume_conversations(sid)
        replay,_=_build_gateway_agent_history(canonical)
        assert next(row for row in replay if row.get('tool_call_id')=='interrupted')['effect_disposition']=='unknown'
        assert bind_native_incremental_replay_projection(a,source_messages=canonical,replay_messages=replay)
        deferred={'name':'continuity_note','arguments':ARGS}
        replies=[response(NS(type='function_call',id='fc-note',call_id='note-1',name='tool_call',arguments=json.dumps(deferred))),response(message('NOTE_SAVED'))]
        calls=[]
        a._interruptible_api_call=lambda req:(calls.append(deepcopy(req)) or replies.pop(0))
        r=a.run_conversation('Record the current continuity note.',conversation_history=replay)
        assert r['completed'] and r['final_response']=='NOTE_SAVED'
        persisted,_=db.get_resume_conversations(a.session_id)
        wrapped=next(m for m in persisted if m.get('role')=='assistant' and m.get('tool_calls') and m['tool_calls'][0].get('function',{}).get('name')=='tool_call')
        note_result=next(m for m in persisted if m.get('role')=='tool' and (m.get('name') or m.get('tool_name'))=='continuity_note')
        assert json.loads(wrapped['tool_calls'][0]['function']['arguments'])==deferred
        assert note_result['tool_call_id']==wrapped['tool_calls'][0]['id']
        assert 'authenticated_by' in note_result['content']
        a.close();db.close()
        db=SessionDB(db_path=path);a=new(db,sid)
        canonical,_=db.get_resume_conversations(sid)
        replay,_=_build_gateway_agent_history(canonical)
        assert next(row for row in replay if row.get('tool_call_id')=='interrupted')['effect_disposition']=='unknown'
        assert bind_native_incremental_replay_projection(a,source_messages=canonical,replay_messages=replay)
        if not in_place:
            # Compression's lease may adopt a longer canonical DB history.
            # Its prior gateway projection must not invalidate the genuine note.
            db.append_message(sid,'user','Concurrent recorded evidence; preserve the next live instruction.')
            db.append_message(sid,'assistant','Concurrent evidence recorded.')
        a.context_compressor.threshold_tokens=100
        a.context_compressor.should_compress_preflight=lambda _:True
        fresh_deferred={'name':'continuity_note','arguments':{**ARGS,'objective':'Fresh projected note'}}
        replies=[
            response(NS(type='compaction',id='cp',encrypted_content='test-checkpoint'),model='gpt-5.6-luna'),
            response(NS(type='function_call',id='fc-fresh',call_id='new-note',name='tool_call',arguments=json.dumps(fresh_deferred))),
            response(message('NEW_INSTRUCTION_OK')),
        ]
        calls=[]
        a._interruptible_api_call=lambda req:(calls.append(deepcopy(req)) or replies.pop(0))
        r=a.run_conversation('Newest correction: reply NEW_INSTRUCTION_OK; never deploy.',conversation_history=replay)
        assert r['completed'] and r['final_response']=='NEW_INSTRUCTION_OK'
        assert [c['model'] for c in calls]==['gpt-5.6-luna','gpt-6-astra','gpt-6-astra']
        assert 'Perform native context compaction only.' not in calls[0]['instructions']
        assert 'reply exactly NATIVE_COMPACTION_COMPLETE' not in calls[0]['instructions']
        for ordinary in calls[1:]:
            assert 'The previous compaction-only operation is finished.' in ordinary['instructions']
            assert any(i.get('type')=='compaction' for i in ordinary['input'])
        assert 'max_output_tokens' not in calls[0]
        assert calls[0]['reasoning']=={'effort':'low'}
        normal=calls[1]['input']
        assert normal[0]['type']=='compaction'
        assert normal[1]['role']=='developer' and 'historical' in normal[1]['content']
        assert any('Newest correction' in str(x.get('content')) for x in normal)
        validate_persisted_native_compaction_history(r['messages'])
        final_sid=a.session_id;a.close();db.close()
        db=SessionDB(db_path=path)
        reloaded,_=db.get_resume_conversations(final_sid)
        validate_persisted_native_compaction_history(reloaded)
        assert reloaded[0]['codex_reasoning_items'][0]['encrypted_content']=='test-checkpoint'
        # The note was minted by the ordinary tool executor AFTER compaction,
        # while the same run held the rebound protected source/projection pair.
        canonical,_=db.get_resume_conversations(final_sid)
        replay,_=_build_gateway_agent_history(canonical)
        fresh=NS(session_id=final_sid,native_incremental_handoff_enabled=True)
        assert bind_native_incremental_replay_projection(fresh,source_messages=canonical,replay_messages=replay)
        assert restore_native_incremental_note(fresh,replay)['objective']=='Fresh projected note'
        db.close()


def test_missing_note_recovers_midturn_without_premature_warning(tmp_path, monkeypatch):
    """Real loop: missing note -> note tool -> checkpoint -> normal reply.

    Only provider responses are fixtures. The warning enters the same public
    warning method as the live pressure gate, before ordinary tool execution
    supplies the missing note and compression persists a rotated checkpoint.
    """
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(
        'tools:\n  tool_search: true\ncompression:\n  enabled: true\n'
        '  native_incremental_handoff: true\n'
        '  native_incremental_model: gpt-5.6-luna\n'
        '  native_incremental_compact_threshold: 1000\n'
    )
    from hermes_state import SessionDB
    from run_agent import AIAgent
    db = SessionDB(db_path=tmp_path / 'state.db')
    sid = 'missing-note-recovery'
    db.create_session(sid, 'cli', model='gpt-6-astra')
    db.append_message(sid, 'user', 'Retain the task objective. ' * 2000)
    db.append_message(sid, 'assistant', 'Preserve the verified evidence. ' * 2000)
    with patch('hermes_cli.plugins.invoke_hook', return_value=[]), patch('hermes_cli.lifecycle.invoke_hook', return_value=[]), patch('agent.turn_context._maybe_title_session_at_turn_start', return_value=None):
        a = AIAgent(
            api_key='test-key', base_url='https://chatgpt.com/backend-api/codex',
            api_mode='codex_responses', model='gpt-6-astra', provider='openai-codex',
            session_db=db, session_id=sid, quiet_mode=True, skip_memory=True,
            skip_context_files=True, skip_background_review=True, enabled_toolsets=['continuity'],
        )
        a.compression_in_place = False
        a._compression_feasibility_checked = True
        a._disable_streaming = True
        a.commit_memory_session = lambda *args, **kwargs: None
        a._emit_status = lambda *args, **kwargs: None
        warnings = []
        a._emit_warning = warnings.append
        a.context_compressor.context_length = 400_000
        a.context_compressor.threshold_tokens = 1000
        replies = [
            response(NS(type='function_call', id='fc-note', call_id='note-1', name='tool_call', arguments=json.dumps({'name': 'continuity_note', 'arguments': ARGS}))),
            response(NS(type='compaction', id='cp', encrypted_content='test-checkpoint'), model='gpt-5.6-luna'),
            response(message('RECOVERED_WITHOUT_RESET')),
        ]
        replies[0].usage = NS(input_tokens=50_000, output_tokens=4, total_tokens=50_004)
        calls = []

        def provider(request):
            calls.append(deepcopy(request))
            if len(calls) == 1:
                a._warn_context_overflow_blocked('structural_backoff:283', 287_761, 1000)
            assert warnings == [], 'A pending recovery must not emit the alarming warning'
            return replies.pop(0)

        a._interruptible_api_call = provider
        history, _ = db.get_resume_conversations(sid)
        try:
            result = a.run_conversation('Keep working; record the continuity note first.', conversation_history=history)
            assert result['completed'] and result['final_response'] == 'RECOVERED_WITHOUT_RESET'
            assert [call['model'] for call in calls] == ['gpt-6-astra', 'gpt-5.6-luna', 'gpt-6-astra']
            assert a.session_id != sid
            persisted, _ = db.get_resume_conversations(a.session_id)
            validate_persisted_native_compaction_history(persisted)
            assert persisted[0]['codex_reasoning_items'][0]['encrypted_content'] == 'test-checkpoint'
            assert warnings == []
        finally:
            a.close()
            db.close()


def test_real_stream_shape_keeps_only_last_checkpoint_and_its_suffix():
    from agent.native_incremental_handoff import _latest_checkpoint_and_suffix
    cp1={'type':'compaction','id':'cp1','encrypted_content':'early'}
    cp2={'type':'compaction','id':'cp2','encrypted_content':'last'}
    reasoning={'type':'reasoning','id':'r','encrypted_content':'opaque-thought','summary':[]}
    cp,suffix=_latest_checkpoint_and_suffix(response(cp1, reasoning, cp2, reasoning, message('suffix')))
    assert cp['encrypted_content']=='last'
    # Duplicate stream delivery is ignored once, not silently reinterpreted.
    assert len(suffix)==1 and suffix[0]['content']=='suffix'
    cp,suffix=_latest_checkpoint_and_suffix(response(cp1, cp2, reasoning, message('suffix'), deepcopy(cp2)))
    assert cp['encrypted_content']=='last'
    assert suffix[0]['codex_reasoning_items'][0]['encrypted_content']=='opaque-thought'
    assert suffix[1]['content']=='suffix'
    with pytest.raises(ValueError,match='conflicting duplicate'):
        _latest_checkpoint_and_suffix(response(cp1,{**cp1,'encrypted_content':'changed'}))


def test_empty_fallback_projects_valid_checkpoint_tail_for_minimax_chat(monkeypatch):
    """MiniMax chat fallback receives a disposable generic projection, not Responses-only state."""
    from tests.run_agent.test_native_incremental_handoff import _agent,_source,_note,_response
    from run_agent import AIAgent

    producer,_=_agent([_response({'type':'compaction','encrypted_content':'cipher'})])
    source=_source()
    source[3]['tool_calls'][0]={'id':'call_1','type':'function','function':{'name':None,'arguments':'{}'}}
    _note(producer,source)
    protected=native_incremental_compact_context(producer,source)
    persisted_before=deepcopy(protected)

    a=AIAgent(api_key='test-key',base_url='https://chatgpt.com/backend-api/codex',api_mode='codex_responses',model='gpt-6-astra',provider='openai-codex',quiet_mode=True,skip_memory=True,skip_context_files=True,skip_background_review=True)
    a._disable_streaming=True
    a.native_incremental_handoff_enabled=True
    a.native_incremental_handoff_model='gpt-5.6-luna'
    a.context_compressor.should_compress_preflight=lambda _:False
    a.context_compressor.should_compress=lambda _:False
    a._fallback_chain=[{'provider':'minimax','model':'MiniMax-M3','base_url':'https://api.minimax.io/v1','api_key':'test-key','api_mode':'chat_completions'}]
    requests=[]
    empty=response(NS(type='message',role='assistant',content=[]))
    replies=[empty,empty,empty,chat_response('FALLBACK_OK')]
    a._interruptible_api_call=lambda request:(requests.append(deepcopy(request)) or replies.pop(0))

    monkeypatch.setattr('agent.auxiliary_client.resolve_provider_client',lambda *args,**kwargs:(a.client,'MiniMax-M3'))
    monkeypatch.setattr('agent.conversation_loop.jittered_backoff',lambda *a,**k:0.0)
    monkeypatch.setattr('agent.conversation_loop._empty_guard.empty_retry_budget',lambda *a,**k:1)
    try:
        result=a.run_conversation('Return FALLBACK_OK.',conversation_history=protected)
    finally:
        a.close()

    assert result['completed'] and result['final_response']=='FALLBACK_OK'
    assert a.provider=='minimax' and a.model=='MiniMax-M3' and a.api_mode=='chat_completions'
    assert len(requests)==4 and 'input' in requests[0] and 'messages' in requests[-1]
    chat_messages=requests[-1]['messages']
    assert all('codex_reasoning_items' not in row for row in chat_messages)
    assert all(call.get('function',{}).get('name') for row in chat_messages for call in row.get('tool_calls') or [])
    assert any('historical' in str(row.get('content')) for row in chat_messages)
    assert any('Return FALLBACK_OK.' in str(row.get('content')) for row in chat_messages)
    assert protected==persisted_before
    validate_persisted_native_compaction_history(result['messages'])


def test_empty_fallback_keeps_valid_checkpoint_tail_immutable(monkeypatch):
    """Responses fallback retains the validated checkpoint for adapter-owned conversion."""
    from tests.run_agent.test_native_incremental_handoff import _agent,_source,_note,_response
    from run_agent import AIAgent

    producer,_=_agent([_response({'type':'compaction','encrypted_content':'cipher'})])
    source=_source();_note(producer,source)
    protected=native_incremental_compact_context(producer,source)
    persisted_before=deepcopy(protected)

    a=AIAgent(api_key='test-key',base_url='https://chatgpt.com/backend-api/codex',api_mode='codex_responses',model='gpt-6-astra',provider='openai-codex',quiet_mode=True,skip_memory=True,skip_context_files=True,skip_background_review=True)
    a._disable_streaming=True
    a.native_incremental_handoff_enabled=True
    a.native_incremental_handoff_model='gpt-5.6-luna'
    a.context_compressor.should_compress_preflight=lambda _:False
    a.context_compressor.should_compress=lambda _:False
    a._fallback_chain=[{'provider':'openai-codex','model':'gpt-6-fallback','base_url':'https://chatgpt.com/backend-api/codex','api_key':'test-key','api_mode':'codex_responses'}]
    requests=[]
    empty=response(NS(type='message',role='assistant',content=[]))
    replies=[empty,empty,empty,response(message('FALLBACK_OK'))]
    a._interruptible_api_call=lambda request:(requests.append(deepcopy(request)) or replies.pop(0))

    monkeypatch.setattr('agent.auxiliary_client.resolve_provider_client',lambda *args,**kwargs:(a.client,'gpt-6-fallback'))
    monkeypatch.setattr('agent.conversation_loop.jittered_backoff',lambda *a,**k:0.0)
    monkeypatch.setattr('agent.conversation_loop._empty_guard.empty_retry_budget',lambda *a,**k:1)
    try:
        result=a.run_conversation('Return FALLBACK_OK.',conversation_history=protected)
    finally:
        a.close()

    assert result['completed'] and result['final_response']=='FALLBACK_OK'
    assert a.api_mode=='codex_responses' and len(requests)==4
    assert any(item.get('type')=='compaction' for item in requests[-1]['input'])
    assert protected==persisted_before
    validate_persisted_native_compaction_history(result['messages'])


def test_newer_note_supersedes_checkpoint_without_renewing_old_evidence():
    from agent.native_incremental_handoff import native_incremental_note_from_history
    from tests.run_agent.test_native_incremental_handoff import _agent,_source,_note,_response
    agent,_=_agent([_response({'type':'compaction','encrypted_content':'cipher'})])
    history=_source();_note(agent,history)
    compacted=native_incremental_compact_context(agent,history)
    old=restore_native_incremental_note(agent,compacted)
    assert old['source_cursor']==2
    args={**ARGS,'objective':'New corrected objective'}
    compacted.append({'role':'assistant','content':'','tool_calls':[{'id':'new-note','function':{'name':'continuity_note','arguments':json.dumps(args)}}]})
    result=record_native_incremental_note_from_tool_call(agent,args,compacted)
    compacted.append({'role':'tool','name':'continuity_note','tool_call_id':'new-note','content':result})
    fresh=NS(session_id=agent.session_id)
    note=restore_native_incremental_note(fresh,compacted)
    assert note['objective']=='New corrected objective'
    assert note['source_cursor']==len(compacted)-1


def test_unbound_tool_and_unpaired_payload_cannot_stage_note():
    from tools.continuity_note_tool import _unbound_handler
    assert 'error' in json.loads(_unbound_handler(ARGS))
    agent=NS(session_id='session',native_incremental_handoff_enabled=True)
    history=note_history(agent)
    history[2]['tool_calls'][0]['function']['name']='terminal'
    assert restore_native_incremental_note(NS(session_id='session'),history) is None


def test_native_operation_default_is_separate_from_main_trigger():
    from agent.native_incremental_handoff import NATIVE_INCREMENTAL_COMPACT_THRESHOLD
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    assert NATIVE_INCREMENTAL_COMPACT_THRESHOLD==32000
    assert DEFAULT_CONFIG['compression']['native_incremental_compact_threshold']==32000


@pytest.mark.parametrize('item',[
    {'type':'message','role':'assistant','content':[{'type':'future_content','payload':'opaque'}]},
    {'type':'message','role':'assistant','content':None},
    {'type':'message','role':'user','content':[{'type':'output_text','text':'unexpected'}]},
])
def test_unknown_message_suffix_retains_source(item):
    from tests.run_agent.test_native_incremental_handoff import _agent,_source,_note,_response
    a,_=_agent([_response({'type':'compaction','encrypted_content':'cipher'},item)])
    history=_source();before=deepcopy(history);_note(a,history)
    with pytest.raises(ValueError,match='message suffix'):
        native_incremental_compact_context(a,history)
    assert history==before


@pytest.mark.parametrize('exc',[TimeoutError('timeout'),InterruptedError('cancelled'),KeyboardInterrupt()])
def test_native_cancellation_keeps_source_and_cannot_summarize(exc):
    from tests.run_agent.test_native_incremental_handoff import _agent,_source,_note
    a,_=_agent([]);history=_source();before=deepcopy(history);_note(a,history);calls=[]
    def fail(req):
        calls.append(req['model']);raise exc
    a._interruptible_api_call=fail
    with pytest.raises(type(exc)):
        native_incremental_compact_context(a,history)
    assert history==before and calls==['gpt-5.6-luna']
    assert a.context_compressor._last_compress_aborted


@pytest.mark.parametrize('model',['gpt-5.4-mini','gpt-6-astra',''])
def test_invalid_pinned_model_does_not_fallback(model):
    from tests.run_agent.test_native_incremental_handoff import _agent,_source,_note
    a,calls=_agent([]);history=_source();_note(a,history);a.native_incremental_handoff_model=model
    assert native_incremental_compact_context(a,history) is history
    assert calls==[] and a._last_native_incremental_compaction['disposition']=='invalid_route'


@pytest.mark.parametrize('platform',['cli','telegram'])
def test_public_cli_enables_continuity_and_platform_exposes_it(tmp_path,platform):
    import os,subprocess,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    cfg=tmp_path/'config.yaml'
    cfg.write_text('tools:\n  tool_search: false\ncompression:\n  native_incremental_handoff: true\n  native_incremental_model: gpt-5.6-luna\n')
    env=os.environ.copy();env['HERMES_HOME']=str(tmp_path);env['HERMES_SKIP_ENV_LOAD']='1'
    cmd=[sys.executable,'-m','hermes_cli.main','tools','enable','continuity','--platform',platform]
    result=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0 and 'Unknown toolset' not in result.stdout,result.stdout
    code='''import json
from hermes_cli.config import load_config
from hermes_cli.tools_config import _get_platform_tools
from toolsets import resolve_toolset
from run_agent import AIAgent
cfg=load_config();sets=_get_platform_tools(cfg,PLATFORM)
assert 'continuity' in sets
assert 'continuity_note' in resolve_toolset('continuity')
a=AIAgent(api_key='test-key',base_url='https://chatgpt.com/backend-api/codex',api_mode='codex_responses',model='gpt-6-astra',provider='openai-codex',enabled_toolsets=sorted(sets),quiet_mode=True,skip_memory=True,skip_context_files=True,skip_background_review=True)
assert any(t['function']['name']=='continuity_note' for t in a.tools)
a.close()
'''.replace('PLATFORM',repr(platform))
    check=subprocess.run([sys.executable,'-c',code],cwd=root,env=env,capture_output=True,text=True,timeout=30)
    assert check.returncode==0,check.stderr[-1200:]
    # Explicit toolset selection must not bypass the feature's off gate.
    import yaml
    parsed=yaml.safe_load(cfg.read_text());parsed['compression']['native_incremental_handoff']=False;cfg.write_text(yaml.safe_dump(parsed))
    code=code.replace("assert any(t['function']['name']=='continuity_note' for t in a.tools)","assert not any(t['function']['name']=='continuity_note' for t in a.tools)")
    check=subprocess.run([sys.executable,'-c',code],cwd=root,env=env,capture_output=True,text=True,timeout=30)
    assert check.returncode==0,check.stderr[-1200:]
