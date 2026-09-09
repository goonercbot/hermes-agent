"""Native /compress must initialize continuity like an ordinary gateway turn."""
import json
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from tests.gateway.test_compress_command import _make_event, _make_runner


@pytest.mark.asyncio
@pytest.mark.parametrize('tampered', [False, True])
async def test_manual_native_restores_note_and_preserves_history(tmp_path, monkeypatch, tampered):
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from agent.native_incremental_handoff import record_native_incremental_note_from_tool_call
    from agent.native_compaction import validate_persisted_native_compaction_history

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(
        'compression:\n  enabled: true\n  in_place: true\n'
        '  codex_responses_native: true\n  native_incremental_handoff: true\n'
        '  native_incremental_model: gpt-5.6-luna\n'
        '  native_incremental_compact_threshold: 1000\n'
    )
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('sess-1', 'telegram', model='gpt-6-astra')
    agent = AIAgent(api_key='fixture', base_url='https://chatgpt.com/backend-api/codex',
                    api_mode='codex_responses', model='gpt-6-astra', provider='openai-codex',
                    session_db=db, session_id='sess-1', quiet_mode=True, skip_memory=True,
                    skip_context_files=True, skip_background_review=True, enabled_toolsets=[])
    agent.compression_in_place = True
    agent._compression_feasibility_checked = True
    agent._emit_status = lambda *args, **kwargs: None
    agent._cached_system_prompt = 'Follow the latest user request.'
    args = dict(objective='Keep weekly and all-time usage', current_plan='Preserve compact layout',
                next_action='Verify compression', blockers=[])
    db.append_message('sess-1', 'user', 'Historical request. ' * 3000)
    db.append_message('sess-1', 'assistant', 'Historical evidence. ' * 3000)
    db.append_message('sess-1', 'assistant', '', tool_calls=[
        {'id': 'note1', 'type': 'function', 'function':
         {'name': 'continuity_note', 'arguments': json.dumps(args)}}])
    source = db.get_messages_as_conversation('sess-1', repair_alternation=False)
    note = record_native_incremental_note_from_tool_call(agent, args, source)
    db.append_message('sess-1', 'tool', note, tool_call_id='note1', tool_name='continuity_note')
    db.append_message('sess-1', 'user', 'Keep the compact report format.')
    history = db.get_messages_as_conversation('sess-1', repair_alternation=False)
    if tampered:
        history[0]['content'] += 'altered after the note'
    before = deepcopy(history)
    # A freshly constructed manual helper has no staged in-memory note.
    agent._native_incremental_handoff_note = None
    calls = []

    def provider(request):
        calls.append(deepcopy(request))
        assert request['model'] == 'gpt-5.6-luna'
        return NS(status='completed', output=[NS(type='compaction', id='fixture-checkpoint',
                  encrypted_content='fixture-opaque-checkpoint')],
                  usage=NS(input_tokens=20000, output_tokens=20))

    agent._interruptible_api_call = provider
    runner = _make_runner(history)
    try:
        with patch('gateway.run._resolve_runtime_agent_kwargs', return_value={'api_key':'fixture'}), \
             patch('gateway.run._resolve_gateway_model', return_value='gpt-6-astra'), \
             patch('run_agent.AIAgent', return_value=agent), \
             patch('hermes_cli.plugins.invoke_hook', return_value=[]), \
             patch('hermes_cli.lifecycle.invoke_hook', return_value=[]):
            result = await runner._handle_compress_command(_make_event())
        assert history == before
        if tampered:
            assert not calls
            assert 'handoff note not ready' in result
            assert 'Compression blocked' in result
            assert 'No changes from compression' not in result
            assert len(db.get_messages_as_conversation('sess-1', repair_alternation=False)) == len(before)
        else:
            assert len(calls) == 1, result
            assert 'Compressed:' in result, result
            persisted = db.get_messages_as_conversation('sess-1', repair_alternation=False)
            assert validate_persisted_native_compaction_history(persisted)
            assert persisted[-1]['content'] == 'Keep the compact report format.'
            assert len(json.dumps(persisted)) < len(json.dumps(before))
            runner.session_store.rewrite_transcript.assert_not_called()
            # Reopen through ordinary turn setup: manual commit alone is not
            # enough if the first normal request damages the protected tail.
            fresh = AIAgent(api_key='fixture', base_url='https://chatgpt.com/backend-api/codex',
                            api_mode='codex_responses', model='gpt-6-astra', provider='openai-codex',
                            session_db=db, session_id='sess-1', quiet_mode=True, skip_memory=True,
                            skip_context_files=True, skip_background_review=True, enabled_toolsets=[])
            fresh._fallback_chain = []
            fresh.context_compressor.should_compress_preflight = lambda _: False
            fresh.context_compressor.should_compress = lambda _: False
            fresh._interruptible_api_call = lambda request: NS(
                status='completed', model='gpt-6-astra',
                output=[NS(type='message', role='assistant',
                           content=[NS(type='output_text', text='CONTINUATION_OK')])],
                usage=NS(input_tokens=600, output_tokens=4, total_tokens=604))
            try:
                with patch('agent.turn_context._maybe_title_session_at_turn_start'), \
                     patch('hermes_cli.plugins.invoke_hook', return_value=[]), \
                     patch('hermes_cli.lifecycle.invoke_hook', return_value=[]):
                    answer = fresh.run_conversation('Reply CONTINUATION_OK.', conversation_history=persisted)
                assert answer['completed'] and answer['final_response'] == 'CONTINUATION_OK'
                validate_persisted_native_compaction_history(
                    db.get_messages_as_conversation('sess-1', repair_alternation=False))
            finally:
                fresh.close()
    finally:
        agent.close()
        db.close()
