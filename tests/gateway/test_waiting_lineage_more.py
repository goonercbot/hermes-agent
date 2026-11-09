"""Additional lineage regressions at the adapter and actual dispatch boundaries."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

from agent.turn_context import _bind_turn_identity
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tools import async_delegation
from tools.delegate_waiting_reply import dispatch_identity
from tools.waiting_lineage import execution_proof, prove_completion


def test_continuation_dispatch_registers_original_identity(monkeypatch):
    from tools.delegate_tool_dispatch import _Batch, _dispatch_background
    import tools.delegate_tool_dispatch as dispatch
    original = SimpleNamespace(session_id='sid', _current_turn_id='prior')
    identity = dispatch_identity(original)
    monkeypatch.setattr(async_delegation, '_records', {'previous': {'parent_session_id': 'sid',
        'session_key': 'route', 'waiting_identity': identity}})
    proof = prove_completion(['previous'], 'sid', 'route')
    agent = SimpleNamespace(session_id='sid')
    with execution_proof(proof, 'sid', 'route'):
        _bind_turn_identity(agent, None, None, None, None, None)
        child = SimpleNamespace(model='test', get_activity_summary=lambda: {})
        task = {'goal': 'next batch'}
        done = threading.Event()
        monkeypatch.setattr(dispatch, '_resolve_async_wake_sid', lambda *a, **k: 'wake')
        monkeypatch.setattr(dispatch, '_detach_child', lambda *a: None)
        monkeypatch.setattr(dispatch, '_execute_and_aggregate', lambda *a, **k: (done.wait(10) and {'results': []}))
        batch = _Batch(task_list=[task], children=[(0, task, child)], parent_agent=agent,
            creds={'model': 'test'}, context=None, top_role='leaf', max_children=1,
            live_deleg_id=None, live_writers=[], live_paths=[], origin_wake_sid='', origin_ui_session_id='',
            origin_owner_transport=None, origin_owner_session_record=None, origin_session_history_delivery=False,
            overall_start=time.monotonic())
        try:
            first = json.loads(_dispatch_background(batch))
            second = json.loads(_dispatch_background(batch))
            assert first['status'] == second['status'] == 'dispatched'
            for item in (first, second):
                assert async_delegation._records[item['delegation_id']]['waiting_identity'] == identity
        finally:
            done.set()


@pytest.mark.asyncio
async def test_non_authority_adapter_queue_keeps_typed_proof(monkeypatch):
    import gateway.session_authorities as authorities
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(platforms={})
    runner._running = True
    source = SessionSource(platform=Platform.SLACK, chat_id='dm', chat_type='dm', user_id='u')
    route = runner._session_key_for_source(source)
    ident = dispatch_identity(SimpleNamespace(session_id='sid', _current_turn_id='prior'))
    monkeypatch.setattr(async_delegation, '_records', {'u': {'parent_session_id': 'sid',
        'session_key': route, 'waiting_identity': ident}})
    proof = prove_completion(['u'], 'sid', route)
    events = []
    class Adapter:
        async def handle_message(self, event):
            events.append(event)
            event._gateway_accepted = True
    adapter = Adapter()
    runner.adapters = {Platform.SLACK: adapter}
    monkeypatch.setattr(runner, '_build_process_event_source', lambda evt: source)
    monkeypatch.setattr(runner, '_resolve_injection_adapter', lambda *a: adapter)
    monkeypatch.setattr(authorities, 'active_authority', lambda runner: None)
    evt = {'session_key': route, 'type': 'async_delegation', '_waiting_lineage_proof': proof}
    assert await runner._inject_watch_notification('completed', evt) is True
    assert getattr(events[0], '_waiting_lineage_proof') is proof
    agent = SimpleNamespace(session_id='sid')
    with execution_proof(events[0]._waiting_lineage_proof, 'sid', route):
        _bind_turn_identity(agent, None, None, None, None, None)
        assert dispatch_identity(agent) == ident


def test_turn_runner_passes_proof_to_selected_agent(monkeypatch):
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext
    original = SimpleNamespace(session_id='sid', _current_turn_id='prior')
    identity = dispatch_identity(original)
    monkeypatch.setattr(async_delegation, '_records', {'unit': {'parent_session_id': 'sid',
        'session_key': 'route', 'waiting_identity': identity}})
    proof = prove_completion(['unit'], 'sid', 'route')
    agent = SimpleNamespace(session_id='sid')
    def run(message, **kwargs):
        _bind_turn_identity(agent, None, None, None, None, None)
        assert dispatch_identity(agent) == identity
        return {'final_response': 'ok'}
    agent.run_conversation = run
    turn = TurnRunner(SimpleNamespace(session_authority=None, _consume_pending_native_image_paths=lambda key: []),
        TurnContext(message='hello', session_id='sid', session_key='route', waiting_lineage_proof=proof,
                    source=SimpleNamespace(user_id='u', user_name='U', is_bot=False)))
    assert turn._run_conversation_with_approval(agent, [], None, None, None)['final_response'] == 'ok'


@pytest.mark.asyncio
@pytest.mark.parametrize('route_matches', [True, False])
async def test_recursive_followup_uses_only_its_matching_origin(monkeypatch, route_matches):
    from unittest.mock import AsyncMock, MagicMock
    from gateway.platforms.event import MessageEvent
    from gateway.turn_context import TurnContext
    from tests.gateway.test_internal_event_pin_wiring import _make_runner, _capture, _human_source, KEY
    runner = _make_runner(monkeypatch)
    calls = []
    _capture(runner, calls)
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value='queued')
    runner._session_key_for_source = lambda source: KEY if route_matches else 'different-route'
    original = SimpleNamespace(session_id='sid', _current_turn_id='prior')
    identity = dispatch_identity(original)
    monkeypatch.setattr(async_delegation, '_records', {'u': {'parent_session_id': 'sid',
        'session_key': KEY, 'waiting_identity': identity}})
    proof = prove_completion(['u'], 'sid', KEY)
    source = _human_source()
    event = MessageEvent(text='queued', source=source, internal=True)
    event._waiting_lineage_proof = proof
    adapter = MagicMock()
    adapter._active_sessions = {}
    ctx = TurnContext(source=source, context_prompt='ctx', session_key=KEY, session_id='sid',
                      run_generation=1, history=[])
    result = {'final_response': 'done', 'messages': []}
    await runner._run_agent_queued_followup(ctx, adapter, 'queued', event, 'done', result, None)
    agent = SimpleNamespace(session_id='sid')
    with execution_proof(calls[0]['waiting_lineage_proof'], 'sid', calls[0]['session_key']):
        _bind_turn_identity(agent, None, None, None, None, None)
        assert (dispatch_identity(agent) == identity) is route_matches
