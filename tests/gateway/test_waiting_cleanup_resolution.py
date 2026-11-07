"""Integration: native classification, delivery and proven completion; no external transport."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from gateway.commentary_delivery import schedule_cleanup
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from tests.gateway.test_waiting_reply_delivery import setup, classify
from tools import async_delegation
from tools.delegate_waiting_reply import dispatch_identity
from tools.waiting_lineage import prove_completion, execution_proof, adopt_bound_turn
from tools.registry import registry
from tests.gateway.test_local_producer_admission import messaging


@pytest.mark.asyncio
@pytest.mark.parametrize('tts', [False, True])
async def test_native_wait_to_proven_final_deletes_only_wait(tts, tmp_path):
    agent = SimpleNamespace(_current_turn_id='first', session_id='sid')
    identity = dispatch_identity(agent)
    unit = 'resolution-unit'
    with async_delegation._records_lock:
        async_delegation._records[unit] = {'status': 'running', 'waiting_identity': identity,
                                          'parent_session_id': 'sid', 'session_key': 'session'}
    try:
        classify(agent, unit)
        adapter, guard, ctx = setup(agent)
        event = MessageEvent(text='question', message_type=MessageType.TEXT, source=ctx.source, message_id='in')
        adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True, message_id='wait', full_payload_delivered=True, fresh_message_ids=('wait',)))
        async def waiting(_):
            schedule_cleanup({'final_response': 'Waiting.'}, adapter, ctx)
            return 'Waiting.'
        adapter.set_message_handler(waiting)
        await adapter._process_message_background(event, 'session')
        async_delegation._records[unit]['status'] = 'completed'
        proof = prove_completion([unit], 'sid', 'session')
        new_agent = SimpleNamespace(_current_turn_id='second', session_id='sid')
        with execution_proof(proof, 'sid', 'session'):
            assert adopt_bound_turn(new_agent, 'second')
        result = json.loads(registry.dispatch('delegate_task', {'action': 'resolve_waiting_reply', 'delegation_ids': [unit]}, parent_agent=new_agent))
        assert result['status'] == 'resolved'
        ctx.agent_holder[0] = new_agent
        guard = asyncio.Event()
        guard._hermes_run_generation = 2
        adapter._active_sessions['session'] = guard
        ctx.run_generation = 2
        adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True, message_id='final', full_payload_delivered=True))
        if tts:
            adapter._wants_auto_tts = lambda *a: True
            adapter._synthesize_auto_tts = AsyncMock(return_value=([str(tmp_path / 'fake-voice')], None))
            async def play(_event, _text, _path, _first, _metadata, record):
                record(SendResult(success=True, message_id='audio'))
                return False
            adapter._play_tts_file = play
        async def final(_):
            schedule_cleanup({'final_response': 'Finished.'}, adapter, ctx)
            return 'Finished.'
        adapter.set_message_handler(final)
        await adapter._process_message_background(event, 'session')
        assert adapter.deleted == ([] if tts else [{'chat_id': 'chat', 'message_id': 'wait'}])
    finally:
        with async_delegation._records_lock:
            async_delegation._records.pop(unit, None)


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['ok', 'no_resolution', 'pending', 'pruned', 'disabled',
                                 'partial', 'failed', 'unknown', 'missing_final_ids', 'media', 'nontext',
                                 'stale_guard', 'stale_generation', 'wrong_session', 'overlap',
                                 'streamed', 'delete_failure', 'mixed_root'])
async def test_resolution_controls_and_receipt_union(case):
    from gateway.platforms.post_delivery import record_final_ids
    from gateway.stream_consumer import GatewayStreamConsumer
    from tools.delegate_waiting_reply import resolving_reply
    agent = SimpleNamespace(_current_turn_id='t', session_id='sid')
    identity = dispatch_identity(agent)
    units = ['case-unit-a', 'case-unit-b']
    with async_delegation._records_lock:
        for uid in units:
            async_delegation._records[uid] = {'status': 'running', 'waiting_identity': identity,
                                               'parent_session_id': 'sid', 'session_key': 'session'}
    try:
        adapter, guard, ctx = setup(agent)
        for index, uid in enumerate(units):
            classify(agent, uid)
            schedule_cleanup({'final_response': 'Waiting.'}, adapter, ctx)
            guard._hermes_final_fresh_message_ids = set()
            record_final_ids(guard, SendResult(success=True, message_id=f'wait{index}', full_payload_delivered=True, fresh_message_ids=(f'wait{index}',)))
            adapter._confirm_final_delivery('session', 1, True)
            await adapter._fire_post_delivery_callback('session', guard)
            guard._hermes_final_delivery_ids.clear()
        key = ('session', 'chat', 'thread', identity.root_request_id)
        assert adapter._waiting_reply_receipts[key]['message_ids'] == {'wait0', 'wait1'}
        assert set(adapter._waiting_reply_receipts[key]['delegation_ids']) == set(units)
        for uid in units:
            async_delegation._records[uid]['status'] = 'completed'
        proof = prove_completion(units, 'sid', 'session')
        agent = SimpleNamespace(_current_turn_id='next', session_id='sid')
        with execution_proof(proof, 'sid', 'session'):
            assert adopt_bound_turn(agent, 'next')
        if case != 'no_resolution':
            assert json.loads(registry.dispatch('delegate_task', {'action': 'resolve_waiting_reply',
                              'delegation_ids': [units[0]]}, parent_agent=agent))['status'] == 'resolved'
        ctx.agent_holder[0] = agent
        if case == 'pending':
            async_delegation._records[units[1]]['status'] = 'running'
        if case == 'pruned':
            async_delegation._records.pop(units[1])
        if case == 'mixed_root':
            async_delegation._records[units[1]]['waiting_identity'] = dispatch_identity(SimpleNamespace(_current_turn_id='other'))
        if case == 'disabled':
            ctx._cleanup_progress = False
        if case == 'wrong_session':
            ctx.session_key = 'other'
            adapter._active_sessions['other'] = guard
        response = {'final_response': 'Done.'}
        if case in ('partial', 'failed'):
            response[case] = True
        if case == 'media':
            response['final_response'] += ' MEDIA:/tmp/file'
        if case == 'nontext':
            guard._hermes_nontext_delivery_attempted = True
        if case == 'streamed':
            consumer = GatewayStreamConsumer(adapter, 'chat')
            adapter.send = AsyncMock(return_value=SendResult(success=True, message_id='final',
                         raw_response={'message_ids': ['final2']}, full_payload_delivered=True))
            assert await consumer._try_fresh_final('Done.')
            ctx.stream_consumer_holder[0] = consumer
            response['_complete_final_body'] = True
        schedule_cleanup(response, adapter, ctx)
        final_id = 'wait0' if case == 'overlap' else 'final'
        if case != 'missing_final_ids':
            record_final_ids(guard, SendResult(success=True, message_id=final_id, full_payload_delivered=True))
        if case != 'unknown':
            adapter._confirm_final_delivery(ctx.session_key, 1, True)
        else:
            adapter._confirm_final_delivery(ctx.session_key, 1, False)
        if case == 'stale_guard':
            adapter._active_sessions['session'] = asyncio.Event()
        if case == 'stale_generation':
            guard._hermes_run_generation = 2
        if case == 'delete_failure':
            adapter.delete_message = AsyncMock(return_value=False)
        await adapter._fire_post_delivery_callback(ctx.session_key, guard)
        if case in ('ok', 'streamed'):
            assert {r['message_id'] for r in adapter.deleted} == {'wait0', 'wait1'}
        elif case == 'overlap':
            assert adapter.deleted == [{'chat_id': 'chat', 'message_id': 'wait1'}]
            assert key not in adapter._waiting_reply_receipts
        else:
            assert not adapter.deleted
            assert key in adapter._waiting_reply_receipts
        if case == 'delete_failure':
            assert adapter.delete_message.await_count == 2
        if case != 'no_resolution':
            assert resolving_reply(agent)
    finally:
        with async_delegation._records_lock:
            for uid in units:
                async_delegation._records.pop(uid, None)


def test_resolution_invalidated_by_classification_or_spawn_attempt():
    from tools.delegate_waiting_reply import resolving_reply
    agent = SimpleNamespace(_current_turn_id='t')
    identity = dispatch_identity(agent)
    uid = 'invalidation-unit'
    async_delegation._records[uid] = {'status': 'completed', 'waiting_identity': identity}
    try:
        for action in ('classify_waiting_reply', 'spawn'):
            assert json.loads(registry.dispatch('delegate_task', {'action': 'resolve_waiting_reply',
                              'delegation_ids': [uid]}, parent_agent=agent))['status'] == 'resolved'
            registry.dispatch('delegate_task', {'action': action, 'delegation_ids': []}, parent_agent=agent)
            assert resolving_reply(agent) is None
    finally:
        async_delegation._records.pop(uid, None)


def test_resolution_admission_rejects_pending_foreign_and_lost_state():
    agent = SimpleNamespace(_current_turn_id='t')
    identity = dispatch_identity(agent)
    foreign = dispatch_identity(SimpleNamespace(_current_turn_id='foreign'))
    async_delegation._records['done'] = {'status': 'completed', 'waiting_identity': identity}
    async_delegation._records['pending'] = {'status': 'running', 'waiting_identity': identity}
    async_delegation._records['foreign'] = {'status': 'completed', 'waiting_identity': foreign}
    try:
        def resolve(ids):
            return json.loads(registry.dispatch('delegate_task', {'action': 'resolve_waiting_reply',
                              'delegation_ids': ids}, parent_agent=agent))
        assert 'error' in resolve(['done'])
        async_delegation._records['pending']['status'] = 'completed'
        assert 'error' in resolve(['done', 'foreign'])
        assert 'error' in resolve(['done', 'missing'])
        assert resolve(['done'])['status'] == 'resolved'
        agent._current_turn_id = 'unrelated-user-turn'
        assert 'error' in resolve(['done'])
    finally:
        for uid in ('done', 'pending', 'foreign'):
            async_delegation._records.pop(uid, None)


@pytest.mark.asyncio
async def test_unrelated_genuine_final_promotes_overlap_before_later_resolution():
    from gateway.platforms.post_delivery import record_final_ids
    agent = SimpleNamespace(_current_turn_id='t', session_id='sid')
    identity = dispatch_identity(agent)
    uid = 'promotion-unit'
    async_delegation._records[uid] = {'status': 'running', 'waiting_identity': identity,
                                      'parent_session_id': 'sid', 'session_key': 'session'}
    try:
        classify(agent, uid)
        adapter, guard, ctx = setup(agent)
        schedule_cleanup({'final_response': 'Waiting.'}, adapter, ctx)
        record_final_ids(guard, SendResult(success=True, message_id='promoted',
                                          raw_response={'message_ids': ['wait']}, full_payload_delivered=True,
                                          fresh_message_ids=('promoted', 'wait')))
        adapter._confirm_final_delivery('session', 1, True)
        await adapter._fire_post_delivery_callback('session', guard)
        agent._current_turn_id = 'unrelated'
        guard._hermes_final_delivery_ids.clear()
        schedule_cleanup({'final_response': 'A substantive answer.'}, adapter, ctx)
        record_final_ids(guard, SendResult(success=True, message_id='promoted', full_payload_delivered=True))
        adapter._confirm_final_delivery('session', 1, True)
        await adapter._fire_post_delivery_callback('session', guard)
        from tools.delegate_waiting_reply import adopt_dispatch_identity
        agent._current_turn_id = 'another-wait'
        guard._hermes_final_fresh_message_ids = set()
        assert adopt_dispatch_identity(agent, 'another-wait', identity)
        classify(agent, uid)
        guard._hermes_final_delivery_ids.clear()
        schedule_cleanup({'final_response': 'Waiting again.'}, adapter, ctx)
        record_final_ids(guard, SendResult(success=True, message_id='promoted', full_payload_delivered=True))
        adapter._confirm_final_delivery('session', 1, True)
        await adapter._fire_post_delivery_callback('session', guard)
        async_delegation._records[uid]['status'] = 'completed'
        proof = prove_completion([uid], 'sid', 'session')
        continuation = SimpleNamespace(_current_turn_id='completion', session_id='sid')
        with execution_proof(proof, 'sid', 'session'):
            assert adopt_bound_turn(continuation, 'completion')
        assert json.loads(registry.dispatch('delegate_task', {'action': 'resolve_waiting_reply',
                          'delegation_ids': [uid]}, parent_agent=continuation))['status'] == 'resolved'
        ctx.agent_holder[0] = continuation
        guard._hermes_final_delivery_ids.clear()
        schedule_cleanup({'final_response': 'Finished.'}, adapter, ctx)
        record_final_ids(guard, SendResult(success=True, message_id='final', full_payload_delivered=True))
        adapter._confirm_final_delivery('session', 1, True)
        await adapter._fire_post_delivery_callback('session', guard)
        assert adapter.deleted == [{'chat_id': 'chat', 'message_id': 'wait'}]
    finally:
        async_delegation._records.pop(uid, None)


@pytest.mark.asyncio
async def test_native_admission_snapshot_to_resolution_delivery(messaging, monkeypatch):
    from agent.turn_context import _bind_turn_identity
    from gateway.session_contract import SessionRef
    from gateway.session_ingress import execute_admission
    from gateway.session_results import execution_result
    from gateway.platforms.post_delivery import record_final_ids
    from hermes_state_runtime import claim_session_input, settle_session_input, list_session_admissions

    m = messaging
    original = SimpleNamespace(_current_turn_id='initial', session_id=m.owner)
    identity = dispatch_identity(original)
    records = {'unit': {'status': 'running', 'parent_session_id': m.owner,
                        'session_key': m.route, 'waiting_identity': identity}}
    monkeypatch.setattr(async_delegation, '_records', records)
    classify(original, 'unit')
    adapter, guard, ctx = setup(original)
    adapter._active_sessions[m.route] = adapter._active_sessions.pop('session')
    ctx.session_key = m.route
    schedule_cleanup({'final_response': 'Waiting.'}, adapter, ctx)
    record_final_ids(guard, SendResult(success=True, message_id='wait', full_payload_delivered=True, fresh_message_ids=('wait',)))
    adapter._confirm_final_delivery(m.route, 1, True)
    await adapter._fire_post_delivery_callback(m.route, guard)
    records['unit']['status'] = 'completed'
    monkeypatch.setattr(m.runner, '_completion_delivery_ready', AsyncMock(return_value=True))
    monkeypatch.setattr(m.runner, '_classify_completion_owner', AsyncMock(return_value=('deliver', m.owner, m.owner)))
    monkeypatch.setattr(async_delegation, 'claim_completion_delivery', lambda *a: True)
    monkeypatch.setattr(m.runner, '_settle_durable_claim', lambda *a: None)
    evt = m.event('async_delegation', delegation_id='unit', status='completed')
    assert await m.runner._deliver_completion_notification('[unit completed]', evt) is True
    rows = list_session_admissions(m.authority.db, session_id=m.owner, pending_only=False)
    queued = next(row for row in rows if row['principal_id'].startswith('automation:'))
    human = claim_session_input(m.authority.db, epoch=m.authority.epoch, session_id=m.owner)
    settle_session_input(m.authority.db, epoch=m.authority.epoch, admission_id=human['admission_id'],
                         generation=human['generation'], outcome='completed')
    row = claim_session_input(m.authority.db, epoch=m.authority.epoch, session_id=m.owner)
    assert row['admission_id'] == queued['admission_id']
    async def handle(event):
        agent = SimpleNamespace(session_id=m.owner)
        with execution_proof(getattr(event, '_waiting_lineage_proof', None), m.owner, m.route):
            _bind_turn_identity(agent, None, None, None, None, None)
        assert json.loads(registry.dispatch('delegate_task', {'action': 'resolve_waiting_reply',
                          'delegation_ids': ['unit']}, parent_agent=agent))['status'] == 'resolved'
        ctx.agent_holder[0] = agent
        guard._hermes_final_delivery_ids.clear()
        schedule_cleanup({'final_response': 'Finished.'}, adapter, ctx)
        record_final_ids(guard, SendResult(success=True, message_id='final', full_payload_delivered=True))
        adapter._confirm_final_delivery(m.route, 1, True)
        await adapter._fire_post_delivery_callback(m.route, guard)
        execution_result.get()['result'] = {'final_response': 'Finished.'}
        return 'Finished.'
    monkeypatch.setattr(m.runner, '_handle_message', handle)
    await execute_admission(m.authority, SessionRef(m.authority.profile_id, m.owner), row)
    assert adapter.deleted == [{'chat_id': 'chat', 'message_id': 'wait'}]


@pytest.mark.asyncio
async def test_deletion_does_not_replace_newer_receipt_ledger():
    from gateway.platforms.post_delivery import record_final_ids
    agent = SimpleNamespace(_current_turn_id='first', session_id='sid')
    identity = dispatch_identity(agent)
    uid = 'race-unit'
    async_delegation._records[uid] = {'status': 'running', 'waiting_identity': identity,
                                      'parent_session_id': 'sid', 'session_key': 'session'}
    try:
        classify(agent, uid)
        adapter, guard, ctx = setup(agent)
        schedule_cleanup({'final_response': 'Wait'}, adapter, ctx)
        record_final_ids(guard, SendResult(success=True, message_id='old1',
                                          raw_response={'message_ids': ['old2']}, full_payload_delivered=True,
                                          fresh_message_ids=('old1', 'old2')))
        adapter._confirm_final_delivery('session', 1, True)
        await adapter._fire_post_delivery_callback('session', guard)
        key = ('session', 'chat', 'thread', identity.root_request_id)
        async_delegation._records[uid]['status'] = 'completed'
        proof = prove_completion([uid], 'sid', 'session')
        resumed = SimpleNamespace(_current_turn_id='resume', session_id='sid')
        with execution_proof(proof, 'sid', 'session'):
            assert adopt_bound_turn(resumed, 'resume')
        assert json.loads(registry.dispatch('delegate_task', {'action': 'resolve_waiting_reply',
                          'delegation_ids': [uid]}, parent_agent=resumed))['status'] == 'resolved'
        ctx.agent_holder[0] = resumed
        guard._hermes_final_delivery_ids.clear()
        schedule_cleanup({'final_response': 'Done'}, adapter, ctx)
        record_final_ids(guard, SendResult(success=True, message_id='final', full_payload_delivered=True))
        adapter._confirm_final_delivery('session', 1, True)
        newer = {'message_ids': frozenset({'new'}), 'delegation_ids': (uid,)}
        async def delete(_chat, _mid):
            adapter._waiting_reply_receipts[key] = newer
            return True
        adapter.delete_message = AsyncMock(side_effect=delete)
        await adapter._fire_post_delivery_callback('session', guard)
        assert adapter.delete_message.await_count == 1
        assert adapter._waiting_reply_receipts[key] is newer
    finally:
        async_delegation._records.pop(uid, None)
