"""Completion proof crosses the real native FIFO but never its serialized payload."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_local_producer_admission import messaging
from tools import async_delegation
from tools.delegate_waiting_reply import dispatch_identity, DispatchIdentity
from tools.waiting_lineage import execution_proof, prove_completion
from agent.turn_context import _bind_turn_identity


def bind(agent):
    _bind_turn_identity(agent, None, None, None, None, None)
    return dispatch_identity(agent)


def test_proof_rejects_missing_mixed_and_wrong_origin(monkeypatch):
    ident = DispatchIdentity('owner', 'root')
    records = {'a': {'parent_session_id': 'sid', 'session_key': 'route', 'waiting_identity': ident},
               'b': {'parent_session_id': 'sid', 'session_key': 'route', 'waiting_identity': ident}}
    monkeypatch.setattr(async_delegation, '_records', records)
    assert prove_completion(['a', 'b'], 'sid', 'route')
    assert not prove_completion(['a', 'missing'], 'sid', 'route')
    assert not prove_completion(['a', 'a'], 'sid', 'route')
    assert not prove_completion(['a'], 'other', 'route')
    assert not prove_completion(['a'], 'sid', 'other')
    records['b']['waiting_identity'] = DispatchIdentity('owner', 'different')
    assert not prove_completion(['a', 'b'], 'sid', 'route')
    proof = prove_completion(['a'], 'sid', 'route')
    records.pop('a')
    agent = SimpleNamespace(session_id='sid')
    with execution_proof(proof, 'sid', 'route'):
        assert bind(agent) != ident


def test_recreated_and_same_agent_adopt_once_then_human_gets_fresh_root(monkeypatch):
    original = SimpleNamespace(session_id='sid', _current_turn_id='old')
    identity = dispatch_identity(original)
    monkeypatch.setattr(async_delegation, '_records', {'a': {'parent_session_id': 'sid',
        'session_key': 'route', 'waiting_identity': identity}})
    for agent in (original, SimpleNamespace(session_id='sid')):
        proof = prove_completion(['a'], 'sid', 'route')
        with execution_proof(proof, 'sid', 'route'):
            assert bind(agent) == identity
        with execution_proof(proof, 'sid', 'route'):
            assert bind(agent) != identity
        assert bind(agent) != identity


@pytest.mark.asyncio
async def test_native_committed_snapshot_restores_one_use_proof(messaging, monkeypatch):
    from gateway.session_contract import SessionRef
    from gateway.session_ingress import execute_admission
    from hermes_state_runtime import claim_session_input, settle_session_input, list_session_admissions

    m = messaging
    old = SimpleNamespace(session_id=m.owner, _current_turn_id='old')
    identity = dispatch_identity(old)
    records = {'unit': {'parent_session_id': m.owner, 'session_key': m.route, 'waiting_identity': identity}}
    monkeypatch.setattr(async_delegation, '_records', records)
    monkeypatch.setattr(m.runner, '_completion_delivery_ready', AsyncMock(return_value=True))
    monkeypatch.setattr(m.runner, '_classify_completion_owner', AsyncMock(return_value=('deliver', m.owner, m.owner)))
    monkeypatch.setattr(async_delegation, 'claim_completion_delivery', lambda *a: True)
    monkeypatch.setattr(m.runner, '_settle_durable_claim', lambda *a: None)
    evt = m.event('async_delegation', delegation_id='unit', status='completed')
    assert await m.runner._deliver_completion_notification('[unit completed]', evt) is True
    rows = list_session_admissions(m.authority.db, session_id=m.owner, pending_only=False)
    queued = next(row for row in rows if row['principal_id'].startswith('automation:'))
    assert '_waiting_lineage_proof' not in str(queued['payload'])
    human = claim_session_input(m.authority.db, epoch=m.authority.epoch, session_id=m.owner)
    settle_session_input(m.authority.db, epoch=m.authority.epoch, admission_id=human['admission_id'],
                         generation=human['generation'], outcome='completed')
    row = claim_session_input(m.authority.db, epoch=m.authority.epoch, session_id=m.owner)
    assert row['admission_id'] == queued['admission_id']
    seen = []
    async def handle(event):
        proof = getattr(event, '_waiting_lineage_proof', None)
        seen.append(proof)
        agent = SimpleNamespace(session_id=m.owner)
        with execution_proof(proof, m.owner, m.route):
            assert bind(agent) == identity
        from gateway.session_results import execution_result
        execution_result.get()['result'] = {'final_response': 'ok'}
        return 'ok'
    monkeypatch.setattr(m.runner, '_handle_message', handle)
    ref = SessionRef(m.authority.profile_id, m.owner)
    await execute_admission(m.authority, ref, row)
    assert seen[0] is not None
    assert m.authority.take_waiting_lineage(ref, row, m.route) is None


@pytest.mark.asyncio
async def test_failed_claim_never_creates_association(messaging, monkeypatch):
    m = messaging
    monkeypatch.setattr(m.runner, '_completion_delivery_ready', AsyncMock(return_value=True))
    monkeypatch.setattr(async_delegation, 'claim_completion_delivery', lambda *a: False)
    evt = m.event('async_delegation', delegation_id='bad', status='completed',
                  _waiting_lineage_proof={'owner': 'forged', 'root': 'forged'})
    assert await m.runner._deliver_completion_notification('[done]', evt) is None
    assert not m.authority._waiting_lineage


@pytest.mark.asyncio
async def test_committed_association_fences_route_restart_and_claim(messaging, monkeypatch):
    from gateway.session_contract import SessionRef
    from gateway.session_authority import SessionAuthority
    from gateway.platforms.event import MessageEvent
    from hermes_state_runtime import list_session_admissions
    m = messaging
    ident = DispatchIdentity('owner', 'root')
    monkeypatch.setattr(async_delegation, '_records', {'u': {'parent_session_id': m.owner,
        'session_key': m.route, 'waiting_identity': ident}})
    proof = prove_completion(['u'], m.owner, m.route)
    event = MessageEvent(text='notice', source=m.authority.sessions[m.owner].source, internal=True)
    event._waiting_lineage_proof = proof
    row = list_session_admissions(m.authority.db, session_id=m.owner)[0]
    ref = SessionRef(m.authority.profile_id, m.owner)
    m.authority._admitted(ref, event, row)
    started = dict(row, status='started', generation=1)
    assert m.authority.take_waiting_lineage(ref, row, m.route) is None  # not claimed
    m.authority._admitted(ref, event, row)
    m.authority.fence_route(m.route)
    assert m.authority.take_waiting_lineage(ref, started, m.route) is None
    m.authority._admitted(ref, event, row)
    restarted = SessionAuthority(m.runner, profile_id=m.authority.profile_id, instance_id='new',
                                 db=m.authority.db, epoch=m.authority.epoch + 1)
    assert restarted.take_waiting_lineage(ref, started, m.route) is None
