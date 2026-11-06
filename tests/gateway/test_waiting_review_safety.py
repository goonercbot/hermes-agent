"""Genuine finals remain protected across both ephemeral cleanup doors."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from gateway.commentary_delivery import schedule_cleanup, track_progress
from gateway.platforms.base import SendResult
from gateway.platforms.post_delivery import record_final_ids
from gateway.turn_context import TurnContext
from tests.gateway.test_waiting_reply_delivery import setup, classify
from tools import async_delegation
from tools.delegate_waiting_reply import dispatch_identity
from tools.registry import registry
from tools.waiting_lineage import execution_proof, prove_completion, adopt_bound_turn


@pytest.fixture
def owned(monkeypatch):
    agent = SimpleNamespace(_current_turn_id='initial', session_id='sid')
    identity = dispatch_identity(agent)
    records = {'unit': {'status': 'running', 'waiting_identity': identity,
                        'parent_session_id': 'sid', 'session_key': 'session'}}
    records['trigger'] = {'status': 'completed', 'waiting_identity': identity,
                          'parent_session_id': 'sid', 'session_key': 'session'}
    monkeypatch.setattr(async_delegation, '_records', records)
    adapter, guard, ctx = setup(agent)
    return agent, identity, records, adapter, guard, ctx


async def delivered(adapter, guard, ctx, text, ids):
    guard._hermes_final_delivery_ids.clear()
    schedule_cleanup({'final_response': text}, adapter, ctx)
    record_final_ids(guard, SendResult(success=True, message_id=ids[0],
                      raw_response={'message_ids': list(ids)}, full_payload_delivered=True))
    adapter._confirm_final_delivery(ctx.session_key, ctx.run_generation, True)
    await adapter._fire_post_delivery_callback(ctx.session_key, guard)


def resume(owned, turn='continuation'):
    _, _, records, _, _, ctx = owned
    records['unit']['status'] = 'completed'
    proof = prove_completion(['unit'], 'sid', 'session')
    agent = SimpleNamespace(_current_turn_id=turn, session_id='sid')
    with execution_proof(proof, 'sid', 'session'):
        assert adopt_bound_turn(agent, turn)
    assert json.loads(registry.dispatch('delegate_task', {
        'action': 'resolve_waiting_reply', 'delegation_ids': ['unit']},
        parent_agent=agent))['status'] == 'resolved'
    ctx.agent_holder[0] = agent
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize('initial_wait', [False, True])
async def test_every_known_genuine_final_survives_later_wait_alias(owned, initial_wait):
    agent, identity, records, adapter, guard, ctx = owned
    if initial_wait:
        classify(agent, 'unit')
        await delivered(adapter, guard, ctx, 'Waiting initially.', ['initial-wait'])
    # Native resolution intent is not needed for an ordinary substantive answer.
    agent._current_turn_id = 'substantive'
    await delivered(adapter, guard, ctx, 'A genuine substantive answer.', ['genuine-answer'])
    # Another original-request continuation emits a wait whose receipt aliases that answer.
    # The trigger is completed; the classified unit is a distinct still-running sibling.
    proof = prove_completion(['trigger'], 'sid', 'session')
    waiting_agent = SimpleNamespace(_current_turn_id='wait-again', session_id='sid')
    with execution_proof(proof, 'sid', 'session'):
        assert adopt_bound_turn(waiting_agent, 'wait-again')
    ctx.agent_holder[0] = waiting_agent
    classify(waiting_agent, 'unit')
    await delivered(adapter, guard, ctx, 'Waiting again.', ['genuine-answer'])
    resume(owned)
    await delivered(adapter, guard, ctx, 'Resolved.', ['resolving-final'])
    deleted = [item['message_id'] for item in adapter.deleted]
    print('known-final-alias observed deletions:', deleted)
    assert 'genuine-answer' not in deleted


@pytest.mark.asyncio
async def test_concurrent_promotion_is_rechecked_before_each_delete(owned):
    agent, identity, records, adapter, guard, ctx = owned
    classify(agent, 'unit')
    await delivered(adapter, guard, ctx, 'Waiting.', ['wait-a', 'wait-b', 'wait-c'])
    key = ('session', 'chat', 'thread', identity.root_request_id)
    order = tuple(adapter._waiting_reply_receipts[key]['message_ids'])
    promoted_id = order[1]
    resume(owned)
    other_guard = asyncio.Event()
    other_guard._hermes_run_generation = 1
    other_guard._hermes_final_delivery_ids = set()
    adapter._active_sessions['other-session'] = other_guard
    other_ctx = TurnContext(source=ctx.source, session_key='other-session', run_generation=1,
                            _cleanup_progress=True, _run_still_current=lambda: True)
    other_ctx.agent_holder[0] = SimpleNamespace(_current_turn_id='other-answer')
    deleted = []
    async def delete(chat, mid):
        deleted.append(mid)
        if len(deleted) == 1:
            # Another live route finishes while the first destructive transport awaits.
            await delivered(adapter, other_guard, other_ctx, 'Genuine final.', [promoted_id])
            assert ('chat', promoted_id) in adapter._waiting_promoted_ids
            assert promoted_id not in adapter._waiting_reply_receipts[key]['message_ids']
        return True
    adapter.delete_message = delete
    await delivered(adapter, guard, ctx, 'Resolved.', ['new-final'])
    print('concurrent-promotion target:', promoted_id, 'observed deletions:', deleted)
    assert promoted_id not in deleted


@pytest.mark.asyncio
async def test_same_turn_sibling_cleanup_honors_permanent_promotion(owned):
    agent, _, _, adapter, guard, ctx = owned
    classify(agent, 'unit')
    await delivered(adapter, guard, ctx, 'Waiting.', ['genuine-answer', 'other-wait'])
    agent._current_turn_id = 'answer'
    await delivered(adapter, guard, ctx, 'A genuine answer.', ['genuine-answer'])
    assert ('chat', 'genuine-answer') in adapter._waiting_promoted_ids
    agent._current_turn_id = 'later-turn'
    # A tracked commentary/progress receipt aliases the already promoted answer.
    track_progress(ctx, SendResult(success=True, message_id='genuine-answer'))
    await delivered(adapter, guard, ctx, 'A later final.', ['later-final'])
    deleted = [item['message_id'] for item in adapter.deleted]
    print('same-turn sibling observed deletions:', deleted)
    assert 'genuine-answer' not in deleted


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['cancel', 'active_unit', 'new_dispatch', 'ledger_replacement'])
async def test_existing_per_delete_ownership_fences_stop_following_deletes(owned, change):
    agent, identity, records, adapter, guard, ctx = owned
    classify(agent, 'unit')
    await delivered(adapter, guard, ctx, 'Waiting.', ['wait-a', 'wait-b', 'wait-c'])
    resumed = resume(owned)
    deleted = []
    async def delete(chat, mid):
        deleted.append(mid)
        if change == 'cancel':
            guard.set()
        elif change == 'active_unit':
            records['new-running'] = {'status': 'running', 'waiting_identity': identity}
        elif change == 'new_dispatch':
            dispatch_identity(resumed)
        else:
            adapter._waiting_reply_receipts = {}
        return True
    adapter.delete_message = delete
    await delivered(adapter, guard, ctx, 'Resolved.', ['final'])
    assert len(deleted) == 1


@pytest.mark.asyncio
async def test_promotion_bound_disables_capture_without_deleting(owned, monkeypatch):
    import gateway.waiting_reply_delivery as delivery
    monkeypatch.setattr(delivery, '_MAX_IDS', 2)
    agent, _, _, adapter, guard, ctx = owned
    for index in range(3):
        agent._current_turn_id = 'waiting-' + str(index)
        identity = dispatch_identity(agent)
        async_delegation._records['unit']['waiting_identity'] = identity
        classify(agent, 'unit')
        await delivered(adapter, guard, ctx, 'Waiting.', [f'message-{index}'])
        agent._current_turn_id = 'answer-' + str(index)
        await delivered(adapter, guard, ctx, 'A genuine answer.', [f'message-{index}'])
    assert adapter._waiting_receipts_disabled is True
    assert not adapter._waiting_reply_receipts
    assert not adapter.deleted
    agent._current_turn_id = 'capture-after-bound'
    identity = dispatch_identity(agent)
    async_delegation._records['unit']['waiting_identity'] = identity
    classify(agent, 'unit')
    await delivered(adapter, guard, ctx, 'Waiting.', ['never-record'])
    assert not adapter._waiting_reply_receipts


@pytest.mark.asyncio
@pytest.mark.parametrize('form', ['normal-split', 'partial', 'stream', 'preview', 'stream-message'])
async def test_all_final_receipt_forms_protected_before_wait(owned, form):
    agent, _, _, adapter, guard, ctx = owned
    consumer = SimpleNamespace(_final_receipt_ids=set(), _segment_preview_message_ids=set(),
                               message_id=None, final_content_delivered=True)
    ctx.stream_consumer_holder[0] = consumer
    response = {'final_response': 'A genuine answer.', 'partial': form == 'partial'}
    if form == 'stream':
        response['_complete_final_body'] = True
        consumer._final_receipt_ids.add('genuine-answer')
    elif form == 'preview':
        consumer._segment_preview_message_ids.add('genuine-answer')
    elif form == 'stream-message':
        consumer.message_id = 'genuine-answer'
    schedule_cleanup(response, adapter, ctx)
    if form in ('normal-split', 'partial'):
        record_final_ids(guard, SendResult(success=True, message_id='first',
                         continuation_message_ids=('genuine-answer',), full_payload_delivered=False))
    adapter._confirm_final_delivery(ctx.session_key, ctx.run_generation, True)
    await adapter._fire_post_delivery_callback(ctx.session_key, guard)
    assert ('chat', 'genuine-answer') in adapter._waiting_promoted_ids
    ctx.stream_consumer_holder[0] = None
    guard._hermes_final_delivery_ids.clear()
    classify(agent, 'unit')
    await delivered(adapter, guard, ctx, 'Waiting.', ['genuine-answer'])
    resume(owned)
    await delivered(adapter, guard, ctx, 'Resolved.', ['final'])
    assert 'genuine-answer' not in [item['message_id'] for item in adapter.deleted]


@pytest.mark.asyncio
async def test_capacity_exhaustion_also_stops_same_turn_cleanup(owned, monkeypatch):
    import gateway.waiting_reply_delivery as delivery
    monkeypatch.setattr(delivery, '_MAX_IDS', 1)
    agent, _, _, adapter, guard, ctx = owned
    await delivered(adapter, guard, ctx, 'First.', ['first'])
    track_progress(ctx, SendResult(success=True, message_id='untracked-final'))
    agent._current_turn_id = 'next'
    await delivered(adapter, guard, ctx, 'Second.', ['second'])
    assert adapter._waiting_receipts_disabled is True
    assert not adapter.deleted
