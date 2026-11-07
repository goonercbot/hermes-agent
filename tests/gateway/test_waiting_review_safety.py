"""Genuine finals remain protected across both ephemeral cleanup doors."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from gateway.commentary_delivery import schedule_cleanup, track_progress
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
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
    guard._hermes_final_fresh_message_ids = set()
    # A reused id models an edit/alias, never a fresh send.
    seen = getattr(adapter, '_test_seen_ids', set())
    fresh = tuple(mid for mid in ids if mid not in seen)
    seen.update(ids)
    adapter._test_seen_ids = seen
    schedule_cleanup({'final_response': text}, adapter, ctx)
    record_final_ids(guard, SendResult(success=True, message_id=ids[0],
                      raw_response={'message_ids': list(ids)}, full_payload_delivered=True,
                      fresh_message_ids=fresh))
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
    # The alias is an edit, not a new progress send, so it has no fresh proof.
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
async def test_many_promotions_do_not_disable_later_capture(owned):
    agent, _, _, adapter, guard, ctx = owned
    for index in range(300):
        agent._current_turn_id = 'waiting-' + str(index)
        identity = dispatch_identity(agent)
        async_delegation._records['unit']['waiting_identity'] = identity
        classify(agent, 'unit')
        await delivered(adapter, guard, ctx, 'Waiting.', [f'message-{index}'])
        agent._current_turn_id = 'answer-' + str(index)
        await delivered(adapter, guard, ctx, 'A genuine answer.', [f'message-{index}'])
    assert not adapter._waiting_reply_receipts
    assert not adapter.deleted
    agent._current_turn_id = 'capture-after-many'
    identity = dispatch_identity(agent)
    async_delegation._records['unit']['waiting_identity'] = identity
    classify(agent, 'unit')
    await delivered(adapter, guard, ctx, 'Waiting.', ['new-receipt'])
    assert next(iter(adapter._waiting_reply_receipts.values()))['message_ids'] == {'new-receipt'}


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
    # The later waiting delivery aliases the previously delivered final, not a new send.
    adapter._test_seen_ids = {'genuine-answer', 'first'}
    ctx.stream_consumer_holder[0] = None
    guard._hermes_final_delivery_ids.clear()
    classify(agent, 'unit')
    await delivered(adapter, guard, ctx, 'Waiting.', ['genuine-answer'])
    resume(owned)
    await delivered(adapter, guard, ctx, 'Resolved.', ['final'])
    assert 'genuine-answer' not in [item['message_id'] for item in adapter.deleted]


@pytest.mark.asyncio
async def test_progress_cleanup_continues_after_prior_final(owned):
    agent, _, _, adapter, guard, ctx = owned
    await delivered(adapter, guard, ctx, 'First.', ['first'])
    track_progress(ctx, SendResult(success=True, message_id='untracked-final'))
    track_progress(ctx, SendResult(success=True, message_id='fresh-progress',
                                   fresh_message_ids=('fresh-progress',)))
    agent._current_turn_id = 'next'
    await delivered(adapter, guard, ctx, 'Second.', ['second'])
    assert [item['message_id'] for item in adapter.deleted] == ['fresh-progress']


@pytest.mark.asyncio
async def test_two_thousand_fresh_request_lifecycles_leave_no_receipts(owned):
    _, _, records, adapter, _, _ = owned
    for index in range(2001):
        agent = SimpleNamespace(_current_turn_id=f'wait-{index}', session_id='sid')
        identity = dispatch_identity(agent)
        records['unit'] = {'status': 'running', 'waiting_identity': identity,
                           'parent_session_id': 'sid', 'session_key': 'session'}
        _, guard, ctx = setup(agent)
        adapter._active_sessions['session'] = guard
        classify(agent, 'unit')
        await delivered(adapter, guard, ctx, 'Waiting.', [f'wait-{index}'])
        assert len(adapter._waiting_reply_receipts) == 1
        resume((agent, identity, records, adapter, guard, ctx), f'resolve-{index}')
        await delivered(adapter, guard, ctx, 'Resolved.', [f'final-{index}'])
        assert not adapter._waiting_reply_receipts
        assert adapter.deleted[-1]['message_id'] == f'wait-{index}'


@pytest.mark.asyncio
async def test_live_wait_capture_revoked_by_concurrent_final(owned):
    agent, _, _, adapter, guard, ctx = owned
    classify(agent, 'unit')
    schedule_cleanup({'final_response': 'Waiting.'}, adapter, ctx)
    record_final_ids(guard, SendResult(success=True, message_id='alias',
                                     fresh_message_ids=('alias',)))
    adapter._confirm_final_delivery('session', 1, True)
    other = asyncio.Event()
    other._hermes_run_generation = 1
    other._hermes_final_delivery_ids = set()
    adapter._active_sessions['other-session'] = other
    other_ctx = TurnContext(source=ctx.source, session_key='other-session', run_generation=1,
                            _cleanup_progress=True, _run_still_current=lambda: True)
    other_ctx.agent_holder[0] = SimpleNamespace(_current_turn_id='answer')
    adapter._test_seen_ids = {'alias'}  # the final edits an existing message
    await delivered(adapter, other, other_ctx, 'Genuine final.', ['alias'])
    await adapter._fire_post_delivery_callback('session', guard)
    assert not getattr(adapter, '_waiting_reply_receipts', {})
    assert not adapter.deleted


@pytest.mark.asyncio
async def test_failed_unrelated_final_releases_barrier_for_later_resolution(owned):
    agent, _, _, adapter, guard, ctx = owned
    classify(agent, 'unit')
    await delivered(adapter, guard, ctx, 'Waiting.', ['wait'])
    other = asyncio.Event()
    other._hermes_run_generation = 1
    other._hermes_final_delivery_ids = set()
    adapter._active_sessions['other-session'] = other
    other_ctx = TurnContext(source=ctx.source, session_key='other-session', run_generation=1,
                            _cleanup_progress=True, _run_still_current=lambda: True)
    other_ctx.agent_holder[0] = SimpleNamespace(_current_turn_id='failed')
    schedule_cleanup({'final_response': 'Failed.', 'failed': True}, adapter, other_ctx)
    await adapter._fire_post_delivery_callback('other-session', other)
    resume(owned)
    await delivered(adapter, guard, ctx, 'Resolved.', ['final'])
    assert [item['message_id'] for item in adapter.deleted] == ['wait']


def other_turn(adapter, ctx, name='other-session'):
    guard = asyncio.Event()
    guard._hermes_run_generation = 1
    guard._hermes_final_delivery_ids = set()
    adapter._active_sessions[name] = guard
    other = TurnContext(source=ctx.source, session_key=name, run_generation=1,
                        _cleanup_progress=True, _run_still_current=lambda: True)
    other.agent_holder[0] = SimpleNamespace(_current_turn_id='other')
    return guard, other


@pytest.mark.asyncio
async def test_new_capture_waits_for_concurrent_final_and_alias_is_revoked(owned):
    agent, _, _, adapter, guard, ctx = owned
    other, other_ctx = other_turn(adapter, ctx)
    schedule_cleanup({'final_response': 'Other final.'}, adapter, other_ctx)
    classify(agent, 'unit')
    await delivered(adapter, guard, ctx, 'Waiting.', ['alias', 'safe'])
    resume(owned)
    await delivered(adapter, guard, ctx, 'Resolved.', ['first-final'])
    assert not adapter.deleted
    record_final_ids(other, SendResult(success=True, message_id='alias'))
    await adapter._fire_post_delivery_callback('other-session', other)
    await delivered(adapter, guard, ctx, 'Resolved again.', ['second-final'])
    assert [item['message_id'] for item in adapter.deleted] == ['safe']


@pytest.mark.asyncio
@pytest.mark.parametrize('disposal', ['replace', 'cancel-callback', 'raise-callback'])
async def test_dropped_callback_retains_only_overlapping_receipts(owned, disposal):
    agent, identity, records, adapter, guard, ctx = owned
    classify(agent, 'unit')
    await delivered(adapter, guard, ctx, 'Waiting.', ['uncertain'])
    other, other_ctx = other_turn(adapter, ctx)
    if disposal != 'replace':
        async def broken():
            if disposal == 'cancel-callback':
                raise asyncio.CancelledError()
            raise RuntimeError('callback failed')
        adapter.register_post_delivery_callback('other-session', broken, generation=1)
    schedule_cleanup({'final_response': 'Other.'}, adapter, other_ctx)
    if disposal == 'replace':
        adapter.register_post_delivery_callback('other-session', lambda: None, generation=2)
        assert adapter._waiting_pending_finals  # replacement does not end the old send
    if disposal == 'cancel-callback':
        with pytest.raises(asyncio.CancelledError):
            await adapter._fire_post_delivery_callback('other-session', other)
    else:
        await adapter._fire_post_delivery_callback('other-session', other)
    resume(owned)
    await delivered(adapter, guard, ctx, 'Resolved.', ['final'])
    if disposal == 'raise-callback':
        # Callback isolation continues to the resolution callback.
        assert [x['message_id'] for x in adapter.deleted] == ['uncertain']
    else:
        assert not adapter.deleted
        assert not adapter._waiting_reply_receipts  # resolved uncertainty is retained in chat, not eligible again
    # An unrelated later request remains eligible, without clearing uncertain evidence.
    fresh = SimpleNamespace(_current_turn_id='fresh', session_id='sid')
    new_identity = dispatch_identity(fresh)
    records['unit'] = {'status': 'running', 'waiting_identity': new_identity,
                       'parent_session_id': 'sid', 'session_key': 'session'}
    ctx.agent_holder[0] = fresh
    classify(fresh, 'unit')
    await delivered(adapter, guard, ctx, 'Waiting again.', ['fresh-wait'])
    resume((fresh, new_identity, records, adapter, guard, ctx), 'fresh-resolution')
    await delivered(adapter, guard, ctx, 'Fresh resolved.', ['fresh-final'])
    assert adapter.deleted[-1]['message_id'] == 'fresh-wait'


@pytest.mark.asyncio
async def test_unknown_delivery_receipt_preserves_overlap_but_not_future(owned):
    agent, identity, records, adapter, guard, ctx = owned
    classify(agent, 'unit')
    await delivered(adapter, guard, ctx, 'Waiting.', ['uncertain'])
    other, other_ctx = other_turn(adapter, ctx)
    schedule_cleanup({'final_response': 'Other.'}, adapter, other_ctx)
    other._hermes_final_receipt_uncertain = True
    await adapter._fire_post_delivery_callback('other-session', other)
    resume(owned)
    await delivered(adapter, guard, ctx, 'Resolved.', ['final'])
    assert not adapter.deleted
    assert not adapter._waiting_reply_receipts  # resolved uncertainty is retained in chat, not eligible again


@pytest.mark.asyncio
async def test_cancelling_native_inflight_send_retains_overlap(owned):
    agent, _, _, adapter, guard, ctx = owned
    classify(agent, 'unit')
    await delivered(adapter, guard, ctx, 'Waiting.', ['uncertain'])
    other, other_ctx = other_turn(adapter, ctx)
    started = asyncio.Event()

    async def handler(event):
        schedule_cleanup({'final_response': 'Other.'}, adapter, other_ctx)
        return 'Other.'

    async def send(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    adapter.set_message_handler(handler)
    adapter._send_with_retry = send
    event = MessageEvent(text='question', message_type=MessageType.TEXT,
                         source=ctx.source, message_id='incoming')
    task = asyncio.create_task(adapter._process_message_background(event, 'other-session'))
    await asyncio.wait_for(started.wait(), 2)
    other.set()
    assert adapter._waiting_pending_finals
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not adapter._waiting_pending_finals
    resume(owned)
    await delivered(adapter, guard, ctx, 'Resolved.', ['final'])
    assert not adapter.deleted
    assert not adapter._waiting_reply_receipts  # resolved uncertainty is retained in chat, not eligible again


@pytest.mark.asyncio
async def test_native_send_without_receipt_retains_overlapping_wait(owned):
    agent, _, _, adapter, guard, ctx = owned
    classify(agent, 'unit')
    await delivered(adapter, guard, ctx, 'Waiting.', ['uncertain'])
    _, other_ctx = other_turn(adapter, ctx)

    async def handler(event):
        schedule_cleanup({'final_response': 'Other.'}, adapter, other_ctx)
        return 'Other.'

    async def send(*args, **kwargs):
        return SendResult(success=True, full_payload_delivered=True)

    adapter.set_message_handler(handler)
    adapter._send_with_retry = send
    event = MessageEvent(text='question', message_type=MessageType.TEXT,
                         source=ctx.source, message_id='incoming')
    await adapter._process_message_background(event, 'other-session')
    resume(owned)
    await delivered(adapter, guard, ctx, 'Resolved.', ['final'])
    assert not adapter.deleted
    assert not adapter._waiting_reply_receipts  # resolved uncertainty is retained in chat, not eligible again
