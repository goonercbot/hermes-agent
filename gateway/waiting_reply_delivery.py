"""Capture and resolve explicitly typed waiting finals in adapter-local memory."""
from collections import OrderedDict


def finish_pending_final(adapter, token, *, uncertain=False):
    """Retire one delivery barrier; unknown disposal taints only overlapping receipts."""
    pending = getattr(adapter, '_waiting_pending_finals', {})
    state = pending.pop(token, None)
    if state is None:
        return
    chat, guard, consumer, _generation = state
    keep = set(getattr(guard, '_hermes_final_delivery_ids', ()))
    if consumer is not None:
        keep.update(getattr(consumer, '_final_receipt_ids', ()))
        keep.update(getattr(consumer, '_segment_preview_message_ids', ()))
        if getattr(consumer, 'message_id', None):
            keep.add(str(consumer.message_id))
    for live_guard in tuple(adapter._active_sessions.values()):
        for capture in getattr(live_guard, '_waiting_live_captures', ()):
            if capture['chat'] == chat:
                capture['revoked'].update(keep)
                if uncertain:
                    capture['uncertain'] = True
    ledger = getattr(adapter, '_waiting_reply_receipts', {})
    for key, record in list(ledger.items()):
        if key[1] != chat:
            continue
        record['message_ids'] = record['message_ids'] - keep
        if uncertain:
            record.setdefault('_uncertain_ids', set()).update(record['message_ids'])
        if not record['message_ids']:
            ledger.pop(key, None)


def dispose_pending_finals(adapter, guard, generation):
    """The native delivery task ended without running its registered callback."""
    for token, state in list(getattr(adapter, '_waiting_pending_finals', {}).items()):
        if state[1] is guard and state[3] == generation:
            finish_pending_final(adapter, token, uncertain=True)
    captures = getattr(guard, '_waiting_live_captures', [])
    captures[:] = [capture for capture in captures if capture.get('generation') != generation]


def schedule_waiting_receipt(response, adapter, ctx):
    from tools.delegate_waiting_reply import waiting_reply

    if not (ctx._cleanup_progress and adapter is not None and ctx.session_key
            and ctx.run_generation is not None and isinstance(response, dict)
            and not response.get('failed') and not response.get('partial')
            and not response.get('interrupted') and response.get('completed') is not False
            and callable(getattr(adapter, 'register_post_delivery_callback', None))
            and callable(getattr(adapter, '_has_confirmed_final_delivery', None))):
        return
    agent = ctx.agent_holder[0]
    waiting = waiting_reply(agent)
    source = ctx.source
    chat = getattr(source, 'chat_id', None)
    text = response.get('final_response')
    if not (waiting and waiting.delegation_ids
            and isinstance(ctx.session_key, str) and ctx.session_key
            and chat is not None
            and isinstance(text, str) and text.strip()):
        return
    # There is no complete attachment receipt. Reject potential attachment-bearing replies,
    # including directives that the delivery extractor could otherwise strip or filter.
    from gateway.platforms.base import BasePlatformAdapter
    if ('MEDIA:' in text.upper() or '[[audio_as_voice]]' in text.lower()
            or '[[as_document]]' in text.lower()):
        return
    images, _ = BasePlatformAdapter.extract_images(text)
    files, _ = BasePlatformAdapter.extract_local_files(text)
    if images or files:
        return
    guard = adapter._active_sessions.get(ctx.session_key)
    current = ctx._run_still_current
    if (guard is None or getattr(guard, '_hermes_run_generation', None) != ctx.run_generation
            or not callable(current) or not current() or guard.is_set()):
        return
    thread = getattr(source, 'thread_id', None)
    key = (ctx.session_key, str(chat), None if thread is None else str(thread), waiting.root_request_id)
    consumer = ctx.stream_consumer_holder[0]
    streamed = response.get('_complete_final_body') is True
    if streamed:
        if (consumer is None or not getattr(consumer, '_final_receipt_ids', None)
                or not callable(getattr(adapter, '_confirm_final_delivery', None))):
            return
        adapter._confirm_final_delivery(ctx.session_key, ctx.run_generation, True)

    # A live capture owns its revocations until its callback runs.  This is not
    # an adapter-lifetime history of final IDs.
    capture_state = {'chat': str(chat), 'revoked': set(), 'generation': ctx.run_generation}
    captures = getattr(guard, '_waiting_live_captures', None)
    if captures is None:
        captures = guard._waiting_live_captures = []
    captures.append(capture_state)

    async def capture():
        if not any(state is capture_state for state in captures):
            return
        captures[:] = [state for state in captures if state is not capture_state]
        if (adapter._active_sessions.get(ctx.session_key) is not guard
                or getattr(guard, '_hermes_run_generation', None) != ctx.run_generation
                or guard.is_set() or not current() or waiting_reply(agent) != waiting
                or getattr(guard, '_hermes_nontext_delivery_attempted', False)
                or not adapter._has_confirmed_final_delivery(ctx.session_key, ctx.run_generation)):
            return
        # The streamed proof is established by the consumer before scheduling; normal
        # sends are confirmed by the adapter after the handler returns.
        if streamed:
            ids = getattr(consumer, '_final_fresh_message_ids', ()) if consumer is not None else ()
        else:
            ids = getattr(guard, '_hermes_final_fresh_message_ids', ())
        ids = frozenset(mid for mid in ids if isinstance(mid, str) and mid) - capture_state['revoked']
        if not ids:
            return
        ledger = getattr(adapter, '_waiting_reply_receipts', None)
        if ledger is None:
            ledger = adapter._waiting_reply_receipts = OrderedDict()
        previous = ledger.get(key)
        if previous:
            ids = ids | previous['message_ids']
            units = tuple(dict.fromkeys((*previous['delegation_ids'], *waiting.delegation_ids)))

        else:
            units = waiting.delegation_ids
        ledger[key] = {'message_ids': ids, 'delegation_ids': units}
        uncertain_ids = set(previous.get('_uncertain_ids', ())) if previous else set()
        if capture_state.get('uncertain'):
            uncertain_ids.update(ids)
        if uncertain_ids:
            ledger[key]['_uncertain_ids'] = uncertain_ids
        ledger.move_to_end(key)


    adapter.register_post_delivery_callback(ctx.session_key, capture, generation=ctx.run_generation)


def schedule_waiting_resolution(response, adapter, ctx):
    """Protect delivered finals first; delete only with an explicit, still-owned resolution."""
    from tools.delegate_waiting_reply import waiting_reply, resolving_reply, completed_owned

    if (adapter is None or not ctx.session_key or ctx.run_generation is None
            or not callable(getattr(adapter, 'register_post_delivery_callback', None))):
        return
    agent = ctx.agent_holder[0]
    if waiting_reply(agent):
        return
    resolution = resolving_reply(agent)
    guard = adapter._active_sessions.get(ctx.session_key)
    if guard is None:
        return
    source = ctx.source
    chat = getattr(source, 'chat_id', None)
    thread = getattr(source, 'thread_id', None)
    route = (ctx.session_key, str(chat), None if thread is None else str(thread))
    current = ctx._run_still_current
    consumer = ctx.stream_consumer_holder[0]
    token = object()
    ledger = getattr(adapter, '_waiting_reply_receipts', None)
    if ledger is None:
        ledger = adapter._waiting_reply_receipts = OrderedDict()
    pending = getattr(adapter, '_waiting_pending_finals', None)
    if pending is None:
        pending = adapter._waiting_pending_finals = {}
    pending[token] = (route[1], guard, consumer, ctx.run_generation)
    streamed = isinstance(response, dict) and response.get('_complete_final_body') is True
    if streamed and consumer is not None and getattr(consumer, '_final_receipt_ids', None):
        adapter._confirm_final_delivery(ctx.session_key, ctx.run_generation, True)

    async def cleanup():
        finish_pending_final(adapter, token, uncertain=(
            getattr(guard, '_hermes_final_receipt_uncertain', False)
            or getattr(guard, '_hermes_delivery_in_flight', False)))
        final_ids = set(getattr(guard, '_hermes_final_delivery_ids', ()))
        stream_ids = set(getattr(consumer, '_final_receipt_ids', ())) if consumer is not None else set()
        keep = final_ids | stream_ids
        if consumer is not None:
            keep.update(getattr(consumer, '_segment_preview_message_ids', ()))
            if getattr(consumer, 'message_id', None):
                keep.add(str(consumer.message_id))
        # Even an incomplete or nontext genuine final must never remain eligible.
        for live_guard in tuple(adapter._active_sessions.values()):
            for capture in getattr(live_guard, '_waiting_live_captures', ()):
                if capture['chat'] == route[1]:
                    capture['revoked'].update(keep)
        if getattr(adapter, '_waiting_reply_receipts', None) is not ledger:
            return
        for key, record in list(ledger.items()):
            if key[1] == route[1] and keep.intersection(record['message_ids']):
                record['message_ids'] = record['message_ids'] - keep
                if not record['message_ids']:
                    ledger.pop(key, None)

        text = response.get('final_response') if isinstance(response, dict) else None
        from gateway.platforms.base import BasePlatformAdapter
        text_only = (isinstance(text, str) and bool(text.strip()) and 'MEDIA:' not in text.upper()
                     and '[[audio_as_voice]]' not in text.lower() and '[[as_document]]' not in text.lower()
                     and not BasePlatformAdapter.extract_images(text)[0]
                     and not BasePlatformAdapter.extract_local_files(text)[0])
        complete = (isinstance(response, dict) and not response.get('failed')
                    and not response.get('partial') and not response.get('interrupted')
                    and response.get('completed') is not False and text_only
                    and bool(stream_ids if streamed else final_ids)
                    and not getattr(guard, '_hermes_nontext_delivery_attempted', False)
                    and adapter._has_confirmed_final_delivery(ctx.session_key, ctx.run_generation))
        def owned():
            return (adapter._active_sessions.get(ctx.session_key) is guard
                    and getattr(guard, '_hermes_run_generation', None) == ctx.run_generation
                    and not guard.is_set() and callable(current) and current()
                    and resolving_reply(agent) == resolution
                    and getattr(adapter, '_waiting_reply_receipts', None) is ledger)

        if not (ctx._cleanup_progress and resolution and owned() and complete):
            return
        key = (*route, resolution.identity.root_request_id)
        record = ledger.get(key)
        def blocked():
            return any(state[0] == route[1] for state in pending.values())

        if not record or blocked():
            return
        for mid in tuple(record['message_ids']):
            if not owned() or ledger.get(key) is not record or blocked():
                return
            units = tuple(dict.fromkeys((*resolution.delegation_ids, *record['delegation_ids'])))
            if not completed_owned(resolution.identity, units):
                return
            if mid not in record['message_ids'] or mid in record.get('_uncertain_ids', ()):
                continue
            try:
                deleted = await adapter.delete_message(chat, mid)
            except Exception:
                continue
            if deleted and ledger.get(key) is record:
                record['message_ids'] = record['message_ids'] - {mid}
                if not record['message_ids']:
                    ledger.pop(key, None)
                    return
        # After an owned resolution, uncertainty means retain the messages, not
        # retain deletion eligibility forever. Failed deletes remain retryable.
        if owned() and ledger.get(key) is record and not blocked():
            uncertain = record.pop('_uncertain_ids', set())
            record['message_ids'] = record['message_ids'] - uncertain
            if not record['message_ids']:
                ledger.pop(key, None)

    adapter.register_post_delivery_callback(ctx.session_key, cleanup, generation=ctx.run_generation)
