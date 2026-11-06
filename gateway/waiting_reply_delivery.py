"""Capture and resolve explicitly typed waiting finals in adapter-local memory."""
from collections import OrderedDict

# Eviction discards metadata only. Large individual receipts are not recorded at all.
_MAX_RECORDS = 256
_MAX_IDS = 256
_MAX_UNITS = 256


def schedule_waiting_receipt(response, adapter, ctx):
    from tools.delegate_waiting_reply import waiting_reply

    if not (ctx._cleanup_progress and adapter is not None and ctx.session_key
            and not getattr(adapter, '_waiting_receipts_disabled', False)
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
    if not (waiting and waiting.delegation_ids and len(waiting.delegation_ids) <= _MAX_UNITS
            and all(len(unit) <= 1024 for unit in waiting.delegation_ids)
            and len(waiting.root_request_id) <= 1024
            and isinstance(ctx.session_key, str) and len(ctx.session_key) <= 1024
            and chat is not None and 0 < len(str(chat)) <= 1024
            and len(str(getattr(source, 'thread_id', '') or '')) <= 1024
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

    async def capture():
        if (getattr(adapter, '_waiting_receipts_disabled', False)
                or adapter._active_sessions.get(ctx.session_key) is not guard
                or getattr(guard, '_hermes_run_generation', None) != ctx.run_generation
                or guard.is_set() or not current() or waiting_reply(agent) != waiting
                or getattr(guard, '_hermes_nontext_delivery_attempted', False)
                or not adapter._has_confirmed_final_delivery(ctx.session_key, ctx.run_generation)):
            return
        # The streamed proof is established by the consumer before scheduling; normal
        # sends are confirmed by the adapter after the handler returns.
        if streamed:
            ids = getattr(consumer, '_final_receipt_ids', ()) if consumer is not None else ()
        else:
            ids = getattr(guard, '_hermes_final_delivery_ids', ())
        ids = frozenset(ids)
        if (not ids or len(ids) > _MAX_IDS
                or any(not isinstance(mid, str) or not 0 < len(mid) <= 1024 for mid in ids)):
            return
        protected = getattr(adapter, '_waiting_promoted_ids', set())
        ids = frozenset(mid for mid in ids if (str(chat), mid) not in protected)
        if not ids:
            return
        ledger = getattr(adapter, '_waiting_reply_receipts', None)
        if ledger is None:
            ledger = adapter._waiting_reply_receipts = OrderedDict()
        previous = ledger.get(key)
        if previous:
            ids = ids | previous['message_ids']
            units = tuple(dict.fromkeys((*previous['delegation_ids'], *waiting.delegation_ids)))
            if len(ids) > _MAX_IDS or len(units) > _MAX_UNITS:
                # Lost completeness means none of these receipts are safe to delete.
                ledger.pop(key, None)
                return
        else:
            units = waiting.delegation_ids
        ledger[key] = {'message_ids': ids, 'delegation_ids': units}
        if previous and previous.get('_pending_finals'):
            ledger[key]['_pending_finals'] = previous['_pending_finals'].copy()
        ledger.move_to_end(key)
        while len(ledger) > _MAX_RECORDS:
            ledger.popitem(last=False)

    adapter.register_post_delivery_callback(ctx.session_key, capture, generation=ctx.run_generation)


def schedule_waiting_resolution(response, adapter, ctx):
    """Protect delivered finals first; delete only with an explicit, still-owned resolution."""
    from tools.delegate_waiting_reply import waiting_reply, resolving_reply, completed_owned

    if (adapter is None or not ctx.session_key or ctx.run_generation is None
            or not callable(getattr(adapter, 'register_post_delivery_callback', None))
            or not getattr(adapter, '_waiting_reply_receipts', None)):
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
    ledger = adapter._waiting_reply_receipts
    for key, record in list(ledger.items()):
        if key[1] == route[1]:
            pending = record.setdefault('_pending_finals', set())
            if len(pending) >= _MAX_IDS:
                ledger.pop(key, None)  # Unknown outstanding finals: retain messages, forget eligibility.
            else:
                pending.add(token)
    streamed = isinstance(response, dict) and response.get('_complete_final_body') is True
    if streamed and consumer is not None and getattr(consumer, '_final_receipt_ids', None):
        adapter._confirm_final_delivery(ctx.session_key, ctx.run_generation, True)

    async def cleanup():
        if getattr(adapter, '_waiting_reply_receipts', None) is not ledger:
            return
        final_ids = set(getattr(guard, '_hermes_final_delivery_ids', ()))
        stream_ids = set(getattr(consumer, '_final_receipt_ids', ())) if consumer is not None else set()
        keep = final_ids | stream_ids
        if consumer is not None:
            keep.update(getattr(consumer, '_segment_preview_message_ids', ()))
            if getattr(consumer, 'message_id', None):
                keep.add(str(consumer.message_id))
        # Even an incomplete or nontext genuine final must never remain eligible.
        promoted = {(route[1], mid) for key, record in list(ledger.items()) if key[1] == route[1]
                    for mid in keep.intersection(record['message_ids'])}
        protected = getattr(adapter, '_waiting_promoted_ids', set())
        if len(protected | promoted) > _MAX_IDS:
            adapter._waiting_receipts_disabled = True
            ledger.clear()
            return
        if promoted:
            adapter._waiting_promoted_ids = protected | promoted
        for key, record in list(ledger.items()):
            if key[1] == route[1] and keep.intersection(record['message_ids']):
                record['message_ids'] = record['message_ids'] - keep
                if not record['message_ids']:
                    ledger.pop(key, None)

        valid_guard = (adapter._active_sessions.get(ctx.session_key) is guard
                       and getattr(guard, '_hermes_run_generation', None) == ctx.run_generation
                       and not guard.is_set() and callable(current) and current())
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
        if valid_guard and complete:
            for key, record in list(ledger.items()):
                if key[1] == route[1] and '_pending_finals' in record:
                    record['_pending_finals'].discard(token)
                    if not record['_pending_finals']:
                        del record['_pending_finals']

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
        if not record or record.get('_pending_finals'):
            return
        for mid in tuple(record['message_ids']):
            if not owned() or ledger.get(key) is not record or record.get('_pending_finals'):
                return
            units = tuple(dict.fromkeys((*resolution.delegation_ids, *record['delegation_ids'])))
            if not completed_owned(resolution.identity, units):
                return
            try:
                deleted = await adapter.delete_message(chat, mid)
            except Exception:
                continue
            if deleted and ledger.get(key) is record:
                record['message_ids'] = record['message_ids'] - {mid}
                if not record['message_ids']:
                    ledger.pop(key, None)
                    return

    adapter.register_post_delivery_callback(ctx.session_key, cleanup, generation=ctx.run_generation)
