"""Capture proven, turn-owned waiting finals in adapter-local memory; never delete."""
from collections import OrderedDict

# Eviction discards metadata only. Large individual receipts are not recorded at all.
_MAX_RECORDS = 256
_MAX_IDS = 256
_MAX_UNITS = 256


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
        if (adapter._active_sessions.get(ctx.session_key) is not guard
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
        ledger = getattr(adapter, '_waiting_reply_receipts', None)
        if ledger is None:
            ledger = adapter._waiting_reply_receipts = OrderedDict()
        ledger[key] = {'message_ids': ids, 'delegation_ids': waiting.delegation_ids}
        ledger.move_to_end(key)
        while len(ledger) > _MAX_RECORDS:
            ledger.popitem(last=False)

    adapter.register_post_delivery_callback(ctx.session_key, capture, generation=ctx.run_generation)
