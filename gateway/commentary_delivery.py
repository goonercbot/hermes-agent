"""Turn-owned commentary receipts; deletion requires complete final-body proof."""
from contextlib import suppress


def message_ids(result):
    raw = getattr(result, 'raw_response', None)
    ids = raw.get('message_ids', []) if isinstance(raw, dict) else []
    delivered = raw.get('delivered_message_ids', []) if isinstance(raw, dict) else []
    return list(dict.fromkeys(str(mid) for mid in [getattr(result, 'message_id', None), *ids,
        *(delivered or ()), *(getattr(result, 'continuation_message_ids', None) or ())] if mid is not None))


def track_progress(ctx, result):
    if ctx._cleanup_progress and getattr(result, 'success', False):
        from gateway.config import Platform
        # Telegram supplies creation receipts; other routes retain their existing
        # ordinary-progress behavior, never used to classify a WAITING final.
        ids = (getattr(result, 'fresh_message_ids', None) or ()) if getattr(ctx.source, 'platform', None) == Platform.TELEGRAM else message_ids(result)
        for mid in (str(mid) for mid in ids if mid is not None):
            if mid not in ctx._cleanup_msg_ids:
                ctx._cleanup_msg_ids.append(mid)


def track_commentary(ctx, result, text):
    if ctx._cleanup_progress and getattr(result, 'success', False):
        track_progress(ctx, result)
        ctx._commentary_messages.extend((mid, text) for mid in message_ids(result))


def remember_commentary(ctx, future, text):
    if future is None or not ctx._cleanup_progress:
        return
    ctx._commentary_delivery_futures.append(future)
    def remember(done):
        with suppress(Exception):
            result = done.result()
            track_commentary(ctx, result, text)
    future.add_done_callback(remember)


def schedule_cleanup(response, adapter, ctx):
    import asyncio
    from gateway.waiting_reply_delivery import schedule_waiting_receipt, schedule_waiting_resolution
    schedule_waiting_resolution(response, adapter, ctx)
    schedule_waiting_receipt(response, adapter, ctx)
    if not (ctx._cleanup_progress and adapter is not None and ctx.session_key
            and (ctx._cleanup_msg_ids or ctx._commentary_delivery_futures)
            and isinstance(response, dict) and not response.get('failed')
            and not response.get('partial') and not response.get('interrupted')
            and response.get('completed') is not False
            and hasattr(adapter, 'register_post_delivery_callback')):
        return
    final_text = str(response.get('final_response') or '').strip()
    from gateway.platforms.base import _strip_media_directives
    final_body = _strip_media_directives(final_text).strip()
    delivery_guard = adapter._active_sessions.get(ctx.session_key)
    if delivery_guard is None:
        return
    capture_state = {'chat': str(ctx.source.chat_id), 'revoked': set(), 'generation': ctx.run_generation}
    captures = getattr(delivery_guard, '_waiting_live_captures', None)
    if captures is None:
        captures = delivery_guard._waiting_live_captures = []
    captures.append(capture_state)
    if response.get('_complete_final_body') and final_text:
        adapter._confirm_final_delivery(ctx.session_key, ctx.run_generation, True)
    async def cleanup_body():
        if not adapter._has_confirmed_final_delivery(ctx.session_key, ctx.run_generation):
            return
        for future in ctx._commentary_delivery_futures:
            with suppress(Exception):
                await asyncio.wrap_future(future)
        keep = {mid for mid, text in ctx._commentary_messages
                if str(text).strip() and str(text).strip() in (final_text, final_body)}
        keep.update(getattr(delivery_guard, '_hermes_final_delivery_ids', ()))
        consumer = ctx.stream_consumer_holder[0]
        if consumer:
            keep.update(consumer._final_receipt_ids)
        if consumer and consumer.final_content_delivered:
            keep.update(consumer._segment_preview_message_ids)
            if consumer.message_id:
                keep.add(str(consumer.message_id))
        for mid in dict.fromkeys(ctx._cleanup_msg_ids):
            current = ctx._run_still_current
            if (adapter._active_sessions.get(ctx.session_key) is not delivery_guard
                    or getattr(delivery_guard, '_hermes_run_generation', None) != ctx.run_generation
                    or delivery_guard.is_set() or callable(current) and not current()):
                break
            pending = getattr(adapter, '_waiting_pending_finals', {})
            if capture_state.get('uncertain') or any(
                    state[0] == capture_state['chat'] for state in pending.values()):
                break
            if mid not in keep and mid not in capture_state['revoked']:
                with suppress(Exception):
                    await adapter.delete_message(ctx.source.chat_id, mid)
    async def cleanup():
        if not any(state is capture_state for state in captures) or capture_state.get('started'):
            return
        capture_state['started'] = True
        try:
            await cleanup_body()
        finally:
            captures[:] = [state for state in captures if state is not capture_state]
    adapter.register_post_delivery_callback(ctx.session_key, cleanup, generation=ctx.run_generation)


async def edit_streamed_final(consumer, source, response, content, *, _sk, ok, fail_result, fail_exc):
    import logging
    from gateway.platforms.post_delivery import complete_body
    logger = logging.getLogger(__name__)
    try:
        result = await consumer.adapter.edit_message(chat_id=source.chat_id, message_id=consumer.message_id,
                                                      content=content, finalize=True)
    except Exception as error:
        logger.warning(fail_exc, _sk, error)
        return False
    raw = getattr(result, 'raw_response', None)
    if getattr(result, 'success', False) and not (isinstance(raw, dict) and raw.get('skipped')):
        consumer._track_final_receipt(result)
        if consumer.message_id and consumer.message_id != '__no_edit__':
            consumer._final_receipt_ids.add(str(consumer.message_id))
    if (not getattr(result, 'success', False) or getattr(result, 'full_payload_delivered', None) is False
            or isinstance(raw, dict) and (raw.get('partial_overflow') or raw.get('skipped'))):
        logger.warning(fail_result, _sk, getattr(result, 'error', None))
        return False
    response['already_sent'] = True
    response['_complete_final_body'] = complete_body(result)
    logger.info(*ok)
    return True
