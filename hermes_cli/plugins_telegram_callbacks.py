"""Compatibility registrar for authorization-gated Telegram callback prefixes."""
import inspect
import logging
from contextlib import suppress

logger = logging.getLogger(__name__)


def register_callback(ctx, prefix, callback):
    if not isinstance(prefix, str) or not prefix:
        raise ctx._refuse("a Telegram callback handler with an empty prefix")
    if not callable(callback):
        raise ctx._refuse("a Telegram callback handler with a non-callable callback")
    manager = ctx._manager
    entry = (prefix, callback, ctx.manifest.name)
    revoked = False

    def release():
        nonlocal revoked
        with manager._telegram_callback_lock:
            revoked = True
            manager._remove_identity(manager._telegram_callback_handlers, entry)

    # Do not hold the registry lock across tracking: timeout rollback must be
    # able to revoke a handle even while its registering worker is paused.
    registration = ctx._track("telegram_callback_handler", prefix, release)
    with manager._telegram_callback_lock:
        abandoned = ctx._load_abandoned
        if not abandoned and not revoked and registration.active:
            manager._telegram_callback_handlers.append(entry)
            _publish_factory(manager)
    if abandoned:
        # Tracking itself may have finished after the timeout's ledger sweep.
        registration.dispose()
    return registration


def _publish_factory(manager):
    """Publish the shared factory while holding the callback registry lock."""
    factory = manager._telegram_callback_dispatch_factory
    if factory is None:
        def factory(application, adapter):
            from telegram.ext import CallbackQueryHandler

            class PrefixCallbackHandler(CallbackQueryHandler):
                # PTB matches queries with neither data nor game_short_name even when
                # a pattern is set; those must remain available to core handlers.
                def check_update(self, update):
                    query = getattr(update, "callback_query", None)
                    data = getattr(query, "data", None)
                    if not isinstance(data, str) or not data:
                        return False
                    return super().check_update(update)

            def matches(data):
                return isinstance(data, str) and any(data.startswith(item[0])
                    for item in manager.get_telegram_callback_handlers())

            async def dispatch(update, context):
                query = getattr(update, "callback_query", None)
                data = getattr(query, "data", None)
                if query is None or not isinstance(data, str):
                    return
                selected = next((item for item in manager.get_telegram_callback_handlers()
                                 if data.startswith(item[0])), None)
                if selected is None:
                    return
                selected_prefix, selected_callback, plugin_name = selected
                message = getattr(query, "message", None)
                chat = getattr(message, "chat", None)
                user = getattr(query, "from_user", None)
                thread = getattr(message, "message_thread_id", None)
                if not adapter._is_callback_user_authorized(
                    str(getattr(user, "id", None) or ""), chat_id=getattr(message, "chat_id", None),
                    chat_type=str(getattr(chat, "type", "")) or None,
                    thread_id=str(thread) if thread is not None else None,
                    user_name=getattr(user, "first_name", None),
                ):
                    await query.answer(text="⛔ You are not authorized to use this button.", show_alert=True)
                    return
                try:
                    result = selected_callback(update=update, query=query, adapter=adapter)
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    logger.error("Plugin %s Telegram callback handler failed for prefix %s",
                                 plugin_name, selected_prefix, exc_info=True)
                    with suppress(Exception):
                        await query.answer(text="This action failed. Please try again.", show_alert=True)

            application.add_handler(PrefixCallbackHandler(dispatch, pattern=matches))
        manager._telegram_callback_dispatch_factory = factory
    factories = manager._platform_handler_factories.setdefault("telegram", [])
    if not any(registered is factory for registered, _ in factories):
        factories.append((factory, "hermes.callback-dispatch"))
