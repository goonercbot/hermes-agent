"""Persistent, non-final model notices with delivery receipts and destination checks."""
import asyncio
import logging

from agent.model_switch import bind_model_switch_state, flush_model_switch_notices
from gateway.bot_peer_notices import automatic_notices_allowed

logger = logging.getLogger(__name__)


async def deliver_model_switch(adapter, source, text, metadata, still_current, *, chat_id=None):
    """Never seal an open draft; verify ownership at the actual send, not just scheduling."""
    if not still_current() or not automatic_notices_allowed(source) or not adapter or not source.chat_id:
        return False
    from gateway.run import _interim_metadata, _redact_gateway_user_facing_secrets
    try:
        receipt = await adapter.send(chat_id or source.chat_id, _redact_gateway_user_facing_secrets(str(text)),
                                     metadata=_interim_metadata(metadata))
    except Exception as exc:
        logger.exception("Model-switch delivery failed (%s); retained for retry", type(exc).__name__, exc_info=False)
        return False
    if getattr(receipt, "success", None) is not True:
        logger.warning("Model-switch delivery was not acknowledged; retained for retry")
        return False
    return True


def bind_compression_notice_sink(gateway, agent, source, entry, metadata=None):
    """Used by detached hygiene and /compress, whose agents have no ordinary chat callback."""
    bind_model_switch_state(agent, entry)
    loop = asyncio.get_running_loop()
    key = entry.session_key
    # The entry's identity survives in-place/child compression, but not /new or route replacement.
    def current():
        return gateway.session_store._entries.get(key) is entry
    def send(text):
        return asyncio.run_coroutine_threadsafe(
            deliver_model_switch(gateway._delivery_adapter_for(source), source, text, metadata, current), loop)
    agent._model_switch_callback = send
    flush_model_switch_notices(agent)
