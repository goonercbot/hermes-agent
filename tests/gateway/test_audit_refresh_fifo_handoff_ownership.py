"""Independent focused F1 boundary checks; no candidate edits."""
import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, build_session_key
from gateway.run import GatewayRunner
from tests.gateway.test_audit_refresh_fifo_handoff_integration import Adapter
from tests.gateway.test_queued_followup_processing_hooks import _make_runner


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["valid", "adapter", "key", "equal_event", "owner", "child", "metadata", "malformed"])
async def test_handoff_proof_is_narrow(mode):
    adapter = Adapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
    other = Adapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
    runner = _make_runner(adapter)
    source = SessionSource(platform=Platform.TELEGRAM, chat_type="dm", chat_id="42", user_id="42")
    key = build_session_key(source)
    runner._delivery_adapter_for = lambda source: adapter
    event = MessageEvent(text="head", source=source, message_id="head")
    orphans = [MessageEvent(text=f"tail-{i}", source=source, message_id=f"tail-{i}") for i in range(2)]
    runner._session_state(key).conversation.queued_events.extend(orphans)
    task = asyncio.current_task()
    proof = (adapter, key, event)
    if mode == "adapter": proof = (other, key, event)
    elif mode == "key": proof = (adapter, key + ":foreign", event)
    elif mode == "equal_event":
        clone = replace(event)
        assert clone == event and clone is not event
        proof = (adapter, key, clone)
    elif mode == "malformed": proof = (adapter, key)
    setattr(task, "_hermes_fifo_handoff", proof)
    adapter._session_tasks[key] = object() if mode == "owner" else task
    if mode == "metadata":
        delattr(task, "_hermes_fifo_handoff")
        event.metadata = {"_hermes_fifo_handoff": proof, "fifo_handoff": True}
    def check():
        result = runner._hm_rescue_orphaned_fifo(event, source, False, key)
        if mode == "valid":
            assert result[0] is event
            assert key not in adapter._pending_messages
            assert runner._overflow_queue(key) == orphans
        else:
            assert result[0] is orphans[0]
            assert adapter._pending_messages[key] is orphans[1]
            assert runner._overflow_queue(key) == [event]
    try:
        if mode == "child":
            async def child(): check()
            await asyncio.create_task(child())
        else:
            check()
    finally:
        if hasattr(task, "_hermes_fifo_handoff"):
            delattr(task, "_hermes_fifo_handoff")
        adapter._session_tasks.clear()
