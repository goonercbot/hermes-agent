"""Red regression: a classified waiting final must have a delivery receipt owner."""
import asyncio
import json
from types import SimpleNamespace

from gateway.commentary_delivery import schedule_cleanup
from gateway.turn_context import TurnContext
from tools import async_delegation
from tools.delegate_waiting_reply import dispatch_identity
from tools.registry import registry
import tools.delegate_tool  # register the native delegate_task handler


def test_classified_wait_registers_post_delivery_receipt_callback():
    agent = SimpleNamespace(_current_turn_id="turn-1")
    identity = dispatch_identity(agent)
    unit = "test-wait-receipt-unit"
    with async_delegation._records_lock:
        async_delegation._records[unit] = {"status": "running", "waiting_identity": identity}
    try:
        result = json.loads(registry.dispatch("delegate_task", {
            "action": "classify_waiting_reply", "delegation_ids": [unit],
        }, parent_agent=agent))
        assert result["status"] == "classified"
        from tests.gateway.test_run_cleanup_progress import CleanupCaptureAdapter
        adapter = CleanupCaptureAdapter()
        guard = asyncio.Event()
        guard._hermes_run_generation = 1
        guard._hermes_final_delivery_ids = set()
        adapter._active_sessions['session'] = guard
        ctx = TurnContext(source=SimpleNamespace(chat_id="chat"), session_key="session",
                          run_generation=1, _cleanup_progress=True, _run_still_current=lambda: True)
        ctx.agent_holder[0] = agent
        schedule_cleanup({"final_response": "Waiting for the delegated work."}, adapter, ctx)
        assert adapter.pop_post_delivery_callback('session', generation=1), (
            "classified waiting final has no post-delivery receipt callback")
    finally:
        with async_delegation._records_lock:
            async_delegation._records.pop(unit, None)
