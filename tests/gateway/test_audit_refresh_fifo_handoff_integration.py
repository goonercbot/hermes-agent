"""Real recursive turns, adapter drain handoff and inbound FIFO rescue.

Model, setup and delivery are isolated; the queue and gateway lifecycle are real.
"""
import asyncio
from unittest.mock import AsyncMock

import pytest
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key

class Adapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect=False): return True
    async def disconnect(self): pass
    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise AssertionError("No platform calls permitted in review probe")
    async def get_chat_info(self, chat_id): return {"id": chat_id, "type": "dm"}

@pytest.mark.asyncio
@pytest.mark.parametrize("count,mixed", [(1, True), (4, True), (8, False), (10, False), (8, True), (10, True)])
async def test_real_handoff_preserves_fifo(count, mixed, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    from tests.gateway.test_queued_followup_processing_hooks import _make_runner, _install_fake_agent
    a = Adapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
    r = _make_runner(a)
    r._draining = False
    r._external_drain_active = False
    s = SessionSource(platform=Platform.TELEGRAM, chat_type="dm", chat_id="42", user_id="42")
    key = build_session_key(s)
    r._delivery_adapter_for = lambda source: a
    r._intake_adapter_for = lambda source: None
    r._session_key_for_source = lambda source: key
    r._hm_admit_event = AsyncMock(side_effect=lambda event: (event, event.source, event.internal))
    r._hm_estop_gate = lambda *args: None
    r._hm_pending_reply_intercepts = AsyncMock(return_value=None)
    r._hm_evict_idle_stale_agent = lambda *args: None
    r._hm_evict_reaped_agent = lambda *args: None
    r._hm_dispatch_idle_commands = AsyncMock(return_value=(False,None))
    r._is_telegram_topic_root_lobby = lambda *args: False
    r._claim_active_session_slot = lambda *args: (None,None)
    r._persist_active_agents = lambda: None
    r._begin_session_run_generation = lambda *args: 1
    r._restore_pending_one_turn_model_override = lambda *args: None
    r._clear_durable_active_turn = AsyncMock()
    r._release_turn_lease = lambda *args: None
    def release(k, **kwargs): r._session_state(k).turn.agent = None
    r._release_running_agent_state = release
    r._run_post_turn_hooks = AsyncMock()
    r._run_agent_deliver_first_response = AsyncMock()
    r._refresh_agent_cache_message_count = AsyncMock()
    r._prepare_profile_scoped_inbound_message_text = AsyncMock(side_effect=lambda **kw: kw["event"].text)
    r._reply_anchor_for_event = lambda event: event.message_id
    a._start_typing_refresh = lambda *args, **kw: None
    a._stop_typing_refresh = AsyncMock()
    a._release_turn_marker = AsyncMock()
    a.send_typing = AsyncMock()
    events = [MessageEvent(text=f"notice-{i}", source=s, message_id=f"id-{i}",
                          internal=not mixed or i%2==0,
                          metadata={"index":i}, message_type=MessageType.TEXT)
              for i in range(count)]
    by_id = {e.message_id:e for e in events}
    root = MessageEvent(text="root", source=s, message_id="root", internal=True)
    by_id["root"] = root
    consumed, handoffs, rescues = [], [], []
    original_rescue = r._hm_rescue_orphaned_fifo
    def rescue(event, source, is_internal, k):
        out = original_rescue(event,source,is_internal,k)
        if out[0] is not event: rescues.append((event.message_id,out[0].message_id))
        return out
    r._hm_rescue_orphaned_fifo = rescue
    by_text = {e.text:e for e in [root,*events]}
    class FakeAgent:
        def __init__(self, **kwargs): self.tools = []
        def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
            event = by_text[message]
            if event is not root: consumed.append(event)
            return {"final_response":"", "messages":[], "api_calls":1}
    _install_fake_agent(monkeypatch, tmp_path, FakeAgent)
    async def inner(event, source, k, generation):
        await r._run_agent(message=event.text,context_prompt="",history=[],source=source,
                           session_id="review",session_key=k,run_generation=generation,
                           inbound_message_id=event.message_id)
        return None
    r._handle_message_with_agent = inner
    async def handler(event):
        handoffs.append(event.message_id)
        assert event is by_id[event.message_id]
        if event is not root:
            index = int(event.message_id.removeprefix("id-"))
            assert event.metadata == {"index": index}
            assert event.internal is (not mixed or index % 2 == 0)
        return await r._handle_message(event)
    a._message_handler = handler
    for e in events: r._enqueue_fifo(key,e,a)
    assert a._start_session_processing(root,key)
    async def settle():
        while a._background_tasks:
            await asyncio.gather(*list(a._background_tasks))
            await asyncio.sleep(0)
    try:
        await asyncio.wait_for(settle(),timeout=10)
        actual = [e.message_id for e in consumed]
        expected = [e.message_id for e in events]
        print({"count":count,"mixed":mixed,"actual":actual,"handoffs":handoffs,"rescues":rescues})
        assert len(consumed)==count
        assert len({id(e) for e in consumed})==count
        assert not r._overflow_queue(key)
        assert key not in a._pending_messages
        assert key not in a._active_sessions
        assert actual==expected
    finally:
        await a.cancel_background_tasks()


@pytest.mark.asyncio
async def test_new_arrival_metadata_cannot_claim_handoff_and_orphans_are_rescued():
    from tests.gateway.test_queued_followup_processing_hooks import _make_runner

    adapter = Adapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
    runner = _make_runner(adapter)
    source = SessionSource(platform=Platform.TELEGRAM, chat_type="dm", chat_id="42", user_id="42")
    key = build_session_key(source)
    runner._delivery_adapter_for = lambda source: adapter
    orphan = MessageEvent(text="old", source=source, message_id="old")
    incoming = MessageEvent(text="new", source=source, message_id="new",
                            metadata={"_hermes_fifo_handoff": True})
    runner._session_state(key).conversation.queued_events.append(orphan)
    rescued, rescued_source, is_internal = runner._hm_rescue_orphaned_fifo(
        incoming, source, False, key)
    assert rescued is orphan
    assert rescued_source is source
    assert not is_internal
    assert adapter._pending_messages[key] is incoming


@pytest.mark.asyncio
async def test_handoff_does_not_bypass_admission():
    from tests.gateway.test_queued_followup_processing_hooks import _make_runner

    adapter = Adapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
    runner = _make_runner(adapter)
    source = SessionSource(platform=Platform.TELEGRAM, chat_type="dm", chat_id="42", user_id="42")
    key = build_session_key(source)
    event = MessageEvent(text="denied", source=source, message_id="denied")
    runner._delivery_adapter_for = lambda source: adapter
    runner._session_key_for_source = lambda source: key
    runner._hm_admit_event = AsyncMock(return_value=None)
    task = asyncio.current_task()
    assert task is not None
    setattr(task, "_hermes_fifo_handoff", (adapter, key, event))
    adapter._session_tasks[key] = task
    try:
        assert await runner._handle_message(event) is None
        runner._hm_admit_event.assert_awaited_once_with(event)
        assert not hasattr(task, "_hermes_fifo_handoff")
    finally:
        if hasattr(task, "_hermes_fifo_handoff"):
            delattr(task, "_hermes_fifo_handoff")
        adapter._session_tasks.pop(key)
