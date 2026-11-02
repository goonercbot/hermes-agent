"""F1 integration through real admission and supported dispatch rewrites."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key
from tests.gateway.test_audit_refresh_fifo_handoff_integration import Adapter
from tests.gateway.test_queued_followup_processing_hooks import _make_runner, _install_fake_agent


@pytest.mark.asyncio
@pytest.mark.parametrize("count,rewrite", [(8, False), (10, False), (8, True), (10, True)])
async def test_real_admission_fifo(count, rewrite, tmp_path, monkeypatch):
    import hermes_cli.lifecycle
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    a = Adapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
    r = _make_runner(a)
    r._draining = False
    r._external_drain_active = False
    s = SessionSource(platform=Platform.TELEGRAM, chat_type="dm", chat_id="42", user_id="42")
    key = build_session_key(s)
    r._delivery_adapter_for = lambda source: a
    r._intake_adapter_for = lambda source: None
    r._session_key_for_source = lambda source: key
    # Exercise production _hm_admit_event and _hm_pre_gateway_dispatch_hook.
    # Only authorization and the hook's external result are fixture inputs.
    r._scale_to_zero_note_real_inbound = lambda: None
    r._is_user_authorized_for_source = lambda source: True
    r._admit_bot_message_for_source = lambda source: True
    hook_calls = []
    async def hook(name, **kwargs):
        assert name == "pre_gateway_dispatch"
        hook_calls.append(kwargs["event"].message_id)
        return [{"action": "rewrite", "text": kwargs["event"].text.upper()}] if rewrite else []
    monkeypatch.setattr(hermes_cli.lifecycle, "ainvoke_hook", hook)
    r._hm_estop_gate = lambda *args: None
    r._hm_pending_reply_intercepts = AsyncMock(return_value=None)
    r._hm_evict_idle_stale_agent = lambda *args: None
    r._hm_evict_reaped_agent = lambda *args: None
    r._hm_dispatch_idle_commands = AsyncMock(return_value=(False, None))
    r._is_telegram_topic_root_lobby = lambda *args: False
    r._claim_active_session_slot = lambda *args: (None, None)
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
                          internal=i % 2 == 0, metadata={"index": i}, message_type=MessageType.TEXT)
              for i in range(count)]
    root = MessageEvent(text="root", source=s, message_id="root", internal=True)
    by_text = {text: e for e in [root, *events] for text in (e.text, e.text.upper())}
    consumed, handoffs, rescues = [], [], []
    original_rescue = r._hm_rescue_orphaned_fifo
    def rescue(event, source, is_internal, k):
        if event.message_id and event.message_id.startswith("id-"):
            original = events[int(event.message_id.removeprefix("id-"))]
            from dataclasses import fields
            for field in fields(MessageEvent):
                # replace() deliberately reconstructs init=False bookkeeping.
                if field.init and field.name != "text":
                    assert getattr(event, field.name) == getattr(original, field.name)
            assert event.source is original.source
            if rewrite and not original.internal:
                assert event is not original
                assert event.text == original.text.upper()
        out = original_rescue(event, source, is_internal, k)
        if out[0] is not event: rescues.append((event.message_id, out[0].message_id))
        return out
    r._hm_rescue_orphaned_fifo = rescue
    class FakeAgent:
        def __init__(self, **kwargs): self.tools = []
        def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
            event = by_text[message]
            if event is not root: consumed.append(event.message_id)
            return {"final_response": "", "messages": [], "api_calls": 1}
    _install_fake_agent(monkeypatch, tmp_path, FakeAgent)
    async def inner(event, source, k, generation):
        await r._run_agent(message=event.text, context_prompt="", history=[], source=source,
                           session_id="review", session_key=k, run_generation=generation,
                           inbound_message_id=event.message_id)
        return None
    r._handle_message_with_agent = inner
    async def handler(event):
        handoffs.append(event.message_id)
        # Admission may legally replace event objects; require semantic preservation.
        if event is not root:
            index = int(event.message_id.removeprefix("id-"))
            assert event.metadata == {"index": index}
            assert event.internal is (index % 2 == 0)
        return await r._handle_message(event)
    a._message_handler = handler
    for event in events: r._enqueue_fifo(key, event, a)
    assert a._start_session_processing(root, key)
    async def settle():
        while a._background_tasks:
            await asyncio.gather(*list(a._background_tasks))
            await asyncio.sleep(0)
    try:
        await asyncio.wait_for(settle(), timeout=10)
        expected = [e.message_id for e in events]
        print({"count": count, "rewrite": rewrite, "actual": consumed, "handoffs": handoffs,
               "rescues": rescues, "hook_calls": hook_calls})
        assert len(consumed) == count
        assert len(set(consumed)) == count
        assert not r._overflow_queue(key)
        assert key not in a._pending_messages
        assert key not in a._active_sessions
        assert "id-3" in hook_calls
        assert consumed == expected
    finally:
        await a.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("denial", ["plugin", "auth"])
async def test_real_admission_denial_consumes_proof_and_nested_cannot_inherit(denial, monkeypatch):
    import hermes_cli.lifecycle
    from dataclasses import replace

    a = Adapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
    r = _make_runner(a)
    s = SessionSource(platform=Platform.TELEGRAM, chat_type="dm", chat_id="42", user_id="42")
    key = build_session_key(s)
    r._delivery_adapter_for = lambda source: a
    r._session_key_for_source = lambda source: key
    r._intake_adapter_for = lambda source: None
    r._scale_to_zero_note_real_inbound = lambda: None
    r._is_user_authorized_for_source = lambda source: False
    r._get_unauthorized_dm_behavior = lambda *args, **kwargs: None
    r._hm_report_ignored_dm = AsyncMock()
    event = MessageEvent(text="denied", source=s, message_id="denied")
    task = asyncio.current_task()
    a._session_tasks[key] = task
    setattr(task, "_hermes_fifo_handoff", (a, key, event))
    nested_seen = []

    async def hook(name, **kwargs):
        assert not hasattr(asyncio.current_task(), "_hermes_fifo_handoff")
        if kwargs["event"] is event:
            # An equal clone invoked recursively in the same task and a child
            # both see no proof, even while the original admission is suspended.
            nested = replace(event)
            assert await r._handle_message(nested) is None
            assert await asyncio.create_task(r._handle_message(nested)) is None
            return [{"action": "skip"}] if denial == "plugin" else [{"action": "rewrite", "text": "DENIED"}]
        nested_seen.append(kwargs["event"])
        return [{"action": "skip"}]

    monkeypatch.setattr(hermes_cli.lifecycle, "ainvoke_hook", hook)
    try:
        assert await r._handle_message(event) is None
        assert len(nested_seen) == 2
        assert not hasattr(task, "_hermes_fifo_handoff")
    finally:
        a._session_tasks.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["owner", "adapter", "key"])
async def test_proof_rechecked_after_admission_awaits(change, monkeypatch):
    import hermes_cli.lifecycle

    a = Adapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
    other = Adapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
    r = _make_runner(a)
    s = SessionSource(platform=Platform.TELEGRAM, chat_type="dm", chat_id="42", user_id="42")
    key = build_session_key(s)
    route = {"adapter": a, "key": key}
    r._delivery_adapter_for = lambda source: route["adapter"]
    r._session_key_for_source = lambda source: route["key"]
    r._intake_adapter_for = lambda source: None
    r._scale_to_zero_note_real_inbound = lambda: None
    r._is_user_authorized_for_source = lambda source: True
    r._admit_bot_message_for_source = lambda source: True
    async def hook(*args, **kwargs):
        return [{"action": "rewrite", "text": "HEAD"}]
    monkeypatch.setattr(hermes_cli.lifecycle, "ainvoke_hook", hook)
    r._hm_estop_gate = lambda *args: None
    async def pending(*args):
        if change == "owner": a._session_tasks[key] = object()
        elif change == "adapter": route["adapter"] = other
        else: route["key"] = key + ":other"
        return None
    r._hm_pending_reply_intercepts = pending
    r._hm_evict_idle_stale_agent = lambda *args: None
    r._is_session_running = lambda *args: False
    r._hm_dispatch_idle_commands = AsyncMock(return_value=(False, None))
    r._is_telegram_topic_root_lobby = lambda *args: False
    r._external_drain_active = False
    r._claim_active_session_slot = lambda *args: (None, None)
    event = MessageEvent(text="head", source=s, message_id="head")
    task = asyncio.current_task()
    a._session_tasks[key] = task
    setattr(task, "_hermes_fifo_handoff", (a, key, event))
    orphan = MessageEvent(text="orphan", source=s, message_id="orphan")
    r._session_state(key).conversation.queued_events.append(orphan)
    real_rescue = r._hm_rescue_orphaned_fifo
    class Done(Exception): pass
    def check(admitted, source, internal, k):
        assert admitted is not event and admitted.text == "HEAD"
        assert not hasattr(task, "_hermes_fifo_handoff")
        if change != "adapter":
            assert real_rescue(admitted, source, internal, k)[0] is orphan
        raise Done
    r._hm_rescue_orphaned_fifo = check
    try:
        with pytest.raises(Done):
            await r._handle_message(event)
        assert not hasattr(task, "_hermes_fifo_handoff")
    finally:
        a._session_tasks.clear()
