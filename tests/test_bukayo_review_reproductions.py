"""Parent reproductions of the standard-candidate independent review findings."""
import asyncio
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from hermes_state import SessionDB
from gateway.run import GatewayRunner
from gateway.config import Platform
from gateway.platforms.base import SendResult
from tests.gateway.test_run_progress_topics import (
    ProgressCaptureAdapter, SessionSource, MessageEvent, MessageType,
)


def test_f1_failed_marker_rolls_back_all_baselines(tmp_path):
    path = tmp_path / "fault.sqlite"
    db = SessionDB(path)
    db.create_session("legacy", source="cli")
    db.update_token_counts("legacy", model="test-model", input_tokens=42, api_call_count=1)
    db.close()
    with sqlite3.connect(path) as c:
        c.executescript("""
            DROP TRIGGER usage_events_no_delete;
            DELETE FROM usage_events;
            DELETE FROM usage_event_meta;
            CREATE TRIGGER inject_marker_failure BEFORE INSERT ON usage_event_meta
            BEGIN SELECT RAISE(ABORT, 'injected marker failure'); END;
        """)
    with pytest.raises(sqlite3.IntegrityError, match="injected marker failure"):
        SessionDB(path)
    with sqlite3.connect(path) as observer:
        committed_baselines = observer.execute("SELECT count(*) FROM usage_events").fetchone()[0]
    assert committed_baselines == 0, "baseline escaped failed initialization transaction"


def test_f4_coalesced_signed_parts_remain_in_ledger(tmp_path):
    db = SessionDB(tmp_path / "signed.sqlite")
    try:
        db.create_session("queued", source="cli")
        batch = []
        for amount in (1.0, -1.0):
            kwargs = {"model": "same", "billing_provider": "same", "actual_cost_usd": amount}
            kwargs["_usage_event_parts"] = [db._new_usage_event_part(kwargs)]
            batch.append(("queued", kwargs))
        db._apply_token_batch(batch)
        costs = [row[0] for row in db._conn.execute(
            "SELECT actual_cost_usd FROM usage_events WHERE session_id='queued' ORDER BY recorded_at, event_id"
        )]
        assert sorted(costs) == [-1.0, 1.0], "net-zero queue erased admitted signed events"
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", ["group", "forum"])
async def test_f3_home_setup_notice_is_not_shared(monkeypatch, chat_type):
    import agent.secret_scope
    monkeypatch.setattr(agent.secret_scope, "get_secret", lambda *_args, **_kw: None)
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    runner = SimpleNamespace(
        async_session_store=SimpleNamespace(has_any_sessions=AsyncMock(return_value=True)),
        config=SimpleNamespace(get_home_channel=lambda *_: None),
        _deliver_platform_notice=AsyncMock(),
    )
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-42", chat_type=chat_type)
    await GatewayRunner._hmwa_first_contact_notes(runner, source, [], [])
    runner._deliver_platform_notice.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", ["dm", "private"])
async def test_f3_home_setup_notice_remains_available_in_private_chats(monkeypatch, chat_type):
    import agent.secret_scope
    monkeypatch.setattr(agent.secret_scope, "get_secret", lambda *_args, **_kw: None)
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    runner = SimpleNamespace(
        async_session_store=SimpleNamespace(has_any_sessions=AsyncMock(return_value=True)),
        config=SimpleNamespace(get_home_channel=lambda *_: None),
        _deliver_platform_notice=AsyncMock(),
    )
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type=chat_type)
    await GatewayRunner._hmwa_first_contact_notes(runner, source, [], [])
    runner._deliver_platform_notice.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["failed_result", "exception"])
async def test_f2_failed_final_delivery_keeps_commentary(mode):
    adapter = ProgressCaptureAdapter()
    adapter.config.typing_indicator = False
    adapter.delete_message = AsyncMock(return_value=True)
    adapter.send = AsyncMock(return_value=SendResult(success=False, error="injected send failure"))
    if mode == "exception":
        adapter.send.side_effect = RuntimeError("injected send exception")
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm")
    event = MessageEvent(text="canary", message_type=MessageType.TEXT, source=source, message_id="incoming")
    key = "test-review-delivery"
    interrupt = asyncio.Event()
    interrupt._hermes_run_generation = 1
    adapter._active_sessions[key] = interrupt
    ctx = SimpleNamespace(_cleanup_progress=True, _cleanup_msg_ids=["commentary"],
                          _commentary_messages=[("commentary", "Working")], session_key=key,
                          source=source, run_generation=1)
    runner = object.__new__(GatewayRunner)

    async def handler(_event):
        runner._run_agent_schedule_bubble_cleanup(
            {"failed": False, "final_response": "Complete answer"}, adapter, ctx)
        return "Complete answer"

    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, key)
    for _ in range(5):
        await asyncio.sleep(0)
    assert adapter.send.await_count > 0
    adapter.delete_message.assert_not_awaited()
