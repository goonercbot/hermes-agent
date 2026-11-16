"""Durable completion admission and route resolution reject foreign owners.

Port of the owner-corrected v0.22.5 audit with real stores and synthetic chats.
"""
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["own", "foreign_live", "foreign_idle", "foreign_child"])
async def test_completion_cannot_take_over_foreign_chat(tmp_path, request, case):
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner, _profile_runtime_scope
    from gateway.session import SessionSource
    from tools import async_delegation as ad

    ad._reset_for_tests()
    request.addfinalizer(ad._reset_for_tests)
    with _profile_runtime_scope(tmp_path):
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
        store = runner.session_store
        request.addfinalizer(runner.close_all_session_db_handles)
        request.addfinalizer(store.close_all_db_handles)
        current = store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="chat-A", chat_type="dm"))
        foreign = store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="chat-B", chat_type="dm"))
        pinned = current.session_id if case == "own" else foreign.session_id
        if case == "foreign_idle":
            store._db.end_session(pinned, end_reason="idle")
        if case == "foreign_child":
            store._db.create_session("child-of-B", source="subagent", parent_session_id=pinned,
                                     model_config={"_delegate_from": pinned})
            pinned = "child-of-B"
        accepted = []

        async def accept(event):
            route = store.lookup_by_session_key(current.session_key)
            result = await runner._resolve_async_delegation_session(route, event.metadata["gateway_session_id"])
            accepted.append(result.session_id if result else None)
            event._gateway_accepted = result is not None

        runner.adapters[Platform.TELEGRAM] = SimpleNamespace(handle_message=accept)
        evt = {"type": "async_delegation", "delegation_id": "preservation-ownership",
               "session_key": current.session_key, "parent_session_id": pinned,
               "dispatched_at": 1.0, "summary": "synthetic complete", "status": "completed"}
        ad._persist_dispatch(evt)
        ad._persist_completion(evt, {"status": "completed", "summary": evt["summary"]})
        await runner._deliver_completion_notification(evt["summary"], evt)
        row = ad.get_durable_delegation(evt["delegation_id"])
        if case == "own":
            assert accepted == [current.session_id]
            assert row["delivery_state"] == "delivered"
        else:
            assert accepted == []
            assert row["delivery_state"] == "dropped"
            assert await runner._resolve_async_delegation_session(current, pinned) is None
        assert store.lookup_by_session_key(current.session_key).session_id == current.session_id
