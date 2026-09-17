"""Gateway-side session binding for async delegations (#57498, #55578).

Three invariants on the messaging-gateway surface, mirroring the TUI rules:

1. Completions are pinned to the spawning session (contributor commit).
2. A dead/ended spawning session is never resurrected: the injection is
   dropped, fail-closed (never rerouted to the peer's current session).
3. /new interrupts the old conversation's in-flight async delegations.
"""

import asyncio
import threading
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import tools.async_delegation as ad


@pytest.fixture(autouse=True)
def _reset_async_delegation():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


def _seed_record(delegation_id, session_key="", parent_session_id="", status="running"):
    fn = MagicMock()
    with ad._records_lock:
        ad._records[delegation_id] = {
            "delegation_id": delegation_id,
            "status": status,
            "session_key": session_key,
            "parent_session_id": parent_session_id,
            "interrupt_fn": fn,
        }
    return fn


class TestInterruptForSessionByParentId:
    def test_parent_session_id_selector(self):
        mine = _seed_record("d1", session_key="agent:main:telegram:dm:1", parent_session_id="sess_old")
        other = _seed_record("d2", session_key="agent:main:telegram:dm:2", parent_session_id="sess_other")
        n = ad.interrupt_for_session(parent_session_id="sess_old")
        assert n == 1
        mine.assert_called_once()
        other.assert_not_called()


class TestGatewayPinningFailsClosed:
    """The gateway must follow only verified compression continuations."""

    @staticmethod
    def _entry(session_id):
        from datetime import datetime

        from gateway.config import Platform
        from gateway.session import SessionEntry

        return SessionEntry(
            session_key="agent:main:telegram:group:-100:4",
            session_id=session_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            platform=Platform.TELEGRAM,
            chat_type="group",
        )

    def _make_runner(
        self,
        rows,
        *,
        compression_tip=None,
        compression_error=None,
        switched_entry=None,
    ):
        from gateway.run import GatewayRunner
        from gateway.session import AsyncSessionStore

        runner = object.__new__(GatewayRunner)
        db = MagicMock()
        db.get_session = AsyncMock(side_effect=lambda session_id: rows.get(session_id))
        db.get_compression_tip = AsyncMock(
            return_value=compression_tip,
            side_effect=compression_error,
        )
        runner._session_db = db
        runner.session_store = MagicMock()
        runner.session_store.switch_session = MagicMock(return_value=switched_entry)
        runner.session_store.advance_compression_session = MagicMock(
            return_value=switched_entry
        )
        runner._async_session_store = AsyncSessionStore(runner.session_store)
        return runner

    @staticmethod
    def _assert_no_route_change(runner):
        getattr(runner.session_store, "switch_session").assert_not_called()
        getattr(
            runner.session_store, "advance_compression_session"
        ).assert_not_called()


    @pytest.mark.asyncio
    async def test_live_foreign_session_cannot_rebind_gateway_route(self):
        current = self._entry("sess_current")
        runner = self._make_runner(
            {"sess_live": {"id": "sess_live", "source": "telegram", "ended_at": None}},
        )

        resolved = await runner._resolve_async_delegation_session(
            current, "sess_live"
        )

        assert resolved is None
        self._assert_no_route_change(runner)

    @pytest.mark.asyncio
    async def test_live_subagent_row_in_real_sqlite_cannot_rebind_gateway_route(self, tmp_path, request):
        """A live delegate child is never a gateway route owner, even with a valid parent id."""
        from gateway.run import GatewayRunner
        from gateway.session import AsyncSessionStore
        from hermes_state import AsyncSessionDB, SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        request.addfinalizer(db.close)
        db.create_session("gateway-owner", source="telegram")
        db.create_session(
            "active-subagent", source="subagent", parent_session_id="gateway-owner"
        )
        owner = self._entry("gateway-owner")
        runner = object.__new__(GatewayRunner)
        runner._session_db = AsyncSessionDB(db)
        runner.session_store = MagicMock()
        runner.session_store.switch_session = MagicMock()
        runner.session_store.advance_compression_session = MagicMock()
        runner._async_session_store = AsyncSessionStore(runner.session_store)

        resolved = await runner._resolve_async_delegation_session(owner, "active-subagent")

        assert resolved is None
        self._assert_no_route_change(runner)
        assert db.get_session("gateway-owner")["ended_at"] is None
        assert db.get_session("active-subagent")["ended_at"] is None

    @pytest.mark.asyncio
    async def test_real_sqlite_profiles_isolate_active_children_and_preserve_arrival_order(self, tmp_path, monkeypatch, request):
        """Policy: distinct same-turn completions are consolidated in queue-arrival order per route."""
        from gateway.run import GatewayRunner
        from gateway.session import AsyncSessionStore
        from hermes_state import AsyncSessionDB, SessionDB

        def make_profile(profile, chat_id):
            db = SessionDB(db_path=tmp_path / profile / "state.db")
            request.addfinalizer(db.close)
            owner_id, child_id = f"{profile}-owner", f"{profile}-child"
            db.create_session(owner_id, source="telegram", profile_name=profile)
            db.create_session(
                child_id, source="subagent", parent_session_id=owner_id, profile_name=profile,
            )
            entry = self._entry(owner_id)
            entry.session_key = f"agent:{profile}:telegram:dm:{chat_id}"
            runner = object.__new__(GatewayRunner)
            runner._session_db = AsyncSessionDB(db)
            runner.session_store = MagicMock()
            runner.session_store.switch_session = MagicMock()
            runner.session_store.advance_compression_session = MagicMock()
            runner._async_session_store = AsyncSessionStore(runner.session_store)
            runner._completion_delivery_lock = threading.Lock()
            runner._completion_deliveries_inflight = set()
            runner._completion_deliveries_delivered = OrderedDict()
            runner._completion_delivery_retention = 2048
            return runner, db, entry, owner_id, child_id

        runner_a, db_a, owner_a, owner_a_id, child_a_id = make_profile("profile-a", "100")
        runner_b, db_b, owner_b, owner_b_id, child_b_id = make_profile("profile-b", "200")
        assert await runner_a._resolve_async_delegation_session(owner_a, child_a_id) is None
        assert await runner_b._resolve_async_delegation_session(owner_b, child_b_id) is None
        assert db_a.get_session(child_b_id) is None
        assert db_b.get_session(child_a_id) is None

        delivered = []

        async def capture(text, event, **_kwargs):
            delivered.append((event["session_key"], text, event["delegation_id"]))
            return True

        async def ready(_event):
            return True

        runner_a._deliver_completion_notification = capture
        runner_b._deliver_completion_notification = capture
        runner_a._completion_delivery_ready = ready
        runner_b._completion_delivery_ready = ready
        monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda event, _claim: event["delegation_id"])

        def event(delegation_id, session_key, parent_session_id, summary):
            return {
                "type": "async_delegation", "delegation_id": delegation_id,
                "session_key": session_key, "parent_session_id": parent_session_id,
                "goal": delegation_id, "status": "completed", "summary": summary,
                "api_calls": 1, "duration_seconds": 1.0,
            }

        a_first = event("a-first", owner_a.session_key, owner_a_id, "first arrived")
        a_second = event("a-second", owner_a.session_key, owner_a_id, "second arrived")
        b_only = event("b-only", owner_b.session_key, owner_b_id, "profile-b only")
        assert await asyncio.gather(
            runner_a._deliver_async_delegation_group([a_second, a_first]),
            runner_b._deliver_async_delegation_group([b_only]),
        ) == [True, True]

        a_delivery = next(item for item in delivered if item[0] == owner_a.session_key)
        b_delivery = next(item for item in delivered if item[0] == owner_b.session_key)
        assert a_delivery[1].index("second arrived") < a_delivery[1].index("first arrived")
        assert "profile-b only" not in a_delivery[1]
        assert "profile-b only" in b_delivery[1]
        assert db_a.get_session(owner_a_id)["profile_name"] == "profile-a"
        assert db_b.get_session(owner_b_id)["profile_name"] == "profile-b"

    @pytest.mark.asyncio
    async def test_non_compression_ended_parent_drops(self):
        current = self._entry("sess_old")
        runner = self._make_runner(
            {
                "sess_old": {
                    "id": "sess_old",
                    "ended_at": "2026-07-08T00:00:00",
                    "end_reason": "session_reset",
                }
            }
        )

        resolved = await runner._resolve_async_delegation_session(
            current, "sess_old"
        )

        assert resolved is None
        self._assert_no_route_change(runner)


    @pytest.mark.asyncio
    async def test_intermediate_compression_route_advances_to_same_live_tip(self):
        current = self._entry("sess_middle")
        tip = self._entry("sess_tip")
        runner = self._make_runner(
            {
                "sess_parent": {
                    "id": "sess_parent",
                    "ended_at": "2026-07-08T00:00:00",
                    "end_reason": "compression",
                },
                "sess_middle": {
                    "id": "sess_middle",
                    "ended_at": "2026-07-08T00:01:00",
                    "end_reason": "compression",
                    "parent_session_id": "sess_parent",
                },
                "sess_tip": {
                    "id": "sess_tip",
                    "ended_at": None,
                    "parent_session_id": "sess_middle",
                },
            },
            compression_tip="sess_tip",
            switched_entry=tip,
        )

        resolved = await runner._resolve_async_delegation_session(
            current, "sess_parent"
        )

        assert resolved is tip
        getattr(
            runner.session_store, "advance_compression_session"
        ).assert_called_once_with(current.session_key, "sess_middle", "sess_tip")

    @pytest.mark.asyncio
    async def test_compression_parent_follows_real_sessiondb_lineage(self, tmp_path):
        from gateway.run import GatewayRunner
        from gateway.session import AsyncSessionStore
        from hermes_state import AsyncSessionDB, SessionDB

        session_db = SessionDB(db_path=tmp_path / "state.db")
        session_db.create_session("sess_parent", source="telegram")
        session_db.end_session("sess_parent", end_reason="compression")
        session_db.create_session(
            "sess_tip",
            source="telegram",
            parent_session_id="sess_parent",
        )

        current = self._entry("sess_parent")
        tip = self._entry("sess_tip")
        runner = object.__new__(GatewayRunner)
        runner._session_db = AsyncSessionDB(session_db)
        runner.session_store = MagicMock()
        runner.session_store.switch_session = MagicMock(return_value=tip)
        runner.session_store.advance_compression_session = MagicMock(return_value=tip)
        runner._async_session_store = AsyncSessionStore(runner.session_store)

        resolved = await runner._resolve_async_delegation_session(
            current, "sess_parent"
        )

        assert resolved is tip
        getattr(
            runner.session_store, "advance_compression_session"
        ).assert_called_once_with(current.session_key, "sess_parent", "sess_tip")


class TestResetHandlerInterruptsDelegations:
    def test_reset_command_calls_interrupt_for_session(self):
        """The /new handler must sever the old conversation's delegations."""
        import inspect
        from gateway import slash_commands

        src = inspect.getsource(slash_commands.GatewaySlashCommandsMixin._handle_reset_command)
        assert "interrupt_for_session" in src
        assert "session_reset" in src
