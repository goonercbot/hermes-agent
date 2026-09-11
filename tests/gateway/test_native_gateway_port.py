"""Gateway-native continuity regressions for the v0.21.1 mixin split."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace

from gateway.run import _build_gateway_agent_history, _hygiene_requires_exclusive_completion
from gateway.run_turn import GatewayTurnMixin
from gateway.run_turn_runner import TurnRunner


def _sealed_history():
    """Build a valid protected carrier/handoff/tail with the native producer."""
    from tests.run_agent.test_native_incremental_handoff import _agent, _note, _response, _source
    from agent.native_incremental_handoff import native_incremental_compact_context

    agent, _ = _agent([_response({"type": "compaction", "id": "cp", "encrypted_content": "sealed"})])
    source = _source()
    _note(agent, source)
    return native_incremental_compact_context(agent, source)


def test_gateway_history_preserves_native_sealed_carrier_bytes():
    history = _sealed_history()
    expected = deepcopy(history)
    for index, row in enumerate(history):
        row["timestamp"] = 1_700_000_000 + index
        row["observed"] = True

    replay, observed = _build_gateway_agent_history(history)

    assert observed is None
    assert replay == expected


def test_turn_runner_binds_canonical_source_and_clean_replay(monkeypatch):
    history = _sealed_history()
    agent = SimpleNamespace(native_incremental_handoff_enabled=True)
    calls = []
    monkeypatch.setattr(
        "agent.native_incremental_handoff.bind_native_incremental_replay_projection",
        lambda target, *, source_messages, replay_messages: calls.append((target, source_messages, replay_messages)),
    )
    turn = TurnRunner.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(
        history=history,
        channel_prompt=None,
        user_config={},
        session_id="native-session",
    )

    replay, _observed, _media = turn._load_turn_history(agent, reused_cached_agent=False)

    assert calls == [(agent, history, replay)]
    assert replay == history


def test_native_hygiene_waits_past_ordinary_turn_hold():
    class Fence:
        is_cancelled = False

        @staticmethod
        def seconds_since_progress():
            return 99.0

    async def exercise():
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        loop.call_later(0.03, future.set_result, ([{"role": "assistant", "content": "sealed"}], None))
        attempt = SimpleNamespace(
            commit_fence=Fence(), future=future, wait_started=loop.time(), require_exclusive_completion=True,
        )
        settings = SimpleNamespace(timeout_seconds=0.001, total_ceiling_seconds=0.2, max_turn_hold_seconds=0.001)
        session = SimpleNamespace(session_id="native-session")
        return await GatewayTurnMixin._hmwa_hygiene_wait_for_summary(SimpleNamespace(), attempt, settings, session)

    assert asyncio.run(exercise()) == [{"role": "assistant", "content": "sealed"}]


def test_native_hygiene_exclusive_gate_follows_native_continuity(monkeypatch):
    route = SimpleNamespace(is_codex_backend=True, is_xai_responses=False, is_github_responses=False)
    monkeypatch.setattr("agent.codex_responses_adapter.classify_responses_route", lambda _agent: route)
    monkeypatch.setattr("agent.native_compaction.native_continuity_capable", lambda *_args, **_kwargs: True)

    assert _hygiene_requires_exclusive_completion(SimpleNamespace(api_mode="codex_responses"))
    assert not _hygiene_requires_exclusive_completion(SimpleNamespace(api_mode="chat_completions"))
