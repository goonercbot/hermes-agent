"""Warnings describe unresolved pressure, not a transient mid-turn snapshot."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent

    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        instance = AIAgent(
            model="test-model", provider="openai", api_key="test-key",
            base_url="https://api.example.invalid/v1", quiet_mode=True,
            skip_memory=True, skip_context_files=True, skip_background_review=True,
            enabled_toolsets=[],
        )
    instance.context_compressor = SimpleNamespace(
        threshold_tokens=204_000, context_length=400_000,
        last_prompt_tokens=287_761, reason="structural_backoff:283",
        awaiting_real_usage_after_compression=False,
    )
    compressor = instance.context_compressor
    compressor.should_compress_info = lambda tokens: (
        False, compressor.reason if tokens >= compressor.threshold_tokens else None
    )
    instance._warnings = []
    instance._emit_warning = instance._warnings.append
    instance._touch_activity = lambda *args, **kwargs: None
    yield instance
    instance.close()


def run_turn(agent, body):
    """Use the public turn owner, without provider traffic or live persistence."""
    with patch("agent.conversation_loop.run_conversation", side_effect=lambda *_a, **_k: body()):
        return agent.run_conversation("Continue the current task")


def queue_warning(agent):
    agent._warn_context_overflow_blocked("structural_backoff:283", 287_761, 204_000)


def test_successful_same_turn_recovery_never_emits_warning(agent):
    def recover():
        queue_warning(agent)
        assert agent._warnings == [], "do not alarm the user before recovery has finished"
        agent.context_compressor.last_prompt_tokens = 30_237
        agent.context_compressor.reason = None
        return {"completed": True, "final_response": "Recovered"}

    assert run_turn(agent, recover)["completed"]
    assert agent._warnings == []


@pytest.mark.parametrize("reason", ["structural_backoff:280", "cooldown:28"])
def test_unresolved_block_warns_at_turn_exit_and_deduplicates(agent, reason):
    def blocked():
        agent.context_compressor.reason = reason
        agent._warn_context_overflow_blocked(reason, 287_761, 204_000)
        assert agent._warnings == []
        return {"completed": True, "final_response": "Still answering"}

    run_turn(agent, blocked)
    assert len(agent._warnings) == 1
    assert f"blocked ({reason})" in agent._warnings[0]
    # The next turn sees the already-delivered warning, not a second emission.
    with patch("agent.conversation_loop.run_conversation", side_effect=lambda *_a, **_k: (
        agent._warn_context_overflow_blocked(reason, 287_761, 204_000) or {"completed": True}
    )):
        agent.run_conversation("Next turn")
    assert len(agent._warnings) == 1


@pytest.mark.parametrize("exc", [RuntimeError("failed"), TimeoutError("timeout"), KeyboardInterrupt()])
def test_terminal_failure_does_not_drop_pending_warning(agent, exc):
    def fail():
        queue_warning(agent)
        assert agent._warnings == []
        raise exc

    with pytest.raises(type(exc)):
        run_turn(agent, fail)
    assert len(agent._warnings) == 1


def test_expired_guard_does_not_emit_stale_warning(agent):
    def resume():
        queue_warning(agent)
        agent.context_compressor.reason = None
        return {"completed": True}

    run_turn(agent, resume)
    assert agent._warnings == []


def test_new_pressure_after_recovery_still_warns(agent):
    def recover_then_block_again():
        queue_warning(agent)
        agent._clear_context_overflow_warn()
        agent.context_compressor.last_prompt_tokens = 310_000
        agent.context_compressor.reason = "cooldown:50"
        agent._warn_context_overflow_blocked("cooldown:50", 310_000, 204_000)
        return {"completed": False}

    run_turn(agent, recover_then_block_again)
    assert len(agent._warnings) == 1
    assert "310,000" in agent._warnings[0]
    assert "cooldown:50" in agent._warnings[0]


@pytest.mark.parametrize("reason,tokens", [("ineffective", 287_761), ("cooldown:30", 410_000)])
def test_permanent_block_or_hard_limit_warns_immediately(agent, reason, tokens):
    def urgent():
        agent.context_compressor.reason = reason
        agent._warn_context_overflow_blocked(reason, tokens, 204_000)
        assert len(agent._warnings) == 1
        return {"completed": False}

    run_turn(agent, urgent)
    assert len(agent._warnings) == 1
