"""Structural no-op backoff (#93022).

A compression attempt that finds nothing eligible inside the protection
window (too few messages / empty window / post-handoff residue) is "nothing
to compress right now", not an ineffective attempt: it must defer retries
transiently instead of arming the permanent anti-thrash breaker, so a short
session can still auto-compact after it grows real compressible material.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.conversation_compression import (
    _automatic_compression_blocked_for_attempt,
    compress_context,
)
from agent.context_compressor import ContextCompressor


def _compressor(protect_first_n: int = 1) -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=protect_first_n,
            protect_last_n=1,
            quiet_mode=True,
        )


def _response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def test_insufficient_messages_backs_off_without_strike():
    """Too few messages -> structural backoff, breaker stays untouched."""
    compressor = _compressor()
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hello"},
    ]

    result = compressor.compress(messages, current_tokens=90_000)

    assert result == messages
    assert compressor._ineffective_compression_count == 0
    assert compressor._structural_no_op_backoff_until > 0.0
    telemetry = compressor._last_compression_telemetry or {}
    assert telemetry.get("failure_class") == "insufficient_messages"


def test_no_compressible_window_backs_off_without_strike():
    """Transcript inside the tail budget -> backoff, breaker untouched."""
    compressor = _compressor()
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "turn one"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "turn two"},
        {"role": "assistant", "content": "answer two"},
        {"role": "user", "content": "turn three"},
        {"role": "assistant", "content": "answer three"},
        {"role": "user", "content": "latest request in protected tail"},
    ]

    with patch.object(compressor, "_find_tail_cut_by_tokens", return_value=2):
        result = compressor.compress(messages, current_tokens=90_000)

    assert result == messages
    assert compressor._ineffective_compression_count == 0
    assert compressor._structural_no_op_backoff_until > 0.0
    telemetry = compressor._last_compression_telemetry or {}
    assert telemetry.get("failure_class") == "no_compressible_window"


def test_gate_blocked_during_backoff_then_resumes():
    """should_compress defers during the backoff and recovers after it lapses.

    The transcript sits over the compression threshold the whole time; only
    the clock changes, proving the block is transient rather than a latched
    breaker state.
    """
    compressor = _compressor()

    # While the structural backoff is live the gate must say blocked.
    compressor._structural_no_op_backoff_until = time.monotonic() + 300.0
    with patch.object(
        compressor, "_automatic_compression_blocked", return_value=True
    ):
        should, reason = compressor.should_compress_info(prompt_tokens=300_000)
        assert should is False
        assert reason is not None
        assert reason.startswith("structural_backoff:")
        assert compressor._compression_block_reason().startswith(
            "structural_backoff:"
        )

    # After the backoff lapses nothing blocks: same over-threshold
    # transcript compresses again (real gate, real state).
    compressor._structural_no_op_backoff_until = (
        time.monotonic() - 1.0
    )
    should, reason = compressor.should_compress_info(prompt_tokens=300_000)
    assert should is True
    assert reason is None


SUMMARY_RESPONSE = "fresh replacement summary body"


def _messages_with_old_handoff():
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": (
            "CONTEXT SUMMARY (from previous session):\nold summary body"
        )},
        {"role": "assistant", "content": "handoff acknowledged after resume"},
        {"role": "user", "content": "new user turn after resume"},
        {"role": "assistant", "content": "new assistant work after resume"},
        {"role": "user", "content": "more new work after resume"},
        {"role": "assistant", "content": "latest tail response"},
        {"role": "user", "content": "final active request stays in protected tail"},
    ]


def test_forced_attempt_and_success_lift_the_backoff():
    """Manual /compress clears an active backoff; a completed boundary lifts it.

    Both are proof the transcript is being actively worked on — neither may
    leave auto-compaction deferred by a stale structural no-op.
    """
    compressor = _compressor()
    compressor._structural_no_op_backoff_until = time.monotonic() + 300.0

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response(SUMMARY_RESPONSE),
    ):
        compressed = compressor.compress(
            _messages_with_old_handoff(), force=True
        )

    assert compressor._structural_no_op_backoff_until == 0.0
    # The forced attempt actually committed a boundary.
    assert len(compressed) < len(_messages_with_old_handoff())


def test_real_attempt_underperformance_still_strikes_breaker():
    """Only genuine attempted-but-underperformed compressions strike.

    A real summary pass that saves <10% goes through the ineffective
    verdict (persisted); structural no-ops must not touch that counter —
    that distinction IS this fix.
    """
    compressor = _compressor()
    before = compressor._ineffective_compression_count
    compressor._record_ineffective_compression_verdict(before + 1)
    assert compressor._ineffective_compression_count == before + 1

    compressor._record_structural_no_op("test reason")
    assert compressor._structural_no_op_backoff_until > 0.0
    assert compressor._ineffective_compression_count == before + 1


def test_native_automatic_attempt_bypasses_only_structural_no_op_backoff():
    """Codex-native compaction must not inherit generic-window no-op state."""
    compressor = _compressor()
    now = time.monotonic()
    compressor._structural_no_op_backoff_until = now + 300.0

    # Native compaction owns a provider-side boundary, so the generic
    # compressor's 300-second local no-op backoff must not delay its attempt.
    assert not _automatic_compression_blocked_for_attempt(
        compressor, native_continuity_eligible=True, prompt_tokens=90_000
    )

    # Generic and Responses-compatible-but-ineligible routes retain the full
    # 300-second anti-thrash behavior.
    assert compressor._structural_no_op_backoff_until > now + 299.0
    assert _automatic_compression_blocked_for_attempt(
        compressor, native_continuity_eligible=False, prompt_tokens=90_000
    )

    # The exemption is structural-only: real cooldown and ineffective-breaker
    # state still blocks an otherwise native-eligible automatic attempt.
    compressor._summary_failure_cooldown_until = time.monotonic() + 300.0
    assert _automatic_compression_blocked_for_attempt(
        compressor, native_continuity_eligible=True, prompt_tokens=90_000
    )
    compressor._summary_failure_cooldown_until = 0.0
    compressor._ineffective_compression_count = 2
    assert _automatic_compression_blocked_for_attempt(
        compressor, native_continuity_eligible=True, prompt_tokens=90_000
    )


def test_native_no_progress_rearms_only_after_meaningful_request_growth():
    """Native no-checkpoint retries follow pressure growth, not a blind timer."""
    compressor = _compressor()
    compressor._structural_no_op_backoff_until = time.monotonic() + 300.0
    compressor._native_no_progress_rearm_tokens = 91_024

    assert _automatic_compression_blocked_for_attempt(
        compressor, native_continuity_eligible=True, prompt_tokens=90_500
    )
    assert not _automatic_compression_blocked_for_attempt(
        compressor, native_continuity_eligible=True, prompt_tokens=91_024
    )


class _NativeAttemptAgent:
    """Small real-compress_context fixture for the automatic native gate."""

    def __init__(self, compressor):
        self.context_compressor = compressor
        self.api_mode = "codex_responses"
        self.model = "gpt-5.6-test"
        self.provider = "openai-codex"
        self.base_url = "https://chatgpt.com/backend-api/codex"
        self._base_url_hostname = "chatgpt.com"
        self._base_url_lower = self.base_url
        self.codex_responses_native_compaction = True
        self.compression_enabled = True
        self._codex_reasoning_replay_enabled = True
        self.runtime_capabilities = {"native_compaction": True}
        self.capabilities = {"native_compaction": True}
        self._compression_feasibility_checked = True
        self.compression_in_place = False
        self._memory_manager = None
        self._session_db = None
        self._todo_store = SimpleNamespace(format_for_injection=lambda: "")
        self._cached_system_prompt = "system"
        self.session_id = "native-structural-backoff"
        self.platform = "cli"
        self.tools = []

    def _emit_status(self, _message):
        pass

    def _emit_warning(self, _message):
        pass

    def _invalidate_system_prompt(self):
        self._cached_system_prompt = None

    def _build_system_prompt(self, system_message):
        return system_message


def test_native_eligible_automatic_compression_reaches_native_dispatch_after_no_op():
    """A stale generic structural no-op must not short-circuit native dispatch."""
    compressor = _compressor()
    compressor._structural_no_op_backoff_until = time.monotonic() + 300.0
    agent = _NativeAttemptAgent(compressor)
    messages = [
        {"role": "user", "content": "old context"},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "current request"},
    ]
    dispatched = []

    def _native_dispatch(_agent, _messages, _system_message):
        dispatched.append(True)
        compressor._last_compress_aborted = False
        compressor._last_summary_error = None
        compressor._last_compression_made_progress = True
        compressor._last_summary_fallback_used = False
        return [{"role": "user", "content": "native compacted context"}]

    with patch(
        "agent.native_compaction.native_compact_context", side_effect=_native_dispatch
    ):
        compressed, _ = compress_context(
            agent, messages, "system", approx_tokens=90_000
        )

    assert dispatched == [True]
    assert compressed == [{"role": "user", "content": "native compacted context"}]
