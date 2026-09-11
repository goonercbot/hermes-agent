"""Focused regression coverage for the extracted missing-native-note seam."""

from types import SimpleNamespace


def test_pending_native_maintenance_defers_only_blocked_context_warning():
    """A pending authenticated repair hides its stale-tail warning, then restores it."""
    from agent.conversation_loop import _suppress_pending_native_overflow_warning

    warnings = []
    agent = SimpleNamespace(
        _warn_context_overflow_blocked=lambda reason, tokens, threshold: warnings.append(
            (reason, tokens, threshold)
        )
    )

    with _suppress_pending_native_overflow_warning(agent, object()):
        agent._warn_context_overflow_blocked("structural_backoff:283", 287_761, 1_000)

    assert warnings == []
    agent._warn_context_overflow_blocked("structural_backoff:283", 287_761, 1_000)
    assert warnings == [("structural_backoff:283", 287_761, 1_000)]


def test_turn_start_compaction_defers_for_pending_native_note(monkeypatch):
    """The turn-start gate cannot warn or summarize before maintenance admission."""
    from agent import turn_context
    from agent import turn_context_compaction as compaction

    compressor = SimpleNamespace(
        protect_first_n=0,
        protect_last_n=0,
        threshold_tokens=1_000,
        last_real_prompt_tokens=10,
    )
    warnings = []
    agent = SimpleNamespace(
        compression_enabled=True,
        context_compressor=compressor,
        _turn_received_provider_response=False,
        _turn_preflight_display_snapshot=None,
        _warn_context_overflow_blocked=lambda *args: warnings.append(args),
    )
    out = compaction.CompactionOutcome(
        messages=[{"role": "user", "content": "stale tail"}],
        active_system_prompt=None,
        conversation_history=[],
        current_turn_user_idx=0,
    )
    monkeypatch.setattr(turn_context, "_review_fork_first_request_pending", lambda *_args: False)
    monkeypatch.setattr(turn_context, "_should_run_preflight_estimate", lambda *_args: True)
    monkeypatch.setattr(turn_context, "_preflight_request_tokens", lambda *_args: 2_000)
    monkeypatch.setattr(compaction, "_native_note_refresh_pending", lambda *_args: True)

    compaction._preflight_compression(agent, out, None, "current user", "task")

    assert warnings == []
    assert out.compressed is False


def test_native_note_admission_leaves_disabled_mode_on_ordinary_preflight():
    """Disabled native continuity cannot suppress ordinary compression warnings."""
    from agent.turn_preflight import _native_note_refresh_pending

    agent = SimpleNamespace(
        api_mode="codex_responses",
        native_incremental_handoff_enabled=False,
    )

    assert _native_note_refresh_pending(agent, [{"role": "user", "content": "history"}]) is False
