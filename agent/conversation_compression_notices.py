"""Focused extraction preserving the existing caller contract."""
from __future__ import annotations
from typing import Any


def _warn_summary_or_aux_fallback(agent: Any) -> None:
    """Surface a failed summary, or a recovered-but-broken aux compression model, once."""
    from agent.conversation_compression import logger
    summary_error = getattr(agent.context_compressor, "_last_summary_error", None)
    if summary_error:
        if getattr(agent, "_last_compression_summary_warning", None) != summary_error:
            agent._last_compression_summary_warning = summary_error
            agent._emit_warning(f"⚠ Compression summary failed: {summary_error}. Inserted a fallback context marker.")
    else:
        # Aux model may have errored and been recovered on main; tell the user their
        # auxiliary.compression.model is broken even though compression succeeded.
        _aux_fail_model = getattr(agent.context_compressor, "_last_aux_model_failure_model", None)
        _aux_fail_err = getattr(agent.context_compressor, "_last_aux_model_failure_error", None)
        # Dedup on (model, error) so we don't spam on every compaction
        _aux_key = (_aux_fail_model, _aux_fail_err)
        if _aux_fail_model and getattr(agent, "_last_aux_fallback_warning_key", None) != _aux_key:
            agent._last_aux_fallback_warning_key = _aux_key
            logger.warning(
                "Configured compression model %r failed (%s); recovered using the main model.",
                _aux_fail_model, _aux_fail_err or "unknown error",
            )
            agent._emit_warning(
                f"ℹ Configured compression model '{_aux_fail_model}' failed, so Hermes summarised "
                "with your main model instead. Check auxiliary.compression.model in your config."
            )
