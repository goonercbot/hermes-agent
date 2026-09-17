"""Delegated workers never create detached gateway-owned completion routes."""

from types import SimpleNamespace
from unittest.mock import patch

from tools.delegate_tool_dispatch import _Batch, _dispatch_background


def _batch(platform: str) -> _Batch:
    return _Batch(
        task_list=[], children=[], parent_agent=SimpleNamespace(platform=platform),
        creds={}, context=None, top_role="leaf", max_children=1, live_deleg_id=None,
        live_writers=[], live_paths=[], origin_wake_sid="gateway-owner", origin_ui_session_id="",
        origin_owner_transport=None, origin_owner_session_record=None, overall_start=0.0,
    )


def test_subagent_origin_forces_sync_before_async_route_resolution():
    """Inherited gateway keys cannot enter the detached async registry from a child."""
    batch = _batch("subagent")
    with (
        patch("tools.delegate_tool_dispatch._run_sync_with_note", return_value="inline-result") as sync,
        patch("tools.delegate_tool_dispatch._resolve_async_wake_sid") as wake,
    ):
        assert _dispatch_background(batch) == "inline-result"

    sync.assert_called_once_with(batch, "subagent_origin")
    wake.assert_not_called()


def test_gateway_origin_still_uses_async_route_resolution():
    """The ownership guard does not disable ordinary gateway background delivery."""
    batch = _batch("telegram")
    with (
        patch("tools.delegate_tool_dispatch._resolve_async_wake_sid", return_value=None) as wake,
        patch("tools.delegate_tool_dispatch._run_sync_with_note", return_value="fallback") as sync,
    ):
        assert _dispatch_background(batch) == "fallback"

    wake.assert_called_once_with("gateway-owner")
    sync.assert_called_once_with(batch, "no_async")
