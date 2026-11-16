"""Focused extraction preserving the existing caller contract."""
from __future__ import annotations


def _local_completion_card(evt: dict) -> dict[str, str]:
    """The card of a local process notice: the async-result card the in-process TUI wrote for a
    completion or async delegation (a gateway-side ``/review`` reports as one), and a liveness tick's
    one-line receipt. The claim's ``message.start`` carries it, and an idle classic CLI only opens a
    turn that has one. ``{}`` for a watch match."""
    from tools.process_registry_notifications import (
        PROCESS_COMPLETE_DISPLAY_KIND, async_delegation_display_text, process_completion_display_text)
    evt_type = evt.get("type", "completion")
    if evt_type == "completion":
        return {"display_kind": PROCESS_COMPLETE_DISPLAY_KIND,
                "display_text": evt.get("_display_text") or process_completion_display_text([evt])}
    if evt_type == "async_delegation":
        return {"display_kind": "async_delegation_complete", "display_text": async_delegation_display_text(evt)}
    if evt_type == "heartbeat":
        # The wake stays machinery (no bubble, and a quiet tick may end in silence); the receipt lets an
        # idle CLI open the turn with one line and show the agent's reply.
        from gateway.response_filters import INTERNAL_NOTIFICATION_DISPLAY_KIND
        from tools.process_registry_notifications import heartbeat_display_text
        return {"display_kind": INTERNAL_NOTIFICATION_DISPLAY_KIND, "display_text": heartbeat_display_text(evt)}
    return {}
