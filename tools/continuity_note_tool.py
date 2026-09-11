"""Registered opt-in continuity-note schema for native incremental compaction.

The runtime binds the model call to the live agent and transcript in
``agent.native_incremental_handoff``.  Keeping that binding out of this module
means pasted JSON or a tool-name lookalike cannot manufacture authenticated
state: only the normal executor can call the host-side recorder.
"""
from __future__ import annotations

from tools.registry import registry

CONTINUITY_NOTE_TOOL_NAME = "continuity_note"
CONTINUITY_NOTE_AUTHENTICATOR = "native_incremental_continuity_note_v1"

CONTINUITY_NOTE_SCHEMA = {
    "name": CONTINUITY_NOTE_TOOL_NAME,
    "description": (
        "Record the current objective, plan, next action, and blockers for the "
        "opt-in native incremental continuity route. Use during active work; "
        "current user instructions always take precedence over this agent-authored note."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "objective": {"type": "string", "description": "Current work objective."},
            "current_plan": {"type": "string", "description": "Current concise plan."},
            "next_action": {"type": "string", "description": "Single next action."},
            "blockers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Current blockers, if any.",
            },
        },
        "required": ["objective", "current_plan", "next_action"],
        "additionalProperties": False,
    },
}


def _opted_in() -> bool:
    """Expose only when the profile explicitly enabled the new route."""
    try:
        from hermes_cli.config import load_config
        from utils import is_truthy_value

        cfg = load_config() or {}
        compression = cfg.get("compression") if isinstance(cfg, dict) else None
        return bool(
            isinstance(compression, dict)
            and is_truthy_value(compression.get("native_incremental_handoff", False))
        )
    except Exception:
        # A failed config read must not advertise an opt-in operational tool.
        return False


def _unbound_handler(_args, **_kwargs) -> str:
    # The ordinary AIAgent dispatcher supplies the authenticated live-agent
    # binding. Generic registry dispatch has no transcript ownership and must
    # fail closed rather than accepting a caller-provided source/history.
    return '{"error":"continuity_note requires an active AIAgent turn"}'


registry.register(
    name=CONTINUITY_NOTE_TOOL_NAME,
    toolset="continuity",
    schema=CONTINUITY_NOTE_SCHEMA,
    handler=_unbound_handler,
    check_fn=_opted_in,
    emoji="🧭",
)
