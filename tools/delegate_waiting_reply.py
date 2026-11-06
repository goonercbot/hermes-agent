"""Process-local, explicit waiting-reply intent; this module never delivers or deletes messages.

A gateway may inherit a root only with independently proven continuation lineage. Missing
state, ambiguous roots and process restart must leave prior messages untouched.
"""
from dataclasses import dataclass
import threading
from typing import Any, Optional
from uuid import uuid4

from tools.registry import tool_error

_lock = threading.RLock()


@dataclass(frozen=True)
class DispatchIdentity:
    owner_id: str
    root_request_id: str


@dataclass(frozen=True)
class WaitingReply:
    turn_id: str
    root_request_id: str
    delegation_ids: tuple[str, ...]


def _turn_state(agent: Any) -> Optional[dict]:
    turn = getattr(agent, "_current_turn_id", None)
    if not isinstance(turn, str) or not turn:
        return None
    state = getattr(agent, "_waiting_reply_turn", None)
    if not isinstance(state, dict) or state["turn"] != turn:
        owner = getattr(agent, "_waiting_reply_owner", None) or uuid4().hex
        agent._waiting_reply_owner = owner
        state = {"turn": turn, "root": uuid4().hex, "owner": owner, "dispatched": False, "waiting": None}
        agent._waiting_reply_turn = state
    return state


def inherit_request_root(agent: Any, turn_id: str, root_request_id: str) -> bool:
    """Trusted runtime input before dispatch; never call with an inferred session root."""
    with _lock:
        state = _turn_state(agent)
        if (state is None or state["turn"] != turn_id or state["dispatched"] or state["waiting"] is not None
                or not isinstance(root_request_id, str) or not root_request_id):
            return False
        state["root"] = root_request_id
        return True


def dispatch_identity(agent: Any) -> Optional[DispatchIdentity]:
    """Capture an opaque request identity before admitting any units of a dispatch."""
    with _lock:
        state = _turn_state(agent)
        if state is None:
            return None
        state["dispatched"] = True
        return DispatchIdentity(state["owner"], state["root"])


def waiting_reply(agent: Any) -> Optional[WaitingReply]:
    """Return only this turn's opt-in; it is not proof of delivery or a complete final."""
    with _lock:
        state = getattr(agent, "_waiting_reply_turn", None)
        if not isinstance(state, dict) or state["turn"] != getattr(agent, "_current_turn_id", None):
            return None
        return state["waiting"]


def root_for_owned_units(agent: Any, delegation_ids: list[str]) -> Optional[str]:
    """Resolve admitted completion IDs in memory; caller must prove admission and route.

    Completed records may be pruned or absent after restart. Mixed roots fail closed.
    """
    from tools import async_delegation

    if (not isinstance(delegation_ids, list) or not delegation_ids
            or any(not isinstance(uid, str) or not uid for uid in delegation_ids)
            or len(set(delegation_ids)) != len(delegation_ids)):
        return None
    with _lock:
        owner = getattr(agent, "_waiting_reply_owner", None)
        if not owner:
            return None
        with async_delegation._records_lock:
            identities = [async_delegation._records.get(uid, {}).get("waiting_identity") for uid in delegation_ids]
            if any(not isinstance(identity, DispatchIdentity) or identity.owner_id != owner for identity in identities):
                return None
            roots = {identity.root_request_id for identity in identities}
            return next(iter(roots)) if len(roots) == 1 else None


def classify_waiting_reply(agent: Any, delegation_ids: Any) -> str:
    """Mark the upcoming pure waiting response only for exact live, owned units."""
    import json
    from tools import async_delegation

    with _lock:
        state = _turn_state(agent)
        if state is not None:
            state["waiting"] = None
        if (state is None or not isinstance(delegation_ids, list) or not delegation_ids
                or any(not isinstance(uid, str) or not uid for uid in delegation_ids)
                or len(set(delegation_ids)) != len(delegation_ids)):
            return tool_error("Provide a nonempty list of distinct live delegation unit IDs in an active turn.")
        identity = DispatchIdentity(state["owner"], state["root"])
        with async_delegation._records_lock:
            records = [async_delegation._records.get(uid) for uid in delegation_ids]
            if any(not record or record.get("status") != "running" or record.get("waiting_identity") != identity
                   for record in records):
                return tool_error("Every unit must be live, owned, and from this exact request root.")
            state["waiting"] = WaitingReply(state["turn"], state["root"], tuple(delegation_ids))
        return json.dumps({"status": "classified", "note": "Your next final must be only a temporary waiting status, not an answer."})
