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


@dataclass(frozen=True)
class ResolvingReply:
    turn_id: str
    identity: DispatchIdentity
    delegation_ids: tuple[str, ...]


def _turn_state(agent: Any) -> Optional[dict]:
    turn = getattr(agent, "_current_turn_id", None)
    if not isinstance(turn, str) or not turn:
        return None
    state = getattr(agent, "_waiting_reply_turn", None)
    if not isinstance(state, dict) or state["turn"] != turn:
        owner = getattr(agent, "_waiting_reply_owner", None) or uuid4().hex
        agent._waiting_reply_owner = owner
        state = {"turn": turn, "root": uuid4().hex, "owner": owner, "dispatched": False, "waiting": None, "resolution": None}
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
        state["resolution"] = None
        state["waiting"] = None
        return DispatchIdentity(state["owner"], state["root"])


def invalidate_reply_intent(agent):
    with _lock:
        state = _turn_state(agent)
        if state is not None:
            state['waiting'] = None
            state['resolution'] = None


def adopt_dispatch_identity(agent: Any, turn_id: str, identity: DispatchIdentity) -> bool:
    """Adopt a proven owner and root before the first dispatch of this turn."""
    with _lock:
        state = _turn_state(agent)
        if (type(identity) is not DispatchIdentity or not identity.owner_id or not identity.root_request_id
                or state is None or state['turn'] != turn_id or state['dispatched'] or state['waiting'] is not None):
            return False
        agent._waiting_reply_owner = identity.owner_id
        state['owner'], state['root'] = identity.owner_id, identity.root_request_id
        return True


def waiting_reply(agent: Any) -> Optional[WaitingReply]:
    """Return only this turn's opt-in; it is not proof of delivery or a complete final."""
    with _lock:
        state = getattr(agent, "_waiting_reply_turn", None)
        if not isinstance(state, dict) or state["turn"] != getattr(agent, "_current_turn_id", None):
            return None
        return state["waiting"]


def resolving_reply(agent: Any) -> Optional[ResolvingReply]:
    with _lock:
        state = getattr(agent, '_waiting_reply_turn', None)
        if not isinstance(state, dict) or state['turn'] != getattr(agent, '_current_turn_id', None):
            return None
        return state.get('resolution')


def completed_owned(identity, ids):
    """All supplied units still exist and no unit of this identity remains active."""
    from tools import async_delegation
    if not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        return False
    with async_delegation._records_lock:
        if any(not (r := async_delegation._records.get(i)) or r.get('status') != 'completed'
               or r.get('waiting_identity') != identity for i in ids):
            return False
        return not any(r.get('waiting_identity') == identity and r.get('status') in ('running', 'finalizing')
                       for r in async_delegation._records.values())


def resolve_waiting_reply(agent, delegation_ids):
    import json
    with _lock:
        state = _turn_state(agent)
        if state is not None:
            state['resolution'] = None
            state['waiting'] = None
        if state is None or not isinstance(delegation_ids, list):
            return tool_error('Provide completed owned unit IDs in an active turn.')
        identity = DispatchIdentity(state['owner'], state['root'])
        if not completed_owned(identity, delegation_ids):
            return tool_error('Units must be completed, owned, and from this root; no root units may remain active.')
        state['resolution'] = ResolvingReply(state['turn'], identity, tuple(delegation_ids))
        return json.dumps({'status': 'resolved', 'note': 'Your next final resolves this request.'})


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
            state["resolution"] = None
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
