"""Ephemeral completion lineage. Never derived from message metadata or persisted."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import Optional

from tools.delegate_waiting_reply import DispatchIdentity


@dataclass
class CompletionProof:
    identity: DispatchIdentity
    session_id: str
    route: str
    delegation_ids: tuple[str, ...]
    _used: bool = False

    def take(self, session_id, route):
        with _proof_lock:
            if self._used or (self.session_id, self.route) != (session_id, route):
                return None
            self._used = True
            return self.identity


_proof_lock = Lock()
_current_proof: ContextVar[Optional[tuple[DispatchIdentity, str]]] = ContextVar('waiting_lineage_proof', default=None)


def prove_completion(ids, session_id, route):
    """Only call after native claim/owner preflight, on the actually claimed IDs."""
    from tools import async_delegation
    if (not session_id or not route or not ids or len(set(ids)) != len(ids)
            or any(not isinstance(i, str) or not i for i in ids)):
        return None
    with async_delegation._records_lock:
        records = [async_delegation._records.get(i) for i in ids]
        if any(not r or r.get('parent_session_id') != session_id or r.get('session_key') != route
               for r in records):
            return None
        identities = [r.get('waiting_identity') for r in records]
        identity = identities[0]
        if (not isinstance(identity, DispatchIdentity) or not identity.owner_id or not identity.root_request_id
                or any(i != identity for i in identities)):
            return None
        return CompletionProof(identity, session_id, route, tuple(ids))


@contextmanager
def execution_proof(proof, session_id, route):
    identity = None
    if type(proof) is CompletionProof:
        # A queued proof cannot outlive its in-memory evidence (pruning/restart).
        fresh = prove_completion(list(proof.delegation_ids), session_id, route)
        if fresh is not None and fresh.identity == proof.identity:
            identity = proof.take(session_id, route)
    token = _current_proof.set((identity, session_id) if identity else None)
    try:
        yield
    finally:
        _current_proof.reset(token)


def adopt_bound_turn(agent, turn_id):
    value = _current_proof.get()
    if value is None or getattr(agent, 'session_id', None) != value[1]:
        return False
    from tools.delegate_waiting_reply import adopt_dispatch_identity
    _current_proof.set(None)
    return adopt_dispatch_identity(agent, turn_id, value[0])
