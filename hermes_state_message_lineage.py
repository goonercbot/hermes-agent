"""Focused extraction preserving the existing caller contract."""
from __future__ import annotations


def _resume_lineage_ids(self, session_id: str) -> list[str]:
    """Session ids a display resume materializes: the VERIFIED compression lineage (root → tip; each
    hop's parent ended ``compression`` and the child is a genuine continuation), or the session alone
    for anything else — an explicit ``/branch`` copy, an API fork, a reset child, or a plain session.
    The raw parent walker ``_session_lineage_root_to_tip`` is intentionally NOT used here: it crosses
    fork/reset boundaries and misclassifies API forks (parent ended ``branched``, no marker) as
    compression lineages. Shared with the resume guard so it counts exactly what a resume loads."""
    if not session_id:
        return [session_id]
    session = self.get_session(session_id)
    if not session or self._is_explicit_fork_child_row(session):
        return [session_id]
    # _is_compression_child_row is the per-hop gate: the child must not be an explicit
    # branch/delegate/tool child AND its parent must have ended 'compression'.
    chain = [session_id]
    current = session
    seen = {session_id}
    while len(chain) < 100:  # defensive bound, same as _session_lineage_root_to_tip
        if not self._is_compression_child_row(current):
            break
        parent = self.get_session(current["parent_session_id"])
        if not parent or parent["id"] in seen:
            break
        seen.add(parent["id"])
        chain.append(parent["id"])
        current = parent
    return list(reversed(chain))
