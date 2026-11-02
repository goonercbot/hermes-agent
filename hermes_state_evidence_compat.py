"""Compatibility with the dormant historical evidence schema, when present."""


def has_evidence_references(conn, *, session_id=None, row_id=None):
    """Check both immutable source pointers within the caller's write transaction."""
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='evidence_events'").fetchone() is None:
        return False
    if row_id is not None:
        return conn.execute(
            "SELECT 1 FROM evidence_events WHERE source_message_id=? OR source_call_message_id=? LIMIT 1",
            (row_id, row_id)).fetchone() is not None
    return conn.execute(
        "SELECT 1 FROM evidence_events e JOIN messages m ON "
        "(m.id=e.source_message_id OR m.id=e.source_call_message_id) "
        "WHERE m.session_id=? LIMIT 1", (session_id,)).fetchone() is not None