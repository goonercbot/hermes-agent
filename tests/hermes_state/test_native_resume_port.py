"""Persisted native-compaction resume boundaries."""

import sqlite3

from hermes_state import SessionDB


def test_native_checkpoint_handoff_roundtrip_preserves_sealed_rows_and_repairs_ordinary_tail(tmp_path):
    """A real SQLite reload keeps a sealed native handoff byte-exact.

    The ordinary consecutive-user tail still takes the public resume repair;
    only rows authenticated by the checkpoint carrier bypass normalization.
    """
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    metadata = {
        "version": 2,
        "identity": "native-checkpoint",
        "handoff": "authenticated handoff",
        "tail_count": 1,
        "tail_fence": "sealed tail fence",
    }
    try:
        db.create_session("native-resume", "cli", model="gpt-5.6-luna")
        db.append_message(
            "native-resume", "assistant", "",
            codex_reasoning_items=[{
                "type": "compaction",
                "encrypted_content": "sealed-checkpoint",
                "_hermes_native_compaction": metadata,
            }],
        )
        db.append_message("native-resume", "user", "  sealed handoff  ")
        db.append_message("native-resume", "assistant", "  sealed tail  ")
        db.append_message("native-resume", "user", "  ordinary first  ")
        db.append_message("native-resume", "user", "  ordinary second  ")

        # Inspect the physical persisted values before any model projection.
        with sqlite3.connect(path) as conn:
            raw_rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
                ("native-resume",),
            ).fetchall()
        assert raw_rows == [
            ("assistant", ""),
            ("user", "  sealed handoff  "),
            ("assistant", "  sealed tail  "),
            ("user", "  ordinary first  "),
            ("user", "  ordinary second  "),
        ]

        canonical = db.get_messages_as_conversation("native-resume", repair_alternation=False)
        resumed, _display = db.get_resume_conversations("native-resume")
    finally:
        db.close()

    assert [row["content"] for row in canonical] == [
        "", "  sealed handoff  ", "  sealed tail  ", "ordinary first", "ordinary second",
    ]
    assert [row["content"] for row in resumed] == [
        "", "  sealed handoff  ", "  sealed tail  ", "ordinary first\n\nordinary second",
    ]
    assert resumed[0]["codex_reasoning_items"] == [{
        "type": "compaction",
        "encrypted_content": "sealed-checkpoint",
        "_hermes_native_compaction": metadata,
    }]
