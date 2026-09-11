"""Focused regression coverage for the immutable timestamped usage ledger."""

import sqlite3

import pytest

from hermes_state import SessionDB


def _events(db, session_id):
    return [dict(row) for row in db._conn.execute(
        "SELECT * FROM usage_events WHERE session_id = ? ORDER BY recorded_at, event_id", (session_id,)
    ).fetchall()]


def test_direct_queued_absolute_reconciliation_and_immutable_rows(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("usage", source="cli")

    db.update_token_counts(
        "usage", input_tokens=10, output_tokens=2, api_call_count=1,
        model="main", billing_provider="provider",
        billing_base_url="https://token@example.test/secret?key=nope", occurred_at=123.0,
    )
    db.queue_token_counts("usage", input_tokens=3, api_call_count=1, model="main", billing_provider="provider")
    db.queue_token_counts("usage", input_tokens=4, api_call_count=1, model="main", billing_provider="provider")
    assert db.flush_token_counts()

    # Cumulative gateway reports reconcile the actual row transition, including a negative correction.
    db.update_token_counts("usage", input_tokens=20, output_tokens=2, api_call_count=3, absolute=True)
    db.update_token_counts("usage", input_tokens=17, output_tokens=2, api_call_count=3, absolute=True)
    rows = _events(db, "usage")
    deltas = [row for row in rows if row["event_kind"] == "delta"]
    assert len(deltas) == 5  # one direct, two queued admissions, two absolute reconciliations
    assert deltas[0]["occurred_at"] == 123.0
    assert deltas[0]["billing_base_url"] == "https://example.test"
    assert [row["input_tokens"] for row in deltas[-2:]] == [3, -3]
    assert [row["timing_kind"] for row in deltas[-2:]] == ["unknown", "unknown"]

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db._conn.execute("UPDATE usage_events SET input_tokens = 0 WHERE event_id = ?", (deltas[0]["event_id"],))
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db._conn.execute("DELETE FROM usage_events WHERE event_id = ?", (deltas[0]["event_id"],))
    event_ids = [row["event_id"] for row in deltas]
    db.close()

    # A store already carrying the ledger retains both immutable rows and its marker.
    reopened = SessionDB(tmp_path / "state.db")
    assert [row["event_id"] for row in _events(reopened, "usage")] == event_ids
    assert reopened._conn.execute(
        "SELECT value FROM usage_event_meta WHERE key = 'schema_version'"
    ).fetchone()[0] == "1"
    reopened.close()


def test_existing_aggregate_baselines_seed_exactly_once_and_preserve_sessions_messages(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("legacy", source="cli")
    db.append_message("legacy", "user", content="preserved message")
    db.update_token_counts("legacy", input_tokens=42, api_call_count=1, model="legacy-model")
    db.close()

    # Simulate a state.db from immediately before the ledger migration: keep all legacy data.
    conn = sqlite3.connect(path)
    conn.executescript("""
        DROP TRIGGER usage_events_no_update;
        DROP TRIGGER usage_events_no_delete;
        DROP TABLE usage_events;
        DROP TABLE usage_event_meta;
    """)
    conn.commit()
    conn.close()

    migrated = SessionDB(path)
    baseline = migrated._conn.execute(
        "SELECT event_kind, input_tokens FROM usage_events WHERE session_id = 'legacy'"
    ).fetchall()
    assert [tuple(row) for row in baseline] == [("baseline", 42)]
    assert migrated._conn.execute("SELECT content FROM messages WHERE session_id = 'legacy'").fetchone()[0] == "preserved message"
    migrated.close()

    reopened = SessionDB(path)
    assert reopened._conn.execute(
        "SELECT COUNT(*) FROM usage_events WHERE session_id = 'legacy'"
    ).fetchone()[0] == 1
    assert reopened._conn.execute(
        "SELECT value FROM usage_event_meta WHERE key = 'schema_version'"
    ).fetchone()[0] == "1"
    reopened.close()


def test_auxiliary_interval_event_retains_task_and_signed_actual_cost(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.record_auxiliary_usage(
        "aux", "background_review", model="reviewer", input_tokens=5,
        actual_cost_usd=-0.25, execution_role="independent_review", task_id="review-1",
        interval_start=10.0, interval_end=20.0,
    )
    row = _events(db, "aux")[0]
    assert (row["task"], row["execution_role"], row["task_id"]) == ("background_review", "independent_review", "review-1")
    assert (row["timing_kind"], row["occurred_at"], row["interval_start"], row["interval_end"]) == ("interval", None, 10.0, 20.0)
    assert row["actual_cost_usd"] == -0.25
    db.close()
