"""Immutable timestamped usage evidence at the SessionDB accounting seam."""

import sqlite3
import threading

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A disposable store and Hermes home; no profile state is opened."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    result = SessionDB(db_path=tmp_path / "state.db")
    yield result
    result.close()


def _events(db):
    with db._lock:
        rows = db._conn.execute(
            "SELECT * FROM usage_events ORDER BY recorded_at, event_id"
        ).fetchall()
    return [dict(row) for row in rows]


def test_ledger_schema_stamps_immutable_meta(db):
    with db._lock:
        meta = dict(
            db._conn.execute("SELECT key, value FROM usage_event_meta").fetchall()
        )
        columns = {
            row["name"]
            for row in db._conn.execute("PRAGMA table_info(usage_events)").fetchall()
        }

    assert meta["schema_version"] == "1"
    assert float(meta["initialized_at"]) > 0
    assert {
        "event_id", "event_kind", "session_id", "parent_session_id",
        "session_source", "model", "billing_base_url", "task",
        "execution_role", "task_id", "occurred_at", "recorded_at",
        "interval_start", "interval_end", "timing_kind", "actual_cost_usd",
    } <= columns


def test_main_delta_preserves_explicit_role_task_and_zero_actual_cost(db):
    db.create_session("main", source="cli")
    db.update_token_counts(
        "main",
        input_tokens=12,
        output_tokens=3,
        model="model-a",
        billing_provider="provider-a",
        billing_base_url="https://secret:token@example.test/v1?api_key=leak",
        api_call_count=1,
        actual_cost_usd=0.0,
        execution_role="implementation",
        task_id="task-42",
        occurred_at=1_700_000_001.0,
    )

    events = _events(db)
    assert len(events) == 1
    event = events[0]
    assert event["event_kind"] == "delta"
    assert event["session_id"] == "main"
    assert event["session_source"] == "cli"
    assert event["execution_role"] == "implementation"
    assert event["task_id"] == "task-42"
    assert event["occurred_at"] == 1_700_000_001.0
    assert event["timing_kind"] == "point"
    assert event["actual_cost_usd"] == 0.0
    assert event["billing_base_url"] == "https://example.test"


def test_queued_coalescing_retains_one_delta_per_admission_and_timing(db):
    db.create_session("queued", source="cli")
    applied = []
    started = threading.Event()
    release = threading.Event()
    original = db.update_token_counts

    def gated(session_id, **kwargs):
        applied.append(kwargs)
        if len(applied) == 1:
            started.set()
            assert release.wait(timeout=10)
        return original(session_id, **kwargs)

    db.update_token_counts = gated
    try:
        db.queue_token_counts(
            "queued", input_tokens=2, api_call_count=1, occurred_at=1_700_000_010.0
        )
        assert started.wait(timeout=10)
        db.queue_token_counts(
            "queued", input_tokens=3, api_call_count=1, occurred_at=1_700_000_020.0
        )
        db.queue_token_counts(
            "queued", input_tokens=4, api_call_count=1, occurred_at=1_700_000_030.0
        )
        release.set()
        assert db.flush_token_counts()
    finally:
        db.update_token_counts = original

    # The final two SQL updates coalesced, while their retained evidence did not.
    assert len(applied) < 3
    events = _events(db)
    assert [event["input_tokens"] for event in events] == [2, 3, 4]
    assert [event["occurred_at"] for event in events] == [
        1_700_000_010.0,
        1_700_000_020.0,
        1_700_000_030.0,
    ]
    assert len({event["event_id"] for event in events}) == 3


def test_absolute_reconciliation_uses_persisted_transition_and_unknown_timing(db):
    db.create_session("absolute", source="gateway")
    db.update_token_counts("absolute", input_tokens=100, api_call_count=1)
    # A DB-level normalizer is the boundary: reconciliation must use the row
    # persisted after SQL, rather than arithmetic over the input kwargs.
    with db._lock:
        db._conn.execute(
            """CREATE TRIGGER normalize_absolute_input AFTER UPDATE OF input_tokens ON sessions
               WHEN NEW.id = 'absolute' AND NEW.input_tokens >= 125
               BEGIN
                   UPDATE sessions SET input_tokens = input_tokens + 5 WHERE id = NEW.id;
               END"""
        )
    db.update_token_counts("absolute", input_tokens=125, absolute=True)
    # An explicit lower cumulative total is a real correction, not a loss to
    # hide; it remains an unknown-timed signed ledger delta.
    db.update_token_counts("absolute", input_tokens=120, api_call_count=0, absolute=True)

    events = _events(db)
    assert [(event["input_tokens"], event["api_call_count"]) for event in events] == [
        (100, 1), (30, 0), (-10, -1),
    ]
    assert [event["timing_kind"] for event in events] == ["point", "unknown", "unknown"]
    assert events[1]["occurred_at"] is None
    assert events[2]["occurred_at"] is None


def test_usage_events_are_immutable_and_point_timing_requires_timestamp(db):
    db.create_session("immutable", source="cli")
    db.update_token_counts("immutable", input_tokens=1, api_call_count=1)
    event_id = _events(db)[0]["event_id"]
    with db._lock, pytest.raises(sqlite3.IntegrityError):
        db._conn.execute("UPDATE usage_events SET input_tokens = 2 WHERE event_id = ?", (event_id,))
    with db._lock, pytest.raises(sqlite3.IntegrityError):
        db._conn.execute("DELETE FROM usage_events WHERE event_id = ?", (event_id,))
    with db._lock, pytest.raises(sqlite3.IntegrityError):
        db._conn.execute(
            """INSERT INTO usage_events (
                   event_id, schema_version, event_kind, session_id, session_source,
                   model, execution_role, occurred_at, recorded_at, timing_kind
               ) VALUES ('bad-point', 1, 'delta', 'immutable', 'cli',
                         'model', 'unknown', NULL, 1.0, 'point')"""
        )


def test_transaction_retry_reuses_the_same_direct_event_id(db, monkeypatch):
    db.create_session("retry", source="cli")
    original = db._record_usage_events
    attempts = 0

    def fail_after_insert(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        original(*args, **kwargs)
        if attempts == 1:
            raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "_record_usage_events", fail_after_insert)
    monkeypatch.setattr(db, "_sleep_before_write_retry", lambda *args: True)
    db.update_token_counts("retry", input_tokens=7, api_call_count=1)

    assert attempts == 2
    events = _events(db)
    assert len(events) == 1
    assert events[0]["input_tokens"] == 7
    assert db.get_session("retry")["input_tokens"] == 7


def test_auxiliary_background_aggregate_is_explicitly_unknown_timing(db):
    db.create_session("aux", source="cron")
    db.record_auxiliary_usage(
        "aux", "background_review", model="review-model", api_call_count=4
    )

    events = _events(db)
    assert len(events) == 1
    event = events[0]
    assert event["task"] == "background_review"
    assert event["execution_role"] == "auxiliary"
    assert event["timing_kind"] == "unknown"
    assert event["occurred_at"] is None


def test_legacy_usage_rows_seed_baselines_once(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    path = tmp_path / "legacy.db"
    original = SessionDB(db_path=path)
    original.create_session("legacy", source="cli")
    original.update_token_counts(
        "legacy", input_tokens=9, output_tokens=2, model="legacy-model", api_call_count=1
    )
    original.close()

    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE usage_events")
    conn.execute("DROP TABLE usage_event_meta")
    conn.commit()
    conn.close()

    migrated = SessionDB(db_path=path)
    try:
        assert len(_events(migrated)) == 1
        assert _events(migrated)[0]["event_kind"] == "baseline"
        assert _events(migrated)[0]["input_tokens"] == 9
        migrated.close()
        reopened = SessionDB(db_path=path)
        try:
            assert len(_events(reopened)) == 1
        finally:
            reopened.close()
    finally:
        migrated.close()


def test_legacy_baseline_requires_both_interval_boundaries(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    path = tmp_path / "legacy-boundaries.db"
    original = SessionDB(db_path=path)
    original.create_session("legacy-boundaries", source="cli")
    original.update_token_counts(
        "legacy-boundaries", input_tokens=2, model="missing-boundary", api_call_count=1
    )
    original.update_token_counts(
        "legacy-boundaries", input_tokens=3, model="known-boundaries", api_call_count=1
    )
    original.close()

    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE usage_events")
    conn.execute("DROP TABLE usage_event_meta")
    conn.execute(
        "UPDATE session_model_usage SET first_seen = NULL, last_seen = 20 "
        "WHERE model = 'missing-boundary'"
    )
    conn.execute(
        "UPDATE session_model_usage SET first_seen = 10, last_seen = 20 "
        "WHERE model = 'known-boundaries'"
    )
    conn.commit()
    conn.close()

    migrated = SessionDB(db_path=path)
    try:
        baselines = {event["model"]: event for event in _events(migrated)}
        assert baselines["missing-boundary"]["timing_kind"] == "unknown"
        assert baselines["missing-boundary"]["interval_start"] is None
        assert baselines["missing-boundary"]["interval_end"] is None
        assert baselines["known-boundaries"]["timing_kind"] == "interval"
        assert baselines["known-boundaries"]["interval_start"] == 10
        assert baselines["known-boundaries"]["interval_end"] == 20
    finally:
        migrated.close()
