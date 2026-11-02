"""Real SQLite acquisition deadlines, exact timeout restoration and recovery.

Adapted from the accepted v0.21.6 preservation suite and the v0.22.5 strict audit.
"""
import sqlite3
import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    with SessionDB(tmp_path / "state.db") as handle:
        yield handle


def timeout(db):
    return db._conn.execute("PRAGMA busy_timeout").fetchone()[0]


@pytest.mark.parametrize("configured", [0, 237, 1000])
def test_success_restores_exact_timeout(db, configured):
    db._conn.execute(f"PRAGMA busy_timeout={configured}")
    db.set_meta("success", "yes")
    assert timeout(db) == configured


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_callback_failure_rolls_back_and_restores_timeout(db, failure):
    db._conn.execute("PRAGMA busy_timeout=337")

    def fail(conn):
        conn.execute("INSERT INTO state_meta(key,value) VALUES('probe','discard')")
        raise failure("synthetic failure")

    with pytest.raises(failure):
        db._execute_write(fail)
    assert db.get_meta("probe") is None
    assert not db._conn.in_transaction
    assert timeout(db) == 337
    db.set_meta("probe", "recovered")
    assert db.get_meta("probe") == "recovered"


@pytest.mark.parametrize("patience", [0, 0.1])
def test_real_acquisition_is_bounded_and_same_connection_recovers(db, patience):
    conn = db._conn
    conn.execute("PRAGMA busy_timeout=4000")
    holder = sqlite3.connect(str(db.db_path), isolation_level=None)
    try:
        holder.execute("BEGIN IMMEDIATE")
        start = time.monotonic()
        with pytest.raises(sqlite3.OperationalError, match="another Hermes process"):
            db._execute_write(
                lambda c: c.execute("INSERT INTO state_meta(key,value) VALUES('probe','no')"),
                patience_s=patience,
            )
        elapsed = time.monotonic() - start
    finally:
        holder.close()
    assert db.get_meta("probe") is None
    assert not conn.in_transaction
    assert timeout(db) == 4000
    db.set_meta("probe", "recovered")
    assert db._conn is conn
    assert db.get_meta("probe") == "recovered"
    assert timeout(db) == 4000
    print(f"ACQUISITION requested={patience}s elapsed={elapsed:.6f}s; same-handle recovery")
    assert elapsed < 2, f"requested={patience}, elapsed={elapsed}"


def test_reopen_preserves_new_connection_timeout(db):
    original = db._conn
    db.close()
    db.set_meta("reopen", "yes")
    assert db._conn is not original
    assert timeout(db) == 1000


def test_busy_timeout_restore_failure_rolls_back_before_callback(db):
    conn = db._conn
    calls = []

    class RestoreFailure:
        def execute(self, sql, *args):
            if sql == "PRAGMA busy_timeout=1000":
                raise RuntimeError("restore failed")
            return conn.execute(sql, *args)

        def rollback(self):
            return conn.rollback()

    db._conn = RestoreFailure()
    try:
        with pytest.raises(RuntimeError, match="restore failed"):
            db._execute_write(lambda c: calls.append(c))
    finally:
        db._conn = conn
        conn.execute("PRAGMA busy_timeout=1000")
    assert calls == []
    assert not conn.in_transaction
    db.set_meta("restore-recovery", "yes")
    assert db.get_meta("restore-recovery") == "yes"
