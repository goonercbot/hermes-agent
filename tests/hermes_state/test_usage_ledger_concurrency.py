"""Deterministically reproduce concurrent first-ledger initialization."""
import sqlite3
import threading

from hermes_state import SessionDB
from hermes_state_schema import SessionSchemaMixin


def test_concurrent_first_initializers_seed_once(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path/'home'))
    path = tmp_path/'state.db'
    db = SessionDB(db_path=path)
    db.create_session('legacy', source='cli')
    db.close()
    with sqlite3.connect(path) as conn:
        conn.execute('DELETE FROM usage_event_meta')
        conn.execute("INSERT INTO session_model_usage(session_id,model,first_seen,last_seen,input_tokens,api_call_count) VALUES('legacy','fixture-model',100,200,10,1)")
    barrier = threading.Barrier(2)
    errors = []

    class ConcurrentCursor(sqlite3.Cursor):
        def execute(self, sql, parameters=()):
            self.marker_query = sql.strip().startswith('SELECT value FROM usage_event_meta')
            return super().execute(sql, parameters)

        def fetchone(self):
            row = super().fetchone()
            if self.marker_query and not self.connection.in_transaction:
                # Both contenders see the old no-marker result before either
                # starts a write. A fixed implementation rechecks under lock.
                barrier.wait(timeout=5)
            return row

    def initialize():
        conn = sqlite3.connect(path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            SessionSchemaMixin()._initialize_usage_event_ledger(conn.cursor(factory=ConcurrentCursor))
            conn.commit()
        except Exception as exc:
            errors.append(f'{type(exc).__name__}: {exc}')
            conn.rollback()
        finally:
            conn.close()

    threads = [threading.Thread(target=initialize) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=12)
    assert not any(thread.is_alive() for thread in threads)
    assert not errors, errors
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT count(*) FROM usage_events').fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM usage_event_meta WHERE key='schema_version'").fetchone()[0] == 1
