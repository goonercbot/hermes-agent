"""Logical history remains coupled to the native aggregates."""
import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import _worker_usage


def test_native_delta_absolute_auxiliary_and_reopen(tmp_path):
    path = tmp_path / 'state.db'
    db = SessionDB(path)
    db.update_token_counts('s', input_tokens=11, output_tokens=3, api_call_count=1,
                           model='m', cost_status='estimated', cost_source='native')
    db.record_auxiliary_usage('s', 'vision', model='a', input_tokens=5)
    db._execute_write(lambda c: _worker_usage(db, c, 's', {'task': 'vision', 'model': 'a', 'input_tokens': 7}, auxiliary=True))
    db._execute_write(lambda c: _worker_usage(db, c, 's', {'model': 'm', 'input_tokens': 2}, auxiliary=False))
    db.update_token_counts('s', input_tokens=4, output_tokens=3, api_call_count=1, absolute=True)
    rows = [dict(r) for r in db._conn.execute('select * from usage_events')]
    assert len(rows) == 5
    assert rows[0]['timing_kind'] == 'point' and rows[0]['cost_status'] == 'unknown'
    assert 'estimated' in rows[0]['cost_source']
    assert rows[2]['timing_kind'] == rows[3]['timing_kind'] == rows[4]['timing_kind'] == 'unknown'
    assert rows[4]['input_tokens'] == -9
    ids = {r['event_id'] for r in rows}
    db.close()
    reopened = SessionDB(path)
    assert {r[0] for r in reopened._conn.execute('select event_id from usage_events')} == ids
    reopened.close()


@pytest.mark.parametrize('kept', [None, 'schema_version', 'initialized_at'])
def test_adopt_existing_events_without_inventing_aggregate_coverage(tmp_path, kept):
    path = tmp_path / 'state.db'
    db = SessionDB(path)
    db.update_token_counts('s', input_tokens=3, model='m')
    db.close()
    with sqlite3.connect(path) as conn:
        conn.execute('UPDATE session_model_usage SET input_tokens=100')
        conn.execute('DELETE FROM usage_event_meta WHERE key != ?', (kept or '',))
        before_events = conn.execute('SELECT * FROM usage_events').fetchall()
        before_routes = conn.execute('SELECT * FROM session_model_usage').fetchall()
        before_sessions = conn.execute('SELECT * FROM sessions').fetchall()
        original_meta = dict(conn.execute('SELECT key,value FROM usage_event_meta'))
    for _ in range(2):
        db = SessionDB(path)
        assert [tuple(r) for r in db._conn.execute('SELECT * FROM usage_events')] == before_events
        assert [tuple(r) for r in db._conn.execute('SELECT * FROM session_model_usage')] == before_routes
        assert [tuple(r) for r in db._conn.execute('SELECT * FROM sessions')] == before_sessions
        meta = dict(db._conn.execute('SELECT key,value FROM usage_event_meta'))
        assert all(meta[k] == v for k, v in original_meta.items())
        assert meta['legacy_adoption'] == 'existing_events_coverage_unknown'
        assert db._conn.execute('SELECT sum(input_tokens) FROM usage_events').fetchone()[0] == 3
        assert db._conn.execute('SELECT sum(input_tokens) FROM session_model_usage').fetchone()[0] == 100
        db.close()
    db = SessionDB(path)
    db.update_token_counts('s', input_tokens=2, model='m')
    events = [tuple(r) for r in db._conn.execute('SELECT * FROM usage_events ORDER BY rowid')]
    assert events[:1] == before_events and len(events) == 2
    assert db._conn.execute('SELECT sum(input_tokens) FROM usage_events').fetchone()[0] == 5
    db.close()


@pytest.mark.parametrize('key,value', [('schema_version', '2'), ('initialized_at', 'nan')])
def test_invalid_existing_marker_refused_without_mutation(tmp_path, key, value):
    from hermes_state_usage_history import initialize
    path = tmp_path / 'state.db'
    db = SessionDB(path)
    db.update_token_counts('s', input_tokens=1)
    db.close()
    with sqlite3.connect(path) as conn:
        conn.execute('DELETE FROM usage_event_meta')
        conn.execute('INSERT INTO usage_event_meta VALUES (?,?)', (key, value))
        before = conn.execute('SELECT * FROM usage_events').fetchall()
    with sqlite3.connect(path) as conn:
        with pytest.raises(ValueError):
            initialize(conn)
        assert conn.execute('SELECT * FROM usage_event_meta').fetchall() == [(key, value)]
        assert conn.execute('SELECT * FROM usage_events').fetchall() == before


def test_adoption_metadata_rolls_back_together(tmp_path):
    from hermes_state_usage_history import initialize
    path = tmp_path / 'state.db'
    db = SessionDB(path)
    db.update_token_counts('s', input_tokens=1)
    db.close()
    with sqlite3.connect(path) as conn:
        conn.execute('DELETE FROM usage_event_meta')
        conn.execute("CREATE TRIGGER reject_initialization BEFORE INSERT ON usage_event_meta WHEN NEW.key='initialized_at' BEGIN SELECT RAISE(ABORT, 'blocked'); END")
        before = conn.execute('SELECT * FROM usage_events').fetchall()
    with sqlite3.connect(path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match='blocked'):
            initialize(conn)
        assert conn.execute('SELECT * FROM usage_event_meta').fetchall() == []
        assert conn.execute('SELECT * FROM usage_events').fetchall() == before


def test_atomic_rollback(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    db.update_token_counts('s', input_tokens=1)
    try:
        def write(conn):
            db._update_token_counts_in_transaction(conn, 's', input_tokens=4)
            raise RuntimeError('rollback')
        try:
            db._execute_write(write)
        except RuntimeError:
            pass
        assert db._conn.execute('select input_tokens from sessions where id="s"').fetchone()[0] == 1
        assert db._conn.execute('select count(*) from usage_events').fetchone()[0] == 1
    finally:
        db.close()


def test_legacy_aggregate_is_baselined_once(tmp_path):
    from hermes_state_common import SCHEMA_SQL
    path = tmp_path / 'state.db'
    before, rest = SCHEMA_SQL.split('CREATE TABLE IF NOT EXISTS usage_events (', 1)
    _, after = rest.split('CREATE TABLE IF NOT EXISTS state_meta (', 1)
    with sqlite3.connect(path) as conn:
        conn.executescript(before + 'CREATE TABLE IF NOT EXISTS state_meta (' + after)
        conn.execute("INSERT INTO sessions(id, source, started_at) VALUES ('s','cli',100)")
        conn.execute("""INSERT INTO session_model_usage
            (session_id,model,billing_provider,billing_base_url,billing_mode,task,input_tokens,
             api_call_count,first_seen,last_seen,cost_status,cost_source)
            VALUES ('s','m','','','','',10,1,100,200,'estimated','legacy')""")
        conn.execute("""INSERT INTO session_model_usage
            (session_id,model,billing_provider,billing_base_url,billing_mode,task,input_tokens)
            VALUES ('s','correction','','','','',-2)""")
    db = SessionDB(path)
    rows = {row['model']: dict(row) for row in db._conn.execute('select * from usage_events')}
    assert len(rows) == 2 and rows['m']['event_kind'] == 'baseline'
    assert rows['m']['input_tokens'] == 10 and rows['m']['timing_kind'] == 'interval'
    assert rows['m']['cost_status'] == 'unknown' and 'estimated' in rows['m']['cost_source']
    assert (rows['correction']['event_kind'], rows['correction']['timing_kind'], rows['correction']['input_tokens']) == ('delta', 'unknown', -2)
    db.close()
    db = SessionDB(path)
    assert db._conn.execute('select count(*) from usage_events').fetchone()[0] == 2
    db.close()


def test_public_queue_admits_distinct_equal_deltas(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    try:
        db.queue_token_counts('s', input_tokens=2, model='m')
        db.queue_token_counts('s', input_tokens=2, model='m')
        assert db.flush_token_counts(timeout=5)
        rows = [dict(r) for r in db._conn.execute('SELECT * FROM usage_events')]
        assert len(rows) == 2
        assert len({r['event_id'] for r in rows}) == 2
        assert all(r['input_tokens'] == 2 and r['timing_kind'] == 'point' and r['occurred_at'] is not None for r in rows)
        assert db.get_session('s')['input_tokens'] == 4
    finally:
        db.close()


def test_coalesced_parts_keep_individual_times_and_identity(tmp_path):
    from hermes_state_usage_history import new_part
    db = SessionDB(tmp_path / 'state.db')
    parts = [new_part({'input_tokens': 2}, occurred_at=100), new_part({'input_tokens': 2}, occurred_at=200)]
    batch = [('s', {'input_tokens': 2, '_usage_event_parts': [part]}) for part in parts]
    merged = db._coalesce_token_deltas(batch)
    assert len(merged) == 1 and len(batch[0][1]['_usage_event_parts']) == 1
    db._apply_token_batch(batch)
    rows = list(db._conn.execute('select event_id,occurred_at,input_tokens from usage_events order by occurred_at'))
    assert [(row[1], row[2]) for row in rows] == [(100, 2), (200, 2)]
    assert {row[0] for row in rows} == {part['event_id'] for part in parts}
    db.close()
