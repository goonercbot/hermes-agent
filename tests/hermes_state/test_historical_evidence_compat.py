"""Synthetic compatibility tests for dormant historical evidence references."""
import hashlib
import json
import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_errors import PruneRowUnresolvedError

SCHEMA = """
CREATE TABLE evidence_events (
 id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id),
 source_message_id INTEGER NOT NULL REFERENCES messages(id),
 source_sha256 TEXT NOT NULL CHECK(length(source_sha256)=64),
 source_call_message_id INTEGER REFERENCES messages(id),
 source_call_sha256 TEXT CHECK(source_call_sha256 IS NULL OR length(source_call_sha256)=64),
 tool_call_id TEXT, tool_name TEXT NOT NULL, event_type TEXT NOT NULL,
 lifecycle_dimension TEXT NOT NULL, lifecycle_state TEXT NOT NULL, outcome TEXT NOT NULL,
 issuer TEXT NOT NULL, adapter TEXT NOT NULL, adapter_version TEXT NOT NULL,
 subject_type TEXT NOT NULL, subject_id TEXT NOT NULL, candidate_id TEXT,
 parent_candidate_id TEXT, artifact_id TEXT, metadata_json TEXT NOT NULL DEFAULT '{}',
 occurred_at REAL NOT NULL, supersedes_event_id INTEGER, previous_event_hash TEXT,
 event_hash TEXT NOT NULL UNIQUE CHECK(length(event_hash)=64),
 trusted INTEGER NOT NULL DEFAULT 0 CHECK(trusted IN (0,1)),
 freshness_scope TEXT NOT NULL DEFAULT 'historical',
 UNIQUE(session_id,source_message_id,event_type,issuer),
 FOREIGN KEY(supersedes_event_id) REFERENCES evidence_events(id));
CREATE TABLE evidence_retention_deletions (session_id TEXT PRIMARY KEY);
CREATE TRIGGER evidence_events_no_update BEFORE UPDATE ON evidence_events BEGIN
 SELECT RAISE(ABORT, 'evidence_events are append-only'); END;
CREATE TRIGGER evidence_events_no_delete BEFORE DELETE ON evidence_events WHEN NOT EXISTS
 (SELECT 1 FROM evidence_retention_deletions WHERE session_id=OLD.session_id) BEGIN
 SELECT RAISE(ABORT, 'evidence_events are append-only'); END;
"""


SOURCE_FIELDS = ('id', 'session_id', 'role', 'content', 'tool_call_id', 'tool_calls',
                 'tool_name', 'effect_disposition', 'timestamp')


def historical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def source_digest(conn, row_id):
    row = conn.execute('SELECT ' + ','.join(SOURCE_FIELDS) + ' FROM messages WHERE id=?', (row_id,)).fetchone()
    return historical_hash(dict(zip(SOURCE_FIELDS, row)))


def add_chain(conn, result_id, call_id):
    previous = None
    for event_type in ('tool_result', 'lifecycle'):
        record = dict(session_id='s', source_message_id=result_id,
                      source_sha256=source_digest(conn, result_id), source_call_message_id=call_id,
                      source_call_sha256=source_digest(conn, call_id), tool_call_id='c', tool_name='search',
                      event_type=event_type, lifecycle_dimension='test', lifecycle_state='done', outcome='ok',
                      issuer='test', adapter='test', adapter_version='1', subject_type='test', subject_id='test',
                      candidate_id=None, parent_candidate_id=None, artifact_id=None, metadata_json='{}',
                      occurred_at=1.0, supersedes_event_id=None, previous_event_hash=previous,
                      trusted=0, freshness_scope='historical')
        record['event_hash'] = historical_hash(record)
        conn.execute('INSERT INTO evidence_events (' + ','.join(record) + ') VALUES (' +
                     ','.join('?' for _ in record) + ')', tuple(record.values()))
        previous = record['event_hash']
    conn.commit()


def assert_chain(conn):
    previous = None
    for raw in conn.execute('SELECT * FROM evidence_events ORDER BY id').fetchall():
        event = dict(raw)
        assert event.pop('id') > 0
        digest = event.pop('event_hash')
        assert event['previous_event_hash'] == previous
        assert event['source_sha256'] == source_digest(conn, event['source_message_id'])
        assert event['source_call_sha256'] == source_digest(conn, event['source_call_message_id'])
        assert historical_hash(event) == digest
        previous = digest
    assert previous is not None


def make_db(path):
    db = SessionDB(path)
    db.create_session('s', source='cli')
    db.append_messages_batch('s', [
        {'role': 'assistant', 'content': 'call', 'tool_calls': [{'id': 'c', 'type': 'function', 'function': {'name': 'search', 'arguments': '{}'}}]},
        {'role': 'tool', 'content': 'original result', 'tool_call_id': 'c', 'tool_name': 'search'},
    ])
    conn = db._conn
    conn.executescript(SCHEMA)
    rows = conn.execute("SELECT id,content,tool_calls FROM messages WHERE session_id='s' ORDER BY id").fetchall()
    add_chain(conn, rows[1][0], rows[0][0])
    assert_chain(conn)
    return db, [tuple(row) for row in rows]


def test_replace_preserves_referenced_source_and_archived_rows(tmp_path):
    db, original = make_db(tmp_path / 'state.db')
    db.replace_messages('s', [{'role':'user','content':'new'}])
    assert [m['content'] for m in db.get_messages_as_conversation('s')] == ['new']
    db.replace_messages('s', [{'role':'user','content':'next'}])
    assert [tuple(r) for r in db._conn.execute('SELECT id,content,tool_calls FROM messages WHERE id IN (?,?) ORDER BY id', (original[0][0],original[1][0]))] == original
    assert_chain(db._conn)
    assert not db._conn.execute('PRAGMA foreign_key_check').fetchall()
    db.close()
    db = SessionDB(tmp_path / 'state.db')
    assert [m['content'] for m in db.get_messages_as_conversation('s')] == ['next']
    assert db.search_messages('original result', include_inactive=True)
    db.close()


@pytest.mark.parametrize('index', [0, 1])
def test_prune_refuses_referenced_call_and_result_atomically(tmp_path, index):
    db, original = make_db(tmp_path / 'state.db')
    messages = db.get_messages_as_conversation('s', include_row_ids=True)
    selected = messages[index]
    with pytest.raises(PruneRowUnresolvedError):
        db.rewrite_pruned_rows('s', [(selected, {**selected, 'content':'pruned'})])
    assert [tuple(r) for r in db._conn.execute('SELECT id,content,tool_calls FROM messages WHERE session_id=? ORDER BY id', ('s',))] == original
    assert_chain(db._conn)
    assert not db._conn.execute('PRAGMA foreign_key_check').fetchall()
    db.close()


def test_archived_row_stale_flush_inserts_without_mutating_archive(tmp_path):
    from agent.message_metadata import DB_ROW_SNAPSHOT
    from agent.transcript_repair import transcript_row_snapshot

    db = SessionDB(tmp_path / 'state.db')
    db.create_session('s', source='cli')
    row_id = db.append_message('s', 'user', 'original')
    stale = db.get_messages_as_conversation('s', include_row_ids=True)[0]
    stale[DB_ROW_SNAPSHOT] = transcript_row_snapshot(
        db._conn.execute('SELECT * FROM messages WHERE id=?', (row_id,)).fetchone())
    assert db._conn.execute("SELECT 1 FROM sqlite_master WHERE name='evidence_events'").fetchone() is None
    db.replace_messages('s', [], archive_dropped=True)
    stale['content'] = 'stale edit'
    assert db.append_messages_batch('s', [stale]) == 1
    assert tuple(db._conn.execute('SELECT content, active FROM messages WHERE id=?', (row_id,)).fetchone()) == ('original', 0)
    assert [m['content'] for m in db.get_messages_as_conversation('s')] == ['stale edit']
    db.close()


def test_active_row_flush_remains_mutable(tmp_path):
    from agent.message_metadata import DB_ROW_SNAPSHOT
    from agent.transcript_repair import transcript_row_snapshot

    db = SessionDB(tmp_path / 'state.db')
    db.create_session('s', source='cli')
    row_id = db.append_message('s', 'user', 'original')
    message = db.get_messages_as_conversation('s', include_row_ids=True)[0]
    message[DB_ROW_SNAPSHOT] = transcript_row_snapshot(
        db._conn.execute('SELECT * FROM messages WHERE id=?', (row_id,)).fetchone())
    message['content'] = 'active edit'
    assert db.append_messages_batch('s', [message]) == 0
    assert tuple(db._conn.execute('SELECT content, active FROM messages WHERE id=?', (row_id,)).fetchone()) == ('active edit', 1)
    db.close()


def test_archived_evidence_row_stale_flush_keeps_chain(tmp_path):
    from agent.message_metadata import DB_ROW_SNAPSHOT
    from agent.transcript_repair import transcript_row_snapshot

    db, original = make_db(tmp_path / 'state.db')
    stale = db.get_messages_as_conversation('s', include_row_ids=True)[1]
    stale[DB_ROW_SNAPSHOT] = transcript_row_snapshot(db._conn.execute(
        'SELECT * FROM messages WHERE id=?', (original[1][0],)).fetchone())
    db.replace_messages('s', [], archive_dropped=True)
    stale['content'] = 'stale edit'
    assert db.append_messages_batch('s', [stale]) == 1
    assert_chain(db._conn)
    assert db.get_messages_as_conversation('s')[-1]['content'] == 'stale edit'
    db.close()


def test_row_addressed_flush_keeps_historical_source(tmp_path):
    from agent.message_metadata import DB_ROW_SNAPSHOT
    from agent.transcript_repair import transcript_row_snapshot
    db, original = make_db(tmp_path / 'state.db')
    messages = db.get_messages_as_conversation('s', include_row_ids=True)
    result = messages[1]
    result[DB_ROW_SNAPSHOT] = transcript_row_snapshot(
        db._conn.execute('SELECT * FROM messages WHERE id=?', (original[1][0],)).fetchone())
    result['content'] = 'updated result'
    db.append_messages_batch('s', [result])
    assert_chain(db._conn)
    assert db.get_messages_as_conversation('s')[-1]['content'] == 'updated result'
    db.close()


@pytest.mark.parametrize('index', [0, 1])
def test_flush_generation_order_repeat_and_stale_winner(tmp_path, index):
    from agent.message_metadata import DB_ROW_SNAPSHOT, CANONICAL_ROW
    from agent.transcript_repair import transcript_row_snapshot
    db, original = make_db(tmp_path / 'state.db')
    db.append_messages_batch('s', [{'role': 'user', 'content': 'later'},
                                   {'role': 'assistant', 'content': 'tail'}])
    target = db.get_messages_as_conversation('s', include_row_ids=True)[index]
    target[DB_ROW_SNAPSHOT] = transcript_row_snapshot(db._conn.execute(
        'SELECT * FROM messages WHERE id=?', (target['_row_id'],)).fetchone())
    stale = target.copy()
    target['content'] = 'updated'
    counters = tuple(db._conn.execute('SELECT message_count, tool_call_count FROM sessions WHERE id=?', ('s',)).fetchone())
    assert db.append_messages_batch('s', [target]) == 0
    assert target['_row_id'] != original[index][0]
    expected = ['call', 'original result', 'later', 'tail']
    expected[index] = 'updated'
    assert [m['content'] for m in db.get_messages_as_conversation('s')] == expected
    count = db._conn.execute('SELECT count(*) FROM messages').fetchone()[0]
    assert db.append_messages_batch('s', [target]) == 0
    assert db._conn.execute('SELECT count(*) FROM messages').fetchone()[0] == count
    stale['content'] = 'stale edit'
    db.append_messages_batch('s', [stale])
    assert stale[CANONICAL_ROW]['content'] == 'updated'
    assert [m['content'] for m in db.get_messages_as_conversation('s')] == expected
    assert tuple(db._conn.execute('SELECT message_count, tool_call_count FROM sessions WHERE id=?', ('s',)).fetchone()) == counters
    assert_chain(db._conn)
    assert not db._conn.execute('PRAGMA foreign_key_check').fetchall()
    db.close()


def test_both_sources_changed_in_one_batch(tmp_path):
    from agent.message_metadata import DB_ROW_SNAPSHOT
    from agent.transcript_repair import transcript_row_snapshot
    db, original = make_db(tmp_path / 'state.db')
    rows = db.get_messages_as_conversation('s', include_row_ids=True)
    for i, row in enumerate(rows):
        row[DB_ROW_SNAPSHOT] = transcript_row_snapshot(db._conn.execute(
            'SELECT * FROM messages WHERE id=?', (original[i][0],)).fetchone())
        row['content'] = f'updated {i}'
    assert db.append_messages_batch('s', rows) == 0
    assert [m['content'] for m in db.get_messages_as_conversation('s')] == ['updated 0', 'updated 1']
    assert len({r['_row_id'] for r in rows}) == 2
    assert_chain(db._conn)
    db.close()


def test_generation_rollback_restores_batch_dict(tmp_path, monkeypatch):
    from agent.message_metadata import DB_ROW_SNAPSHOT
    from agent.transcript_repair import transcript_row_snapshot
    db, original = make_db(tmp_path / 'state.db')
    target = db.get_messages_as_conversation('s', include_row_ids=True)[1]
    target[DB_ROW_SNAPSHOT] = transcript_row_snapshot(db._conn.execute(
        'SELECT * FROM messages WHERE id=?', (original[1][0],)).fetchone())
    target['content'] = 'updated'
    before = target.copy()
    def fail(*args, **kwargs):
        raise RuntimeError('injected failure')
    monkeypatch.setattr(db, '_insert_message_rows', fail)
    with pytest.raises(RuntimeError, match='injected failure'):
        db.append_messages_batch('s', [target])
    assert target == before
    assert db._conn.execute('SELECT count(*) FROM messages').fetchone()[0] == 2
    assert_chain(db._conn)
    db.close()


def test_marker_cleanup_skips_historical_call_source(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('s', source='cli')
    db.append_messages_batch('s', [
        {'role': 'assistant', 'content': '[memory]', 'tool_calls': [{'id': 'c', 'type': 'function',
         'function': {'name': 'search', 'arguments': '{}'}}]},
        {'role': 'tool', 'content': 'original result', 'tool_call_id': 'c', 'tool_name': 'search'},
    ])
    db._conn.executescript(SCHEMA)
    rows = db._conn.execute('SELECT id FROM messages ORDER BY id').fetchall()
    add_chain(db._conn, rows[1][0], rows[0][0])
    assert db.purge_stale_tool_call_markers(dry_run=True, backup=False)['rows_affected'] == 0
    result = db.purge_stale_tool_call_markers(backup=False)
    assert result['rows_affected'] == 0
    assert db._conn.execute('SELECT content FROM messages WHERE id=?', (rows[0][0],)).fetchone()[0] == '[memory]'
    assert_chain(db._conn)
    db.close()


def test_real_prune_caller_uses_archive_fallback(tmp_path):
    from unittest.mock import patch
    from tests.agent.test_proactive_prune_restart_safety import _build_agent, _configure_pruning, _history

    db = SessionDB(tmp_path / 'state.db')
    db.create_session('s', source='cli')
    db.append_messages_batch('s', _history())
    conn = db._conn
    conn.executescript(SCHEMA)
    result = conn.execute("SELECT id,content FROM messages WHERE session_id='s' AND tool_call_id='call_0'").fetchone()
    call = conn.execute("SELECT id,tool_calls FROM messages WHERE session_id='s' AND role='assistant' ORDER BY id LIMIT 1").fetchone()
    add_chain(conn, result[0], call[0])
    assert_chain(conn)
    agent = _build_agent(db, 's')
    _configure_pruning(agent)
    before = db.get_messages_as_conversation('s')
    with patch.object(db, 'archive_and_compact', wraps=db.archive_and_compact) as fallback:
        pruned, count = agent.context_compressor.prune_tool_results_only(before, current_tokens=120_000)
    assert count >= 1 and fallback.call_count == 1
    assert [m['content'] for m in db.get_messages_as_conversation('s')] == [m['content'] for m in pruned]
    assert conn.execute('SELECT content FROM messages WHERE id=?', (result[0],)).fetchone()[0] == result[1]
    assert conn.execute('SELECT tool_calls FROM messages WHERE id=?', (call[0],)).fetchone()[0] == call[1]
    assert_chain(conn)
    assert not conn.execute('PRAGMA foreign_key_check').fetchall()
    db.close()
