"""Cross-process state.db lifecycle contracts with real SQLite and OS processes.

These tests deliberately use separate interpreters, file barriers, a real SQLite
write lock, and real SIGTERM/SIGKILL.  They do not make a gateway/cache claim:
SessionDB proves the durable-storage seam only.  Gateway, CLI, and subagent
controllers must rebuild/evict their own cached projections after their process
lifecycle transition and then read this durable state.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from hermes_state import SessionDB
from tests.conformance.persistence._harness import (
    kill9_and_reap,
    reap,
    spawn_child,
    wait_for,
)


def _isolated_env(tmp_path: Path) -> dict[str, str]:
    """Give every child a disposable HOME/HERMES_HOME before its imports."""
    home = tmp_path / "home"
    hermes_home = tmp_path / "hermes-home"
    home.mkdir()
    hermes_home.mkdir()
    return {"HOME": str(home), "HERMES_HOME": str(hermes_home)}


def _assert_integrity(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    finally:
        conn.close()


def test_cross_process_lock_deadline_then_shutdown_recovery(tmp_path):
    """A pre-opened writer obeys its 0.2s budget under another process's lock.

    The client and lock holder are distinct OS processes.  Once the holder is
    released, a third fresh process proves that the same database accepts a
    post-contention recovery write.  The 0.6s margin covers process scheduling
    and SQLite timer granularity; it remains well below the holder's 2s hold.
    """
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()
    env = _isolated_env(tmp_path)
    client_ready, start_write = tmp_path / "client-ready", tmp_path / "start-write"
    holder_ready, release_holder = tmp_path / "holder-ready", tmp_path / "release-holder"
    client_result, recovery_result = tmp_path / "client.json", tmp_path / "recovery.json"

    client = spawn_child(
        f"""
import json, sqlite3, time
from pathlib import Path
from hermes_state import SessionDB

db = SessionDB(db_path=Path({str(db_path)!r}))
SessionDB._WRITE_PATIENCE_S = 0.2
Path({str(client_ready)!r}).write_text('ready', encoding='utf-8')
while not Path({str(start_write)!r}).exists():
    time.sleep(0.01)
started = time.monotonic()
try:
    db.set_meta('deadline-key', 'value')
except sqlite3.OperationalError as exc:
    payload = {{'outcome': 'locked', 'elapsed_s': time.monotonic() - started, 'error': str(exc)}}
else:
    payload = {{'outcome': 'success', 'elapsed_s': time.monotonic() - started}}
finally:
    db.close()
Path({str(client_result)!r}).write_text(json.dumps(payload), encoding='utf-8')
""",
        env=env,
    )
    holder = None
    try:
        wait_for(client_ready.exists, what="pre-opened client", child=client)
        holder = spawn_child(
            f"""
import sqlite3, time
from pathlib import Path

conn = sqlite3.connect({str(db_path)!r}, timeout=0, isolation_level=None)
try:
    conn.execute('BEGIN IMMEDIATE')
    Path({str(holder_ready)!r}).write_text('locked', encoding='utf-8')
    end = time.monotonic() + 30
    while not Path({str(release_holder)!r}).exists():
        if time.monotonic() >= end:
            raise RuntimeError('holder release barrier timed out')
        time.sleep(0.01)
    conn.execute('COMMIT')
finally:
    conn.close()
""",
            env=env,
        )
        wait_for(holder_ready.exists, what="cross-process SQLite write lock", child=holder)
        start_write.write_text('go', encoding='utf-8')
        rc, _out, err = reap(client)
        assert rc == 0, err
        payload = json.loads(client_result.read_text(encoding="utf-8"))
        assert payload["outcome"] == "locked", payload
        assert "another Hermes process" in payload["error"]
        assert 0.10 <= payload["elapsed_s"] <= 0.80, payload
    finally:
        release_holder.write_text('release', encoding='utf-8')
        if holder is not None:
            rc, _out, err = reap(holder)
            assert rc == 0, err
        if client.poll() is None:
            kill9_and_reap(client)

    recovery = spawn_child(
        f"""
import json
from pathlib import Path
from hermes_state import SessionDB

db = SessionDB(db_path=Path({str(db_path)!r}))
try:
    db.set_meta('recovered-after-contention', 'yes')
    payload = {{'value': db.get_meta('recovered-after-contention')}}
finally:
    db.close()
Path({str(recovery_result)!r}).write_text(json.dumps(payload), encoding='utf-8')
""",
        env=env,
    )
    rc, _out, err = reap(recovery)
    assert rc == 0, err
    assert json.loads(recovery_result.read_text(encoding="utf-8")) == {"value": "yes"}
    _assert_integrity(db_path)


def test_sigterm_worker_then_fresh_process_recovers_pending_transcript(tmp_path):
    """A killed process leaves its committed prefix readable and appendable.

    This is deliberately a persistence proof, not a cache-lifecycle proof: a
    controller that owns a gateway/CLI/subagent cache must reacquire its agent
    and project these rows after its own shutdown/restart boundary.
    """
    db_path = tmp_path / "state.db"
    env = _isolated_env(tmp_path)
    ready, recovered = tmp_path / "pending-ready", tmp_path / "pending-recovered.json"
    worker = spawn_child(
        f"""
import time
from pathlib import Path
from hermes_state import SessionDB

db = SessionDB(db_path=Path({str(db_path)!r}))
db.create_session('shutdown-session', 'conformance')
db.append_message('shutdown-session', 'user', content='pending before shutdown')
Path({str(ready)!r}).write_text('durable', encoding='utf-8')
while True:
    time.sleep(1)
""",
        env=env,
    )
    try:
        wait_for(ready.exists, what="durable pending transcript", child=worker)
    finally:
        worker.terminate()
        worker.wait(timeout=20)
    assert worker.returncode != 0

    fresh = spawn_child(
        f"""
import json
from pathlib import Path
from hermes_state import SessionDB

db = SessionDB(db_path=Path({str(db_path)!r}))
try:
    db.append_message('shutdown-session', 'assistant', content='recovered after shutdown')
    payload = {{'rows': [{{'role': row['role'], 'content': row['content']}} for row in db.get_messages('shutdown-session')]}}
finally:
    db.close()
Path({str(recovered)!r}).write_text(json.dumps(payload), encoding='utf-8')
""",
        env=env,
    )
    rc, _out, err = reap(fresh)
    assert rc == 0, err
    payload = json.loads(recovered.read_text(encoding="utf-8"))
    assert payload["rows"] == [
        {"role": "user", "content": "pending before shutdown"},
        {"role": "assistant", "content": "recovered after shutdown"},
    ]
    _assert_integrity(db_path)


def test_sigkill_after_disposable_effect_keeps_unknown_without_replay(tmp_path):
    """A real disposable append before result publication stays UNKNOWN on cold recovery.

    The worker fsyncs an append to a task-owned artifact, publishes only the
    assistant tool call, then receives SIGKILL before a result can be written.
    A fresh process runs the production replay sanitizer.  It may annotate the
    replay projection as UNKNOWN, but neither it nor SessionDB is allowed to
    append the artifact again or mutate canonical history automatically.
    """
    db_path, effect_path = tmp_path / "state.db", tmp_path / "disposable-effect.log"
    env = _isolated_env(tmp_path)
    effect_ready, recovery_result = tmp_path / "effect-ready", tmp_path / "effect-recovery.json"
    worker = spawn_child(
        f"""
import os, time
from pathlib import Path
from hermes_state import SessionDB

db = SessionDB(db_path=Path({str(db_path)!r}))
db.create_session('effect-session', 'conformance')
db.append_message(
    'effect-session', 'assistant', content='writing disposable artifact',
    tool_calls=[{{'id': 'effect-call', 'type': 'function',
                 'function': {{'name': 'write_file', 'arguments': '{{}}'}}}}],
)
with Path({str(effect_path)!r}).open('a', encoding='utf-8') as effect:
    effect.write('effect-applied-once\\n')
    effect.flush()
    os.fsync(effect.fileno())
Path({str(effect_ready)!r}).write_text('effect-durable-result-not-published', encoding='utf-8')
while True:
    time.sleep(1)
""",
        env=env,
    )
    try:
        wait_for(effect_ready.exists, what="durable disposable effect", child=worker)
    finally:
        kill9_and_reap(worker)
    assert effect_path.read_text(encoding="utf-8") == "effect-applied-once\n"
    effect_before_recovery = effect_path.read_bytes()

    fresh = spawn_child(
        f"""
import json
from pathlib import Path
from agent.replay_cleanup import sanitize_replay_history
from hermes_state import SessionDB

db = SessionDB(db_path=Path({str(db_path)!r}))
try:
    canonical = db.get_messages_as_conversation('effect-session')
    recovered = sanitize_replay_history(canonical)
    payload = {{'canonical': canonical, 'recovered': recovered}}
finally:
    db.close()
Path({str(recovery_result)!r}).write_text(json.dumps(payload), encoding='utf-8')
""",
        env=env,
    )
    rc, _out, err = reap(fresh)
    assert rc == 0, err
    payload = json.loads(recovery_result.read_text(encoding="utf-8"))
    assert [row["role"] for row in payload["canonical"]] == ["assistant"]
    assert [row["role"] for row in payload["recovered"]] == ["assistant", "tool"]
    recovered_result = payload["recovered"][-1]
    assert recovered_result["tool_call_id"] == "effect-call"
    assert recovered_result["effect_disposition"] == "unknown"
    assert "UNKNOWN" in recovered_result["content"]
    assert effect_path.read_bytes() == effect_before_recovery
    _assert_integrity(db_path)
