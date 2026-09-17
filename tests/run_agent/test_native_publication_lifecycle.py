"""Publication must authenticate before ANY caller resumes, not just preflight."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from agent.native_incremental_handoff import (
    _projection_source_for_messages,
    bind_native_incremental_replay_projection,
    restore_native_incremental_note,
)
from agent.native_compaction import validate_persisted_native_compaction_history
from tests.run_agent.test_native_preflight_projection_rebind import (
    _agent_with_authenticated_history, _response, _message,
)


def _assert_fresh_process_reload(tmp_path, session_id, platform, model):
    """Reload the exact SQLite generation through a new interpreter/agent."""
    root = Path(__file__).resolve().parents[2]
    script = f"""
from hermes_state import SessionDB
from pathlib import Path
from run_agent import AIAgent
from agent.native_compaction import validate_persisted_native_compaction_history
from agent.native_incremental_handoff import bind_native_incremental_replay_projection, restore_native_incremental_note

db = SessionDB(db_path=Path({str(tmp_path / 'state.db')!r}))
agent = AIAgent(
    api_key='fixture', base_url='https://chatgpt.com/backend-api/codex',
    api_mode='codex_responses', model={model!r}, platform={platform!r},
    provider='openai-codex', session_db=db, session_id={session_id!r},
    quiet_mode=True, skip_memory=True, skip_context_files=True,
    skip_background_review=True, enabled_toolsets=[]
)
rows = db.get_messages_as_conversation({session_id!r}, repair_alternation=False)
validate_persisted_native_compaction_history(rows)
assert bind_native_incremental_replay_projection(agent, source_messages=rows, replay_messages=rows)
assert restore_native_incremental_note(agent, rows) is not None
assert agent._native_incremental_replay_projection['source'] == rows
agent.close()
db.close()
print('fresh-process native publication reload ok')
"""
    env = os.environ.copy()
    env.update({
        "HERMES_HOME": str(tmp_path),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(root) + os.pathsep + env.get("PYTHONPATH", ""),
    })
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, env=env,
        text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "fresh-process native publication reload ok"


def _append_durable_tool_and_steering(agent, db):
    """Make an authenticated note's tail contain durable evidence and steering."""
    db.append_message(
        agent.session_id,
        "assistant",
        "",
        tool_calls=[{
            "id": "durable-terminal-1",
            "type": "function",
            "function": {"name": "terminal", "arguments": "{\"command\":\"true\"}"},
        }],
    )
    db.append_message(
        agent.session_id,
        "tool",
        json.dumps({"exit_code": 0, "output": "DURABLE_TOOL_PAYLOAD " * 2000}),
        tool_call_id="durable-terminal-1",
        tool_name="terminal",
    )
    db.append_message(
        agent.session_id,
        "user",
        "STEERING: retain the durable evidence; do not deploy.",
    )
    source = db.get_messages_as_conversation(
        agent.session_id, repair_alternation=False
    )
    assert bind_native_incremental_replay_projection(
        agent, source_messages=source, replay_messages=source
    )
    assert restore_native_incremental_note(agent, source) is not None
    return source


@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("platform", ["telegram", "subagent"])
def test_publication_authenticates_final_tail_before_return(tmp_path, monkeypatch, in_place, platform):
    agent, db, source = _agent_with_authenticated_history(tmp_path, monkeypatch)
    agent.platform = platform
    agent.compression_in_place = in_place
    agent._native_compaction_turn_active = False  # post-tool/manual/overflow boundary
    agent._todo_store.format_for_injection = lambda: "[Current task state]\nTASK_STATE_AMBER; deployment paused"
    source = _append_durable_tool_and_steering(agent, db)
    before = deepcopy(source)
    requests = []

    def provider(request):
        requests.append(deepcopy(request))
        assert request["model"] == "gpt-5.6-luna"
        return _response(
            NS(type="compaction", id="publication-cp", encrypted_content="fixture-checkpoint"),
            _message("Continue with the current task."), model="gpt-5.6-luna",
        )

    agent._interruptible_api_call = provider
    try:
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            compressed, _ = agent._compress_context(source, "Test native publication", approx_tokens=210000)
        assert len(requests) == 1
        assert len(compressed) < len(source) or len(str(compressed)) < len(str(source))
        assert validate_persisted_native_compaction_history(compressed)
        assert "TASK_STATE_AMBER" in str(compressed)
        assert "DURABLE_TOOL_PAYLOAD" in str(compressed)
        assert "STEERING: retain the durable evidence; do not deploy." in str(compressed)
        canonical, _ = _projection_source_for_messages(agent, compressed)
        assert canonical is not None, "published tail diverged from pre-publication replay fence"
        assert restore_native_incremental_note(agent, compressed) is not None
        durable = db.get_messages_as_conversation(agent.session_id, repair_alternation=False)
        assert validate_persisted_native_compaction_history(durable)
        _assert_fresh_process_reload(tmp_path, agent.session_id, platform, agent.model)
        tampered = deepcopy(compressed)
        tampered[-1]["content"] = "tampered published boundary"
        with pytest.raises(ValueError):
            validate_persisted_native_compaction_history(tampered)
        assert _projection_source_for_messages(agent, tampered) == (None, None)
        assert restore_native_incremental_note(agent, tampered) is None
        if not in_place:
            assert db.get_messages_as_conversation("native-preflight-projection", repair_alternation=False)[:len(before)] == before
    finally:
        agent.close()
        db.close()


@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("failure", ["commit", "readback", "tail", "carrier"])
def test_publication_failure_never_returns_success(tmp_path, monkeypatch, in_place, failure):
    agent, db, source = _agent_with_authenticated_history(tmp_path, monkeypatch)
    agent.compression_in_place = in_place
    agent._native_compaction_turn_active = False
    before = deepcopy(source)
    original_sid = agent.session_id
    read = db.get_messages_as_conversation
    method = "archive_and_compact" if in_place else "publish_compression_child"
    commit = getattr(db, method)
    def failed_read(*args, **kwargs):
        rows = read(*args, **kwargs)
        if failure == "readback":
            raise OSError("fixture readback unavailable")
        if failure == "tail":
            rows[-1]["content"] = "tampered saved tail"
        if failure == "carrier":
            rows[0]["codex_reasoning_items"][0]["encrypted_content"] = "wrong checkpoint"
        return rows
    def controlled_commit(*args, **kwargs):
        if failure == "commit":
            raise OSError("fixture atomic commit failure")
        result = commit(*args, **kwargs)
        monkeypatch.setattr(db, "get_messages_as_conversation", failed_read)
        return result
    monkeypatch.setattr(db, method, controlled_commit)
    agent._interruptible_api_call = lambda request: _response(
        NS(type="compaction",id="failure-cp",encrypted_content="fixture-checkpoint"),
        _message("Continue."),model="gpt-5.6-luna",
    )
    try:
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            with pytest.raises((ValueError, OSError)):
                agent._compress_context(source, "Test failed publication", approx_tokens=210000)
        if failure == "commit":
            assert agent.session_id == original_sid
            assert read(original_sid, repair_alternation=False) == before
        else:
            # A failed reader is NOT permission to erase/restore a committed
            # checkpoint. The durable generation remains valid for recovery.
            saved = read(agent.session_id, repair_alternation=False)
            assert validate_persisted_native_compaction_history(saved)
        assert db._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        agent.close()
        db.close()


def test_post_commit_crash_reconciles_in_fresh_process_and_allows_changed_turn_retry(tmp_path, monkeypatch):
    """A committed child survives a process death before native readback.

    This intentionally terminates a separate interpreter at the exact native
    authentication seam after SQLite publication. The parent test then reads
    the durable generation, and a second fresh interpreter restores it before
    completing a materially changed ordinary turn. Provider responses remain
    local fixture objects; the database is this test's isolated SQLite file.
    """
    agent, db, source = _agent_with_authenticated_history(tmp_path, monkeypatch)
    session_id = agent.session_id
    db_path = tmp_path / "state.db"
    try:
        # The spawned process owns its connection. Close this fixture reader
        # before the deliberate os._exit so neither test process holds a lock.
        agent.close()
        db.close()

        root = Path(__file__).resolve().parents[2]
        crash_script = f"""
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from hermes_state import SessionDB
from run_agent import AIAgent
from agent.native_incremental_handoff import bind_native_incremental_replay_projection, restore_native_incremental_note
import agent.native_incremental_handoff as handoff

path = Path({str(db_path)!r})
session_id = {session_id!r}
db = SessionDB(db_path=path)
agent = AIAgent(
    api_key='fixture', base_url='https://chatgpt.com/backend-api/codex',
    api_mode='codex_responses', model='gpt-6-astra', provider='openai-codex',
    session_db=db, session_id=session_id, quiet_mode=True, skip_memory=True,
    skip_context_files=True, skip_background_review=True, enabled_toolsets=[]
)
agent._disable_streaming = True
agent.compression_in_place = False
agent._compression_feasibility_checked = True
agent._emit_status = lambda *args, **kwargs: None
agent._emit_warning = lambda *args, **kwargs: None
agent.commit_memory_session = lambda *args, **kwargs: None
agent.context_compressor.threshold_tokens = 100
agent.context_compressor.should_compress_preflight = lambda _: True
agent.context_compressor.should_compress = lambda _: False
agent.context_compressor.update_from_response({{'prompt_tokens': 20000}})
source = db.get_messages_as_conversation(session_id, repair_alternation=False)
assert bind_native_incremental_replay_projection(agent, source_messages=source, replay_messages=source)
assert restore_native_incremental_note(agent, source) is not None
agent._native_compaction_turn_active = False
agent._interruptible_api_call = lambda request: NS(
    output=[
        NS(type='compaction', id='crash-cp', encrypted_content='crash-checkpoint'),
        NS(type='message', role='assistant', content=[NS(type='output_text', text='published before crash')]),
    ],
    usage=NS(input_tokens=210000, output_tokens=4, total_tokens=210004),
    status='completed', model='gpt-5.6-luna'
)
# Publication is already committed when this authentication/readback seam runs.
# os._exit prevents finally blocks from supplying an in-process recovery path.
handoff.authenticate_native_compaction_publication = lambda *args, **kwargs: os._exit(73)
with patch('hermes_cli.plugins.invoke_hook', return_value=[]):
    agent._compress_context(source, 'crash after durable publication', approx_tokens=210000)
raise AssertionError('post-commit crash seam was not reached')
"""
        env = os.environ.copy()
        env.update({
            "HOME": str(tmp_path),
            "HERMES_HOME": str(tmp_path),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(root) + os.pathsep + env.get("PYTHONPATH", ""),
        })
        crashed = subprocess.run(
            [sys.executable, "-B", "-c", crash_script], cwd=root, env=env,
            text=True, capture_output=True, timeout=60,
        )
        assert crashed.returncode == 73, crashed.stderr

        db = __import__("hermes_state").SessionDB(db_path=db_path)
        children = db._conn.execute(
            "SELECT id FROM sessions WHERE parent_session_id = ? ORDER BY started_at", (session_id,)
        ).fetchall()
        assert len(children) == 1
        child_session_id = children[0][0]
        published = db.get_messages_as_conversation(child_session_id, repair_alternation=False)
        validate_persisted_native_compaction_history(published)
        assert restore_native_incremental_note(NS(session_id=child_session_id), published) is not None
        assert db._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        db.close()
        db = None

        retry_script = f"""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from hermes_state import SessionDB
from run_agent import AIAgent
from agent.native_compaction import validate_persisted_native_compaction_history
from agent.native_incremental_handoff import bind_native_incremental_replay_projection, restore_native_incremental_note

path = Path({str(db_path)!r})
session_id = {child_session_id!r}
db = SessionDB(db_path=path)
agent = AIAgent(
    api_key='fixture', base_url='https://chatgpt.com/backend-api/codex',
    api_mode='codex_responses', model='gpt-6-astra', provider='openai-codex',
    session_db=db, session_id=session_id, quiet_mode=True, skip_memory=True,
    skip_context_files=True, skip_background_review=True, enabled_toolsets=[]
)
agent._disable_streaming = True
agent._compression_feasibility_checked = True
agent._emit_status = lambda *args, **kwargs: None
agent._emit_warning = lambda *args, **kwargs: None
agent.commit_memory_session = lambda *args, **kwargs: None
agent.context_compressor.should_compress_preflight = lambda _: False
agent.context_compressor.should_compress = lambda _: False
source = db.get_messages_as_conversation(session_id, repair_alternation=False)
validate_persisted_native_compaction_history(source)
assert bind_native_incremental_replay_projection(agent, source_messages=source, replay_messages=source)
assert restore_native_incremental_note(agent, source) is not None
agent._interruptible_api_call = lambda request: NS(
    output=[NS(type='message', role='assistant', content=[NS(type='output_text', text='CHANGED_TURN_RETRY_OK')])],
    usage=NS(input_tokens=42, output_tokens=4, total_tokens=46),
    status='completed', model='gpt-6-astra'
)
with patch('hermes_cli.plugins.invoke_hook', return_value=[]), patch(
    'hermes_cli.lifecycle.invoke_hook', return_value=[]
), patch('agent.turn_context._maybe_title_session_at_turn_start', return_value=None):
    result = agent.run_conversation(
        'Changed retry instruction: preserve the committed checkpoint; do not replay work.',
        conversation_history=source,
    )
assert result['completed'] and result['final_response'] == 'CHANGED_TURN_RETRY_OK'
reloaded = db.get_messages_as_conversation(session_id, repair_alternation=False)
validate_persisted_native_compaction_history(reloaded)
assert any(row.get('content') == 'Changed retry instruction: preserve the committed checkpoint; do not replay work.' for row in reloaded)
assert any(row.get('content') == 'CHANGED_TURN_RETRY_OK' for row in reloaded)
assert restore_native_incremental_note(NS(session_id=session_id), reloaded) is not None
agent.close()
db.close()
print('fresh-process crash reconciliation and changed-turn retry ok')
"""
        retried = subprocess.run(
            [sys.executable, "-B", "-c", retry_script], cwd=root, env=env,
            text=True, capture_output=True, timeout=60,
        )
        assert retried.returncode == 0, retried.stderr
        assert retried.stdout.strip() == "fresh-process crash reconciliation and changed-turn retry ok"
    finally:
        if db is not None:
            db.close()
