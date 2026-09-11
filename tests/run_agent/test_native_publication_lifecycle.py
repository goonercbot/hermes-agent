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
