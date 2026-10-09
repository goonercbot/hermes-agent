"""Real isolated board/session stores: continuity must preserve task and role boundaries."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.cli_init_mixin import CLIInitMixin
from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
from hermes_cli.kanban_session import record_worker_session, resolve_worker_resume
from hermes_state import SessionDB


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    workspace = tmp_path / "work"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    for name in ("builder", "reviewer"):
        p = home / "profiles" / name
        p.mkdir(parents=True)
        (p / "config.yaml").write_text("kanban:\n  resume_sessions: true\n")
    kb.init_db()
    stores = {name: SessionDB(db_path=home / "profiles" / name / "state.db") for name in ("builder", "reviewer")}
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="Preserve edited work", assignee="builder", workspace_kind="dir", workspace_path=str(workspace))
    yield SimpleNamespace(home=home, workspace=workspace, db=stores["builder"], stores=stores, tid=tid, monkeypatch=monkeypatch)
    for db in stores.values():
        db.close()


def claim(e, profile="builder", review=False):
    e.monkeypatch.setenv("HERMES_HOME", str(e.home / "profiles" / profile))
    e.db = e.stores[profile]
    with kbc.connect_closing() as conn:
        task = (kb.claim_review_task if review else kb.claim_task)(conn, e.tid, claimer=profile + ":test")
    assert task
    for key, value in {"HERMES_PROFILE":profile, "HERMES_KANBAN_TASK":e.tid,
                       "HERMES_KANBAN_RUN_ID":str(task.current_run_id),
                       "HERMES_KANBAN_CLAIM_LOCK":task.claim_lock}.items():
        e.monkeypatch.setenv(key, value)
    return task


class _CLIProbe(CLIInitMixin, CLIAgentSetupMixin):
    def _init_session_store(self):
        pass

    def _init_ui_state(self):
        pass


def start_cli(e):
    # Exercise the real CLI startup seam without credentials, network or UI initialization.
    obj = _CLIProbe()
    obj._session_db = e.db
    obj.tool_progress_mode = "off"
    obj._init_runtime_state(None)
    if obj._resumed:
        assert obj._load_resumed_history_late()
    return obj


def seed(e, sid, text):
    e.db.create_session(sid, source="kanban")
    e.db.append_message(sid, role="user", content="Continue the assigned task")
    e.db.append_message(sid, role="assistant", content=text)


def test_cli_retry_review_and_repair_keep_separate_histories(env):
    e = env
    task = claim(e)
    first = start_cli(e)
    seed(e, first.session_id, "unfinished edit and failed approach retained")
    (e.workspace / "unfinished.txt").write_text("preserve this edit")
    with kbc.connect_closing() as conn:
        kb.block_task(conn, e.tid, reason="interrupted", expected_run_id=task.current_run_id)
        kb.unblock_task(conn, e.tid)
    task = claim(e)
    again = start_cli(e)
    assert again.session_id == first.session_id and again._resumed
    assert any(m.get("content") == "unfinished edit and failed approach retained" for m in again.conversation_history)
    assert (e.workspace / "unfinished.txt").read_text() == "preserve this edit"
    with kbc.connect_closing() as conn:
        kb.request_review(conn, e.tid, summary="candidate", reviewer="reviewer", expected_run_id=task.current_run_id)
    review_task = claim(e, "reviewer", review=True)
    review = start_cli(e)
    assert review.session_id != first.session_id
    seed(e, review.session_id, "independent finding retained")
    with kbc.connect_closing() as conn:
        kb.request_changes(conn, e.tid, reason="repair required", expected_run_id=review_task.current_run_id)
    task = claim(e)
    assert start_cli(e).session_id == first.session_id
    with kbc.connect_closing() as conn:
        kb.request_review(conn, e.tid, summary="repaired", reviewer="reviewer", expected_run_id=task.current_run_id)
    claim(e, "reviewer", review=True)
    assert start_cli(e).session_id == review.session_id
    history = e.db.get_messages_as_conversation(review.session_id)
    assert any(m.get("content") == "independent finding retained" for m in history)
    assert not any(m.get("content") == "unfinished edit and failed approach retained" for m in history)


@pytest.mark.parametrize("boundary", ["missing", "workspace", "profile", "task", "board", "claim", "delegated", "restore", "disabled", "compression", "explicit"])
def test_continuity_never_imports_wrong_history_or_old_authority(env, boundary):
    e = env
    task = claim(e)
    seed(e, "bound-session", "retained context")
    record_worker_session("bound-session")
    with kbc.connect_closing() as conn:
        kb.block_task(conn, e.tid, reason="pause", expected_run_id=task.current_run_id)
        kb.unblock_task(conn, e.tid)
    task = claim(e)
    if boundary == "disabled":
        (e.home / "profiles" / "builder" / "config.yaml").write_text("kanban:\n  resume_sessions: false\n")
        assert resolve_worker_resume(e.db) is None
        assert not start_cli(e)._resumed
        return
    if boundary == "explicit":
        with pytest.raises(RuntimeError, match="not task/role-bound"):
            resolve_worker_resume(e.db, "unrelated-session")
        return
    if boundary == "compression":
        e.db.end_session("bound-session", "compression")
        e.db.create_session("compressed-child", source="kanban", parent_session_id="bound-session")
        e.db.append_message("compressed-child", role="user", content="retained task checkpoint after compression")
        e.db.append_message("compressed-child", role="assistant", content="continue unfinished edits")
        assert start_cli(e).session_id == "compressed-child"
        return
    if boundary == "restore":
        called = []
        probe = _CLIProbe()
        probe._restore_session_cwd = lambda *a, **k: called.append("cwd")
        probe._restore_session_yolo = lambda *a, **k: called.append("yolo")
        probe._restore_session_model = lambda *a, **k: called.append("model")
        probe._restore_session_state({"model_config":{"yolo_mode": True}})
        assert called == []
        return
    if boundary == "missing":
        e.db.delete_session("bound-session")
    elif boundary == "workspace":
        other = e.workspace.parent / "other"
        other.mkdir()
        e.monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(other))
    elif boundary == "profile":
        e.monkeypatch.setenv("HERMES_PROFILE", "reviewer")
    elif boundary == "claim":
        e.monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "stale")
    elif boundary == "board":
        e.monkeypatch.setenv("HERMES_KANBAN_DB", str(e.home / "other.db"))
        kb.init_db()
    elif boundary == "task":
        with kbc.connect_closing() as conn:
            e.tid = kb.create_task(conn, title="different", assignee="builder", workspace_kind="dir", workspace_path=str(e.workspace))
        claim(e)
    elif boundary == "delegated":
        from agent.delegation_context import delegated_child_context
        with delegated_child_context():
            assert resolve_worker_resume(e.db) is None
        return
    if boundary in {"workspace", "profile", "board", "claim"}:
        with pytest.raises(RuntimeError):
            resolve_worker_resume(e.db)
    else:
        assert resolve_worker_resume(e.db) is None
