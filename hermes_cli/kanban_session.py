"""Task/role-bound CLI conversation continuity using existing run metadata.

No new store or launcher: the claimed child selects history from its own SessionDB.
Unbound/legacy attempts recover from the task checkpoint, never from a guessed session.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
_BINDING = "conversation_binding"


def worker_continuity_enabled():
    from agent.delegation_context import owned_kanban_task
    from hermes_cli.config import load_config_readonly

    return bool(owned_kanban_task()) and load_config_readonly().get("kanban", {}).get("resume_sessions") is True


def _context(conn):
    from agent.delegation_context import owned_kanban_task
    from hermes_cli import kanban_db as kb
    from hermes_constants import get_hermes_home

    task_id = owned_kanban_task()
    if not task_id:
        raise RuntimeError("Kanban conversation requires an owned task")
    task = kb.get_task(conn, task_id)
    try:
        run_id = int(os.environ.get("HERMES_KANBAN_RUN_ID", ""))
    except ValueError:
        raise RuntimeError("Kanban conversation requires a valid current run") from None
    run = kb.get_run(conn, run_id)
    profile = os.environ.get("HERMES_PROFILE")
    lock = os.environ.get("HERMES_KANBAN_CLAIM_LOCK")
    workspace = os.environ.get("HERMES_KANBAN_WORKSPACE")
    if not (task and run and task.current_run_id == run_id
            and task.status == "running" and run.ended_at is None
            and run.task_id == task_id and task.assignee == run.profile == profile
            and lock and task.claim_lock == run.claim_lock == lock
            and workspace and Path(workspace).is_dir()):
        raise RuntimeError("Kanban conversation rejected: stale or mismatched claim")
    if task.workspace_path and Path(task.workspace_path).resolve() != Path(workspace).resolve():
        raise RuntimeError("Kanban conversation rejected: mismatched workspace")
    scope = {
        "profile_home": str(get_hermes_home().resolve()),
        "workspace": str(Path(workspace).resolve()),
        "branch": task.branch_name,
        "role": kb._retry_status_for_run(conn, task_id, run_id),
        "step": run.step_key,
    }
    return task, run, scope


def resolve_worker_resume(session_db, requested=None):
    """Return a validated exact session, or None for checkpoint recovery.

    Only an active claimed worker can use this route. The board connection and the
    receiving profile's native SessionDB supply the two isolation boundaries.
    """
    from hermes_cli import kanban_db_connect as kbc

    if not worker_continuity_enabled():
        return requested
    with kbc.connect_closing() as conn:
        task, run, scope = _context(conn)
        rows = conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id = ? AND profile = ? "
            "AND id < ? ORDER BY id DESC", (task.id, run.profile, run.id),
        )
        binding = None
        for row in rows:
            data = json.loads(row["metadata"] or "{}")
            candidate = data.get(_BINDING) if isinstance(data, dict) else None
            if (isinstance(candidate, dict) and isinstance(candidate.get("scope"), dict)
                    and candidate["scope"].get("role") == scope["role"]):
                binding = candidate
                break
    selected = None
    if binding and binding.get("scope") == scope and session_db is not None:
        sid = binding.get("session_id")
        # Native resolution follows compression continuations, not other tasks' latest sessions.
        if isinstance(sid, str) and session_db.get_session(sid):
            resolved = session_db.resolve_resume_session_id(sid)
            meta = session_db.get_session(resolved) if resolved else None
            if meta and meta.get("source") == "kanban" and not meta.get("archived"):
                from hermes_state import SessionResumeTooLargeError
                try:
                    session_db.assert_resume_safe(resolved, tip_only=True)
                except SessionResumeTooLargeError:
                    logger.warning("Kanban task %s: history exceeds safe-resume limit; recover from checkpoint", task.id)
                else:
                    if meta.get("message_count", 0) > 0:
                        selected = resolved
    if requested and requested != selected:
        raise RuntimeError("Kanban conversation rejected: requested session is not task/role-bound")
    if binding and not selected:
        logger.warning("Kanban task %s: prior conversation unavailable or scope changed; recover from checkpoint", task.id)
    return selected


def record_worker_session(session_id):
    """Bind before the first turn, so a crash need not reach a lifecycle tool.

    Terminal transitions preserve this metadata. Compression is followed using
    native session lineage on the next attempt.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    if not worker_continuity_enabled():
        return
    with kbc.connect_closing() as conn, kb.write_txn(conn):
        task, run, scope = _context(conn)
        metadata = dict(run.metadata or {})
        metadata[_BINDING] = {"session_id": session_id, "scope": scope}
        conn.execute("UPDATE task_runs SET metadata = ? WHERE id = ?", (json.dumps(metadata), run.id))


def retain_conversation_binding(conn, run_id, metadata):
    """Lifecycle callers may add evidence, but cannot overwrite the native binding."""
    row = conn.execute("SELECT metadata FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    old = json.loads(row["metadata"] or "{}") if row else {}
    result = dict(metadata or {})
    result.pop(_BINDING, None)
    if isinstance(old, dict) and _BINDING in old:
        result[_BINDING] = old[_BINDING]
    return result or None
