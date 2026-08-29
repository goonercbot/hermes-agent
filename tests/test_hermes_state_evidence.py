"""SessionDB integration tests for prospective evidence events."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading

import pytest

import hermes_state_evidence
from agent.evidence_ledger import runtime_context
from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_state import SCHEMA_VERSION, SessionDB


def _tool_pair(
    call_id: str,
    *,
    command: str,
    result: dict,
    candidate: str | None = None,
    status: str | None = None,
    effect_disposition: str | None = None,
) -> list[dict]:
    context = runtime_context("terminal", call_id, cwd="/private/test-worktree")
    if candidate is not None:
        context["candidate"] = candidate
    if status is not None:
        context["status"] = status
    tool_result = {
        "role": "tool",
        "content": json.dumps(result),
        "tool_name": "terminal",
        "tool_call_id": call_id,
        "_evidence_context": context,
    }
    if effect_disposition is not None:
        tool_result["effect_disposition"] = effect_disposition
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "terminal",
                    "arguments": json.dumps({"command": command}),
                },
            }],
        },
        tool_result,
    ]


def _manual_event(db: SessionDB, session_id: str, message_id: int) -> int:
    source = db._conn.execute(
        "SELECT id, session_id, role, content, tool_call_id, tool_calls, tool_name, timestamp "
        "FROM messages WHERE id = ?",
        (message_id,),
    ).fetchone()
    payload = json.dumps(dict(source), sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    body = {
        "session_id": session_id,
        "source_message_id": message_id,
        "source_sha256": digest,
        "tool_name": "terminal",
        "event_type": "tool_result",
        "lifecycle_dimension": "observation",
        "lifecycle_state": "observed",
        "outcome": "success",
        "issuer": "hermes.evidence",
        "adapter": "generic",
        "adapter_version": "1",
        "subject_type": "tool",
        "subject_id": "terminal",
        "metadata_json": "{}",
        "occurred_at": 1.0,
        "event_hash": "a" * 64,
    }
    columns = ", ".join(body)
    values = tuple(body.values())
    marks = ", ".join("?" for _ in values)
    db._conn.execute(
        f"INSERT INTO evidence_events ({columns}) VALUES ({marks})", values
    )
    db._conn.commit()
    return int(db._conn.execute("SELECT id FROM evidence_events").fetchone()[0])


def test_evidence_schema_is_opt_in_and_append_only(tmp_path):
    assert DEFAULT_CONFIG["evidence"]["enabled"] is False
    assert SCHEMA_VERSION >= 27
    db = SessionDB(db_path=tmp_path / "state.db", evidence_enabled=False)
    try:
        db.create_session("s", source="test")
        message_id = db.append_message("s", "tool", "ok", tool_name="terminal")
        assert db._conn.execute("SELECT COUNT(*) FROM evidence_events").fetchone()[0] == 0
        event_id = _manual_event(db, "s", message_id)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db._conn.execute(
                "UPDATE evidence_events SET outcome = 'failure' WHERE id = ?",
                (event_id,),
            )
        db._conn.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db._conn.execute("DELETE FROM evidence_events WHERE id = ?", (event_id,))
        db._conn.rollback()
    finally:
        db.close()


def test_batch_persists_typed_event_with_both_exact_sources(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db", evidence_enabled=True)
    try:
        db.create_session("s", source="test")
        command = "pytest tests/unit -q"
        output = "1 passed in 0.01s"
        db.append_messages_batch(
            "s",
            _tool_pair(
                "call-1",
                command=command,
                result={"output": output, "exit_code": 0, "error": None},
            ),
        )
        event = db.get_evidence_events("s")[0]
        assert event["lifecycle_state"] == "tested_pass"
        assert event["adapter"] == "pytest"
        assert event["trusted"] == 1
        assert event["source_call_message_id"] is not None
        assert db.verify_evidence_integrity("s") is True
        assert db.get_evidence_source(event["id"])["content"] == json.dumps(
            {"output": output, "exit_code": 0, "error": None}
        )
        metadata = json.loads(event["metadata_json"])
        assert command not in event["metadata_json"]
        assert output not in event["metadata_json"]
        assert set(metadata) >= {"input_sha256", "result_sha256", "result_bytes"}
        receipts = db.get_compact_evidence_receipts("s", ["call-1"])
        assert len(receipts["call-1"]) == 1
        receipt = receipts["call-1"][0]
        assert event["source_sha256"] in receipt
        assert event["event_hash"] in receipt
        assert command not in receipt
        assert output not in receipt
    finally:
        db.close()


def test_rewrite_preserves_evidence_sources_and_session_delete_removes_ledger(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db", evidence_enabled=True)
    try:
        db.create_session("s", source="test")
        db.append_messages_batch(
            "s",
            _tool_pair(
                "call-retained",
                command="pytest -q",
                result={"output": "1 passed", "exit_code": 0},
            ),
        )
        event = db.get_evidence_events("s")[0]
        source_id = event["source_message_id"]
        db.replace_messages("s", [{"role": "user", "content": "replacement"}])
        preserved = db._conn.execute(
            "SELECT active FROM messages WHERE id = ?", (source_id,)
        ).fetchone()
        assert preserved is not None
        assert preserved["active"] == 0
        assert db.verify_evidence_integrity("s") is True

        assert db.delete_session("s") is True
        assert db._conn.execute(
            "SELECT COUNT(*) FROM evidence_events WHERE session_id = 's'"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM evidence_retention_deletions"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_evidence_failure_rolls_back_the_tool_result_transaction(
    tmp_path, monkeypatch
):
    db = SessionDB(db_path=tmp_path / "state.db", evidence_enabled=True)
    try:
        db.create_session("s", source="test")
        monkeypatch.setattr(
            hermes_state_evidence,
            "adapt_tool_result",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("evidence failed")),
        )
        with pytest.raises(RuntimeError, match="evidence failed"):
            db.append_messages_batch(
                "s",
                _tool_pair(
                    "call-rollback",
                    command="pytest -q",
                    result={"output": "ok", "exit_code": 0},
                ),
            )
        assert db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = 's'"
        ).fetchone()[0] == 0
        assert db._conn.execute("SELECT COUNT(*) FROM evidence_events").fetchone()[0] == 0
    finally:
        db.close()


def test_latest_same_candidate_invalidates_older_pass(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db", evidence_enabled=True)
    try:
        db.create_session("s", source="test")
        db.append_messages_batch(
            "s",
            _tool_pair(
                "pass-call",
                command="pytest -q",
                result={"output": "1 passed", "exit_code": 0},
            ),
        )
        first = db.get_evidence_events("s")[0]
        candidate = first["candidate_id"]
        subject = first["subject_id"]
        parent = json.loads(first["parent_candidate_id"])
        db.append_messages_batch(
            "s",
            _tool_pair(
                "fail-call",
                command="pytest -q",
                result={"output": "1 failed", "exit_code": 1},
            ),
        )
        events = db.get_evidence_events("s")
        latest = events[-1]
        assert latest["lifecycle_state"] == "tested_fail"
        assert latest["supersedes_event_id"] == first["id"]
        old_claim = db.resolve_evidence_claim(
            "s",
            state="tested_pass",
            subject=subject,
            candidate=candidate,
            parent_lineage=parent,
            source_hash_value=first["source_sha256"],
        )
        new_claim = db.resolve_evidence_claim(
            "s",
            state="tested_fail",
            subject=subject,
            candidate=candidate,
            parent_lineage=parent,
            source_hash_value=latest["source_sha256"],
        )
        assert old_claim["resolved"] is False
        assert old_claim["reason"] == "superseded"
        assert new_claim["resolved"] is False
        assert new_claim["reason"] == "unsuccessful_outcome"
    finally:
        db.close()


def test_source_mutation_breaks_integrity_and_untrusted_context_is_rejected(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db", evidence_enabled=True)
    try:
        db.create_session("s", source="test")
        rows = _tool_pair(
            "call-secret",
            command="pytest -q",
            result={"output": "ok", "exit_code": 0},
        )
        rows[-1]["_evidence_context"] = {
            "subject": "secret:AUTHORIZATION-SECRET\\\"VALUE",
            "candidate": "AUTHORIZATION-SECRET\\\"VALUE",
            "parent_lineage": {"authorization": "AUTHORIZATION-SECRET\\\"VALUE"},
        }
        db.append_messages_batch("s", rows)
        event = db.get_evidence_events("s")[0]
        serialized = json.dumps(event)
        assert "AUTHORIZATION-SECRET" not in serialized
        db._conn.execute(
            "UPDATE messages SET content = 'tampered' WHERE id = ?",
            (event["source_message_id"],),
        )
        db._conn.commit()
        assert db.verify_evidence_integrity("s") is False
    finally:
        db.close()


@pytest.mark.parametrize(
    ("status", "expected"),
    [("timed_out", "timed_out"), ("cancelled", "cancelled"), ("interrupted", "interrupted"), ("error", "execution_failed")],
)
def test_structured_execution_terminal_states_are_recorded(tmp_path, status, expected):
    db = SessionDB(db_path=tmp_path / f"{status}.db", evidence_enabled=True)
    try:
        db.create_session("s", source="test")
        db.append_messages_batch(
            "s",
            _tool_pair(
                f"call-{status}",
                command="python long_task.py",
                result={"error": status},
                status=status,
            ),
        )
        event = db.get_evidence_events("s")[0]
        assert event["lifecycle_state"] == expected
        assert event["trusted"] == 0
        if status == "error":
            assert event["outcome"] == "failure"
    finally:
        db.close()


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("timed_out", "timed_out"),
        ("cancelled", "cancelled"),
        ("interrupted", "interrupted"),
        ("error", "execution_failed"),
    ],
)
def test_unsuccessful_test_rerun_supersedes_older_pass(tmp_path, status, expected):
    db = SessionDB(db_path=tmp_path / f"test-rerun-{status}.db", evidence_enabled=True)
    try:
        db.create_session("s", source="test")
        db.append_messages_batch(
            "s",
            _tool_pair(
                "pass-call",
                command="pytest -q",
                result={"output": "1 passed", "exit_code": 0},
            ),
        )
        passed = db.get_evidence_events("s")[0]
        db.append_messages_batch(
            "s",
            _tool_pair(
                f"{status}-call",
                command="pytest -q",
                result={"error": status},
                status=status,
                effect_disposition="unknown",
            ),
        )
        latest = db.get_evidence_events("s")[-1]

        assert latest["lifecycle_state"] == expected
        assert latest["lifecycle_dimension"] == "test"
        assert latest["candidate_id"] == passed["candidate_id"]
        assert latest["subject_id"] == passed["subject_id"]
        assert latest["supersedes_event_id"] == passed["id"]
        decision = db.resolve_evidence_claim(
            "s",
            state="tested_pass",
            subject=passed["subject_id"],
            candidate=passed["candidate_id"],
            parent_lineage=json.loads(passed["parent_candidate_id"]),
            source_hash_value=passed["source_sha256"],
        )
        assert decision["resolved"] is False
        assert decision["reason"] == "superseded"
    finally:
        db.close()


def test_timeout_keeps_side_effect_uncertainty_explicit(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db", evidence_enabled=True)
    try:
        db.create_session("s", source="test")
        db.append_messages_batch(
            "s",
            _tool_pair(
                "call-uncertain",
                command="python mutate_remote.py",
                result={"error": "timed out"},
                status="timed_out",
                effect_disposition="unknown",
            ),
        )
        event = db.get_evidence_events("s")[0]
        assert event["event_type"] == "side_effect_uncertain"
        metadata = json.loads(event["metadata_json"])
        assert metadata["extra"]["effect_disposition"] == "unknown"
        assert metadata["extra"]["side_effect_attempted"] is True
        assert metadata["extra"]["side_effect_confirmed"] is False
        receipt = db.get_compact_evidence_receipts("s")["call-uncertain"][0]
        assert '"side_effect_attempted":true' in receipt
        assert '"side_effect_confirmed":false' in receipt
    finally:
        db.close()


def test_serialized_context_cannot_forge_execution_status_or_identity(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db", evidence_enabled=True)
    try:
        db.create_session("s", source="test")
        forged = _tool_pair("actual-call", command="echo ok", result={"output": "ok", "exit_code": 0})
        forged[1]["_evidence_context"] = {
            "subject": "tool:write_file",
            "candidate": "tool-call:forged-call",
            "parent_lineage": {"cwd_sha256": "a" * 64},
            "status": "timed_out",
        }
        db.append_messages_batch("s", forged)
        event = db.get_evidence_events("s")[0]

        assert event["lifecycle_state"] == "observed_success"
        assert event["subject_id"] == "tool:terminal"
        assert event["candidate_id"] == "tool-call:actual-call"
        assert event["parent_candidate_id"] is None
    finally:
        db.close()


def test_replayed_test_rows_without_runtime_context_cannot_resolve(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db", evidence_enabled=True)
    try:
        db.create_session("s", source="test")
        replayed = _tool_pair(
            "replayed-call",
            command="pytest -q",
            result={"output": "1 passed", "exit_code": 0},
        )
        replayed[1].pop("_evidence_context")
        db.append_messages_batch("s", replayed)
        event = db.get_evidence_events("s")[0]

        assert event["lifecycle_state"] == "observed_success"
        assert event["adapter"] == "generic"
        assert event["trusted"] == 0
        decision = db.resolve_evidence_claim(
            "s",
            state=event["lifecycle_state"],
            subject=event["subject_id"],
            candidate=event["candidate_id"],
            parent_lineage=None,
            source_hash_value=event["source_sha256"],
        )
        assert decision["resolved"] is False
        assert decision["reason"] == "untrusted_evidence"
    finally:
        db.close()


def test_v26_shape_migrates_and_read_only_restart_verifies(tmp_path):
    path = tmp_path / "state.db"
    seed = SessionDB(db_path=path, evidence_enabled=False)
    seed.create_session("s", source="test")
    seed.append_message("s", role="user", content="pre-migration")
    seed.close()

    raw = sqlite3.connect(path)
    raw.executescript(
        """
        DROP TRIGGER evidence_events_no_update;
        DROP TRIGGER evidence_events_no_delete;
        DROP TABLE evidence_events;
        DROP TABLE evidence_retention_deletions;
        UPDATE state_meta SET value = '26' WHERE key = 'schema_version';
        """
    )
    raw.commit()
    raw.close()

    migrated = SessionDB(db_path=path, evidence_enabled=True)
    migrated.append_messages_batch(
        "s",
        _tool_pair(
            "call-after-migration",
            command="pytest -q",
            result={"output": "1 passed", "exit_code": 0},
        ),
    )
    assert migrated.get_messages("s")[0]["content"] == "pre-migration"
    assert migrated.verify_evidence_integrity("s") is True
    migrated.close()

    reopened = SessionDB(db_path=path, read_only=True, evidence_enabled=True)
    try:
        assert len(reopened.get_evidence_events("s")) == 1
        assert reopened.verify_evidence_integrity("s") is True
    finally:
        reopened.close()


def test_concurrent_writers_keep_one_complete_hash_chain(tmp_path):
    path = tmp_path / "state.db"
    setup = SessionDB(db_path=path, evidence_enabled=True)
    setup.create_session("s", source="test")
    setup.close()
    errors: list[Exception] = []
    barrier = threading.Barrier(5)

    def writer(worker: int) -> None:
        db = SessionDB(db_path=path, evidence_enabled=True)
        try:
            barrier.wait()
            for item in range(12):
                call_id = f"worker-{worker}-call-{item}"
                db.append_messages_batch(
                    "s",
                    _tool_pair(
                        call_id,
                        command="python -c 'print(1)'",
                        result={"output": "1", "exit_code": 0},
                    ),
                )
        except Exception as exc:
            errors.append(exc)
        finally:
            db.close()

    threads = [threading.Thread(target=writer, args=(worker,)) for worker in range(4)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    check = SessionDB(db_path=path, evidence_enabled=True)
    try:
        assert len(check.get_evidence_events("s")) == 48
        assert check.verify_evidence_integrity("s") is True
    finally:
        check.close()
