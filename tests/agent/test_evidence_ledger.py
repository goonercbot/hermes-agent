"""Adversarial contract tests for the pure evidence-ledger adapters."""
from __future__ import annotations

import json

from agent.evidence_ledger import (
    ISSUER,
    adapt_tool_result,
    canonical_json,
    event_hash,
    redact_value,
    resolve_claim,
    verify_hash_chain,
)


def _resolve(events, event):
    return resolve_claim(
        events,
        state=event["state"],
        subject=event["subject"],
        candidate=event["candidate"],
        parent_lineage=event["parent_lineage"],
        source_hash_value=event["source_hash"],
        integrity_chain=verify_hash_chain,
    )


def test_direct_test_adapter_uses_explicit_exit_code_and_later_failure_wins():
    passed = adapt_tool_result(
        "terminal", {"command": "pytest -q"}, {"exit_code": 0, "output": "1 passed"},
        subject="suite", candidate="tree-a", parent_lineage={"base": "base-a"},
        trusted_execution=True,
    )
    failed = adapt_tool_result(
        "terminal", {"command": "pytest -q"}, {"exit_code": 1, "output": "1 failed"},
        subject="suite", candidate="tree-a", parent_lineage={"base": "base-a"},
        previous_event_hash=passed["event_hash"],
        trusted_execution=True,
    )

    assert passed["state"] == "tested_pass"
    assert failed["state"] == "tested_fail"
    assert verify_hash_chain([passed, failed])
    decision = _resolve([passed, failed], passed)
    assert decision["resolved"] is False
    assert decision["reason"] == "superseded"


def test_later_unsuccessful_test_attempt_supersedes_older_pass():
    for status in ("timed_out", "cancelled", "interrupted", "error"):
        parent = {"cwd_sha256": "a" * 64}
        passed = adapt_tool_result(
            "terminal",
            {"command": "pytest -q"},
            {"exit_code": 0, "output": "1 passed"},
            parent_lineage=parent,
            trusted_execution=True,
        )
        incomplete = adapt_tool_result(
            "terminal",
            {"command": "pytest -q"},
            {"status": status},
            parent_lineage=parent,
            previous_event_hash=passed["event_hash"],
            trusted_execution=True,
        )

        assert incomplete["candidate"] == passed["candidate"]
        assert incomplete["subject"] == passed["subject"]
        assert incomplete["lifecycle_dimension"] == "test"
        decision = _resolve([passed, incomplete], passed)
        assert decision["resolved"] is False
        assert decision["reason"] == "superseded"


def test_generic_shell_script_mentioning_git_merge_is_never_promoted():
    event = adapt_tool_result(
        "terminal",
        {"command": "sh -c 'echo git merge; exit 0'"},
        {"exit_code": 0, "output": "git merge"},
    )

    spoofed = adapt_tool_result("web_search", {"command": "pytest"}, {"exit_code": 0})
    assert event["state"] == "observed_success"
    assert event["outcome"] == "success"
    assert event["trusted"] is False
    assert spoofed["state"] == "observed_success"
    assert spoofed["trusted"] is False


def test_closed_adapter_without_authenticated_execution_is_never_promoted():
    replayed = adapt_tool_result(
        "terminal",
        {"command": "pytest -q"},
        {"exit_code": 0, "output": "1 passed"},
    )

    assert replayed["state"] == "observed_success"
    assert replayed["adapter"] == "generic"
    assert replayed["trusted"] is False


def test_generic_explicit_failure_is_preserved_without_lifecycle_promotion():
    event = adapt_tool_result(
        "terminal",
        {"command": "python failing_script.py"},
        {"exit_code": 7, "output": "failed"},
    )

    assert event["state"] == "observed_failure"
    assert event["outcome"] == "failure"
    assert event["trusted"] is False


def test_manually_injected_high_risk_state_is_rejected_even_with_valid_hash():
    event = adapt_tool_result("terminal", {"command": "echo pass"}, {"exit_code": 0})
    event.update(
        state="tested_pass",
        outcome="success",
        issuer=ISSUER,
        trusted=True,
        subject="suite",
        candidate="tree-a",
        parent_lineage={"base": "base-a"},
    )
    event["event_hash"] = event_hash(event)

    decision = _resolve([event], event)
    assert decision["resolved"] is False
    assert decision["reason"] == "untrusted_evidence"


def test_known_secrets_and_sensitive_keys_are_redacted_before_serialization():
    secret = 's3cr"et\\value'
    escaped = json.dumps(secret)[1:-1]
    event = adapt_tool_result(
        "read_file",
        {"path": "x", "authorization": secret},
        {"exit_code": 0, "output": f"raw={secret}; escaped={escaped}", "nested": {"api_key": secret, "accessToken": secret}},
        known_secrets=[secret],
    )

    stored = canonical_json(event)
    assert secret not in stored
    assert escaped not in stored
    assert "input" not in event["metadata"]
    assert "result" not in event["metadata"]
    assert set(event["metadata"]) >= {
        "input_sha256", "result_sha256", "result_bytes"
    }
    assert redact_value({"authorization": secret})["authorization"] == "[REDACTED]"


def test_exact_git_and_pr_adapters_bind_artifact_lineage():
    sha = "a" * 40
    commit = adapt_tool_result(
        "terminal", {"command": "git rev-parse HEAD"}, {"exit_code": 0, "output": sha},
        parent_lineage={"repository": "owner/repo"},
        trusted_execution=True,
    )
    pr = adapt_tool_result(
        "terminal",
        {"command": "gh pr view 42 --json number,state,headRefOid"},
        {"exit_code": 0, "output": json.dumps({"number": 42, "state": "MERGED", "headRefOid": sha})},
        trusted_execution=True,
    )

    assert commit["state"] == "commit_observed"
    assert commit["candidate"] == sha
    assert pr["state"] == "pr_merged"
    assert pr["candidate"] == sha
    assert pr["parent_lineage"] == {"pr_number": 42, "head_ref_oid": sha}
    live = _resolve([pr], pr)
    assert live["resolved"] is False
    assert live["reason"] == "fresh_runtime_verification_required"


def test_exact_approved_pr_snapshot_has_separate_review_state():
    sha = "b" * 40
    approved = adapt_tool_result(
        "terminal",
        {"command": "gh pr view 42 --json number,state,headRefOid,reviewDecision"},
        {
            "exit_code": 0,
            "output": json.dumps(
                {
                    "number": 42,
                    "state": "OPEN",
                    "headRefOid": sha,
                    "reviewDecision": "APPROVED",
                }
            ),
        },
        trusted_execution=True,
    )

    assert approved["state"] == "accepted"
    assert approved["trusted"] is True
    assert approved["candidate"] == sha

    revoked = adapt_tool_result(
        "terminal",
        {"command": "gh pr view 42 --json number,state,headRefOid,reviewDecision"},
        {
            "exit_code": 0,
            "output": json.dumps(
                {
                    "number": 42,
                    "state": "OPEN",
                    "headRefOid": sha,
                    "reviewDecision": "CHANGES_REQUESTED",
                }
            ),
        },
        previous_event_hash=approved["event_hash"],
        trusted_execution=True,
    )
    decision = _resolve([approved, revoked], approved)
    assert revoked["state"] == "acceptance_revoked"
    assert revoked["outcome"] == "failure"
    assert decision["resolved"] is False
    assert decision["reason"] == "superseded"


def test_interrupted_file_mutation_is_not_implemented():
    event = adapt_tool_result(
        "write_file", {"path": "agent/new.py"},
        {"status": "interrupted", "bytes_written": 12},
        subject="file:agent/new.py", candidate="tree-a", parent_lineage={"base": "base-a"},
    )

    assert event["state"] == "interrupted"
    assert event["trusted"] is False


def test_file_mutator_requires_structured_landed_success():
    generic = adapt_tool_result("write_file", {"path": "x"}, "looks good")
    landed = adapt_tool_result(
        "patch", {"path": "x"}, {"success": True}, trusted_execution=True
    )

    assert generic["state"] == "observed"
    assert landed["state"] == "implemented"
    assert landed["trusted"] is True


def test_stale_claim_rejected_on_source_hash_or_lineage_mismatch():
    event = adapt_tool_result(
        "terminal", {"command": "python -m unittest"}, {"exit_code": 0},
        subject="suite", candidate="tree-a", parent_lineage={"base": "base-a"},
        trusted_execution=True,
    )

    wrong_source = resolve_claim(
        [event], state="tested_pass", subject=event["subject"], candidate=event["candidate"],
        parent_lineage=event["parent_lineage"], source_hash_value="0" * 64,
        integrity_chain=verify_hash_chain,
    )
    wrong_lineage = resolve_claim(
        [event], state="tested_pass", subject=event["subject"], candidate="tree-b",
        parent_lineage=event["parent_lineage"], source_hash_value=event["source_hash"],
        integrity_chain=verify_hash_chain,
    )

    assert wrong_source["reason"] == "source_hash_mismatch"
    assert wrong_lineage["reason"] == "no_exact_lineage"
