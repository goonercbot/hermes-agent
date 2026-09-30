"""A handoff must ask one stage-specific question without losing criteria."""
import json
from types import SimpleNamespace

import pytest

from hermes_cli import goals


@pytest.mark.parametrize("review_handoff", [False, True])
@pytest.mark.parametrize("criteria_kind", ["plain", "subgoals", "contract", "both"])
def test_phase_target_preserves_task_evidence_and_criteria(monkeypatch, review_handoff, criteria_kind):
    from agent import auxiliary_client

    captured = {}
    def call_llm(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({"verdict": "continue", "reason": "required proof missing"})))])
    monkeypatch.setattr(auxiliary_client, "call_llm", call_llm)
    task = "Implement tenant isolation. " + "x" * 1650 + " REQUIRED_LATE_IMPLEMENTATION_CHECK " + "x" * 700
    evidence = "Candidate built; validation evidence: " + "y" * 5000
    subgoals = ["Verify explicit pre-review publication", "Preserve protected source records"] if criteria_kind in ("subgoals", "both") else None
    contract = goals.GoalContract(
        outcome="Implement isolation and obtain independent acceptance",
        verification="Missing required tenant-isolation test must reject handoff",
        constraints="Do not change protected records",
        boundaries="Only scoped repository files",
        stop_when="Credentials unavailable",
    ) if criteria_kind in ("contract", "both") else None
    result = goals.judge_goal(task, evidence, review_handoff=review_handoff,
                             subgoals=subgoals, contract=contract, timeout=30.0)
    assert result[0] == "continue"
    system, user = [m["content"] for m in captured["messages"]]
    assert goals._truncate(task, 2000) in user
    assert "REQUIRED_LATE_IMPLEMENTATION_CHECK" in user
    assert goals._truncate(evidence, goals._JUDGE_RESPONSE_SNIPPET_CHARS) in user
    if subgoals:
        assert all(criterion in user for criterion in subgoals)
    if contract:
        assert contract.render_block() in user
    if review_handoff:
        assert system == goals.REVIEW_HANDOFF_SYSTEM_PROMPT
        assert "DONE — the goal is fully satisfied" not in system
        assert "Is the goal satisfied" not in user
        assert "Evaluation target: implementation-ready handoff" in user
        assert "review request that has NOT executed yet" in system
        assert "ONLY AFTER review/approval" in system
        assert "BEFORE review" in system
        assert "Requested repairs from an earlier review" in system
        assert "Continue to reject missing implementation or failing required tests." in system
    else:
        assert system == goals.JUDGE_SYSTEM_PROMPT
        assert "DONE — the goal is fully satisfied" in system
        assert user.startswith("Goal:\n")
        assert "Evaluation target: implementation-ready handoff" not in user
