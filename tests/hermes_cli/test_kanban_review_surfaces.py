"""Cross-surface regressions for the complete Kanban review lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def review_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    # kanban_request_review now rejects reviewers that are not installed profiles (#106163).
    (home / "profiles" / "reviewer").mkdir(parents=True)
    (home / "profiles" / "reviewer" / "config.yaml").write_text("{}\n")  # identity marker
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="Review tool contract", assignee="builder")
        task = kb.claim_task(conn, task_id, claimer="builder:1")
        assert task is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    return task_id


def test_review_tools_redact_handoff_and_route_changes(
    review_worker: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import kanban_tools as tools

    secret = "ghp_" + "A" * 40
    requested = json.loads(
        tools._handle_request_review({
            "summary": f"Ready; temporary token was {secret}",
            "metadata": {"token": secret, "tests_run": 7},
            "reviewer": "reviewer",
        })
    )
    assert requested["ok"] is True

    with kbc.connect() as conn:
        task = kb.get_task(conn, review_worker)
        assert task is not None
        assert task.status == "review"
        assert task.assignee == "reviewer"
        handoff = kb.latest_run(conn, review_worker)
        assert handoff is not None
        assert secret not in (handoff.summary or "")
        assert secret not in json.dumps(handoff.metadata)
        review = kb.claim_review_task(conn, review_worker, claimer="reviewer:1")
        assert review is not None

    monkeypatch.setenv("HERMES_PROFILE", "reviewer")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))
    change_secret = "sk-" + "B" * 32
    changed = json.loads(
        tools._handle_request_changes({
            "reason": f"Add a boundary assertion; leaked={change_secret}",
        })
    )
    assert changed["ok"] is True
    assert changed["implementer"] == "builder"

    with kbc.connect() as conn:
        task = kb.get_task(conn, review_worker)
        assert task is not None
        assert task.status == "ready"
        assert task.assignee == "builder"
        event = [
            item
            for item in kb.list_events(conn, review_worker)
            if item.kind == "changes_requested"
        ][-1]
        assert event.payload is not None
        assert change_secret not in event.payload["reason"]
        assert event.payload["reason"] != (
            "Add a boundary assertion; leaked=" + change_secret
        )


def test_review_tools_are_gated_and_visible_to_kanban_workers(
    review_worker: str,
) -> None:
    import tools.kanban_tools  # noqa: F401 - registers the tools
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    invalidate_check_fn_cache()
    definitions = registry.get_definitions(
        set(resolve_toolset("hermes-cli")), quiet=True
    )
    names = {
        definition["function"]["name"]
        for definition in definitions
        if "function" in definition
    }
    assert "kanban_request_review" in names
    assert "kanban_request_changes" in names

    from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS

    assert "kanban_request_changes" in EXPOSED_TOOLS
    assert "kanban_request_changes" in resolve_toolset("kanban")




def test_review_cli_round_trip_preserves_handoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="CLI review", assignee="builder")
        implementation = kb.claim_task(conn, task_id, claimer="builder:1")
        assert implementation is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(implementation.current_run_id))

    output = kc.run_slash(
        f"request-review {task_id} --summary 'ready for review' "
        "--reviewer reviewer --metadata '{\"tests_run\": 3}'"
    )
    assert "Requested review" in output

    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.assignee == "reviewer"
        handoff = kb.latest_run(conn, task_id)
        assert handoff is not None
        assert handoff.metadata == {"tests_run": 3}
        review = kb.claim_review_task(conn, task_id, claimer="reviewer:1")
        assert review is not None
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))

    output = kc.run_slash(
        f"request-changes {task_id} 'cover the malformed payload case'"
    )
    assert "Requested changes" in output
    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
        assert task.assignee == "builder"


def test_domain_and_cli_review_handoffs_redact_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    secret = "ghp_" + "R" * 40

    with kbc.connect() as conn:
        direct_id = kb.create_task(conn, title="direct redaction", assignee="builder")
        direct_run = kb.claim_task(conn, direct_id)
        assert direct_run is not None
        assert kb.request_review(
            conn,
            direct_id,
            summary=f"direct {secret}",
            metadata={"nested": [secret]},
            expected_run_id=direct_run.current_run_id,
        )
        run = kb.latest_run(conn, direct_id)
        event = [
            item for item in kb.list_events(conn, direct_id)
            if item.kind == "review_requested"
        ][-1]
        assert run is not None
        assert secret not in str(run.summary)
        assert secret not in json.dumps(run.metadata)
        assert secret not in json.dumps(event.payload)

        review = kb.claim_review_task(conn, direct_id)
        assert review is not None
        assert kb.request_changes(
            conn,
            direct_id,
            reason=f"change {secret}",
            expected_run_id=review.current_run_id,
        ) == (True, "builder")
        run = kb.latest_run(conn, direct_id)
        event = [
            item for item in kb.list_events(conn, direct_id)
            if item.kind == "changes_requested"
        ][-1]
        assert run is not None
        assert secret not in str(run.summary)
        assert secret not in json.dumps(event.payload)

        cli_id = kb.create_task(conn, title="CLI redaction", assignee="builder")
    cli_output = kc.run_slash(
        f'request-review {cli_id} --summary "cli {secret}" '
        f"--metadata '{{\"token\":\"{secret}\"}}'"
    )
    assert "Requested review" in cli_output
    assert secret not in cli_output
    with kbc.connect() as conn:
        run = kb.latest_run(conn, cli_id)
        event = [
            item for item in kb.list_events(conn, cli_id)
            if item.kind == "review_requested"
        ][-1]
        assert run is not None
        assert secret not in str(run.summary)
        assert secret not in json.dumps(run.metadata)
        assert secret not in json.dumps(event.payload)




def test_cli_reopen_review_is_transition_first_and_redacts_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    secret = "ghp_" + "Q" * 40
    with kbc.connect() as conn:
        invalid_id = kb.create_task(conn, title="not review", assignee="builder")
        review_id = kb.create_task(conn, title="review", assignee="builder")
        assert kb.request_review(conn, review_id, summary="ready")

    invalid_output = kc.run_slash(
        f'reopen-review {invalid_id} --reason "invalid {secret}"'
    )
    assert "cannot reopen" in invalid_output
    with kbc.connect() as conn:
        assert kb.list_comments(conn, invalid_id) == []

    success_output = kc.run_slash(
        f'reopen-review {review_id} --reason "revise {secret}"'
    )
    assert "Reopened" in success_output
    assert secret not in success_output
    with kbc.connect() as conn:
        task = kb.get_task(conn, review_id)
        assert task is not None
        assert task.status == "ready"
        comments = kb.list_comments(conn, review_id)
        assert len(comments) == 1
        assert secret not in comments[0].body


def test_goal_mode_review_handoff_cannot_bypass_judge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    with kbc.connect() as conn:
        tool_task = kb.create_task(
            conn,
            title="Goal-mode tool task",
            assignee="builder",
            goal_mode=True,
        )
        claimed = kb.claim_task(conn, tool_task, claimer="builder:1")
        assert claimed is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", tool_task)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))

    from tools import kanban_tools as tools

    monkeypatch.setattr(tools, "_goal_judge_available", lambda: True)
    monkeypatch.setattr(
        tools,
        "judge_goal",
        lambda *args, **kwargs: (
            "continue",
            "acceptance evidence is missing",
            False,
            None,
            False,
        ),
    )
    rejected = json.loads(tools._handle_request_review({"summary": "Looks ready."}))
    assert "error" in rejected
    assert "rejected by judge" in rejected["error"]
    with kbc.connect() as conn:
        tool_after = kb.get_task(conn, tool_task)
        assert tool_after is not None
        assert tool_after.status == "running"

    # The shell/CLI path applies the same gate and must not bypass the tool.
    with kbc.connect() as conn:
        cli_task = kb.create_task(
            conn,
            title="Goal-mode CLI task",
            assignee="builder",
            goal_mode=True,
        )
        cli_claimed = kb.claim_task(conn, cli_task, claimer="builder:2")
        assert cli_claimed is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", cli_task)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(cli_claimed.current_run_id))

    import agent.auxiliary_client as auxiliary_client
    from hermes_cli import goals

    monkeypatch.setattr(
        auxiliary_client,
        "get_text_auxiliary_client",
        lambda purpose: (object(), "judge-model"),
    )
    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *args, **kwargs: ("continue", "tests are missing", False, None, False),
    )
    output = kc.run_slash(f"request-review {cli_task} --summary 'Looks ready.'")
    assert "rejected by judge" in output
    with kbc.connect() as conn:
        cli_after = kb.get_task(conn, cli_task)
        assert cli_after is not None
        assert cli_after.status == "running"


def test_goal_loop_stops_after_reviewer_requests_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import goals

    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *args, **kwargs: pytest.fail(
            "a terminal review verdict must not be judged"
        ),
    )
    result = goals.run_kanban_goal_loop(
        task_id="t_review",
        goal_text="review the change",
        run_turn=lambda prompt: pytest.fail("must not run another reviewer turn"),
        task_status_fn=lambda: "changes_requested",
        block_fn=lambda reason: pytest.fail("must not block"),
        first_response="Changes requested.",
    )
    assert result["outcome"] == "changes_requested_by_reviewer"
    assert result["turns_used"] == 1


def _goal_mode_review_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, body: str = "Implement the change; reviewer approval is required before completion.",
) -> str:
    """A disposable claimed goal-mode card for public review-handoff tests."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True)
    (home / "profiles" / "reviewer").mkdir(parents=True)
    (home / "profiles" / "reviewer" / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn, title="Goal-mode review readiness", body=body, assignee="builder", goal_mode=True,
        )
        claimed = kb.claim_task(conn, task_id, claimer="builder:fixture")
        assert claimed is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    return task_id


def test_goal_mode_review_readiness_scopes_public_tool_and_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Fixture proof, not a live reviewer proof: both public handoff paths ask
    whether implementation is ready, while incomplete work and unreviewed
    completion remain rejected by the ordinary full-goal judge."""
    from tools import kanban_tools as tools
    import agent.auxiliary_client as auxiliary_client
    from hermes_cli import goals

    def judge(goal, last_response, **_kwargs):
        if "incomplete" in last_response.lower():
            return "continue", "implementation evidence is incomplete", False, None, False
        if _kwargs.get("review_handoff"):
            return "done", "implementation is ready for fixture review", False, None, False
        return "continue", "reviewer approval evidence is absent", False, None, False

    # Tool handler: the same-card review criterion must not deadlock the first handoff.
    tool_task = _goal_mode_review_worker(monkeypatch, tmp_path / "tool")
    monkeypatch.setattr(tools, "_goal_judge_available", lambda: True)
    monkeypatch.setattr(tools, "judge_goal", judge)
    requested = json.loads(tools._handle_request_review({
        "summary": "Implemented the change with passing fixture checks.", "reviewer": "reviewer",
    }))
    assert requested.get("ok") is True, requested
    with kbc.connect() as conn:
        assert kb.get_task(conn, tool_task).status == "review"
        review = kb.claim_review_task(conn, tool_task, claimer="reviewer:fixture")
        assert review is not None
    monkeypatch.setenv("HERMES_PROFILE", "reviewer")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))
    assert json.loads(tools._handle_request_changes({"reason": "Fixture: add a boundary assertion."}))["ok"]
    with kbc.connect() as conn:
        repair = kb.claim_task(conn, tool_task, claimer="builder:repair")
        assert repair is not None
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(repair.current_run_id))
    assert json.loads(tools._handle_request_review({
        "summary": "Fixture repair implemented and tests pass.", "reviewer": "reviewer",
    }))["ok"]
    with kbc.connect() as conn:
        rereview = kb.claim_review_task(conn, tool_task, claimer="reviewer:rereview")
        assert rereview is not None
    monkeypatch.setenv("HERMES_PROFILE", "reviewer")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(rereview.current_run_id))
    # First prove completion cannot use the implementation-readiness waiver.
    rejected_completion = json.loads(tools._handle_complete({"summary": "Implementation ready."}))
    assert "rejected by judge" in rejected_completion.get("error", "")
    monkeypatch.setattr(tools, "judge_goal", lambda **kw: ("done", "fixture review approved", False, None, False))
    assert json.loads(tools._handle_complete({"summary": "Fixture reviewer accepted repaired candidate."}))["ok"]
    with kbc.connect() as conn:
        assert kb.get_task(conn, tool_task).status == "done"
    monkeypatch.setattr(tools, "judge_goal", judge)

    # CLI handler: independently exercise the shell/slash implementation.
    cli_task = _goal_mode_review_worker(monkeypatch, tmp_path / "cli")
    monkeypatch.setattr(auxiliary_client, "get_text_auxiliary_client", lambda _purpose: (object(), "fixture-judge"))
    monkeypatch.setattr(goals, "judge_goal", judge)
    output = kc.run_slash(f"request-review {cli_task} --summary 'Implemented the change with passing fixture checks.'")
    assert "Requested review" in output
    with kbc.connect() as conn:
        assert kb.get_task(conn, cli_task).status == "review"

    # The scoped question does not waive implementation evidence on either public path.
    incomplete_tool = _goal_mode_review_worker(monkeypatch, tmp_path / "incomplete-tool")
    rejected = json.loads(tools._handle_request_review({"summary": "Implementation incomplete."}))
    assert "rejected by judge" in rejected.get("error", "")
    with kbc.connect() as conn:
        assert kb.get_task(conn, incomplete_tool).status == "running"

    incomplete_cli = _goal_mode_review_worker(monkeypatch, tmp_path / "incomplete-cli")
    output = kc.run_slash(f"request-review {incomplete_cli} --summary 'Implementation incomplete.'")
    assert "rejected by judge" in output
    with kbc.connect() as conn:
        assert kb.get_task(conn, incomplete_cli).status == "running"

    # Completion remains unscoped: review approval is still required there.
    completion_task = _goal_mode_review_worker(monkeypatch, tmp_path / "completion")
    output = kc.run_slash(f"complete {completion_task} --summary 'Implemented the change.'")
    assert "rejected by judge" in output
    with kbc.connect() as conn:
        assert kb.get_task(conn, completion_task).status == "running"


def test_cli_review_readiness_note_survives_real_judge_truncation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The CLI scope stays outside the real 2000-character task truncation.
    The fake transport labels this fixture proof."""
    from unittest.mock import patch
    import agent.auxiliary_client as auxiliary_client

    body = "Acceptance criteria:\n" + "\n".join(
        f"- Detailed implementation criterion {index}: verify a meaningful boundary."
        for index in range(80)
    )
    assert len(body) > 2000
    task_id = _goal_mode_review_worker(monkeypatch, tmp_path, body=body)
    monkeypatch.setattr(auxiliary_client, "get_text_auxiliary_client", lambda _purpose: (object(), "fixture-judge"))
    captured: dict = {}

    class _Message:
        content = '{"done": true, "reason": "fixture implementation evidence is sufficient"}'

    class _Response:
        choices = [type("Choice", (), {"message": _Message()})()]

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return _Response()

    with patch("agent.auxiliary_client.call_llm", side_effect=fake_call_llm):
        output = kc.run_slash(f"request-review {task_id} --summary 'Implemented all listed criteria.'")

    assert "Requested review" in output
    prompt = next(message["content"] for message in captured["messages"] if message["role"] == "system")
    assert "do not withhold DONE merely because reviewer/approval evidence is absent" in prompt


@pytest.mark.parametrize("surface", ["tool", "cli"])
def test_review_scope_preserves_previously_visible_implementation_criterion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, surface: str,
) -> None:
    """Real prompt construction, fake transport: scope must not consume task budget."""
    from types import SimpleNamespace
    import agent.auxiliary_client as auxiliary_client
    from tools import kanban_tools as tools

    criterion = "MANDATORY: enforce tenant isolation before review."
    title_prefix = "Goal-mode review readiness\n\n"
    body = "Implement the documented behavior.\n".ljust(1701 - len(title_prefix), ".") + criterion
    assert (title_prefix + body).index(criterion) == 1701
    task_id = _goal_mode_review_worker(monkeypatch, tmp_path, body=body)
    monkeypatch.setattr(tools, "_goal_judge_available", lambda: True)
    monkeypatch.setattr(auxiliary_client, "get_text_auxiliary_client", lambda _purpose: (object(), "fixture-judge"))
    captured = {}

    def transport(**kwargs):
        captured.update(kwargs)
        user = next(m["content"] for m in kwargs["messages"] if m["role"] == "user")
        verdict = {"done": criterion not in user, "reason": "tenant isolation is not implemented"}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(verdict)))])

    monkeypatch.setattr(auxiliary_client, "call_llm", transport)
    if surface == "tool":
        result = json.loads(tools._handle_request_review({"summary": "Basic implementation done; tenant isolation not implemented."}))
        assert "rejected by judge" in result.get("error", "")
    else:
        result = kc.run_slash(f"request-review {task_id} --summary 'Basic implementation done; tenant isolation not implemented.'")
        assert "rejected by judge" in result
    user = next(m["content"] for m in captured["messages"] if m["role"] == "user")
    system = next(m["content"] for m in captured["messages"] if m["role"] == "system")
    assert criterion in user
    assert title_prefix + body in user
    assert "ready to hand off to a reviewer" in system
    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "running"


def test_cli_and_dashboard_receive_graph_aware_deadlock_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    with kbc.connect() as conn:
        parent_id = kb.create_task(conn, title="Implementation", assignee="builder")
        child_id = kb.create_task(
            conn,
            title="Review",
            assignee="reviewer",
            parents=[parent_id],
        )
        parent = kb.claim_task(conn, parent_id, claimer="builder:1")
        assert parent is not None
        assert kb.block_task(
            conn,
            parent_id,
            reason="review-required: ready",
            expected_run_id=parent.current_run_id,
        )

    payload = json.loads(kc.run_slash(f"diagnostics --task {parent_id} --json"))
    assert any(
        item["kind"] == "review_dependency_deadlock"
        for item in payload[0]["diagnostics"]
    )

    from plugins.kanban.dashboard.plugin_api import _compute_task_diagnostics

    with kbc.connect() as conn:
        dashboard = _compute_task_diagnostics(conn, task_ids=[parent_id])
    assert dashboard[parent_id][0]["kind"] == "review_dependency_deadlock"
    assert dashboard[parent_id][0]["data"]["waiting_child_ids"] == [child_id]
