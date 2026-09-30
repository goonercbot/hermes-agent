"""Native single-query goal drivers retain unfinished Kanban turns for their judge.

These fixtures exercise the real stop gate and CLI/goal-loop wiring with scripted
model responses.  They prove control flow, not provider autonomy.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

import cli
from agent.delegation_context import is_kanban_goal_driver_context
from agent.kanban_stop import kanban_stop_nudge_enabled
from agent.turn_stop_gates import apply_stop_gates
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import goals


@pytest.fixture(autouse=True)
def _isolated_kanban_goal_worker(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_GOAL_MODE", "1")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Continue concrete work",
            body="Make three concrete steps before the goal budget ends.",
            assignee="worker",
            goal_mode=True,
            goal_max_turns=3,
        )
        claimed = kb.claim_task(conn, task_id, claimer="worker:fixture")
        assert claimed is not None and claimed.current_run_id is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    return task_id


class _StopGateProbe:
    """A scripted model boundary that invokes Hermes' actual text-response gate."""

    def __init__(self):
        self.session_id = "goal-worker-session"
        self.responses = iter(("concrete step one", "concrete step two", "concrete step three"))
        self.stop_gate_continues: list[bool] = []
        self.prompts: list[str] = []

    def _turn(self) -> str:
        response = next(self.responses)
        messages = [{"role": "user", "content": "work the task"}]
        verdict = apply_stop_gates(
            self,
            {"role": "assistant", "content": response},
            final_response=response,
            messages=messages,
            conversation_history=messages,
            pending_verification_response=None,
            pending_verification_response_previewed=False,
        )
        self.stop_gate_continues.append(verdict.continue_turn)
        return response

    def run_conversation(self, **kwargs):
        self.prompts.append(kwargs["user_message"])
        return {"final_response": self._turn(), "messages": []}

    def chat(self, prompt, **_kwargs):
        self.prompts.append(prompt)
        return self._turn()

    # The ordinary stop gate only calls these if the regression returns.
    def _emit_interim_assistant_message(self, _message):
        pass

    def _flush_messages_to_session_db(self, _messages, _history):
        pass

    def _interim_content_was_streamed(self, _content):
        return False

    def _emit_diagnostic_status(self, _message):
        pass


@pytest.mark.parametrize("quiet", [True, False], ids=["quiet", "nonquiet"])
def test_native_goal_driver_reaches_repeated_continuations_without_terminalizing(
    monkeypatch, _isolated_kanban_goal_worker, quiet,
):
    """Regression: both entry paths let the judge drive the same worker to its real budget."""
    probe = _StopGateProbe()
    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *args, **kwargs: ("continue", "another concrete step is required", False, None, False),
    )

    if quiet:
        worker = SimpleNamespace(agent=probe, conversation_history=[], session_id=probe.session_id)
        with pytest.raises(SystemExit) as exc:
            cli._run_quiet_single_query(worker, "start the task")
    else:
        monkeypatch.setattr(cli, "_should_seed_interactive", lambda *args, **kwargs: False)
        monkeypatch.setattr(cli, "_collect_query_images", lambda query, image: (query, []))
        monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda images: [])
        monkeypatch.setattr(cli, "_finalize_single_query", lambda worker: None)
        worker = SimpleNamespace(
            _single_query_mode=False,
            _claim_active_session=lambda *args, **kwargs: True,
            console=SimpleNamespace(print=lambda *args, **kwargs: None),
            _show_security_advisories=lambda: None,
            chat=probe.chat,
            _print_exit_summary=lambda **kwargs: None,
            _last_turn_result={"final_response": "concrete step one"},
        )
        with pytest.raises(SystemExit) as exc:
            cli._run_single_query_mode(worker, "start the task", None, False, True)

    assert exc.value.code == 0
    assert probe.stop_gate_continues == [False, False, False]
    assert all("report concrete progress" in prompt for prompt in probe.prompts[1:])
    assert not is_kanban_goal_driver_context(), "the exception/budget path must restore the ordinary guard"
    with kbc.connect() as conn:
        task = kb.get_task(conn, _isolated_kanban_goal_worker)
    assert task is not None and task.status == "blocked", "the unchanged goal budget fails closed"


def test_goal_driver_scope_requires_live_owned_run_and_restores_the_guard(
    _isolated_kanban_goal_worker,
):
    """Goal-mode env alone, a descendant, and a terminated claim cannot bypass the guard."""
    from agent.delegation_context import delegated_child_context

    with cli._kanban_goal_driver_context() as active:
        assert active is True
        assert kanban_stop_nudge_enabled() is False
    assert kanban_stop_nudge_enabled() is True
    assert not is_kanban_goal_driver_context()

    with pytest.raises(RuntimeError):
        with cli._kanban_goal_driver_context() as active:
            assert active is True
            raise RuntimeError("simulated inner turn failure")
    assert kanban_stop_nudge_enabled() is True
    assert not is_kanban_goal_driver_context()

    with delegated_child_context():
        with cli._kanban_goal_driver_context() as active:
            assert active is False

    with kbc.connect() as conn:
        assert kb.block_task(
            conn,
            _isolated_kanban_goal_worker,
            reason="waiting for the user",
            kind="needs_input",
            expected_run_id=int(os.environ["HERMES_KANBAN_RUN_ID"]),
        )
    with cli._kanban_goal_driver_context() as active:
        assert active is False
        assert kanban_stop_nudge_enabled() is True


@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        ("blocked", "blocked_by_worker"),
        ("review", "review_requested_by_worker"),
        ("changes_requested", "changes_requested_by_reviewer"),
        ("done", "completed_by_worker"),
    ],
)
def test_goal_loop_preserves_terminal_handoffs_without_reopening(status, outcome):
    """Blocks (including dependency/input waits) and review ownership end before judging."""
    result = goals.run_kanban_goal_loop(
        task_id="t_terminal",
        goal_text="do not reopen terminal handoffs",
        run_turn=lambda prompt: pytest.fail("terminal task must not receive another worker turn"),
        task_status_fn=lambda: status,
        block_fn=lambda reason: pytest.fail("terminal task must not be blocked again"),
        first_response="terminal handoff already recorded",
    )
    assert result["outcome"] == outcome
    assert result["turns_used"] == 1
