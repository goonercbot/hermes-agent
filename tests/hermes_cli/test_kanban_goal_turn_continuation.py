"""Native single-query goal drivers retain unfinished Kanban turns for their judge.

These fixtures exercise the real stop gate and CLI/goal-loop wiring with scripted
model responses.  They prove control flow, not provider autonomy.
"""

from __future__ import annotations

import os
import threading
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
        self.thread_ids: list[int] = []
        self._pending_cli_user_message = {"display_metadata": None}

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
        self.thread_ids.append(threading.get_ident())
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
        def _chat(prompt, **kwargs):
            response = probe.chat(prompt, **kwargs)
            worker._last_turn_result = {"final_response": response}
            return response

        worker = SimpleNamespace(
            _single_query_mode=False,
            _claim_active_session=lambda *args, **kwargs: True,
            console=SimpleNamespace(print=lambda *args, **kwargs: None),
            _show_security_advisories=lambda: None,
            chat=_chat,
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


def test_nonquiet_goal_driver_carries_only_validated_scope_into_real_agent_threads(
    monkeypatch, _isolated_kanban_goal_worker,
):
    """``-q`` first and continued turns cross the actual CLI Thread boundary.

    Python does not inherit ContextVars into ``threading.Thread``. This test uses
    the native chat entry point (rather than replacing ``cli.chat`` with a
    synchronous probe) so a stop-gate result proves the scope reached each
    worker thread.
    """
    probe = _StopGateProbe()
    shell = cli.HermesCLI(compact=True, max_turns=1)
    shell.agent = probe

    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: (
        "continue", "another concrete step is required", False, None, False,
    ))
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda query, image: (query, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda images: [])
    monkeypatch.setattr(cli, "_finalize_single_query", lambda worker: None)
    monkeypatch.setattr(shell, "_claim_active_session", lambda *a, **k: True)
    monkeypatch.setattr(shell, "_show_security_advisories", lambda: None)
    monkeypatch.setattr(shell, "_print_exit_summary", lambda **k: None)
    monkeypatch.setattr(shell, "_ensure_runtime_credentials", lambda: True)
    monkeypatch.setattr(shell, "_resolve_turn_agent_config", lambda message: {
        "signature": shell._active_agent_route_signature, "model": None, "runtime": None,
    })
    monkeypatch.setattr(shell, "_init_agent", lambda **kwargs: True)
    monkeypatch.setattr(shell, "_sync_fallback_chain_with_config", lambda agent: None)
    monkeypatch.setattr(shell, "_chat_route_images", lambda message, images: message)
    monkeypatch.setattr(shell, "_chat_expand_context_references", lambda message: (message, None))
    monkeypatch.setattr(shell, "_chat_stage_user_message", lambda agent, message: None)
    monkeypatch.setattr(shell, "_reset_stream_state", lambda: None)
    monkeypatch.setattr(shell, "_chat_setup_turn_audio", lambda turn, message, voice_input: None)
    monkeypatch.setattr(shell, "_chat_monitor_agent_thread", lambda turn, thread: (thread.join(5), None)[1])
    monkeypatch.setattr(shell, "_chat_settle_turn", lambda turn: setattr(shell, "_last_turn_result", turn.result))
    monkeypatch.setattr(shell, "_chat_render_turn", lambda turn, thread, interrupt: turn.result["final_response"])
    monkeypatch.setattr(shell, "_chat_release_turn_audio", lambda turn: None)
    monkeypatch.setattr(shell, "_flush_credit_notices", lambda: None)

    with pytest.raises(SystemExit) as exc:
        cli._run_single_query_mode(shell, "start the task", None, False, True)

    assert exc.value.code == 0
    assert probe.stop_gate_continues == [False, False, False]
    assert len(probe.thread_ids) == 3
    assert all(thread_id != threading.get_ident() for thread_id in probe.thread_ids)
    with kbc.connect() as conn:
        task = kb.get_task(conn, _isolated_kanban_goal_worker)
    assert task is not None and task.status == "blocked"


@pytest.mark.parametrize("resumed", [False, True], ids=["fresh", "resumed"])
def test_quiet_goal_turns_carry_returned_history_including_compaction(monkeypatch, resumed):
    """A retained session ID alone is not contextual continuation."""
    from copy import deepcopy
    from hermes_cli import quiet_single_query

    initial = [{"role": "user", "content": "retained constraints"}] if resumed else []
    first = initial + [
        {"role": "user", "content": "start the task"},
        {"role": "assistant", "content": "step one", "tool_calls": [{
            "id": "call_step", "type": "function",
            "function": {"name": "terminal", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "call_step", "content": "step-one evidence"},
        {"role": "assistant", "content": "concrete progress one"},
    ]
    compacted = [{"role": "user", "content": "compacted constraints and step-one evidence"},
                 {"role": "assistant", "content": "concrete progress two"}]
    final = compacted + [{"role": "assistant", "content": "concrete progress three"}]
    returned = [first, compacted, final]
    seen = []
    probe = _StopGateProbe()

    def _run(**kwargs):
        seen.append(deepcopy(kwargs["conversation_history"]))
        if len(seen) == 2:
            probe.session_id = "compressed-goal-session"
        return {"final_response": f"step {len(seen)}", "messages": deepcopy(returned[len(seen) - 1])}

    monkeypatch.setattr(probe, "run_conversation", _run)
    monkeypatch.setattr(quiet_single_query, "continue_quiet_notify_completions", lambda *a, **k: None)
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: (
        "continue", "another concrete step", False, None, False,
    ))
    worker = SimpleNamespace(agent=probe, conversation_history=deepcopy(initial), session_id=probe.session_id)
    with pytest.raises(SystemExit) as exc:
        cli._run_quiet_single_query(worker, "start the task")
    assert exc.value.code == 0
    assert seen == [initial, first, compacted]
    assert worker.conversation_history == final
    assert worker.session_id == "compressed-goal-session"


class _OneShotResultAgent:
    def __init__(self, result):
        self.result = result
        self.calls = 0
        self.session_id = "one-shot-goal-worker"

    def run_conversation(self, **_kwargs):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return dict(self.result)


@pytest.mark.parametrize("quiet", [True, False], ids=["quiet", "nonquiet"])
@pytest.mark.parametrize(
    ("turn_result", "expected_exit"),
    [
        ({"final_response": "budget hit", "completed": False, "api_calls": 500}, 1),
        ({"final_response": "failed", "failed": True, "failure_reason": "task_error"}, 1),
        ({"final_response": "partial", "partial": True, "completed": False}, 1),
        ({"final_response": "cancelled", "interrupted": True, "completed": False}, 130),
        ({"final_response": "rate limited", "failed": True, "failure_reason": "rate_limit"}, kb.KANBAN_RATE_LIMIT_EXIT_CODE),
        ({"final_response": "auth rejected", "failed": True, "failure_reason": "auth"}, kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE),
        (RuntimeError("turn raised"), 1),
    ],
    ids=["inner-cap", "failed", "partial", "interrupted", "rate-limit", "auth", "exception"],
)
@pytest.mark.parametrize("failure_turn", [1, 2], ids=["initial", "continued"])
def test_goal_mode_preserves_unsuccessful_inner_turn_exit_without_another_model_turn(
    monkeypatch, _isolated_kanban_goal_worker, quiet, turn_result, expected_exit, failure_turn,
):
    """The real one-shot entries retain cap/failure/partial/interrupt outcomes.

    A goal loop owns only a successful completed turn. In particular, it must
    not turn an inner execution cap into an extra model call or a sticky block.
    """
    loop_calls = []
    if failure_turn == 1 and isinstance(turn_result, Exception):
        pytest.skip("This regression exercises continuation exceptions; initial CLI exceptions are separate.")
    agent = _OneShotResultAgent(turn_result)
    chat_calls = []
    if failure_turn == 2:
        original_run = agent.run_conversation

        def _run(**kwargs):
            if agent.calls == 0:
                agent.calls += 1
                return {"final_response": "concrete progress", "completed": True}
            return original_run(**kwargs)

        monkeypatch.setattr(agent, "run_conversation", _run)
        monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: (
            "continue", "another step is required", False, None, False,
        ))
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda query, image: (query, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda images: [])
    monkeypatch.setattr(cli, "_finalize_single_query", lambda worker: None)
    if failure_turn == 1:
        monkeypatch.setattr(cli, "_run_kanban_goal_loop_q", lambda *a, **k: loop_calls.append("quiet"))
        monkeypatch.setattr(cli, "_run_kanban_goal_loop_chat", lambda *a, **k: loop_calls.append("nonquiet"))

    if quiet:
        from hermes_cli import quiet_single_query

        monkeypatch.setattr(quiet_single_query, "continue_quiet_notify_completions", lambda *a, **k: None)
        worker = SimpleNamespace(
            _single_query_mode=False,
            _claim_active_session=lambda *a, **k: True,
            _ensure_runtime_credentials=lambda: True,
            _resolve_turn_agent_config=lambda query: {"signature": None, "model": None, "runtime": None},
            _active_agent_route_signature=None,
            _init_agent=lambda **kwargs: True,
            tool_progress_mode="all",
            agent=agent,
            conversation_history=[],
            session_id=agent.session_id,
            model="scripted",
        )
    else:

        def _chat(*_args, **_kwargs):
            chat_calls.append("turn")
            worker._last_turn_result = agent.run_conversation()
            return str(worker._last_turn_result.get("final_response") or "")

        worker = SimpleNamespace(
            _single_query_mode=False,
            _claim_active_session=lambda *a, **k: True,
            console=SimpleNamespace(print=lambda *a, **k: None),
            _show_security_advisories=lambda: None,
            chat=_chat,
            _print_exit_summary=lambda **k: None,
            _last_turn_result=turn_result,
        )

    with pytest.raises(SystemExit) as exc:
        cli._run_single_query_mode(worker, "start the task", None, quiet, True)

    assert exc.value.code == expected_exit
    assert loop_calls == []
    assert agent.calls == failure_turn
    if not quiet:
        assert len(chat_calls) == failure_turn
    with kbc.connect() as conn:
        task = kb.get_task(conn, _isolated_kanban_goal_worker)
    assert task is not None and task.status == "running"


def test_goal_driver_scope_requires_live_owned_run_and_restores_the_guard(
    monkeypatch, _isolated_kanban_goal_worker,
):
    """Goal-mode env alone, a descendant, and a terminated claim cannot bypass the guard."""
    from agent.delegation_context import delegated_child_context

    with cli._kanban_goal_driver_context() as active:
        assert active is True
        assert kanban_stop_nudge_enabled() is False
    assert kanban_stop_nudge_enabled() is True
    assert not is_kanban_goal_driver_context()

    # A normal Kanban worker remains nudged when goal mode is disabled, and
    # stale/mismatched dispatcher markers cannot manufacture the exemption.
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE")
    with cli._kanban_goal_driver_context() as active:
        assert active is False
        assert kanban_stop_nudge_enabled() is True
    monkeypatch.setenv("HERMES_KANBAN_GOAL_MODE", "1")
    live_run_id = os.environ["HERMES_KANBAN_RUN_ID"]
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "999999")
    with cli._kanban_goal_driver_context() as active:
        assert active is False
        assert kanban_stop_nudge_enabled() is True
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", live_run_id)
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    with cli._kanban_goal_driver_context() as active:
        assert active is False
        assert kanban_stop_nudge_enabled() is False
    monkeypatch.setenv("HERMES_KANBAN_TASK", _isolated_kanban_goal_worker)

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
        # A dependency block is parked as todo rather than reopened as a goal turn.
        ("todo", "stopped"),
    ],
)
def test_goal_loop_preserves_handoffs_and_dependency_parks_without_reopening(status, outcome):
    """Blocks (including dependency/input waits), review ownership and done end before judging."""
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
