import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import tool_executor
from agent.tool_executor import _ToolCancelledResult, _ToolTimeoutResult
from run_agent import AIAgent


def _tool_defs():
    return [{
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "terminal",
            "parameters": {"type": "object", "properties": {}},
        },
    }]


def _agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        return AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (_ToolTimeoutResult("timed out"), "timed_out"),
        (_ToolCancelledResult("cancelled"), "cancelled"),
        ("Error executing tool 'terminal': boom", "error"),
    ],
)
def test_sequential_executor_attaches_terminal_status(result, expected, monkeypatch):
    agent = _agent()
    messages = []
    tool_call = SimpleNamespace(
        id=f"call-{expected}",
        function=SimpleNamespace(
            name="terminal",
            arguments=json.dumps({"command": "python long_task.py"}),
        ),
    )
    assistant = SimpleNamespace(content="", tool_calls=[tool_call])
    monkeypatch.setattr(
        tool_executor,
        "get_active_env",
        lambda _task_id: SimpleNamespace(cwd="/private/isolated-worktree"),
    )
    monkeypatch.setattr(
        tool_executor,
        "_NEVER_PARALLEL_TOOLS",
        frozenset({*tool_executor._NEVER_PARALLEL_TOOLS, "terminal"}),
    )
    if expected == "error":
        monkeypatch.setattr(
            tool_executor,
            "_detect_tool_failure",
            lambda _name, _result: (True, "synthetic error"),
        )
    with patch("run_agent.handle_function_call", return_value=result):
        agent._execute_tool_calls_sequential(assistant, messages, "isolated-task")

    assert len(messages) == 1
    context = messages[0]["_evidence_context"]
    assert context["status"] == expected
    assert context["subject"] == "tool:terminal"
    assert context["candidate"] == f"tool-call:call-{expected}"
    assert "/private/isolated-worktree" not in json.dumps(context)


def test_concurrent_preflight_interrupt_records_explicit_cancellation(monkeypatch):
    agent = _agent()
    agent._interrupt_requested = True
    messages = []
    tool_call = SimpleNamespace(
        id="call-preflight-cancelled",
        function=SimpleNamespace(
            name="terminal",
            arguments=json.dumps({"command": "touch must-not-run"}),
        ),
    )
    assistant = SimpleNamespace(content="", tool_calls=[tool_call])
    monkeypatch.setattr(
        tool_executor,
        "get_active_env",
        lambda _task_id: SimpleNamespace(cwd="/private/isolated-worktree"),
    )

    tool_executor.execute_tool_calls_concurrent(
        agent, assistant, messages, "isolated-task"
    )

    assert len(messages) == 1
    assert messages[0]["effect_disposition"] == "none"
    context = messages[0]["_evidence_context"]
    assert context["status"] == "cancelled"
    assert context["candidate"] == "tool-call:call-preflight-cancelled"


def test_sequential_preflight_interrupt_records_explicit_cancellation(monkeypatch):
    agent = _agent()
    agent._interrupt_requested = True
    messages = []
    calls = [
        SimpleNamespace(
            id=f"call-sequential-{index}",
            function=SimpleNamespace(
                name="terminal",
                arguments=json.dumps({"command": f"touch must-not-run-{index}"}),
            ),
        )
        for index in range(2)
    ]
    assistant = SimpleNamespace(content="", tool_calls=calls)
    monkeypatch.setattr(
        tool_executor,
        "get_active_env",
        lambda _task_id: SimpleNamespace(cwd="/private/isolated-worktree"),
    )

    agent._execute_tool_calls_sequential(assistant, messages, "isolated-task")

    assert len(messages) == 2
    for message in messages:
        assert message["effect_disposition"] == "none"
        assert message["_evidence_context"]["status"] == "cancelled"


def test_sequential_mid_batch_interrupt_authenticates_remaining_cancellation(monkeypatch):
    agent = _agent()
    messages = []
    calls = [
        SimpleNamespace(
            id=f"call-mid-{index}",
            function=SimpleNamespace(
                name="terminal",
                arguments=json.dumps({"command": f"python task_{index}.py"}),
            ),
        )
        for index in range(2)
    ]
    assistant = SimpleNamespace(content="", tool_calls=calls)
    monkeypatch.setattr(
        tool_executor,
        "get_active_env",
        lambda _task_id: SimpleNamespace(cwd="/private/isolated-worktree"),
    )
    monkeypatch.setattr(
        tool_executor,
        "_NEVER_PARALLEL_TOOLS",
        frozenset({*tool_executor._NEVER_PARALLEL_TOOLS, "terminal"}),
    )

    def _first_result(*_args, **_kwargs):
        agent._interrupt_requested = True
        return '{"ok": true}'

    with patch("run_agent.handle_function_call", side_effect=_first_result):
        agent._execute_tool_calls_sequential(assistant, messages, "isolated-task")

    assert len(messages) == 2
    cancelled = messages[1]
    assert cancelled["effect_disposition"] == "none"
    assert cancelled["_evidence_context"]["status"] == "cancelled"
    assert cancelled["_evidence_context"]["candidate"] == "tool-call:call-mid-1"


def test_special_tool_keyboard_interrupt_persists_current_and_remaining(monkeypatch):
    agent = _agent()
    messages = []
    calls = [
        SimpleNamespace(
            id="call-special-current",
            function=SimpleNamespace(
                name="todo",
                arguments=json.dumps({"todos": []}),
            ),
        ),
        SimpleNamespace(
            id="call-special-remaining",
            function=SimpleNamespace(
                name="terminal",
                arguments=json.dumps({"command": "touch must-not-run"}),
            ),
        ),
    ]
    assistant = SimpleNamespace(content="", tool_calls=calls)
    monkeypatch.setattr(
        tool_executor,
        "get_active_env",
        lambda _task_id: SimpleNamespace(cwd="/private/isolated-worktree"),
    )
    monkeypatch.setattr(
        tool_executor,
        "_run_sequential_tool_execution_middleware",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        agent._execute_tool_calls_sequential(assistant, messages, "isolated-task")

    assert len(messages) == 2
    assert messages[0]["effect_disposition"] == "unknown"
    assert messages[1]["effect_disposition"] == "none"
    assert all(
        message["_evidence_context"]["status"] == "cancelled"
        for message in messages
    )
