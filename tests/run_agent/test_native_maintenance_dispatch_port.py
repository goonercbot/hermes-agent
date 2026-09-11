"""Native note-refresh must cross the real executor/policy boundary once."""
import importlib.util
from pathlib import Path
import sys
import threading
import uuid
from types import SimpleNamespace

import pytest


@pytest.fixture
def router_policy(monkeypatch, tmp_path):
    """Install the bundled router through its actual PluginManager hook surface."""
    from hermes_cli.plugins import PluginManager

    path = Path(__file__).resolve().parents[1] / "fixtures/native_maintenance_router/__init__.py"
    name = f"native_dispatch_router_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    module._ROUTER = module.RouteStore(state_root=tmp_path / "router-state")

    post_calls = []
    manager = PluginManager(scope_key=str(tmp_path))
    manager._discovered = True
    manager._hooks["pre_tool_call"] = [module.pre_tool_call]

    def observe_post(*args, **kwargs):
        post_calls.append(kwargs)
        return module.post_tool_call(*args, **kwargs)

    manager._hooks["post_tool_call"] = [observe_post]
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    return post_calls


def _capability():
    from tests.run_agent.test_native_note_refresh_capability import capability_fixture

    agent, capability = capability_fixture()
    agent._current_turn_id = "native-dispatch-turn"
    agent._tool_worker_threads_lock = threading.Lock()
    agent._tool_worker_threads = set()
    agent._interrupt_requested = False
    agent._tool_guardrails = SimpleNamespace(
        before_call=lambda *_args: SimpleNamespace(allows_execution=True),
    )
    agent._guardrail_block_result = lambda decision: (
        '{"error": "' + decision.message + '"}'
    )
    agent._checkpoint_mgr = SimpleNamespace(enabled=False)
    agent._touch_activity = lambda *_args: None
    agent._current_tool = None
    agent.quiet_mode = True
    agent.tool_progress_callback = None
    agent.tool_start_callback = None
    return agent, capability


def test_authenticated_maintenance_dispatches_once_through_real_router(router_policy):
    """No mocked middleware: router admission sees the consumed opaque capability."""
    from agent.tool_executor import _run_sequential_tool_execution_middleware
    from tests.run_agent.test_native_note_refresh_capability import ARGS

    agent, capability = _capability()
    executed = []
    result = _run_sequential_tool_execution_middleware(
        agent,
        function_name="continuity_note",
        function_args=dict(ARGS),
        effective_task_id=agent.session_id,
        tool_call_id="note-call",
        maintenance_capability=capability,
        execute=lambda final_args: executed.append(final_args) or "recorded",
    )

    assert result.dispatched and not result.blocked and result.result == "recorded"
    assert executed == [ARGS]
    assert capability.consumed is True
    assert len(router_policy) == 1
    assert router_policy[0]["maintenance_context"] is capability
    assert router_policy[0]["args"] == ARGS


def test_ordinary_continuity_note_still_hits_router_guard(router_policy):
    from agent.tool_executor import _run_sequential_tool_execution_middleware
    from tests.run_agent.test_native_note_refresh_capability import ARGS

    agent, _ = _capability()
    result = _run_sequential_tool_execution_middleware(
        agent,
        function_name="continuity_note",
        function_args=dict(ARGS),
        effective_task_id=agent.session_id,
        tool_call_id="ordinary-note",
        execute=lambda _args: pytest.fail("ordinary call must not dispatch"),
    )

    assert result.blocked and result.dispatched
    assert "Record fleet_route_task first" in result.result
    assert len(router_policy) == 1
    assert "maintenance_context" not in router_policy[0]


def test_capability_does_not_bypass_an_independent_guardrail_denial(router_policy):
    from agent.tool_executor import _run_sequential_tool_execution_middleware
    from tests.run_agent.test_native_note_refresh_capability import ARGS

    agent, capability = _capability()
    agent._tool_guardrails.before_call = lambda *_args: SimpleNamespace(
        allows_execution=False, message="independent guardrail denial",
    )
    result = _run_sequential_tool_execution_middleware(
        agent,
        function_name="continuity_note",
        function_args=dict(ARGS),
        effective_task_id=agent.session_id,
        tool_call_id="note-call",
        maintenance_capability=capability,
        execute=lambda _args: pytest.fail("denied maintenance must not dispatch"),
    )

    assert result.blocked and result.dispatched
    assert "independent guardrail denial" in result.result
    assert capability.consumed is True
    assert len(router_policy) == 1
    assert router_policy[0]["maintenance_context"] is capability


def test_policy_dispatch_failure_blocks_maintenance_instead_of_bypassing(router_policy, monkeypatch):
    from agent.tool_executor import _run_sequential_tool_execution_middleware
    from tests.run_agent.test_native_note_refresh_capability import ARGS

    agent, capability = _capability()
    monkeypatch.setattr(
        "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("policy boundary failed")),
    )
    result = _run_sequential_tool_execution_middleware(
        agent,
        function_name="continuity_note",
        function_args=dict(ARGS),
        effective_task_id=agent.session_id,
        tool_call_id="note-call",
        maintenance_capability=capability,
        execute=lambda _args: pytest.fail("policy-dispatch failure must not dispatch"),
    )

    assert result.blocked and result.dispatched
    assert "pre-tool policy evaluation failed" in result.result
    assert capability.consumed is True
    # The terminal observer still receives the synthesized blocked result; the
    # pre-hook never ran because its dispatcher boundary failed.
    assert len(router_policy) == 1
    assert "pre-tool policy evaluation failed" in router_policy[0]["result"]
