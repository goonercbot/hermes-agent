"""The explicit waiting classification uses live owned units, not chat text or sessions."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

from tools import async_delegation, delegate_tool
from tools.delegate_waiting_reply import dispatch_identity, inherit_request_root, root_for_owned_units, waiting_reply
from tools.delegate_tool_dispatch import _Batch, _dispatch_background
from tools.registry import registry
from run_agent import AIAgent


@pytest.fixture
def units():
    done = threading.Event()
    handles = []
    def add(agent):
        identity = dispatch_identity(agent)
        result = async_delegation.dispatch_async_delegation_batch(
            goals=["work"], context=None, toolsets=None, role="leaf", model=None,
            session_key="same-session", parent_session_id="same-session",
            origin_session_id="same-session", runner=lambda: (done.wait(10) and {"results": []}),
            waiting_identity=identity,
        )
        assert result["status"] == "dispatched"
        handles.append(result["delegation_id"])
        return handles[-1]
    yield add
    done.set()


def parent(turn="turn-1"):
    return SimpleNamespace(session_id="same-session", _current_turn_id=turn)


def mark(agent, ids):
    return json.loads(registry.dispatch("delegate_task", {
        "action": "classify_waiting_reply", "delegation_ids": ids,
    }, parent_agent=agent))


def test_model_handler_and_native_intercept(units):
    agent = parent()
    a, b = units(agent), units(agent)
    assert mark(agent, [a, b])["status"] == "classified"
    state = waiting_reply(agent)
    assert state.delegation_ids == (a, b)
    assert state.root_request_id and state.turn_id == "turn-1"
    result = json.loads(AIAgent._dispatch_delegate_task(agent, {
        "action": "classify_waiting_reply", "delegation_ids": [a],
    }))
    assert result["status"] == "classified"
    assert waiting_reply(agent).delegation_ids == (a,)
    assert "error" in mark(agent, ["not-a-unit"])
    assert waiting_reply(agent) is None
    agent._current_turn_id = "turn-2"
    assert waiting_reply(agent) is None


def test_invalid_foreign_completed_and_mixed_roots_fail_closed(units):
    agent, foreign = parent(), parent()
    first, outsider = units(agent), units(foreign)
    original = dispatch_identity(agent).root_request_id
    agent._current_turn_id = "turn-2"
    second = units(agent)
    assert dispatch_identity(agent).root_request_id != original
    assert root_for_owned_units(agent, [first, second]) is None
    assert root_for_owned_units(foreign, [first]) is None
    for ids in ([], ["not-a-unit"], [outsider], [first, second], [second, second], [None], "bad"):
        assert "error" in mark(agent, ids)
        assert waiting_reply(agent) is None
    with async_delegation._records_lock:
        async_delegation._records[second]["status"] = "completed"
    assert "error" in mark(agent, [second])
    assert waiting_reply(agent) is None


def test_explicit_inheritance_never_inferred_from_session(units):
    agent = parent()
    first = units(agent)
    root = dispatch_identity(agent).root_request_id
    agent._current_turn_id = "turn-2"
    assert "error" in mark(agent, [first])
    assert inherit_request_root(agent, "turn-2", root)
    second = units(agent)
    assert mark(agent, [first, second])["status"] == "classified"
    assert waiting_reply(agent).root_request_id == root
    assert root_for_owned_units(agent, [first, second]) == root
    assert not inherit_request_root(agent, "stale-turn", root)
    assert not inherit_request_root(agent, "turn-2", "another-root")


def test_missing_turn_cannot_classify(units):
    agent = parent("")
    assert dispatch_identity(agent) is None
    assert "error" in mark(agent, ["anything"])
    assert waiting_reply(agent) is None


def test_real_background_dispatch_reaches_model_handler(monkeypatch):
    import tools.delegate_tool_dispatch as dispatch

    agent = parent()
    child = SimpleNamespace(model="test", get_activity_summary=lambda: {})
    task = {"goal": "work"}
    done = threading.Event()
    monkeypatch.setattr(dispatch, "_resolve_async_wake_sid", lambda *a, **k: "wake")
    monkeypatch.setattr(dispatch, "_detach_child", lambda *a: None)
    monkeypatch.setattr(dispatch, "_execute_and_aggregate", lambda *a, **k: (done.wait(10) and {"results": []}))
    batch = _Batch(
        task_list=[task], children=[(0, task, child)], parent_agent=agent,
        creds={"model": "test"}, context=None, top_role="leaf", max_children=1,
        live_deleg_id=None, live_writers=[], live_paths=[], origin_wake_sid="",
        origin_ui_session_id="", origin_owner_transport=None, origin_owner_session_record=None,
        origin_session_history_delivery=False, overall_start=time.monotonic(),
    )
    try:
        payload = json.loads(_dispatch_background(batch))
        assert payload["status"] == "dispatched"
        assert mark(agent, [payload["delegation_id"]])["status"] == "classified"
        assert waiting_reply(agent).delegation_ids == (payload["delegation_id"],)
    finally:
        done.set()


def test_model_schema_advertises_only_pure_wait_and_oneshot_hides_it():
    schema = delegate_tool._build_dynamic_schema_overrides()
    props = schema["parameters"]["properties"]
    assert "classify_waiting_reply" in props["action"]["enum"]
    assert "never use it for a genuine or substantive answer" in schema["description"]
    inline = delegate_tool.oneshot_delegate_function({"name": "delegate_task", **schema})
    assert "delegation_ids" not in inline["parameters"]["properties"]
    assert "classify_waiting_reply" not in inline["description"]
