"""Final provider-boundary checks for maintenance-only requests."""
from copy import deepcopy
from types import SimpleNamespace as NS
from typing import Any

import pytest

from agent import relay_llm, turn_api_call
from agent.native_note_refresh import NativeNoteRefreshFailure
from hermes_cli import middleware


def _call(agent, request, capability: Any = False):
    return turn_api_call.perform_api_call(
        agent, api_kwargs=request, _original_api_kwargs=deepcopy(request),
        _llm_middleware_trace=[], _moa_prepared_request=None, _retry=NS(),
        thinking_spinner=None, retry_count=0, api_call_count=1,
        api_request_id="turn:api:1", effective_task_id="task", turn_id="turn",
        interrupted=False, native_note_refresh_request=capability,
    )


def _agent():
    return NS(
        api_mode="codex_responses", provider="openai-codex", model="test-model",
        session_id="test-session", platform="cli", base_url="https://example.invalid",
        _get_transport=lambda: NS(preflight_kwargs=lambda request, **_kwargs: request),
        _is_copilot_url=lambda: False, _is_codex_backend=lambda: True,
        _has_pending_redirect=lambda: False,
    )


@pytest.mark.parametrize("mutation_stage", [None, "middleware", "relay"])
def test_maintenance_checks_final_payload_and_never_streams(monkeypatch, mutation_stage):
    request = {"model": "test-model", "input": [], "instructions": "maintenance only"}
    expected = deepcopy(request)
    checked, sent = [], []
    response = object()
    agent = _agent()
    agent._interruptible_api_call = lambda payload: sent.append(deepcopy(payload)) or response
    agent._interruptible_streaming_api_call = lambda *_args, **_kwargs: pytest.fail(
        "maintenance must not stream"
    )
    monkeypatch.setattr(turn_api_call, "_should_stream", lambda _agent: True)

    def check_request(_agent, payload):
        checked.append(deepcopy(payload))
        if payload != expected:
            raise NativeNoteRefreshFailure("maintenance request changed after preparation")

    def execute_middleware(payload, invoke, **_kwargs):
        payload = deepcopy(payload)
        if mutation_stage == "middleware":
            payload["instructions"] = "changed by middleware"
        return invoke(payload)

    def execute_relay(payload, invoke, **_kwargs):
        payload = deepcopy(payload)
        if mutation_stage == "relay":
            payload["instructions"] = "changed by relay"
        return invoke(payload)

    monkeypatch.setattr(middleware, "run_llm_execution_middleware", execute_middleware)
    monkeypatch.setattr(relay_llm, "execute", execute_relay)
    capability = NS(check_request=check_request)
    if mutation_stage is None:
        verdict = _call(agent, request, capability)
        assert verdict.action == "fallthrough"
        assert verdict.response is response
        assert sent == [expected]
    else:
        with pytest.raises(NativeNoteRefreshFailure, match="changed after preparation"):
            _call(agent, request, capability)
        assert sent == []
    assert len(checked) == 1


def test_ordinary_request_keeps_streaming_path(monkeypatch):
    agent = _agent()
    streamed = []
    response = object()
    agent._interruptible_streaming_api_call = lambda payload, **_kwargs: streamed.append(payload) or response
    agent._interruptible_api_call = lambda *_args, **_kwargs: pytest.fail("ordinary stream was disabled")
    monkeypatch.setattr(turn_api_call, "_should_stream", lambda _agent: True)
    monkeypatch.setattr(
        middleware, "run_llm_execution_middleware",
        lambda payload, invoke, **_kwargs: invoke(payload),
    )
    request = {"model": "test-model", "input": []}
    verdict = _call(agent, request)
    assert verdict.response is response
    assert streamed == [request]
