"""Real SDK response-body drops must preserve managed-stream recovery gates."""
import json
from unittest.mock import Mock

import anthropic
import httpx2
import pytest

from agent import relay_runtime
from agent.chat_completion_helpers import _StreamingCall
from run_agent import AIAgent


def _sse(kind, **fields):
    return (f"event: {kind}\ndata: " + json.dumps({"type": kind, **fields}) + "\n\n").encode()


def _start():
    return _sse("message_start", message={
        "id": "msg_transport_test", "type": "message", "role": "assistant",
        "content": [], "model": "claude-test", "stop_reason": None,
        "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 0},
    })


def _text(text):
    return b"".join([
        _sse("content_block_start", index=0, content_block={"type": "text", "text": ""}),
        _sse("content_block_delta", index=0, delta={"type": "text_delta", "text": text}),
        _sse("content_block_stop", index=0),
    ])


def _complete():
    return _start() + _text("hello") + _sse(
        "message_delta", delta={"stop_reason": "end_turn", "stop_sequence": None},
        usage={"output_tokens": 1},
    ) + _sse("message_stop")


@pytest.fixture
def managed_stream(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("agent.chat_completion_helpers._wait_stream_retry_backoff", lambda *args: None)
    relay_runtime._reset_for_tests()
    delivered = []
    agent = AIAgent(
        api_key="synthetic-test-key", base_url="https://api.anthropic.com", provider="anthropic",
        model="claude-test", quiet_mode=True, skip_context_files=True, skip_memory=True,
        stream_delta_callback=delivered.append,
    )
    agent.api_mode = "anthropic_messages"
    agent.session_id = "transport-recovery-test"
    agent._interrupt_requested = False
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(), session_id=agent.session_id, platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(lease, turn_id="transport-turn", task_id="transport-task")
    lease.host.retain_managed_execution("test.transport_recovery")
    clients = []
    try:
        yield agent, delivered, clients
    finally:
        lease.host.release_managed_execution("test.transport_recovery")
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="test")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()
        for client in clients:
            client.close()


@pytest.mark.parametrize("error_type", [httpx2.ReadTimeout, httpx2.ReadError, httpx2.RemoteProtocolError])
@pytest.mark.parametrize("phase", ["before_delivery", "partial_tool", "delivered_text"])
def test_managed_sdk_body_drop_recovers_without_duplicate_text(managed_stream, error_type, phase):
    agent, delivered, clients = managed_stream
    prefix = _start()
    if phase != "before_delivery":
        prefix += _text("preamble")
    if phase == "partial_tool":
        prefix += _sse("content_block_start", index=1, content_block={
            "type": "tool_use", "id": "tool_test", "name": "read_file", "input": {},
        }) + _sse("content_block_delta", index=1, delta={
            "type": "input_json_delta", "partial_json": '{"path":',
        })

    class DroppedBody(httpx2.SyncByteStream):
        def __iter__(self):
            yield prefix
            raise error_type("synthetic response-body drop")

    requests = []

    def respond(request):
        body = json.loads(request.content)
        assert body["temperature"] == 1
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}
        requests.append(request)
        options = {"stream": DroppedBody()} if len(requests) == 1 else {"content": _complete()}
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, request=request, **options)

    def make_client(*args, **kwargs):
        client = anthropic.Anthropic(
            api_key="synthetic-test-key", max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
        )
        clients.append(client)
        return client

    agent._create_request_anthropic_client = make_client
    kwargs = {"model": "claude-test", "max_tokens": 2048, "messages": [{"role": "user", "content": "hi"}],
              "temperature": 1, "thinking": {"type": "enabled", "budget_tokens": 1024}}
    if phase == "delivered_text":
        result = agent._interruptible_streaming_api_call(kwargs)
        assert len(requests) == 1
        assert "".join(delivered) == "preamble"
        assert result.content[0].text == "preamble"
        assert result.stop_reason == "max_tokens"
    else:
        result = agent._interruptible_streaming_api_call(kwargs)
        assert len(requests) == 2
        assert result.content[0].text == "hello"
        assert result.stop_reason == "end_turn"
        assert "".join(delivered).count("hello") == 1
        if phase == "partial_tool":
            assert "".join(delivered).count("preamble") == 1
            assert agent._current_streamed_assistant_text == "hello"


@pytest.mark.parametrize("error_type", [httpx2.ReadTimeout, httpx2.ReadError, httpx2.RemoteProtocolError])
def test_cancelled_stream_never_reconnects(error_type):
    call = _StreamingCall.__new__(_StreamingCall)
    call._request_cancelled = {"value": True}
    call._retry_after_drop = Mock()
    call.result = {}
    assert call._handle_stream_error(error_type("cancelled"), 0, 2) is False
    call._retry_after_drop.assert_not_called()
    assert call.result == {}
