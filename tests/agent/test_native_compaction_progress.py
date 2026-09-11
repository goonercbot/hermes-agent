"""Native compression progress belongs to one request and one commit fence."""
import threading
import time
from types import SimpleNamespace as NS

import pytest

from agent.chat_completion_helpers import interruptible_api_call
from agent.codex_runtime import run_codex_stream
from agent.conversation_compression import (
    CompressionCommitFence,
    run_compress_context_with_progress_timeout,
)
from agent.native_compaction_progress import (
    current_native_compaction_request,
    native_compaction_request,
)
from tools.thread_context import propagate_context_to_thread


def test_scope_is_propagated_but_not_rebound_to_later_request():
    first, second = CompressionCommitFence(), CompressionCommitFence()
    observed = []
    with native_compaction_request(first) as old:
        target = propagate_context_to_thread(lambda: observed.append(current_native_compaction_request()))
        with native_compaction_request(second) as new:
            thread = threading.Thread(target=target)
            thread.start(); thread.join(2)
            assert observed == [old]
            assert current_native_compaction_request() is new
            old.on_event()
            assert first.progress_observed and not second.progress_observed
        assert current_native_compaction_request() is old
        with pytest.raises(InterruptedError):
            new.on_event()
    assert current_native_compaction_request() is None
    with pytest.raises(InterruptedError):
        old.on_event()


def test_cancelled_fence_cannot_receive_late_progress_or_affect_next_fence():
    old, new = CompressionCommitFence(), CompressionCommitFence()
    with native_compaction_request(old) as request:
        assert old.cancel_before_commit()
        with native_compaction_request(new):
            with pytest.raises(InterruptedError): request.on_event()
        assert not old.progress_observed and not new.progress_observed


@pytest.mark.parametrize('keepalive', [False, True])
def test_idle_and_total_ceilings_still_cancel_native_work(keepalive):
    done = threading.Event()
    fence = CompressionCommitFence()
    causes = []
    original = [{'role':'user', 'content':'fixture'}]
    def worker(bound_fence):
        try:
            with native_compaction_request(bound_fence) as request:
                while True:
                    request.check_cancelled()
                    if keepalive: request.on_event()
                    time.sleep(.02)
        except InterruptedError:
            return original, 'fixture'
        finally:
            done.set()
    returned = run_compress_context_with_progress_timeout(
        worker=worker, messages=original, system_prompt_fallback='fixture',
        idle_timeout_seconds=.2, total_ceiling_seconds=.5, fence=fence,
        on_timeout_cause=lambda *args: causes.append(args), stall_fallback=False,
    )
    assert done.wait(2)
    assert returned[0] is original and fence.is_cancelled
    assert causes == [(keepalive, keepalive)]


def _agent_with_stream(stream):
    aborts = []
    client = NS(responses=NS(create=lambda **kwargs: stream))
    agent = NS(
        model='gpt-6-astra', provider='openai-codex', api_mode='codex_responses',
        base_url='https://chatgpt.com/backend-api/codex', session_id='fixture',
        _interrupt_requested=False, _codex_stream_last_event_ts=None,
        _is_codex_backend=lambda: True, _compute_non_stream_stale_timeout=lambda request: 10,
        _touch_activity=lambda *args,**kwargs: None,
        _fire_stream_delta=lambda *args: None, _fire_reasoning_delta=lambda *args: None,
        _fire_streamed_codex_commentary=lambda *args: None,
        _client_log_context=lambda: 'fixture', _buffer_status=lambda *args: None,
        _emit_wait_notice=lambda *args: None,
        _create_request_openai_client=lambda *args,**kwargs: client,
        _close_request_openai_client=lambda *args,**kwargs: None,
    )
    def abort(client, **kwargs):
        aborts.append((client, kwargs.get('reason')))
        stream.release.set()
    agent._abort_request_openai_client = abort
    agent._run_codex_stream=lambda kwargs,client=None,on_first_delta=None: run_codex_stream(agent, kwargs, client, on_first_delta)
    return agent, client, aborts


class _HeldStream:
    def __init__(self):
        self.ready=threading.Event(); self.release=threading.Event(); self.closed=threading.Event()
    def __iter__(self):
        yield {'type':'response.created','response':{'id':'fixture'}}
        self.ready.set()
        assert self.release.wait(3)
        yield {'type':'response.completed','response':{'id':'fixture','status':'completed','output':[]}}
    def close(self): self.closed.set()


def _start_request(agent, fence):
    outcome={}; done=threading.Event()
    def call():
        try:
            with native_compaction_request(fence) as request:
                outcome['request']=request
                outcome['response']=interruptible_api_call(agent, {'model':'gpt-5.6-luna','input':[]})
        except BaseException as exc: outcome['error']=exc
        finally: done.set()
    thread=threading.Thread(target=call)
    thread.start()
    return thread, done, outcome


def test_next_request_timestamp_reset_does_not_trigger_native_ttfb(monkeypatch):
    monkeypatch.setenv('HERMES_CODEX_TTFB_TIMEOUT_SECONDS', '.1')
    monkeypatch.setenv('HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS', '5')
    stream=_HeldStream(); agent, client, aborts=_agent_with_stream(stream)
    thread, done, outcome=_start_request(agent, CompressionCommitFence())
    try:
        assert stream.ready.wait(2)
        assert outcome['request'].last_event_ts is not None
        # Exact shared-agent reset performed when the next ordinary call starts.
        agent._codex_stream_last_event_ts=None
        assert not done.wait(.4)
        assert not aborts
        stream.release.set()
        assert done.wait(2)
        assert 'error' not in outcome
    finally:
        stream.release.set(); thread.join(3)
    assert not thread.is_alive()


def test_host_cancellation_aborts_only_native_client(monkeypatch):
    monkeypatch.setenv('HERMES_CODEX_TTFB_TIMEOUT_SECONDS', '5')
    stream=_HeldStream(); agent, client, aborts=_agent_with_stream(stream)
    fence=CompressionCommitFence()
    thread, done, outcome=_start_request(agent, fence)
    try:
        assert stream.ready.wait(2)
        newer=CompressionCommitFence()
        agent._active_compression_commit_fence=newer
        assert fence.cancel_before_commit()
        assert done.wait(2)
        assert isinstance(outcome.get('error'), InterruptedError)
        assert aborts == [(client, 'native_compression_cancel')]
        assert not newer.progress_observed and not newer.is_cancelled
        assert stream.closed.wait(2)
    finally:
        stream.release.set(); thread.join(3)
    assert not thread.is_alive()


def test_ordinary_codex_stream_retains_request_watchdog_marker():
    from agent.codex_runtime import _codex_watchdog_state_var

    stream=_HeldStream(); stream.release.set()
    agent, client, aborts=_agent_with_stream(stream)
    assert current_native_compaction_request() is None
    # v0.21.1 deliberately moved ordinary watchdogs off shared agent fields.
    state=NS(token=None,lock=threading.Lock(),last_event_ts=None,
             last_progress_ts=None,retry_started_ts=None)
    token=_codex_watchdog_state_var.set(state)
    try:
        run_codex_stream(agent, {'model':'gpt-6-astra','input':[]}, client)
        assert state.last_event_ts is not None
    finally:
        _codex_watchdog_state_var.reset(token)
    assert not aborts
