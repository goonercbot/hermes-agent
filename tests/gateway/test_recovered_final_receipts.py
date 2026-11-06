"""Recovered final fallback preserves every successful delivery receipt."""
import asyncio
import threading

import pytest

from gateway.config import Platform, StreamingConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from tests.gateway.test_commentary_cleanup_integration import CommentaryAgent
from tests.gateway.test_run_cleanup_progress import CleanupCaptureAdapter, _install_fakes, _make_runner


async def recovered_flow(monkeypatch, tmp_path, reuse):
    commentary_seen = threading.Event()
    final = 'the complete final answer'

    class Adapter(CleanupCaptureAdapter):
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            if content.startswith('working on it'):
                mid, raw, tail = 'comment-1', {'message_ids': ['comment-1', 'comment-2']}, ('comment-3',)
                commentary_seen.set()
            else:
                mid = 'final-tail'
                raw = {'message_ids': ['final-tail', 'comment-2' if reuse else 'final-2']}
                tail = ('comment-3' if reuse else 'final-3',)
            self.sent.append({'content': content, 'message_id': mid, 'raw': raw, 'continuation': tail})
            return SendResult(success=True, message_id=mid, raw_response=raw,
                              continuation_message_ids=tail, full_payload_delivered=True)

        async def edit_message(self, chat_id, message_id, content, **kwargs):
            self.edits.append({'content': content, 'message_id': message_id})
            if content == final:
                return SendResult(success=False, message_id=message_id, error='editing unsupported')
            return SendResult(success=True, message_id=message_id, full_payload_delivered=True)

    class Agent(CommentaryAgent):
        def run_conversation(self, *args, **kwargs):
            self.stream_delta_callback('working on it')
            assert commentary_seen.wait(5)
            self.interim_assistant_callback('working on it', already_streamed=True)
            self.stream_delta_callback('NO_REPLY')
            return {'final_response': final, 'messages': [], 'api_calls': 1}

    adapter = Adapter()
    runner = _make_runner(adapter)
    runner.config.streaming = StreamingConfig(enabled=True, buffer_threshold=1)
    gateway_run = _install_fakes(monkeypatch, Agent, cleanup_on=True)
    monkeypatch.setattr(gateway_run, '_hermes_home', tmp_path)
    monkeypatch.setattr(gateway_run, '_load_gateway_config', lambda: {
        'display': {'tool_progress': 'off', 'interim_assistant_messages': True,
                    'platforms': {'telegram': {'streaming': True, 'cleanup_progress': True}}}})
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='4242', chat_type='dm')
    event = MessageEvent(text='question', message_type=MessageType.TEXT, source=source, message_id='incoming')
    key, generation = 'recovered-fallback-recheck', 1
    guard = asyncio.Event()
    guard._hermes_run_generation = generation
    adapter._active_sessions[key] = guard
    runner._session_run_generation[key] = generation
    results = []

    async def handler(_event):
        result = await runner._run_agent(message='question', context_prompt='', history=[], source=source,
            session_id=key, session_key=key, run_generation=generation)
        results.append(result)
        return None if result.get('already_sent') else result['final_response']

    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, key)
    assert any(row['message_id'] == 'final-tail' and row['content'] == final for row in adapter.sent)
    assert any(row['message_id'] == 'comment-1' and row['content'] == final for row in adapter.edits)
    assert results[0].get('already_sent') is True
    assert results[0].get('_complete_final_body') is True
    deleted = {row['message_id'] for row in adapter.deleted}
    protected = {'comment-2', 'comment-3'} if reuse else {'final-2', 'final-3'}
    assert not (deleted & protected), 'recovered cleanup deleted final split receipts'
    assert 'final-tail' not in deleted
    assert 'comment-1' in deleted
    if not reuse:
        assert {'comment-2', 'comment-3'} <= deleted


@pytest.mark.asyncio
@pytest.mark.parametrize("reuse", [True, False], ids=["overlap", "fresh"])
async def test_recovered_fallback_final_receipts_survive(monkeypatch, tmp_path, reuse):
    await recovered_flow(monkeypatch, tmp_path, reuse)
