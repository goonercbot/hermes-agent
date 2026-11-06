"""Real stream deltas classified as interim commentary are cleaned only safely."""
import asyncio
import threading

import pytest

from gateway.config import Platform, StreamingConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from tests.gateway.test_commentary_cleanup_integration import CommentaryAgent
from tests.gateway.test_run_cleanup_progress import CleanupCaptureAdapter, _install_fakes, _make_runner


@pytest.mark.asyncio
@pytest.mark.parametrize('scenario', [
    'complete', 'partial', 'disabled', 'stale', 'matching', 'matching_attachment',
    'attachment', 'reused_final', 'reused_attachment', 'unclassified', 'mismatched',
    'prior_boundary', 'streamed_final', 'recovered_final',
])
async def test_streamed_commentary_receipts_are_cleaned(monkeypatch, tmp_path, scenario):
    seen = threading.Event()
    attachment = scenario in ('matching_attachment', 'attachment', 'reused_attachment')
    document = tmp_path / 'report.pdf'
    if attachment:
        document.write_bytes(b'test document')
    final = 'working on it' if scenario.startswith('matching') else 'the answer'
    if attachment:
        final += f'\nMEDIA:{document}'

    class Adapter(CleanupCaptureAdapter):
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            if content.startswith('working on it'):
                self.sent.append({'content': content, 'message_id': 'comment-1'})
                seen.set()
                return SendResult(success=True, message_id='comment-1',
                                  raw_response={'message_ids': ['comment-1', 'comment-2']},
                                  continuation_message_ids=('comment-3',), full_payload_delivered=True)
            final_id = 'comment-2' if scenario == 'reused_final' else 'final'
            self.sent.append({'content': content, 'message_id': final_id})
            return SendResult(success=True, message_id=final_id, full_payload_delivered=scenario != 'partial')

        async def send_document(self, chat_id, file_path, **kwargs):
            self.documents.append(file_path)
            return SendResult(success=True, message_id='comment-3' if scenario == 'reused_attachment' else 'attachment')

        async def edit_message(self, chat_id, message_id, content, **kwargs):
            self.edits.append({'content': content, 'message_id': message_id})
            return SendResult(success=True, message_id=message_id, full_payload_delivered=True)

    class Agent(CommentaryAgent):
        def run_conversation(self, *args, **kwargs):
            self.stream_delta_callback('working on it')
            assert seen.wait(5), 'streamed commentary was not delivered'
            if scenario == 'prior_boundary':
                self.stream_delta_callback(None)
            if scenario != 'unclassified':
                self.interim_assistant_callback(
                    'other text' if scenario == 'mismatched' else 'working on it', already_streamed=True)
            if scenario == 'streamed_final':
                self.stream_delta_callback('the answer')
            if scenario == 'recovered_final':
                self.stream_delta_callback('NO_REPLY')
            return {'final_response': final, 'messages': [], 'api_calls': 1}

    adapter = Adapter()
    adapter.documents = []
    runner = _make_runner(adapter)
    runner.config.streaming = StreamingConfig(enabled=True, buffer_threshold=1)
    gateway_run = _install_fakes(monkeypatch, Agent, cleanup_on=scenario != 'disabled')
    monkeypatch.setattr(gateway_run, '_hermes_home', tmp_path)
    monkeypatch.setattr(gateway_run, '_load_gateway_config', lambda: {
        'display': {'tool_progress': 'off', 'interim_assistant_messages': True,
                    'platforms': {'telegram': {'streaming': True, 'cleanup_progress': scenario != 'disabled'}}}})
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='4242', chat_type='dm')
    event = MessageEvent(text='question', message_type=MessageType.TEXT, source=source, message_id='incoming')
    key, generation = 'actual-stream', 1
    guard = asyncio.Event()
    guard._hermes_run_generation = generation
    adapter._active_sessions[key] = guard
    runner._session_run_generation[key] = generation

    async def handler(_event):
        result = await runner._run_agent(message='question', context_prompt='', history=[], source=source,
                                         session_id='actual-stream', session_key=key, run_generation=generation)
        if scenario == 'stale':
            runner._session_run_generation[key] = generation + 1
        return result['final_response']

    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, key)
    if attachment:
        assert adapter.documents == [str(document)]
    if scenario == 'streamed_final':
        assert any('the answer' in row['content'] and row['message_id'] == 'final' for row in adapter.sent)
    deleted = sorted(row['message_id'] for row in adapter.deleted)
    expected = {
        'partial': [], 'disabled': [], 'stale': [], 'matching': [], 'matching_attachment': [],
        'unclassified': [], 'mismatched': [], 'reused_final': ['comment-1', 'comment-3'],
        'reused_attachment': ['comment-1', 'comment-2'],
    }
    if scenario == 'recovered_final':
        assert 'comment-1' not in deleted
        assert any(row['message_id'] == 'comment-1' and row['content'] == 'the answer' for row in adapter.edits)
    else:
        assert deleted == expected.get(scenario, ['comment-1', 'comment-2', 'comment-3'])
    assert 'final' not in deleted and 'attachment' not in deleted
