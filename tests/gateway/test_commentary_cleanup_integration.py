"""Interim-only consumer receipts participate in generation-bound cleanup."""
import asyncio

import pytest

from gateway.config import Platform, StreamingConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from tests.gateway.test_run_cleanup_progress import CleanupCaptureAdapter, _install_fakes, _make_runner


class CommentaryAgent:
    commentary = 'working on it'
    final = 'the answer'

    def __init__(self, **kwargs):
        self.interim_assistant_callback = None
        self.tools = []

    def run_conversation(self, *args, **kwargs):
        assert self.interim_assistant_callback is not None
        self.interim_assistant_callback(self.commentary, already_streamed=False)
        return {'final_response': self.final, 'messages': [], 'api_calls': 1}


class SplitAdapter(CleanupCaptureAdapter):
    def __init__(self, outcome='complete', delete_raises=False):
        super().__init__()
        self.outcome = outcome
        self.delete_raises = delete_raises
        self.documents = []
        self.final_count = 0

    async def send_document(self, chat_id, file_path, caption=None, file_name=None, reply_to=None, metadata=None, **kwargs):
        self.documents.append(file_path)
        return SendResult(success=True, message_id='attachment')

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        if metadata and metadata.get('_interim_send'):
            self.sent.append({'content': content, 'message_id': 'comment-1'})
            return SendResult(success=True, message_id='comment-1',
                              raw_response={'message_ids': ['comment-1', 'comment-2']},
                              continuation_message_ids=['comment-3'])
        self.final_count += 1
        final_id = f'final-{self.final_count}'
        self.sent.append({'content': content, 'message_id': final_id})
        if self.outcome == 'failure':
            return SendResult(success=False, error='refused')
        return SendResult(success=True, message_id=final_id,
                          full_payload_delivered=self.outcome == 'complete')

    async def delete_message(self, chat_id, message_id):
        self.deleted.append({'chat_id': chat_id, 'message_id': str(message_id)})
        if self.delete_raises and message_id == 'comment-1':
            raise RuntimeError('deletion refused')
        return True


@pytest.mark.asyncio
@pytest.mark.parametrize('attachment', [False, True])
@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('outcome,cleanup,identical,stale,delete_raises,expected', [
    ('complete', True, False, False, False, ['comment-1', 'comment-2', 'comment-3']),
    ('failure', True, False, False, False, []),
    ('partial', True, False, False, False, []),
    ('complete', False, False, False, False, []),
    ('complete', True, True, False, False, []),
    ('complete', True, False, True, False, []),
    ('complete', True, False, False, True, ['comment-1', 'comment-2', 'comment-3']),
])
async def test_consumer_receipts_through_background_delivery(
        monkeypatch, tmp_path, outcome, cleanup, identical, stale, delete_raises, expected, attachment, streaming):
    adapter = SplitAdapter(outcome, delete_raises)
    runner = _make_runner(adapter)
    runner.config.streaming = StreamingConfig(enabled=streaming)
    gateway_run = _install_fakes(monkeypatch, CommentaryAgent, cleanup_on=cleanup)
    monkeypatch.setattr(gateway_run, '_hermes_home', tmp_path)
    monkeypatch.setattr(gateway_run, '_load_gateway_config', lambda: {
        'display': {'tool_progress': 'off', 'interim_assistant_messages': True,
                    'platforms': {'telegram': {'streaming': streaming, 'cleanup_progress': cleanup}}}})
    monkeypatch.setattr(CommentaryAgent, 'commentary', 'the answer' if identical else 'working on it')
    document = tmp_path / 'report.pdf'
    if attachment:
        document.write_bytes(b'test document')
        monkeypatch.setattr(CommentaryAgent, 'final', f'the answer\nMEDIA:{document}')
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='4242', chat_type='dm')
    event = MessageEvent(text='question', message_type=MessageType.TEXT, source=source, message_id='incoming')
    key, generation = 'commentary-cleanup', 1
    interrupt = asyncio.Event()
    interrupt._hermes_run_generation = generation
    adapter._active_sessions[key] = interrupt
    runner._session_run_generation[key] = generation

    async def handler(_event):
        result = await runner._run_agent(message='question', context_prompt='', history=[], source=source,
                                         session_id='test-commentary', session_key=key, run_generation=generation)
        if stale:
            runner._session_run_generation[key] = generation + 1
        return result['final_response']

    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, key)
    assert adapter.sent[0]['content'] == CommentaryAgent.commentary
    assert [row['message_id'] for row in adapter.deleted] == expected
    assert 'final-1' not in [row['message_id'] for row in adapter.deleted]
    assert 'attachment' not in [row['message_id'] for row in adapter.deleted]
    if attachment:
        assert adapter.documents == [str(document)]


@pytest.mark.asyncio
async def test_later_cleanup_preserves_an_earlier_final(monkeypatch, tmp_path):
    adapter = SplitAdapter()
    runner = _make_runner(adapter)
    runner.config.streaming = StreamingConfig(enabled=False)
    gateway_run = _install_fakes(monkeypatch, CommentaryAgent, cleanup_on=True)
    monkeypatch.setattr(gateway_run, '_hermes_home', tmp_path)
    monkeypatch.setattr(gateway_run, '_load_gateway_config', lambda: {
        'display': {'tool_progress': 'off', 'interim_assistant_messages': True,
                    'platforms': {'telegram': {'streaming': False, 'cleanup_progress': True}}}})
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='4242', chat_type='dm')
    event = MessageEvent(text='question', message_type=MessageType.TEXT, source=source, message_id='incoming')
    key = 'commentary-two-turns'

    async def old_handler(_event):
        return 'earlier final'

    adapter.set_message_handler(old_handler)
    await adapter._process_message_background(event, key)
    assert adapter.sent == [{'content': 'earlier final', 'message_id': 'final-1'}]

    generation = 2
    guard = asyncio.Event()
    guard._hermes_run_generation = generation
    adapter._active_sessions[key] = guard
    runner._session_run_generation[key] = generation

    async def new_handler(_event):
        result = await runner._run_agent(message='question', context_prompt='', history=[], source=source,
                                         session_id='test-two-turns', session_key=key, run_generation=generation)
        return result['final_response']

    adapter.set_message_handler(new_handler)
    await adapter._process_message_background(event, key)
    assert [row['message_id'] for row in adapter.deleted] == ['comment-1', 'comment-2', 'comment-3']
    assert [(row['message_id'], row['content']) for row in adapter.sent
            if row['message_id'].startswith('final-')] == [
        ('final-1', 'earlier final'), ('final-2', 'the answer')]
