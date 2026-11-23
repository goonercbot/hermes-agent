"""The standalone sender must distinguish accepted attachments from text fallback."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import BasePlatformAdapter
from plugins.platforms.telegram import standalone


class TimedOut(Exception):
    """A request whose Telegram response did not arrive."""


def document_message(message_id: int | None = 42, chat_id: int = 123):
    return SimpleNamespace(message_id=message_id, chat=SimpleNamespace(id=chat_id),
                           effective_attachment=SimpleNamespace(file_id="document-id"))


@pytest.fixture
def bot(monkeypatch):
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    monkeypatch.setitem(sys.modules, "telegram.constants", SimpleNamespace(ParseMode=parse_mode))
    monkeypatch.setattr(standalone, "_telegram_mdv2", lambda text, **_: text)
    fixture = SimpleNamespace(
        send_document=AsyncMock(return_value=document_message()),
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=17)),
    )
    monkeypatch.setattr(standalone, '_telegram_bot', lambda *_: fixture)
    return fixture


def send(paths):
    request = '\n'.join(['Tennis scorecard test', *(f'MEDIA:{path}' for path in paths)])
    media, text = BasePlatformAdapter.extract_media(request)
    assert len(media) == len(paths)
    return asyncio.run(standalone._send_telegram('fixture-token', '123', text, media_files=media))


def test_html_attachment_receipt_uses_the_document_message_id(tmp_path, bot):
    report = tmp_path / 'scorecard.html'
    report.write_text('<p>Test only</p>', encoding='utf-8')

    result = send([report])

    bot.send_document.assert_awaited_once()
    assert result['success'] is True
    assert result['message_id'] == '42'
    assert result['media_delivered'] is True


@pytest.mark.parametrize('case', ['text_only', 'missing', 'refused', 'partial', 'no_document_id',
                                  'zero_id', 'wrong_chat', 'no_attachment', 'timed_out'])
def test_receipt_does_not_claim_unconfirmed_attachments(tmp_path, bot, case):
    report = tmp_path / 'scorecard.html'
    if case != 'missing':
        report.write_text('<p>Test only</p>', encoding='utf-8')
    if case == 'refused':
        bot.send_document.side_effect = ValueError('Controlled upload refusal')
    if case == 'no_document_id':
        bot.send_document.return_value = document_message(message_id=None)
    if case == 'zero_id':
        bot.send_document.return_value = document_message(message_id=0)
    if case == 'wrong_chat':
        bot.send_document.return_value = document_message(chat_id=456)
    if case == 'no_attachment':
        bot.send_document.return_value = SimpleNamespace(message_id=42, chat=SimpleNamespace(id=123))
    if case == 'timed_out':
        bot.send_document.side_effect = TimedOut('Upload may have reached Telegram')
    paths = [] if case == 'text_only' else [report]
    if case == 'partial':
        paths.append(tmp_path / 'missing.html')

    result = send(paths)

    assert result['success'] is True  # The heading can be sent independently.
    assert result.get('media_delivered') is not True
    assert bot.send_document.await_count == (0 if case in ('missing', 'text_only') else 1)


def test_receipt_confirms_all_individually_acknowledged_documents(tmp_path, bot):
    paths = [tmp_path / name for name in ('summary.html', 'details.html')]
    for report in paths:
        report.write_text('<p>Test only</p>', encoding='utf-8')
    bot.send_document.side_effect = [document_message(41), document_message(42)]

    result = send(paths)

    assert bot.send_document.await_count == 2
    assert result['message_id'] == '42'
    assert result['media_delivered'] is True
