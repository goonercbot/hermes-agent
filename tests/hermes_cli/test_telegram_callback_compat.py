"""Backward-compatible authorized Telegram callback-prefix registration."""

import asyncio
import re
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


class _FakeCallbackQueryHandler:
    def __init__(self, callback, pattern):
        self.callback = callback
        self.pattern = re.compile(pattern)


def _context():
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="callback-fixture"), manager)
    return manager, context


def _wired_handler(manager, adapter, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "telegram.ext",
        SimpleNamespace(CallbackQueryHandler=_FakeCallbackQueryHandler),
    )
    application = SimpleNamespace(handlers=[])
    application.add_handler = application.handlers.append
    factories = manager.get_platform_handler_factories("telegram")
    assert len(factories) == 1
    factory, plugin_name = factories[0]
    assert plugin_name == "callback-fixture"
    factory(application, adapter)
    assert len(application.handlers) == 1
    handler = application.handlers[0]
    assert isinstance(handler, _FakeCallbackQueryHandler)
    return handler


def _update(data="tx:approve"):
    query = SimpleNamespace(
        data=data,
        answer=AsyncMock(),
        from_user=SimpleNamespace(id=123, first_name="AJ"),
        message=SimpleNamespace(
            chat_id=-1001,
            message_thread_id=42,
            chat=SimpleNamespace(type="supergroup"),
        ),
    )
    return SimpleNamespace(callback_query=query), query


def test_register_callback_prefix_wires_authorized_handler(monkeypatch):
    manager, context = _context()
    callback = AsyncMock()
    context.register_telegram_callback_handler("tx:", callback)
    adapter = SimpleNamespace(_is_callback_user_authorized=lambda *_a, **_kw: True)

    handler = _wired_handler(manager, adapter, monkeypatch)
    update, query = _update()
    asyncio.run(handler.callback(update, None))

    callback.assert_awaited_once_with(
        update=update,
        query=query,
        adapter=adapter,
    )


def test_callback_prefix_denies_unauthorized_user(monkeypatch):
    manager, context = _context()
    callback = AsyncMock()
    context.register_telegram_callback_handler("tx:", callback)
    adapter = SimpleNamespace(_is_callback_user_authorized=lambda *_a, **_kw: False)

    handler = _wired_handler(manager, adapter, monkeypatch)
    update, query = _update()
    asyncio.run(handler.callback(update, None))

    callback.assert_not_awaited()
    query.answer.assert_awaited_once()
    assert "not authorized" in query.answer.await_args.kwargs["text"].lower()
    assert query.answer.await_args.kwargs["show_alert"] is True


def test_callback_prefix_is_regex_escaped(monkeypatch):
    manager, context = _context()
    context.register_telegram_callback_handler("tx.+:", AsyncMock())
    adapter = SimpleNamespace(_is_callback_user_authorized=lambda *_a, **_kw: True)

    handler = _wired_handler(manager, adapter, monkeypatch)

    assert handler.pattern.pattern == r"^tx\.\+:"


@pytest.mark.parametrize("prefix", ["", None, 123])
def test_callback_prefix_rejects_invalid_prefix(prefix):
    _, context = _context()
    with pytest.raises(ValueError, match="empty prefix"):
        context.register_telegram_callback_handler(prefix, AsyncMock())


def test_callback_prefix_rejects_non_callable_callback():
    _, context = _context()
    with pytest.raises(ValueError, match="non-callable"):
        context.register_telegram_callback_handler("tx:", "not-callable")
