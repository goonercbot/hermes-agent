import asyncio
from types import SimpleNamespace

from gateway.config import Platform, PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter():
    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = object.__new__(TelegramAdapter)
    adapter.config = config
    adapter._config = config
    adapter._platform = Platform.TELEGRAM
    adapter._connected = True
    adapter._message_handler = None
    return adapter


def test_plugin_callback_prefix_is_dispatched(monkeypatch):
    seen = []

    async def handler(**kwargs):
        seen.append(kwargs)

    manager = SimpleNamespace(
        get_telegram_callback_handlers=lambda: [("ai:", handler, "agentic-tap")]
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_manager", lambda: manager
    )

    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_is_callback_user_authorized", lambda *_args, **_kwargs: True)
    query = SimpleNamespace(
        data="ai:20260716:4148:approve",
        from_user=SimpleNamespace(id=123),
        message=SimpleNamespace(chat_id=-1001, message_thread_id=None),
    )
    update = SimpleNamespace(callback_query=query)

    asyncio.run(adapter._handle_callback_query(update, None))

    assert len(seen) == 1
    assert seen[0]["query"] is query
    assert seen[0]["update"] is update
    assert seen[0]["adapter"] is adapter


def test_plugin_callback_denies_unauthorized_user(monkeypatch):
    called = False

    async def handler(**_kwargs):
        nonlocal called
        called = True

    manager = SimpleNamespace(
        get_telegram_callback_handlers=lambda: [("ai:", handler, "agentic-tap")]
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_manager", lambda: manager
    )

    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_is_callback_user_authorized", lambda *_args, **_kwargs: False)

    answers = []

    async def answer(text, **kwargs):
        answers.append((text, kwargs))

    query = SimpleNamespace(
        data="ai:20260716:4148:approve",
        from_user=SimpleNamespace(id=999),
        message=SimpleNamespace(chat_id=-1001, message_thread_id=None),
        answer=answer,
    )
    update = SimpleNamespace(callback_query=query)

    asyncio.run(adapter._handle_callback_query(update, None))

    assert called is False
    assert answers and "not authorized" in answers[0][0].lower()
