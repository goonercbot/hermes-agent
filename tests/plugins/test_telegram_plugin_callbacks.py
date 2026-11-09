"""Behavior of the plugin callback registrar through Telegram's native dispatcher."""
import asyncio

import pytest
from telegram import CallbackQuery, Update, User
from telegram.ext import Application, CallbackQueryHandler

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.update_admission import TelegramApplication


def test_callback_dispatch_and_native_cleanup(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="test", version="1", description="test"), manager)
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "1")
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter._authorization_check = None
    adapter._update_admission = None
    adapter._update_receipt_dir = tmp_path / "receipts"
    adapter._update_receipts_loaded = set()
    adapter._seen_update_ids = {}
    adapter._inflight_update_ids = {}
    adapter._update_receipts_dirty = set()
    adapter._update_receipt_flush = None
    adapter._updates_dispatched_total = 0
    calls = []
    import sys
    import types
    state = types.ModuleType("callback_generation_probe")
    state.generation = 0
    state.calls = calls
    monkeypatch.setitem(sys.modules, state.__name__, state)
    plugin = home / "plugins" / "generation_probe"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text('name: generation_probe\nversion: "1"\ndescription: test\n')
    (plugin / "__init__.py").write_text(
        'import callback_generation_probe as state\n'
        'def register(ctx):\n'
        '    state.generation += 1\n'
        '    generation = state.generation\n'
        '    ctx.register_telegram_callback_handler("x:", lambda **kw: state.calls.append(generation))\n'
    )
    (home / "config.yaml").write_text('plugins:\n  enabled: [generation_probe]\n')
    monkeypatch.setattr("hermes_cli.plugins.get_bundled_plugins_dir", lambda: tmp_path / "empty")
    monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [])

    async def answer(self, *args, **kwargs):
        calls.append("denied")
    monkeypatch.setattr(CallbackQuery, "answer", answer)
    async def edit_query(self, *args, **kwargs):
        calls.append("edited")
    monkeypatch.setattr(CallbackQuery, "edit_message_text", edit_query)
    import tools.approval
    import tools.slash_confirm
    monkeypatch.setattr(tools.approval, "resolve_gateway_approval_card",
                        lambda session, choice, request: calls.append(("approval", session, choice, request)) or 1)
    async def resolve_command(session, confirm, choice):
        calls.append(("confirm", session, confirm, choice))
    monkeypatch.setattr(tools.slash_confirm, "resolve", resolve_command)
    adapter.resume_typing_for_chat = lambda *args: None
    adapter._approval_state = {123: "session"}
    adapter._approval_request_ids = {123: "request-123"}
    adapter._slash_confirm_state = {"abc": "session"}

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True, extra={})}
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.pairing_store = None
    adapter._message_handler = runner._handle_message
    sources = []
    real_policy = runner._is_user_authorized
    def record_policy(source):
        sources.append(source)
        return real_policy(source)
    monkeypatch.setattr(runner, "_is_user_authorized", record_policy)

    def app():
        application = Application.builder().token("123456:synthetic").application_class(
            TelegramApplication, {"adapter": adapter}).build()
        application.bot._bot_user = User(id=123456, first_name="bot", is_bot=True)
        application._initialized = True
        return application

    application = app()
    handle = ctx.register_telegram_callback_handler("x:", lambda **kw: calls.append("first"))
    ctx.register_telegram_callback_handler("x:more", lambda **kw: calls.append("second"))
    adapter._wire_plugin_handlers(application)
    adapter._wire_plugin_handlers(application)
    async def core(update, context):
        calls.append("core")
        # Core currently expects text (or missing data), not arbitrary PTB objects.
        if update.callback_query.data is None or isinstance(update.callback_query.data, str):
            await adapter._handle_callback_query(update, context)
    application.add_handler(CallbackQueryHandler(core))

    def update(uid, actor, data, *, chat_id=1, chat_type="private", thread=None, name="Example"):
        message = {"message_id": 1, "date": 0, "chat": {"id": chat_id, "type": chat_type}, "text": "test"}
        if thread is not None:
            message["message_thread_id"] = thread
        return Update.de_json({"update_id": uid, "callback_query": {"id": str(uid),
            "from": {"id": actor, "is_bot": False, "first_name": name},
            "chat_instance": "synthetic", "data": data, "message": message}}, application.bot)

    async def exercise():
        await application.process_update(update(1, 2, "x:more"))
        assert calls == ["denied"]
        assert sources[-1].user_id == "2"
        calls.clear()
        await application.process_update(update(2, 1, "x:more"))
        assert sources[-1].user_id == "1"
        assert calls == ["first"]
        calls.clear()
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS")
        monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", "-100")
        await application.process_update(update(10, 7, "x:more", chat_id=-100, chat_type="supergroup", thread=42, name="Alice"))
        assert calls == ["first"]
        source = sources[-1]
        assert (source.chat_id, source.chat_type, source.thread_id, source.user_name) == ("-100", "forum", "42", "Alice")
        calls.clear()
        await application.process_update(update(11, 7, "x:more", chat_id=-101, chat_type="supergroup", thread=43))
        assert calls == ["denied"]
        calls.clear()
        monkeypatch.delenv("TELEGRAM_GROUP_ALLOWED_CHATS")
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "Alice")
        await application.process_update(update(18, 7, "x:more", name="Alice"))
        assert calls == ["denied"]  # Telegram policy does not grant by display name.
        calls.clear()
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS")
        await application.process_update(update(12, 1, "x:more"))
        assert calls == ["denied"]
        calls.clear()
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
        await application.process_update(update(13, None, "x:more"))
        assert calls == ["denied"]
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "1")
        calls.clear()
        await application.process_update(update(19, 1, "ea:once:123"))
        assert ("approval", "session", "once", "request-123") in calls
        assert "edited" in calls
        calls.clear()
        await application.process_update(update(20, 1, "sc:once:abc"))
        assert ("confirm", "session", "abc", "once") in calls
        assert "edited" in calls
        calls.clear()
        handle.dispose()
        await application.process_update(update(3, 1, "x:more"))
        assert calls == ["second"]
        calls.clear()
        for uid, data in ((5, None), (6, ""), (7, 123), (8, "unmatched")):
            await application.process_update(update(uid, 1, data))
            assert calls == ["core"]
            calls.clear()
        manager.unload()
        calls.clear()
        await application.process_update(update(4, 1, "x:more"))
        assert calls == ["core"]
        calls.clear()
        ctx.register_telegram_callback_handler("x:", lambda **kw: calls.append("new-generation"))
        adapter._wire_plugin_handlers(application)
        await application.process_update(update(9, 1, "x:more"))
        assert calls == ["new-generation"]
        calls.clear()
        manager.discover_and_load(force=True)
        assert manager._plugins["generation_probe"].enabled
        first_generation = state.generation
        await application.process_update(update(14, 1, "x:more"))
        assert calls == [first_generation]
        calls.clear()
        manager.discover_and_load(force=True)
        assert state.generation > first_generation
        await application.process_update(update(15, 1, "x:more"))
        assert calls == [state.generation]
        calls.clear()
        reconnected = app()
        adapter._wire_plugin_handlers(reconnected)
        adapter._wire_plugin_handlers(reconnected)
        assert len(reconnected.handlers[0]) == 1
        await reconnected.process_update(update(16, 1, "x:more"))
        assert calls == [state.generation]
        calls.clear()
        manager.unload()
        late_app = app()
        core_handler = CallbackQueryHandler(core)
        late_app.add_handler(core_handler)
        adapter._core_handler_ids = {id(core_handler)}
        adapter._wire_plugin_handlers(late_app)
        ctx.register_telegram_callback_handler("x:", lambda **kw: calls.append("late"))
        adapter._wire_plugin_handlers(late_app)
        adapter._wire_plugin_handlers(late_app)
        assert len(late_app.handlers[0]) == 2
        assert late_app.handlers[0][1] is core_handler
        await late_app.process_update(update(17, 1, "x:more"))
        assert calls == ["late"]
        if adapter._update_receipt_flush:
            await adapter._update_receipt_flush
    asyncio.run(exercise())


def test_registrar_validation():
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="test", version="1", description="test"), manager)
    for prefix, callback in (("", lambda **kw: None), (None, lambda **kw: None), ("x:", None)):
        with pytest.raises(ValueError):
            ctx.register_telegram_callback_handler(prefix, callback)
    assert manager.get_telegram_callback_handlers() == []


def test_registration_failure_before_tracking_leaves_no_entry(monkeypatch):
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="test", version="1", description="test"), manager)
    def fail(*args, **kwargs):
        raise RuntimeError("ledger unavailable")
    monkeypatch.setattr(manager, "_track_registration", fail)
    with pytest.raises(RuntimeError, match="ledger unavailable"):
        ctx.register_telegram_callback_handler("x:", lambda **kw: None)
    assert manager.get_telegram_callback_handlers() == []
    assert manager.get_platform_handler_factories("telegram") == []


def test_failed_plugin_rolls_back_callback(tmp_path, monkeypatch):
    home = tmp_path / "home"
    plugin = home / "plugins" / "failure_probe"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text('name: failure_probe\nversion: "1"\ndescription: test\n')
    (plugin / "__init__.py").write_text(
        'def register(ctx):\n'
        '    ctx.register_telegram_callback_handler("failed:", lambda **kw: None)\n'
        '    raise RuntimeError("registration failed")\n'
    )
    (home / "config.yaml").write_text('plugins:\n  enabled: [failure_probe]\n')
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_cli.plugins.get_bundled_plugins_dir", lambda: tmp_path / "empty")
    monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [])
    manager = PluginManager()
    manager.discover_and_load()
    assert not manager._plugins["failure_probe"].enabled
    assert manager.get_telegram_callback_handlers() == []
    # The manager-owned factory can remain, but its matcher consults the empty registry.
    app = Application.builder().token("123456:synthetic").build()
    for factory, _ in manager.get_platform_handler_factories("telegram"):
        factory(app, None)
    query = Update.de_json({"update_id": 1, "callback_query": {"id": "1",
        "from": {"id": 1, "is_bot": False, "first_name": "Example"},
        "chat_instance": "x", "data": "failed:x"}}, app.bot)
    assert len(app.handlers.get(0, [])) == 1
    assert not app.handlers[0][0].check_update(query)
    seen = []
    async def core(update, context):
        seen.append("core")
    app.add_handler(CallbackQueryHandler(core))
    app.bot._bot_user = User(id=123456, first_name="bot", is_bot=True)
    app._initialized = True
    asyncio.run(app.process_update(query))
    assert seen == ["core"]
    ctx = PluginContext(PluginManifest(name="later", version="1", description="test"), manager)
    ctx.register_telegram_callback_handler("later:", lambda **kw: None)
    assert len(manager.get_platform_handler_factories("telegram")) == 1


def test_host_isolation_audit_refuses_callback(tmp_path):
    from hermes_cli.plugin_isolation_audit import audit_plugin_dir
    (tmp_path / "__init__.py").write_text(
        'def register(ctx):\n    ctx.register_telegram_callback_handler("x:", callback)\n'
    )
    report = audit_plugin_dir(tmp_path, manifest={})
    assert not report.host_ready
    assert any("register_telegram_callback_handler" in reason for reason in report.reasons)
