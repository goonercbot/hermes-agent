"""Deadline rollback must revoke callbacks regardless of publication order."""
import asyncio
import sys
import threading
import types

import pytest
from telegram import Update, User
from telegram.ext import Application, CallbackQueryHandler

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.update_admission import TelegramApplication


@pytest.mark.parametrize("pause_at", ["before_tracking", "after_tracking", "after_publication"])
def test_timeout_revokes_callback_before_and_after_publication(tmp_path, monkeypatch, pause_at):
    home = tmp_path / "home"
    plugin = home / "plugins" / "timeout_callback"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text('name: timeout_callback\nversion: "1"\ndescription: test\n')
    state = types.ModuleType("callback_timeout_test_state")
    state.calls = []
    state.registered = threading.Event()
    state.resume = threading.Event()
    state.returned = threading.Event()
    state.pause_at = pause_at
    monkeypatch.setitem(sys.modules, state.__name__, state)
    (plugin / "__init__.py").write_text(
        'import callback_timeout_test_state as state\n'
        'def register(ctx):\n'
        '    try:\n'
        '        ctx.register_telegram_callback_handler("zombie:", lambda **kw: state.calls.append("zombie"))\n'
        '        if state.pause_at == "after_publication":\n'
        '            state.registered.set()\n'
        '            assert state.resume.wait(10)\n'
        '    finally:\n'
        '        state.returned.set()\n'
    )
    (home / "config.yaml").write_text('plugins:\n  enabled: [timeout_callback]\n')
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "1")
    monkeypatch.setattr("hermes_cli.plugins.get_bundled_plugins_dir", lambda: tmp_path / "empty")
    monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [])
    monkeypatch.setattr("hermes_cli.plugins_loader._resolve_plugin_load_timeout", lambda: 0.3)
    manager = PluginManager()
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    ctx = PluginContext(PluginManifest(name="existing", version="1", description="test"), manager)
    ctx.register_telegram_callback_handler("existing:", lambda **kw: state.calls.append("existing"))
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._authorization_check = None
    adapter._update_admission = None
    adapter._update_receipt_dir = home / "receipts"
    adapter._update_receipts_loaded = set()
    adapter._seen_update_ids = {}
    adapter._inflight_update_ids = {}
    adapter._update_receipts_dirty = set()
    adapter._update_receipt_flush = None
    adapter._updates_dispatched_total = 0
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.config.platforms = {Platform.TELEGRAM: adapter.config}
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.pairing_store = None
    adapter._message_handler = runner._handle_message
    app = Application.builder().token("123456:synthetic").application_class(
        TelegramApplication, {"adapter": adapter}).build()
    app.bot._bot_user = User(id=123456, first_name="bot", is_bot=True)
    app._initialized = True
    adapter._wire_plugin_handlers(app)
    async def core(update, context):
        state.calls.append("core")
    app.add_handler(CallbackQueryHandler(core))
    handles = []
    real_track = manager._track_registration
    def scheduled_track(manifest, kind, key, release, **kwargs):
        if kind == "telegram_callback_handler" and key == "zombie:" and pause_at == "before_tracking":
            state.registered.set()
            assert state.resume.wait(10)
        handle = real_track(manifest, kind, key, release, **kwargs)
        if kind == "telegram_callback_handler" and key == "zombie:":
            handles.append(handle)
            if pause_at == "after_tracking":
                state.registered.set()
                assert state.resume.wait(10)
        return handle
    monkeypatch.setattr(manager, "_track_registration", scheduled_track)
    try:
        manager.discover_and_load()
        assert state.registered.is_set()
        assert "timed out" in manager._plugins["timeout_callback"].error
        if handles:
            assert not handles[0].active
        assert all(entry[0] != "zombie:" for entry in manager.get_telegram_callback_handlers())
        # Sweep while the abandoned worker is still paused, then let it return.
        manager.unload()
    finally:
        state.resume.set()
    assert state.returned.wait(10), "deadline worker failed to return"
    assert not handles[0].active
    assert manager.get_telegram_callback_handlers() == []
    assert manager.get_platform_handler_factories("telegram") == []
    async def dispatch():
        update = Update.de_json({"update_id": 12, "callback_query": {"id": "12",
            "from": {"id": 1, "is_bot": False, "first_name": "Example"},
            "chat_instance": "synthetic", "data": "zombie:x"}}, app.bot)
        await app.process_update(update)
        if adapter._update_receipt_flush:
            await adapter._update_receipt_flush
    asyncio.run(dispatch())
    assert state.calls == ["core"]
