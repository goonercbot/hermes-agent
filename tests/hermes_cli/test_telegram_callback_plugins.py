import yaml

from hermes_cli.plugins import PluginManager


def test_plugin_registers_telegram_callback_prefix(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    plugin_dir = hermes_home / "plugins" / "telegram_callback_test"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "telegram_callback_test"}), encoding="utf-8"
    )
    (plugin_dir / "__init__.py").write_text(
        "async def handle(**kwargs):\n"
        "    return True\n\n"
        "def register(ctx):\n"
        "    ctx.register_telegram_callback_handler('ai:', handle)\n",
        encoding="utf-8",
    )
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["telegram_callback_test"]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    manager = PluginManager()
    manager.discover_and_load()

    handlers = manager.get_telegram_callback_handlers()
    assert len(handlers) == 1
    prefix, callback, plugin_name = handlers[0]
    assert prefix == "ai:"
    assert callable(callback)
    assert plugin_name == "telegram_callback_test"
