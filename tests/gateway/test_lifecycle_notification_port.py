"""Latest local lifecycle settings survive the v0.21.1 config/mixin split."""
import json
from unittest.mock import AsyncMock

import pytest
import gateway.run as gateway_run
from gateway.config import HomeChannel, Platform, PlatformConfig, load_gateway_config
from gateway.platforms.base import SendResult
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.parametrize("raw,expected", [({}, True), ({"gateway_restart_notification": False}, False), ({"gateway_restart_notification": False, "home_channel_startup_notification": True}, True), ({"extra": {"home_channel_startup_notification": False}}, False), ({"home_channel_startup_notification": True, "extra": {"home_channel_startup_notification": False}}, True)])
def test_home_notice_resolution_and_roundtrip(raw, expected):
    config = PlatformConfig.from_dict(raw)
    assert config.home_channel_startup_notification is expected
    assert PlatformConfig.from_dict(config.to_dict()).home_channel_startup_notification is expected


def test_nested_platform_config_bridge(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("platforms:\n  telegram:\n    enabled: true\n    home_channel_startup_notification: false\n")
    assert load_gateway_config().platforms[Platform.TELEGRAM].home_channel_startup_notification is False


@pytest.mark.asyncio
async def test_suppressed_home_startup_sends_nothing():
    runner, adapter = make_restart_runner()
    config = runner.config.platforms[Platform.TELEGRAM]
    config.home_channel = HomeChannel(platform=Platform.TELEGRAM, chat_id="-42", name="Home")
    config.home_channel_startup_notification = False
    adapter.send = AsyncMock()
    assert await runner._send_home_channel_startup_notifications() == set()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_direct_restart_reply_survives_home_suppression(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / ".restart_notify.json").write_text(json.dumps({"platform": "telegram", "chat_id": "42"}))
    runner, adapter = make_restart_runner()
    config = runner.config.platforms[Platform.TELEGRAM]
    config.gateway_restart_notification = True
    config.home_channel_startup_notification = False
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="restart"))
    assert await runner._send_restart_notification() == ("telegram", "42", None)
    adapter.send.assert_awaited_once()
