"""Gateway model notices survive cleanup and reach the originating chat for hygiene too."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.run_turn_progress import GatewayTurnProgressMixin
from gateway.run_turn_hygiene import GatewayTurnHygieneMixin
from gateway.session import Platform, SessionSource


@pytest.mark.asyncio
async def test_model_switch_is_persistent_chat_message_even_with_warnings_disabled(monkeypatch):
    sends = []
    adapter = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(success=True, message_id="notice")))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="human")
    ctx = SimpleNamespace(mute_notification_reply=False, _status_adapter=adapter,
        _run_still_current=lambda: True, source=source, _status_chat_id="chat",
        _status_thread_metadata={"thread_id": "topic"}, _cleanup_progress=True,
        user_config={"display": {"warning_notifications": False}})
    owner = SimpleNamespace(_ctx=ctx, _approval_owner=None,
        _status_live=lambda: True, _track_future_cleanup_id=Mock())
    def schedule(coro, message):
        sends.append(coro)
        return Mock()
    owner._schedule = schedule
    GatewayTurnProgressMixin._status_callback_sync(owner, "model_switch", "🔔 Compression model selected: backup")
    assert len(sends) == 1
    await sends[0]
    adapter.send.assert_awaited_once_with("chat", "🔔 Compression model selected: backup", metadata={"thread_id": "topic"})
    owner._track_future_cleanup_id.assert_not_called()


@pytest.mark.asyncio
async def test_stale_gateway_generation_does_not_emit_switch():
    owner = SimpleNamespace(_ctx=SimpleNamespace(mute_notification_reply=False), _status_live=lambda: False)
    GatewayTurnProgressMixin._status_callback_sync(owner, "model_switch", "stale")


@pytest.mark.asyncio
async def test_hygiene_switch_uses_same_chat_and_thread_not_warning_progress():
    adapter = SimpleNamespace(send=AsyncMock(), emit_warning=AsyncMock())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="origin-chat", user_id="human")
    owner = SimpleNamespace(_delivery_adapter_for=lambda s: adapter)
    await GatewayTurnHygieneMixin._hmwa_hygiene_notify(owner, source, {"thread_id": "origin-topic"},
        "🔔 Compression model selected: backup", "test switch", model_switch=True)
    adapter.send.assert_awaited_once_with("origin-chat", "🔔 Compression model selected: backup",
                                         metadata={"thread_id": "origin-topic"})
    adapter.emit_warning.assert_not_awaited()
    await GatewayTurnHygieneMixin._hmwa_hygiene_notify(owner, source, {}, "ordinary warning", "control")
    adapter.emit_warning.assert_awaited_once()


@pytest.mark.asyncio
async def test_hygiene_delivery_error_is_logged_not_raised(caplog):
    adapter = SimpleNamespace(send=AsyncMock(side_effect=RuntimeError("fixture transport failure")))
    owner = SimpleNamespace(_delivery_adapter_for=lambda s: adapter)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="origin", user_id="human")
    await GatewayTurnHygieneMixin._hmwa_hygiene_notify(owner, source, {}, "model switch", "test switch", model_switch=True)
    assert "Failed to deliver test switch" in caplog.text
