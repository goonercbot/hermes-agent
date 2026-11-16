"""Status notices respect bot-peer suppression without losing cleanup receipts."""
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run_turn_progress import GatewayTurnProgressMixin


@pytest.mark.parametrize("is_bot", [False, True])
def test_status_notice_respects_peer_gate_and_tracks_created_receipt(is_bot):
    future = Future()
    ctx = SimpleNamespace(
        source=SimpleNamespace(is_bot=is_bot, platform=Platform.TELEGRAM),
        _status_adapter=SimpleNamespace(send=Mock(return_value="scheduled-send")),
        _status_chat_id="test-chat", _cleanup_progress=True,
        _cleanup_msg_ids=[], _commentary_messages=[], _commentary_delivery_futures=[],
    )
    runner = SimpleNamespace(_ctx=ctx, _schedule=Mock(return_value=future))
    GatewayTurnProgressMixin._send_status_text(runner, "working", {}, "notice")
    if is_bot:
        ctx._status_adapter.send.assert_not_called()
        runner._schedule.assert_not_called()
        assert ctx._commentary_delivery_futures == []
        assert ctx._cleanup_msg_ids == []
    else:
        ctx._status_adapter.send.assert_called_once_with("test-chat", "working", metadata={})
        runner._schedule.assert_called_once_with("scheduled-send", "notice")
        assert ctx._commentary_delivery_futures == [future]
        future.set_result(SendResult(success=True, message_id="notice-1", fresh_message_ids=("notice-1",)))
        assert ctx._cleanup_msg_ids == ["notice-1"]
        assert ctx._commentary_messages == [("notice-1", "working")]
