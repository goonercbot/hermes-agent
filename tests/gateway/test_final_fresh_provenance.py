"""Final identity and creation provenance remain distinct for cleanup consumers."""
import asyncio

from gateway.platforms.base import SendResult
from gateway.platforms.post_delivery import record_final_ids
from gateway.stream_consumer import GatewayStreamConsumer


def test_normal_guard_distinguishes_edited_and_created_partial_ids():
    guard = asyncio.Event()
    guard._hermes_final_delivery_ids = set()
    record_final_ids(guard, SendResult(success=False, message_id='edited',
                     raw_response={'delivered_message_ids': ['created']},
                     fresh_message_ids=('created',)))
    assert guard._hermes_final_delivery_ids == {'edited', 'created'}
    assert guard._hermes_final_fresh_message_ids == {'created'}


def test_stream_final_preserves_fresh_preview_but_not_reused_identity():
    consumer = GatewayStreamConsumer(object(), 'chat')
    consumer._track_preview_ids_from_result(SendResult(
        success=True, message_id='preview', fresh_message_ids=('preview',)))
    consumer._track_final_receipt(SendResult(success=True, message_id='preview'))
    consumer._track_final_receipt(SendResult(success=True, message_id='old'))
    assert consumer._final_receipt_ids == {'preview', 'old'}
    assert consumer._final_fresh_message_ids == {'preview'}
