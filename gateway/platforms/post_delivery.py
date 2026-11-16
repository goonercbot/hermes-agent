"""Generation-owned delivery callbacks and ephemeral complete-body receipts."""



def complete_body(result):
    """Unknown, clipped or partial receipts cannot authorize deleting evidence."""
    raw = getattr(result, 'raw_response', None)
    return (getattr(result, 'success', False)
            and getattr(result, 'full_payload_delivered', None) is True
            and not (isinstance(raw, dict) and (raw.get('partial_overflow') or raw.get('skipped'))))


def record_final_ids(guard, result):
    # Even a failed or partial send can have created or edited messages.  Their
    # identity protects them from deletion; freshness is a separate proof.
    from gateway.commentary_delivery import message_ids
    if not hasattr(guard, '_hermes_final_delivery_ids'):
        guard._hermes_final_delivery_ids = set()
    guard._hermes_final_delivery_ids.update(message_ids(result))
    if not hasattr(guard, '_hermes_final_fresh_message_ids'):
        guard._hermes_final_fresh_message_ids = set()
    guard._hermes_final_fresh_message_ids.update(
        str(mid) for mid in (getattr(result, 'fresh_message_ids', None) or ()) if mid is not None)


class FinalDeliveryReceiptMixin:
    def _confirm_final_delivery(self, session_key, generation, complete):
        if not hasattr(self, '_confirmed_final_deliveries'):
            self._confirmed_final_deliveries = {}
        self._confirmed_final_deliveries[session_key, generation] = bool(complete)

    def _has_confirmed_final_delivery(self, session_key, generation):
        return getattr(self, '_confirmed_final_deliveries', {}).get((session_key, generation), False)
