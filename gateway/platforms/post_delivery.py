"""Generation-owned delivery callbacks and ephemeral complete-body receipts."""



def complete_body(result):
    """Unknown, clipped or partial receipts cannot authorize deleting evidence."""
    raw = getattr(result, 'raw_response', None)
    return (getattr(result, 'success', False)
            and getattr(result, 'full_payload_delivered', None) is True
            and not (isinstance(raw, dict) and (raw.get('partial_overflow') or raw.get('skipped'))))


def record_final_ids(guard, result):
    if getattr(result, 'success', False):
        from gateway.commentary_delivery import message_ids
        guard._hermes_final_delivery_ids.update(message_ids(result))


class FinalDeliveryReceiptMixin:
    def _confirm_final_delivery(self, session_key, generation, complete):
        if not hasattr(self, '_confirmed_final_deliveries'):
            self._confirmed_final_deliveries = {}
        self._confirmed_final_deliveries[session_key, generation] = bool(complete)

    def _has_confirmed_final_delivery(self, session_key, generation):
        return getattr(self, '_confirmed_final_deliveries', {}).get((session_key, generation), False)
