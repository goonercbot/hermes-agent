"""Bounded exact-parent routine replay, NOT a full parent-build test."""
import pytest
from gateway import waiting_reply_delivery
from tests.gateway import _cap_free_review_parent_commentary as parent_commentary
from tests.gateway import _cap_free_review_parent_waiting as parent_waiting
from tests.gateway import test_cap_free_review_probe as probes
from tests.gateway import test_waiting_review_safety as fixtures
from tests.gateway.test_cap_free_review_probe import (
    owned,
    test_commentary_creation_before_concurrent_final_keeps_final,
    test_streamed_wait_created_before_concurrent_final_cannot_be_readmitted,
)


@pytest.fixture(autouse=True)
def exact_parent_routines(monkeypatch):
    for name in ('schedule_waiting_receipt', 'schedule_waiting_resolution'):
        monkeypatch.setattr(waiting_reply_delivery, name, getattr(parent_waiting, name))
    for module in (probes, fixtures):
        monkeypatch.setattr(module, 'schedule_cleanup', parent_commentary.schedule_cleanup)
    monkeypatch.setattr(probes, 'track_progress', parent_commentary.track_progress)
