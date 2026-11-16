"""Legacy and modern transports share the same bounded stream-retry contract."""
import sys

import httpx
import httpx2
import pytest

from agent.stream_transport_errors import http_transport_error_types


@pytest.mark.parametrize("transport", [httpx, httpx2])
@pytest.mark.parametrize("name", ["ReadTimeout", "ConnectTimeout", "PoolTimeout", "ConnectError", "ReadError", "RemoteProtocolError"])
def test_both_transport_generations_are_classified(transport, name):
    timeouts, connections = http_transport_error_types()
    error = getattr(transport, name)("synthetic drop")
    assert isinstance(error, timeouts) == name.endswith("Timeout")
    assert isinstance(error, connections) == (not name.endswith("Timeout"))
    assert not isinstance(ValueError("not a transport error"), timeouts + connections)


def test_legacy_error_does_not_import_optional_transport(monkeypatch):
    monkeypatch.delitem(sys.modules, "httpx2")
    timeouts, connections = http_transport_error_types()
    assert isinstance(httpx.ReadTimeout("legacy"), timeouts)
    assert isinstance(httpx.ReadError("legacy"), connections)
    assert "httpx2" not in sys.modules
