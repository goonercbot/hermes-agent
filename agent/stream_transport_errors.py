"""Transient stream-body exception classes from already loaded HTTP transports."""
import sys

import httpx


def http_transport_error_types() -> tuple[tuple, tuple]:
    # A raw httpx2 exception implies that transport is loaded. Do not import or
    # provision an optional SDK from another provider's error handler.
    transports = [httpx]
    modern = sys.modules.get("httpx2")
    if modern is not None:
        transports.append(modern)
    timeouts = tuple(
        getattr(transport, name) for transport in transports
        for name in ("ReadTimeout", "ConnectTimeout", "PoolTimeout")
    )
    connections = tuple(
        getattr(transport, name) for transport in transports
        for name in ("ConnectError", "ReadError", "RemoteProtocolError")
    )
    return timeouts, connections
