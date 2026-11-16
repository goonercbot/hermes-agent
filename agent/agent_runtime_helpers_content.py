"""Focused extraction preserving the existing caller contract."""
from __future__ import annotations


def _normalize_sentinel_encoded_content(messages: list[dict]) -> None:
    """Decode any sentinel-encoded row content in place before the alternation passes run, so an
    image-bearing turn is never merged as text (#125299). A multimodal turn can re-enter the working set
    as its ``\\x00json:[…]`` string (e.g. after a proactive prune re-inserts history, #124102). A body
    that no longer parses stays a sentinel string; ``_merge_consecutive_users`` refuses to weld it."""
    from hermes_state import SessionDB  # lazy: the persistence layer owns the sentinel codec

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        decoded = SessionDB._decode_content(content)
        if decoded is not content:
            msg["content"] = decoded
