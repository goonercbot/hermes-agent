"""Focused extraction preserving the existing caller contract."""
from __future__ import annotations


def _looks_structured_value(value: str) -> bool:
    """True when *value* plausibly encodes a YAML/JSON list or mapping. Deliberately conservative:
    a bare leading ``-`` is not a trigger (``-5``, ``--flag`` must stay strings)."""
    stripped = value.lstrip()
    if stripped[:1] in ('[', '{'):
        return True
    if '\n' not in value:
        return False
    for line in value.splitlines():
        item = line.strip()
        if item == '-' or item.startswith('- '):
            return True
        # ``key: value`` / ``key:`` mapping-entry shape (no whitespace in the key).
        head, sep, _rest = item.partition(': ')
        if sep and head and ' ' not in head and not head.startswith('#'):
            return True
        if item.endswith(':') and ' ' not in item[:-1] and item[:-1]:
            return True
    return False
