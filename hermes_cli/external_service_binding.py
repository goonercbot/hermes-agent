"""Explicit interpreter binding recorded by an external installation owner."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from pm.paths import install_stamp_path

_KEY = 'servicePython'


def _validate(root: Path, value: object) -> str:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError('external service Python must be an absolute executable path')
    python = Path(value)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError('external service Python is missing or not executable')
    if not (root / 'hermes_bootstrap.py').is_file():
        raise ValueError('external installation source is missing its bootstrap')
    try:
        result = subprocess.run([str(python), '-I', '-c', 'import sys; print(sys.executable)'],
                                capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('external service Python cannot start') from exc
    if result.stdout.strip() != str(python):
        raise ValueError('external service Python identity does not match its binding')
    return str(python)


def external_service_python(root: Path, *, require_if_store: bool = False) -> str | None:
    root = Path(root).resolve()
    path = install_stamp_path(root)
    try:
        stamp = json.loads(path.read_text(encoding='utf-8-sig'))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise ValueError('invalid installation stamp') from exc
    if not isinstance(stamp, dict):
        raise ValueError('invalid installation stamp')
    if _KEY not in stamp:
        if stamp.get('updateMechanism') == 'external' and require_if_store:
            from hermes_cli._launchers import resolve_store_python
            if resolve_store_python(root, publication=True) is not None:
                raise ValueError('external installation has no service Python binding')
        return None
    binding = stamp[_KEY]
    if stamp.get('updateMechanism') != 'external' or not isinstance(binding, dict):
        raise ValueError('service Python binding requires an external installation')
    if binding.get('source') != str(root):
        raise ValueError('external service binding belongs to another source')
    return _validate(root, binding.get('python'))


def bind_external_service_python(root: Path, python: Path) -> None:
    """Record an owner's explicit choice; never claim or publish a launcher."""
    root = Path(root).resolve()
    path = install_stamp_path(root)
    stamp = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(stamp, dict) or stamp.get('updateMechanism') != 'external':
        raise ValueError('installation is not externally owned')
    value = _validate(root, str(python))
    stamp[_KEY] = {'source': str(root), 'python': value}
    fd, temp = tempfile.mkstemp(prefix='.install-stamp-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(stamp, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
