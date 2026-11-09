"""Explicit interpreter binding recorded by an external installation owner."""
from __future__ import annotations

from contextlib import suppress
import ctypes
import errno
import json
import os
import stat
import subprocess
import sys
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


def _refuse_acl(fd: int) -> None:
    """Fail closed rather than lose a restrictive or inherited access ACL."""
    if sys.platform == 'darwin':
        libc = ctypes.CDLL(None, use_errno=True)
        get_acl = libc.acl_get_fd_np
        get_acl.argtypes = [ctypes.c_int, ctypes.c_int]
        get_acl.restype = ctypes.c_void_p
        free_acl = libc.acl_free
        free_acl.argtypes = [ctypes.c_void_p]
        free_acl.restype = ctypes.c_int
        acl = get_acl(fd, 0x100)  # ACL_TYPE_EXTENDED
        if acl:
            free_acl(acl)
            raise OSError('installation stamp ACL cannot be preserved')
        error = ctypes.get_errno()
        if error != errno.ENOENT:
            raise OSError(error, 'cannot inspect installation stamp ACL')
    elif sys.platform.startswith('linux'):
        if any(name.startswith('system.posix_acl_') for name in os.listxattr(fd)):
            raise OSError('installation stamp ACL cannot be preserved')
    else:
        raise OSError('cannot safely inspect installation stamp ACL')


def bind_external_service_python(root: Path, python: Path) -> None:
    """Record an owner's explicit choice; never claim or publish a launcher."""
    root = Path(root).resolve()
    path = install_stamp_path(root)
    with path.open(encoding='utf-8-sig') as original:
        metadata = os.fstat(original.fileno())
        _refuse_acl(original.fileno())
        stamp = json.load(original)
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
            _refuse_acl(stream.fileno())
            staged = os.fstat(stream.fileno())
            if (staged.st_uid, staged.st_gid) != (metadata.st_uid, metadata.st_gid):
                if not hasattr(os, 'fchown'):
                    raise OSError('cannot preserve installation stamp ownership')
                os.fchown(stream.fileno(), metadata.st_uid, metadata.st_gid)
            if not hasattr(os, 'fchmod'):
                raise OSError('cannot preserve installation stamp mode')
            os.fchmod(stream.fileno(), stat.S_IMODE(metadata.st_mode))
            staged = os.fstat(stream.fileno())
            if ((staged.st_uid, staged.st_gid, stat.S_IMODE(staged.st_mode)) !=
                    (metadata.st_uid, metadata.st_gid, stat.S_IMODE(metadata.st_mode))):
                raise OSError('cannot preserve installation stamp access metadata')
            os.fsync(stream.fileno())
        os.replace(temp, path)
        with suppress(OSError):
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
