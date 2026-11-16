"""Focused extraction preserving the existing caller contract."""
from __future__ import annotations
from typing import Optional
from pathlib import Path


def _translate_docker_container_media_path(
    candidate: Path, session_key: str = "", workspace_key: str = "", *, quiet: bool = False, workspace_cwd: str = "",
) -> Optional[Path]:
    """Container-absolute path -> host path via longest-prefix match over ``docker_volumes``, the
    auto-mounted cache dirs (``/root/.hermes/...``), persistent ``/workspace`` and ``/root``.

    ``workspace_key`` is the 16-hex digest of one session's host cwd ``workspace_cwd`` (its cwd
    mount). It selects that session's ``<base>:ws:<digest>`` sandbox and is empty for MEDIA delivery.
    """
    from gateway.platforms.base import (Path, _cache_dir_container_mounts, _parse_docker_volume_mounts, _path_is_within, _resolve_path, _warn_unresolved_docker_media, contextlib)
    if not candidate.as_posix().startswith("/"):  # POSIX even on Windows gateways (not is_absolute)
        return None
    # In-process gateways (Desktop, `hermes serve`) may not have bridged terminal.* config into
    # TERMINAL_* env yet; the bridge is idempotent.
    with contextlib.suppress(Exception):
        from tools.terminal_tool import _ensure_terminal_env_bridged
        _ensure_terminal_env_bridged()
    from gateway.platforms.docker_media_paths import (
        _default_docker_workspace_host_roots, _docker_persistent_sandbox_roots,
    )
    mounts = [*_parse_docker_volume_mounts(), *_cache_dir_container_mounts()]
    mounted = {c.as_posix() for _, c in mounts}
    # Synthetic /workspace mounts: this session's workspace sandbox, then the
    # shared profile layout, then the legacy per-session dir.
    if "/workspace" not in mounted:
        mounts.extend(
            (root, Path("/workspace"))
            for root in _default_docker_workspace_host_roots(session_key, workspace_key, workspace_cwd))
    # Synthetic /root mounts catch stray home writes (/root/out.png; cache mounts are longer
    # prefixes). /root/.hermes/* that missed a cache mount is the container's credential surface —
    # translating it via the home mount would dodge the host denylist.
    if "/root" not in mounted and not candidate.as_posix().startswith("/root/.hermes"):
        mounts.extend(
            (root, Path("/root"))
            for root in _docker_persistent_sandbox_roots(session_key, "home", workspace_key))
    if not mounts:
        _warn_unresolved_docker_media(candidate, session_key, "no sandbox mounts resolved", quiet)
        return None
    # Longest container-prefix match; equal-length prefixes are tried in insertion order.
    candidate_posix = candidate.as_posix()
    matched = [(host_root, container_root, len(prefix)) for host_root, container_root in mounts
               for prefix in (container_root.as_posix().rstrip("/") or "/",)
               if candidate_posix == prefix or candidate_posix.startswith(prefix + "/")]
    if not matched:
        _warn_unresolved_docker_media(candidate, session_key, "no mounted prefix matches", quiet)
        return None
    for host_root, container_root, _score in sorted(matched, key=lambda m: -m[2]):
        translated = _resolve_path(host_root / candidate.relative_to(container_root), strict=True)
        if translated is not None and (
                translated == host_root or _path_is_within(translated, host_root)):
            return translated
    _warn_unresolved_docker_media(candidate, session_key, "host file missing from sandbox", quiet)
    return None
