"""External service binding is explicit and cannot republish shared launchers."""
import json
import os
import plistlib
import stat
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.platforms('posix')
@pytest.mark.parametrize('mode', [0o644, 0o600, 0o640])
def test_binding_preserves_stamp_access_and_unrelated_fields(tmp_path, mode):
    from hermes_cli.external_service_binding import bind_external_service_python

    (tmp_path / 'hermes_bootstrap.py').touch()
    stamp = tmp_path / 'install-stamp.json'
    original = {'updateMechanism': 'external', 'other': {'kept': [1, 2]}}
    stamp.write_text(json.dumps(original))
    stamp.chmod(mode)
    before = stamp.stat()
    bind_external_service_python(tmp_path, Path(sys.executable))
    after = stamp.stat()
    assert stat.S_IMODE(after.st_mode) == mode
    assert (after.st_uid, after.st_gid) == (before.st_uid, before.st_gid)
    updated = json.loads(stamp.read_text())
    assert all(updated[key] == value for key, value in original.items())
    assert updated['servicePython'] == {'source': str(tmp_path.resolve()), 'python': sys.executable}


@pytest.mark.platforms('posix')
def test_binding_metadata_failure_leaves_original_unchanged(tmp_path, monkeypatch):
    from hermes_cli.external_service_binding import bind_external_service_python

    (tmp_path / 'hermes_bootstrap.py').touch()
    stamp = tmp_path / 'install-stamp.json'
    stamp.write_text('{"updateMechanism": "external", "other": 42}')
    stamp.chmod(0o640)
    before = stamp.stat()
    content = stamp.read_bytes()

    def fail_chmod(*args):
        raise PermissionError('cannot preserve mode')
    monkeypatch.setattr(os, 'fchmod', fail_chmod)
    with pytest.raises(PermissionError, match='cannot preserve mode'):
        bind_external_service_python(tmp_path, Path(sys.executable))
    after = stamp.stat()
    assert stamp.read_bytes() == content
    assert (after.st_ino, after.st_uid, after.st_gid, stat.S_IMODE(after.st_mode)) == (
        before.st_ino, before.st_uid, before.st_gid, stat.S_IMODE(before.st_mode))
    assert not list(tmp_path.glob('.install-stamp-*'))


@pytest.mark.platforms('posix')
def test_binding_ownership_failure_leaves_original_unchanged(tmp_path, monkeypatch):
    from hermes_cli.external_service_binding import bind_external_service_python

    (tmp_path / 'hermes_bootstrap.py').touch()
    stamp = tmp_path / 'install-stamp.json'
    stamp.write_text('{"updateMechanism": "external", "other": 42}')
    stamp.chmod(0o640)
    before = stamp.stat()
    content = stamp.read_bytes()
    real_fstat = os.fstat
    calls = 0

    def different_staged_group(fd):
        nonlocal calls
        result = real_fstat(fd)
        calls += 1
        if calls == 2:
            values = list(result)
            values[5] = result.st_gid + 1
            return os.stat_result(values)
        return result

    def fail_chown(*args):
        raise PermissionError('cannot preserve ownership')

    monkeypatch.setattr(os, 'fstat', different_staged_group)
    monkeypatch.setattr(os, 'fchown', fail_chown)
    with pytest.raises(PermissionError, match='cannot preserve ownership'):
        bind_external_service_python(tmp_path, Path(sys.executable))
    after = stamp.stat()
    assert stamp.read_bytes() == content
    assert (after.st_ino, after.st_uid, after.st_gid, stat.S_IMODE(after.st_mode)) == (
        before.st_ino, before.st_uid, before.st_gid, stat.S_IMODE(before.st_mode))
    assert not list(tmp_path.glob('.install-stamp-*'))


@pytest.mark.platforms('macos')
def test_binding_refuses_stamp_with_acl_without_replacing(tmp_path):
    from hermes_cli.external_service_binding import bind_external_service_python

    (tmp_path / 'hermes_bootstrap.py').touch()
    stamp = tmp_path / 'install-stamp.json'
    stamp.write_text('{"updateMechanism": "external"}')
    subprocess.run(['/bin/chmod', '+a', 'everyone deny write', str(stamp)], check=True)
    before = stamp.stat()
    content = stamp.read_bytes()
    with pytest.raises(OSError, match='ACL cannot be preserved'):
        bind_external_service_python(tmp_path, Path(sys.executable))
    after = stamp.stat()
    assert stamp.read_bytes() == content
    assert (after.st_ino, after.st_uid, after.st_gid, after.st_mode) == (
        before.st_ino, before.st_uid, before.st_gid, before.st_mode)
    assert '+ ' in subprocess.run(['/bin/ls', '-le', str(stamp)], check=True,
                                 capture_output=True, text=True).stdout


@pytest.mark.platforms('macos')
def test_external_service_generator_uses_binding(tmp_path, monkeypatch):
    from hermes_cli import gateway
    from hermes_cli.external_service_binding import bind_external_service_python

    root = tmp_path / 'source'
    root.mkdir()
    (root / 'install-stamp.json').write_text(json.dumps({'updateMechanism': 'external'}))
    (root / 'hermes_bootstrap.py').touch()
    python = Path('/Library/Developer/CommandLineTools/usr/bin/python3')
    bind_external_service_python(root, python)
    monkeypatch.setattr(gateway, 'PROJECT_ROOT', root)
    from hermes_cli import _launchers
    monkeypatch.setattr(_launchers, 'resolve_store_python', lambda *a, **k: Path('/foreign/store/bin/python'))
    generated = plistlib.loads(gateway.generate_launchd_plist().encode())
    command = generated['ProgramArguments'][-1]
    assert command.count(str(python)) == 2
    assert str(root) in command


def test_binding_refuses_missing_invalid_and_foreign(tmp_path, monkeypatch):
    from hermes_cli.external_service_binding import external_service_python, bind_external_service_python
    root = tmp_path / 'source'; root.mkdir()
    stamp = root / 'install-stamp.json'
    (root / 'hermes_bootstrap.py').touch()
    stamp.write_text(json.dumps({'updateMechanism': 'external'}))
    assert external_service_python(root, require_if_store=True) is None  # legacy external, no PM
    from hermes_cli import _launchers
    monkeypatch.setattr(_launchers, 'resolve_store_python', lambda *a, **k: tmp_path / 'store-python')
    with pytest.raises(ValueError, match='no service Python'):
        external_service_python(root, require_if_store=True)
    with pytest.raises(ValueError):
        bind_external_service_python(root, Path('relative'))
    assert 'servicePython' not in json.loads(stamp.read_text())
    stamp.write_text(json.dumps({'updateMechanism': 'external', 'servicePython': {
        'source': str(tmp_path / 'other'), 'python': '/missing'}}))
    with pytest.raises(ValueError, match='another source'):
        external_service_python(root)
    stamp.write_text(json.dumps({'updateMechanism': 'external', 'servicePython': {
        'source': str(root), 'python': '/missing'}}))
    with pytest.raises(ValueError, match='missing'):
        external_service_python(root)
    stamp.write_text(json.dumps({'updateMechanism': 'managed'}))
    with pytest.raises(ValueError, match='not externally owned'):
        bind_external_service_python(root, Path('/missing'))
    assert external_service_python(root) is None


def test_managed_store_command_unchanged(tmp_path, monkeypatch):
    from hermes_cli import _launchers
    root = tmp_path / 'source'; root.mkdir()
    stamp = root / 'install-stamp.json'
    stamp.write_text(json.dumps({'updateMechanism': 'git'}))
    store_python = tmp_path / 'store-python'; store_python.touch()
    monkeypatch.setattr(_launchers, 'resolve_store_python', lambda *a, **k: store_python)
    command = _launchers.installation_command(root, ['gateway', 'run'], python='/different/python')
    assert command == [str(root / '.hermes/bin/hermes'), 'gateway', 'run']


@pytest.mark.platforms('macos')
def test_refresh_intercepts_reload_and_refuses_foreign(tmp_path, monkeypatch):
    from hermes_cli import gateway, gateway_launchd
    from hermes_cli.external_service_binding import bind_external_service_python
    root = tmp_path / 'source'; root.mkdir()
    (root / 'hermes_bootstrap.py').touch()
    (root / 'install-stamp.json').write_text(json.dumps({'updateMechanism': 'external'}))
    python = Path('/Library/Developer/CommandLineTools/usr/bin/python3')
    bind_external_service_python(root, python)
    launcher = root / '.hermes/bin/hermes'; launcher.parent.mkdir(parents=True)
    launcher.write_text('external launcher must not change')
    home = tmp_path / 'home'; home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(gateway, 'PROJECT_ROOT', root)
    from hermes_cli import _launchers
    monkeypatch.setattr(_launchers, 'resolve_store_python', lambda *a, **k: Path('/foreign/store/bin/python'))
    monkeypatch.setattr(_launchers, 'ensure_install_launchers', lambda *a, **k: pytest.fail('external launcher published'))
    monkeypatch.setattr(gateway, 'get_hermes_home', lambda: home)
    plist = tmp_path / 'unit.plist'
    monkeypatch.setattr(gateway, 'get_launchd_plist_path', lambda: plist)
    monkeypatch.setattr(gateway, 'launchd_plist_is_current', lambda: False)
    monkeypatch.setattr(gateway, '_launchd_domain', lambda: 'gui/test')
    # The real safety gate must reject this disposable home. Intercept only after
    # asserting it, so the lifecycle below can exercise its synthetic plist.
    assert gateway_launchd.gateway_service_owner.refuse_temp_home_service_write(
        gateway.generate_launchd_plist(), 'launchd plist')
    monkeypatch.setattr(gateway_launchd.gateway_service_owner, 'refuse_temp_home_service_write', lambda *args: False)
    calls = []
    monkeypatch.setattr(gateway_launchd, '_bootout_drain_and_rebootstrap', lambda *args: calls.append(args) or True)
    import gateway.status
    monkeypatch.setattr(gateway.status, 'get_running_pid', lambda: None)
    plist.write_bytes(plistlib.dumps({'EnvironmentVariables': {'HERMES_HOME': str(tmp_path / 'foreign')}}))
    before = plist.read_bytes()
    assert gateway_launchd.refresh_launchd_plist_if_needed() is False
    assert plist.read_bytes() == before and not calls
    plist.write_bytes(plistlib.dumps({'EnvironmentVariables': {'HERMES_HOME': str(home)}}))
    assert gateway_launchd.refresh_launchd_plist_if_needed() is True
    assert len(calls) == 1
    generated = plistlib.loads(plist.read_bytes())
    assert generated['ProgramArguments'][-1].count(str(python)) == 2
    assert launcher.read_text() == 'external launcher must not change'
