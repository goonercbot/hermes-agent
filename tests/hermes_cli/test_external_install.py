"""Externally maintained archive detection and update ownership."""
import json

from hermes_cli.config import detect_install_method, recommended_update_command_for_method
from hermes_cli.steward import classify_install
from hermes_cli.update_contract import evaluate_update_admission


def stamp(root, **values):
    (root / "install-stamp.json").write_text(json.dumps(values), encoding="utf-8")


def test_external_archive_remains_owned_and_routes_updates(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    root = tmp_path / "code"
    root.mkdir()
    stamp(root, source="local", distribution="custom-git-archive", updateMechanism="external")
    assert detect_install_method(root) == "external"
    assert classify_install(root) == ("custom-git-archive", False)
    assert recommended_update_command_for_method("external") != "hermes update"
    refusal = evaluate_update_admission(root)
    assert refusal is not None and refusal.code == "custom-git-archive"
    assert "installed it" in refusal.message


def test_external_with_no_named_owner_does_not_self_update(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    stamp(tmp_path, updateMechanism="external")
    assert detect_install_method(tmp_path) == "external"
    assert classify_install(tmp_path) == ("unknown", False)
    refusal = evaluate_update_admission(tmp_path)
    assert refusal is not None and refusal.code == "external"
    assert "tool that installed" in refusal.update_command


def test_stale_external_stamp_does_not_override_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    (tmp_path / ".git").mkdir()
    stamp(tmp_path, distribution="custom-git-archive", updateMechanism="external")
    assert detect_install_method(tmp_path) == "git"
    assert classify_install(tmp_path) == ("git", True)
    assert evaluate_update_admission(tmp_path) is None


def test_invalid_stamp_remains_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    (tmp_path / "install-stamp.json").write_text("{", encoding="utf-8")
    assert detect_install_method(tmp_path) == "unknown"
    assert classify_install(tmp_path) == ("unknown", False)
