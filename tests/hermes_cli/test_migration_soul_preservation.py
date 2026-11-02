"""A stale profile's config upgrade must not rewrite anyone's identity."""
from pathlib import Path

import hermes_yaml as yaml
import pytest


@pytest.mark.parametrize("target_name", ["default", "target"])
def test_versioned_migration_preserves_fleet_identity(tmp_path, monkeypatch, target_name):
    from hermes_cli.config import DEFAULT_CONFIG, migrate_config

    root = tmp_path / ".hermes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    homes = {"default": root, **{n: root / "profiles" / n for n in ("target", "stale", "current", "plain")}}
    identities = {
        "default": b"\xef\xbb\xbf# Default\r\n\r\n## Messaging other agents\r\nuser notes\r\n",
        "target": b"# Target\n\n## Messaging other agents\nkeep\n\n## More\ntext\n",
        "stale": b"# Stale\n\n## Messaging other agents\nkeep\n",
        "current": b"# Current\n\n## Messaging other agents\nkeep\n",
        "plain": b"# No heading\r\nunchanged\r\n",
    }
    for name, home in homes.items():
        home.mkdir(parents=True, exist_ok=True)
        version = DEFAULT_CONFIG["_config_version"] if name == "current" else 40
        (home / "config.yaml").write_text(f"_config_version: {version}\nfixture: retained\n")
        (home / "SOUL.md").write_bytes(identities[name])

    def migrate(name):
        monkeypatch.setenv("HERMES_HOME", str(homes[name]))
        migrate_config(interactive=False, quiet=True)
        assert yaml.safe_load((homes[name] / "config.yaml").read_text())["_config_version"] == DEFAULT_CONFIG["_config_version"]
        assert {n: (h / "SOUL.md").read_bytes() for n, h in homes.items()} == identities

    excluded = {n: (h / "config.yaml").read_bytes() for n, h in homes.items() if n != target_name}
    migrate(target_name)
    assert all((homes[n] / "config.yaml").read_bytes() == b for n, b in excluded.items())
    target_config = (homes[target_name] / "config.yaml").read_bytes()
    migrate(target_name)
    assert (homes[target_name] / "config.yaml").read_bytes() == target_config
    others = {n: (h / "config.yaml").read_bytes() for n, h in homes.items() if n != "stale"}
    migrate("stale")
    assert all((homes[n] / "config.yaml").read_bytes() == b for n, b in others.items())
