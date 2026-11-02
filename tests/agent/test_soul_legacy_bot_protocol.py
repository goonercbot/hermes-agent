"""Prompt filtering must not rewrite user-owned identity files on disk."""

from tools import bot_mode_probe

LEGACY = "# Me\n\nBe terse.\n\n## Messaging other agents\n\nold plugin text\n- `bot-a`\n"


def test_load_soul_md_drops_legacy_protocol_section(tmp_path, monkeypatch):
    from agent import prompt_builder

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "SOUL.md").write_text(LEGACY + "\n## Symbols\n\nuse ◆\n", encoding="utf-8")

    soul = prompt_builder.load_soul_md(home_override=home)

    assert "Messaging other agents" not in soul and "old plugin text" not in soul
    assert "Be terse." in soul and "use ◆" in soul  # neighbouring sections survive
    # Bot Chat itself keeps the live section: SOUL no longer suppresses the probe.
    bot_mode_probe._reset_cache_for_tests()
    (home / "profiles" / "bot").mkdir(parents=True)
    (home / "profiles" / "bot" / "profile.yaml").write_text("ui_meta:\n  hermes-bots: {}\n", encoding="utf-8")
    assert "`@bot`" in bot_mode_probe.get_bot_mode_protocol_section(home)


def test_migration_41_preserves_every_profile_soul(tmp_path, monkeypatch):
    from hermes_cli.config_migrations import _migrate_to_41

    home = tmp_path / ".hermes"
    (home / "profiles" / "worker").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "SOUL.md").write_text(LEGACY, encoding="utf-8")
    (home / "profiles" / "worker" / "SOUL.md").write_text("# Worker\n\n## Messaging other agents\nx\n", encoding="utf-8")

    results = {"config_added": []}
    _migrate_to_41(results, quiet=True)

    assert (home / "SOUL.md").read_text(encoding="utf-8") == LEGACY
    assert (home / "profiles" / "worker" / "SOUL.md").read_text(encoding="utf-8") == "# Worker\n\n## Messaging other agents\nx\n"
    assert results["config_added"] == []
    _migrate_to_41(results := {"config_added": []}, quiet=True)
    assert results["config_added"] == []
