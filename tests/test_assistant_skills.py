from pathlib import Path

from heta import assistants


def test_install_assistant_skills_installs_codex_and_claude(monkeypatch, tmp_path: Path) -> None:
    heta_dir = tmp_path / "heta" / "skills" / "heta"
    codex_dir = tmp_path / "codex" / "skills" / "heta"
    claude_dir = tmp_path / "claude" / "skills" / "heta"
    monkeypatch.setattr(assistants, "HETA_SKILL_DIR", heta_dir)
    monkeypatch.setattr(assistants, "CODEX_SKILL_DIR", codex_dir)
    monkeypatch.setattr(assistants, "CLAUDE_SKILL_DIR", claude_dir)

    installed = assistants.install_assistant_skills()

    assert [(item.assistant, item.path) for item in installed] == [
        ("Little Heta", heta_dir),
        ("Codex", codex_dir),
        ("Claude Code", claude_dir),
    ]
    for directory in (heta_dir, codex_dir, claude_dir):
        assert (directory / "SKILL.md").read_text(encoding="utf-8").startswith("---")
        assert (directory / "COMMANDS.md").read_text(encoding="utf-8").startswith("# Little Heta")
