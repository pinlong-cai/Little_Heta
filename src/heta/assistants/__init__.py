"""Install Little Heta usage guides into AI coding assistants."""

from __future__ import annotations

import importlib.resources
import os
from dataclasses import dataclass
from pathlib import Path

from heta.config.io import CONFIG_DIR

_CODEX_DIR = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
_CLAUDE_DIR = Path.home() / ".claude"

HETA_SKILL_DIR = CONFIG_DIR / "skills" / "heta"
CODEX_SKILL_DIR = _CODEX_DIR / "skills" / "heta"
CLAUDE_SKILL_DIR = _CLAUDE_DIR / "skills" / "heta"

_SKILL_FILES = ("SKILL.md", "COMMANDS.md")


@dataclass(frozen=True)
class InstalledSkill:
    assistant: str
    path: Path


def claude_code_detected() -> bool:
    """True when a Claude Code config directory exists for the current user."""
    return _CLAUDE_DIR.is_dir()


def codex_detected() -> bool:
    """True when a Codex config directory exists for the current user."""
    return _CODEX_DIR.is_dir()


def install_codex_skill(target_dir: Path | None = None) -> Path:
    """Copy the bundled `heta` skill into the Codex global skills directory."""
    return _copy_skill(target_dir or CODEX_SKILL_DIR)


def install_claude_skill(target_dir: Path | None = None) -> Path:
    """Copy the bundled `heta` skill into the Claude Code skills directory.

    Overwrites any existing copy so re-running `heta init` refreshes the skill.
    Returns the directory the skill was written to.
    """
    return _copy_skill(target_dir or CLAUDE_SKILL_DIR)


def install_portable_skill(target_dir: Path | None = None) -> Path:
    """Copy the bundled `heta` skill into Little Heta's own config directory."""
    return _copy_skill(target_dir or HETA_SKILL_DIR)


def install_assistant_skills() -> list[InstalledSkill]:
    """Install Little Heta skills into supported global assistant folders."""
    return [
        InstalledSkill("Little Heta", install_portable_skill()),
        InstalledSkill("Codex", install_codex_skill()),
        InstalledSkill("Claude Code", install_claude_skill()),
    ]


def skill_template_files() -> tuple[str, ...]:
    """Files that make up the portable Little Heta skill."""
    return _SKILL_FILES


def skill_template_dir() -> Path:
    """Stable user-facing directory containing Little Heta's portable skill."""
    return HETA_SKILL_DIR


def skill_template_hint() -> str:
    """Human-readable location hint for manual agent-framework installation."""
    return "copy SKILL.md and COMMANDS.md from the Little Heta skill folder"


def _copy_skill(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)

    source = importlib.resources.files("heta.assistants") / "templates" / "claude_skill"
    for name in _SKILL_FILES:
        text = (source / name).read_text(encoding="utf-8")
        (dest / name).write_text(text, encoding="utf-8")
    return dest


__all__ = [
    "CLAUDE_SKILL_DIR",
    "CODEX_SKILL_DIR",
    "HETA_SKILL_DIR",
    "InstalledSkill",
    "claude_code_detected",
    "codex_detected",
    "install_assistant_skills",
    "install_claude_skill",
    "install_codex_skill",
    "install_portable_skill",
    "skill_template_dir",
    "skill_template_files",
    "skill_template_hint",
]
