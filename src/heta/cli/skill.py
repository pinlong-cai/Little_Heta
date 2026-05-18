"""`heta skill` command."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from heta.assistants import install_assistant_skills, skill_template_dir, skill_template_files, skill_template_hint
from heta.cli.branding import HETA, MUTED, OK, WARN

console = Console()


def skill_command() -> None:
    """Install the Little Heta skill into supported agent frameworks."""
    try:
        installed = install_assistant_skills()
    except Exception as exc:
        console.print(f"[{WARN}]?[/] Could not install assistant skills: {exc}")
        raise typer.Exit(1) from exc

    table = Table.grid(padding=(0, 2))
    table.add_column(style=f"bold {HETA}")
    table.add_column(overflow="fold")
    for item in installed:
        table.add_row(item.assistant, _short_path(item.path))

    template_dir = skill_template_dir()
    for filename in skill_template_files():
        table.add_row(filename, _short_path(template_dir / filename))
    table.add_row("Manual use", f"{skill_template_hint()}.")

    console.print(
        Panel(
            table,
            title="skill",
            border_style=OK,
            padding=(1, 2),
        )
    )


def _short_path(path: Path) -> str:
    home = Path.home()
    try:
        return "~/" + str(path.expanduser().resolve().relative_to(home.resolve()))
    except (OSError, ValueError):
        return str(path)
