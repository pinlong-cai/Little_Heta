"""`heta clean` command."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from heta.config.io import CONFIG_PATH, load_config
from heta.kb import paths
from heta.kb.clean import CleanSummary, clean_knowledge_base

console = Console()

HETA = "rgb(52,144,220)"
MUTED = "rgb(126,146,158)"
OK = "rgb(76,196,142)"
WARN = "rgb(238,183,74)"


def clean_command(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Clean wiki knowledge pages and vector database while keeping raw files."""
    config = load_config()
    if config is None:
        console.print(f"[{WARN}]?[/] Little Heta is not initialized.")
        console.print(f"[{MUTED}]  Missing config:[/] {CONFIG_PATH}")
        console.print(f"[{MUTED}]  Next:[/] [bold {HETA}]heta init[/]")
        raise typer.Exit(1)

    pages = _wiki_pages()
    _show_plan(pages)
    if not yes and not Confirm.ask(
        "Clear the current Heta knowledge base? You can restore this deletion later with git",
        default=False,
    ):
        console.print(f"[{MUTED}]Clean cancelled.[/]")
        raise typer.Exit(0)

    try:
        with console.status(f"[bold {HETA}]heta clean[/] [{MUTED}]cleaning wiki[/]", spinner="dots"):
            summary = clean_knowledge_base()
    except Exception as exc:
        console.print(f"[{WARN}]?[/] Clean failed.")
        console.print(f"[{MUTED}]  Reason:[/] {exc}")
        raise typer.Exit(1) from exc

    _show_result(summary)


def _show_plan(pages: list[Path]) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style=f"bold {HETA}", no_wrap=True)
    table.add_column()
    if pages:
        for page in pages:
            table.add_row("delete", f"pages/{page.name}")
    else:
        table.add_row("delete", "no wiki pages")
    table.add_row("restore", "available through wiki git history after clean commit")

    console.print(
        Panel(
            table,
            title="clean",
            border_style=HETA,
            padding=(1, 2),
        )
    )


def _wiki_pages() -> list[Path]:
    pages_dir = paths.pages_dir()
    if not pages_dir.exists():
        return []
    return sorted(page for page in pages_dir.glob("*.md") if page.is_file())


def _show_result(summary: CleanSummary) -> None:
    console.print()
    console.print(f"[{OK}]✓[/] Clean completed.")
    console.print(f"[{MUTED}]pages deleted:[/] {summary.deleted_pages}")
    console.print(f"[{MUTED}]vector files deleted:[/] {summary.deleted_vector_files}")
    console.print(f"[{MUTED}]invalidated memories:[/] {summary.invalidated_memories}")
    if summary.commit_id:
        console.print(f"[{MUTED}]wiki commit:[/] [bold {HETA}]{summary.commit_id}[/]")
    else:
        console.print(f"[{MUTED}]wiki commit:[/] no changes")
