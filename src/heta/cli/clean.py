"""`heta clean` command."""

from __future__ import annotations

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

    _show_plan()
    if not yes and not Confirm.ask("Continue?", default=False):
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


def _show_plan() -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style=f"bold {HETA}")
    table.add_column()
    table.add_row("clean", "wiki pages")
    table.add_row("reset", str(paths.index_path()))
    table.add_row("append", str(paths.log_path()))
    table.add_row("delete", str(paths.vector_db_path()))
    table.add_row("keep", str(paths.raw_dir()))

    console.print(
        Panel(
            table,
            title="clean",
            border_style=HETA,
            padding=(1, 2),
        )
    )


def _show_result(summary: CleanSummary) -> None:
    console.print()
    console.print(f"[{OK}]✓[/] Clean completed.")
    console.print(f"[{MUTED}]pages deleted:[/] {summary.deleted_pages}")
    console.print(f"[{MUTED}]vector files deleted:[/] {summary.deleted_vector_files}")
    if summary.commit_id:
        console.print(f"[{MUTED}]wiki commit:[/] [bold {HETA}]{summary.commit_id}[/]")
    else:
        console.print(f"[{MUTED}]wiki commit:[/] no changes")

