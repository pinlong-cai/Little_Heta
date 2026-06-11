"""CLI command: heta remember."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from heta.cli.branding import ERR, HETA, MUTED, OK
from heta.config.io import load_config
from heta.cli.errors import print_error
from heta.mem.pipeline import remember

console = Console()


def remember_command(
    text: str = typer.Argument(..., help="Text to remember."),
) -> None:
    """Save something for Little Heta to remember."""
    config = load_config()
    if config is None:
        console.print(f"[{ERR}]Heta is not initialised. Run `heta init` first.[/]")
        raise typer.Exit(1)

    try:
        with console.status(f"[bold {HETA}]Extracting memories...[/]"):
            result = remember(text, config)
    except Exception as exc:
        print_error(console, "Remember failed.", exc)
        raise typer.Exit(1) from None

    console.print(
        Panel(
            f"[bold {HETA}]L1 episodes:[/] {result.l1_count}\n"
            f"[bold {HETA}]L2 facts:[/]    {result.l2_count}\n"
            f"[{MUTED}]session: {result.session_id}[/]\n"
            f"[{MUTED}]elapsed: {result.elapsed_s}s[/]",
            title="remember",
            border_style=OK,
        )
    )
