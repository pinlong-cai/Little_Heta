"""CLI command: heta remember."""

from __future__ import annotations

import os

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
    mode: str = typer.Option("high", "--mode", help="Memory mode: fast (raw only) or high (full memory extraction)."),
    fast: bool = typer.Option(False, "--fast", help="Shortcut for --mode fast; store only the raw text."),
) -> None:
    """Save something for Little Heta to remember."""
    selected_mode = "fast" if fast else mode.lower()
    if selected_mode not in {"fast", "high"}:
        console.print(f"[{ERR}]Invalid remember mode: {selected_mode}. Use fast or high.[/]")
        raise typer.Exit(1)

    config = load_config()
    if config is None:
        console.print(f"[{ERR}]Heta is not initialised. Run `heta init` first.[/]")
        raise typer.Exit(1)

    try:
        with console.status(f"[bold {HETA}]Extracting memories...[/]"):
            result = remember(text, config, mode=selected_mode)
    except Exception as exc:
        print_error(console, "Remember failed.", exc)
        raise typer.Exit(1) from None

    body = (
        f"[bold {HETA}]L0 turns:[/]    {result.l0_count}\n"
        f"[bold {HETA}]L1 episodes:[/] {result.l1_count}\n"
        f"[bold {HETA}]L2 facts:[/]    {result.l2_count}\n"
        f"[{MUTED}]session: {result.session_id}[/]\n"
        f"[{MUTED}]elapsed: {result.elapsed_s}s[/]"
    )
    if selected_mode != "high":
        body = f"{body}\n[{MUTED}]mode: {selected_mode}[/]"
    if os.environ.get("HETA_REMEMBER_TIMING") in {"1", "true", "TRUE", "yes", "on"} and result.timings:
        timing_lines = "\n".join(
            f"[{MUTED}]  {name}: {duration:.3f}s[/]"
            for name, duration in result.timings.items()
        )
        body = f"{body}\n\n[bold {HETA}]timing:[/]\n{timing_lines}"

    console.print(
        Panel(
            body,
            title="remember",
            border_style=OK,
        )
    )
