"""Little Heta CLI."""

from __future__ import annotations

import typer

from heta.cli.clean import clean_command
from heta.cli import init as init_module
from heta.cli.init import interactive_init
from heta.cli.insert import insert_command
from heta.cli.query import query_command
from heta.cli.status import status_command
from heta.cli.vector import app as vector_app

app = typer.Typer(
    name="heta",
    help="Little Heta command line interface.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.callback()
def main() -> None:
    """Little Heta command line interface."""


@app.command("init")
def init_command() -> None:
    """Run the first-time Little Heta initialization wizard."""
    try:
        interactive_init()
    except (KeyboardInterrupt, EOFError):
        init_module.console.print("\n[yellow]Initialization cancelled.[/yellow]")
        raise typer.Exit(130) from None


app.command("insert")(insert_command)
app.command("query")(query_command)
app.command("clean")(clean_command)
app.command("status")(status_command)
app.add_typer(vector_app)
