"""Little Heta CLI."""

from __future__ import annotations

import typer

from heta.cli.ask import ask_command
from heta.cli.branding import apply_typer_theme
from heta.cli.clean import clean_command
from heta.cli.dynamic_insert import app as dynamic_insert_app
from heta.cli.mem_clean import mem_clean_command
from heta.cli.mem_show import app as mem_show_app
from heta.cli import init as init_module
from heta.cli.init import interactive_init
from heta.cli.insert import insert_command
from heta.cli.insert_planning import app as insert_planning_app
from heta.cli.query import query_command
from heta.cli.recall import recall_command
from heta.cli.remember import remember_command
from heta.cli.skill import skill_command
from heta.cli.status import status_command
from heta.cli.vector import app as vector_app

apply_typer_theme()

app = typer.Typer(
    name="heta",
    help="Little Heta command line interface.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)


@app.callback()
def main() -> None:
    """Little Heta command line interface."""


@app.command("init")
def init_command() -> None:
    """Set up Little Heta for the first time."""
    try:
        interactive_init()
    except (KeyboardInterrupt, EOFError):
        init_module.console.print("\n[yellow]Initialization cancelled.[/yellow]")
        raise typer.Exit(130) from None


app.command("ask")(ask_command)
app.command("mem-clean")(mem_clean_command)
app.command("insert")(insert_command)
app.command("query")(query_command)
app.command("clean")(clean_command)
app.command("remember")(remember_command)
app.command("recall")(recall_command)
app.command("skill")(skill_command)
app.command("status")(status_command)
app.add_typer(dynamic_insert_app)
app.add_typer(insert_planning_app)
app.add_typer(vector_app)
app.add_typer(mem_show_app)
