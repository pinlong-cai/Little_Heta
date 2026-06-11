"""`heta query` command."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from heta.cli.branding import HETA, MUTED, WARN
from heta.config.io import CONFIG_PATH, load_config
from heta.cli.errors import print_error
from heta.query import QueryResult, run_wiki_query

console = Console()


def query_command(
    question: str = typer.Argument(..., help="Question to answer from the Little Heta wiki."),
    top_k: int = typer.Option(5, "--top-k", min=1, max=10, help="Initial vector matches to include."),
) -> None:
    """Ask a question about your inserted documents."""
    config = load_config()
    if config is None:
        console.print(f"[{WARN}]?[/] Little Heta is not initialized.")
        console.print(f"[{MUTED}]  Missing config:[/] {CONFIG_PATH}")
        console.print(f"[{MUTED}]  Next:[/] [bold {HETA}]heta init[/]")
        raise typer.Exit(1)

    try:
        with console.status(f"[bold {HETA}]heta query[/] [{MUTED}]reading wiki[/]", spinner="dots"):
            result = run_wiki_query(question, config, top_k=top_k)
    except Exception as exc:
        print_error(console, "Query failed.", exc)
        raise typer.Exit(1) from None

    _show_result(result)


def _show_result(result: QueryResult) -> None:
    markdown = Markdown(result.answer.strip() or "No answer returned.")
    console.print(
        Panel(
            _ResultRenderable(markdown, _sources_text(result)),
            title="query",
            border_style=HETA,
            padding=(1, 2),
        )
    )


class _ResultRenderable:
    def __init__(self, answer: Markdown, sources: Text) -> None:
        self.answer = answer
        self.sources = sources

    def __rich_console__(self, console: Console, options):
        yield Text("Answer:", style=f"bold {HETA}")
        yield self.answer
        if self.sources.plain:
            yield Text("")
            yield Text("Sources:", style=f"bold {HETA}")
            yield self.sources


def _sources_text(result: QueryResult) -> Text:
    text = Text()
    for source in result.sources:
        label = f"[{source.wiki_id}]" if source.wiki_id is not None else "[?]"
        detail = f"{label} {source.title}"
        if source.heading_path:
            detail += f" — {source.heading_path}"
        detail += f" ({source.path})"
        text.append(detail + "\n")
    text.rstrip()
    return text
