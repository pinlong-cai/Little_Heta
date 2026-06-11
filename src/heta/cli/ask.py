"""CLI command: heta ask."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from heta.cli.branding import CYAN, ERR, HETA, MUTED, OK, WARN
from heta.config.io import load_config
from heta.cli.errors import print_error
from heta.query.smart_query import smart_query

console = Console()

_SOURCE_STYLES = {
    "memory": ("● memory", f"bold {OK}"),
    "kb": ("● KB", f"bold {HETA}"),
    "both": ("● memory + KB", f"bold {CYAN}"),
}


def ask_command(
    question: str = typer.Argument(..., help="Question to ask."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Results per layer / KB vector match."),
    debug: bool = typer.Option(False, "--debug", "-d", help="Print agent steps, memory evidence, and KB output."),
) -> None:
    """Ask anything — answered from your memory and documents."""
    config = load_config()
    if config is None:
        console.print(f"[{ERR}]Heta is not initialised. Run `heta init` first.[/]")
        raise typer.Exit(1)

    try:
        with console.status(f"[bold {HETA}]Thinking...[/]"):
            result = smart_query(question, config, top_k=top_k)
    except Exception as exc:
        print_error(console, "Ask failed.", exc)
        raise typer.Exit(1) from None

    if debug:
        console.print(f"\n[bold {WARN}]── DEBUG ──[/]")
        console.print(f"agent steps: {' → '.join(result.agent_steps) or '(none)'}")

        if result.memory_evidence:
            console.print("\n[bold]memory evidence:[/bold]")
            for layer_ev in result.memory_evidence:
                console.print(f"  [bold]{layer_ev.layer}[/bold] ({len(layer_ev.items)} hits)")
                for i, item in enumerate(layer_ev.items, 1):
                    score = item.get("score", 0)
                    console.print(f"    [dim][{i}; score={score:.4f}][/dim]")
                    if layer_ev.layer == "raw":
                        console.print(f"      {item.get('text_content', '')}")
                    elif layer_ev.layer == "episode":
                        console.print(f"      {item.get('summary', '')}")
                    elif layer_ev.layer == "kb_insight":
                        console.print(f"      [dim]source:[/dim] {item.get('source_path', '')}")
                        console.print(f"      {item.get('insight', '')}")
                    else:
                        console.print(f"      {item.get('fact_text', '')}")

        if result.kb_result:
            console.print("\n[bold]kb result:[/bold]")
            paths = [s.path for s in result.kb_result.sources]
            console.print(f"  used sources: {paths or '(empty)'}")
            if result.kb_result.insights:
                console.print(f"  agent insights ({len(result.kb_result.insights)}):")
                for i, qi in enumerate(result.kb_result.insights, 1):
                    console.print(f"    [dim][{i}] sources:[/dim] {qi.source_paths}")
                    console.print(f"        {qi.text}")
            console.print(f"  written_back: {result.written_back}")
        console.print(f"[bold {WARN}]──────────[/]\n")

    label, style = _SOURCE_STYLES[result.source]
    source_line = Text()
    source_line.append("Source: ", style=f"bold {HETA}")
    source_line.append(label, style=style)
    if result.written_back:
        source_line.append(f"  ({result.written_back} memories written back)", style=MUTED)

    console.print(
        Panel(
            _AnswerRenderable(Markdown(result.answer), source_line, _kb_sources_text(result)),
            title="ask",
            border_style=HETA,
            padding=(1, 2),
        )
    )


def _kb_sources_text(result) -> Text:
    text = Text()
    if not (result.kb_result and result.kb_result.sources):
        return text
    for src in result.kb_result.sources:
        title = src.title or src.path
        heading = f" — {src.heading_path}" if src.heading_path else ""
        text.append(f"[{src.wiki_id}] ", style=MUTED)
        text.append(f"{title}{heading}\n")
    text.rstrip()
    return text


class _AnswerRenderable:
    def __init__(self, answer: Markdown, source: Text, kb_sources: Text) -> None:
        self.answer = answer
        self.source = source
        self.kb_sources = kb_sources

    def __rich_console__(self, console: Console, options):
        yield Text("Answer:", style=f"bold {HETA}")
        yield self.answer
        yield Text("")
        yield self.source
        if self.kb_sources.plain:
            yield Text("")
            yield Text("KB Sources:", style=f"bold {HETA}")
            yield self.kb_sources
