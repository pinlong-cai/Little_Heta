"""CLI command: heta ask."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from heta.config.io import load_config
from heta.query.smart_query import smart_query

console = Console()

_SOURCE_STYLES = {
    "memory": ("● memory", "bold green"),
    "kb": ("● KB", "bold cyan"),
    "both": ("● memory + KB", "bold magenta"),
}


def ask_command(
    question: str = typer.Argument(..., help="Question to ask."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Results per layer / KB vector match."),
    debug: bool = typer.Option(False, "--debug", "-d", help="Print agent steps, memory evidence, and KB output."),
) -> None:
    """Answer a question via an outer agent that decides between memory and the KB."""
    config = load_config()
    if config is None:
        console.print("[red]Heta is not initialised. Run `heta init` first.[/red]")
        raise typer.Exit(1)

    with console.status("[cyan]Thinking...[/cyan]"):
        result = smart_query(question, config, top_k=top_k)

    if debug:
        console.print("\n[bold yellow]── DEBUG ──[/bold yellow]")
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
        console.print("[bold yellow]──────────[/bold yellow]\n")

    label, style = _SOURCE_STYLES[result.source]
    header = Text()
    header.append("Source: ")
    header.append(label, style=style)
    if result.written_back:
        header.append(f"  ({result.written_back} memories written back)", style="dim")

    console.print(header)
    console.print()
    console.print(Panel(Markdown(result.answer), border_style="cyan"))

    if result.kb_result and result.kb_result.sources:
        console.print("[dim]KB Sources:[/dim]")
        for src in result.kb_result.sources:
            title = src.title or src.path
            heading = f" — {src.heading_path}" if src.heading_path else ""
            console.print(f"  [dim][{src.wiki_id}][/dim] {title}{heading}")
        console.print()
