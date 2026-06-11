"""CLI command: heta recall."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

from heta.cli.branding import ERR, HETA, MUTED, WARN
from heta.config.io import load_config
from heta.cli.errors import print_error
from heta.mem.recall import recall

console = Console()

# Technical layer names — shown in --debug output.
_LAYER_LABELS = {
    "raw": "L0 Raw",
    "episode": "L1 Episode",
    "atomic_fact": "L2 Atomic Fact",
    "kb_insight": "KB Insight",
}

# User-facing layer names — shown in the recall result box.
_SOURCE_LABELS = {
    "raw": "Conversation",
    "episode": "Episodes",
    "atomic_fact": "Facts",
    "kb_insight": "Documents",
}


def recall_command(
    query: str = typer.Argument(..., help="What to recall."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Results per layer."),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Show layer ranking, reason, and scored evidence."
    ),
) -> None:
    """Look up what Little Heta remembers."""
    config = load_config()
    if config is None:
        console.print(f"[{ERR}]Heta is not initialised. Run `heta init` first.[/]")
        raise typer.Exit(1)

    try:
        with console.status(f"[bold {HETA}]Searching memories...[/]"):
            result = recall(query, config, top_k=top_k)
    except Exception as exc:
        print_error(console, "Recall failed.", exc)
        raise typer.Exit(1) from None

    if debug:
        _show_debug(result)

    _show_result(result)


def _show_result(result) -> None:
    lines = Text()
    lines.append("Query: ", style=f"bold {HETA}")
    lines.append(f'"{result.query}"\n\n')

    lines.append("Answer:\n", style=f"bold {HETA}")
    if result.answer:
        lines.append(result.answer)
    else:
        lines.append(
            "I couldn't find a confident answer in your memories yet — "
            "the most relevant pieces are listed below.",
            style=MUTED,
        )

    source = _source_text(result)
    if source.plain:
        lines.append("\n\n")
        lines.append("Source:\n", style=f"bold {HETA}")
        lines.append(source)

    console.print(Panel(lines, title="recall", border_style=HETA, padding=(1, 2)))


def _source_text(result) -> Text:
    text = Text()
    for layer_ev in result.evidence:
        if not layer_ev.items:
            continue
        label = _SOURCE_LABELS.get(layer_ev.layer, layer_ev.layer)
        text.append(f"{label}\n", style=HETA)
        for item in layer_ev.items:
            text.append("  · ", style=HETA)
            text.append(f"{_item_text(layer_ev.layer, item)}\n", style=MUTED)
    text.rstrip()
    return text


def _show_debug(result) -> None:
    ranking_str = " > ".join(_LAYER_LABELS.get(r, r) for r in result.ranking)
    console.print(f"\n[bold {WARN}]── DEBUG ──[/]\n")

    console.print(f"[bold {HETA}]Ranking[/]")
    console.print(f"  {ranking_str}\n")

    console.print(f"[bold {HETA}]Reason[/]")
    console.print(Padding(Text(result.reason), (0, 0, 0, 2)))
    console.print()

    console.print(f"[bold {HETA}]Evidence[/]")
    for layer_ev in result.evidence:
        if not layer_ev.items:
            continue
        label = _LAYER_LABELS.get(layer_ev.layer, layer_ev.layer)
        console.print(f"  [{HETA}]{label}[/]")
        for item in layer_ev.items:
            score = item.get("score", 0)
            line = Text("    ")
            line.append(f"{score:.3f} · ", style=MUTED)
            line.append(_item_text(layer_ev.layer, item))
            console.print(line)
        console.print()
    console.print(f"[bold {WARN}]──────────[/]\n")


def _item_text(layer: str, item: dict) -> str:
    if layer == "raw":
        return item.get("text_content", "")
    if layer == "episode":
        return item.get("summary", "")
    if layer == "kb_insight":
        return item.get("insight", "")
    return item.get("fact_text", "")
