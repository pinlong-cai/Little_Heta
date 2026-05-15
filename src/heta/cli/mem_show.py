"""`heta mem-show` commands — inspect stored memories."""

from __future__ import annotations

import sqlite3
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table

from heta.mem.db import get_connection, init_db
from heta.mem.paths import db_path

console = Console()

HETA = "rgb(52,144,220)"
MUTED = "rgb(126,146,158)"
WARN = "rgb(238,183,74)"

app = typer.Typer(
    name="mem-show",
    help="Inspect stored memory contents.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.command("insights")
def insights_command(
    source: str | None = typer.Option(None, "--source", "-s", help="Filter by source_path substring (e.g. 'pages/1-foo.md')."),
    question: str | None = typer.Option(None, "--question", "-q", help="Filter by question substring."),
    limit: int = typer.Option(50, "--limit", "-n", help="Max rows to show."),
    full: bool = typer.Option(False, "--full", "-f", help="Show full insight text (no truncation)."),
) -> None:
    """List stored kb_insight memories, newest first."""
    if not db_path().exists():
        console.print(f"[{WARN}]?[/] Memory DB does not exist yet.")
        console.print(f"[{MUTED}]  Run `heta ask` at least once to populate it.[/]")
        raise typer.Exit(0)

    conn = get_connection(db_path(), with_vec=True)
    init_db(conn)
    try:
        rows = _fetch_insights(conn, source=source, question=question, limit=limit)
        total = _count_total(conn, source=source, question=question)
    finally:
        conn.close()

    if not rows:
        console.print(f"[{MUTED}]No insights matched.[/]")
        return

    table = Table(
        title=f"kb_insights ({len(rows)} of {total} shown)",
        show_lines=not full,
        border_style=HETA,
    )
    table.add_column("#", style="dim", justify="right", no_wrap=True)
    table.add_column("created", style=MUTED, no_wrap=True)
    table.add_column("sources", style=MUTED)
    table.add_column("question", style=MUTED)
    table.add_column("insight")

    for i, row in enumerate(rows, 1):
        insight_text = row["insight"] if full else _truncate(row["insight"], 140)
        question_text = row["question"] or ""
        if not full:
            question_text = _truncate(question_text, 50)
        sources_text = "\n".join(row["source_paths"]) if full else _truncate(
            ", ".join(row["source_paths"]), 40
        )
        table.add_row(
            str(i),
            _format_ts(row["created_at"]),
            sources_text,
            question_text,
            insight_text,
        )
    console.print(table)


def _fetch_insights(
    conn: sqlite3.Connection,
    *,
    source: str | None,
    question: str | None,
    limit: int,
) -> list[dict]:
    """Fetch insights and their full source_paths list."""
    base_sql = """
        SELECT i.memory_id, i.insight, i.question, i.created_at
        FROM kb_insight i
        JOIN memory_meta m ON m.memory_id = i.memory_id
        WHERE m.status = 'active'
    """
    clauses, params = _build_filters(source=source, question=question)
    sql = f"{base_sql} {clauses} ORDER BY i.created_at DESC LIMIT ?"
    params.append(max(1, limit))
    rows = conn.execute(sql, params).fetchall()

    results = []
    for r in rows:
        paths = [
            row[0]
            for row in conn.execute(
                "SELECT source_path FROM kb_insight_source WHERE memory_id = ? ORDER BY source_path",
                (r["memory_id"],),
            ).fetchall()
        ]
        results.append({
            "insight": r["insight"],
            "question": r["question"],
            "source_paths": paths,
            "created_at": r["created_at"],
        })
    return results


def _count_total(
    conn: sqlite3.Connection,
    *,
    source: str | None,
    question: str | None,
) -> int:
    base_sql = """
        SELECT COUNT(*) FROM kb_insight i
        JOIN memory_meta m ON m.memory_id = i.memory_id
        WHERE m.status = 'active'
    """
    clauses, params = _build_filters(source=source, question=question)
    row = conn.execute(f"{base_sql} {clauses}", params).fetchone()
    return int(row[0])


def _build_filters(*, source: str | None, question: str | None) -> tuple[str, list]:
    clauses: list[str] = []
    params: list = []
    if source:
        clauses.append(
            "AND i.memory_id IN (SELECT memory_id FROM kb_insight_source WHERE source_path LIKE ?)"
        )
        params.append(f"%{source}%")
    if question:
        clauses.append("AND i.question LIKE ?")
        params.append(f"%{question}%")
    return " ".join(clauses), params


def _truncate(text: str, max_len: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _format_ts(ts: int | None) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")


__all__ = ["app"]
