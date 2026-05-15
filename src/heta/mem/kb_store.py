"""CRUD and search operations for kb_insight."""

from __future__ import annotations

import sqlite3

import sqlite_vec

from heta.mem.models import KBInsight


def insert_kb_insight(conn: sqlite3.Connection, insight: KBInsight) -> None:
    """Insert insight row plus one row per source_path into the join table."""
    conn.execute(
        """INSERT INTO kb_insight
               (memory_id, insight, question, source_path, wiki_id, heading_path, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (insight.memory_id, insight.insight, insight.question, insight.source_path,
         insight.wiki_id, insight.heading_path, insight.created_at),
    )
    for path in insight.source_paths:
        conn.execute(
            "INSERT OR IGNORE INTO kb_insight_source (memory_id, source_path) VALUES (?, ?)",
            (insight.memory_id, path),
        )


def insert_insight_embedding(
    conn: sqlite3.Connection, memory_id: str, embedding: list[float]
) -> None:
    conn.execute(
        "INSERT INTO kb_insight_vec (memory_id, embedding) VALUES (?, ?)",
        (memory_id, sqlite_vec.serialize_float32(embedding)),
    )


def get_source_paths(conn: sqlite3.Connection, memory_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT source_path FROM kb_insight_source WHERE memory_id = ? ORDER BY source_path",
        (memory_id,),
    ).fetchall()
    return [r[0] for r in rows]


def search_kb_insights(
    conn: sqlite3.Connection,
    embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
    rows = conn.execute(
        """SELECT i.memory_id, i.insight, i.source_path, v.distance
           FROM kb_insight_vec v
           JOIN kb_insight i ON i.memory_id = v.memory_id
           JOIN memory_meta m ON m.memory_id = i.memory_id
           WHERE v.embedding MATCH ? AND k = ?
             AND m.status = 'active'
           ORDER BY v.distance""",
        (sqlite_vec.serialize_float32(embedding), top_k),
    ).fetchall()
    results = []
    for r in rows:
        mid = r["memory_id"]
        results.append({
            "memory_id": mid,
            "insight": r["insight"],
            "source_path": r["source_path"],
            "source_paths": get_source_paths(conn, mid),
            "score": 1.0 / (1.0 + float(r["distance"])),
        })
    return results
