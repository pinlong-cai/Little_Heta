"""SQLite connection factory and schema initialisation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

from heta.mem.client import EMBEDDING_DIM


def get_connection(path: Path, *, with_vec: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    if with_vec:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS session (
            session_id      TEXT PRIMARY KEY,
            started_at      INTEGER NOT NULL,
            ended_at        INTEGER,
            consolidated    INTEGER NOT NULL DEFAULT 0,
            consolidated_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS l0_turn (
            session_id   TEXT    NOT NULL REFERENCES session(session_id),
            turn_index   INTEGER NOT NULL,
            role         TEXT    NOT NULL,
            modality     TEXT    NOT NULL DEFAULT 'text',
            text_content TEXT    NOT NULL,
            created_at   INTEGER NOT NULL,
            UNIQUE(session_id, turn_index)
        );

        CREATE TABLE IF NOT EXISTS memory_meta (
            memory_id      TEXT    PRIMARY KEY,
            memory_type    TEXT    NOT NULL,
            session_id     TEXT    REFERENCES session(session_id),
            origin         TEXT    NOT NULL,
            kb_uid         TEXT,
            status         TEXT    NOT NULL DEFAULT 'active',
            deprecated_by  TEXT    REFERENCES memory_meta(memory_id),
            recency_score  REAL    NOT NULL DEFAULT 1.0,
            access_freq    INTEGER NOT NULL DEFAULT 0,
            user_emphasis  REAL    NOT NULL DEFAULT 0.0,
            importance     REAL    NOT NULL DEFAULT 0.5,
            confidence     REAL    NOT NULL DEFAULT 0.9,
            created_at     INTEGER NOT NULL,
            last_access_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS l1_episodic (
            memory_id      TEXT PRIMARY KEY REFERENCES memory_meta(memory_id) ON DELETE CASCADE,
            who            TEXT NOT NULL,
            what           TEXT NOT NULL,
            where_loc      TEXT,
            when_ts        INTEGER,
            when_text      TEXT,
            when_resolved  TEXT,
            when_precision TEXT,
            why            TEXT,
            summary        TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_l1_when ON l1_episodic(when_ts);

        CREATE TABLE IF NOT EXISTS l2_semantic (
            memory_id     TEXT    PRIMARY KEY REFERENCES memory_meta(memory_id) ON DELETE CASCADE,
            subject       TEXT    NOT NULL,
            predicate     TEXT    NOT NULL,
            object        TEXT    NOT NULL,
            object_type   TEXT    NOT NULL DEFAULT 'literal',
            fact_text     TEXT    NOT NULL DEFAULT '',
            t_valid_start INTEGER NOT NULL,
            t_valid_end   INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_l2_predicate ON l2_semantic(predicate);

        CREATE TABLE IF NOT EXISTS kb_insight (
            memory_id    TEXT PRIMARY KEY REFERENCES memory_meta(memory_id) ON DELETE CASCADE,
            insight      TEXT NOT NULL,
            question     TEXT,
            source_path  TEXT NOT NULL,
            wiki_id      INTEGER,
            heading_path TEXT,
            created_at   INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_kb_insight_source ON kb_insight(source_path);
        CREATE INDEX IF NOT EXISTS idx_kb_insight_wiki   ON kb_insight(wiki_id);

        CREATE TABLE IF NOT EXISTS kb_insight_source (
            memory_id   TEXT NOT NULL REFERENCES kb_insight(memory_id) ON DELETE CASCADE,
            source_path TEXT NOT NULL,
            PRIMARY KEY (memory_id, source_path)
        );

        CREATE INDEX IF NOT EXISTS idx_kb_insight_source_path
            ON kb_insight_source(source_path);
    """)
    _migrate(conn)
    _ensure_vec_table(conn)
    conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after initial schema creation."""
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    l2_cols = {row[1] for row in conn.execute("PRAGMA table_info(l2_semantic)")}
    if "fact_text" not in l2_cols:
        conn.execute("ALTER TABLE l2_semantic ADD COLUMN fact_text TEXT NOT NULL DEFAULT ''")
    if "when_text" not in l2_cols:
        conn.execute("ALTER TABLE l2_semantic ADD COLUMN when_text TEXT")
    if "when_resolved" not in l2_cols:
        conn.execute("ALTER TABLE l2_semantic ADD COLUMN when_resolved TEXT")
    if "when_precision" not in l2_cols:
        conn.execute("ALTER TABLE l2_semantic ADD COLUMN when_precision TEXT")

    l1_cols = {row[1] for row in conn.execute("PRAGMA table_info(l1_episodic)")}
    if "when_resolved" not in l1_cols:
        conn.execute("ALTER TABLE l1_episodic ADD COLUMN when_resolved TEXT")
    if "when_precision" not in l1_cols:
        conn.execute("ALTER TABLE l1_episodic ADD COLUMN when_precision TEXT")

    # Backfill kb_insight_source from kb_insight.source_path for pre-existing rows.
    # Idempotent: PRIMARY KEY (memory_id, source_path) prevents duplicates on rerun.
    try:
        conn.execute("""
            INSERT OR IGNORE INTO kb_insight_source (memory_id, source_path)
            SELECT memory_id, source_path FROM kb_insight
            WHERE source_path IS NOT NULL AND source_path != ''
        """)
    except Exception:
        pass

    # legacy tables from earlier design — kept so existing DBs don't break
    if "kb_source" not in tables:
        conn.execute("""CREATE TABLE kb_source (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL,
            wiki_id INTEGER, page_title TEXT,
            page_path TEXT NOT NULL, heading_path TEXT,
            synced_at INTEGER NOT NULL)""")
    if "kb_chunk" not in tables:
        conn.execute("""CREATE TABLE kb_chunk (
            memory_id TEXT PRIMARY KEY, wiki_id INTEGER, page_title TEXT,
            page_path TEXT NOT NULL, heading_path TEXT,
            chunk_text TEXT NOT NULL, synced_at INTEGER NOT NULL)""")
    if "kb_qa" not in tables:
        conn.execute("""CREATE TABLE kb_qa (
            memory_id TEXT PRIMARY KEY,
            question TEXT NOT NULL, answer TEXT NOT NULL,
            created_at INTEGER NOT NULL)""")
    if "kb_qa_chunk" not in tables:
        conn.execute("""CREATE TABLE kb_qa_chunk (
            qa_memory_id TEXT NOT NULL, chunk_memory_id TEXT NOT NULL,
            PRIMARY KEY (qa_memory_id, chunk_memory_id))""")


def _ensure_vec_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""CREATE VIRTUAL TABLE IF NOT EXISTS l2_fact_vec USING vec0(
            memory_id TEXT PRIMARY KEY,
            embedding FLOAT[{EMBEDDING_DIM}]
        )"""
    )
    conn.execute(
        f"""CREATE VIRTUAL TABLE IF NOT EXISTS l1_episode_vec USING vec0(
            memory_id TEXT PRIMARY KEY,
            embedding FLOAT[{EMBEDDING_DIM}]
        )"""
    )
    conn.execute(
        """CREATE VIRTUAL TABLE IF NOT EXISTS l0_turn_fts USING fts5(
            session_id UNINDEXED,
            turn_index UNINDEXED,
            text_content
        )"""
    )
    conn.execute(
        f"""CREATE VIRTUAL TABLE IF NOT EXISTS kb_insight_vec USING vec0(
            memory_id TEXT PRIMARY KEY,
            embedding FLOAT[{EMBEDDING_DIM}]
        )"""
    )
    # legacy vec tables — kept so existing DBs don't break
    conn.execute(
        f"""CREATE VIRTUAL TABLE IF NOT EXISTS kb_chunk_vec USING vec0(
            memory_id TEXT PRIMARY KEY,
            embedding FLOAT[{EMBEDDING_DIM}]
        )"""
    )
    conn.execute(
        f"""CREATE VIRTUAL TABLE IF NOT EXISTS kb_qa_vec USING vec0(
            memory_id TEXT PRIMARY KEY,
            embedding FLOAT[{EMBEDDING_DIM}]
        )"""
    )
