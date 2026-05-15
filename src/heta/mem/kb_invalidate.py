"""Invalidate kb_insight memories whose source wiki pages changed."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable

from heta.mem.db import get_connection, init_db
from heta.mem.paths import db_path

logger = logging.getLogger(__name__)


def invalidate_by_paths(paths: Iterable[str]) -> int:
    """Delete kb_insight memories whose source_path is in `paths`. Returns count deleted.

    Silently returns 0 when the memory DB does not exist yet, so KB operations
    succeed even if the user never initialised memory.
    """
    path_list = [p for p in paths if p]
    if not path_list:
        return 0
    if not db_path().exists():
        return 0
    try:
        conn = get_connection(db_path(), with_vec=True)
    except Exception:
        logger.warning("memory DB open failed; skip kb_insight invalidation", exc_info=True)
        return 0
    try:
        init_db(conn)
        return delete_insights_by_paths(conn, path_list)
    except Exception:
        logger.warning("kb_insight invalidation failed", exc_info=True)
        return 0
    finally:
        conn.close()


def invalidate_all() -> int:
    """Delete every kb_insight memory. Returns count deleted.

    Silently returns 0 when the memory DB does not exist yet.
    """
    if not db_path().exists():
        return 0
    try:
        conn = get_connection(db_path(), with_vec=True)
    except Exception:
        logger.warning("memory DB open failed; skip kb_insight invalidation", exc_info=True)
        return 0
    try:
        init_db(conn)
        return delete_all_insights(conn)
    except Exception:
        logger.warning("kb_insight invalidation (all) failed", exc_info=True)
        return 0
    finally:
        conn.close()


def delete_insights_by_paths(conn: sqlite3.Connection, paths: list[str]) -> int:
    """Connection-level helper. Exposed for tests and callers with an open conn.

    An insight is invalidated if ANY of its source_paths matches a changed page.
    """
    if not paths:
        return 0
    placeholders = ",".join("?" for _ in paths)
    ids = [
        r[0]
        for r in conn.execute(
            f"SELECT DISTINCT memory_id FROM kb_insight_source WHERE source_path IN ({placeholders})",
            paths,
        ).fetchall()
    ]
    if not ids:
        return 0
    id_placeholders = ",".join("?" for _ in ids)
    # vec0 virtual table does not honour FK cascade; delete explicitly.
    conn.execute(f"DELETE FROM kb_insight_vec WHERE memory_id IN ({id_placeholders})", ids)
    # memory_meta delete cascades to kb_insight via ON DELETE CASCADE,
    # which in turn cascades to kb_insight_source.
    conn.execute(f"DELETE FROM memory_meta WHERE memory_id IN ({id_placeholders})", ids)
    conn.commit()
    return len(ids)


def delete_all_insights(conn: sqlite3.Connection) -> int:
    """Connection-level helper to wipe all kb_insight rows. Exposed for tests."""
    ids = [r[0] for r in conn.execute("SELECT memory_id FROM kb_insight").fetchall()]
    if not ids:
        return 0
    conn.execute("DELETE FROM kb_insight_vec")
    placeholders = ",".join("?" for _ in ids)
    conn.execute(f"DELETE FROM memory_meta WHERE memory_id IN ({placeholders})", ids)
    conn.commit()
    return len(ids)
