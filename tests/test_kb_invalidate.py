"""Tests for heta.mem.kb_invalidate."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest

from heta.mem.db import get_connection, init_db
from heta.mem.kb_invalidate import delete_all_insights, delete_insights_by_paths
from heta.mem.kb_store import insert_insight_embedding, insert_kb_insight
from heta.mem.meta_store import insert_meta
from heta.mem.models import KBInsight, MemoryMeta


@pytest.fixture()
def conn(tmp_path: Path):
    db = tmp_path / "test_mem.sqlite3"
    c = get_connection(db, with_vec=True)
    init_db(c)
    yield c
    c.close()


def _now() -> int:
    return int(time.time())


def _insert_insight(conn, source_paths, insight_text: str = "fact") -> str:
    """source_paths can be a single str (for backward-compat tests) or a list."""
    if isinstance(source_paths, str):
        source_paths = [source_paths]
    mid = str(uuid.uuid4())
    insert_meta(conn, MemoryMeta(
        memory_id=mid, memory_type="kb_insight", session_id=None,
        origin="kb_insight", created_at=_now(), last_access_at=_now(),
    ))
    insert_kb_insight(conn, KBInsight(
        memory_id=mid, insight=insight_text, question="q",
        source_paths=source_paths, wiki_id=None, heading_path=None,
        created_at=_now(),
    ))
    # 1024-dim float embedding (matches EMBEDDING_DIM)
    from heta.mem.client import EMBEDDING_DIM
    insert_insight_embedding(conn, mid, [0.0] * EMBEDDING_DIM)
    conn.commit()
    return mid


def _count(conn, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# ── tests ─────────────────────────────────────────────────────────────────────


def test_delete_by_paths_removes_matching_rows(conn):
    _insert_insight(conn, "pages/1-foo.md")
    _insert_insight(conn, "pages/1-foo.md")  # 2 insights on same page
    _insert_insight(conn, "pages/2-bar.md")

    deleted = delete_insights_by_paths(conn, ["pages/1-foo.md"])

    assert deleted == 2
    assert _count(conn, "kb_insight") == 1
    assert _count(conn, "kb_insight_vec") == 1
    assert _count(conn, "memory_meta") == 1


def test_delete_by_paths_leaves_other_pages_untouched(conn):
    _insert_insight(conn, "pages/1-foo.md")
    bar_id = _insert_insight(conn, "pages/2-bar.md")

    delete_insights_by_paths(conn, ["pages/1-foo.md"])

    rows = conn.execute("SELECT memory_id FROM kb_insight").fetchall()
    assert [r[0] for r in rows] == [bar_id]


def test_delete_by_paths_empty_input(conn):
    _insert_insight(conn, "pages/1-foo.md")
    assert delete_insights_by_paths(conn, []) == 0
    assert _count(conn, "kb_insight") == 1


def test_delete_by_paths_no_match(conn):
    _insert_insight(conn, "pages/1-foo.md")
    assert delete_insights_by_paths(conn, ["pages/does-not-exist.md"]) == 0
    assert _count(conn, "kb_insight") == 1


def test_delete_all_clears_everything(conn):
    _insert_insight(conn, "pages/1-foo.md")
    _insert_insight(conn, "pages/2-bar.md")
    _insert_insight(conn, "pages/3-baz.md")

    deleted = delete_all_insights(conn)

    assert deleted == 3
    assert _count(conn, "kb_insight") == 0
    assert _count(conn, "kb_insight_vec") == 0
    assert _count(conn, "memory_meta") == 0


def test_delete_all_on_empty_db_returns_zero(conn):
    assert delete_all_insights(conn) == 0


def test_delete_by_paths_invalidates_multi_source_insight(conn):
    """An insight derived from multiple pages dies when ANY of its sources changes."""
    multi = _insert_insight(conn, ["pages/1-foo.md", "pages/2-bar.md"])
    solo = _insert_insight(conn, ["pages/3-baz.md"])

    deleted = delete_insights_by_paths(conn, ["pages/2-bar.md"])

    assert deleted == 1
    remaining = [r[0] for r in conn.execute("SELECT memory_id FROM kb_insight").fetchall()]
    assert multi not in remaining
    assert solo in remaining
    # both rows in kb_insight_source for the multi insight should be gone
    assert _count(conn, "kb_insight_source") == 1


def test_delete_by_paths_preserves_other_memory_types(conn):
    """Deleting kb_insight by path must not touch L1/L2/etc."""
    _insert_insight(conn, "pages/1-foo.md")
    # an unrelated memory_meta row (e.g. L2)
    other = str(uuid.uuid4())
    insert_meta(conn, MemoryMeta(
        memory_id=other, memory_type="L2", session_id=None,
        origin="extracted", created_at=_now(), last_access_at=_now(),
    ))

    delete_insights_by_paths(conn, ["pages/1-foo.md"])

    assert _count(conn, "memory_meta") == 1
    remaining = conn.execute("SELECT memory_id FROM memory_meta").fetchone()
    assert remaining[0] == other
