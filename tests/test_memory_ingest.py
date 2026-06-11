"""Tests for the heta remember ingestion pipeline (pipeline.py).

All LLM and embedding calls are mocked so tests run offline and deterministically.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from heta.config.schema import InsertPlanningConfig, HetaConfig, LLMConfig, MinerUConfig, VectorIndexConfig
from heta.mem.client import EMBEDDING_DIM
from heta.mem.db import get_connection, init_db
from heta.mem.l1_dedup import EpisodeDedupResult
from heta.mem.l2_conflict import ConflictResult
from heta.mem.pipeline import remember


# ── constants ─────────────────────────────────────────────────────────────────

FAKE_EMB = [0.01] * EMBEDDING_DIM  # deterministic dummy embedding

EPISODE_DICT = {
    "who": ["user"],
    "what": "参加了技术分享会",
    "where_loc": "公司会议室",
    "when_text": "上周",
    "when_resolved": "2026-W19",
    "when_precision": "week",
    "why": None,
    "summary": "用户上周在公司会议室参加了技术分享会",
}

FACT_DICT = {
    "subject": "用户",
    "predicate": "居住在",
    "object": "北京朝阳区",
    "object_type": "literal",
    "when_text": None,
    "when_resolved": None,
    "when_precision": None,
}


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def config() -> HetaConfig:
    return HetaConfig(
        version=1,
        llm=LLMConfig(provider="qwen", api_key="sk-test"),
        mineru=MinerUConfig.disabled(),
        vector_index=VectorIndexConfig.enabled(),
        insert_planning=InsertPlanningConfig.enabled(),
    )


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "mem.sqlite3"


# ── patch helper ──────────────────────────────────────────────────────────────

@contextmanager
def _patch_pipeline(
    tmp_db: Path,
    episodes: list[dict] | None = None,
    facts: list[dict] | None = None,
    conflicts: list[str] | None = None,   # memory_ids to deprecate
):
    """Patch all external I/O in pipeline.remember so tests run offline."""
    if episodes is None:
        episodes = []
    if facts is None:
        facts = []
    if conflicts is None:
        conflicts = []

    mock_client = MagicMock()

    def _open_conn(path, *, with_vec=False):
        c = get_connection(tmp_db, with_vec=True)
        init_db(c)
        return c

    with (
        patch("heta.mem.pipeline.ensure_mem_dir"),
        patch("heta.mem.pipeline.db_path", return_value=tmp_db),
        patch("heta.mem.pipeline.get_connection", side_effect=_open_conn),
        patch("heta.mem.pipeline.init_db"),
        patch("heta.mem.pipeline.build_client", return_value=(mock_client, "mock-llm")),
        patch("heta.mem.pipeline.build_embedding_client", return_value=(mock_client, "mock-emb")),
        patch("heta.mem.pipeline.extract_episodes", return_value=episodes),
        patch("heta.mem.pipeline.extract_facts", return_value=facts),
        patch("heta.mem.pipeline.embed_text", return_value=FAKE_EMB),
        patch(
            "heta.mem.pipeline.detect_episode_duplicates_batch",
            side_effect=lambda new_episode_summaries, **kwargs: [
                EpisodeDedupResult(duplicate_of=None, embedding=FAKE_EMB)
                for _ in new_episode_summaries
            ],
        ),
        patch(
            "heta.mem.pipeline.detect_conflicts_batch",
            side_effect=lambda new_fact_texts, **kwargs: [
                ConflictResult(ids_to_deprecate=list(conflicts), embedding=FAKE_EMB)
                for _ in new_fact_texts
            ],
        ),
    ):
        yield


def _open(tmp_db: Path):
    """Open a fresh read connection to the tmp DB after pipeline closes it."""
    return get_connection(tmp_db, with_vec=True)


# ── basic ingestion ───────────────────────────────────────────────────────────

def test_remember_creates_session_and_l0_turn(config, tmp_db) -> None:
    with _patch_pipeline(tmp_db):
        result = remember("hello world", config)

    conn = _open(tmp_db)
    sessions = conn.execute("SELECT * FROM session WHERE session_id = ?", (result.session_id,)).fetchall()
    turns = conn.execute("SELECT * FROM l0_turn WHERE session_id = ?", (result.session_id,)).fetchall()
    conn.close()

    assert len(sessions) == 1
    assert sessions[0]["session_id"] == result.session_id
    assert sessions[0]["ended_at"] is not None  # session was closed
    assert len(turns) == 1
    assert turns[0]["text_content"] == "hello world"
    assert turns[0]["role"] == "user"


def test_remember_l0_turn_indexed_in_fts(config, tmp_db) -> None:
    with _patch_pipeline(tmp_db):
        remember("爬山打羽毛球", config)

    conn = _open(tmp_db)
    rows = conn.execute(
        "SELECT text_content FROM l0_turn_fts WHERE text_content MATCH ?",
        ('"爬山打羽毛球"',),
    ).fetchall()
    conn.close()

    assert len(rows) == 1


def test_remember_empty_extraction_succeeds(config, tmp_db) -> None:
    with _patch_pipeline(tmp_db, episodes=[], facts=[]):
        result = remember("no events here", config)

    assert result.l1_count == 0
    assert result.l2_count == 0


def test_remember_returns_correct_counts(config, tmp_db) -> None:
    with _patch_pipeline(tmp_db, episodes=[EPISODE_DICT], facts=[FACT_DICT, FACT_DICT]):
        result = remember("some text", config)

    assert result.l1_count == 1
    assert result.l2_count == 2
    assert result.session_id != ""
    assert result.elapsed_s >= 0




def test_remember_extracts_episodes_and_facts_concurrently(config, tmp_db) -> None:
    episodes_started = threading.Event()
    facts_started = threading.Event()
    mock_client = MagicMock()

    def _open_conn(path, *, with_vec=False):
        c = get_connection(tmp_db, with_vec=True)
        init_db(c)
        return c

    def fake_extract_episodes(*args, **kwargs):
        episodes_started.set()
        assert facts_started.wait(1.0)
        return []

    def fake_extract_facts(*args, **kwargs):
        facts_started.set()
        assert episodes_started.wait(1.0)
        return []

    with (
        patch("heta.mem.pipeline.ensure_mem_dir"),
        patch("heta.mem.pipeline.db_path", return_value=tmp_db),
        patch("heta.mem.pipeline.get_connection", side_effect=_open_conn),
        patch("heta.mem.pipeline.init_db"),
        patch("heta.mem.pipeline.build_client", return_value=(mock_client, "mock-llm")),
        patch("heta.mem.pipeline.build_embedding_client", return_value=(mock_client, "mock-emb")),
        patch("heta.mem.pipeline.extract_episodes", side_effect=fake_extract_episodes),
        patch("heta.mem.pipeline.extract_facts", side_effect=fake_extract_facts),
        patch("heta.mem.pipeline.detect_episode_duplicates_batch", return_value=[]),
        patch("heta.mem.pipeline.detect_conflicts_batch", return_value=[]),
    ):
        result = remember("some text", config)

    assert result.l1_count == 0
    assert result.l2_count == 0


# ── L1 episode storage ────────────────────────────────────────────────────────

def test_remember_l1_episode_stored_correctly(config, tmp_db) -> None:
    with _patch_pipeline(tmp_db, episodes=[EPISODE_DICT]):
        result = remember("some text", config)

    conn = _open(tmp_db)
    row = conn.execute(
        "SELECT * FROM l1_episodic e JOIN memory_meta m ON e.memory_id = m.memory_id "
        "WHERE m.session_id = ?",
        (result.session_id,),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row["what"] == EPISODE_DICT["what"]
    assert row["where_loc"] == EPISODE_DICT["where_loc"]
    assert row["when_text"] == EPISODE_DICT["when_text"]
    assert row["when_resolved"] == EPISODE_DICT["when_resolved"]
    assert row["when_precision"] == EPISODE_DICT["when_precision"]
    assert row["summary"] == EPISODE_DICT["summary"]
    assert json.loads(row["who"]) == EPISODE_DICT["who"]
    assert row["memory_type"] == "L1"
    assert row["status"] == "active"


def test_remember_l1_episode_embedding_inserted(config, tmp_db) -> None:
    with _patch_pipeline(tmp_db, episodes=[EPISODE_DICT]):
        result = remember("some text", config)

    conn = _open(tmp_db)
    meta = conn.execute(
        "SELECT memory_id FROM memory_meta WHERE session_id = ? AND memory_type = 'L1'",
        (result.session_id,),
    ).fetchone()
    vec_row = conn.execute(
        "SELECT memory_id FROM l1_episode_vec WHERE memory_id = ?",
        (meta["memory_id"],),
    ).fetchone()
    conn.close()

    assert vec_row is not None


def test_remember_l1_who_defaults_to_user_when_missing(config, tmp_db) -> None:
    ep = {**EPISODE_DICT}
    del ep["who"]
    with _patch_pipeline(tmp_db, episodes=[ep]):
        remember("some text", config)

    conn = _open(tmp_db)
    row = conn.execute("SELECT who FROM l1_episodic").fetchone()
    conn.close()

    assert json.loads(row["who"]) == ["user"]


# ── L2 fact storage ───────────────────────────────────────────────────────────

def test_remember_l2_fact_stored_correctly(config, tmp_db) -> None:
    with _patch_pipeline(tmp_db, facts=[FACT_DICT]):
        result = remember("some text", config)

    conn = _open(tmp_db)
    row = conn.execute(
        "SELECT * FROM l2_semantic s JOIN memory_meta m ON s.memory_id = m.memory_id "
        "WHERE m.session_id = ?",
        (result.session_id,),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row["subject"] == FACT_DICT["subject"]
    assert row["predicate"] == FACT_DICT["predicate"]
    assert row["object"] == FACT_DICT["object"]
    assert row["object_type"] == "literal"
    assert row["t_valid_end"] is None   # active, not yet expired
    assert row["status"] == "active"
    assert row["memory_type"] == "L2"


def test_remember_l2_fact_embedding_inserted(config, tmp_db) -> None:
    with _patch_pipeline(tmp_db, facts=[FACT_DICT]):
        result = remember("some text", config)

    conn = _open(tmp_db)
    meta = conn.execute(
        "SELECT memory_id FROM memory_meta WHERE session_id = ? AND memory_type = 'L2'",
        (result.session_id,),
    ).fetchone()
    vec_row = conn.execute(
        "SELECT memory_id FROM l2_fact_vec WHERE memory_id = ?",
        (meta["memory_id"],),
    ).fetchone()
    conn.close()

    assert vec_row is not None


def test_remember_l2_object_type_list_coerced_to_string(config, tmp_db) -> None:
    """LLM occasionally returns object_type as a list; pipeline should normalise it."""
    fact = {**FACT_DICT, "object_type": ["literal", "extra"]}
    with _patch_pipeline(tmp_db, facts=[fact]):
        remember("some text", config)

    conn = _open(tmp_db)
    row = conn.execute("SELECT object_type FROM l2_semantic").fetchone()
    conn.close()

    assert isinstance(row["object_type"], str)
    assert row["object_type"] == "literal"


# ── conflict resolution ───────────────────────────────────────────────────────

def test_remember_conflict_deprecates_old_fact(config, tmp_db) -> None:
    """When detect_conflicts returns an old memory_id, that fact is expired + deprecated."""
    # session 1: insert a fact directly so we have an old_id to conflict
    from heta.mem.l2_store import insert_fact, insert_fact_embedding
    from heta.mem.meta_store import insert_meta
    from heta.mem.models import L2Semantic, MemoryMeta
    from heta.mem.session_store import create_session, close_session
    from heta.mem.models import Session
    import time, uuid

    old_id = str(uuid.uuid4())
    now = int(time.time())

    setup_conn = get_connection(tmp_db, with_vec=True)
    init_db(setup_conn)
    sid0 = str(uuid.uuid4())
    create_session(setup_conn, Session(session_id=sid0, started_at=now))
    close_session(setup_conn, sid0, now)
    insert_meta(setup_conn, MemoryMeta(
        memory_id=old_id, memory_type="L2", session_id=sid0,
        origin="extracted", created_at=now, last_access_at=now,
    ))
    insert_fact(setup_conn, L2Semantic(
        memory_id=old_id, subject="用户", predicate="居住在", object="北京朝阳区",
        object_type="literal", fact_text="用户 居住在 北京朝阳区",
        t_valid_start=now,
    ))
    insert_fact_embedding(setup_conn, old_id, FAKE_EMB)
    setup_conn.close()

    # session 2: new fact conflicts with old_id
    new_fact = {**FACT_DICT, "object": "北京海淀区"}
    with _patch_pipeline(tmp_db, facts=[new_fact], conflicts=[old_id]):
        remember("搬家了", config)

    conn = _open(tmp_db)
    old_row = conn.execute(
        "SELECT s.t_valid_end, m.status, m.deprecated_by "
        "FROM l2_semantic s JOIN memory_meta m ON s.memory_id = m.memory_id "
        "WHERE s.memory_id = ?",
        (old_id,),
    ).fetchone()
    active_rows = conn.execute(
        "SELECT object FROM l2_semantic WHERE t_valid_end IS NULL"
    ).fetchall()
    conn.close()

    assert old_row["t_valid_end"] is not None     # expired
    assert old_row["status"] == "deprecated"
    assert old_row["deprecated_by"] is not None   # FK to new fact
    assert len(active_rows) == 1
    assert active_rows[0]["object"] == "北京海淀区"


def test_remember_no_conflict_keeps_both_facts(config, tmp_db) -> None:
    """When detect_conflicts returns [], both old and new facts remain active."""
    fact_a = {**FACT_DICT, "predicate": "喜欢", "object": "爬山"}
    fact_b = {**FACT_DICT, "predicate": "喜欢", "object": "羽毛球"}

    with _patch_pipeline(tmp_db, facts=[fact_a], conflicts=[]):
        remember("喜欢爬山", config)
    with _patch_pipeline(tmp_db, facts=[fact_b], conflicts=[]):
        remember("喜欢羽毛球", config)

    conn = _open(tmp_db)
    active = conn.execute(
        "SELECT object FROM l2_semantic WHERE t_valid_end IS NULL ORDER BY object"
    ).fetchall()
    conn.close()

    objects = {r["object"] for r in active}
    assert objects == {"爬山", "羽毛球"}


def test_remember_detect_conflicts_batch_receives_session_id(config, tmp_db) -> None:
    """detect_conflicts_batch must be called with the current session_id so same-session
    facts are excluded from conflict candidates."""
    captured_kwargs: dict = {}

    def _fake_detect_conflicts_batch(new_fact_texts, **kwargs):
        captured_kwargs.update(kwargs)
        return [ConflictResult(ids_to_deprecate=[], embedding=FAKE_EMB) for _ in new_fact_texts]

    with (
        _patch_pipeline(tmp_db, facts=[FACT_DICT]),
        patch("heta.mem.pipeline.detect_conflicts_batch", side_effect=_fake_detect_conflicts_batch),
    ):
        result = remember("some text", config)

    assert "session_id" in captured_kwargs
    assert captured_kwargs["session_id"] == result.session_id


# ── multiple sessions ─────────────────────────────────────────────────────────

def test_remember_multiple_sessions_accumulate(config, tmp_db) -> None:
    with _patch_pipeline(tmp_db, facts=[FACT_DICT]):
        r1 = remember("first", config)
    with _patch_pipeline(tmp_db, facts=[FACT_DICT]):
        r2 = remember("second", config)

    assert r1.session_id != r2.session_id

    conn = _open(tmp_db)
    n_sessions = conn.execute("SELECT COUNT(*) FROM session").fetchone()[0]
    n_turns = conn.execute("SELECT COUNT(*) FROM l0_turn").fetchone()[0]
    conn.close()

    assert n_sessions == 2
    assert n_turns == 2


def test_remember_skips_duplicate_l2_fact(config, tmp_db) -> None:
    def _duplicate_result(new_fact_texts, **kwargs):
        return [
            ConflictResult(ids_to_deprecate=[], embedding=FAKE_EMB, duplicate_of="old-fact")
            for _ in new_fact_texts
        ]

    with (
        _patch_pipeline(tmp_db, facts=[FACT_DICT]),
        patch("heta.mem.pipeline.detect_conflicts_batch", side_effect=_duplicate_result),
    ):
        result = remember("duplicate fact", config)

    conn = _open(tmp_db)
    l2_rows = conn.execute("SELECT * FROM l2_semantic").fetchall()
    conn.close()

    assert result.l2_count == 0
    assert l2_rows == []


def test_remember_skips_duplicate_l1_episode(config, tmp_db) -> None:
    def _duplicate_episodes(new_episode_summaries, **kwargs):
        return [
            EpisodeDedupResult(duplicate_of="old-episode", embedding=FAKE_EMB)
            for _ in new_episode_summaries
        ]

    with (
        _patch_pipeline(tmp_db, episodes=[EPISODE_DICT]),
        patch("heta.mem.pipeline.detect_episode_duplicates_batch", side_effect=_duplicate_episodes),
    ):
        result = remember("duplicate episode", config)

    conn = _open(tmp_db)
    l1_rows = conn.execute("SELECT * FROM l1_episodic").fetchall()
    conn.close()

    assert result.l1_count == 0
    assert l1_rows == []


def test_remember_dedups_episodes_and_detects_fact_conflicts_concurrently(config, tmp_db) -> None:
    episode_dedup_started = threading.Event()
    fact_conflict_started = threading.Event()

    def _episode_dedup(new_episode_summaries, **kwargs):
        episode_dedup_started.set()
        assert fact_conflict_started.wait(1.0)
        return [EpisodeDedupResult(duplicate_of=None, embedding=FAKE_EMB) for _ in new_episode_summaries]

    def _fact_conflicts(new_fact_texts, **kwargs):
        fact_conflict_started.set()
        assert episode_dedup_started.wait(1.0)
        return [ConflictResult(ids_to_deprecate=[], embedding=FAKE_EMB) for _ in new_fact_texts]

    with (
        _patch_pipeline(tmp_db, episodes=[EPISODE_DICT], facts=[FACT_DICT]),
        patch("heta.mem.pipeline.detect_episode_duplicates_batch", side_effect=_episode_dedup),
        patch("heta.mem.pipeline.detect_conflicts_batch", side_effect=_fact_conflicts),
    ):
        result = remember("episode and fact", config)

    assert result.l1_count == 1
    assert result.l2_count == 1
