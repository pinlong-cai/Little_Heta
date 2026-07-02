"""Orchestrator for the heta remember pipeline."""

from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from heta.config.schema import HetaConfig
from heta.mem import l0_store, l1_store, l2_store, meta_store, session_store
from heta.mem.client import build_chat_model, build_embedding_model
from heta.mem.db import get_connection, init_db
from heta.mem.embedder import fact_text
from heta.mem.l1_dedup import detect_episode_duplicates_batch
from heta.mem.l1_extractor import extract_episodes, resolve_when_ts
from heta.mem.l2_conflict import detect_conflicts_batch
from heta.mem.l2_extractor import extract_facts
from heta.mem.models import L0Turn, L1Episodic, L2Semantic, MemoryMeta, Session
from heta.mem.paths import db_path, ensure_mem_dir
from heta.providers.model_protocols import ChatModelProtocol, EmbeddingModelProtocol


@dataclass
class RememberResult:
    session_id: str
    l0_count: int
    l1_count: int
    l2_count: int
    elapsed_s: float
    timings: dict[str, float] | None = None


def _detect_episode_duplicates(
    path,
    summaries,
    chat_model: ChatModelProtocol,
    embedding_model: EmbeddingModelProtocol,
    config,
):
    conn = get_connection(path, with_vec=True)
    init_db(conn)
    try:
        return detect_episode_duplicates_batch(
            conn=conn,
            new_episode_summaries=summaries,
            chat_model=chat_model,
            embedding_model=embedding_model,
            config=config,
        )
    finally:
        conn.close()


def _detect_fact_conflicts(
    path,
    fact_texts,
    chat_model: ChatModelProtocol,
    embedding_model: EmbeddingModelProtocol,
    config,
    session_id,
):
    conn = get_connection(path, with_vec=True)
    init_db(conn)
    try:
        return detect_conflicts_batch(
            conn=conn,
            new_fact_texts=fact_texts,
            chat_model=chat_model,
            embedding_model=embedding_model,
            config=config,
            session_id=session_id,
        )
    finally:
        conn.close()


def remember(
    text: str,
    config: HetaConfig,
    *,
    mode: str = "high",
) -> RememberResult:
    if mode not in {"fast", "high"}:
        raise ValueError("remember mode must be one of: fast, high")

    t_total = time.perf_counter()
    timings: dict[str, float] = {}

    t_stage = time.perf_counter()
    ensure_mem_dir()
    conn = get_connection(db_path(), with_vec=True)
    init_db(conn)

    now = int(time.time())
    session_id = str(uuid.uuid4())
    timings["setup"] = round(time.perf_counter() - t_stage, 3)

    # --- session + L0 ---
    t_stage = time.perf_counter()
    session_store.create_session(conn, Session(session_id=session_id, started_at=now))
    l0_store.insert_turn(
        conn,
        L0Turn(
            session_id=session_id,
            turn_index=0,
            role="user",
            modality="text",
            text_content=text,
            created_at=now,
        ),
    )
    timings["l0_write"] = round(time.perf_counter() - t_stage, 3)
    if mode == "fast":
        t_stage = time.perf_counter()
        session_store.close_session(conn, session_id, int(time.time()))
        conn.close()
        timings["close"] = round(time.perf_counter() - t_stage, 3)
        timings["total"] = round(time.perf_counter() - t_total, 3)
        return RememberResult(
            session_id=session_id,
            l0_count=1,
            l1_count=0,
            l2_count=0,
            elapsed_s=round(time.perf_counter() - t_total, 2),
            timings=timings,
        )

    chat_model = build_chat_model(config)
    embedding_model = build_embedding_model(config)

    # --- extract ---
    t_stage = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        episodes_future = executor.submit(
            extract_episodes, chat_model, text, config, now
        )
        facts_future = executor.submit(
            extract_facts, chat_model, text, config, now
        )
        raw_episodes = episodes_future.result()
        raw_facts = facts_future.result()
    timings["extract"] = round(time.perf_counter() - t_stage, 3)

    # --- prepare L1/L2 records ---
    t_stage = time.perf_counter()
    prepared_episodes = []
    for ep in raw_episodes:
        summary = ep.get("summary", ep.get("what", ""))
        prepared_episodes.append((
            ep,
            json.dumps(ep.get("who", ["user"]), ensure_ascii=False),
            ep.get("what", ""),
            ep.get("where_loc"),
            resolve_when_ts(ep.get("when_resolved")),
            ep.get("when_text"),
            ep.get("when_resolved"),
            ep.get("when_precision"),
            ep.get("why"),
            summary,
        ))

    prepared_facts = []
    for raw_fact in raw_facts:
        subject = str(raw_fact.get("subject", ""))
        predicate = str(raw_fact.get("predicate", ""))
        object_ = str(raw_fact.get("object", ""))
        raw_object_type = raw_fact.get("object_type", "literal")
        object_type_val = raw_object_type[0] if isinstance(raw_object_type, list) else str(raw_object_type)
        ft = fact_text(subject, predicate, object_)
        prepared_facts.append((raw_fact, subject, predicate, object_, object_type_val, ft))
    timings["prepare"] = round(time.perf_counter() - t_stage, 3)

    # --- detect L1 duplicates and L2 conflicts concurrently ---
    t_stage = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        episode_future = executor.submit(
            _detect_episode_duplicates,
            db_path(),
            [item[9] for item in prepared_episodes],
            chat_model,
            embedding_model,
            config,
        ) if prepared_episodes else None
        conflict_future = executor.submit(
            _detect_fact_conflicts,
            db_path(),
            [item[5] for item in prepared_facts],
            chat_model,
            embedding_model,
            config,
            session_id,
        ) if prepared_facts else None

        episode_dedup_results = episode_future.result() if episode_future else []
        conflict_results = conflict_future.result() if conflict_future else []
    timings["dedup_conflict"] = round(time.perf_counter() - t_stage, 3)

    # --- persist L1 ---
    t_stage = time.perf_counter()
    l1_count = 0
    for prepared, dedup_result in zip(prepared_episodes, episode_dedup_results, strict=True):
        if dedup_result.duplicate_of is not None:
            continue

        _, who, what, where_loc, when_ts, when_text, when_resolved, when_precision, why, summary = prepared
        memory_id = str(uuid.uuid4())
        meta = MemoryMeta(
            memory_id=memory_id,
            memory_type="L1",
            session_id=session_id,
            origin="extracted",
            created_at=now,
            last_access_at=now,
        )
        episode = L1Episodic(
            memory_id=memory_id,
            who=who,
            what=what,
            where_loc=where_loc,
            when_ts=when_ts,
            when_text=when_text,
            when_resolved=when_resolved,
            when_precision=when_precision,
            why=why,
            summary=summary,
        )
        meta_store.insert_meta(conn, meta)
        l1_store.insert_episodic(conn, episode)
        l1_store.insert_episode_embedding(conn, memory_id, dedup_result.embedding)
        l1_count += 1
    timings["persist_l1"] = round(time.perf_counter() - t_stage, 3)

    # --- persist L2 (semantic conflict resolution) ---
    t_stage = time.perf_counter()
    l2_count = 0
    for prepared, conflict_result in zip(prepared_facts, conflict_results, strict=True):
        if conflict_result.duplicate_of is not None:
            continue

        raw_fact, subject, predicate, object_, object_type_val, ft = prepared
        memory_id = str(uuid.uuid4())
        meta = MemoryMeta(
            memory_id=memory_id,
            memory_type="L2",
            session_id=session_id,
            origin="extracted",
            created_at=now,
            last_access_at=now,
        )
        fact_record = L2Semantic(
            memory_id=memory_id,
            subject=subject,
            predicate=predicate,
            object=object_,
            object_type=object_type_val,
            fact_text=ft,
            t_valid_start=now,
            when_text=raw_fact.get("when_text"),
            when_resolved=raw_fact.get("when_resolved"),
            when_precision=raw_fact.get("when_precision"),
        )

        # insert new meta + fact first so FK reference is valid
        meta_store.insert_meta(conn, meta)
        for old_id in conflict_result.ids_to_deprecate:
            l2_store.expire_fact(conn, old_id, now)
            meta_store.deprecate(conn, old_id, memory_id)
        l2_store.insert_fact(conn, fact_record)
        l2_store.insert_fact_embedding(conn, memory_id, conflict_result.embedding)
        l2_count += 1
    timings["persist_l2"] = round(time.perf_counter() - t_stage, 3)

    t_stage = time.perf_counter()
    session_store.close_session(conn, session_id, int(time.time()))
    conn.close()
    timings["close"] = round(time.perf_counter() - t_stage, 3)
    timings["total"] = round(time.perf_counter() - t_total, 3)

    return RememberResult(
        session_id=session_id,
        l0_count=1,
        l1_count=l1_count,
        l2_count=l2_count,
        elapsed_s=round(time.perf_counter() - t_total, 2),
        timings=timings,
    )
