"""Semantic conflict detection for L2 fact memories."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from heta.config.schema import HetaConfig
from heta.mem.l2_store import search_similar_facts
from heta.mem.prompts import BATCH_CONFLICT_JUDGE_PROMPT, CONFLICT_JUDGE_PROMPT
from heta.providers.model_protocols import (
    ChatCompletionRequest,
    ChatMessage,
    ChatModelOptions,
    ChatModelProtocol,
    EmbeddingModelProtocol,
    EmbeddingRequest,
)

logger = logging.getLogger(__name__)

MIN_CONFLICT_CANDIDATE_SCORE = 0.60


@dataclass(frozen=True)
class ConflictResult:
    ids_to_deprecate: list[str]
    embedding: list[float]
    duplicate_of: str | None = None


def detect_conflicts(
    conn: Any,
    new_fact_text: str,
    chat_model: ChatModelProtocol,
    embedding_model: EmbeddingModelProtocol,
    config: HetaConfig,
    top_k: int = 10,
    session_id: str | None = None,
) -> tuple[list[str], list[float]]:
    """Return memory_ids of existing facts that the new fact contradicts."""
    result = detect_conflicts_batch(
        conn=conn,
        new_fact_texts=[new_fact_text],
        chat_model=chat_model,
        embedding_model=embedding_model,
        config=config,
        top_k=top_k,
        session_id=session_id,
    )[0]
    return result.ids_to_deprecate, result.embedding


def detect_conflicts_batch(
    conn: Any,
    new_fact_texts: list[str],
    chat_model: ChatModelProtocol,
    embedding_model: EmbeddingModelProtocol,
    config: HetaConfig,
    top_k: int = 10,
    session_id: str | None = None,
    min_candidate_score: float = MIN_CONFLICT_CANDIDATE_SCORE,
) -> list[ConflictResult]:
    """Detect contradicted facts for multiple new fact texts with one judge call."""
    if not new_fact_texts:
        return []

    embeddings = _embed_texts(embedding_model, new_fact_texts)
    filtered_candidates: dict[int, list[dict]] = {}

    for index, embedding in enumerate(embeddings):
        candidates = search_similar_facts(
            conn,
            embedding,
            top_k=top_k,
            exclude_session_id=session_id,
        )
        candidates = _filter_candidates(candidates, min_candidate_score)
        duplicate = _find_exact_duplicate(new_fact_texts[index], candidates)
        if duplicate is not None:
            filtered_candidates[index] = [duplicate]
            continue
        if candidates:
            filtered_candidates[index] = candidates

    duplicates = {
        index: candidates[0]["memory_id"]
        for index, candidates in filtered_candidates.items()
        if candidates and _same_fact_text(new_fact_texts[index], candidates[0].get("fact_text", ""))
    }
    judge_candidates = {
        index: candidates
        for index, candidates in filtered_candidates.items()
        if index not in duplicates
    }

    deprecations: dict[int, list[str]] = {}
    if judge_candidates:
        deprecations = _judge_batch(
            chat_model,
            new_fact_texts,
            judge_candidates,
            config,
        )

    return [
        ConflictResult(
            ids_to_deprecate=deprecations.get(index, []),
            embedding=embedding,
            duplicate_of=duplicates.get(index),
        )
        for index, embedding in enumerate(embeddings)
    ]


def _find_exact_duplicate(new_fact_text: str, candidates: list[dict]) -> dict | None:
    for candidate in candidates:
        if _same_fact_text(new_fact_text, candidate.get("fact_text", "")):
            return candidate
    return None


def _same_fact_text(left: str, right: str) -> bool:
    return " ".join(left.split()).casefold() == " ".join(str(right).split()).casefold()


def _filter_candidates(candidates: list[dict], min_candidate_score: float) -> list[dict]:
    filtered = []
    for candidate in candidates:
        score = _candidate_score(candidate)
        if score >= min_candidate_score:
            enriched = dict(candidate)
            enriched["score"] = score
            filtered.append(enriched)
    return filtered


def _candidate_score(candidate: dict) -> float:
    distance = float(candidate.get("distance", 0.0) or 0.0)
    return 1.0 / (1.0 + max(distance, 0.0))


def _embed_texts(embedding_model: EmbeddingModelProtocol, texts: list[str]) -> list[list[float]]:
    response = embedding_model.embed(EmbeddingRequest(texts=texts))
    return response.vectors


def _judge(
    chat_model: ChatModelProtocol,
    new_fact_text: str,
    candidates: list[dict],
    config: HetaConfig,
) -> list[str]:
    candidate_lines = "\n".join(
        f'- id: "{c["memory_id"]}"  fact: "{c["fact_text"]}"'
        for c in candidates
    )
    user_msg = f'New fact: "{new_fact_text}"\n\nExisting facts:\n{candidate_lines}'

    response = chat_model.complete(
        ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=CONFLICT_JUDGE_PROMPT),
                ChatMessage(role="user", content=user_msg),
            ],
            options=ChatModelOptions(temperature=0.0),
        )
    )
    raw = (response.message.content or "").strip()
    return _parse_judge_response(raw)


def _judge_batch(
    chat_model: ChatModelProtocol,
    new_fact_texts: list[str],
    candidates_by_index: dict[int, list[dict]],
    config: HetaConfig,
) -> dict[int, list[str]]:
    user_msg = _batch_user_message(new_fact_texts, candidates_by_index)
    response = chat_model.complete(
        ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=BATCH_CONFLICT_JUDGE_PROMPT),
                ChatMessage(role="user", content=user_msg),
            ],
            options=ChatModelOptions(temperature=0.0),
        )
    )
    raw = (response.message.content or "").strip()
    return _parse_batch_judge_response(raw)


def _batch_user_message(new_fact_texts: list[str], candidates_by_index: dict[int, list[dict]]) -> str:
    blocks: list[str] = []
    for index in sorted(candidates_by_index):
        candidate_lines = "\n".join(
            f'- id: "{candidate["memory_id"]}"  score: {candidate.get("score", 0):.3f}  fact: "{candidate["fact_text"]}"'
            for candidate in candidates_by_index[index]
        )
        blocks.append(
            f'New fact index: {index}\n'
            f'New fact: "{new_fact_texts[index]}"\n'
            f'Existing candidate facts:\n{candidate_lines}'
        )
    return "\n\n".join(blocks)


def _parse_judge_response(raw: str) -> list[str]:
    text = _strip_json_fence(raw)
    try:
        data = json.loads(text)
        result = data.get("deprecate", [])
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Failed to parse conflict judge response: %s", raw[:200])
        return []


def _parse_batch_judge_response(raw: str) -> dict[int, list[str]]:
    text = _strip_json_fence(raw)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Failed to parse batch conflict judge response: %s", raw[:200])
        return {}

    result = data.get("deprecate", [])
    if not isinstance(result, list):
        return {}

    parsed: dict[int, list[str]] = {}
    for item in result:
        if not isinstance(item, dict):
            continue
        index = item.get("new_fact_index")
        memory_ids = item.get("memory_ids", [])
        if not isinstance(index, int) or not isinstance(memory_ids, list):
            continue
        parsed[index] = [memory_id for memory_id in memory_ids if isinstance(memory_id, str)]
    return parsed


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return text
