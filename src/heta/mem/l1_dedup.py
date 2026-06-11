"""Semantic duplicate detection for L1 episodic memories."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from heta.config.schema import HetaConfig
from heta.mem.client import EMBEDDING_DIM, extra_body
from heta.mem.l1_search import search_episodes
from heta.mem.prompts import EPISODE_DEDUP_PROMPT

logger = logging.getLogger(__name__)

MIN_EPISODE_DUP_CANDIDATE_SCORE = 0.72


@dataclass(frozen=True)
class EpisodeDedupResult:
    duplicate_of: str | None
    embedding: list[float]


def detect_episode_duplicates_batch(
    conn: Any,
    new_episode_summaries: list[str],
    llm_client: OpenAI,
    llm_model: str,
    emb_client: OpenAI,
    emb_model: str,
    config: HetaConfig,
    top_k: int = 5,
    min_candidate_score: float = MIN_EPISODE_DUP_CANDIDATE_SCORE,
) -> list[EpisodeDedupResult]:
    """Return duplicate decisions for new episode summaries."""
    if not new_episode_summaries:
        return []

    embeddings = _embed_texts(emb_client, emb_model, new_episode_summaries)
    candidates_by_index: dict[int, list[dict]] = {}

    for index, embedding in enumerate(embeddings):
        candidates = search_episodes(conn, embedding, top_k=top_k)
        candidates = [c for c in candidates if float(c.get("score", 0.0) or 0.0) >= min_candidate_score]
        if candidates:
            candidates_by_index[index] = candidates

    duplicates: dict[int, str] = {}
    if candidates_by_index:
        duplicates = _judge_batch(
            llm_client,
            llm_model,
            new_episode_summaries,
            candidates_by_index,
            config,
        )

    return [
        EpisodeDedupResult(
            duplicate_of=duplicates.get(index),
            embedding=embedding,
        )
        for index, embedding in enumerate(embeddings)
    ]


def _embed_texts(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=model,
        input=texts,
        dimensions=EMBEDDING_DIM,
    )
    return [item.embedding for item in response.data]


def _judge_batch(
    client: OpenAI,
    model: str,
    new_episode_summaries: list[str],
    candidates_by_index: dict[int, list[dict]],
    config: HetaConfig,
) -> dict[int, str]:
    user_msg = _batch_user_message(new_episode_summaries, candidates_by_index)
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": EPISODE_DEDUP_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
    }
    body = extra_body(config)
    if body:
        kwargs["extra_body"] = body

    response = client.chat.completions.create(**kwargs)
    raw = (response.choices[0].message.content or "").strip()
    return _parse_judge_response(raw)


def _batch_user_message(new_episode_summaries: list[str], candidates_by_index: dict[int, list[dict]]) -> str:
    blocks: list[str] = []
    for index in sorted(candidates_by_index):
        candidate_lines = "\n".join(
            f'- id: "{candidate["memory_id"]}"  score: {candidate.get("score", 0):.3f}  summary: "{candidate["summary"]}"'
            for candidate in candidates_by_index[index]
        )
        blocks.append(
            f"New episode index: {index}\n"
            f'New episode summary: "{new_episode_summaries[index]}"\n'
            f"Existing candidate episodes:\n{candidate_lines}"
        )
    return "\n\n".join(blocks)


def _parse_judge_response(raw: str) -> dict[int, str]:
    text = _strip_json_fence(raw)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Failed to parse episode dedup judge response: %s", raw[:200])
        return {}

    result = data.get("duplicates", [])
    if not isinstance(result, list):
        return {}

    parsed: dict[int, str] = {}
    for item in result:
        if not isinstance(item, dict):
            continue
        index = item.get("new_episode_index")
        memory_id = item.get("memory_id")
        if isinstance(index, int) and isinstance(memory_id, str):
            parsed[index] = memory_id
    return parsed


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return text
