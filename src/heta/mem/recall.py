"""Orchestrator for the heta recall pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from heta.config.schema import HetaConfig
from heta.mem.client import build_chat_model, build_embedding_model
from heta.mem.db import get_connection, init_db
from heta.mem.embedder import embed_text
from heta.mem.kb_store import search_kb_insights
from heta.mem.l0_search import search_turns
from heta.mem.l1_search import search_episodes
from heta.mem.l2_store import search_similar_facts
from heta.mem.paths import db_path, ensure_mem_dir
from heta.mem.prompts import RECALL_ANSWER_PROMPT, RECALL_RANKING_PROMPT
from heta.providers.model_protocols import ChatCompletionRequest, ChatMessage, ChatModelOptions, ChatModelProtocol

logger = logging.getLogger(__name__)


def _open_conn_and_embed(query: str, config: HetaConfig):
    ensure_mem_dir()
    conn = get_connection(db_path(), with_vec=True)
    init_db(conn)
    embedding_model = build_embedding_model(config)
    embedding = embed_text(embedding_model, query)
    return conn, embedding


@dataclass
class LayerEvidence:
    layer: str          # raw / episode / atomic_fact
    items: list[dict] = field(default_factory=list)


@dataclass
class RecallResult:
    query: str
    ranking: list[str]
    answer: str
    reason: str
    evidence: list[LayerEvidence]
    sufficient: bool = False


def retrieve_evidence(query: str, config: HetaConfig, top_k: int = 5) -> list[LayerEvidence]:
    """Pure retrieval — no LLM calls. Used by smart_query to inject context into the KB agent."""
    conn, query_embedding = _open_conn_and_embed(query, config)
    l0_hits = search_turns(conn, query, top_k=top_k)
    l1_hits = search_episodes(conn, query_embedding, top_k=top_k)
    l2_hits = search_similar_facts(conn, query_embedding, top_k=top_k)
    kb_insight_hits = search_kb_insights(conn, query_embedding, top_k=top_k)
    conn.close()
    return [
        LayerEvidence(layer="raw", items=l0_hits),
        LayerEvidence(layer="episode", items=l1_hits),
        LayerEvidence(layer="atomic_fact", items=l2_hits),
        LayerEvidence(layer="kb_insight", items=kb_insight_hits),
    ]


def recall(query: str, config: HetaConfig, top_k: int = 10) -> RecallResult:
    conn, query_embedding = _open_conn_and_embed(query, config)
    chat_model = build_chat_model(config)

    l0_hits = search_turns(conn, query, top_k=top_k)
    l1_hits = search_episodes(conn, query_embedding, top_k=top_k)
    l2_hits = search_similar_facts(conn, query_embedding, top_k=top_k)
    kb_insight_hits = search_kb_insights(conn, query_embedding, top_k=top_k)
    conn.close()

    evidence = [
        LayerEvidence(layer="raw", items=l0_hits),
        LayerEvidence(layer="episode", items=l1_hits),
        LayerEvidence(layer="atomic_fact", items=l2_hits),
        LayerEvidence(layer="kb_insight", items=kb_insight_hits),
    ]

    ranking, answer, reason, sufficient = _rank(
        query=query,
        evidence=evidence,
        chat_model=chat_model,
    )

    return RecallResult(
        query=query,
        ranking=ranking,
        answer=answer,
        reason=reason,
        evidence=evidence,
        sufficient=sufficient,
    )


def _rank(
    query: str,
    evidence: list[LayerEvidence],
    chat_model: ChatModelProtocol,
) -> tuple[list[str], str, str, bool]:
    """Two-phase: rank layers first, then generate a strictly grounded answer."""
    evidence_text = format_evidence(evidence)

    # Phase A: rank layers (no answer generation)
    ranking, reason = _rank_layers(
        query=query,
        evidence_text=evidence_text,
        chat_model=chat_model,
    )

    # Phase B: generate grounded answer from the top-ranked useful layers.
    answer_evidence = _select_ranked_evidence(evidence, ranking, max_layers=2)
    answer, sufficient = _generate_grounded_answer(
        query=query,
        evidence_text=format_evidence(answer_evidence),
        chat_model=chat_model,
    )

    return ranking, answer, reason, sufficient


def _select_ranked_evidence(
    evidence: list[LayerEvidence],
    ranking: list[str],
    *,
    max_layers: int = 2,
) -> list[LayerEvidence]:
    """Return the top-ranked non-empty layers for answer generation."""
    by_layer = {layer_ev.layer: layer_ev for layer_ev in evidence}
    selected: list[LayerEvidence] = []
    seen: set[str] = set()

    for layer in ranking:
        layer_ev = by_layer.get(layer)
        if layer_ev is None or not layer_ev.items or layer in seen:
            continue
        selected.append(layer_ev)
        seen.add(layer)
        if len(selected) >= max(1, max_layers):
            return selected

    if selected:
        return selected

    # If the ranker fails or returns only empty/unknown layers, preserve prior behavior.
    return evidence


def _rank_layers(
    query: str,
    evidence_text: str,
    chat_model: ChatModelProtocol,
) -> tuple[list[str], str]:
    try:
        response = chat_model.complete(
            ChatCompletionRequest(
                messages=[
                    ChatMessage(role="system", content=RECALL_RANKING_PROMPT),
                    ChatMessage(role="user", content=f"Question:\n{query}\n\nEvidence:\n{evidence_text}"),
                ],
                options=ChatModelOptions(temperature=0.1),
            )
        )
        raw = (response.message.content or "").strip()
        data = _parse_json(raw)
        return data.get("ranking", []), data.get("reason", "")
    except Exception:
        logger.warning("ranking call failed", exc_info=True)
        return [], ""


def _generate_grounded_answer(
    query: str,
    evidence_text: str,
    chat_model: ChatModelProtocol,
) -> tuple[str, bool]:
    try:
        response = chat_model.complete(
            ChatCompletionRequest(
                messages=[
                    ChatMessage(role="system", content=RECALL_ANSWER_PROMPT),
                    ChatMessage(role="user", content=f"Question:\n{query}\n\nEvidence:\n{evidence_text}"),
                ],
                options=ChatModelOptions(temperature=0.2),
            )
        )
        raw = (response.message.content or "").strip()
        data = _parse_json(raw)
        answer = data.get("answer", "")
        sufficient = bool(data.get("sufficient", False))
        if answer == "[INSUFFICIENT]" or not sufficient:
            return "", False
        return answer, True
    except Exception:
        logger.warning("answer generation call failed", exc_info=True)
        return "", False



def format_evidence(evidence: list[LayerEvidence]) -> str:
    parts = []
    for layer_ev in evidence:
        parts.append(f"## {layer_ev.layer}")
        if not layer_ev.items:
            parts.append("(no results)")
        else:
            for i, item in enumerate(layer_ev.items, 1):
                score = item.get("score", 0)
                if layer_ev.layer == "raw":
                    parts.append(f"[{i}; score={score:.4f}] {item['text_content']}")
                elif layer_ev.layer == "episode":
                    parts.append(f"[{i}; score={score:.4f}] {item['summary']}")
                elif layer_ev.layer == "kb_insight":
                    src = item.get("source_path", "")
                    parts.append(f"[{i}; score={score:.4f}] [{src}] {item.get('insight', '')}")
                else:
                    parts.append(f"[{i}; score={score:.4f}] {item['fact_text']}")
    return "\n".join(parts)


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Failed to parse LLM JSON response: %s", raw[:200])
        return {}
