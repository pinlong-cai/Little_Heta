"""Store agent-distilled kb_insights into memory, with dedup."""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

from heta.config.schema import HetaConfig
from heta.mem import meta_store
from heta.mem.client import build_chat_model, build_embedding_model
from heta.mem.db import get_connection, init_db
from heta.mem.embedder import embed_text
from heta.mem.kb_store import insert_insight_embedding, insert_kb_insight, search_kb_insights
from heta.mem.models import KBInsight, MemoryMeta
from heta.mem.paths import db_path, ensure_mem_dir
from heta.mem.prompts import INSIGHT_DEDUP_PROMPT
from heta.providers.model_protocols import ChatCompletionRequest, ChatMessage, ChatModelOptions, ChatModelProtocol
from heta.query.models import QueryInsight, QuerySource

logger = logging.getLogger(__name__)

# Below this similarity score the candidate is almost certainly a new insight,
# so we skip the LLM dedup call entirely.
_DEDUP_SIMILARITY_THRESHOLD = 0.7
_DEDUP_TOP_K = 5


def remember_kb_insights(
    question: str,
    insights: list[QueryInsight],
    sources: list[QuerySource],
    config: HetaConfig,
    base_dir: Path | None = None,
) -> int:
    """Persist agent-distilled insights into memory. Returns count stored after dedup."""
    if not insights:
        return 0

    ensure_mem_dir()
    conn = get_connection(db_path(), with_vec=True)
    init_db(conn)

    chat_model = build_chat_model(config)
    embedding_model = build_embedding_model(config)
    now = int(time.time())

    # Build a path → QuerySource map so insights can inherit wiki_id / heading
    # from the primary source.
    source_index = {s.path: s for s in sources}
    total = 0

    for qi in insights:
        text = qi.text.strip()
        if not text:
            continue

        emb = embed_text(embedding_model, text)
        similar = search_kb_insights(conn, emb, top_k=_DEDUP_TOP_K)
        if similar and similar[0]["score"] >= _DEDUP_SIMILARITY_THRESHOLD:
            if _is_duplicate(chat_model, text, similar):
                logger.debug("skip duplicate insight: %.80s", text)
                continue

        primary_path = qi.source_paths[0] if qi.source_paths else ""
        primary = source_index.get(primary_path)
        wiki_id = primary.wiki_id if primary else None
        heading_path = primary.heading_path if primary else None

        memory_id = str(uuid.uuid4())
        meta = MemoryMeta(
            memory_id=memory_id,
            memory_type="kb_insight",
            session_id=None,
            origin="kb_insight",
            kb_uid=str(wiki_id) if wiki_id is not None else None,
            created_at=now,
            last_access_at=now,
        )
        insight = KBInsight(
            memory_id=memory_id,
            insight=text,
            source_paths=list(qi.source_paths),
            created_at=now,
            question=question,
            wiki_id=wiki_id,
            heading_path=heading_path,
        )
        meta_store.insert_meta(conn, meta)
        insert_kb_insight(conn, insight)
        insert_insight_embedding(conn, memory_id, emb)
        total += 1

    conn.commit()
    conn.close()
    return total


def _is_duplicate(
    chat_model: ChatModelProtocol,
    insight_text: str,
    similar: list[dict],
) -> bool:
    """Ask the LLM whether the new insight is fully covered by the similar set."""
    existing_block = "\n".join(
        f"[{i + 1}] {s['insight']}" for i, s in enumerate(similar)
    )
    user_msg = f"NEW insight:\n{insight_text}\n\nEXISTING similar insights:\n{existing_block}"
    try:
        response = chat_model.complete(
            ChatCompletionRequest(
                messages=[
                    ChatMessage(role="system", content=INSIGHT_DEDUP_PROMPT),
                    ChatMessage(role="user", content=user_msg),
                ],
                options=ChatModelOptions(temperature=0.0),
            )
        )
        raw = (response.message.content or "").strip()
        data = _parse_json(raw)
        return bool(data.get("duplicate", False))
    except Exception:
        logger.warning("dedup check failed, defaulting to store: %.80s", insight_text)
        return False


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        return {}
