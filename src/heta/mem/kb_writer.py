"""Extract and store KB insights into memory."""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

from heta.config.schema import HetaConfig
from heta.mem import meta_store
from heta.mem.client import build_client, build_embedding_client, extra_body
from heta.mem.db import get_connection, init_db
from heta.mem.embedder import embed_text
from heta.mem.kb_store import insert_insight_embedding, insert_kb_insight, search_kb_insights
from heta.mem.models import KBInsight, MemoryMeta
from heta.mem.paths import db_path, ensure_mem_dir
from heta.mem.prompts import INSIGHT_DEDUP_PROMPT, KB_INSIGHT_EXTRACTION_PROMPT
from heta.query.tools import read_page

logger = logging.getLogger(__name__)

# Only invoke the LLM dedup check when semantic similarity is this high.
# Below the threshold the candidate is almost certainly a new insight.
_DEDUP_SIMILARITY_THRESHOLD = 0.7
_DEDUP_TOP_K = 5


def remember_kb_insights(
    question: str,
    sources,                       # list[QuerySource] — already validated to "used" by the KB agent
    config: HetaConfig,
    base_dir: Path | None = None,
) -> int:
    """Distil KB page content into insights and store them. Returns number of insights written."""
    if not sources:
        return 0

    ensure_mem_dir()
    conn = get_connection(db_path(), with_vec=True)
    init_db(conn)

    llm_client, llm_model = build_client(config)
    emb_client, emb_model = build_embedding_client(config)
    now = int(time.time())
    total = 0

    for qs in sources:
        path = qs.path
        page_content = read_page(path, base_dir)
        if page_content.startswith("error:"):
            logger.warning("skip insight extraction for %s: %s", path, page_content)
            continue

        wiki_id = qs.wiki_id
        heading_path = qs.heading_path

        insights = _extract_insights(llm_client, llm_model, question, page_content, config)
        if not insights:
            logger.info("no insights extracted from %s", path)
            continue

        for insight_text in insights:
            # Embed first so we can run the dedup check before writing.
            emb = embed_text(emb_client, emb_model, insight_text)

            similar = search_kb_insights(conn, emb, top_k=_DEDUP_TOP_K)
            if similar and similar[0]["score"] >= _DEDUP_SIMILARITY_THRESHOLD:
                if _is_duplicate(llm_client, llm_model, insight_text, similar, config):
                    logger.debug("skip duplicate insight: %.80s", insight_text)
                    continue

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
                insight=insight_text,
                question=question,
                source_path=path,
                wiki_id=wiki_id,
                heading_path=heading_path,
                created_at=now,
            )
            meta_store.insert_meta(conn, meta)
            insert_kb_insight(conn, insight)
            insert_insight_embedding(conn, memory_id, emb)
            total += 1

    conn.commit()
    conn.close()
    return total


def _is_duplicate(
    client,
    model: str,
    insight_text: str,
    similar: list[dict],
    config: HetaConfig,
) -> bool:
    """Ask the LLM whether insight_text is already covered by any of the similar insights."""
    existing_block = "\n".join(
        f"[{i + 1}] {s['insight']}" for i, s in enumerate(similar)
    )
    user_msg = f"New insight:\n{insight_text}\n\nExisting similar insights:\n{existing_block}"
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": INSIGHT_DEDUP_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
    }
    body = extra_body(config)
    if body:
        kwargs["extra_body"] = body
    try:
        raw = (client.chat.completions.create(**kwargs).choices[0].message.content or "").strip()
        data = _parse_json(raw)
        return bool(data.get("duplicate", False))
    except Exception:
        logger.warning("dedup check failed, defaulting to store: %.80s", insight_text)
        return False


def _extract_insights(client, model: str, question: str, page_content: str, config: HetaConfig) -> list[str]:
    user_msg = f"Question:\n{question}\n\nKB page content:\n{page_content}"
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": KB_INSIGHT_EXTRACTION_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.2,
    }
    body = extra_body(config)
    if body:
        kwargs["extra_body"] = body
    try:
        response = client.chat.completions.create(**kwargs)
        raw = (response.choices[0].message.content or "").strip()
        return _parse_insights(raw)
    except Exception:
        logger.exception("insight extraction failed for question: %.80s", question)
        return []


def _parse_insights(raw: str) -> list[str]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(text)
        items = data.get("insights", [])
        return [s for s in items if isinstance(s, str) and s.strip()]
    except (json.JSONDecodeError, AttributeError):
        logger.warning("failed to parse insights response: %s", raw[:200])
        return []


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        return {}
