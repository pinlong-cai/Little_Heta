"""LLM-based semantic fact extraction."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from heta.config.schema import HetaConfig
from heta.mem.prompts import FACT_EXTRACTION_PROMPT
from heta.providers.model_protocols import ChatCompletionRequest, ChatMessage, ChatModelOptions, ChatModelProtocol

logger = logging.getLogger(__name__)


def extract_facts(
    chat_model: ChatModelProtocol,
    text: str,
    config: HetaConfig,
    session_ts: int | None = None,
) -> list[dict[str, Any]]:
    """Call the LLM and return a list of raw fact dicts."""
    anchor_date = _fmt_date(session_ts)
    user_content = f"Anchor date: {anchor_date}\n\nText:\n{text}"

    response = chat_model.complete(
        ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=FACT_EXTRACTION_PROMPT),
                ChatMessage(role="user", content=user_content),
            ],
            options=ChatModelOptions(temperature=0.2),
        )
    )
    raw = response.message.content or ""
    return _parse_facts(raw)


def _fmt_date(ts: int | None) -> str:
    if ts is None:
        return datetime.now().strftime("%Y-%m-%d")
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _parse_facts(raw: str) -> list[dict[str, Any]]:
    for text in _json_candidates(raw):
        try:
            data = json.loads(text)
            facts = data.get("facts", [])
            if not isinstance(facts, list):
                return []
            return [
                f for f in facts
                if isinstance(f, dict) and all(k in f for k in ("subject", "predicate", "object"))
            ]
        except (json.JSONDecodeError, AttributeError):
            continue

    logger.warning("Failed to parse fact extraction response: %s", raw[:200])
    return []


def _json_candidates(raw: str) -> list[str]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text = text.strip()

    candidates = [text]
    if text.startswith("{{"):
        candidates.append(text[1:])
        if text.endswith("}}"):
            candidates.append(text[1:-1])
    return candidates
