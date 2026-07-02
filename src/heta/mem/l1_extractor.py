"""LLM-based episodic memory extraction."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from heta.config.schema import HetaConfig
from heta.mem.prompts import EPISODE_EXTRACTION_PROMPT
from heta.providers.model_protocols import ChatCompletionRequest, ChatMessage, ChatModelOptions, ChatModelProtocol

logger = logging.getLogger(__name__)


def extract_episodes(
    chat_model: ChatModelProtocol,
    text: str,
    config: HetaConfig,
    session_ts: int | None = None,
) -> list[dict[str, Any]]:
    """Call the LLM and return a list of raw episode dicts."""
    anchor_date = _fmt_date(session_ts)
    user_content = f"Anchor date: {anchor_date}\n\nText:\n{text}"

    response = chat_model.complete(
        ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=EPISODE_EXTRACTION_PROMPT),
                ChatMessage(role="user", content=user_content),
            ],
            options=ChatModelOptions(temperature=0.2),
        )
    )
    raw = response.message.content or ""
    return _parse_episodes(raw)


def resolve_when_ts(when_resolved: str | None) -> int | None:
    """Parse a variable-precision date string to unix timestamp of period start.

    Accepts: YYYY-MM-DD, YYYY-Www (ISO week), YYYY-MM, YYYY
    """
    if not when_resolved:
        return None
    s = when_resolved.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            pass
    # ISO week: "2026-W21"
    try:
        dt = datetime.strptime(s + "-1", "%Y-W%W-%w")
        return int(dt.replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        pass
    return None


def _fmt_date(ts: int | None) -> str:
    if ts is None:
        return datetime.now().strftime("%Y-%m-%d")
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _parse_episodes(raw: str) -> list[dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(text)
        episodes = data.get("episodes", [])
        if not isinstance(episodes, list):
            return []
        return [e for e in episodes if isinstance(e, dict) and "what" in e]
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Failed to parse episode extraction response: %s", raw[:200])
        return []
