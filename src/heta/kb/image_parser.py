"""Image parsing for Little Heta KB inserts."""

from __future__ import annotations

import base64
import json
from datetime import date
from pathlib import Path
from typing import Any

from heta.config.schema import HetaConfig
from heta.kb.agent import _chat_completion, _get_client

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def parse_image_markdown(source_path: Path, archived_path: Path, config: HetaConfig) -> str:
    """Describe an image with a VLM and return stable wiki-flavored Markdown."""
    description = describe_image(source_path=source_path, config=config)
    return build_image_markdown(
        title=f"Image - {source_path.stem}",
        source_name=archived_path.name,
        image_path=f"../../raw/{archived_path.name}",
        summary=description["summary"],
        visual_facts=description["visual_facts"],
        visible_text=description["visible_text"],
        interpretation_keywords=description["interpretation_keywords"],
    )


def describe_image(*, source_path: Path, config: HetaConfig) -> dict[str, str]:
    client, model = _get_client(config)
    response = _chat_completion(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": _image_system_prompt()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _image_user_prompt(source_path.name)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _data_url(source_path),
                        },
                    },
                ],
            },
        ],
        tools=None,
        temperature=0.1,
        config=config,
    )
    raw = response.choices[0].message.content or ""
    return _normalize_description(_extract_json_object(raw))


def build_image_markdown(
    *,
    title: str,
    source_name: str,
    image_path: str,
    summary: str,
    visual_facts: str,
    visible_text: str,
    interpretation_keywords: str,
) -> str:
    return (
        "---\n"
        f"title: {title}\n"
        f"sources: [{source_name}]\n"
        f"updated: {date.today().isoformat()}\n"
        "---\n\n"
        "## Summary\n\n"
        f"{summary.strip()}\n\n"
        "## Content\n\n"
        f"![{source_name}](<{image_path}>)\n\n"
        "### Visual Facts\n\n"
        f"{visual_facts.strip()}\n\n"
        "### Visible Text\n\n"
        f"{visible_text.strip() or 'None detected.'}\n\n"
        "### Interpretation and Keywords\n\n"
        f"{interpretation_keywords.strip()}\n\n"
        "## Related Pages\n\n"
        "- None yet\n\n"
        "## Source\n\n"
        f"- {source_name}\n"
    )


def _image_system_prompt() -> str:
    return """You are an image-to-Markdown parser for Little Heta KB inserts.
Return only one valid JSON object. Do not wrap it in Markdown fences.
Be detailed, factual, and efficient. Do not invent hidden context.
If visible text exists, transcribe it faithfully. If there is no visible text,
write "None detected."."""


def _image_user_prompt(filename: str) -> str:
    return f"""Describe this image for semantic retrieval.

Filename: {filename}

Return JSON with exactly these string fields:
- summary: one concise paragraph describing what the image is and why it matters.
- visual_facts: detailed factual description of scene type, main subject, objects, people, layout, colors, labels, numbers, and spatial relations.
- visible_text: visible text transcription, or "None detected."
- interpretation_keywords: likely meaning or purpose with uncertainty if needed, ending with compact search keywords.
"""


def _data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = _MIME_TYPES.get(suffix)
    if mime is None:
        raise ValueError(f"Unsupported image type: {suffix}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Image model did not return JSON.")
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Image model JSON must be an object.")
    return value


def _normalize_description(data: dict[str, Any]) -> dict[str, str]:
    fields = {
        "summary": "Imported image.",
        "visual_facts": "No visual facts extracted.",
        "visible_text": "None detected.",
        "interpretation_keywords": "Image; visual document.",
    }
    normalized: dict[str, str] = {}
    for key, fallback in fields.items():
        value = data.get(key)
        normalized[key] = str(value).strip() if value else fallback
    return normalized


__all__ = ["IMAGE_EXTENSIONS", "build_image_markdown", "describe_image", "parse_image_markdown"]
