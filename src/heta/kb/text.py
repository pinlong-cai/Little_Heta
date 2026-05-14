"""Text utilities for Little Heta KB."""

from __future__ import annotations

import re
from datetime import date


def slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "untitled"


def extract_title(markdown: str, fallback: str) -> str:
    in_frontmatter = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter and stripped.startswith("title:"):
            title = stripped.split(":", 1)[1].strip()
            if title:
                return title
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
    return fallback


def frontmatter_page(title: str, source_name: str, summary: str, content: str) -> str:
    today = date.today().isoformat()
    body = content.strip() or "No content."
    return (
        "---\n"
        f"title: {title}\n"
        f"sources: [{source_name}]\n"
        f"updated: {today}\n"
        "---\n\n"
        "## Summary\n\n"
        f"{summary.strip() or title}\n\n"
        "## Content\n\n"
        f"{body}\n\n"
        "## Related Pages\n\n"
        "- None yet\n\n"
        "## Source\n\n"
        f"- {source_name}\n"
    )


def summarize(markdown: str, *, max_chars: int = 180) -> str:
    lines: list[str] = []
    in_frontmatter = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter or not stripped:
            continue
        if stripped.startswith("#"):
            continue
        stripped = stripped.lstrip("-*").strip()
        if stripped:
            lines.append(stripped)
    compact = " ".join(lines)
    if not compact:
        return "Imported knowledge page."
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."
