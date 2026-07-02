"""Static wiki page generation for `heta insert`."""

from __future__ import annotations

import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

from heta.config.schema import HetaConfig
from heta.kb.models import FileChange, ParsedDocument
from heta.kb.text import slugify
from heta.providers.clients import build_chat_model
from heta.providers.model_protocols import (
    ChatCompletionRequest,
    ChatMessage,
    ChatModelOptions,
    ChatModelRequestError,
)

SUMMARY_MAX_CHARS = 12000
SUMMARY_MAX_TOKENS = 512
SUMMARY_RETRIES = 3

SUMMARY_PROMPT = """Write a concise Little Heta wiki Summary for one parsed source document.
Return only the summary paragraph, normally 1-3 sentences.
Do not use Markdown headings or bullets.
Mention the main object/topic, document purpose, and important identifiers if visible.
"""


def write_static_page(
    *,
    root_dir: Path,
    document: ParsedDocument,
    config: HetaConfig,
) -> dict[str, Any]:
    """Write exactly one static wiki page for a parsed document."""
    pages = root_dir / "pages"
    pages.mkdir(parents=True, exist_ok=True)

    summary = generate_summary(document=document, config=config)
    page_name = _available_page_name(pages, document.title)
    page_rel = f"pages/{page_name}"
    page = _build_page(document=document, summary=summary)
    (pages / page_name).write_text(page, encoding="utf-8")
    _append_index_entry(root_dir / "index.md", document.title, page_rel, summary)
    _append_log(root_dir / "log.md", f"Created static page: {document.title} from {document.source_name}")

    change = FileChange("added", document.title, page_rel)
    return {"added": [change], "updated": [], "deleted": []}


def generate_summary(*, document: ParsedDocument, config: HetaConfig) -> str:
    chat_model = build_chat_model(config, timeout=300, max_retries=2)
    prompt = _summary_user_prompt(document)
    last_exc: Exception | None = None
    for attempt in range(1, SUMMARY_RETRIES + 1):
        try:
            response = chat_model.complete(
                ChatCompletionRequest(
                    messages=[
                        ChatMessage(role="system", content=SUMMARY_PROMPT),
                        ChatMessage(role="user", content=prompt),
                    ],
                    options=ChatModelOptions(
                        temperature=0.1,
                        max_output_tokens=SUMMARY_MAX_TOKENS,
                    ),
                )
            )
            summary = _normalize_summary(response.message.content or "")
            if summary:
                return summary
            last_exc = RuntimeError("LLM returned an empty summary.")
        except ChatModelRequestError as exc:
            last_exc = exc
        if attempt < SUMMARY_RETRIES:
            time.sleep(min(2**attempt, 20))
    assert last_exc is not None
    raise last_exc


def normalize_content_for_static_page(document: ParsedDocument) -> str:
    text = _normalize_model_markdown(document.markdown_content)
    lines: list[str] = []
    has_level3 = False
    in_code = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            lines.append(line)
            continue
        if not in_code and (match := re.match(r"^(#{1,6})\s+(.*)$", line)):
            level = len(match.group(1))
            title = match.group(2).strip()
            new_level = max(3, level + 2)
            if new_level >= 3:
                has_level3 = True
            line = f"{'#' * min(new_level, 6)} {title}"
        lines.append(line)

    content = "\n".join(lines).strip()
    if not content:
        content = "No content."
    if not has_level3:
        content = f"### {document.title}\n\n{content}"
    return content


def _summary_user_prompt(document: ParsedDocument) -> str:
    sample = document.markdown_content[:SUMMARY_MAX_CHARS]
    return f"""Title: {document.title}
Source file: {document.source_name}

Parsed markdown excerpt:
```markdown
{sample}
```
"""


def _build_page(*, document: ParsedDocument, summary: str) -> str:
    content = normalize_content_for_static_page(document)
    today = date.today().isoformat()
    return (
        "---\n"
        f"title: {document.title}\n"
        f"sources: [{document.source_name}]\n"
        f"updated: {today}\n"
        "---\n\n"
        "## Summary\n\n"
        f"{summary.strip() or document.title}\n\n"
        "## Content\n\n"
        f"{content.strip()}\n\n"
        "## Related Pages\n\n"
        "- None yet\n\n"
        "## Source\n\n"
        f"- {document.source_name}\n"
    )


def _available_page_name(pages: Path, title: str) -> str:
    next_id = _next_wiki_id(pages)
    slug = slugify(title)
    for wiki_id in range(next_id, next_id + 10000):
        candidate = f"{wiki_id}-{slug}.md"
        if not (pages / candidate).exists():
            return candidate
    raise RuntimeError(f"Too many wiki pages while creating: {title}")


def _next_wiki_id(pages: Path) -> int:
    ids = []
    for page in pages.glob("*.md"):
        match = re.match(r"^(\d+)-.+\.md$", page.name)
        if match:
            ids.append(int(match.group(1)))
    return max(ids, default=0) + 1


def _append_index_entry(index_path: Path, title: str, page_rel: str, summary: str) -> None:
    index = index_path.read_text(encoding="utf-8") if index_path.exists() else "# Wiki Index\n"
    wiki_id = _wiki_id_from_page_rel(page_rel)
    prefix = f"- [{wiki_id}] " if wiki_id is not None else "- "
    entry = f"{prefix}[[{title}]] ({page_rel}) — {summary.strip() or title}"
    index_path.write_text(index.rstrip() + "\n" + entry + "\n", encoding="utf-8")


def _append_log(log_path: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else "# Wiki Log\n\n"
    log_path.write_text(existing.rstrip() + f"\n- [{timestamp}] {message}\n", encoding="utf-8")


def _wiki_id_from_page_rel(page_rel: str) -> int | None:
    match = re.match(r"^pages/(\d+)-", page_rel)
    return int(match.group(1)) if match else None


def _normalize_summary(markdown: str) -> str:
    text = _normalize_model_markdown(markdown)
    text = re.sub(r"^#+\s*", "", text).strip()
    text = " ".join(line.strip().lstrip("-*").strip() for line in text.splitlines() if line.strip())
    return re.sub(r"\s+", " ", text).strip()


def _normalize_model_markdown(markdown: str) -> str:
    text = markdown.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:markdown)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


__all__ = ["normalize_content_for_static_page", "write_static_page"]
