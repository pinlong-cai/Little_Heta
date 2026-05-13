"""LLM agent loop for merging documents into the Little Heta Wiki."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from openai import OpenAI

from heta.config.schema import HetaConfig
from heta.kb.models import FileChange, ParsedDocument
from heta.kb.text import slugify
from heta.kb.wiki import detect_wiki_changes

logger = logging.getLogger(__name__)

FAST_AGENT_MODELS = {
    "qwen": "qwen3.5-flash",
    "chatgpt": "gpt-5.4-nano",
    "gemini": "gemini-2.5-flash",
}

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Read a wiki file from the working copy. Valid paths: index.md, log.md, or pages/*.md.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_page",
            "description": "Create a new wiki page. Valid paths: pages/*.md only. Fails if the file already exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_page",
            "description": "Edit a wiki file by replacing an exact string. Valid paths: index.md or pages/*.md.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                },
                "required": ["path", "old_str", "new_str"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_page",
            "description": "Delete a wiki page. Valid paths: pages/*.md only. Cannot delete index.md or log.md.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_log",
            "description": "Append a one-line message to log.md. Timestamp formatting is added automatically.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass
class AgentStats:
    task_id: str
    max_steps: int = 20
    max_seconds: int = 600
    steps: int = 0
    tokens: int = 0
    start: float = field(default_factory=time.time)
    events: list[str] = field(default_factory=list)

    def should_continue(self) -> bool:
        if self.steps >= self.max_steps:
            self.log("aborted: max steps reached")
            return False
        if time.time() - self.start > self.max_seconds:
            self.log("aborted: timeout")
            return False
        return True

    def record(self, tool_names: str, usage: Any) -> None:
        self.steps += 1
        self._add_usage(usage)
        self.log(f"step {self.steps} | {tool_names}")

    def record_completion(self, usage: Any) -> None:
        self._add_usage(usage)

    def finish(self, summary: str) -> dict[str, Any]:
        elapsed = int(time.time() - self.start)
        self.log(f"done | {summary} | {self.tokens} tokens | {elapsed}s")
        return {
            "steps": self.steps,
            "tokens": self.tokens,
            "elapsed_s": elapsed,
            "events": self.events,
        }

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.events.append(f"[{timestamp}] {message}")
        logger.info("[kb-agent:%s] %s", self.task_id, message)

    def _add_usage(self, usage: Any) -> None:
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        self.tokens += prompt_tokens + completion_tokens


def run_merge_agent(
    *,
    task_id: str,
    documents: list[ParsedDocument],
    root_dir: Path,
    config: HetaConfig,
    max_steps: int = 20,
    max_seconds: int = 600,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Run a real tool-calling LLM agent against a wiki working copy."""
    before = _snapshot_pages(root_dir)
    client, model = _get_client(config)
    stats = AgentStats(task_id=task_id, max_steps=max_steps, max_seconds=max_seconds)
    messages: list[dict[str, Any]] = [{"role": "user", "content": _initial_message(documents, root_dir)}]
    written_paths: set[str] = set()
    read_paths: set[str] = set()

    while stats.should_continue():
        response = _chat_completion(
            client=client,
            model=model,
            messages=[{"role": "system", "content": _system_prompt()}, *messages],
            tools=AGENT_TOOLS,
            temperature=temperature,
            config=config,
        )
        message = response.choices[0].message
        tool_calls = list(message.tool_calls or [])

        if not tool_calls:
            stats.record_completion(response.usage)
            final_text = message.content or "completed"
            changes = detect_wiki_changes(root_dir, before)
            return {
                "final_response": final_text,
                "written_paths": sorted(written_paths),
                "read_paths": sorted(read_paths),
                "usage": stats.finish("completed"),
                "added": changes.added,
                "updated": changes.updated,
                "deleted": changes.deleted,
            }

        assistant_message: dict[str, Any] = {"role": "assistant"}
        if message.content:
            assistant_message["content"] = message.content
        assistant_message["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in tool_calls
        ]
        messages.append(assistant_message)

        tool_results = _execute_tools(root_dir, tool_calls, written_paths, read_paths)
        messages.extend(tool_results)
        stats.record(", ".join(tool.function.name for tool in tool_calls), response.usage)

    messages.append(
        {
            "role": "user",
            "content": "You reached the step/time limit. Do not call tools. Summarize what was completed.",
        }
    )
    final = _chat_completion(
        client=client,
        model=model,
        messages=[{"role": "system", "content": _system_prompt()}, *messages],
        tools=None,
        temperature=temperature,
        config=config,
    )
    stats.record_completion(final.usage)
    changes = detect_wiki_changes(root_dir, before)
    return {
        "final_response": final.choices[0].message.content or "completed",
        "written_paths": sorted(written_paths),
        "read_paths": sorted(read_paths),
        "usage": stats.finish("stopped at limit"),
        "added": changes.added,
        "updated": changes.updated,
        "deleted": changes.deleted,
    }


def _get_client(config: HetaConfig) -> tuple[OpenAI, str]:
    # API keys are intentionally read only from ~/.heta/heta.yaml, which is
    # created by `heta init`. Model choice stays fixed to fast defaults here.
    provider = config.llm.provider
    if provider == "qwen":
        return (
            OpenAI(
                api_key=config.llm.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=300,
            ),
            FAST_AGENT_MODELS["qwen"],
        )
    if provider == "chatgpt":
        return OpenAI(api_key=config.llm.api_key, timeout=300), FAST_AGENT_MODELS["chatgpt"]
    if provider == "gemini":
        return (
            OpenAI(
                api_key=config.llm.api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=300,
            ),
            FAST_AGENT_MODELS["gemini"],
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _extra_body(config: HetaConfig) -> dict[str, Any] | None:
    if config.llm.provider == "qwen":
        return {"enable_thinking": False}
    return None


def _chat_completion(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    temperature: float,
    config: HetaConfig,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools is not None:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    extra_body = _extra_body(config)
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    return client.chat.completions.create(**kwargs)


def _system_prompt() -> str:
    return """You are running Little Heta KB merge ingest.

Your job is to absorb parsed source documents into the Markdown wiki working copy.
You must use tools to inspect and edit files. Do not claim completion until the
working copy contains the final wiki changes.
In normal ingest, you receive one source document at a time. Treat that document
as the current unit of truth and preserve its concrete details in the wiki.

Required workflow:
1. read_page("index.md") to understand the current wiki.
2. Identify up to 5 related pages from index.md for each source document.
3. read_page each related page before deciding whether it is genuinely related.
4. For each source document, either create one complete new page or edit one
   existing page that already covers the same topic. When editing an existing
   page, add or extend a source-specific section unless the current source is
   genuinely duplicate.
5. Update index.md with one entry per created or substantially updated page.
   The entry must use exactly this format:
   - [id] [[Title]] (pages/file-name.md) — one-line summary
   If a page does not have a numeric filename prefix yet, omit the id and use
   the semantic path you created. The system will assign stable numeric
   filename prefixes and normalize index.md after you finish.
6. Maintain bidirectional [[Wiki Links]] only when the relationship is real.
7. append_log with a concise summary of created, updated, linked, or deleted pages.

Page format:
---
title: Title
sources: [source_filename]
updated: YYYY-MM-DD
---

## Summary
One paragraph.

## Content
Full self-contained content. Preserve the source document's definitions,
examples, procedures, named entities, important lists, formulas, constraints,
and concrete facts. Do not replace the source with a high-level summary.

## Related Pages
- [[Related Title]]

## Source
- source_filename

If there are no related pages, write "- None yet".

Rules:
- Paths are limited to index.md, log.md, and pages/*.md.
- One source document becomes one complete wiki page unless it clearly belongs in an existing page.
- Every source document must be represented in page content and in the Source list.
- Merge overlapping sources only when they describe the same thing; even then,
  keep new details from the current source.
- Do not discard details just because they seem minor or because the page already
  has a summary.
- Do not invent or maintain wiki ids, chunk ids, or numeric page prefixes.
- Keep [[Wiki Links]] semantic, e.g. [[HetaGen]], never [[1-HetaGen]].
- Always read a page before editing it.
- Use exact old_str when calling edit_page.
- Keep log.md append-only.
- Every page must include frontmatter fields: title, sources, updated.
- index.md must include every created page with its pages/*.md path and summary.
- Do not leave broken [[Wiki Links]].
"""


def _initial_message(documents: list[ParsedDocument], root_dir: Path) -> str:
    parts = [
        f"Current date: {datetime.now().date().isoformat()}",
        f"Current index.md:\n{(root_dir / 'index.md').read_text(encoding='utf-8')}",
    ]
    for index, document in enumerate(documents, start=1):
        parts.append(
            "\n".join(
                [
                    f"Source document {index}:",
                    f"source_filename: {document.source_name}",
                    f"suggested_title: {document.title}",
                    "parsed_markdown:",
                    document.markdown_content,
                ]
            )
        )
    return "\n\n".join(parts)


def _execute_tools(
    root_dir: Path,
    tool_calls: list[Any],
    written_paths: set[str],
    read_paths: set[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            output = f"error: invalid tool arguments: {exc}"
        else:
            try:
                if name == "read_page":
                    output = read_page(root_dir, **arguments)
                    if not output.startswith("error:"):
                        read_paths.add(_normalize_path(arguments.get("path", "")))
                elif name == "create_page":
                    output = create_page(root_dir, written_paths=written_paths, **arguments)
                elif name == "edit_page":
                    output = edit_page(root_dir, written_paths=written_paths, **arguments)
                elif name == "delete_page":
                    output = delete_page(root_dir, written_paths=written_paths, **arguments)
                elif name == "append_log":
                    output = append_log(root_dir, **arguments)
                else:
                    output = f"error: unknown tool {name}"
            except TypeError as exc:
                output = f"error: invalid tool arguments for {name}: {exc}"
            except Exception as exc:
                output = f"error: tool {name} failed: {exc}"

        results.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})
    return results


def read_page(root_dir: Path, path: str) -> str:
    try:
        normalized = _validate_read_path(path)
        full = _resolve_safe(root_dir, normalized)
        if not full.exists():
            return f"error: {normalized} does not exist"
        return full.read_text(encoding="utf-8")
    except Exception as exc:
        return f"error: {exc}"


def create_page(root_dir: Path, path: str, content: str, written_paths: set[str]) -> str:
    try:
        _validate_pages_only(path)
        title = _frontmatter_title(content)
        normalized = f"pages/{slugify(title)}.md" if title else _validate_pages_only(path)
        full = _resolve_safe(root_dir, normalized)
        if full.exists():
            return f"error: {normalized} already exists; use edit_page"
        full.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(full, content)
        written_paths.add(normalized)
        return f"ok: created {normalized} ({len(content)} chars)"
    except Exception as exc:
        return f"error: {exc}"


def edit_page(root_dir: Path, path: str, old_str: str, new_str: str, written_paths: set[str]) -> str:
    try:
        normalized = _validate_edit_path(path)
        full = _resolve_safe(root_dir, normalized)
        if not full.exists():
            return f"error: {normalized} does not exist"
        content = full.read_text(encoding="utf-8")
        count = content.count(old_str)
        if count == 0:
            return "error: old_str not found"
        if count > 1:
            return f"error: old_str matches {count} locations; provide more context"
        _atomic_write(full, content.replace(old_str, new_str, 1))
        written_paths.add(normalized)
        return f"ok: edited {normalized}"
    except Exception as exc:
        return f"error: {exc}"


def delete_page(root_dir: Path, path: str, written_paths: set[str]) -> str:
    try:
        normalized = _validate_pages_only(path)
        full = _resolve_safe(root_dir, normalized)
        if not full.exists():
            return f"error: {normalized} does not exist"
        full.unlink()
        written_paths.add(normalized)
        return f"ok: deleted {normalized}"
    except Exception as exc:
        return f"error: {exc}"


def append_log(root_dir: Path, message: str) -> str:
    try:
        log_path = root_dir / "log.md"
        if not log_path.exists():
            return "error: log.md does not exist"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with log_path.open("a", encoding="utf-8") as file:
            file.write(f"- [{timestamp}] {message}\n")
        return "ok: appended to log.md"
    except Exception as exc:
        return f"error: {exc}"


def _snapshot_pages(root_dir: Path) -> dict[str, str]:
    pages = root_dir / "pages"
    if not pages.exists():
        return {}
    return {
        f"pages/{page.name}": page.read_text(encoding="utf-8")
        for page in sorted(pages.glob("*.md"))
    }


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def _validate_pages_only(path: str) -> str:
    normalized = _normalize_path(path)
    if normalized.startswith("pages/") and normalized.endswith(".md"):
        return normalized
    raise ValueError(f"path must be pages/*.md, got: {path!r}")


def _validate_read_path(path: str) -> str:
    normalized = _normalize_path(path)
    if normalized in {"index.md", "log.md"}:
        return normalized
    return _validate_pages_only(path)


def _validate_edit_path(path: str) -> str:
    normalized = _normalize_path(path)
    if normalized == "index.md":
        return normalized
    return _validate_pages_only(path)


def _resolve_safe(root_dir: Path, normalized: str) -> Path:
    root = root_dir.resolve()
    candidate = (root_dir / normalized).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path escapes working copy: {normalized}")
    return candidate


def _atomic_write(path: Path, content: str) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as temp:
        temp.write(content)
        temp_path = Path(temp.name)
    temp_path.replace(path)


def _frontmatter_title(content: str) -> str | None:
    for line in content.splitlines():
        if line.startswith("title:"):
            return line.split(":", 1)[1].strip()
    return None
