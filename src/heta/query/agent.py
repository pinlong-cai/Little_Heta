"""Read-only agent loop for Little Heta wiki query."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from heta.config.schema import HetaConfig
from heta.kb.agent import AgentStats, _chat_completion, _get_client
from heta.query.models import QueryResult, QuerySource, VectorMatch
from heta.query.tools import (
    format_vector_matches,
    read_index,
    read_page,
    read_raw,
    search_vector,
    source_from_page_path,
)

READ_PAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_page",
        "description": "Read a wiki page. Valid paths: pages/*.md.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}

READ_RAW_TOOL = {
    "type": "function",
    "function": {
        "name": "read_raw",
        "description": "Read an original raw file referenced by a wiki page. Valid paths stay under raw/.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}

SEARCH_VECTOR_TOOL = {
    "type": "function",
    "function": {
        "name": "search_vector",
        "description": "Search semantic wiki chunks. Returns wiki id, page path, heading path, content, and score.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

QUERY_TOOLS = [
    READ_PAGE_TOOL,
    READ_RAW_TOOL,
    SEARCH_VECTOR_TOOL,
]

QUERY_TOOLS_NO_VECTOR = [
    READ_PAGE_TOOL,
    READ_RAW_TOOL,
]

RAW_SNIPPET_MAX_CHARS = 16000


@dataclass(frozen=True)
class FinalAnswer:
    answer: str
    sources: list[QuerySource]


def run_query_agent(
    *,
    question: str,
    config: HetaConfig,
    base_dir: Path | None = None,
    top_k: int = 5,
    extra_context: str | None = None,
    max_steps: int = 8,
    max_seconds: int = 180,
    temperature: float = 0.2,
) -> QueryResult:
    client, model = _get_client(config)
    stats = AgentStats(task_id="query", max_steps=max_steps, max_seconds=max_seconds)
    index_text = read_index(base_dir)
    initial_matches = search_vector(question, config, top_k=top_k, base_dir=base_dir)
    vector_matches = _vector_match_map(initial_matches)
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": _initial_message(
                question=question,
                index_text=index_text,
                vector_matches=initial_matches,
                extra_context=extra_context,
            ),
        }
    ]
    read_paths: set[str] = set()
    tools = QUERY_TOOLS if config.vector_index.enable else QUERY_TOOLS_NO_VECTOR

    while stats.should_continue():
        response = _chat_completion(
            client=client,
            model=model,
            messages=[{"role": "system", "content": _system_prompt(config.vector_index.enable)}, *messages],
            tools=tools,
            temperature=temperature,
            config=config,
        )
        message = response.choices[0].message
        tool_calls = list(message.tool_calls or [])

        if not tool_calls:
            final_answer = _parse_final_answer(
                text=message.content or "",
                read_paths=read_paths,
                vector_matches=vector_matches,
                base_dir=base_dir,
            )
            stats.record_completion(response.usage)
            return QueryResult(
                answer=final_answer.answer,
                sources=final_answer.sources,
                usage=stats.finish("completed"),
            )

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
        messages.extend(_execute_tools(tool_calls, config, base_dir, top_k, read_paths, vector_matches))
        stats.record(", ".join(tool.function.name for tool in tool_calls), response.usage)

    messages.append(
        {
            "role": "user",
            "content": "You reached the step or time limit. Do not call tools. Answer with the evidence already available.",
        }
    )
    final = _chat_completion(
        client=client,
        model=model,
        messages=[{"role": "system", "content": _system_prompt(config.vector_index.enable)}, *messages],
        tools=None,
        temperature=temperature,
        config=config,
    )
    stats.record_completion(final.usage)
    final_answer = _parse_final_answer(
        text=final.choices[0].message.content or "",
        read_paths=read_paths,
        vector_matches=vector_matches,
        base_dir=base_dir,
    )
    return QueryResult(
        answer=final_answer.answer,
        sources=final_answer.sources,
        usage=stats.finish("stopped at limit"),
    )


def _system_prompt(vector_enabled: bool) -> str:
    vector_rule = (
        "- You may call search_vector again with a refined query if the current evidence is insufficient."
        if vector_enabled
        else "- Vector search is disabled; rely on the index and pages you read."
    )
    return f"""You are Little Heta's read-only wiki query agent.

Answer the user's question using the Little Heta wiki. You can inspect the wiki,
but you must not create, edit, delete, rename, or commit anything.

Rules:
- Treat index.md as the global map of pages, ids, paths, and summaries.
- Treat semantic matches as starting evidence, not final truth.
- If a chunk is relevant but incomplete, call read_page(path) for the full page.
- You may call read_raw(path) only for original raw files referenced by wiki pages.
  Raw files help inspect details, but raw files must never appear in used_sources.
- Follow useful [[Wiki Links]] by reading the linked pages when the index gives their paths.
{vector_rule}
- Stop reading when the context is enough.
- If the wiki does not contain enough evidence, say what is missing.
- Your final response must be exactly one valid JSON object, with no Markdown fence:
  {{"answer": "Markdown answer text", "used_sources": [{{"path": "pages/example.md", "heading_path": "Content > Section"}}]}}
- In used_sources, include only evidence you actually used. You may cite a semantic
  match directly when its chunk is sufficient; use its exact path and heading.
- Do not include a Sources, References, or Citations section in answer.
  The CLI renders validated sources separately.
"""


def _initial_message(
    *,
    question: str,
    index_text: str,
    vector_matches: list[VectorMatch],
    extra_context: str | None,
) -> str:
    parts = [
        f"Current date: {datetime.now().date().isoformat()}",
        f"Question:\n{question}",
        f"Wiki Index:\n{index_text or '(index.md is missing or empty)'}",
        f"Semantic Matches:\n{format_vector_matches(vector_matches)}",
    ]
    if extra_context:
        parts.append(f"Extra Context:\n{extra_context}")
    return "\n\n".join(parts)


def _execute_tools(
    tool_calls: list[Any],
    config: HetaConfig,
    base_dir: Path | None,
    default_top_k: int,
    read_paths: set[str],
    vector_matches: dict[tuple[str, str], VectorMatch],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            output = f"error: invalid tool arguments: {exc}"
        else:
            if name == "read_page":
                path = str(arguments.get("path", ""))
                output = read_page(path, base_dir)
                if not output.startswith("error:"):
                    read_paths.add(path.replace("\\", "/").strip("/"))
            elif name == "read_raw":
                path = str(arguments.get("path", ""))
                output = _trim_raw_output(read_raw(path, base_dir))
            elif name == "search_vector":
                query = str(arguments.get("query", ""))
                top_k = int(arguments.get("top_k") or default_top_k)
                matches = search_vector(query, config, top_k=top_k, base_dir=base_dir)
                vector_matches.update(_vector_match_map(matches))
                output = format_vector_matches(matches)
            else:
                output = f"error: unknown tool {name}"
        results.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})
    return results


def _trim_raw_output(output: str) -> str:
    if output.startswith("error:") or len(output) <= RAW_SNIPPET_MAX_CHARS:
        return output
    return output[:RAW_SNIPPET_MAX_CHARS] + "\n\n[truncated raw output]"


def _vector_match_map(matches: list[VectorMatch]) -> dict[tuple[str, str], VectorMatch]:
    return {(_normalize_candidate_path(match.path), match.heading_path): match for match in matches}


def _parse_final_answer(
    *,
    text: str,
    read_paths: set[str],
    vector_matches: dict[tuple[str, str], VectorMatch],
    base_dir: Path | None,
) -> FinalAnswer:
    data = _extract_json_object(text)
    if data is None:
        return FinalAnswer(answer=text, sources=[])

    answer = data.get("answer")
    used_sources = data.get("used_sources")
    if not isinstance(answer, str):
        answer = text
    if not isinstance(used_sources, list):
        used_sources = []

    sources: dict[str, QuerySource] = {}
    normalized_read_paths = {_normalize_candidate_path(path) for path in read_paths}
    for source in used_sources:
        if not isinstance(source, dict):
            continue
        raw_path = source.get("path")
        if not isinstance(raw_path, str):
            continue
        try:
            path = _normalize_candidate_path(raw_path)
        except ValueError:
            continue
        heading = source.get("heading_path")
        heading_path = str(heading).strip() if heading else ""
        key = (path, heading_path)

        if path in normalized_read_paths:
            display_heading = heading_path or None
        elif key in vector_matches:
            display_heading = heading_path
        else:
            continue
        sources[f"{path}#{display_heading or ''}"] = source_from_page_path(path, base_dir, heading_path=display_heading)
    return FinalAnswer(answer=answer, sources=list(sources.values()))


def _normalize_candidate_path(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match is None:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None
