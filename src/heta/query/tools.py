"""Read-only wiki query tools."""

from __future__ import annotations

import re
from pathlib import Path

from heta.config.schema import HetaConfig
from heta.kb import paths
from heta.kb.vector_index import search_wiki_vector_index
from heta.query.models import QuerySource, VectorMatch

PAGE_ID_RE = re.compile(r"^(?P<wiki_id>\d+)-.+\.md$")


def read_index(base_dir: Path | None = None) -> str:
    index = paths.index_path(base_dir)
    if not index.exists():
        return ""
    return index.read_text(encoding="utf-8")


def read_page(path: str, base_dir: Path | None = None) -> str:
    try:
        normalized = normalize_page_path(path)
        full = _resolve_safe(paths.wiki_dir(base_dir), normalized)
        if not full.exists():
            return f"error: {normalized} does not exist"
        return full.read_text(encoding="utf-8")
    except Exception as exc:
        return f"error: {exc}"


def read_raw(path: str, base_dir: Path | None = None) -> str:
    try:
        normalized = normalize_raw_path(path)
        full = _resolve_safe(paths.raw_dir(base_dir), normalized)
        if not full.exists():
            return f"error: raw/{normalized} does not exist"
        if not full.is_file():
            return f"error: raw/{normalized} is not a file"
        return full.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"error: {exc}"


def search_vector(
    query: str,
    config: HetaConfig,
    *,
    top_k: int = 5,
    base_dir: Path | None = None,
) -> list[VectorMatch]:
    if not config.vector_index.enable:
        return []
    return [
        VectorMatch(
            wiki_id=match.wiki_id,
            page_name=match.page_name,
            path=f"pages/{match.page_name}",
            chunk_id=match.chunk_id,
            heading_path=match.heading_path,
            content=match.content,
            score=match.score,
        )
        for match in search_wiki_vector_index(query=query, config=config, top_k=top_k, base_dir=base_dir)
    ]


def source_from_page_path(path: str, base_dir: Path | None = None, heading_path: str | None = None) -> QuerySource:
    normalized = normalize_page_path(path)
    full = paths.wiki_dir(base_dir) / normalized
    text = full.read_text(encoding="utf-8") if full.exists() else ""
    title = _frontmatter_value(text, "title") or Path(normalized).stem
    return QuerySource(
        wiki_id=wiki_id_from_page_name(Path(normalized).name),
        title=title,
        path=normalized,
        heading_path=heading_path,
    )


def normalize_page_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip("/")
    if normalized.startswith("pages/") and normalized.endswith(".md"):
        return normalized
    raise ValueError(f"path must be pages/*.md, got: {path!r}")


def normalize_raw_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    if "/raw/" in normalized:
        normalized = normalized.split("/raw/", 1)[1]
    elif normalized.startswith("raw/"):
        normalized = normalized[4:]
    normalized = normalized.strip("/")
    if not normalized or normalized.startswith("../") or "/../" in normalized:
        raise ValueError(f"path must stay within raw/, got: {path!r}")
    return normalized


def wiki_id_from_page_name(page_name: str) -> int | None:
    match = PAGE_ID_RE.match(page_name)
    if match is None:
        return None
    return int(match.group("wiki_id"))


def format_vector_matches(matches: list[VectorMatch]) -> str:
    if not matches:
        return "None."

    parts: list[str] = []
    for index, match in enumerate(matches, start=1):
        parts.append(
            "\n".join(
                [
                    f"[{index}]",
                    f"wiki_id: {match.wiki_id}",
                    f"page: {match.page_name}",
                    f"path: {match.path}",
                    f"chunk_id: {match.chunk_id}",
                    f"heading: {match.heading_path}",
                    f"score: {match.score:.3f}",
                    "content:",
                    match.content,
                ]
            )
        )
    return "\n\n".join(parts)


def _resolve_safe(root_dir: Path, normalized: str) -> Path:
    root = root_dir.resolve()
    candidate = (root_dir / normalized).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path escapes wiki directory: {normalized}")
    return candidate


def _frontmatter_value(text: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None
