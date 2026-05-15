"""Typed models for read-only wiki query."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VectorMatch:
    wiki_id: int
    page_name: str
    path: str
    chunk_id: str
    heading_path: str
    content: str
    score: float


@dataclass(frozen=True)
class QuerySource:
    wiki_id: int | None
    title: str
    path: str
    heading_path: str | None = None


@dataclass(frozen=True)
class QueryInsight:
    """A distilled knowledge nugget emitted by the KB agent alongside its answer."""
    text: str
    source_paths: list[str]


@dataclass(frozen=True)
class QueryResult:
    answer: str
    sources: list[QuerySource] = field(default_factory=list)
    insights: list[QueryInsight] = field(default_factory=list)
    usage: dict | None = None

