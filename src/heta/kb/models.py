"""Data models for Little Heta knowledge ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ParsedDocument:
    source_path: Path
    archived_path: Path
    title: str
    markdown_content: str
    source_name: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FileChange:
    kind: str
    title: str
    path: str


@dataclass(frozen=True)
class InsertResult:
    commit_id: str | None
    added: list[FileChange]
    updated: list[FileChange]
    deleted: list[FileChange]
    raw_files: list[Path]
    planned_pdf_parts: int = 0
