"""Clean Little Heta knowledge base content."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from heta.kb import paths
from heta.kb.store import append_log, commit_wiki, ensure_wiki_layout


@dataclass(frozen=True)
class CleanSummary:
    deleted_pages: int
    deleted_vector_files: int
    commit_id: str | None
    invalidated_memories: int = 0


def clean_knowledge_base(*, base_dir: Path | None = None) -> CleanSummary:
    """Clear wiki knowledge and vector database while preserving raw files."""
    ensure_wiki_layout(base_dir)

    deleted_pages = _clear_pages(base_dir)
    _reset_index(base_dir)
    append_log("Cleaned Little Heta knowledge base.", paths.wiki_dir(base_dir))
    deleted_vector_files = _clear_vector_db(base_dir)
    commit_id = commit_wiki("chore: clean wiki knowledge base", base_dir)

    from heta.mem.kb_invalidate import invalidate_all
    invalidated = invalidate_all()

    return CleanSummary(
        deleted_pages=deleted_pages,
        deleted_vector_files=deleted_vector_files,
        commit_id=commit_id,
        invalidated_memories=invalidated,
    )


def _clear_pages(base_dir: Path | None) -> int:
    pages = paths.pages_dir(base_dir)
    pages.mkdir(parents=True, exist_ok=True)
    deleted = 0
    for page in pages.glob("*.md"):
        if page.is_file():
            page.unlink()
            deleted += 1
    return deleted


def _reset_index(base_dir: Path | None) -> None:
    paths.index_path(base_dir).write_text("# Wiki Index\n\n", encoding="utf-8")


def _clear_vector_db(base_dir: Path | None) -> int:
    db_path = paths.vector_db_path(base_dir)
    if not db_path.exists():
        return 0
    if db_path.is_file():
        db_path.unlink()
        return 1

    deleted = 0
    for child in db_path.rglob("*"):
        if child.is_file():
            child.unlink()
            deleted += 1
    return deleted


__all__ = ["CleanSummary", "clean_knowledge_base"]
