"""Transactional `heta insert` implementation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from heta.config.schema import HetaConfig
from heta.kb.agent import run_merge_agent
from heta.kb.code_parser import CODE_EXTENSIONS
from heta.kb.discovery import collect_insert_files
from heta.kb.models import FileChange, InsertProgress, InsertResult, ParsedDocument
from heta.kb.parser import parse_document
from heta.kb.pdf_plan import plan_insert_files
from heta.kb.store import commit_wiki, ensure_wiki_layout, reset_wiki
from heta.kb.vector_index import sync_wiki_vector_index
from heta.kb.wiki import apply_path_map, normalize_wiki_pages, repair_broken_wiki_links, validate_wiki
from heta.kb.workspace import cleanup_working_copy, create_working_copy, promote_working_copy


def insert_paths(
    targets: list[Path],
    config: HetaConfig,
    *,
    base_dir: Path | None = None,
    enable_pdf_planning: bool = True,
    on_progress: Callable[[InsertProgress], None] | None = None,
) -> InsertResult:
    _emit_progress(on_progress, "prepare", 1, 0, 0, "preparing files")
    files = collect_insert_files(targets, config)
    if not files:
        raise ValueError("No supported files found.")

    task_id = f"insert_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    raw_files: list[Path] = []

    ensure_wiki_layout(base_dir)

    try:
        prepared_sources, pdf_plans = plan_insert_files(
            files,
            enable_pdf_planning=enable_pdf_planning,
            config=config,
            base_dir=base_dir,
        )
        raw_files.extend(source.archived_path for source in prepared_sources)
        raw_files.extend(
            source.metadata_path
            for source in prepared_sources
            if source.metadata_path is not None and source.metadata_path not in raw_files
        )
        raw_files.extend(
            source.original_path
            for source in prepared_sources
            if source.original_path is not None and source.original_path not in raw_files
        )
        parsed_documents: list[ParsedDocument] = []
        for source in prepared_sources:
            parsed_documents.append(parse_document(source.source_path, source.archived_path, config))

        working_wiki = create_working_copy(task_id, base_dir)
        total_documents = len(parsed_documents)
        _emit_progress(on_progress, "merge", 1, 0, total_documents, "ready to merge documents")
        added = []
        updated = []
        deleted = []
        for index, document in enumerate(parsed_documents, start=1):
            _emit_progress(
                on_progress,
                "merge",
                _merge_percent(index - 1, total_documents),
                index - 1,
                total_documents,
                document.source_name,
            )
            agent_result = run_merge_agent(
                task_id=f"{task_id}_{index}",
                documents=[document],
                root_dir=working_wiki,
                config=config,
            )
            if not (agent_result["added"] or agent_result["updated"] or agent_result["deleted"]):
                raise RuntimeError(f"Agent completed without changing the wiki for: {document.source_name}")
            normalize_result = normalize_wiki_pages(working_wiki)
            repair_broken_wiki_links(working_wiki)
            normalized_added = apply_path_map(agent_result["added"], normalize_result.path_map)
            normalized_updated = apply_path_map(agent_result["updated"], normalize_result.path_map)
            normalized_deleted = apply_path_map(agent_result["deleted"], normalize_result.path_map)
            _ensure_code_raw_links(working_wiki, document, [*normalized_added, *normalized_updated])
            validate_wiki(working_wiki)
            added.extend(normalized_added)
            updated.extend(normalized_updated)
            deleted.extend(normalized_deleted)
            _emit_progress(
                on_progress,
                "merge",
                _merge_percent(index, total_documents),
                index,
                total_documents,
                document.source_name,
            )

        _emit_progress(
            on_progress,
            "finalize",
            99,
            total_documents,
            total_documents,
            "finalizing wiki and vector index",
        )
        promote_working_copy(task_id, base_dir)
        commit_id = commit_wiki(f"ingest: {', '.join(file.name for file in files)}", base_dir)
        if config.vector_index.enable:
            try:
                sync_wiki_vector_index(
                    changes=[*added, *updated, *deleted],
                    config=config,
                    base_dir=base_dir,
                )
            except Exception:
                pass
        cleanup_working_copy(task_id, base_dir)

        from heta.mem.kb_invalidate import invalidate_by_paths
        invalidated = invalidate_by_paths(c.path for c in (*updated, *deleted))

        _emit_progress(on_progress, "done", 100, total_documents, total_documents, "insert completed")

        return InsertResult(
            commit_id=commit_id,
            added=added,
            updated=updated,
            deleted=deleted,
            raw_files=raw_files,
            planned_pdf_parts=sum(plan.parts for plan in pdf_plans if plan.enabled),
            invalidated_memories=invalidated,
        )
    except BaseException:
        for raw in raw_files:
            if raw.exists():
                raw.unlink()
        cleanup_working_copy(task_id, base_dir)
        reset_wiki(base_dir)
        raise


def _merge_percent(done: int, total: int) -> int:
    if total <= 0:
        return 99
    return min(99, 1 + int(done / total * 98))


def _ensure_code_raw_links(wiki_root: Path, document: ParsedDocument, changes: list[FileChange]) -> None:
    if document.metadata.get("extension") not in CODE_EXTENSIONS:
        return
    raw_link = f"[Raw source](<../../raw/{document.source_name}>)"
    for change in changes:
        if not change.path.startswith("pages/") or not change.path.endswith(".md"):
            continue
        page = wiki_root / change.path
        if not page.exists():
            continue
        text = page.read_text(encoding="utf-8")
        if raw_link in text:
            continue
        if "## Content" not in text:
            continue
        updated = text.replace("## Content\n", f"## Content\n\n{raw_link}\n", 1)
        page.write_text(updated, encoding="utf-8")


def _emit_progress(
    callback: Callable[[InsertProgress], None] | None,
    phase: str,
    percent: int,
    current: int,
    total: int,
    label: str,
) -> None:
    if callback is None:
        return
    callback(
        InsertProgress(
            phase=phase,
            percent=max(0, min(100, percent)),
            current=current,
            total=total,
            label=label,
        )
    )
