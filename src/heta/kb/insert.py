"""Transactional `heta insert` implementation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from heta.config.schema import HetaConfig
from heta.kb.discovery import collect_insert_files
from heta.kb.agent import run_merge_agent
from heta.kb.models import InsertResult, ParsedDocument
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
) -> InsertResult:
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
        added = []
        updated = []
        deleted = []
        for index, document in enumerate(parsed_documents, start=1):
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
            validate_wiki(working_wiki)
            added.extend(apply_path_map(agent_result["added"], normalize_result.path_map))
            updated.extend(apply_path_map(agent_result["updated"], normalize_result.path_map))
            deleted.extend(apply_path_map(agent_result["deleted"], normalize_result.path_map))

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

        return InsertResult(
            commit_id=commit_id,
            added=added,
            updated=updated,
            deleted=deleted,
            raw_files=raw_files,
            planned_pdf_parts=sum(plan.parts for plan in pdf_plans if plan.enabled),
        )
    except BaseException:
        for raw in raw_files:
            if raw.exists():
                raw.unlink()
        cleanup_working_copy(task_id, base_dir)
        reset_wiki(base_dir)
        raise
