"""PDF page assessment and static splitting for large inserts."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from heta.kb import paths

PDF_PAGE_THRESHOLD = 80
PDF_PART_MAX_PAGES = 40


@dataclass(frozen=True)
class PreparedSource:
    source_path: Path
    archived_path: Path
    original_path: Path | None = None
    page_start: int | None = None
    page_end: int | None = None


@dataclass(frozen=True)
class PdfPlan:
    source_path: Path
    page_count: int
    enabled: bool
    parts: int


def plan_insert_files(
    files: list[Path],
    *,
    enable_pdf_planning: bool = True,
    base_dir: Path | None = None,
) -> tuple[list[PreparedSource], list[PdfPlan]]:
    prepared: list[PreparedSource] = []
    plans: list[PdfPlan] = []

    for file in files:
        if file.suffix.lower() != ".pdf":
            prepared.append(PreparedSource(source_path=file, archived_path=_save_raw_file(file, base_dir)))
            continue

        page_count = estimate_pdf_pages(file)
        should_split = enable_pdf_planning and page_count > PDF_PAGE_THRESHOLD
        if not should_split:
            prepared.append(PreparedSource(source_path=file, archived_path=_save_raw_file(file, base_dir)))
            plans.append(PdfPlan(source_path=file, page_count=page_count, enabled=False, parts=1))
            continue

        original = _save_original_pdf(file, base_dir)
        parts = split_pdf_to_raw_parts(
            source=file,
            page_count=page_count,
            original=original,
            base_dir=base_dir,
        )
        prepared.extend(parts)
        plans.append(PdfPlan(source_path=file, page_count=page_count, enabled=True, parts=len(parts)))

    return prepared, plans


def estimate_pdf_pages(path: Path) -> int:
    return len(PdfReader(str(path)).pages)


def split_pdf_to_raw_parts(
    *,
    source: Path,
    page_count: int,
    original: Path,
    base_dir: Path | None = None,
    max_pages: int = PDF_PART_MAX_PAGES,
) -> list[PreparedSource]:
    reader = PdfReader(str(source))
    raw = paths.raw_dir(base_dir)
    raw.mkdir(parents=True, exist_ok=True)

    parts: list[PreparedSource] = []
    part_index = 1
    for start in range(0, page_count, max_pages):
        end = min(start + max_pages, page_count)
        writer = PdfWriter()
        for page_index in range(start, end):
            writer.add_page(reader.pages[page_index])

        target = _available_raw_path(
            raw,
            f"{datetime.now():%Y-%m-%d_%H%M%S}_{source.stem}_part-{part_index:03d}{source.suffix}",
        )
        with target.open("wb") as file:
            writer.write(file)
        parts.append(
            PreparedSource(
                source_path=source.with_name(f"{source.stem}_part-{part_index:03d}_pages-{start + 1}-{end}{source.suffix}"),
                archived_path=target,
                original_path=original,
                page_start=start + 1,
                page_end=end,
            )
        )
        part_index += 1

    return parts


def _save_raw_file(source: Path, base_dir: Path | None) -> Path:
    raw = paths.raw_dir(base_dir)
    raw.mkdir(parents=True, exist_ok=True)
    target = _available_raw_path(raw, f"{datetime.now():%Y-%m-%d_%H%M%S}_{source.name}")
    shutil.copy2(source, target)
    return target


def _save_original_pdf(source: Path, base_dir: Path | None) -> Path:
    originals = paths.raw_dir(base_dir) / "originals"
    originals.mkdir(parents=True, exist_ok=True)
    target = _available_raw_path(originals, f"{datetime.now():%Y-%m-%d_%H%M%S}_{source.name}")
    shutil.copy2(source, target)
    return target


def _available_raw_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    for index in range(1, 1000):
        candidate = directory / f"{stem}({index}){suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Too many raw files with the same name: {filename}")


__all__ = [
    "PDF_PAGE_THRESHOLD",
    "PDF_PART_MAX_PAGES",
    "PdfPlan",
    "PreparedSource",
    "estimate_pdf_pages",
    "plan_insert_files",
    "split_pdf_to_raw_parts",
]
