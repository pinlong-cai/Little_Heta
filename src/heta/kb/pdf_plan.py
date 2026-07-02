"""PDF page assessment, lightweight profiling, and splitting for large inserts."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter

from heta.config.schema import HetaConfig
from heta.kb import paths
from heta.kb.agent import _chat_completion, _get_chat_model
from heta.kb.text import slugify

PDF_PAGE_THRESHOLD = 80
PDF_PART_MAX_PAGES = 20
PDF_PROFILE_MAX_CHARS = 60000


@dataclass(frozen=True)
class PreparedSource:
    source_path: Path
    archived_path: Path
    original_path: Path | None = None
    page_start: int | None = None
    page_end: int | None = None
    metadata_path: Path | None = None
    original_name: str | None = None


@dataclass(frozen=True)
class PdfPlan:
    source_path: Path
    page_count: int
    enabled: bool
    parts: int
    document_type: str = "unknown"
    split_strategy: str = "none"


@dataclass(frozen=True)
class PdfProfile:
    filename: str
    page_count: int
    metadata: dict[str, str]
    outline: list[dict[str, Any]]
    page_samples: list[dict[str, Any]]
    heading_candidates: list[dict[str, Any]]


@dataclass(frozen=True)
class SplitUnit:
    title: str
    start_page: int
    end_page: int


def plan_insert_files(
    files: list[Path],
    *,
    enable_pdf_planning: bool = True,
    config: HetaConfig | None = None,
    base_dir: Path | None = None,
) -> tuple[list[PreparedSource], list[PdfPlan]]:
    prepared: list[PreparedSource] = []
    plans: list[PdfPlan] = []

    for file in files:
        if file.suffix.lower() != ".pdf":
            prepared.append(
                PreparedSource(
                    source_path=file,
                    archived_path=_save_raw_file(file, base_dir),
                    original_name=file.name,
                )
            )
            continue

        page_count = estimate_pdf_pages(file)
        should_split = enable_pdf_planning and page_count > PDF_PAGE_THRESHOLD
        if not should_split:
            prepared.append(
                PreparedSource(
                    source_path=file,
                    archived_path=_save_raw_file(file, base_dir),
                    original_name=file.name,
                )
            )
            plans.append(PdfPlan(source_path=file, page_count=page_count, enabled=False, parts=1))
            continue

        original = _save_original_pdf(file, base_dir)
        plan = plan_pdf_split(file, page_count=page_count, config=config)
        parts = split_pdf_to_raw_parts(
            source=file,
            page_count=page_count,
            original=original,
            units=plan["units"],
            document_type=plan["document_type"],
            split_strategy=plan["split_strategy"],
            base_dir=base_dir,
        )
        prepared.extend(parts)
        plans.append(
            PdfPlan(
                source_path=file,
                page_count=page_count,
                enabled=True,
                parts=len(parts),
                document_type=plan["document_type"],
                split_strategy=plan["split_strategy"],
            )
        )

    return prepared, plans


def estimate_pdf_pages(path: Path) -> int:
    return len(PdfReader(str(path)).pages)


def build_pdf_profile(path: Path, *, page_count: int | None = None) -> PdfProfile:
    reader = PdfReader(str(path))
    total_pages = page_count or len(reader.pages)
    outline = _extract_outline(reader)
    sample_pages = _sample_page_numbers(total_pages)
    samples: list[dict[str, Any]] = []
    heading_candidates: list[dict[str, Any]] = []

    for page_number in sample_pages:
        text = _page_text(reader, page_number)
        samples.append(
            {
                "page": page_number,
                "text": _truncate(text, 900),
            }
        )
        heading_candidates.extend(_heading_candidates(text, page_number))

    return PdfProfile(
        filename=path.name,
        page_count=total_pages,
        metadata=_metadata(reader),
        outline=outline,
        page_samples=samples,
        heading_candidates=heading_candidates[:80],
    )


def plan_pdf_split(
    path: Path,
    *,
    page_count: int,
    config: HetaConfig | None,
    max_pages: int = PDF_PART_MAX_PAGES,
) -> dict[str, Any]:
    profile = build_pdf_profile(path, page_count=page_count)
    fallback = _fallback_plan(page_count, max_pages=max_pages)
    if config is None:
        return fallback

    try:
        proposed = run_pdf_planning_agent(profile, config=config)
        return _validate_plan(proposed, page_count=page_count, max_pages=max_pages)
    except Exception:
        return fallback


def run_pdf_planning_agent(profile: PdfProfile, *, config: HetaConfig) -> dict[str, Any]:
    chat_model = _get_chat_model(config)
    response = _chat_completion(
        chat_model=chat_model,
        messages=[
            {"role": "system", "content": _planning_system_prompt()},
            {"role": "user", "content": _planning_user_prompt(profile)},
        ],
        tools=None,
        temperature=0.1,
        config=config,
    )
    content = response.message.content or ""
    return _extract_json_object(content)


def split_pdf_to_raw_parts(
    *,
    source: Path,
    page_count: int,
    original: Path,
    units: list[SplitUnit] | None = None,
    document_type: str = "unknown",
    split_strategy: str = "fixed_page_window",
    base_dir: Path | None = None,
    max_pages: int = PDF_PART_MAX_PAGES,
) -> list[PreparedSource]:
    reader = PdfReader(str(source))
    raw = paths.raw_dir(base_dir)
    raw.mkdir(parents=True, exist_ok=True)
    split_units = units or _fixed_units(page_count, max_pages=max_pages)

    parts: list[PreparedSource] = []
    for part_index, unit in enumerate(split_units, start=1):
        writer = PdfWriter()
        for page_index in range(unit.start_page - 1, unit.end_page):
            writer.add_page(reader.pages[page_index])

        stem = _part_stem(source, part_index, unit)
        target = _available_raw_path(
            raw,
            f"{datetime.now():%Y-%m-%d_%H%M%S}_{stem}{source.suffix}",
        )
        with target.open("wb") as file:
            writer.write(file)
        metadata = _write_part_metadata(
            target,
            original=original,
            unit=unit,
            document_type=document_type,
            split_strategy=split_strategy,
        )
        parts.append(
            PreparedSource(
                source_path=source.with_name(f"{stem}{source.suffix}"),
                archived_path=target,
                original_path=original,
                page_start=unit.start_page,
                page_end=unit.end_page,
                metadata_path=metadata,
                original_name=source.name,
            )
        )

    return parts


def _planning_system_prompt() -> str:
    return """You are Little Heta's PDF split planning agent.

You do not read the full PDF. You receive a lightweight profile containing
metadata, outline/bookmarks (titled leaf entries with real page numbers),
sampled page text, heading-like lines, and page count. Decide how to split
the PDF into smaller source units.

Each unit MUST be at most 20 pages.

Critical: do NOT propose oversized units expecting the system to "split them
later". The system's fallback splitter chops oversized units into mechanical
20-page windows that all share your title, which destroys the outline's
semantic boundaries you saw in the profile. Pick the right granularity
yourself.

Return JSON only with this shape:
{
  "document_type": "textbook | paper_collection | report | slides | manual | scanned_book | mixed",
  "split_strategy": "outline | fixed_page_window | chapter | section | fallback",
  "units": [
    {"title": "Section 1.2: Introduction", "start_page": 12, "end_page": 19}
  ]
}

Granularity rules — use the FINEST outline level whose units fit ≤20 pages:
- Compute the average page span between consecutive outline entries:
  `avg = page_count / len(outline)`.
- If `avg ≤ 20`: use outline entries as unit boundaries directly. Each unit
  spans from one entry's page to the page before the next entry's page (or
  to the document end for the last entry). Inherit the entry's title.
- If individual entries are very small (`avg < 5`): you may merge 2–4
  consecutive entries into one unit to reach a more useful size, but the
  merged title MUST reflect the range (e.g. include the first and last
  entry's identifiers, or the chapter they share). Never let the merged
  unit exceed 20 pages.
- If `avg > 20`: the outline is too coarse. Subdivide each outline entry
  into 20-page fixed windows that inherit the entry's title plus a part
  suffix (e.g. "Chapter 3 (pages 60-80)").
- For paper collections: each paper is one unit (title = paper title). If a
  paper exceeds 20 pages, subdivide that paper alone into 20-page windows.
- For slides, scanned books, or empty/unreliable outline: use fixed 20-page
  windows. Set split_strategy to "fixed_page_window".

Hard rules:
- Page numbers are 1-based and inclusive.
- Every unit MUST be ≤20 pages. A unit larger than 20 pages will be
  rejected and re-split mechanically.
- Cover [1, page_count] without gaps and without overlap.
- Do not invent details that are absent from the profile.
"""


def _planning_user_prompt(profile: PdfProfile) -> str:
    payload = asdict(profile)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    return _truncate(text, PDF_PROFILE_MAX_CHARS)


def _extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError("planning agent did not return JSON")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("planning payload is not an object")
    return payload


def _validate_plan(payload: dict[str, Any], *, page_count: int, max_pages: int) -> dict[str, Any]:
    document_type = str(payload.get("document_type") or "mixed")
    split_strategy = str(payload.get("split_strategy") or "fallback")
    raw_units = payload.get("units")
    if not isinstance(raw_units, list) or not raw_units:
        raise ValueError("planning payload has no units")

    units: list[SplitUnit] = []
    for index, raw in enumerate(raw_units, start=1):
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or f"Part {index}").strip()
        start = int(raw.get("start_page"))
        end = int(raw.get("end_page"))
        if start < 1 or end > page_count or start > end:
            raise ValueError("planning payload has invalid page range")
        units.extend(_split_oversized_unit(SplitUnit(title=title, start_page=start, end_page=end), max_pages))

    if not units:
        raise ValueError("planning payload has no valid units")
    units = _dedupe_and_sort_units(units, page_count=page_count)
    return {"document_type": document_type, "split_strategy": split_strategy, "units": units}


def _fallback_plan(page_count: int, *, max_pages: int) -> dict[str, Any]:
    return {
        "document_type": "mixed",
        "split_strategy": "fixed_page_window",
        "units": _fixed_units(page_count, max_pages=max_pages),
    }


def _fixed_units(page_count: int, *, max_pages: int) -> list[SplitUnit]:
    return [
        SplitUnit(
            title=f"Pages {start + 1}-{min(start + max_pages, page_count)}",
            start_page=start + 1,
            end_page=min(start + max_pages, page_count),
        )
        for start in range(0, page_count, max_pages)
    ]


def _split_oversized_unit(unit: SplitUnit, max_pages: int) -> list[SplitUnit]:
    if unit.end_page - unit.start_page + 1 <= max_pages:
        return [unit]

    units: list[SplitUnit] = []
    start = unit.start_page
    while start <= unit.end_page:
        end = min(start + max_pages - 1, unit.end_page)
        units.append(SplitUnit(title=unit.title, start_page=start, end_page=end))
        start = end + 1
    return units


def _dedupe_and_sort_units(units: list[SplitUnit], *, page_count: int) -> list[SplitUnit]:
    sorted_units = sorted(units, key=lambda unit: (unit.start_page, unit.end_page))
    cleaned: list[SplitUnit] = []
    cursor = 1
    for unit in sorted_units:
        if unit.start_page > cursor:
            cleaned.extend(_fixed_range_units(cursor, unit.start_page - 1, max_pages=PDF_PART_MAX_PAGES))
        start = max(unit.start_page, cursor)
        end = min(unit.end_page, page_count)
        if start <= end:
            cleaned.append(SplitUnit(title=unit.title, start_page=start, end_page=end))
            cursor = end + 1
    if cursor <= page_count:
        cleaned.extend(_fixed_range_units(cursor, page_count, max_pages=PDF_PART_MAX_PAGES))
    return cleaned


def _fixed_range_units(start_page: int, end_page: int, *, max_pages: int) -> list[SplitUnit]:
    units: list[SplitUnit] = []
    start = start_page
    while start <= end_page:
        end = min(start + max_pages - 1, end_page)
        units.append(SplitUnit(title=f"Pages {start}-{end}", start_page=start, end_page=end))
        start = end + 1
    return units


def _extract_outline(reader: PdfReader) -> list[dict[str, Any]]:
    """Collect leaf outline entries (titled bookmarks with a real page target).

    Folder/group bookmarks (e.g. collapsed parents like "正文前资料") have a null
    page destination and are useless for split planning; we skip them so they
    don't crowd out the real leaves under the OUTLINE_MAX cap.
    """
    outline: list[dict[str, Any]] = []
    OUTLINE_MAX = 500

    def visit(items: Any, depth: int = 0) -> None:
        if len(outline) >= OUTLINE_MAX:
            return
        if isinstance(items, list):
            for item in items:
                visit(item, depth)
            return
        title = getattr(items, "title", None)
        if title:
            try:
                page_number = reader.get_destination_page_number(items)
            except Exception:
                page_number = None
            if page_number is None:
                # Folder/group bookmark — keep walking siblings but do not record.
                return
            outline.append({"title": str(title), "page": page_number + 1, "depth": depth})
            return
        try:
            for child in items:
                visit(child, depth + 1)
        except TypeError:
            return

    try:
        visit(reader.outline)
    except Exception:
        return []
    return outline


def _sample_page_numbers(page_count: int) -> list[int]:
    first = list(range(1, min(4, page_count) + 1))
    if page_count <= 12:
        return first
    step = max(1, page_count // 8)
    sampled = set(first)
    sampled.update(range(1, page_count + 1, step))
    sampled.add(page_count)
    return sorted(sampled)


def _page_text(reader: PdfReader, page_number: int) -> str:
    try:
        return reader.pages[page_number - 1].extract_text() or ""
    except Exception:
        return ""


def _heading_candidates(text: str, page_number: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = re.sub(r"\s+", " ", line).strip()
        if not stripped or len(stripped) > 120:
            continue
        if _looks_like_heading(stripped):
            candidates.append({"page": page_number, "text": stripped})
    return candidates[:10]


def _looks_like_heading(line: str) -> bool:
    if re.match(r"^(\d+(\.\d+)*|chapter|section|part|appendix)\b", line, flags=re.IGNORECASE):
        return True
    words = line.split()
    return 2 <= len(words) <= 12 and line[:1].isupper() and not line.endswith(".")


def _metadata(reader: PdfReader) -> dict[str, str]:
    metadata = reader.metadata or {}
    result: dict[str, str] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        result[str(key).lstrip("/")] = _truncate(str(value), 300)
    return result


def _part_stem(source: Path, part_index: int, unit: SplitUnit) -> str:
    title = slugify(unit.title)[:48].strip("-")
    if title:
        return f"{source.stem}_part-{part_index:03d}_{title}_pages-{unit.start_page}-{unit.end_page}"
    return f"{source.stem}_part-{part_index:03d}_pages-{unit.start_page}-{unit.end_page}"


def _write_part_metadata(
    part_path: Path,
    *,
    original: Path,
    unit: SplitUnit,
    document_type: str,
    split_strategy: str,
) -> Path:
    metadata_path = part_path.with_suffix(".meta.json")
    metadata = {
        "original": str(original),
        "part": str(part_path),
        "title": unit.title,
        "start_page": unit.start_page,
        "end_page": unit.end_page,
        "document_type": document_type,
        "split_strategy": split_strategy,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


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
    "PdfProfile",
    "PdfPlan",
    "PreparedSource",
    "SplitUnit",
    "build_pdf_profile",
    "estimate_pdf_pages",
    "plan_pdf_split",
    "plan_insert_files",
    "run_pdf_planning_agent",
    "split_pdf_to_raw_parts",
]
