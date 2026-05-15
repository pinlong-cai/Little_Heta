"""Dataclasses for all memory tables."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Session:
    session_id: str
    started_at: int
    ended_at: int | None = None
    consolidated: int = 0
    consolidated_at: int | None = None


@dataclass
class L0Turn:
    session_id: str
    turn_index: int
    role: str           # user / assistant / system / tool
    modality: str       # text / audio / image / mixed
    text_content: str
    created_at: int


@dataclass
class MemoryMeta:
    memory_id: str
    memory_type: str    # L1 / L2
    session_id: str | None
    origin: str         # extracted / promoted / user_explicit / consolidated
    created_at: int
    last_access_at: int
    kb_uid: str | None = None
    status: str = "active"
    deprecated_by: str | None = None
    recency_score: float = 1.0
    access_freq: int = 0
    user_emphasis: float = 0.0
    importance: float = 0.5
    confidence: float = 0.9


@dataclass
class L1Episodic:
    memory_id: str
    who: str            # JSON array, e.g. '["Alice", "Bob"]'
    what: str
    where_loc: str | None
    when_ts: int | None           # unix timestamp of period start
    when_text: str | None         # original expression ("昨天", "下个月")
    when_resolved: str | None     # variable-precision: "2026-05-12" / "2026-06" / "2026"
    when_precision: str | None    # day / week / month / year
    why: str | None
    summary: str        # used for vector embedding


@dataclass
class L2Semantic:
    memory_id: str
    subject: str
    predicate: str
    object: str
    object_type: str        # literal / entity_ref
    fact_text: str          # natural language form, used for embedding
    t_valid_start: int
    t_valid_end: int | None = None
    when_text: str | None = None      # original relative expression ("下个月")
    when_resolved: str | None = None  # variable-precision: "2026-06" / "2026-05-12"
    when_precision: str | None = None # day / week / month / year


@dataclass
class KBInsight:
    memory_id: str
    insight: str                       # distilled knowledge point
    source_paths: list[str]            # all KB pages this insight derives from
    created_at: int
    question: str | None = None
    wiki_id: int | None = None         # primary wiki id (from first source)
    heading_path: str | None = None    # primary heading (from first source)

    @property
    def source_path(self) -> str:
        """Primary source path — kept for the legacy column / display."""
        return self.source_paths[0] if self.source_paths else ""
