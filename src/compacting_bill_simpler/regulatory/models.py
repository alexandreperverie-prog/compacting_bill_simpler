from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BillRecord:
    bill_id: str
    title: str
    jurisdiction: str | None
    raw_text: str
    text_hash: str | None = None
    cleaned_text: str | None = None  # Set once by orchestrator, reused by segment + chunk


@dataclass
class SentenceRecord:
    bill_id: str
    sentence_id: int
    chunk_hint_id: int
    text: str
    n_tokens: int  # tiktoken cl100k_base count of *text* (pipeline-wide)
    char_start: int | None = None
    char_end: int | None = None
    source_sentence_id: int | None = None
    zone_label: str = "unknown"
    statement_role: str = "unknown"
    operative_score: float = 0.0
    background_score: float = 0.0
    noise_score: float = 0.0


@dataclass
class ChunkRecord:
    bill_id: str
    chunk_id: int
    text: str
    source_sentence_ids: list[int] = field(default_factory=list)
    char_start: int | None = None
    char_end: int | None = None
    sentences: list[SentenceRecord] = field(default_factory=list)
    zone_label: str = "unknown"
    statement_role: str = "unknown"
    operative_ratio: float = 0.0
    background_ratio: float = 0.0
    noise_ratio: float = 0.0


@dataclass
class LegalBlock:
    bill_id: str
    block_id: str
    role: str
    heading: str | None
    text: str
    parent_block_id: str | None = None
    level: int = 1
    chunk_ids: list[int] = field(default_factory=list)
    sentence_ids: list[int] = field(default_factory=list)
    char_start: int | None = None
    char_end: int | None = None
    source_mode: str = "deterministic"
    confidence: float = 0.0
    secondary_roles: list[str] = field(default_factory=list)
    is_mixed: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    bill_id: str
    field_family: str
    chunk_id: int
    chunk_text: str
    score: float
    source_sentence_ids: list[int] = field(default_factory=list)
    char_start: int | None = None
    char_end: int | None = None
    source_sentences: list[SentenceRecord] = field(default_factory=list)
    zone_label: str = "unknown"
    statement_role: str = "unknown"
    operative_ratio: float = 0.0
    background_ratio: float = 0.0
    noise_ratio: float = 0.0
    source_mode: str = "retrieval"
    coverage_type: str | None = None
    section_role: str | None = None
    legal_signal_tags: list[str] = field(default_factory=list)
    coverage_reason: str | None = None
    coverage_confidence: float | None = None
    routing_action: str = "must_extract"
    block_id: str | None = None
    block_role: str | None = None
    block_full_keep: bool = False


Facts = dict[str, Any]
