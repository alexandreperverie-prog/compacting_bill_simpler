from __future__ import annotations

from ...text_processing import chunk_text_by_tokens, clean_legislative_text
from ..models import BillRecord, ChunkRecord, SentenceRecord
from .document_signals import classify_statement_role, classify_text_zone


def _build_overlap(
    sentences: list[SentenceRecord], overlap_tokens: int
) -> list[SentenceRecord]:
    """
    Return a suffix of *sentences* whose combined n_tokens <= overlap_tokens.
    Used to carry context from one chunk into the next.
    """
    if overlap_tokens <= 0:
        return []

    tail: list[SentenceRecord] = []
    total = 0
    for s in reversed(sentences):
        if total + s.n_tokens > overlap_tokens:
            break
        tail.insert(0, s)
        total += s.n_tokens
    return tail


def chunk_bill(
    record: BillRecord,
    sentences: list[SentenceRecord],
    min_tokens: int = 140,
    max_tokens: int = 220,
    overlap: int = 40,
) -> list[ChunkRecord]:
    """
    Build token-bounded chunks that always respect sentence boundaries.

    Token budgets (min/max/overlap) use the same unit as ``SentenceRecord.n_tokens``:
    tiktoken ``cl100k_base`` counts (see ``text_processing.count_cl100k_tokens``).

    Closing logic:
      - A chunk closes before adding the next sentence if that sentence would
        push the chunk past max_tokens.
      - min_tokens is only a soft target for underfilled chunks; max_tokens is
        treated as the hard ceiling for any multi-sentence chunk.
      - A single sentence longer than max_tokens is emitted on its own because
        sentence boundaries must be preserved here.

    Overlap:
      - The last N sentences of a closed chunk (summing to <= overlap tokens)
        are prepended to the next chunk, giving semantic continuity.

    Fallback:
      - If no sentences are provided, falls back to blind token-window chunking
        on record.cleaned_text (or raw_text) for robustness.
    """
    if not sentences:
        cleaned = record.cleaned_text or clean_legislative_text(record.raw_text)
        text_chunks = chunk_text_by_tokens(cleaned, chunk_size=max_tokens, overlap=overlap)
        out: list[ChunkRecord] = []
        for i, text in enumerate(text_chunks):
            zone = classify_text_zone(text)
            out.append(
                ChunkRecord(
                    bill_id=record.bill_id,
                    chunk_id=i,
                    text=text,
                    zone_label=zone["label"],
                    statement_role=classify_statement_role(text)["label"],
                    operative_ratio=float(zone["operative_score"]),
                    background_ratio=float(zone["background_score"]),
                    noise_ratio=float(zone["noise_score"]),
                )
            )
        return out

    chunks: list[ChunkRecord] = []
    current: list[SentenceRecord] = []
    current_tokens: int = 0

    def _flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        text = " ".join(s.text for s in current)
        char_starts = [s.char_start for s in current if s.char_start is not None]
        char_ends = [s.char_end for s in current if s.char_end is not None]
        chunks.append(
            (
                lambda zone, operative_ratio, background_ratio, noise_ratio: ChunkRecord(
                    bill_id=record.bill_id,
                    chunk_id=len(chunks),
                    text=text,
                    source_sentence_ids=[s.sentence_id for s in current],
                    char_start=min(char_starts) if char_starts else None,
                    char_end=max(char_ends) if char_ends else None,
                    sentences=list(current),
                    zone_label=zone["label"],
                    statement_role=classify_statement_role(text)["label"],
                    operative_ratio=operative_ratio,
                    background_ratio=background_ratio,
                    noise_ratio=noise_ratio,
                )
            )(
                classify_text_zone(text),
                round(sum(float(s.operative_score or 0.0) for s in current) / max(1, len(current)), 4),
                round(sum(float(s.background_score or 0.0) for s in current) / max(1, len(current)), 4),
                round(sum(float(s.noise_score or 0.0) for s in current) / max(1, len(current)), 4),
            )
        )
        overlap_sents = _build_overlap(current, overlap)
        current = overlap_sents
        current_tokens = sum(s.n_tokens for s in overlap_sents)

    for sent in sentences:
        if not current:
            current.append(sent)
            current_tokens = sent.n_tokens
            continue

        would_exceed_max = current_tokens + sent.n_tokens > max_tokens

        if would_exceed_max:
            _flush()

        current.append(sent)
        current_tokens += sent.n_tokens

    # Flush remaining sentences
    if current:
        text = " ".join(s.text for s in current)
        char_starts = [s.char_start for s in current if s.char_start is not None]
        char_ends = [s.char_end for s in current if s.char_end is not None]
        chunks.append(
            (
                lambda zone, operative_ratio, background_ratio, noise_ratio: ChunkRecord(
                    bill_id=record.bill_id,
                    chunk_id=len(chunks),
                    text=text,
                    source_sentence_ids=[s.sentence_id for s in current],
                    char_start=min(char_starts) if char_starts else None,
                    char_end=max(char_ends) if char_ends else None,
                    sentences=list(current),
                    zone_label=zone["label"],
                    statement_role=classify_statement_role(text)["label"],
                    operative_ratio=operative_ratio,
                    background_ratio=background_ratio,
                    noise_ratio=noise_ratio,
                )
            )(
                classify_text_zone(text),
                round(sum(float(s.operative_score or 0.0) for s in current) / max(1, len(current)), 4),
                round(sum(float(s.background_score or 0.0) for s in current) / max(1, len(current)), 4),
                round(sum(float(s.noise_score or 0.0) for s in current) / max(1, len(current)), 4),
            )
        )

    return chunks
