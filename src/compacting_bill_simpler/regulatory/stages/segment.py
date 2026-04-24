from __future__ import annotations

import re
from typing import Any

from ...text_processing import clean_legislative_text, count_cl100k_tokens, segment_into_sentences
from ..llm_profiles import build_chat_completion_kwargs
from ..models import BillRecord, SentenceRecord
from .document_signals import classify_statement_role, classify_text_zone

_LONG_SENT_TOKEN_THRESHOLD = 140


def _normalize_space(text: str) -> str:
    return " ".join(text.split())


def _make_sentence_record(
    *,
    bill_id: str,
    sentence_id: int,
    chunk_hint_id: int,
    text: str,
    char_start: int | None,
    char_end: int | None,
    source_sentence_id: int | None,
) -> SentenceRecord:
    zone = classify_text_zone(text)
    return SentenceRecord(
        bill_id=bill_id,
        sentence_id=sentence_id,
        chunk_hint_id=chunk_hint_id,
        text=text,
        n_tokens=count_cl100k_tokens(text),
        char_start=char_start,
        char_end=char_end,
        source_sentence_id=source_sentence_id,
        zone_label=zone["label"],
        statement_role=classify_statement_role(text)["label"],
        operative_score=float(zone["operative_score"]),
        background_score=float(zone["background_score"]),
        noise_score=float(zone["noise_score"]),
    )


def _split_by_llm(text: str, model: str, client: Any) -> list[str]:
    try:
        response = client.chat.completions.create(
            **build_chat_completion_kwargs(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Split this long legal sentence into as many shorter sentences as needed. "
                            "Do not rewrite, paraphrase, normalize, or delete anything. "
                            "Preserve the exact original wording, numbering, punctuation, citations, and casing. "
                            "Only insert sentence breaks. Return one exact sentence fragment per line, in order."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_output_tokens=min(4000, max(1000, count_cl100k_tokens(text) + 400)),
            )
        )
        raw = response.choices[0].message.content
        if not raw:
            return [text]
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if len(lines) < 2:
            return [text]
        if _normalize_space(" ".join(lines)) != _normalize_space(text):
            return [text]
        return lines
    except Exception:
        return [text]


def _map_llm_parts_to_spans(
    original_text: str,
    parts: list[str],
) -> list[tuple[str, int | None, int | None]]:
    mapped: list[tuple[str, int | None, int | None]] = []
    cursor = 0

    for part in parts:
        lookup = original_text.find(part, cursor)
        if lookup >= 0:
            mapped.append((part, lookup, lookup + len(part)))
            cursor = lookup + len(part)
        else:
            mapped.append((part, None, None))
    return mapped


def _split_by_rules(text: str) -> list[tuple[str, int, int]]:
    match = None
    for pattern in (r";\s+(?=provided\s+that\b)", r";\s+(?=except\s+that\b)"):
        match = re.search(pattern, text, flags=re.I)
        if match:
            break
    if not match:
        return [(text, 0, len(text))]
    left = text[: match.start() + 1].strip()
    right_start = match.end()
    right = text[right_start:].strip()
    if not left or not right:
        return [(text, 0, len(text))]
    left_start = text.find(left)
    right_start = text.find(right, right_start)
    return [
        (left, left_start, left_start + len(left)),
        (right, right_start, right_start + len(right)),
    ]


def _repair_long_part(
    text: str,
    threshold: int,
    dry_run: bool,
    llm_model: str,
    llm_client: Any | None,
    *,
    depth: int = 0,
    max_depth: int = 3,
) -> list[tuple[str, int, int]]:
    if count_cl100k_tokens(text) <= threshold:
        return [(text, 0, len(text))]

    rule_parts = _split_by_rules(text)
    if len(rule_parts) > 1:
        return rule_parts

    if dry_run or llm_client is None or depth >= max_depth:
        return [(text, 0, len(text))]

    llm_parts = _split_by_llm(text, model=llm_model, client=llm_client)
    mapped_parts = _map_llm_parts_to_spans(text, llm_parts)
    if len(mapped_parts) < 2 or any(start is None or end is None for _, start, end in mapped_parts):
        return [(text, 0, len(text))]

    repaired: list[tuple[str, int, int]] = []
    for part_text, rel_start, rel_end in mapped_parts:
        child_parts = _repair_long_part(
            part_text,
            threshold,
            dry_run,
            llm_model,
            llm_client,
            depth=depth + 1,
            max_depth=max_depth,
        )
        for child_text, child_start, child_end in child_parts:
            repaired.append((child_text, rel_start + child_start, rel_start + child_end))

    return repaired


def fix_long_sentences(
    sentences: list[SentenceRecord],
    threshold: int = _LONG_SENT_TOKEN_THRESHOLD,
    dry_run: bool = True,
    llm_model: str = "gpt-5-nano",
    llm_client: Any | None = None,
) -> list[SentenceRecord]:
    out: list[SentenceRecord] = []
    sent_idx = 0

    for sent in sentences:
        repaired_parts = _repair_long_part(
            sent.text,
            threshold,
            dry_run,
            llm_model,
            llm_client,
        )

        if len(repaired_parts) == 1:
            out.append(
                _make_sentence_record(
                    bill_id=sent.bill_id,
                    sentence_id=sent_idx,
                    chunk_hint_id=sent.chunk_hint_id,
                    text=sent.text,
                    char_start=sent.char_start,
                    char_end=sent.char_end,
                    source_sentence_id=sent.source_sentence_id or sent.sentence_id,
                )
            )
            sent_idx += 1
            continue

        for part_text, rel_start, rel_end in repaired_parts:
            if not part_text.strip():
                continue
            char_start = sent.char_start + rel_start if sent.char_start is not None else None
            char_end = sent.char_start + rel_end if sent.char_start is not None else None
            out.append(
                _make_sentence_record(
                    bill_id=sent.bill_id,
                    sentence_id=sent_idx,
                    chunk_hint_id=sent.chunk_hint_id,
                    text=part_text,
                    char_start=char_start,
                    char_end=char_end,
                    source_sentence_id=sent.source_sentence_id or sent.sentence_id,
                )
            )
            sent_idx += 1

    return out


def segment_bill(
    record: BillRecord,
    dry_run: bool = True,
    long_sent_model: str = "gpt-5-nano",
    long_sent_threshold: int = _LONG_SENT_TOKEN_THRESHOLD,
    llm_client: Any | None = None,
) -> list[SentenceRecord]:
    text = record.cleaned_text if record.cleaned_text is not None else clean_legislative_text(record.raw_text)
    rows = segment_into_sentences(text)

    out: list[SentenceRecord] = []
    for item in rows:
        sentence_text = item["text"].strip().replace("\n", " ")
        sentence_text = " ".join(sentence_text.split())  # collapse multiple spaces
        if len(sentence_text) < 15:
            continue
        out.append(
            _make_sentence_record(
                bill_id=record.bill_id,
                sentence_id=item["sentence_id"],
                chunk_hint_id=item["chunk_id"],
                text=sentence_text,
                char_start=item.get("char_start"),
                char_end=item.get("char_end"),
                source_sentence_id=item["sentence_id"],
            )
        )

    return fix_long_sentences(
        out,
        threshold=long_sent_threshold,
        dry_run=dry_run,
        llm_model=long_sent_model,
        llm_client=llm_client,
    )
