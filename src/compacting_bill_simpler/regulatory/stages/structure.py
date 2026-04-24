from __future__ import annotations

import json
import re
from typing import Any

from ..llm_profiles import build_chat_completion_kwargs
from ..models import BillRecord, ChunkRecord, LegalBlock, SentenceRecord

_HEADING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?P<heading>(?:\d+[A-Za-z]?(?:[-.]\d+[A-Za-z]?){1,}|SECTION\s+\d+)\s+[A-Z][A-Za-z0-9 ,/&'().-]{3,120})", re.I),
    re.compile(r"(?P<heading>\[SECTION_\d+\]\s+[A-Z][A-Za-z0-9 ,/&'().-]{3,120})", re.I),
)

_CONTINUATION_RE = re.compile(r"^(?:\(\s*(?:\d+|[A-Za-z]|[ivxlcdmIVXLCDM]+)\s*\)|[a-z]|\d+[.)])")
_CODE_HEADING_RE = re.compile(r"(?:\d+[A-Za-z]?(?:[-.]\d+[A-Za-z]?){1,}|SECTION\s+\d+|\[SECTION_\d+\])\b", re.I)
_SECTION_BOUNDARY_RE = re.compile(
    r"(?P<label>(?:\d+[A-Za-z]?(?:[-.]\d+[A-Za-z]?){1,}|SECTION\s+\d+|\[SECTION_\d+\]))\s+",
    re.I,
)

_ROLE_OPTIONS = (
    "definitions",
    "scope",
    "consumer_rights",
    "controller_duties",
    "processor_duties",
    "assessments",
    "exemptions",
    "limitations",
    "enforcement",
    "effective_date",
    "preemption",
    "other",
)

_LLM_BLOCK_ROLE_PROMPT = """\
You are classifying structural legal blocks from a US statute.

Return ONLY valid JSON:
{
  "blocks": [
    {
      "block_id": "BLK-001",
      "role": "definitions|scope|consumer_rights|controller_duties|processor_duties|assessments|exemptions|limitations|enforcement|effective_date|preemption|other",
      "secondary_roles": ["..."],
      "is_mixed": false,
      "confidence": 0.0,
      "notes": ["short"]
    }
  ]
}

Rules:
- Use the provided block heading and text.
- Classify each block exactly once.
- Prefer the main legal function of the block.
- Do not invent block ids.
- If a block is mixed, choose the dominant role.
- Use "secondary_roles" only for genuinely substantial secondary functions.
- Reserve "other" for residual material only.

INPUT:
__INPUT__
"""

_LLM_BLOCK_REFINE_PROMPT = """\
You are refining a coarse legal block from a US statute into a few meaningful child blocks.

Return ONLY valid JSON:
{
  "subblocks": [
    {
      "start_sentence_id": 100,
      "end_sentence_id": 104,
      "role": "definitions|scope|consumer_rights|controller_duties|processor_duties|assessments|exemptions|limitations|enforcement|effective_date|preemption|other",
      "secondary_roles": ["..."],
      "is_mixed": false,
      "heading_hint": "short heading",
      "confidence": 0.0,
      "notes": ["short"]
    }
  ]
}

Rules:
- Produce 1 to 4 child blocks only.
- Child blocks must use contiguous sentence ranges.
- Child blocks must stay entirely inside the provided parent block.
- Prefer semantically meaningful splits, not every list item.
- Do not invent sentence ids.
- If the block is already coherent, return one child block covering the full range.

INPUT:
__INPUT__
"""


def _norm(text: str) -> str:
    return " ".join(str(text or "").split())


def _extract_heading(text: str) -> str | None:
    snippet = _norm(text)[:240]
    for pattern in _HEADING_PATTERNS:
        match = pattern.search(snippet)
        if match:
            return match.group("heading").strip()
    return None


def _looks_like_continuation(text: str) -> bool:
    cleaned = _norm(text)
    if not cleaned:
        return False
    return bool(_CONTINUATION_RE.match(cleaned) or cleaned[:1].islower())


def _clean_source_text(bill: BillRecord, sentences: list[SentenceRecord]) -> str:
    text = bill.cleaned_text or bill.raw_text or ""
    if _norm(text):
        return text
    return " ".join(sentence.text for sentence in sentences)


def _extract_heading_from_span(text: str) -> str | None:
    snippet = _norm(text)[:240]
    if not snippet:
        return None
    match = _CODE_HEADING_RE.search(snippet)
    if not match:
        return None
    candidate = snippet[match.start():]
    return candidate[:160].strip()


def _iter_section_matches(text: str) -> list[re.Match[str]]:
    return [match for match in _SECTION_BOUNDARY_RE.finditer(text)]


def _is_plausible_section_start(text: str, match: re.Match[str]) -> bool:
    start = match.start("label")
    label = match.group("label")
    after = text[match.end(): match.end() + 120]
    if not _norm(after):
        return False
    if label.startswith("[SECTION_"):
        return True
    if label.upper().startswith("SECTION "):
        return True
    after_norm = _norm(after)
    if not after_norm:
        return False
    if after_norm.startswith("("):
        return False
    if after_norm[:1].isdigit():
        return False
    if start > 0:
        prev = text[max(0, start - 2): start]
        if prev and prev[-1].isalnum():
            return False
    return True


def _find_section_boundaries(text: str) -> list[tuple[int, int, str | None]]:
    matches = [match for match in _iter_section_matches(text) if _is_plausible_section_start(text, match)]
    if not matches:
        cleaned = _norm(text)
        return [(0, len(text), _extract_heading(cleaned))] if cleaned else []

    spans: list[tuple[int, int, str | None]] = []
    starts = [match.start("label") for match in matches]

    first_start = starts[0]
    if first_start > 0 and _norm(text[:first_start]):
        spans.append((0, first_start, _extract_heading_from_span(text[:first_start])))

    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(text)
        span_text = text[start:end].strip()
        if not _norm(span_text):
            continue
        spans.append((start, end, _extract_heading_from_span(span_text)))

    return spans


def _span_overlaps(start: int | None, end: int | None, span_start: int, span_end: int) -> bool:
    if start is None or end is None:
        return False
    return end > span_start and start < span_end


def _fallback_role(block: LegalBlock) -> str:
    text = _norm(f"{block.heading or ''} {block.text}")
    lowered = text.lower()
    if "takes effect" in lowered or "effective date" in lowered:
        return "effective_date"
    if any(token in lowered for token in ("preempt", "supersede", "severable", "severability", "conflicting provisions")):
        return "preemption"
    if "enforcement" in lowered or "civil penalty" in lowered or "attorney general" in lowered:
        return "enforcement"
    if "exemptions" in lowered or "does not apply to" in lowered or "shall not apply to" in lowered:
        return "exemptions"
    if "limitations" in lowered or "does not restrict" in lowered:
        return "limitations"
    if "part definitions" in lowered or "as used in this part" in lowered or " means " in lowered or "has the same meaning" in lowered:
        return "definitions"
    if "consumer rights" in lowered or "opt out" in lowered or "appeal" in lowered:
        return "consumer_rights"
    if "privacy notice" in lowered or "controller shall" in lowered:
        return "controller_duties"
    if "processor shall" in lowered or "contract between a controller and a processor" in lowered or "subcontractor" in lowered:
        return "processor_duties"
    if "data protection assessment" in lowered or "profiling" in lowered or "heightened risk" in lowered:
        return "assessments"
    if "scope" in lowered or "this part applies" in lowered or "this act applies" in lowered:
        return "scope"
    return "other"


def _call_block_role_classifier(client: Any, model: str, *, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.chat.completions.create(
        **build_chat_completion_kwargs(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a careful legal analyst. Output only valid JSON."},
                {"role": "user", "content": _LLM_BLOCK_ROLE_PROMPT.replace("__INPUT__", json.dumps(payload, ensure_ascii=False))},
            ],
            max_output_tokens=4000,
        )
    )
    raw = response.choices[0].message.content or "{}"
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def _call_block_refiner(client: Any, model: str, *, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.chat.completions.create(
        **build_chat_completion_kwargs(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a careful legal analyst. Output only valid JSON."},
                {"role": "user", "content": _LLM_BLOCK_REFINE_PROMPT.replace("__INPUT__", json.dumps(payload, ensure_ascii=False))},
            ],
            max_output_tokens=3200,
        )
    )
    raw = response.choices[0].message.content or "{}"
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def _select_refinement_candidates(blocks: list[LegalBlock], *, max_candidates: int = 4) -> list[LegalBlock]:
    scored: list[tuple[int, LegalBlock]] = []
    for block in blocks:
        if block.level != 1:
            continue
        sentence_count = len(block.sentence_ids)
        if sentence_count < 4:
            continue
        score = 0
        if block.is_mixed:
            score += 100
        score += len(block.secondary_roles) * 25
        if block.role in {"consumer_rights", "controller_duties", "processor_duties", "enforcement", "exemptions", "limitations"}:
            score += 20
        if block.role == "definitions":
            score += 10
        if sentence_count >= 8:
            score += 15
        if sentence_count >= 14:
            score += 15
        if score > 0:
            scored.append((score, block))
    scored.sort(key=lambda item: (-item[0], item[1].block_id))
    return [block for _, block in scored[:max_candidates]]


def _safe_heading(base_heading: str | None, hint: str | None) -> str | None:
    hint_norm = _norm(hint or "")
    if not hint_norm:
        return base_heading
    if base_heading and hint_norm.lower() in base_heading.lower():
        return base_heading
    if base_heading:
        return f"{base_heading} - {hint_norm}"[:220]
    return hint_norm[:220]


def _refine_block(
    bill: BillRecord,
    block: LegalBlock,
    *,
    sentence_lookup: dict[int, SentenceRecord],
    ordered_chunks: list[ChunkRecord],
    config: Any,
    llm_client: Any | None,
    next_index: int,
) -> tuple[list[LegalBlock], int]:
    if config.dry_run or llm_client is None or not block.sentence_ids:
        return [block], next_index

    payload = {
        "bill_id": bill.bill_id,
        "title": bill.title,
        "parent_block": {
            "block_id": block.block_id,
            "role": block.role,
            "secondary_roles": block.secondary_roles,
            "is_mixed": block.is_mixed,
            "heading": block.heading,
            "sentence_ids": block.sentence_ids,
            "text": _norm(block.text)[:9000],
        },
        "sentences": [
            {
                "sentence_id": sentence_id,
                "text": sentence_lookup[sentence_id].text,
            }
            for sentence_id in block.sentence_ids
            if sentence_id in sentence_lookup
        ],
    }
    response = _call_block_refiner(llm_client, config.llm_extraction_model, payload=payload)
    proposals = [row for row in list(response.get("subblocks") or []) if isinstance(row, dict)]
    if not proposals:
        return [block], next_index

    valid_ids = set(block.sentence_ids)
    normalized: list[dict[str, Any]] = []
    for row in proposals:
        start = row.get("start_sentence_id")
        end = row.get("end_sentence_id")
        if start not in valid_ids or end not in valid_ids:
            continue
        try:
            start_i = int(start)
            end_i = int(end)
        except Exception:
            continue
        if start_i > end_i:
            continue
        span_ids = [sentence_id for sentence_id in block.sentence_ids if start_i <= sentence_id <= end_i]
        if not span_ids:
            continue
        role = str(row.get("role") or "").strip()
        if role not in _ROLE_OPTIONS:
            role = block.role
        secondary_roles = [
            value
            for value in [str(item).strip() for item in list(row.get("secondary_roles") or [])]
            if value in _ROLE_OPTIONS and value not in {role, "other"}
        ]
        normalized.append(
            {
                "start": start_i,
                "end": end_i,
                "sentence_ids": span_ids,
                "role": role,
                "secondary_roles": secondary_roles,
                "is_mixed": bool(row.get("is_mixed")) or bool(secondary_roles),
                "heading": _safe_heading(block.heading, str(row.get("heading_hint") or "")),
                "confidence": round(float(row.get("confidence") or 0.0), 4),
                "notes": [_norm(str(note)) for note in list(row.get("notes") or []) if _norm(str(note))],
            }
        )

    if not normalized:
        return [block], next_index

    normalized.sort(key=lambda row: (row["start"], row["end"]))
    coverage = set()
    child_blocks: list[LegalBlock] = []
    for row in normalized:
        span_ids = row["sentence_ids"]
        if any(sentence_id in coverage for sentence_id in span_ids):
            continue
        coverage.update(span_ids)
        block_sentences = [sentence_lookup[sentence_id] for sentence_id in span_ids if sentence_id in sentence_lookup]
        if not block_sentences:
            continue
        char_starts = [sentence.char_start for sentence in block_sentences if sentence.char_start is not None]
        char_ends = [sentence.char_end for sentence in block_sentences if sentence.char_end is not None]
        char_start = min(char_starts) if char_starts else None
        char_end = max(char_ends) if char_ends else None
        chunk_ids = sorted(
            chunk.chunk_id
            for chunk in ordered_chunks
            if any(sentence_id in set(chunk.source_sentence_ids or []) for sentence_id in span_ids)
        )
        child_blocks.append(
            LegalBlock(
                bill_id=bill.bill_id,
                block_id=f"BLK-{next_index:03d}",
                role=row["role"],
                heading=row["heading"],
                text=" ".join(sentence.text for sentence in block_sentences),
                parent_block_id=block.block_id,
                level=2,
                chunk_ids=chunk_ids,
                sentence_ids=span_ids,
                char_start=char_start,
                char_end=char_end,
                source_mode="llm_refined",
                confidence=row["confidence"] or block.confidence,
                secondary_roles=row["secondary_roles"],
                is_mixed=row["is_mixed"],
                notes=["llm_refined_subblock"] + row["notes"],
            )
        )
        next_index += 1

    if len(child_blocks) <= 1:
        return [block], next_index

    parent_block = LegalBlock(
        **{
            **block.__dict__,
            "is_mixed": True,
            "notes": list(block.notes) + ["refined_parent"],
        }
    )
    return [parent_block, *child_blocks], next_index


def classify_legal_blocks(
    bill: BillRecord,
    blocks: list[LegalBlock],
    *,
    config: Any,
    llm_client: Any | None,
) -> list[LegalBlock]:
    if not blocks:
        return []

    if config.dry_run or llm_client is None:
        out: list[LegalBlock] = []
        for block in blocks:
            role = _fallback_role(block)
            notes = list(block.notes)
            notes.append("fallback_role")
            out.append(
                LegalBlock(
                    **{**block.__dict__, "role": role, "confidence": 0.75 if role != "other" else 0.4, "notes": notes}
                )
            )
        return out

    payload = {
        "bill_id": bill.bill_id,
        "title": bill.title,
        "blocks": [
            {
                "block_id": block.block_id,
                "level": block.level,
                "parent_block_id": block.parent_block_id,
                "heading": block.heading,
                "text": _norm(block.text)[:6000],
                "sentence_ids": block.sentence_ids,
            }
            for block in blocks
        ],
    }
    response = _call_block_role_classifier(llm_client, config.llm_extraction_model, payload=payload)
    by_id = {
        str(row.get("block_id")): row
        for row in list(response.get("blocks") or [])
        if isinstance(row, dict) and str(row.get("block_id") or "")
    }

    classified: list[LegalBlock] = []
    for block in blocks:
        row = by_id.get(block.block_id, {})
        role = str(row.get("role") or "").strip()
        if role not in _ROLE_OPTIONS:
            role = _fallback_role(block)
        secondary_roles = [
            value
            for value in [str(item).strip() for item in list(row.get("secondary_roles") or [])]
            if value in _ROLE_OPTIONS and value not in {role, "other"}
        ]
        confidence = float(row.get("confidence") or 0.0)
        notes = list(block.notes)
        notes.extend([_norm(str(note)) for note in list(row.get("notes") or []) if _norm(str(note))])
        if row:
            notes.append("llm_role")
        else:
            notes.append("fallback_role")
        classified.append(
            LegalBlock(
                **{
                    **block.__dict__,
                    "role": role,
                    "source_mode": "llm" if row else "deterministic",
                    "confidence": round(confidence if row else (0.75 if role != "other" else 0.4), 4),
                    "secondary_roles": secondary_roles,
                    "is_mixed": bool(row.get("is_mixed")) if row else bool(secondary_roles),
                    "notes": notes,
                }
            )
        )
    return classified


def refine_legal_blocks(
    bill: BillRecord,
    blocks: list[LegalBlock],
    sentences: list[SentenceRecord],
    chunks: list[ChunkRecord],
    *,
    config: Any,
    llm_client: Any | None,
) -> list[LegalBlock]:
    if not blocks:
        return []

    candidates = {block.block_id for block in _select_refinement_candidates(blocks)}
    sentence_lookup = {sentence.sentence_id: sentence for sentence in sentences}
    ordered_chunks = sorted(chunks, key=lambda row: row.chunk_id)
    next_index = len(blocks) + 1
    refined: list[LegalBlock] = []

    for block in blocks:
        if block.block_id not in candidates:
            refined.append(block)
            continue
        new_blocks, next_index = _refine_block(
            bill,
            block,
            sentence_lookup=sentence_lookup,
            ordered_chunks=ordered_chunks,
            config=config,
            llm_client=llm_client,
            next_index=next_index,
        )
        refined.extend(new_blocks)

    refined.sort(key=lambda block: ((block.char_start or 0), block.level, block.block_id))
    return refined


def build_legal_blocks(
    bill: BillRecord,
    sentences: list[SentenceRecord],
    chunks: list[ChunkRecord],
) -> list[LegalBlock]:
    sentence_lookup = {sentence.sentence_id: sentence for sentence in sentences}
    ordered_chunks = sorted(chunks, key=lambda row: row.chunk_id)
    source_text = _clean_source_text(bill, sentences)
    section_spans = _find_section_boundaries(source_text)

    blocks: list[LegalBlock] = []
    for idx, (section_start, section_end, section_heading) in enumerate(section_spans, start=1):
        span_text = source_text[section_start:section_end].strip()
        if not _norm(span_text):
            continue
        sentence_ids = sorted(
            sentence_id
            for sentence_id, sentence in sentence_lookup.items()
            if _span_overlaps(sentence.char_start, sentence.char_end, section_start, section_end)
        )
        chunk_ids = sorted(
            chunk.chunk_id
            for chunk in ordered_chunks
            if _span_overlaps(chunk.char_start, chunk.char_end, section_start, section_end)
        )
        notes = ["structural_section"]
        if section_heading:
            notes.append("heading_detected")
        blocks.append(
            LegalBlock(
                bill_id=bill.bill_id,
                block_id=f"BLK-{idx:03d}",
                role="other",
                heading=section_heading,
                text=span_text,
                level=1,
                chunk_ids=chunk_ids,
                sentence_ids=sentence_ids,
                char_start=section_start,
                char_end=section_end,
                confidence=0.4,
                notes=notes,
            )
        )

    return blocks
