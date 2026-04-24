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
_CODE_HEADING_RE = re.compile(r"^(?:\d+[A-Za-z]?(?:[-.]\d+[A-Za-z]?){1,}|SECTION\s+\d+|\[SECTION_\d+\])\b", re.I)

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
- Reserve "other" for residual material only.

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


def _looks_like_new_block(chunk: ChunkRecord, previous_chunk: ChunkRecord | None) -> bool:
    cleaned = _norm(chunk.text)
    if not cleaned:
        return False
    if previous_chunk is None:
        return True
    if _extract_heading(cleaned):
        return True
    return bool(_CODE_HEADING_RE.match(cleaned) and not _looks_like_continuation(cleaned))


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
                    "notes": notes,
                }
            )
        )
    return classified


def build_legal_blocks(
    bill: BillRecord,
    sentences: list[SentenceRecord],
    chunks: list[ChunkRecord],
) -> list[LegalBlock]:
    sentence_lookup = {sentence.sentence_id: sentence for sentence in sentences}
    ordered_chunks = sorted(chunks, key=lambda row: row.chunk_id)

    raw_groups: list[dict[str, object]] = []
    previous_chunk: ChunkRecord | None = None
    for chunk in ordered_chunks:
        heading = _extract_heading(chunk.text)
        if not raw_groups or _looks_like_new_block(chunk, previous_chunk):
            raw_groups.append({"heading": heading, "chunk_ids": [chunk.chunk_id]})
        else:
            cast_ids = list(raw_groups[-1]["chunk_ids"])
            cast_ids.append(chunk.chunk_id)
            raw_groups[-1]["chunk_ids"] = cast_ids
            if raw_groups[-1].get("heading") is None and heading:
                raw_groups[-1]["heading"] = heading
        previous_chunk = chunk

    blocks: list[LegalBlock] = []
    for idx, group in enumerate(raw_groups, start=1):
        chunk_ids = [int(value) for value in list(group["chunk_ids"])]
        sentence_ids = sorted(
            {
                int(sentence_id)
                for chunk_id in chunk_ids
                for sentence_id in list((next(chunk for chunk in ordered_chunks if chunk.chunk_id == chunk_id)).source_sentence_ids or [])
                if int(sentence_id) in sentence_lookup
            }
        )
        block_sentences = [sentence_lookup[sentence_id] for sentence_id in sentence_ids]
        char_starts = [sentence.char_start for sentence in block_sentences if sentence.char_start is not None]
        char_ends = [sentence.char_end for sentence in block_sentences if sentence.char_end is not None]
        notes: list[str] = []
        if group.get("heading"):
            notes.append("heading_detected")
        blocks.append(
            LegalBlock(
                bill_id=bill.bill_id,
                block_id=f"BLK-{idx:03d}",
                role="other",
                heading=str(group.get("heading")) if group.get("heading") else None,
                text=" ".join(sentence.text for sentence in block_sentences),
                chunk_ids=chunk_ids,
                sentence_ids=sentence_ids,
                char_start=min(char_starts) if char_starts else None,
                char_end=max(char_ends) if char_ends else None,
                confidence=0.4,
                notes=notes or ["structural_block"],
            )
        )
    return blocks
