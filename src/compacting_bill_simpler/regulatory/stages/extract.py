from __future__ import annotations

import json
import re
from typing import Any

from ...text_processing import count_cl100k_tokens
from ..llm_profiles import build_chat_completion_kwargs
from ..models import BillRecord, Facts, LegalBlock

_SCOPE_ROLES = {"definitions", "scope", "exemptions", "limitations", "effective_date", "preemption"}
_EFFECT_ROLES = {"consumer_rights", "controller_duties", "processor_duties", "assessments", "enforcement", "limitations"}

_SCOPE_PROMPT = """\
You are extracting the legal scope and carve-outs of a US statute.

Return ONLY valid JSON:
{
  "definitions": [
    {
      "term": "controller",
      "definition": "means the natural or legal person that determines the purpose and means of processing personal information",
      "evidence_block_ids": ["BLK-001"],
      "confidence": 0.95
    }
  ],
  "applicability": {
    "applies_to_client": "yes|no|uncertain",
    "scope_type": "universal|conditional|mixed|exempted|unknown",
    "covered_entities": ["..."],
    "covered_roles": ["..."],
    "excluded_entities": ["..."],
    "trigger_conditions": ["..."],
    "applicability_criteria": ["..."],
    "thresholds": [
      {
        "type": "revenue|consumer_count|employee_count|time_period|other",
        "operator": ">|>=|<|<=|=",
        "value": "25000000",
        "unit": "USD",
        "metric": "revenue",
        "text": "exceed twenty-five million dollars ($25,000,000) in revenue"
      }
    ],
    "condition_logic_summary": "short natural-language logic summary",
    "scope_summary": "one concise scope sentence",
    "effective_date": "YYYY-MM-DD or null",
    "confidence": 0.0,
    "evidence_block_ids": ["BLK-002"]
  },
  "notes": ["short"]
}

Rules:
- Use only the provided text.
- Prefer complete scope and exemption logic over sparse snippets.
- Keep exclusions and exemptions separate from covered entities.
- Do not invent dates, entities, thresholds, or logic not supported by the text.
- Only include high-value definitions that materially help interpret scope.

INPUT:
__INPUT__
"""

_EFFECTS_PROMPT = """\
You are extracting the operative legal effects of a US statute.

Return ONLY valid JSON:
{
  "obligations": [
    {
      "id": "OBL-001",
      "text": "A controller shall provide a reasonably accessible, clear, and meaningful privacy notice.",
      "subject": "controller",
      "confidence": 0.95,
      "evidence_block_ids": ["BLK-004"]
    }
  ],
  "prohibitions": [
    {
      "id": "PROH-001",
      "text": "A controller shall not discriminate against a consumer for exercising statutory rights.",
      "subject": "controller",
      "confidence": 0.95,
      "evidence_block_ids": ["BLK-005"]
    }
  ],
  "powers": [
    {
      "id": "POW-001",
      "text": "The attorney general and reporter may issue a civil investigative demand.",
      "subject": "attorney general and reporter",
      "confidence": 0.93,
      "evidence_block_ids": ["BLK-009"]
    }
  ],
  "sanctions": [
    {
      "id": "SAN-001",
      "text": "A court may impose a civil penalty of up to $7,500 for each violation.",
      "confidence": 0.96,
      "evidence_block_ids": ["BLK-009"]
    }
  ],
  "eligibility_conditions": [
    {
      "id": "ELG-001",
      "text": "short condition text",
      "confidence": 0.8,
      "evidence_block_ids": ["BLK-00X"]
    }
  ],
  "rights": [
    {
      "id": "RGT-001",
      "text": "A consumer may access personal information processed by the controller.",
      "confidence": 0.9,
      "evidence_block_ids": ["BLK-003"]
    }
  ],
  "notes": ["short"]
}

Rules:
- Extract only black-letter legal effects.
- Do not output chopped fragments, headings, or continuation stubs.
- Prefer fewer clean items over many noisy ones.
- Keep rights separate from obligations and prohibitions.
- Sanctions/consequences must be actual enforcement consequences, not ordinary violations or headings.
- Use only the provided text.

INPUT:
__INPUT__
"""

_QUALITY_PROMPT = """\
You are auditing extracted facts from a US statute.

Return ONLY valid JSON:
{
  "judge_score": 0.0,
  "verdict": "approved|needs_review|rejected",
  "issues": ["short"],
  "confidence": 0.0
}

Focus on:
- whether scope/applicability is resolved clearly
- whether obligations/prohibitions are clean and non-fragmentary
- whether enforcement consequences are real and well anchored
- whether the extraction is missing obviously important material

INPUT:
__INPUT__
"""


def _norm(text: str) -> str:
    return " ".join(str(text or "").split())


def _dedup_strings(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _norm(str(value or ""))
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _safe_block_ids(values: list[Any], valid_ids: set[str]) -> list[str]:
    return [str(value) for value in values if str(value) in valid_ids]


def _format_thresholds(thresholds: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for row in thresholds:
        if not isinstance(row, dict):
            continue
        text = _norm(str(row.get("text") or ""))
        if text:
            parts.append(text)
            continue
        metric = _norm(str(row.get("metric") or row.get("type") or "threshold"))
        operator = _norm(str(row.get("operator") or ""))
        value = _norm(str(row.get("value") or ""))
        unit = _norm(str(row.get("unit") or ""))
        rendered = " ".join(part for part in (metric, operator, value, unit) if part)
        if rendered:
            parts.append(rendered)
    return "; ".join(_dedup_strings(parts))


def _call_json_completion(client: Any, model: str, *, prompt: str, max_output_tokens: int) -> dict[str, Any]:
    response = client.chat.completions.create(
        **build_chat_completion_kwargs(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a careful legal analyst. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=max_output_tokens,
        )
    )
    raw = response.choices[0].message.content or "{}"
    payload = json.loads(raw)
    return payload if isinstance(payload, dict) else {}


def _serialize_blocks(blocks: list[LegalBlock], *, text_limit: int = 4000) -> list[dict[str, Any]]:
    return [
        {
            "block_id": block.block_id,
            "role": block.role,
            "heading": block.heading,
            "sentence_ids": block.sentence_ids,
            "chunk_ids": block.chunk_ids,
            "text": _norm(block.text)[:text_limit],
        }
        for block in blocks
    ]


def _select_blocks(blocks: list[LegalBlock], roles: set[str]) -> list[LegalBlock]:
    return [block for block in blocks if block.role in roles]


def _fits_full_context(bill: BillRecord, config: Any) -> bool:
    return count_cl100k_tokens(bill.cleaned_text or bill.raw_text or "") <= int(config.full_context_max_tokens)


def _build_scope_input(bill: BillRecord, blocks: list[LegalBlock], *, full_context: bool) -> dict[str, Any]:
    selected_blocks = _select_blocks(blocks, _SCOPE_ROLES)
    payload: dict[str, Any] = {
        "bill_id": bill.bill_id,
        "title": bill.title,
        "full_context_mode": full_context,
        "blocks": _serialize_blocks(selected_blocks),
    }
    if full_context:
        payload["cleaned_text"] = bill.cleaned_text or bill.raw_text or ""
    return payload


def _build_effects_input(bill: BillRecord, blocks: list[LegalBlock], *, full_context: bool) -> dict[str, Any]:
    selected_blocks = _select_blocks(blocks, _EFFECT_ROLES)
    payload: dict[str, Any] = {
        "bill_id": bill.bill_id,
        "title": bill.title,
        "full_context_mode": full_context,
        "blocks": _serialize_blocks(selected_blocks),
    }
    if full_context:
        payload["cleaned_text"] = bill.cleaned_text or bill.raw_text or ""
    return payload


def extract_scope_facts(
    bill: BillRecord,
    blocks: list[LegalBlock],
    *,
    config: Any,
    llm_client: Any | None,
) -> dict[str, Any]:
    full_context = _fits_full_context(bill, config)
    payload = _build_scope_input(bill, blocks, full_context=full_context)
    if config.dry_run or llm_client is None:
        return {
            "status": "dry_run",
            "mode": "full_context" if full_context else "block_context",
            "request": payload,
            "response": {},
        }
    response = _call_json_completion(
        llm_client,
        config.llm_extraction_model,
        prompt=_SCOPE_PROMPT.replace("__INPUT__", json.dumps(payload, ensure_ascii=False)),
        max_output_tokens=5000,
    )
    return {
        "status": "ok",
        "mode": "full_context" if full_context else "block_context",
        "request": payload,
        "response": response,
    }


def extract_effect_facts(
    bill: BillRecord,
    blocks: list[LegalBlock],
    *,
    config: Any,
    llm_client: Any | None,
) -> dict[str, Any]:
    full_context = _fits_full_context(bill, config)
    payload = _build_effects_input(bill, blocks, full_context=full_context)
    if config.dry_run or llm_client is None:
        return {
            "status": "dry_run",
            "mode": "full_context" if full_context else "block_context",
            "request": payload,
            "response": {},
        }
    response = _call_json_completion(
        llm_client,
        config.llm_extraction_model,
        prompt=_EFFECTS_PROMPT.replace("__INPUT__", json.dumps(payload, ensure_ascii=False)),
        max_output_tokens=7000,
    )
    return {
        "status": "ok",
        "mode": "full_context" if full_context else "block_context",
        "request": payload,
        "response": response,
    }


def _normalize_item_list(
    rows: list[Any],
    *,
    prefix: str,
    valid_block_ids: set[str],
    keep_subject: bool = False,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    counter = 1
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = _norm(str(row.get("text") or ""))
        if not text:
            continue
        key = re.sub(r"\s+", " ", text.lower())
        if key in seen:
            continue
        seen.add(key)
        item = {
            "id": str(row.get("id") or f"{prefix}-{counter:03d}"),
            "text": text,
            "confidence": round(float(row.get("confidence") or 0.0), 4),
            "evidence_block_ids": _safe_block_ids(list(row.get("evidence_block_ids") or []), valid_block_ids),
        }
        if keep_subject and _norm(str(row.get("subject") or "")):
            item["subject"] = _norm(str(row.get("subject") or ""))
        out.append(item)
        counter += 1
    return out


def consolidate_facts(
    bill: BillRecord,
    blocks: list[LegalBlock],
    scope_trace: dict[str, Any],
    effects_trace: dict[str, Any],
) -> Facts:
    valid_block_ids = {block.block_id for block in blocks}
    scope = dict(scope_trace.get("response") or {})
    effects = dict(effects_trace.get("response") or {})
    app = dict(scope.get("applicability") or {})

    thresholds: list[dict[str, Any]] = []
    seen_thresholds: set[tuple[str, str, str, str, str]] = set()
    for row in list(app.get("thresholds") or []):
        if not isinstance(row, dict):
            continue
        normalized = {
            "type": _norm(str(row.get("type") or "other")).lower() or "other",
            "operator": _norm(str(row.get("operator") or row.get("comparison") or "")),
            "value": _norm(str(row.get("value") or "")),
            "unit": _norm(str(row.get("unit") or "")) or None,
            "metric": _norm(str(row.get("metric") or "")) or None,
            "text": _norm(str(row.get("text") or "")) or None,
        }
        key = (
            str(normalized["type"]),
            str(normalized["operator"]),
            str(normalized["value"]),
            str(normalized["unit"] or ""),
            str(normalized["metric"] or ""),
        )
        if not normalized["value"] or key in seen_thresholds:
            continue
        seen_thresholds.add(key)
        thresholds.append(normalized)

    applicability = {
        "applies_to_client": str(app.get("applies_to_client") or "uncertain"),
        "scope_type": str(app.get("scope_type") or "unknown"),
        "covered_entities": _dedup_strings(list(app.get("covered_entities") or [])),
        "covered_roles": _dedup_strings(list(app.get("covered_roles") or [])),
        "excluded_entities": _dedup_strings(list(app.get("excluded_entities") or [])),
        "trigger_conditions": _dedup_strings(list(app.get("trigger_conditions") or [])),
        "applicability_criteria": _dedup_strings(list(app.get("applicability_criteria") or [])),
        "thresholds": thresholds,
        "condition_logic_summary": _norm(str(app.get("condition_logic_summary") or "")) or None,
        "scope_summary": _norm(str(app.get("scope_summary") or "")) or None,
        "effective_date": app.get("effective_date"),
        "confidence": round(float(app.get("confidence") or 0.0), 4),
        "evidence_block_ids": _safe_block_ids(list(app.get("evidence_block_ids") or []), valid_block_ids),
    }

    definitions = []
    seen_terms: set[str] = set()
    for row in list(scope.get("definitions") or []):
        if not isinstance(row, dict):
            continue
        term = _norm(str(row.get("term") or ""))
        definition = _norm(str(row.get("definition") or ""))
        if not term or not definition:
            continue
        key = term.lower()
        if key in seen_terms:
            continue
        seen_terms.add(key)
        definitions.append(
            {
                "term": term,
                "definition": definition,
                "confidence": round(float(row.get("confidence") or 0.0), 4),
                "evidence_block_ids": _safe_block_ids(list(row.get("evidence_block_ids") or []), valid_block_ids),
            }
        )

    obligations = _normalize_item_list(list(effects.get("obligations") or []), prefix="OBL", valid_block_ids=valid_block_ids, keep_subject=True)
    prohibitions = _normalize_item_list(list(effects.get("prohibitions") or []), prefix="PROH", valid_block_ids=valid_block_ids, keep_subject=True)
    powers = _normalize_item_list(list(effects.get("powers") or []), prefix="POW", valid_block_ids=valid_block_ids, keep_subject=True)
    sanctions = _normalize_item_list(list(effects.get("sanctions") or []), prefix="SAN", valid_block_ids=valid_block_ids)
    eligibility_conditions = _normalize_item_list(list(effects.get("eligibility_conditions") or []), prefix="ELG", valid_block_ids=valid_block_ids)
    rights = _normalize_item_list(list(effects.get("rights") or []), prefix="RGT", valid_block_ids=valid_block_ids)

    legal_effects: list[dict[str, Any]] = []
    for bucket, effect_type in (
        (obligations, "obligation"),
        (prohibitions, "prohibition"),
        (powers, "power"),
        (sanctions, "consequence"),
        (eligibility_conditions, "eligibility"),
        (rights, "right"),
    ):
        for item in bucket:
            legal_effects.append(
                {
                    "id": item["id"],
                    "effect_type": effect_type,
                    "text": item["text"],
                    "confidence": item["confidence"],
                    "evidence_block_ids": item["evidence_block_ids"],
                }
            )

    facts: Facts = {
        "bill_id": bill.bill_id,
        "title": bill.title,
        "jurisdiction": bill.jurisdiction,
        "status": "unknown",
        "effective_date": applicability.get("effective_date"),
        "definitions": definitions,
        "applicability": applicability,
        "rights": rights,
        "eligibility_conditions": eligibility_conditions,
        "obligations": obligations,
        "prohibitions": prohibitions,
        "powers": powers,
        "sanctions": sanctions,
        "legal_effects": legal_effects,
        "trace": {
            "scope_mode": scope_trace.get("mode"),
            "effects_mode": effects_trace.get("mode"),
        },
    }
    return facts


def evaluate_quality(
    facts: Facts,
    *,
    config: Any,
    llm_client: Any | None,
) -> dict[str, Any]:
    applicability = facts.get("applicability") or {}
    obligations = list(facts.get("obligations") or [])
    prohibitions = list(facts.get("prohibitions") or [])
    sanctions = list(facts.get("sanctions") or [])
    critical_present = {
        "scope": bool(applicability.get("scope_summary") or applicability.get("covered_entities") or applicability.get("excluded_entities")),
        "obligations_or_prohibitions": bool(obligations or prohibitions),
        "sanctions": bool(sanctions),
    }
    field_coverage = round(sum(1 for value in critical_present.values() if value) / len(critical_present), 4)
    evidence_items = obligations + prohibitions + sanctions
    with_evidence = sum(1 for item in evidence_items if item.get("evidence_block_ids"))
    evidence_coverage = round(with_evidence / max(1, len(evidence_items)), 4) if evidence_items else 1.0
    issues: list[str] = []
    if applicability.get("applies_to_client") == "uncertain":
        issues.append("applicability unresolved")
    if any(len(str(item.get("text") or "")) < 24 for item in obligations[:5]):
        issues.append("obligation fragments detected")
    if any("heading" in str(item.get("text") or "").lower() for item in sanctions[:5]):
        issues.append("sanction heading leakage detected")

    deterministic = {
        "field_coverage": field_coverage,
        "evidence_coverage": evidence_coverage,
        "missing_fields": [key for key, value in critical_present.items() if not value],
        "issues": issues,
    }

    if config.dry_run or llm_client is None:
        score = round(0.45 * field_coverage + 0.55 * evidence_coverage, 4)
        verdict = "approved" if score >= 0.8 else "needs_review" if score >= 0.55 else "rejected"
        return {
            **deterministic,
            "judge_score": score,
            "judge_verdict": verdict,
            "judge_confidence": 0.0,
            "judge_issues": issues,
            "mode": "deterministic",
        }

    compact = {
        "title": facts.get("title"),
        "applicability": facts.get("applicability"),
        "counts": {
            "definitions": len(facts.get("definitions") or []),
            "rights": len(facts.get("rights") or []),
            "obligations": len(obligations),
            "prohibitions": len(prohibitions),
            "powers": len(facts.get("powers") or []),
            "sanctions": len(sanctions),
        },
        "top_obligations": [item.get("text") for item in obligations[:5]],
        "top_prohibitions": [item.get("text") for item in prohibitions[:5]],
        "top_sanctions": [item.get("text") for item in sanctions[:5]],
        "deterministic": deterministic,
    }
    llm = _call_json_completion(
        llm_client,
        config.llm_summary_model,
        prompt=_QUALITY_PROMPT.replace("__INPUT__", json.dumps(compact, ensure_ascii=False)),
        max_output_tokens=1600,
    )
    return {
        **deterministic,
        "judge_score": round(float(llm.get("judge_score") or 0.0), 4),
        "judge_verdict": str(llm.get("verdict") or "needs_review"),
        "judge_confidence": round(float(llm.get("confidence") or 0.0), 4),
        "judge_issues": _dedup_strings(list(llm.get("issues") or [])) or issues,
        "mode": "llm",
    }


def build_summary(facts: Facts, quality: dict[str, Any]) -> str:
    applicability = facts.get("applicability") or {}
    obligations = [item.get("text") for item in list(facts.get("obligations") or [])[:3]]
    prohibitions = [item.get("text") for item in list(facts.get("prohibitions") or [])[:3]]
    sanctions = [item.get("text") for item in list(facts.get("sanctions") or [])[:3]]
    definitions = [f"{item.get('term')}: {item.get('definition')}" for item in list(facts.get("definitions") or [])[:3]]
    thresholds_summary = _format_thresholds(list(applicability.get("thresholds") or []))
    logic_summary = applicability.get("condition_logic_summary") or "None extracted."
    sections = [
        f"CONTEXT - {facts.get('title') or 'Unknown law'}",
        f"SCOPE - {applicability.get('scope_summary') or 'No clear scope extracted.'}",
        f"THRESHOLDS - {thresholds_summary or 'None extracted.'}",
        f"LOGIC - {logic_summary}",
        f"EXCLUSIONS - {'; '.join(applicability.get('excluded_entities') or []) or 'None extracted.'}",
        f"OBLIGATIONS - {'; '.join(obligations) or 'None extracted.'}",
        f"PROHIBITIONS - {'; '.join(prohibitions) or 'None extracted.'}",
        f"CONSEQUENCES - {'; '.join(sanctions) or 'None extracted.'}",
        f"DEFINITIONS - {'; '.join(definitions) or 'None extracted.'}",
        f"QUALITY - verdict: {quality.get('judge_verdict')}; score: {quality.get('judge_score')}; issues: {'; '.join(quality.get('judge_issues') or []) or 'none'}",
    ]
    return "\n".join(sections)
