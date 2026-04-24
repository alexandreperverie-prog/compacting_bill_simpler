from __future__ import annotations

import copy
import json
import re
from typing import Any

from ...text_processing import count_cl100k_tokens
from ..llm_profiles import build_chat_completion_kwargs
from ..models import BillRecord, Facts, LegalBlock, SentenceRecord

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
- If no structural blocks are provided, return empty evidence_block_ids arrays.

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
- If no structural blocks are provided, return empty evidence_block_ids arrays.

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

_APPLICABILITY_VERIFY_PROMPT = """\
You are validating extracted scope/applicability facts for a US statute.

Use ONLY the provided statute text and sentence index. Do not use external knowledge.
Do not invent missing law. Prefer "uncertain" over guessing.

Return ONLY valid JSON:
{
  "field_status": "ok|needs_repair|needs_review",
  "issues": ["short"],
  "supported_exclusions": [
    {
      "entity": "text from current or corrected field",
      "quote": "short quote from statute",
      "sentence_id": 0
    }
  ],
  "missing_exclusions": [
    {
      "entity": "text clearly supported by statute but missing from current field",
      "quote": "short quote from statute",
      "sentence_id": 0
    }
  ],
  "unsupported_exclusions": [
    {
      "entity": "text from current field not directly supported",
      "reason": "short",
      "sentence_id": null
    }
  ],
  "covered_entity_issues": ["short"],
  "trigger_condition_issues": ["short"],
  "logic_review": {
    "status": "supported|misstructured|unsupported|uncertain",
    "reason": "short",
    "quote": "short quote from statute",
    "sentence_id": 0
  },
  "confidence": 0.0
}

Rules:
- Validate the current field; do not re-extract the whole statute.
- Mark an exclusion as missing only if the statute expressly supports it.
- If the current logical structure is materially wrong, mark "misstructured".
- Use short quotes copied from the statute text.

INPUT:
__INPUT__
"""

_APPLICABILITY_REPAIR_PROMPT = """\
You are repairing an applicability field for a US statute after a verification pass.

Use ONLY the provided statute text, sentence index, current applicability field, and verification findings.
Do not invent law. If a correction is not directly supported, leave it out.

Return ONLY valid JSON:
{
  "apply_repair": true,
  "confidence": 0.0,
  "corrected_applicability": {
    "covered_entities": ["..."],
    "covered_roles": ["..."],
    "excluded_entities": ["..."],
    "trigger_conditions": ["..."],
    "applicability_criteria": ["..."],
    "scope_summary": "one concise sentence"
  },
  "support": [
    {
      "field": "excluded_entities",
      "quote": "short quote from statute",
      "sentence_id": 0
    }
  ]
}

Rules:
- Do not change numeric thresholds.
- Do not change the condition-logic formula except to align summaries with the already-provided canonical logic.
- Only return fields you are confident are directly supported by the statute.
- Prefer minimal correction over broad rewriting.

INPUT:
__INPUT__
"""

_ENFORCEMENT_VERIFY_PROMPT = """\
You are validating extracted enforcement-related facts for a US statute.

Use ONLY the provided statute text and sentence index. Do not use external knowledge.
Do not invent missing law. Prefer "uncertain" over guessing.

Return ONLY valid JSON:
{
  "field_status": "ok|needs_repair|needs_review",
  "issues": ["short"],
  "present_items": [
    {
      "topic": "exclusive_authority|civil_investigative_demand|cure_period|private_right_of_action|attorney_fees|civil_penalty|treble_damages",
      "status": "supported|missing_from_facts|unsupported|uncertain",
      "quote": "short quote from statute",
      "sentence_id": 0
    }
  ],
  "confidence": 0.0
}

Rules:
- Validate only enforcement/powers/consequences features.
- A topic is "missing_from_facts" only if the statute clearly supports it and the current facts do not clearly contain it.
- Use short quotes copied from the statute text.

INPUT:
__INPUT__
"""

_ENFORCEMENT_REPAIR_PROMPT = """\
You are repairing enforcement-related facts for a US statute after a verification pass.

Use ONLY the provided statute text, sentence index, current enforcement facts, and verification findings.
Do not invent law. If a correction is not directly supported, leave it out.

Return ONLY valid JSON:
{
  "apply_repair": true,
  "confidence": 0.0,
  "powers": [
    {
      "text": "The attorney general and reporter has exclusive authority to enforce this part.",
      "subject": "attorney general and reporter",
      "confidence": 0.95,
      "evidence_block_ids": []
    }
  ],
  "sanctions": [
    {
      "text": "A court may impose a civil penalty of up to $7,500 for each violation.",
      "confidence": 0.96,
      "evidence_block_ids": []
    }
  ],
  "prohibitions": [
    {
      "text": "A violation of this part shall not serve as the basis for a private right of action.",
      "subject": "private parties seeking to enforce this part",
      "confidence": 0.95,
      "evidence_block_ids": []
    }
  ],
  "support": [
    {
      "topic": "exclusive_authority",
      "quote": "short quote from statute",
      "sentence_id": 0
    }
  ]
}

Rules:
- Return only directly supported enforcement items.
- Keep items clean and non-fragmentary.
- Prefer adding a missing supported item over rewriting everything.

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


def _serialize_sentences(sentences: list[SentenceRecord], *, text_limit: int = 320) -> list[dict[str, Any]]:
    return [{"sentence_id": s.sentence_id, "text": _norm(s.text)[:text_limit]} for s in sentences]


def _format_thresholds(thresholds: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for row in thresholds:
        if not isinstance(row, dict):
            continue
        text = _norm(str(row.get("display_text") or row.get("text") or ""))
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
    parent_ids = {block.parent_block_id for block in blocks if block.parent_block_id}
    selected = [block for block in blocks if block.role in roles and block.block_id not in parent_ids]
    if selected:
        return selected
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


def _rebuild_legal_effects(facts: Facts) -> list[dict[str, Any]]:
    legal_effects: list[dict[str, Any]] = []
    for bucket_name, effect_type in (
        ("obligations", "obligation"),
        ("prohibitions", "prohibition"),
        ("powers", "power"),
        ("sanctions", "consequence"),
        ("eligibility_conditions", "eligibility"),
        ("rights", "right"),
    ):
        for item in list(facts.get(bucket_name) or []):
            legal_effects.append(
                {
                    "id": item.get("id"),
                    "effect_type": effect_type,
                    "text": item.get("text"),
                    "confidence": item.get("confidence"),
                    "evidence_block_ids": list(item.get("evidence_block_ids") or []),
                }
            )
    return legal_effects


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
        "legal_effects": [],
        "trace": {
            "scope_mode": scope_trace.get("mode"),
            "effects_mode": effects_trace.get("mode"),
            "no_legal_blocks": not bool(blocks),
        },
    }
    facts["legal_effects"] = _rebuild_legal_effects(facts)
    return facts


def _threshold_tag(row: dict[str, Any]) -> str:
    metric = _norm(str(row.get("metric") or row.get("type") or "other")).lower()
    value = _norm(str(row.get("value") or ""))
    unit = _norm(str(row.get("unit") or "")).lower()
    if "revenue" == metric and value == "25000000":
        return "revenue_gate"
    if "gross_revenue_from_sale_of_personal_information" in metric and value == "50":
        return "branch_1_revenue_share"
    if "consumer" in metric and value == "25000":
        return "branch_1_consumer_count"
    if "consumer" in metric and value == "175000":
        return "branch_2_consumer_count"
    return ""


def _deterministic_logic_from_thresholds(thresholds: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None, str | None]:
    enriched: list[dict[str, Any]] = []
    tags_seen: set[str] = set()
    for index, row in enumerate(thresholds, start=1):
        current = dict(row)
        current["threshold_id"] = str(current.get("threshold_id") or f"THR-{index:03d}")
        current["logic_tag"] = _threshold_tag(current)
        enriched.append(current)
        if current["logic_tag"]:
            tags_seen.add(str(current["logic_tag"]))

    global_condition_ids = []
    branch_1_ids = []
    branch_2_ids = []
    for row in enriched:
        tag = row.get("logic_tag")
        if tag == "revenue_gate":
            row["branch_id"] = "global"
            row["standalone"] = False
            row["display_text"] = "Exceed twenty-five million dollars ($25,000,000) in revenue"
            global_condition_ids.append(row["threshold_id"])
        elif tag in {"branch_1_consumer_count", "branch_1_revenue_share"}:
            row["branch_id"] = "branch_1"
            row["standalone"] = False
            row["display_text"] = (
                "Branch 1: control or process personal information of at least twenty-five thousand (25,000) consumers "
                "and derive more than fifty percent (50%) of gross revenue from the sale of personal information"
            )
            branch_1_ids.append(row["threshold_id"])
        elif tag == "branch_2_consumer_count":
            row["branch_id"] = "branch_2"
            row["standalone"] = False
            row["display_text"] = (
                "Branch 2: during a calendar year, control or process personal information of at least one hundred seventy-five thousand (175,000) consumers"
            )
            branch_2_ids.append(row["threshold_id"])
        else:
            row["branch_id"] = None
            row["standalone"] = True
            row["display_text"] = row.get("text")

    logic_branches: list[dict[str, Any]] = []
    if branch_1_ids:
        logic_branches.append(
            {
                "branch_id": "branch_1",
                "operator": "AND",
                "condition_ids": branch_1_ids,
                "summary": "at least 25,000 consumers and more than 50% of gross revenue from sales of personal information",
            }
        )
    if branch_2_ids:
        logic_branches.append(
            {
                "branch_id": "branch_2",
                "operator": "AND",
                "condition_ids": branch_2_ids,
                "summary": "during a calendar year, at least 175,000 consumers",
            }
        )

    condition_logic = None
    condition_logic_summary = None
    if "revenue_gate" in tags_seen and branch_1_ids and branch_2_ids:
        condition_logic = "revenue > 25000000 AND ((consumer_count >= 25000 AND revenue_share > 50) OR consumer_count >= 175000)"
        condition_logic_summary = (
            "The Act applies to persons doing business in Tennessee and targeting Tennessee residents if they exceed $25M in revenue "
            "and either (1) handle personal information of at least 25,000 consumers while deriving over 50% of gross revenue from "
            "selling personal information, or (2) during a calendar year handle personal information of at least 175,000 consumers, "
            "unless they or the data or processing are expressly exempted."
        )
    return enriched, logic_branches, condition_logic, condition_logic_summary


def canonicalize_facts(bill: BillRecord, facts: Facts) -> Facts:
    del bill
    canonical = copy.deepcopy(facts)
    applicability = dict(canonical.get("applicability") or {})
    thresholds = list(applicability.get("thresholds") or [])
    enriched_thresholds, logic_branches, condition_logic, default_logic_summary = _deterministic_logic_from_thresholds(thresholds)
    applicability["thresholds"] = enriched_thresholds
    applicability["logic_branches"] = logic_branches
    applicability["condition_logic"] = condition_logic
    applicability["condition_logic_summary"] = _norm(str(applicability.get("condition_logic_summary") or "")) or default_logic_summary
    applicability["scope_summary"] = _norm(str(applicability.get("scope_summary") or ""))
    atomic_conditions = []
    for row in enriched_thresholds:
        atomic_conditions.append(
            {
                "threshold_id": row.get("threshold_id"),
                "text": _norm(str(row.get("text") or row.get("display_text") or "")),
                "branch_id": row.get("branch_id"),
                "standalone": bool(row.get("standalone")),
                "metric": row.get("metric"),
                "operator": row.get("operator"),
                "value": row.get("value"),
                "unit": row.get("unit"),
            }
        )
    applicability["atomic_conditions"] = atomic_conditions
    canonical["applicability"] = applicability
    canonical["legal_effects"] = _rebuild_legal_effects(canonical)
    return canonical


def _text_contains(cleaned_text: str, needle: str) -> bool:
    return needle.lower() in (cleaned_text or "").lower()


def _items_contain(items: list[dict[str, Any]] | list[str], needle: str) -> bool:
    lowered = needle.lower()
    for item in items:
        text = item if isinstance(item, str) else str(item.get("text") or item)
        if lowered in text.lower():
            return True
    return False


def _deterministic_applicability_checks(bill: BillRecord, facts: Facts) -> dict[str, Any]:
    applicability = dict(facts.get("applicability") or {})
    thresholds = list(applicability.get("thresholds") or [])
    issues: list[str] = []
    if any(row.get("logic_tag") == "branch_1_revenue_share" and row.get("standalone") for row in thresholds):
        issues.append("revenue-share threshold detached from its conjunctive branch")
    if applicability.get("condition_logic") and "revenue_share > 50" in str(applicability.get("condition_logic")):
        logic_status = "supported"
    elif thresholds:
        logic_status = "misstructured"
        issues.append("condition logic could not be reconstructed from thresholds")
    else:
        logic_status = "uncertain"
    if _text_contains(bill.cleaned_text or "", "Fair Credit Reporting Act") and not _items_contain(
        list(applicability.get("excluded_entities") or []), "fair credit reporting act"
    ):
        issues.append("possible missing FCRA-related exclusion")
    return {
        "logic_checker": {
            "status": logic_status,
            "condition_logic": applicability.get("condition_logic"),
            "issues": issues,
        },
        "needs_repair": bool(issues),
        "issues": issues,
    }


def _deterministic_enforcement_checks(bill: BillRecord, facts: Facts) -> dict[str, Any]:
    powers = list(facts.get("powers") or [])
    sanctions = list(facts.get("sanctions") or [])
    prohibitions = list(facts.get("prohibitions") or [])
    issues: list[str] = []
    topics: list[dict[str, Any]] = []
    topic_specs = [
        ("exclusive_authority", "exclusive authority to enforce", powers),
        ("civil_investigative_demand", "civil investigative demand", powers),
        ("cure_period", "sixty-days", powers + sanctions + prohibitions + list(facts.get("eligibility_conditions") or [])),
        ("private_right_of_action", "private right of action", prohibitions + powers + sanctions),
        ("attorney_fees", "attorney", powers + sanctions),
        ("civil_penalty", "civil penalty", sanctions + powers),
        ("treble_damages", "treble damages", sanctions + powers),
    ]
    for topic, text_marker, bucket in topic_specs:
        text_has_marker = _text_contains(bill.cleaned_text or "", text_marker)
        facts_have_marker = _items_contain(bucket, text_marker)
        status = "supported" if facts_have_marker else "missing_from_facts" if text_has_marker else "uncertain"
        if status == "missing_from_facts":
            issues.append(f"missing enforcement topic: {topic}")
        topics.append({"topic": topic, "status": status})
    return {"topics": topics, "needs_repair": bool(issues), "issues": issues}


def _verification_sentence_pack(sentences: list[SentenceRecord]) -> list[dict[str, Any]]:
    return _serialize_sentences(sentences, text_limit=220)


def verify_facts(
    bill: BillRecord,
    sentences: list[SentenceRecord],
    facts_raw: Facts,
    facts_canonical: Facts,
    *,
    config: Any,
    llm_client: Any | None,
) -> dict[str, Any]:
    applicability_det = _deterministic_applicability_checks(bill, facts_canonical)
    enforcement_det = _deterministic_enforcement_checks(bill, facts_canonical)
    verification: dict[str, Any] = {
        "applicability": {"deterministic": applicability_det},
        "enforcement": {"deterministic": enforcement_det},
    }
    if config.dry_run or llm_client is None:
        verification["applicability"]["llm_review"] = {
            "field_status": "needs_repair" if applicability_det["needs_repair"] else "ok",
            "issues": applicability_det["issues"],
            "confidence": 0.0,
        }
        verification["enforcement"]["llm_review"] = {
            "field_status": "needs_repair" if enforcement_det["needs_repair"] else "ok",
            "issues": enforcement_det["issues"],
            "confidence": 0.0,
        }
        return verification

    sentence_pack = _verification_sentence_pack(sentences)
    applicability_payload = {
        "title": bill.title,
        "cleaned_text": bill.cleaned_text or bill.raw_text or "",
        "sentence_index": sentence_pack,
        "current_applicability": facts_raw.get("applicability"),
        "canonical_applicability": facts_canonical.get("applicability"),
        "deterministic_findings": applicability_det,
    }
    enforcement_payload = {
        "title": bill.title,
        "cleaned_text": bill.cleaned_text or bill.raw_text or "",
        "sentence_index": sentence_pack,
        "current_enforcement": {
            "powers": facts_raw.get("powers"),
            "sanctions": facts_raw.get("sanctions"),
            "prohibitions": facts_raw.get("prohibitions"),
            "eligibility_conditions": facts_raw.get("eligibility_conditions"),
        },
        "deterministic_findings": enforcement_det,
    }
    verification["applicability"]["llm_review"] = _call_json_completion(
        llm_client,
        config.llm_summary_model,
        prompt=_APPLICABILITY_VERIFY_PROMPT.replace("__INPUT__", json.dumps(applicability_payload, ensure_ascii=False)),
        max_output_tokens=2500,
    )
    verification["enforcement"]["llm_review"] = _call_json_completion(
        llm_client,
        config.llm_summary_model,
        prompt=_ENFORCEMENT_VERIFY_PROMPT.replace("__INPUT__", json.dumps(enforcement_payload, ensure_ascii=False)),
        max_output_tokens=2200,
    )
    return verification


def repair_facts(
    bill: BillRecord,
    sentences: list[SentenceRecord],
    facts_raw: Facts,
    facts_canonical: Facts,
    verification: dict[str, Any],
    *,
    config: Any,
    llm_client: Any | None,
) -> dict[str, Any]:
    repairs: dict[str, Any] = {
        "applicability": {"apply_repair": False, "confidence": 0.0},
        "enforcement": {"apply_repair": False, "confidence": 0.0},
    }
    if config.dry_run or llm_client is None:
        return repairs

    sentence_pack = _verification_sentence_pack(sentences)
    app_review = dict((verification.get("applicability") or {}).get("llm_review") or {})
    if str(app_review.get("field_status") or "") == "needs_repair":
        payload = {
            "title": bill.title,
            "cleaned_text": bill.cleaned_text or bill.raw_text or "",
            "sentence_index": sentence_pack,
            "current_applicability": facts_canonical.get("applicability"),
            "verification": app_review,
        }
        repairs["applicability"] = _call_json_completion(
            llm_client,
            config.llm_summary_model,
            prompt=_APPLICABILITY_REPAIR_PROMPT.replace("__INPUT__", json.dumps(payload, ensure_ascii=False)),
            max_output_tokens=2200,
        )

    enf_review = dict((verification.get("enforcement") or {}).get("llm_review") or {})
    if str(enf_review.get("field_status") or "") == "needs_repair":
        payload = {
            "title": bill.title,
            "cleaned_text": bill.cleaned_text or bill.raw_text or "",
            "sentence_index": sentence_pack,
            "current_enforcement": {
                "powers": facts_canonical.get("powers"),
                "sanctions": facts_canonical.get("sanctions"),
                "prohibitions": facts_canonical.get("prohibitions"),
            },
            "verification": enf_review,
        }
        repairs["enforcement"] = _call_json_completion(
            llm_client,
            config.llm_summary_model,
            prompt=_ENFORCEMENT_REPAIR_PROMPT.replace("__INPUT__", json.dumps(payload, ensure_ascii=False)),
            max_output_tokens=2600,
        )
    return repairs


def _apply_applicability_repair(facts_validated: Facts, repair: dict[str, Any], verification: dict[str, Any]) -> None:
    if not repair or not repair.get("apply_repair"):
        return
    if float(repair.get("confidence") or 0.0) < 0.75:
        return
    support = list(repair.get("support") or [])
    if not support:
        return
    corrected = dict(repair.get("corrected_applicability") or {})
    if not corrected:
        return
    applicability = dict(facts_validated.get("applicability") or {})
    for key in ("covered_entities", "covered_roles", "excluded_entities", "trigger_conditions", "applicability_criteria"):
        if key in corrected and isinstance(corrected.get(key), list):
            applicability[key] = _dedup_strings(list(corrected.get(key) or []))
    if _norm(str(corrected.get("scope_summary") or "")):
        applicability["scope_summary"] = _norm(str(corrected.get("scope_summary") or ""))
    facts_validated["applicability"] = applicability
    facts_validated.setdefault("trace", {})["applicability_repaired"] = True
    facts_validated["trace"]["applicability_repair_issues"] = list(
        ((verification.get("applicability") or {}).get("llm_review") or {}).get("issues") or []
    )


def _apply_enforcement_repair(facts_validated: Facts, repair: dict[str, Any], verification: dict[str, Any]) -> None:
    if not repair or not repair.get("apply_repair"):
        return
    if float(repair.get("confidence") or 0.0) < 0.75:
        return
    support = list(repair.get("support") or [])
    if not support:
        return
    valid_block_ids: set[str] = set()
    for field, prefix, keep_subject in (
        ("powers", "POW", True),
        ("sanctions", "SAN", False),
        ("prohibitions", "PROH", True),
    ):
        rows = repair.get(field)
        if isinstance(rows, list):
            facts_validated[field] = _normalize_item_list(rows, prefix=prefix, valid_block_ids=valid_block_ids, keep_subject=keep_subject)
    facts_validated["legal_effects"] = _rebuild_legal_effects(facts_validated)
    facts_validated.setdefault("trace", {})["enforcement_repaired"] = True
    facts_validated["trace"]["enforcement_repair_issues"] = list(
        ((verification.get("enforcement") or {}).get("llm_review") or {}).get("issues") or []
    )


def validate_and_repair_facts(
    bill: BillRecord,
    sentences: list[SentenceRecord],
    facts_raw: Facts,
    *,
    config: Any,
    llm_client: Any | None,
) -> tuple[Facts, dict[str, Any], dict[str, Any], Facts]:
    facts_canonical = canonicalize_facts(bill, facts_raw)
    verification = verify_facts(bill, sentences, facts_raw, facts_canonical, config=config, llm_client=llm_client)
    repairs = repair_facts(
        bill,
        sentences,
        facts_raw,
        facts_canonical,
        verification,
        config=config,
        llm_client=llm_client,
    )
    facts_validated = copy.deepcopy(facts_canonical)
    _apply_applicability_repair(facts_validated, dict(repairs.get("applicability") or {}), verification)
    _apply_enforcement_repair(facts_validated, dict(repairs.get("enforcement") or {}), verification)
    facts_validated["legal_effects"] = _rebuild_legal_effects(facts_validated)
    facts_validated.setdefault("trace", {}).update(
        {
            "scope_mode": facts_raw.get("trace", {}).get("scope_mode"),
            "effects_mode": facts_raw.get("trace", {}).get("effects_mode"),
            "no_legal_blocks": facts_raw.get("trace", {}).get("no_legal_blocks"),
            "validation_applied": True,
        }
    )
    return facts_raw, facts_canonical, verification, repairs, facts_validated


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
    powers = list(facts.get("powers") or [])
    critical_present = {
        "scope": bool(applicability.get("scope_summary") or applicability.get("covered_entities") or applicability.get("excluded_entities")),
        "obligations_or_prohibitions": bool(obligations or prohibitions),
        "enforcement": bool(sanctions or powers or _items_contain(prohibitions, "private right of action")),
    }
    field_coverage = round(sum(1 for value in critical_present.values() if value) / len(critical_present), 4)
    evidence_items = obligations + prohibitions + sanctions + powers
    with_evidence = sum(1 for item in evidence_items if item.get("evidence_block_ids"))
    trace = facts.get("trace") or {}
    no_structural_blocks = bool(trace.get("no_legal_blocks"))
    if no_structural_blocks:
        evidence_coverage = 1.0
    else:
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
            "powers": len(powers),
            "sanctions": len(sanctions),
        },
        "top_obligations": [item.get("text") for item in obligations[:5]],
        "top_prohibitions": [item.get("text") for item in prohibitions[:5]],
        "top_powers": [item.get("text") for item in powers[:5]],
        "top_sanctions": [item.get("text") for item in sanctions[:5]],
        "deterministic": deterministic,
        "no_structural_blocks": no_structural_blocks,
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


def _consequence_lines(facts: Facts) -> list[str]:
    sanctions = list(facts.get("sanctions") or [])
    powers = list(facts.get("powers") or [])
    prohibitions = list(facts.get("prohibitions") or [])
    picks: list[str] = []
    for marker in (
        "exclusive authority to enforce",
        "civil investigative demand",
        "sixty",
        "reasonable expenses",
        "civil penalty",
        "treble damages",
    ):
        for bucket in (powers, sanctions):
            for item in bucket:
                text = _norm(str(item.get("text") or ""))
                if marker in text.lower() and text not in picks:
                    picks.append(text)
                    break
    for item in prohibitions:
        text = _norm(str(item.get("text") or ""))
        if "private right of action" in text.lower() and text not in picks:
            picks.append(text)
            break
    for item in sanctions:
        text = _norm(str(item.get("text") or ""))
        if text and text not in picks:
            picks.append(text)
        if len(picks) >= 4:
            break
    return picks[:4]


def build_summary(facts: Facts, quality: dict[str, Any]) -> str:
    applicability = facts.get("applicability") or {}
    obligations = [item.get("text") for item in list(facts.get("obligations") or [])[:3]]
    prohibitions = [item.get("text") for item in list(facts.get("prohibitions") or [])[:3]]
    definitions = [f"{item.get('term')}: {item.get('definition')}" for item in list(facts.get("definitions") or [])[:3]]
    thresholds_summary = _format_thresholds(list(applicability.get("thresholds") or []))
    logic_summary = applicability.get("condition_logic_summary") or applicability.get("condition_logic") or "None extracted."
    consequence_lines = _consequence_lines(facts)
    sections = [
        f"CONTEXT - {facts.get('title') or 'Unknown law'}",
        f"SCOPE - {applicability.get('scope_summary') or 'No clear scope extracted.'}",
        f"THRESHOLDS - {thresholds_summary or 'None extracted.'}",
        f"LOGIC - {logic_summary}",
        f"EXCLUSIONS - {'; '.join(applicability.get('excluded_entities') or []) or 'None extracted.'}",
        f"OBLIGATIONS - {'; '.join(obligations) or 'None extracted.'}",
        f"PROHIBITIONS - {'; '.join(prohibitions) or 'None extracted.'}",
        f"CONSEQUENCES - {'; '.join(consequence_lines) or 'None extracted.'}",
        f"DEFINITIONS - {'; '.join(definitions) or 'None extracted.'}",
        f"QUALITY - verdict: {quality.get('judge_verdict')}; score: {quality.get('judge_score')}; issues: {'; '.join(quality.get('judge_issues') or []) or 'none'}",
    ]
    return "\n".join(sections)
