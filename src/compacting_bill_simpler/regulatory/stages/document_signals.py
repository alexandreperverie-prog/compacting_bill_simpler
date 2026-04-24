from __future__ import annotations

import re
from typing import Any

from ..models import ChunkRecord

_OPERATIVE_RES = (
    re.compile(
        r"\b(?:be it enacted|section\s+\d+|sec\.\s*\d+|is amended|are amended|is repealed|are repealed|"
        r"repealed and recreated|is added|are added|shall read|becomes operative|takes effect|effective date)\b",
        re.I,
    ),
    re.compile(r"\b(?:shall|must|may not|shall not|must not|cannot)\b", re.I),
)
_BACKGROUND_RES = (
    re.compile(
        r"\b(?:analysis by|current law|this bill|bill title|summary|status:|sponsors?|synopsis|"
        r"for further information|introduced|referred to|analysis as introduced)\b",
        re.I,
    ),
    re.compile(r"\b(?:legislative reference bureau|legiscan|state legislature page)\b", re.I),
)
_METADATA_RES = (
    re.compile(r"\b(?:jump to navigation|main menu|register|login|search|comments|track|research|download:)\b", re.I),
    re.compile(r"\b(?:lrb[-\d/]+|lrb\d+)\b", re.I),
    re.compile(r"\b(?:view top 50|feedback|contact us|terms)\b", re.I),
)
_DEFINITION_RE = re.compile(r"\b(?:as used in this|means\b|definition|defined as)\b", re.I)
_THRESHOLD_RE = re.compile(
    r"(?:\b(?:more than|less than|at least|not exceeding|exceeds?)\b\s+\$?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s+(?:days?|years?|hours?|tons?|gallons?|cubic yards?|employees|consumers))",
    re.I,
)
_CROSS_REF_RE = re.compile(r"\b(?:ORS|ILCS|U\.S\.C\.|Tennessee Code Annotated|section\s+\d+|§+\s*[\d\-\.()]+)\b", re.I)
_AMENDMENT_RE = re.compile(r"\b(?:is amended|are amended|amended by|shall read|adding section|new section)\b", re.I)
_REPEAL_RE = re.compile(r"\b(?:is repealed|are repealed|repeal and recreate|repealed and recreated)\b", re.I)
_SYNOPSIS_RE = re.compile(r"\b(?:synopsis|summary|analysis by|current law|bill title|status:|download:)\b", re.I)
_CURRENT_LAW_RE = re.compile(r"\b(?:current law|under current law|subject to certain exceptions, current law)\b", re.I)
_BILL_EFFECT_RE = re.compile(r"\b(?:this bill|this act)\b.*\b(?:eliminates|repeals?|amends?|creates?|requires?|establishes?|supersedes?|preempts?)\b", re.I)
_OPERATIONAL_RULE_RE = re.compile(
    r"\b(?:section\s+\d+|sec\.\s*\d+|is amended|are amended|is repealed|are repealed|repealed and recreated|shall read|takes effect)\b",
    re.I,
)
_BACKGROUND_EXPLANATION_RE = re.compile(r"\b(?:analysis by|summary|synopsis|for further information|sponsors?|introduced|referred to)\b", re.I)
_SCOPE_HEADING_RE = re.compile(r"\b(?:scope|applicability|coverage|covered entities|exemptions?)\b", re.I)
_SCOPE_ANCHOR_RE = re.compile(
    r"\b(?:this\s+(?:act|part|section|chapter)\s+applies\s+to|applies\s+to|does\s+not\s+apply\s+to|shall\s+not\s+apply\s+to)\b",
    re.I,
)
_SCOPE_NUMERIC_RE = re.compile(
    r"\b(?:more than|over|at least|not less than|greater than|under|less than|<=?|>=?)\b[^.;,]{0,80}\b(?:consumers?|residents?|households?|employees?|revenue|turnover|sales)\b",
    re.I,
)
_REVENUE_SHARE_RE = re.compile(r"\b\d{1,3}\s*(?:percent|%)\s+of\s+[^.;,]{0,120}?(?:revenue|sales|turnover)\b", re.I)
_SCOPE_CONDITION_RE = re.compile(r"\b(?:if|when|only if|unless|except|provided that)\b", re.I)
_OBLIGATION_HEAVY_RE = re.compile(
    r"\b(?:privacy notice|consumer rights|respond to the consumer|free of charge|appeal process|authenticate the request)\b",
    re.I,
)


def _clip_score(hits: int, *, factor: float = 0.35) -> float:
    return round(min(1.0, hits * factor), 4)


def classify_text_zone(text: str) -> dict[str, Any]:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return {
            "label": "unknown",
            "operative_score": 0.0,
            "background_score": 0.0,
            "noise_score": 0.0,
            "definition_score": 0.0,
            "threshold_score": 0.0,
            "cross_reference_score": 0.0,
        }

    operative_hits = sum(1 for pattern in _OPERATIVE_RES if pattern.search(cleaned))
    background_hits = sum(1 for pattern in _BACKGROUND_RES if pattern.search(cleaned))
    metadata_hits = sum(1 for pattern in _METADATA_RES if pattern.search(cleaned))
    definition_hits = 1 if _DEFINITION_RE.search(cleaned) else 0
    threshold_hits = len(_THRESHOLD_RE.findall(cleaned))
    cross_ref_hits = len(_CROSS_REF_RE.findall(cleaned))

    operative_score = _clip_score(operative_hits, factor=0.38)
    background_score = _clip_score(background_hits, factor=0.45)
    noise_score = _clip_score(metadata_hits, factor=0.6)
    definition_score = _clip_score(definition_hits, factor=0.5)
    threshold_score = _clip_score(min(threshold_hits, 3), factor=0.34)
    cross_reference_score = _clip_score(min(cross_ref_hits, 4), factor=0.2)

    if noise_score >= 0.6:
        label = "noise"
    elif metadata_hits and operative_score < 0.3 and background_score < 0.45:
        label = "metadata"
    elif operative_score >= max(0.38, background_score + 0.12):
        label = "operative"
    elif background_score >= 0.45:
        label = "background"
    else:
        label = "unknown"

    return {
        "label": label,
        "operative_score": operative_score,
        "background_score": background_score,
        "noise_score": noise_score,
        "definition_score": definition_score,
        "threshold_score": threshold_score,
        "cross_reference_score": cross_reference_score,
    }


def classify_statement_role(text: str) -> dict[str, Any]:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return {"label": "unknown", "scores": {}}

    scores = {
        "describes_current_law": 0.0,
        "states_bill_effect": 0.0,
        "background_or_explanation": 0.0,
        "operational_rule_text": 0.0,
    }
    if _CURRENT_LAW_RE.search(cleaned):
        scores["describes_current_law"] += 1.0
    if _BILL_EFFECT_RE.search(cleaned):
        scores["states_bill_effect"] += 1.0
    if _OPERATIONAL_RULE_RE.search(cleaned):
        scores["operational_rule_text"] += 1.0
    if _BACKGROUND_EXPLANATION_RE.search(cleaned):
        scores["background_or_explanation"] += 1.0

    zone = classify_text_zone(cleaned)
    if zone["label"] == "background":
        scores["background_or_explanation"] += 0.4
    if zone["label"] == "operative":
        scores["operational_rule_text"] += 0.25
    if "this bill" in cleaned.lower() and not _OPERATIONAL_RULE_RE.search(cleaned):
        scores["states_bill_effect"] += 0.35

    best_label = max(scores, key=scores.get)
    if scores[best_label] <= 0.0:
        best_label = "unknown"
    return {
        "label": best_label,
        "scores": {key: round(value, 4) for key, value in scores.items()},
    }


def profile_document(cleaned_text: str, chunks: list[ChunkRecord]) -> dict[str, Any]:
    chunk_profiles = [classify_text_zone(chunk.text) for chunk in chunks]
    n_chunks = max(1, len(chunk_profiles))
    operative_chunks = sum(1 for item in chunk_profiles if item["label"] == "operative")
    background_chunks = sum(1 for item in chunk_profiles if item["label"] == "background")
    noise_chunks = sum(1 for item in chunk_profiles if item["label"] == "noise")

    text = cleaned_text or " ".join(chunk.text for chunk in chunks[:8])
    return {
        "operative_signal": round(sum(item["operative_score"] for item in chunk_profiles) / n_chunks, 4),
        "background_analysis_signal": round(sum(item["background_score"] for item in chunk_profiles) / n_chunks, 4),
        "web_noise_signal": round(sum(item["noise_score"] for item in chunk_profiles) / n_chunks, 4),
        "definition_signal": round(sum(item["definition_score"] for item in chunk_profiles) / n_chunks, 4),
        "threshold_signal": round(sum(item["threshold_score"] for item in chunk_profiles) / n_chunks, 4),
        "cross_reference_density": round(sum(item["cross_reference_score"] for item in chunk_profiles) / n_chunks, 4),
        "amendment_signal": _clip_score(len(_AMENDMENT_RE.findall(text)), factor=0.12),
        "repeal_signal": _clip_score(len(_REPEAL_RE.findall(text)), factor=0.2),
        "synopsis_signal": _clip_score(len(_SYNOPSIS_RE.findall(text)), factor=0.16),
        "operative_chunk_ratio": round(operative_chunks / n_chunks, 4),
        "background_chunk_ratio": round(background_chunks / n_chunks, 4),
        "noise_chunk_ratio": round(noise_chunks / n_chunks, 4),
    }


def score_chunk_for_family(
    family: str,
    chunk_text: str,
    document_profile: dict[str, Any] | None = None,
    *,
    use_structural_rerank: bool = True,
    use_background_penalty: bool = True,
) -> float:
    if not use_structural_rerank:
        return 1.0

    zone = classify_text_zone(chunk_text)
    document_profile = document_profile or {}
    weight = 1.0

    if family in {"obligations", "powers", "prohibitions", "sanctions", "eligibility_conditions", "contract_exclusions", "indirect_restrictions"}:
        weight *= 1.0 + 0.45 * zone["operative_score"]
        if use_background_penalty:
            weight *= 1.0 - 0.42 * zone["background_score"]
        weight *= 1.0 - 0.7 * zone["noise_score"]
        if (
            use_background_penalty
            and document_profile.get("background_analysis_signal", 0.0) >= 0.4
            and zone["background_score"] >= 0.45
        ):
            weight *= 0.78
    elif family in {"applicability", "scope_conditions"}:
        weight *= 1.0 + 0.24 * zone["operative_score"]
        weight *= 1.0 - 0.28 * zone["noise_score"]
        weight *= 1.0 + 0.12 * zone["threshold_score"]
        if _SCOPE_HEADING_RE.search(chunk_text):
            weight *= 1.18
        if _SCOPE_ANCHOR_RE.search(chunk_text):
            weight *= 1.35
        if _SCOPE_NUMERIC_RE.search(chunk_text):
            weight *= 1.18
        if _REVENUE_SHARE_RE.search(chunk_text):
            weight *= 1.16
        if family == "scope_conditions" and _SCOPE_CONDITION_RE.search(chunk_text):
            weight *= 1.08
        if family == "applicability" and _OBLIGATION_HEAVY_RE.search(chunk_text) and not _SCOPE_ANCHOR_RE.search(chunk_text):
            weight *= 0.82
        if family == "applicability" and not (
            _SCOPE_HEADING_RE.search(chunk_text)
            or _SCOPE_ANCHOR_RE.search(chunk_text)
            or _SCOPE_NUMERIC_RE.search(chunk_text)
            or _REVENUE_SHARE_RE.search(chunk_text)
        ):
            weight *= 0.72
        if family == "scope_conditions" and not (
            _SCOPE_CONDITION_RE.search(chunk_text)
            or _SCOPE_ANCHOR_RE.search(chunk_text)
            or _SCOPE_NUMERIC_RE.search(chunk_text)
        ):
            weight *= 0.8
    elif family in {"status_link"}:
        weight *= 1.0 + 0.18 * zone["operative_score"]
        if use_background_penalty:
            weight *= 1.0 + 0.08 * zone["background_score"]
        weight *= 1.0 - 0.28 * zone["noise_score"]
    elif family in {"context", "policy_intent", "global_context"}:
        weight *= 1.0 - 0.2 * zone["noise_score"]
        if use_background_penalty:
            weight *= 1.0 + 0.08 * zone["background_score"]
        weight *= 1.0 + 0.08 * zone["operative_score"]
    else:
        weight *= 1.0 - 0.25 * zone["noise_score"]

    if family == "eligibility_conditions" and zone["definition_score"] >= 0.45:
        weight *= 0.82
    if family == "obligations" and zone["threshold_score"] >= 0.34:
        weight *= 1.08
    if family == "sanctions" and re.search(r"\b(?:fine|penalt|liable|damages|disqualif|ineligib|revok|suspend)\b", chunk_text, re.I):
        weight *= 1.2
    if family == "prohibitions" and re.search(r"\b(?:shall not|may not|must not|cannot|prohibited)\b", chunk_text, re.I):
        weight *= 1.2
    if family == "powers" and re.search(r"\b(?:agency|department|board|attorney general|commission|court)\b.*\bmay\b", chunk_text, re.I):
        weight *= 1.18

    return round(max(0.15, min(1.8, weight)), 4)


def is_descriptive_background_text(text: str) -> bool:
    low = str(text or "").lower()
    if any(
        token in low
        for token in (
            "current law",
            "analysis by",
            "summary",
            "bill title",
            "status:",
            "sponsors",
            "for further information",
            "state legislature page",
        )
    ):
        return True
    return False
