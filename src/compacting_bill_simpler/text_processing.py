"""
Text preprocessing for legislative bills.

Provides:
  - clean_legislative_text(raw) -> str
  - count_cl100k_tokens(text) -> int
  - segment_into_sentences(cleaned_text) -> list[dict]
  - chunk_text_by_tokens(text, chunk_size, overlap) -> list[str]

These functions were previously duplicated across all notebooks.
Single canonical source of truth.
"""
from __future__ import annotations

import re
from typing import Any

import tiktoken
from spacy.language import Language
from spacy.lang.en import English

# ---------------------------------------------------------------------------
# spaCy pipeline (sentencizer only — no heavy NLP needed)
# ---------------------------------------------------------------------------

_nlp = None
_ENUMERATOR_VALUE_RE = re.compile(r"^(?:\d+|[A-Za-z]|[ivxlcdmIVXLCDM]{1,6})$")
_SECTION_TOKEN_RE = re.compile(r"^SECTION_\d+$")
_BILL_HEADER_RE = re.compile(
    r"\b(HOUSE|SENATE|ASSEMBLY)\s+BILL\s+NO\.?\s+(\d+)\b",
    re.IGNORECASE,
)
_CODED_SECTION_LINE_RE = re.compile(r"^\s*\d+(?:\s*-\s*\d+){1,5}(?:\.\s*|\s+)")
_PROTECTED_SECTION_LINE_RE = re.compile(r"^\s*__SECTION_\d+__")
_LINE_ENUMERATOR_RE = re.compile(r"^\s*\(\s*(?:\d+|[A-Za-z]|[ivxlcdmIVXLCDM]{1,6})\s*\)")
_INLINE_SECTION_TOKEN_RE = re.compile(r"(?<!\n)\s+(\[SECTION_\d+\])")
_INLINE_CODED_HEADING_RE = re.compile(
    r"(?<!\n)(?<!§)\s+((?:\d+\s*-\s*)+\d+(?:\.\s*|\s+)(?=[A-Z]))"
)
_INLINE_BLOCK_MARKER_RE = re.compile(
    r"(?<!\n)(?<=\S)\s+(\(\s*(?:\d+|[A-Za-z]|[ivxlcdmIVXLCDM]{1,8})\s*\))(?=\s+[A-Z])"
)
_SIGNATURE_BLOCK_RE = re.compile(
    r"(?:\bHOUSE\s+BILL\s+NO\.?\s+\d+\b.*$|\bSENATE\s+BILL\s+NO\.?\s+\d+\b.*$|\bPASSED:\b.*$|\bAPPROVED\s+this\b.*$)",
    re.IGNORECASE | re.DOTALL,
)


def _previous_non_space_index(doc: Any, start_idx: int) -> int | None:
    for idx in range(start_idx, -1, -1):
        if not doc[idx].is_space:
            return idx
    return None


def _looks_like_section_marker(doc: Any, idx: int) -> bool:
    return (
        doc[idx].text == "["
        and idx + 2 < len(doc)
        and bool(_SECTION_TOKEN_RE.match(doc[idx + 1].text))
        and doc[idx + 2].text == "]"
    )


def _looks_like_enumerator(doc: Any, idx: int) -> bool:
    return (
        doc[idx].text == "("
        and idx + 2 < len(doc)
        and doc[idx + 2].text == ")"
        and bool(_ENUMERATOR_VALUE_RE.match(doc[idx + 1].text.strip()))
    )


def _has_boundary_signal_before(doc: Any, idx: int) -> bool:
    seen = 0
    cursor = idx - 1
    while cursor >= 0 and seen < 5:
        if doc[cursor].is_space:
            cursor -= 1
            continue
        token_text = doc[cursor].text
        if token_text in {";", ":", ".", "?", "!", "]"}:
            return True
        if token_text.lower() in {"and", "or"}:
            seen += 1
            cursor -= 1
            continue
        return False
    return False


def _looks_like_coded_section_heading(doc: Any, idx: int) -> bool:
    if idx >= len(doc) or not doc[idx].text.isdigit():
        return False

    prev_idx = _previous_non_space_index(doc, idx - 1)
    if prev_idx is not None and doc[prev_idx].text == "§":
        return False

    cursor = idx
    groups = 1
    while cursor + 2 < len(doc) and doc[cursor + 1].text == "-" and doc[cursor + 2].text.isdigit():
        groups += 1
        cursor += 2

    if groups < 2 or cursor + 1 >= len(doc):
        return False

    next_text = doc[cursor + 1].text
    if next_text == "." and cursor + 2 < len(doc):
        next_text = doc[cursor + 2].text

    return bool(next_text[:1].isalpha() and next_text[:1].isupper())


@Language.component("legislative_sentence_boundary_hints")
def legislative_sentence_boundary_hints(doc: Any) -> Any:
    if not doc:
        return doc

    doc[0].is_sent_start = True
    for idx in range(1, len(doc)):
        if _looks_like_section_marker(doc, idx):
            doc[idx].is_sent_start = True
            continue
        if _looks_like_enumerator(doc, idx) and _has_boundary_signal_before(doc, idx):
            doc[idx].is_sent_start = True
            continue
        if _looks_like_coded_section_heading(doc, idx):
            doc[idx].is_sent_start = True

    return doc


def _get_nlp() -> Any:
    global _nlp
    if _nlp is None:
        nlp = English()
        nlp.add_pipe("legislative_sentence_boundary_hints")
        nlp.add_pipe("sentencizer", config={"overwrite": False})
        nlp.max_length = 3_000_000
        _nlp = nlp
    return _nlp


def _extract_compact_bill_markers(raw: str) -> set[str]:
    markers: set[str] = set()
    prefixes = {
        "HOUSE": "HB",
        "SENATE": "SB",
        "ASSEMBLY": "AB",
    }
    for chamber, number in _BILL_HEADER_RE.findall(raw):
        prefix = prefixes.get(chamber.upper())
        if prefix:
            markers.add(f"{prefix}{number}")
    return markers


def _remove_isolated_bill_markers(text: str, markers: set[str]) -> str:
    for marker in markers:
        text = re.sub(
            rf"^\s*{re.escape(marker)}\s*$",
            "",
            text,
            flags=re.MULTILINE,
        )
    return text


def _looks_like_structural_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return bool(
        _PROTECTED_SECTION_LINE_RE.match(stripped)
        or _CODED_SECTION_LINE_RE.match(stripped)
        or _LINE_ENUMERATOR_RE.match(stripped)
    )


def _merge_wrapped_lines(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    merged: list[str] = []
    current = lines[0].strip()

    for raw_next in lines[1:]:
        next_line = raw_next.strip()
        if not next_line:
            if current:
                merged.append(current)
                current = ""
            merged.append("")
            continue

        if not current:
            current = next_line
            continue

        if (
            current.endswith((".", "?", "!", ";", ":"))
            or _looks_like_structural_line(next_line)
            or _looks_like_structural_line(current)
        ):
            merged.append(current)
            current = next_line
            continue

        current = f"{current} {next_line}"

    if current:
        merged.append(current)

    return "\n".join(merged)


def _force_inline_structural_breaks(text: str) -> str:
    text = _INLINE_SECTION_TOKEN_RE.sub(r"\n\1", text)
    text = _INLINE_CODED_HEADING_RE.sub(r"\n\1", text)
    text = _INLINE_BLOCK_MARKER_RE.sub(r"\n\1", text)
    return text


def _restore_structural_breaks(text: str) -> str:
    text = _force_inline_structural_breaks(text)
    text = _INLINE_CODED_HEADING_RE.sub(r"\n\1", text)
    text = _INLINE_BLOCK_MARKER_RE.sub(r"\n\1", text)
    return text


# ---------------------------------------------------------------------------
# Web / legislative noise patterns
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# IMPROVED NOISE PATTERNS (SAFE VERSION)
# ---------------------------------------------------------------------------

_WEB_NOISE_PATTERNS = [
    # Web / navigation
    r"skip to main content.*?(?=\bBe it enacted\b|\bAN ACT\b)",
    r"Congress\.gov\s+Site Content.*$",
    r"(?:Sponsor|Committees?|Latest Action|Tracker|Subject|Policy Area)\s*:.*?\n",
    r"\b(?:XML/HTML|PDF|TXT)\b.*",
    r"(?:Listen|Share/Save|Subscribe|Sign In|Site Feedback|Contact Your Member)\b",
    r"(?:Summary|Actions?|Titles?|Amendments?|Cosponsors?|Related Bills?)\s*\(\d+\)",

    # Explanatory / NON-LEGISLATIVE CONTENT (CRITICAL)
    r"Analysis by the Legislative Reference Bureau.*?do enact as follows:",
    r"For further information.*?(?=\n|$)",

    # Wisconsin / PDF headers
    r"\d{4}\s*-\s*\d{4}\s+Legislature\s*-\s*\d+\s*-\s*LRB-[\w/]+",
    r"\b[A-Z]{2,4}:\w+\b",  # MED:wlj
    r"\bSENATE BILL \d+\b",
    r"\bASSEMBLY BILL \d+\b",

    # Page separators
    r"^\s*-\s*\d+\s*-\s*$",
    r"Enrolled\s+Copy[^\n]*",

    # Sources
    r"\(Source:[^)]+\)",
]

_WEB_NOISE_RE = [
    (re.compile(p, re.IGNORECASE | re.MULTILINE | re.DOTALL), "")
    for p in _WEB_NOISE_PATTERNS
]


# ---------------------------------------------------------------------------
# SAFE LINE-LEVEL CLEANING (NO DATA LOSS)
# ---------------------------------------------------------------------------

_WEB_NOISE_RE_NODOT = [
    # REMOVE PURE PAGINATION NUMBERS ONLY (SAFE)
    (re.compile(r"^\s*\d+\s*$", re.MULTILINE), ""),

    # Remove bill headers mid text (H.B. 123 etc.)
    (re.compile(r"\n[A-Z]\.[A-Z]\.\s+\d+\s*\n?", re.MULTILINE), "\n"),

    # Remove isolated legal citation lines (rare but safe)
    (re.compile(r"^\s*\(\d{2}\s+ILCS\s+[\d/]+\)\s*$", re.MULTILINE), ""),
    (re.compile(r"^\s*\(\d+\s+U\.S\.C\.\s+[\d()a-z]+\)\s*$", re.MULTILINE), ""),
]


# ---------------------------------------------------------------------------
# Token counting (cl100k_base — consistent with OpenAI chat tokenization)
# ---------------------------------------------------------------------------

_CL100K_ENCODING: Any = None


def _get_cl100k_encoding() -> Any:
    global _CL100K_ENCODING
    if _CL100K_ENCODING is None:
        _CL100K_ENCODING = tiktoken.get_encoding("cl100k_base")
    return _CL100K_ENCODING


def count_cl100k_tokens(text: str) -> int:
    """
    Count tokens using tiktoken encoding ``cl100k_base`` (GPT-4 class models).

    Used for sentence lengths, chunk budget aggregation, and alignment with
    ``chunk_text_by_tokens``. Falls back to whitespace word count if tiktoken
    cannot load the encoding (offline edge case).
    """
    if not text:
        return 0
    try:
        enc = _get_cl100k_encoding()
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def clean_legislative_text(raw: str) -> str:
    """
    Clean legislative bill text while preserving legal content integrity.

    Improvements over v1:
    - Preserve section structure via [SECTION_X] tokens (FIXED)
    - Fix broken sentences caused by PDF pagination
    - Remove inline artifacts (".", broken fragments)
    - Improve OCR corrections
    - Ensure non-destructive transformations
    """

    text = raw
    compact_bill_markers = _extract_compact_bill_markers(raw)

    # ------------------------------------------------------------------
    # 1. HARD START CUT — keep only legislative core
    # ------------------------------------------------------------------
    match = re.search(
        r"(AN ACT|Be it enacted|SECTION\s+1\b)",
        text,
        re.IGNORECASE,
    )
    if match:
        text = text[match.start():]

    # ------------------------------------------------------------------
    # 2. PROTECT SECTION STRUCTURE (CRITICAL FIX)
    # ------------------------------------------------------------------
    # Capture SECTION X and SECTION X. BEFORE any destructive cleaning
    text = re.sub(
        r"\bSECTION\s+(\d+)\b\.?",
        r"__SECTION_\1__",
        text,
        flags=re.IGNORECASE
    )

    # ------------------------------------------------------------------
    # 3. REMOVE NON-LEGISLATIVE ANALYSIS
    # ------------------------------------------------------------------
    text = re.sub(
        r"For further information.*?(?=\n|$)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # ------------------------------------------------------------------
    # 4. REMOVE GLOBAL NOISE
    # ------------------------------------------------------------------
    for pattern, repl in _WEB_NOISE_RE:
        text = pattern.sub(repl, text)

    # ------------------------------------------------------------------
    # 5. REMOVE LINE-LEVEL NOISE
    # ------------------------------------------------------------------
    for pattern, repl in _WEB_NOISE_RE_NODOT:
        text = pattern.sub(repl, text)

    # ------------------------------------------------------------------
    # 6. REMOVE PURE PAGINATION
    # ------------------------------------------------------------------
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # ------------------------------------------------------------------
    # 7. NORMALIZE WHITESPACE (PRE)
    # ------------------------------------------------------------------
    text = text.replace("\t", " ")
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]+", "", text, flags=re.MULTILINE)
    text = _remove_isolated_bill_markers(text, compact_bill_markers)
    text = _force_inline_structural_breaks(text)

    # ------------------------------------------------------------------
    # 8. RECONSTRUCT TEXT (CONTROLLED MERGE)
    # ------------------------------------------------------------------
    text = _merge_wrapped_lines(text)

    # ------------------------------------------------------------------
    # 8b. RESTORE STRUCTURAL BREAKS LOST TO OCR / WRAP MERGING
    # ------------------------------------------------------------------
    text = _restore_structural_breaks(text)

    # ------------------------------------------------------------------
    # 9. FIX BROKEN PDF ARTIFACTS
    # ------------------------------------------------------------------

    # Remove ". " artifacts
    text = re.sub(r"\n?\s*\.\s+", " ", text)

    # Fix broken fragments like "caddy\n. services"
    text = re.sub(r"\n\s*\.\s*", " ", text)

    # Deduplicate section tokens
    text = re.sub(
        r"(__SECTION_\d+__)\s+\1",
        r"\1",
        text
    )

    # ------------------------------------------------------------------
    # 10. OCR / TEXT CORRECTIONS
    # ------------------------------------------------------------------

    text = re.sub(r"\b(\d)\s+(\d-\d{2}-\d{4})\b", r"\1\2", text)

    # ------------------------------------------------------------------
    # 11. RESTORE SECTION TOKENS (FINAL)
    # ------------------------------------------------------------------
    text = re.sub(
        r"__SECTION_(\d+)__",
        r"[SECTION_\1]",
        text
    )

    # ------------------------------------------------------------------
    # 12. DROP TRAILING SIGNATURE / APPROVAL BLOCKS
    # ------------------------------------------------------------------
    text = _SIGNATURE_BLOCK_RE.sub("", text)

    # ------------------------------------------------------------------
    # 13. FINAL NORMALIZATION
    # ------------------------------------------------------------------
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(r"\(END\)", "", text, flags=re.IGNORECASE)

    return text.strip()

def segment_into_sentences(cleaned_text: str) -> list[dict]:
    """
    Split a cleaned legislative text into sentence records.

    Returns a list of dicts with keys:
        sentence_id, chunk_id, text, n_tokens (cl100k_base), char_start, char_end
    """
    nlp = _get_nlp()
    records: list[dict] = []
    sent_idx = 0
    chunk_idx = 0
    cursor = 0

    for separator in re.finditer(r"\n\n+", cleaned_text):
        chunk = cleaned_text[cursor:separator.start()]
        chunk_start = cursor
        cursor = separator.end()
        if not chunk.strip():
            continue
        chunk_records = _segment_chunk(nlp, chunk, chunk_idx, chunk_start, sent_idx)
        records.extend(chunk_records)
        sent_idx += len(chunk_records)
        chunk_idx += 1

    tail = cleaned_text[cursor:]
    if tail.strip():
        records.extend(_segment_chunk(nlp, tail, chunk_idx, cursor, sent_idx))

    return records


def _segment_chunk(
    nlp: Any,
    chunk: str,
    chunk_idx: int,
    chunk_start: int,
    sent_idx_start: int,
) -> list[dict]:
    records: list[dict] = []
    sent_idx = sent_idx_start

    subchunks = re.split(r"\n", chunk) if len(chunk) > nlp.max_length else [chunk]
    subchunk_cursor = 0

    for subchunk in subchunks:
        relative_start = chunk.find(subchunk, subchunk_cursor)
        if relative_start < 0:
            relative_start = subchunk_cursor
        subchunk_cursor = relative_start + len(subchunk)
        if not subchunk.strip():
            continue

        doc = nlp(subchunk)

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if len(sent_text) < 15:
                continue
            if re.match(r"^[\d\s\W]+$", sent_text):
                continue
            if re.search(r"[A-Z]{2,4}\d{3,5}\s*[-–]\s*\d+\s*[-–]", sent_text):
                continue
            if re.match(r"^(?:[A-Z][a-z]+\n){3,}", sent_text):
                continue

            char_start = chunk_start + relative_start + sent.start_char
            char_end = chunk_start + relative_start + sent.end_char
            records.append(
                {
                    "sentence_id": sent_idx,
                    "chunk_id": chunk_idx,
                    "text": sent_text,
                    "n_tokens": count_cl100k_tokens(sent_text),
                    "char_start": char_start,
                    "char_end": char_end,
                }
            )
            sent_idx += 1

    return records


def chunk_text_by_tokens(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    """
    Split *text* into overlapping token-based chunks.

    Parameters
    ----------
    text:          Raw or cleaned text to chunk.
    chunk_size:    Maximum tokens per chunk (default 400).
    overlap:       Token overlap between consecutive chunks (default 50).
    encoding_name: tiktoken encoding to use.

    Returns
    -------
    List of text chunk strings.
    """
    try:
        enc = tiktoken.get_encoding(encoding_name)
        tokens = enc.encode(text)
    except Exception:
        # Fallback for offline environments where tiktoken encoding files
        # cannot be downloaded on first run.
        words = text.split()
        if not words:
            return []

        word_step = max(1, chunk_size - overlap)
        rough_chunks: list[str] = []
        start_word = 0
        while start_word < len(words):
            end_word = min(start_word + chunk_size, len(words))
            rough_chunks.append(" ".join(words[start_word:end_word]))
            if end_word == len(words):
                break
            start_word += word_step
        return rough_chunks

    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += chunk_size - overlap

    return chunks
