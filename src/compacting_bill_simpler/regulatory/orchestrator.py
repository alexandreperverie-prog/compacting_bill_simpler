from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any

from ..text_processing import clean_legislative_text
from .config import PipelineConfig
from .llm_profiles import preset_model_defaults
from .models import BillRecord, ChunkRecord, SentenceRecord
from .stages import chunk_bill, ingest_bills, profile_document, segment_bill


@dataclass
class PipelineArtifacts:
    bill: BillRecord
    sentences: list[SentenceRecord]
    chunks: list[ChunkRecord]
    document_profile: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bill": asdict(self.bill),
            "sentences": [asdict(sentence) for sentence in self.sentences],
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "document_profile": self.document_profile,
        }


def create_openai_client(config: PipelineConfig) -> Any | None:
    if config.mode != "live":
        return None

    try:
        import openai
    except ImportError as exc:
        raise RuntimeError("Live mode requires the 'openai' package to be installed.") from exc

    return openai.OpenAI()


def select_bills(config: PipelineConfig, *, bill_id: str | None = None) -> list[BillRecord]:
    records = ingest_bills(config)
    if bill_id is None:
        return records

    filtered = [record for record in records if record.bill_id == bill_id]
    if not filtered:
        raise ValueError(f"Bill ID '{bill_id}' not found in {config.input_csv}")
    return filtered


def run_bill_pipeline(
    bill: BillRecord,
    config: PipelineConfig,
    *,
    openai_client: Any | None = None,
) -> PipelineArtifacts:
    bill.cleaned_text = clean_legislative_text(bill.raw_text)
    sentences = segment_bill(
        bill,
        dry_run=config.dry_run,
        long_sent_model=config.long_sentence_model,
        long_sent_threshold=config.long_sentence_threshold,
        llm_client=openai_client,
    )
    chunks = chunk_bill(
        bill,
        sentences,
        min_tokens=config.chunk_min_tokens,
        max_tokens=config.chunk_max_tokens,
        overlap=config.chunk_overlap,
    )
    document_profile = profile_document(bill.cleaned_text, chunks)
    return PipelineArtifacts(
        bill=bill,
        sentences=sentences,
        chunks=chunks,
        document_profile=document_profile,
    )


def run_preview_pipeline(
    config: PipelineConfig,
    *,
    bill_id: str | None = None,
    openai_client: Any | None = None,
) -> list[PipelineArtifacts]:
    bills = select_bills(config, bill_id=bill_id)
    return [run_bill_pipeline(bill, config, openai_client=openai_client) for bill in bills]


def build_preview_payload(
    artifacts: PipelineArtifacts,
    *,
    config: PipelineConfig,
    max_sentences: int = 5,
    max_chunks: int = 3,
    include_cleaned_text: bool = False,
) -> dict[str, Any]:
    preset_defaults = preset_model_defaults(config.model_preset)
    payload = {
        "bill_id": artifacts.bill.bill_id,
        "title": artifacts.bill.title,
        "jurisdiction": artifacts.bill.jurisdiction,
        "text_hash": artifacts.bill.text_hash,
        "pipeline": {
            "mode": config.mode,
            "input_csv": str(config.input_csv),
            "text_column": config.text_column,
            "id_column": config.id_column,
            "title_column": config.title_column,
            "chunk_min_tokens": config.chunk_min_tokens,
            "chunk_max_tokens": config.chunk_max_tokens,
            "chunk_overlap": config.chunk_overlap,
            "long_sentence_threshold": config.long_sentence_threshold,
        },
        "llm": {
            "model_preset": config.model_preset,
            "preset_defaults": preset_defaults,
            "long_sentence_model": config.long_sentence_model,
            "llm_extraction_model": config.llm_extraction_model,
            "llm_summary_model": config.llm_summary_model,
            "summary_verify_model": config.summary_verify_model,
            "summary_postprocess_model": config.summary_postprocess_model,
        },
        "counts": {
            "sentences": len(artifacts.sentences),
            "chunks": len(artifacts.chunks),
            "cleaned_text_chars": len(artifacts.bill.cleaned_text or ""),
        },
        "document_profile": artifacts.document_profile,
        "sentences_preview": [asdict(sentence) for sentence in artifacts.sentences[:max_sentences]],
        "chunks_preview": [asdict(chunk) for chunk in artifacts.chunks[:max_chunks]],
    }
    if include_cleaned_text:
        payload["cleaned_text"] = artifacts.bill.cleaned_text or ""
    return payload


def write_preview_payload(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _json_default(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, rows: list[Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    return path


def default_trace_root(config: PipelineConfig) -> Path:
    if config.output_dir.name.startswith("preview_"):
        return config.output_dir.parent
    return config.output_dir


def allocate_trace_dir(config: PipelineConfig, trace_root: Path | None = None) -> Path:
    root = trace_root or default_trace_root(config)
    root.mkdir(parents=True, exist_ok=True)

    version_numbers: list[int] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = re.fullmatch(r"trace_v(\d+)", path.name)
        if match:
            version_numbers.append(int(match.group(1)))

    next_version = max(version_numbers, default=0) + 1
    return root / f"trace_v{next_version}"


def trace_bill(
    config: PipelineConfig,
    bill_id: str,
    trace_dir: Path,
    *,
    openai_client: Any | None = None,
) -> Path:
    selected = select_bills(config, bill_id=bill_id)
    bill = selected[0]
    artifacts = run_bill_pipeline(bill, config, openai_client=openai_client)
    payload = build_preview_payload(
        artifacts,
        config=config,
        max_sentences=len(artifacts.sentences),
        max_chunks=len(artifacts.chunks),
        include_cleaned_text=False,
    )

    trace_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "bill_id": artifacts.bill.bill_id,
        "title": artifacts.bill.title,
        "mode": config.mode,
        "trace_version": "v1",
        "pipeline_steps": ["ingest", "clean", "segment", "chunk", "document_profile"],
        "raw_text_artifact": "01_raw_text.txt",
        "bill_artifact": "01_bill.json",
        "cleaned_text_artifact": "02_cleaned_text.txt",
        "sentences_artifact": "03_sentences.jsonl",
        "chunks_artifact": "04_chunks.jsonl",
        "document_profile_artifact": "05_document_profile.json",
        "preview_artifact": "06_preview.json",
        "input_csv": str(config.input_csv),
        "text_column": config.text_column,
        "id_column": config.id_column,
        "title_column": config.title_column,
        "model_preset": config.model_preset,
    }

    (trace_dir / "01_raw_text.txt").write_text(artifacts.bill.raw_text or "", encoding="utf-8")
    _write_json(trace_dir / "01_bill.json", artifacts.bill)
    (trace_dir / "02_cleaned_text.txt").write_text(artifacts.bill.cleaned_text or "", encoding="utf-8")
    _write_jsonl(trace_dir / "03_sentences.jsonl", artifacts.sentences)
    _write_jsonl(trace_dir / "04_chunks.jsonl", artifacts.chunks)
    _write_json(trace_dir / "05_document_profile.json", artifacts.document_profile)
    _write_json(trace_dir / "06_preview.json", payload)
    _write_json(trace_dir / "00_manifest.json", manifest)
    return trace_dir
