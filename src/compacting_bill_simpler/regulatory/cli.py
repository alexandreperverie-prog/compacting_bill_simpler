from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .llm_profiles import LEGACY_MODEL_PRESET, MODEL_PRESET_DEFAULTS, preset_model_defaults
from .orchestrator import (
    allocate_trace_dir,
    build_preview_payload,
    create_openai_client,
    run_preview_pipeline,
    trace_bill,
    write_preview_payload,
)


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Short preprocessing pipeline for legislative bills: ingest -> clean -> segment -> chunk."
    )
    parser.add_argument("--input-csv", default="dataset/df_legiscan_title_text_id.csv.gz")
    parser.add_argument("--output-dir", default="dataset/processed/preview_v1")
    parser.add_argument("--text-column", default="full_text")
    parser.add_argument("--id-column", default="code")
    parser.add_argument("--title-column", default="title")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bill-id", default=None, help="Process only this bill ID.")
    parser.add_argument("--mode", choices=["dry-run", "live"], default="dry-run")
    parser.add_argument(
        "--model-preset",
        choices=sorted(MODEL_PRESET_DEFAULTS),
        default=LEGACY_MODEL_PRESET,
    )
    parser.add_argument("--chunk-min-tokens", type=int, default=140)
    parser.add_argument("--chunk-max-tokens", type=int, default=220)
    parser.add_argument("--chunk-overlap", type=int, default=40)
    parser.add_argument("--long-sentence-threshold", type=int, default=140)
    parser.add_argument("--write-json", action="store_true", help="Persist the preview payload(s) to --output-dir.")
    parser.add_argument("--show-cleaned-text", action="store_true")
    parser.add_argument("--max-preview-sentences", type=int, default=5)
    parser.add_argument("--max-preview-chunks", type=int, default=3)
    parser.add_argument("--trace-bill-id", default=None, help="Write stage-by-stage trace artifacts for this bill ID.")
    parser.add_argument("--trace-dir", default=None, help="Optional explicit trace directory. Defaults to dataset/processed/trace_vN.")
    return parser


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    preset_defaults = preset_model_defaults(args.model_preset)
    limit = args.limit
    if limit is None and args.bill_id is None and args.trace_bill_id is None:
        limit = 1

    return PipelineConfig(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        text_column=args.text_column,
        id_column=args.id_column,
        title_column=args.title_column,
        chunk_overlap=args.chunk_overlap,
        chunk_min_tokens=args.chunk_min_tokens,
        chunk_max_tokens=args.chunk_max_tokens,
        long_sentence_threshold=args.long_sentence_threshold,
        long_sentence_model=preset_defaults["long_sentence_model"],
        model_preset=args.model_preset,
        mode=args.mode,
        dry_run=args.mode != "live",
        limit=limit,
        llm_extraction_model=preset_defaults["llm_extraction_model"],
        llm_summary_model=preset_defaults["llm_summary_model"],
        summary_verify_model=preset_defaults["summary_verify_model"],
        summary_postprocess_model=preset_defaults["summary_postprocess_model"],
    )


def _print_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    _load_dotenv_if_available()
    args = build_parser().parse_args()
    config = _build_config(args)
    openai_client = create_openai_client(config)

    if args.trace_bill_id:
        trace_dir = Path(args.trace_dir) if args.trace_dir else allocate_trace_dir(config)
        written_dir = trace_bill(
            config,
            args.trace_bill_id,
            trace_dir,
            openai_client=openai_client,
        )
        print(f"Trace written to: {written_dir}")
        return

    artifacts_list = run_preview_pipeline(
        config,
        bill_id=args.bill_id,
        openai_client=openai_client,
    )

    for artifacts in artifacts_list:
        payload = build_preview_payload(
            artifacts,
            config=config,
            max_sentences=args.max_preview_sentences,
            max_chunks=args.max_preview_chunks,
            include_cleaned_text=args.show_cleaned_text,
        )
        _print_payload(payload)

        if args.write_json:
            output_path = config.output_dir / f"{artifacts.bill.bill_id}.preview.json"
            write_preview_payload(output_path, payload)
            print(f"Wrote preview to {output_path}")
