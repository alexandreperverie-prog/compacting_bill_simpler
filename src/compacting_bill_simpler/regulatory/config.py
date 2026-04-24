from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .llm_profiles import LEGACY_MODEL_PRESET, normalize_model_preset, preset_model_defaults


@dataclass(frozen=True)
class PipelineConfig:
    input_csv: Path = Path("dataset/df_legiscan_title_text_id.csv.gz")
    output_dir: Path = Path("dataset/processed/regulatory_v1")
    text_column: str = "full_text"
    id_column: str = "code"
    title_column: str = "title"
    chunk_size: int = 900           # legacy — kept for backward compat
    chunk_overlap: int = 40         # token budget used to select whole overlap sentences
    chunk_min_tokens: int = 140     # soft lower target for chunk size
    chunk_max_tokens: int = 220     # maximum cl100k_base tokens per chunk (hard stop)
    long_sentence_threshold: int = 140   # cl100k_base tokens above which a sentence is "too long"
    long_sentence_model: str = "gpt-5-nano"  # LLM used to split long sentences
    model_preset: str = LEGACY_MODEL_PRESET
    retrieval_topk: int = 8
    mode: str = "dry-run"
    dry_run: bool = True
    limit: int | None = None
    embedding_model: str = "text-embedding-3-large"
    llm_extraction_model: str = "gpt-4o-mini"
    llm_summary_model: str = "gpt-5-nano"
    summary_mode: str = "strict"
    profile_pipeline: bool = False
    summary_verification: bool = True
    summary_verify_model: str | None = None
    summary_verify_apply_reformat: bool = True
    summary_postprocessing: bool = True
    summary_postprocess_model: str | None = "gpt-4o-mini"
    stop_after: str = "postprocess"
    use_structural_rerank: bool = True
    use_background_penalty: bool = True
    use_claim_bank_rebuild: bool = True
    use_llm_extractors: bool = True
    use_embeddings: bool = True

    @staticmethod
    def from_env() -> "PipelineConfig":
        limit_raw = os.getenv("PIPELINE_LIMIT")
        mode = os.getenv("PIPELINE_MODE", "dry-run").strip().lower()
        if mode not in {"dry-run", "live"}:
            mode = "dry-run"
        summary_mode = os.getenv("SUMMARY_MODE", "strict").strip().lower()
        if summary_mode not in {"strict", "llm_assisted"}:
            summary_mode = "strict"
        stop_after = os.getenv("PIPELINE_STOP_AFTER", "postprocess").strip().lower()
        if stop_after not in {"extract", "esg", "quality", "summary", "summary_verify", "postprocess"}:
            stop_after = "postprocess"
        model_preset = normalize_model_preset(os.getenv("MODEL_PRESET", LEGACY_MODEL_PRESET))
        preset_defaults = preset_model_defaults(model_preset)
        return PipelineConfig(
            input_csv=Path(os.getenv("PIPELINE_INPUT_CSV", "dataset/df_legiscan_title_text_id.csv.gz")),
            output_dir=Path(os.getenv("PIPELINE_OUTPUT_DIR", "dataset/processed/regulatory_v1")),
            text_column=os.getenv("PIPELINE_TEXT_COLUMN", "full_text"),
            id_column=os.getenv("PIPELINE_ID_COLUMN", "code"),
            title_column=os.getenv("PIPELINE_TITLE_COLUMN", "title"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "40")),
            chunk_min_tokens=int(os.getenv("CHUNK_MIN_TOKENS", "140")),
            chunk_max_tokens=int(os.getenv("CHUNK_MAX_TOKENS", "220")),
            long_sentence_threshold=int(os.getenv("LONG_SENT_THRESHOLD", "140")),
            long_sentence_model=os.getenv("LONG_SENT_MODEL", preset_defaults["long_sentence_model"]),
            model_preset=model_preset,
            retrieval_topk=int(os.getenv("RETRIEVAL_VECTOR_TOPK", "8")),
            mode=mode,
            dry_run=mode != "live",
            limit=int(limit_raw) if limit_raw else None,
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            llm_extraction_model=os.getenv("LLM_EXTRACTION_MODEL", preset_defaults["llm_extraction_model"]),
            llm_summary_model=os.getenv("LLM_SUMMARY_MODEL", preset_defaults["llm_summary_model"]),
            summary_mode=summary_mode,
            summary_verification=os.getenv("SUMMARY_VERIFICATION", "1").strip() not in {"0", "false", "no"},
            summary_verify_model=(
                (m := (os.getenv("SUMMARY_VERIFY_MODEL") or "").strip())
                and m
                or preset_defaults["summary_verify_model"]
            ),
            summary_verify_apply_reformat=os.getenv("SUMMARY_VERIFY_APPLY_REFORMAT", "1").strip()
            not in {"0", "false", "no"},
            summary_postprocessing=os.getenv("SUMMARY_POSTPROCESSING", "1").strip()
            not in {"0", "false", "no"},
            summary_postprocess_model=(
                (m := (os.getenv("SUMMARY_POSTPROCESS_MODEL") or str(preset_defaults["summary_postprocess_model"] or "")).strip())
                and m
                or None
            ),
            stop_after=stop_after,
            use_structural_rerank=os.getenv("ABLATION_NO_STRUCTURAL_RERANK", "0").strip() in {"0", "false", "no"},
            use_background_penalty=os.getenv("ABLATION_NO_BACKGROUND_PENALTY", "0").strip() in {"0", "false", "no"},
            use_claim_bank_rebuild=os.getenv("ABLATION_NO_CLAIM_BANK_REBUILD", "0").strip() in {"0", "false", "no"},
            use_llm_extractors=os.getenv("ABLATION_NO_LLM_EXTRACTORS", "0").strip() in {"0", "false", "no"},
            use_embeddings=os.getenv("ABLATION_NO_EMBEDDINGS", "0").strip() in {"0", "false", "no"},
        )
