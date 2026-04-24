from __future__ import annotations

from typing import Any

LEGACY_MODEL_PRESET = "legacy"
GPT51_MODEL_PRESET = "gpt51_all"
GPT54_MODEL_PRESET = "gpt54_all"

MODEL_PRESET_DEFAULTS: dict[str, dict[str, Any]] = {
    LEGACY_MODEL_PRESET: {
        "long_sentence_model": "gpt-5-nano",
        "llm_extraction_model": "gpt-4o-mini",
        "llm_summary_model": "gpt-5-nano",
        "summary_verify_model": None,
        "summary_postprocess_model": "gpt-4o-mini",
    },
    GPT51_MODEL_PRESET: {
        "long_sentence_model": "gpt-5.1",
        "llm_extraction_model": "gpt-5.1",
        "llm_summary_model": "gpt-5.1",
        "summary_verify_model": "gpt-5.1",
        "summary_postprocess_model": "gpt-5.1",
    },
    GPT54_MODEL_PRESET: {
        "long_sentence_model": "gpt-5.4",
        "llm_extraction_model": "gpt-5.4",
        "llm_summary_model": "gpt-5.4",
        "summary_verify_model": None,
        "summary_postprocess_model": "gpt-5.4",
    },
}

GPT5_REASONING_EFFORT_DEFAULT = "low"


def normalize_model_preset(raw: str | None) -> str:
    value = str(raw or LEGACY_MODEL_PRESET).strip().lower()
    return value if value in MODEL_PRESET_DEFAULTS else LEGACY_MODEL_PRESET


def preset_model_defaults(preset: str | None) -> dict[str, Any]:
    return dict(MODEL_PRESET_DEFAULTS[normalize_model_preset(preset)])


def is_gpt5_family_model(model: str | None) -> bool:
    return str(model or "").strip().lower().startswith("gpt-5")


def build_chat_completion_kwargs(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_output_tokens: int,
    response_format: dict[str, Any] | None = None,
    temperature: float | None = 0,
    reasoning_effort: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    if is_gpt5_family_model(model):
        kwargs["max_completion_tokens"] = max_output_tokens
        kwargs["reasoning_effort"] = reasoning_effort or GPT5_REASONING_EFFORT_DEFAULT
    else:
        kwargs["max_tokens"] = max_output_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature

    kwargs.update(extra)
    return kwargs
