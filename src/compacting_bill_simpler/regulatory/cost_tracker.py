from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PRICE_PER_MILLION: dict[str, dict[str, float]] = {
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.0, "reasoning": 10.0},
    "gpt-5.1-chat-latest": {"input": 1.25, "cached_input": 0.125, "output": 10.0, "reasoning": 10.0},
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.0, "reasoning": 10.0},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40, "reasoning": 0.40},
    "gpt-5.4": {"input": 2.50, "cached_input": 0.25, "output": 15.0, "reasoning": 15.0},
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50, "reasoning": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02, "output": 1.25, "reasoning": 1.25},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60, "reasoning": 0.60},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40, "reasoning": 0.40},
    "text-embedding-3-large": {"input": 0.13, "cached_input": 0.0, "output": 0.0, "reasoning": 0.0},
    "text-embedding-3-small": {"input": 0.02, "cached_input": 0.0, "output": 0.0, "reasoning": 0.0},
}

_UNKNOWN_PRICE: dict[str, float] = {"input": 0.0, "cached_input": 0.0, "output": 0.0, "reasoning": 0.0}


def _resolve_price(model: str) -> dict[str, float]:
    normalized = str(model or "").strip()
    if normalized in _PRICE_PER_MILLION:
        return _PRICE_PER_MILLION[normalized]
    for prefix, prices in _PRICE_PER_MILLION.items():
        if normalized.startswith(prefix):
            return prices
    return _UNKNOWN_PRICE


class UsageTracker:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, int]] = {}

    def _ensure(self, model: str) -> None:
        if model not in self._data:
            self._data[model] = {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "embedding_tokens": 0,
                "calls": 0,
            }

    def record_chat(
        self,
        model: str,
        *,
        input_tokens: int,
        cached_input_tokens: int = 0,
        output_tokens: int,
        reasoning_tokens: int = 0,
    ) -> None:
        self._ensure(model)
        self._data[model]["input_tokens"] += max(0, int(input_tokens))
        self._data[model]["cached_input_tokens"] += max(0, int(cached_input_tokens))
        self._data[model]["output_tokens"] += max(0, int(output_tokens))
        self._data[model]["reasoning_tokens"] += max(0, int(reasoning_tokens))
        self._data[model]["calls"] += 1

    def record_embedding(self, model: str, *, total_tokens: int) -> None:
        self._ensure(model)
        self._data[model]["embedding_tokens"] += max(0, int(total_tokens))
        self._data[model]["calls"] += 1

    def to_report(self) -> dict[str, Any]:
        models: dict[str, Any] = {}
        total_cost = 0.0
        total_tokens = 0

        for model, counts in self._data.items():
            prices = _resolve_price(model)
            cached_input = counts["cached_input_tokens"]
            total_input = counts["input_tokens"]
            billable_input = max(0, total_input - cached_input)
            reasoning = counts["reasoning_tokens"]
            pure_output = max(0, counts["output_tokens"] - reasoning)

            input_cost = billable_input / 1_000_000 * prices["input"]
            cached_input_cost = cached_input / 1_000_000 * prices["cached_input"]
            output_cost = pure_output / 1_000_000 * prices["output"]
            reasoning_cost = reasoning / 1_000_000 * prices["reasoning"]
            embedding_cost = counts["embedding_tokens"] / 1_000_000 * prices["input"]

            model_cost = input_cost + cached_input_cost + output_cost + reasoning_cost + embedding_cost
            total_cost += model_cost
            total_tokens += total_input + counts["output_tokens"] + counts["embedding_tokens"]

            models[model] = {
                "calls": counts["calls"],
                "tokens": {
                    "input": total_input,
                    "cached_input": cached_input,
                    "billable_input": billable_input,
                    "output": pure_output,
                    "reasoning": reasoning,
                    "embedding": counts["embedding_tokens"],
                    "total": total_input + counts["output_tokens"] + counts["embedding_tokens"],
                },
                "cost_usd": round(model_cost, 6),
                "cost_breakdown_usd": {
                    "input": round(input_cost, 6),
                    "cached_input": round(cached_input_cost, 6),
                    "output": round(output_cost, 6),
                    "reasoning": round(reasoning_cost, 6),
                    "embedding": round(embedding_cost, 6),
                },
            }

        return {
            "total_cost_usd": round(total_cost, 6),
            "total_cost_eur": round(total_cost * 0.92, 6),
            "total_tokens": total_tokens,
            "models": models,
            "pricing_note": "Uses OpenAI per-model token pricing, including cached input tokens when the API reports them.",
        }

    def write_report(self, path: Path) -> None:
        report = self.to_report()
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


class _TrackedCompletions:
    def __init__(self, completions: Any, tracker: UsageTracker) -> None:
        self._completions = completions
        self._tracker = tracker

    def create(self, model: str, messages: Any, **kwargs: Any) -> Any:
        response = self._completions.create(model=model, messages=messages, **kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            completion_details = getattr(usage, "completion_tokens_details", None)
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) if completion_details else 0
            cached_tokens = getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0
            self._tracker.record_chat(
                model=model,
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                cached_input_tokens=cached_tokens or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                reasoning_tokens=reasoning_tokens or 0,
            )
        return response


class _TrackedChat:
    def __init__(self, chat: Any, tracker: UsageTracker) -> None:
        self.completions = _TrackedCompletions(chat.completions, tracker)


class _TrackedEmbeddings:
    def __init__(self, embeddings: Any, tracker: UsageTracker) -> None:
        self._embeddings = embeddings
        self._tracker = tracker

    def create(self, input: Any, model: str, **kwargs: Any) -> Any:
        response = self._embeddings.create(input=input, model=model, **kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._tracker.record_embedding(model=model, total_tokens=getattr(usage, "total_tokens", 0) or 0)
        return response


class TrackedOpenAI:
    def __init__(self, client: Any, tracker: UsageTracker) -> None:
        self._client = client
        self.tracker = tracker
        self.chat = _TrackedChat(client.chat, tracker)
        self.embeddings = _TrackedEmbeddings(client.embeddings, tracker)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def get_usage_tracker(client: Any | None) -> UsageTracker | None:
    tracker = getattr(client, "tracker", None)
    return tracker if isinstance(tracker, UsageTracker) else None
