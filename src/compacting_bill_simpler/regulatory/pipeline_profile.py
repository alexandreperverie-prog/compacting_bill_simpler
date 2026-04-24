from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PipelineProfile:
    def __init__(self, tqdm_bar: Any | None = None) -> None:
        self._bar = tqdm_bar
        self.nodes: list[dict[str, Any]] = []
        self.run_started_at = _utc_now()
        self._perf_start = time.perf_counter()

    @contextmanager
    def step(self, node_id: str, *, label: str | None = None) -> Iterator[None]:
        started_at = _utc_now()
        perf_start = time.perf_counter()
        if self._bar is not None:
            self._bar.set_postfix_str(node_id, refresh=False)
        try:
            yield
        finally:
            ended_at = _utc_now()
            seconds = round(time.perf_counter() - perf_start, 6)
            self.nodes.append(
                {
                    "id": node_id,
                    "label": label or node_id,
                    "seconds": seconds,
                    "started_at": started_at,
                    "ended_at": ended_at,
                }
            )
            if self._bar is not None:
                self._bar.update(1)

    def to_report(self) -> dict[str, Any]:
        return {
            "run_started_at": self.run_started_at,
            "run_finished_at": _utc_now(),
            "total_seconds": round(time.perf_counter() - self._perf_start, 6),
            "nodes": list(self.nodes),
        }
