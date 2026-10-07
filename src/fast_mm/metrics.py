from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


class MetricsRecorder:
    def __init__(
        self,
        path: Path | None,
        enabled: bool,
        live: bool,
        interval: int,
    ) -> None:
        self.path = path
        self.enabled = enabled
        self.live = live and enabled
        self.interval = interval
        self.started = time.perf_counter()
        self._last_live_length = 0
        if self.enabled and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def emit(self, event: str, **values: Any) -> None:
        if not self.enabled:
            return
        record = {
            "schema_version": 1,
            "event": event,
            "elapsed_seconds": self.elapsed(),
            **values,
        }
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def progress(
        self,
        restart: int,
        restart_count: int,
        iteration: int,
        relative_residual: float,
        best_relative_residual: float,
    ) -> None:
        if not self.live:
            return
        message = (
            f"restart {restart + 1}/{restart_count}  "
            f"iter {iteration}  "
            f"residual {relative_residual:.3e}  "
            f"best {best_relative_residual:.3e}  "
            f"elapsed {self.elapsed():.1f}s"
        )
        padding = max(0, self._last_live_length - len(message))
        sys.stdout.write("\r" + message + (" " * padding))
        sys.stdout.flush()
        self._last_live_length = len(message)

    def finish_live_line(self) -> None:
        if self.live and self._last_live_length:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._last_live_length = 0


