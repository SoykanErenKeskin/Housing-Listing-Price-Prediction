"""Sticky training progress line: percent + ETA at bottom of terminal output."""
from __future__ import annotations

import sys
import time
from typing import Any


def _fmt_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN
        return "--:--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}sa {m:02d}dk {s:02d}sn"
    if m > 0:
        return f"{m}dk {s:02d}sn"
    return f"{s}sn"


class TrainProgress:
    """Tracks coarse training units and redraws a status line under logs."""

    def __init__(self, total_units: int, enabled: bool = True):
        self.total = max(1, int(total_units))
        self.done = 0
        self.t0 = time.time()
        self.enabled = bool(enabled)
        self._sticky = sys.stdout.isatty()
        self._last_len = 0
        self._label = "başlıyor"
        self._closed = False
        self._last_print_t = 0.0
        self._last_print_pct = -1.0

    def _bar(self, pct: float, width: int = 24) -> str:
        filled = int(round(width * pct / 100.0))
        filled = max(0, min(width, filled))
        return "█" * filled + "░" * (width - filled)

    def _status_text(self) -> str:
        if self._closed or self.done >= self.total:
            pct = 100.0 * min(self.done, self.total) / self.total
        else:
            pct = min(99.9, 100.0 * self.done / self.total)
        elapsed = time.time() - self.t0
        remaining_units = max(0, self.total - self.done)
        if self.done > 0 and remaining_units > 0:
            rate = elapsed / self.done
            eta = rate * remaining_units
        elif remaining_units == 0:
            eta = 0.0
        else:
            eta = float("nan")
        shown_done = min(self.done, self.total)
        return (
            f"İlerleme [{self._bar(pct)}] {pct:5.1f}%  "
            f"{shown_done}/{self.total}  "
            f"geçen {_fmt_duration(elapsed)}  "
            f"kalan ~{_fmt_duration(eta)}  "
            f"| {self._label}"
        )

    def clear_line(self) -> None:
        if not self.enabled or self._closed or not self._sticky:
            return
        sys.stdout.write("\r" + (" " * max(self._last_len, 1)) + "\r")
        sys.stdout.flush()

    def render(self, label: str | None = None) -> None:
        if not self.enabled or self._closed:
            return
        if label is not None:
            self._label = label
        line = self._status_text()
        if self._sticky:
            sys.stdout.write("\r" + (" " * max(self._last_len, 1)) + "\r")
            sys.stdout.write(line)
            sys.stdout.flush()
            self._last_len = len(line)
            return
        # Non-TTY: throttle so logs aren't flooded
        now = time.time()
        pct = 100.0 * self.done / self.total
        if (now - self._last_print_t) < 2.0 and (pct - self._last_print_pct) < 5.0 and not self._closed:
            return
        print(line, flush=True)
        self._last_print_t = now
        self._last_print_pct = pct

    def log(self, msg: str = "", **kwargs: Any) -> None:
        """Print a normal log line, then redraw the progress status underneath."""
        end = kwargs.pop("end", "\n")
        self.clear_line()
        print(msg, end=end, **kwargs)
        if end.endswith("\n") or end == "\n":
            self.render()

    def tick(self, label: str = "", n: int = 1) -> None:
        self.done += max(0, int(n))
        # Expand budget if we underestimated (keeps ETA honest instead of stuck at 100%).
        if self.done > self.total:
            self.total = self.done
        self.render(label or self._label)

    def set_label(self, label: str) -> None:
        self.render(label)

    def finish(self, label: str = "tamamlandı") -> None:
        if self._closed:
            return
        self.total = max(self.total, self.done, 1)
        self.done = self.total
        self._label = label
        if self.enabled:
            line = self._status_text()
            if self._sticky:
                sys.stdout.write("\r" + (" " * max(self._last_len, 1)) + "\r")
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
            else:
                print(line, flush=True)
        self._closed = True


# Module-level handle used by training helpers
PROGRESS: TrainProgress | None = None


def get_progress() -> TrainProgress | None:
    return PROGRESS


def set_progress(progress: TrainProgress | None) -> None:
    global PROGRESS
    PROGRESS = progress


def estimate_training_units(
    n_models: int,
    n_attr_runs: int,
    n_demo_runs: int,
    include_final: bool = True,
    n_segments: int = 4,
    n_counties_est: int = 4,
) -> int:
    """Rough unit count matching tick points in train_and_evaluate / layers."""
    # per run: anomaly(1) + base OOF+fit(2*m) + segments(m*seg) + counties(m*c) + wrap(2)
    per_run = 1 + (2 * n_models) + (n_models * n_segments) + (n_models * n_counties_est) + 2
    n_runs = n_attr_runs + n_demo_runs + (1 if include_final else 0)
    return max(1, n_runs * per_run)
