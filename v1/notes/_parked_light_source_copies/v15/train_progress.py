"""Sticky training progress with heartbeat spinner (V15)."""
from __future__ import annotations

import sys
import threading
import time
from typing import Any

_SPINNERS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _fmt_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN
        return "hesaplanıyor"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}sa {m:02d}dk {s:02d}sn"
    if m > 0:
        return f"{m}dk {s:02d}sn"
    return f"{s}sn"


class TrainProgress:
    """Tracks training units; heartbeat thread keeps spinner/elapsed alive."""

    def __init__(self, total_units: int, enabled: bool = True, heartbeat_s: float = 0.35):
        self.total = max(1, int(total_units))
        self.done = 0
        self.t0 = time.time()
        self.enabled = bool(enabled)
        self._sticky = sys.stdout.isatty()
        self._last_len = 0
        self._label = "başlıyor"
        self._stage = "init"
        self._closed = False
        self._last_print_t = 0.0
        self._last_print_pct = -1.0
        self._lock = threading.Lock()
        self._spin_i = 0
        self._heartbeat_s = float(heartbeat_s)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "TrainProgress":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is KeyboardInterrupt:
            self.finish("iptal edildi")
        elif exc_type is not None:
            self.finish("hata")
        else:
            self.finish("tamamlandı")
        return False

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._heartbeat_loop, name="v15-progress", daemon=True)
        self._thread.start()
        self.render()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self._heartbeat_s):
            with self._lock:
                if self._closed:
                    break
                self._spin_i = (self._spin_i + 1) % len(_SPINNERS)
                self._render_unlocked()

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
            eta = (elapsed / self.done) * remaining_units
            eta_s = _fmt_duration(eta)
        elif remaining_units == 0:
            eta_s = _fmt_duration(0.0)
        else:
            eta_s = "hesaplanıyor"
        shown_done = min(self.done, self.total)
        spin = _SPINNERS[self._spin_i % len(_SPINNERS)]
        stage = self._stage or self._label
        return (
            f"V15 {spin} [{self._bar(pct)}] {pct:5.1f}% | "
            f"{shown_done}/{self.total} | "
            f"geçen {_fmt_duration(elapsed)} | "
            f"kalan ~{eta_s} | {stage}"
        )

    def clear_line(self) -> None:
        if not self.enabled or self._closed or not self._sticky:
            return
        sys.stdout.write("\r" + (" " * max(self._last_len, 1)) + "\r")
        sys.stdout.flush()

    def _render_unlocked(self, label: str | None = None) -> None:
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
        now = time.time()
        pct = 100.0 * self.done / self.total
        if (now - self._last_print_t) < 3.0 and (pct - self._last_print_pct) < 5.0:
            return
        print(line, flush=True)
        self._last_print_t = now
        self._last_print_pct = pct

    def render(self, label: str | None = None) -> None:
        with self._lock:
            self._render_unlocked(label)

    def log(self, msg: str = "", **kwargs: Any) -> None:
        end = kwargs.pop("end", "\n")
        with self._lock:
            if self._sticky and not self._closed:
                sys.stdout.write("\r" + (" " * max(self._last_len, 1)) + "\r")
                sys.stdout.flush()
            print(msg, end=end, **kwargs)
            if end.endswith("\n") or end == "\n":
                self._render_unlocked()

    def tick(self, label: str = "", n: int = 1) -> None:
        with self._lock:
            self.done += max(0, int(n))
            if self.done > self.total:
                self.total = self.done
            if label:
                self._label = label
                self._stage = label
            self._render_unlocked()

    def set_label(self, label: str) -> None:
        self.set_stage(label)

    def set_stage(self, stage: str) -> None:
        with self._lock:
            self._stage = stage
            self._label = stage
            self._render_unlocked()

    def finish(self, label: str = "tamamlandı") -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self._lock:
            if self._closed:
                return
            self.total = max(self.total, self.done, 1)
            self.done = self.total
            self._label = label
            self._stage = label
            if self.enabled:
                line = self._status_text()
                if self._sticky:
                    sys.stdout.write("\r" + (" " * max(self._last_len, 1)) + "\r")
                    sys.stdout.write(line + "\n")
                    sys.stdout.flush()
                else:
                    print(line, flush=True)
            self._closed = True


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
    n_detail_runs: int = 0,
    include_final: bool = True,
    n_segments: int = 4,
    n_counties_est: int = 4,
) -> int:
    per_run = 1 + (2 * n_models) + (n_models * n_segments) + (n_models * n_counties_est) + 2
    n_runs = n_attr_runs + n_demo_runs + n_detail_runs + (1 if include_final else 0)
    return max(1, n_runs * per_run)
