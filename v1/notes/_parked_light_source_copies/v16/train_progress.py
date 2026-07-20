"""Sticky training progress with heartbeat spinner (V16).

Progress bar stays on one terminal line (bottom), cleared/redrawn in place.
Log lines print above it; long status text is truncated to terminal width so
carriage-return updates do not wrap and stack.
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from typing import Any

_SPINNERS = "⠋⠙⠹⠼⠴⠦⠧⠇⠏"


def _enable_windows_vt() -> None:
    """Enable ANSI escape processing on Windows consoles (incl. Cursor/Windows Terminal)."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    except Exception:
        pass


def _fmt_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN
        return "?"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}sa{m:02d}dk"
    if m > 0:
        return f"{m}dk{s:02d}sn"
    return f"{s}sn"


def _term_width(fallback: int = 100) -> int:
    try:
        return max(40, int(shutil.get_terminal_size(fallback=(fallback, 24)).columns))
    except Exception:
        return fallback


def _truncate(text: str, width: int) -> str:
    if width <= 0 or len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: max(0, width - 1)] + "…"


def _want_sticky() -> bool:
    """Decide whether to overwrite one progress line in-place."""
    off = str(os.getenv("NO_STICKY_PROGRESS", "")).strip().lower() in {"1", "true", "yes"}
    if off:
        return False
    force = str(os.getenv("FORCE_STICKY_PROGRESS", "")).strip().lower() in {"1", "true", "yes"}
    if force:
        return True
    if sys.stdout.isatty() or sys.stderr.isatty():
        return True
    # Windows Terminal / VS Code / Cursor integrated terminal hints
    if os.environ.get("WT_SESSION"):
        return True
    term_program = str(os.environ.get("TERM_PROGRAM", "")).lower()
    if term_program in {"vscode", "cursor", "windows-terminal"}:
        return True
    return False


class TrainProgress:
    """Tracks training units; heartbeat thread keeps spinner/elapsed alive on one sticky line."""

    def __init__(self, total_units: int, enabled: bool = True, heartbeat_s: float = 0.35):
        self.total = max(1, int(total_units))
        self.done = 0
        self.t0 = time.time()
        self.enabled = bool(enabled)
        self._sticky = _want_sticky()
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
        # Same stream for bar + coordinated logs so clear/redraw stays on one visual line.
        self._stream = sys.stdout
        _enable_windows_vt()

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
        self._thread = threading.Thread(target=self._heartbeat_loop, name="v16-progress", daemon=True)
        self._thread.start()
        self.render()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self._heartbeat_s):
            with self._lock:
                if self._closed:
                    break
                self._spin_i = (self._spin_i + 1) % len(_SPINNERS)
                self._render_unlocked()

    def _bar(self, pct: float, width: int = 20) -> str:
        filled = int(round(width * pct / 100.0))
        filled = max(0, min(width, filled))
        return "█" * filled + "░" * (width - filled)

    def _status_text(self, max_width: int | None = None) -> str:
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
            eta_s = "?"
        shown_done = min(self.done, self.total)
        spin = _SPINNERS[self._spin_i % len(_SPINNERS)]
        stage = str(self._stage or self._label or "").replace("\n", " ").strip()
        # Compact prefix so stage gets remaining width.
        prefix = (
            f"V16 {spin} [{self._bar(pct)}] {pct:4.1f}% "
            f"{shown_done}/{self.total} "
            f"+{_fmt_duration(elapsed)} ~{eta_s} | "
        )
        width = max_width if max_width is not None else _term_width()
        # Leave 1 col margin so the line never wraps.
        budget = max(10, width - 1)
        return _truncate(prefix + stage, budget)

    def _clear_progress_line(self) -> None:
        """Erase the sticky progress line in-place (no newline)."""
        # CSI 2K = erase entire line; \r = return to column 0
        self._stream.write("\033[2K\r")
        self._stream.flush()
        self._last_len = 0

    def clear_line(self) -> None:
        if not self.enabled or self._closed or not self._sticky:
            return
        with self._lock:
            self._clear_progress_line()

    def _render_unlocked(self, label: str | None = None) -> None:
        if not self.enabled or self._closed:
            return
        if label is not None:
            self._label = label
        width = _term_width()
        line = self._status_text(max_width=width)
        if self._sticky:
            # Erase + rewrite same line; never emit \n while running.
            self._stream.write("\033[2K\r")
            self._stream.write(line)
            self._stream.flush()
            self._last_len = len(line)
            return
        # Non-TTY fallback: throttle so we don't spam a new line every heartbeat.
        now = time.time()
        pct = 100.0 * self.done / self.total
        if (now - self._last_print_t) < 5.0 and (pct - self._last_print_pct) < 10.0:
            return
        print(line, flush=True)
        self._last_print_t = now
        self._last_print_pct = pct

    def render(self, label: str | None = None) -> None:
        with self._lock:
            self._render_unlocked(label)

    def log(self, msg: str = "", **kwargs: Any) -> None:
        """Print a log above the sticky bar, then redraw the bar on the next line."""
        end = kwargs.pop("end", "\n")
        file = kwargs.pop("file", self._stream)
        with self._lock:
            if self._sticky and not self._closed:
                self._clear_progress_line()
            print(msg, end=end, file=file, **kwargs)
            if self._sticky and not self._closed and (end.endswith("\n") or end == "\n"):
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
                    self._stream.write("\033[2K\r")
                    self._stream.write(line + "\n")
                    self._stream.flush()
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
