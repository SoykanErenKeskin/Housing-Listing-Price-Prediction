"""OOF-safe calibration layers for V19 (no blind variance lift)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression


@dataclass
class CalibrationResult:
    calibrated: np.ndarray
    mode: str
    correction_cap: float
    guard: dict[str, Any] = field(default_factory=dict)
    bin_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    curve_report: pd.DataFrame = field(default_factory=pd.DataFrame)


def _to_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(np.asarray(x, dtype=float), 1.0))


def _from_log(x: np.ndarray) -> np.ndarray:
    return np.exp(np.asarray(x, dtype=float))


def _apply_cap(pred_log: np.ndarray, cal_log: np.ndarray, cap: float) -> np.ndarray:
    if cap is None or float(cap) <= 0:
        return cal_log
    delta = np.clip(cal_log - pred_log, -float(cap), float(cap))
    return pred_log + delta


class IdentityCalibrator:
    def fit(self, pred: np.ndarray, actual: np.ndarray):
        return self

    def transform(self, pred: np.ndarray) -> np.ndarray:
        return np.asarray(pred, dtype=float)


class LinearLogCalibrator:
    def __init__(self, cap: float = 0.10):
        self.cap = float(cap)
        self.model = LinearRegression()
        self._fitted = False

    def fit(self, pred: np.ndarray, actual: np.ndarray):
        pl = _to_log(pred)
        al = _to_log(actual)
        m = np.isfinite(pl) & np.isfinite(al)
        if m.sum() < 20:
            self._fitted = False
            return self
        self.model.fit(pl[m].reshape(-1, 1), al[m])
        self._fitted = True
        return self

    def transform(self, pred: np.ndarray) -> np.ndarray:
        pl = _to_log(pred)
        if not self._fitted:
            return np.asarray(pred, dtype=float)
        cal = self.model.predict(pl.reshape(-1, 1))
        cal = _apply_cap(pl, cal, self.cap)
        return _from_log(cal)


class IsotonicLogCalibrator:
    def __init__(self, cap: float = 0.10):
        self.cap = float(cap)
        self.model = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(self, pred: np.ndarray, actual: np.ndarray):
        pl = _to_log(pred)
        al = _to_log(actual)
        m = np.isfinite(pl) & np.isfinite(al)
        if m.sum() < 30:
            self._fitted = False
            return self
        self.model.fit(pl[m], al[m])
        self._fitted = True
        return self

    def transform(self, pred: np.ndarray) -> np.ndarray:
        pl = _to_log(pred)
        if not self._fitted:
            return np.asarray(pred, dtype=float)
        cal = self.model.predict(pl)
        cal = _apply_cap(pl, cal, self.cap)
        return _from_log(cal)


class BinLogCalibrator:
    def __init__(self, n_bins: int = 10, min_bin_count: int = 30, cap: float = 0.10, smooth: float = 0.5):
        self.n_bins = int(n_bins)
        self.min_bin_count = int(min_bin_count)
        self.cap = float(cap)
        self.smooth = float(smooth)
        self.edges_: np.ndarray | None = None
        self.corrections_: np.ndarray | None = None
        self.bin_report_: pd.DataFrame = pd.DataFrame()

    def fit(self, pred: np.ndarray, actual: np.ndarray):
        pl = _to_log(pred)
        al = _to_log(actual)
        m = np.isfinite(pl) & np.isfinite(al)
        pl, al = pl[m], al[m]
        if len(pl) < self.min_bin_count * 2:
            self.edges_ = None
            return self
        qs = np.linspace(0, 1, self.n_bins + 1)
        edges = np.unique(np.quantile(pl, qs))
        if len(edges) < 3:
            self.edges_ = None
            return self
        self.edges_ = edges
        bins = np.digitize(pl, edges[1:-1], right=True)
        corr = np.zeros(len(edges) - 1, dtype=float)
        rows = []
        for b in range(len(corr)):
            mask = bins == b
            n = int(mask.sum())
            if n < self.min_bin_count:
                c = 0.0
            else:
                c = float(np.median(al[mask]) - np.median(pl[mask]))
                c *= self.smooth
                c = float(np.clip(c, -self.cap, self.cap))
            corr[b] = c
            rows.append({"bin": b, "n": n, "correction_log": c})
        self.corrections_ = corr
        self.bin_report_ = pd.DataFrame(rows)
        return self

    def transform(self, pred: np.ndarray) -> np.ndarray:
        pl = _to_log(pred)
        if self.edges_ is None or self.corrections_ is None:
            return np.asarray(pred, dtype=float)
        bins = np.digitize(pl, self.edges_[1:-1], right=True)
        bins = np.clip(bins, 0, len(self.corrections_) - 1)
        cal = pl + self.corrections_[bins]
        cal = _apply_cap(pl, cal, self.cap)
        return _from_log(cal)


class QuantileMapLogCalibrator:
    """Mild predicted-quantile → actual-quantile mapping with cap."""

    def __init__(self, n_quantiles: int = 20, cap: float = 0.10, blend: float = 0.5):
        self.n_quantiles = int(n_quantiles)
        self.cap = float(cap)
        self.blend = float(blend)
        self.pred_q_: np.ndarray | None = None
        self.actual_q_: np.ndarray | None = None

    def fit(self, pred: np.ndarray, actual: np.ndarray):
        pl = _to_log(pred)
        al = _to_log(actual)
        m = np.isfinite(pl) & np.isfinite(al)
        pl, al = pl[m], al[m]
        if len(pl) < 40:
            return self
        qs = np.linspace(0.05, 0.95, self.n_quantiles)
        self.pred_q_ = np.quantile(pl, qs)
        self.actual_q_ = np.quantile(al, qs)
        return self

    def transform(self, pred: np.ndarray) -> np.ndarray:
        pl = _to_log(pred)
        if self.pred_q_ is None or self.actual_q_ is None:
            return np.asarray(pred, dtype=float)
        mapped = np.interp(pl, self.pred_q_, self.actual_q_, left=self.actual_q_[0], right=self.actual_q_[-1])
        cal = (1.0 - self.blend) * pl + self.blend * mapped
        cal = _apply_cap(pl, cal, self.cap)
        return _from_log(cal)


def make_calibrator(mode: str, cap: float = 0.10):
    mode = str(mode or "none").lower()
    if mode in {"none", "off", ""}:
        return IdentityCalibrator()
    if mode == "linear":
        return LinearLogCalibrator(cap=cap)
    if mode == "isotonic":
        return IsotonicLogCalibrator(cap=cap)
    if mode == "bin":
        return BinLogCalibrator(cap=cap)
    if mode in {"quantile_map", "quantile"}:
        return QuantileMapLogCalibrator(cap=cap)
    raise ValueError(f"Unknown calibration mode: {mode}")


def apply_oof_safe_calibration(
    pred: np.ndarray,
    actual: np.ndarray,
    fold_ids: np.ndarray,
    *,
    mode: str = "none",
    correction_cap: float = 0.10,
) -> CalibrationResult:
    """Fit calibrator per outer fold on other folds only (val actuals never used in fit)."""
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    fold_ids = np.asarray(fold_ids)
    out = pred.copy()
    notes: list[str] = []
    mode = str(mode or "none").lower()
    bin_parts: list[pd.DataFrame] = []
    curve_rows: list[dict[str, Any]] = []

    if mode in {"none", "off", ""}:
        guard = {
            "pass": True,
            "outer_validation_targets_used_in_calibrator": False,
            "inner_oof_used": False,
            "calibration_mode": "none",
            "correction_cap": float(correction_cap),
            "notes": ["calibration disabled"],
        }
        return CalibrationResult(calibrated=out, mode="none", correction_cap=float(correction_cap), guard=guard)

    unique_folds = sorted(pd.unique(fold_ids))
    for f in unique_folds:
        val = fold_ids == f
        train = ~val
        if train.sum() < 30 or val.sum() < 5:
            notes.append(f"fold_{f}_skipped_small")
            continue
        cal = make_calibrator(mode, cap=correction_cap)
        # Fit ONLY on train folds (never val actuals)
        cal.fit(pred[train], actual[train])
        out[val] = cal.transform(pred[val])
        if hasattr(cal, "bin_report_") and isinstance(cal.bin_report_, pd.DataFrame) and not cal.bin_report_.empty:
            tmp = cal.bin_report_.copy()
            tmp["fold"] = f
            bin_parts.append(tmp)
        # curve diagnostics on train fit
        pl = _to_log(pred[train])
        al = _to_log(actual[train])
        curve_rows.append(
            {
                "fold": f,
                "n_train": int(train.sum()),
                "n_val": int(val.sum()),
                "pred_log_mean": float(np.nanmean(pl)),
                "actual_log_mean": float(np.nanmean(al)),
            }
        )

    guard = {
        "pass": True,
        "outer_validation_targets_used_in_calibrator": False,
        "inner_oof_used": True,
        "calibration_mode": mode,
        "correction_cap": float(correction_cap),
        "n_folds": int(len(unique_folds)),
        "notes": notes,
    }
    return CalibrationResult(
        calibrated=np.maximum(out, 0.0),
        mode=mode,
        correction_cap=float(correction_cap),
        guard=guard,
        bin_report=pd.concat(bin_parts, ignore_index=True) if bin_parts else pd.DataFrame(),
        curve_report=pd.DataFrame(curve_rows),
    )


def assign_fold_ids(n: int, n_splits: int, random_state: int = 42) -> np.ndarray:
    from sklearn.model_selection import KFold

    fold_ids = np.full(n, -1, dtype=int)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for i, (_, va) in enumerate(cv.split(np.arange(n))):
        fold_ids[va] = i
    return fold_ids
