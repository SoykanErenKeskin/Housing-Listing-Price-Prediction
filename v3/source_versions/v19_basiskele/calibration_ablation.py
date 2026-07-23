"""Minimal V19 calibration × ensemble ablation (residual_log only)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# experiment, calibration_mode, ensemble_profile
MINIMAL_CALIBRATION_EXPERIMENTS: list[tuple[str, str, str]] = [
    ("control_none_balanced", "none", "balanced"),
    ("linear_balanced", "linear", "balanced"),
    ("isotonic_balanced", "isotonic", "balanced"),
    ("control_none_no_ridge", "none", "no_ridge"),
    ("linear_no_ridge", "linear", "no_ridge"),
    ("isotonic_no_ridge", "isotonic", "no_ridge"),
]

CONTROL_EXPERIMENT = "control_none_balanced"
TARGET_PROFILE_FIXED = "residual_log"
# Soft floor: variance_ratio must not drop more than this vs control
VARIANCE_RATIO_MAX_DROP = 0.02

ABLATION_CSV_COLUMNS = [
    "experiment",
    "calibration_mode",
    "ensemble_profile",
    "target_profile",
    "rows",
    "r2",
    "log_r2",
    "mape",
    "mae",
    "median_ape",
    "variance_ratio",
    "large_home_r2",
    "large_home_mape",
    "cheap_decile_bias",
    "expensive_decile_bias",
    "selected",
    "notes",
]


def _f(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _read_decile_bias(reports_dir: Path) -> tuple[float, float]:
    path = reports_dir / "basiskele_decile_bias_v19_basiskele.csv"
    if not path.is_file():
        return float("nan"), float("nan")
    try:
        df = pd.read_csv(path)
        if df.empty or "mean_bias" not in df.columns:
            return float("nan"), float("nan")
        # Prefer numeric decile extremes when present
        if "decile" in df.columns:
            d = pd.to_numeric(df["decile"], errors="coerce")
            cheap = df.loc[d == d.min(), "mean_bias"]
            rich = df.loc[d == d.max(), "mean_bias"]
            return _f(cheap.iloc[0] if len(cheap) else np.nan), _f(rich.iloc[0] if len(rich) else np.nan)
        return _f(df["mean_bias"].iloc[0]), _f(df["mean_bias"].iloc[-1])
    except Exception:
        return float("nan"), float("nan")


def _read_large_home_mape(reports_dir: Path, metrics_summary: dict[str, Any]) -> float:
    path = reports_dir / "basiskele_large_home_error_v19_basiskele.csv"
    if path.is_file():
        try:
            df = pd.read_csv(path)
            if "is_large_home" in df.columns and "mape" in df.columns:
                hit = df.loc[pd.to_numeric(df["is_large_home"], errors="coerce") == 1, "mape"]
                if len(hit):
                    return _f(hit.iloc[0])
        except Exception:
            pass
    for row in metrics_summary.get("segment_layer") or []:
        if str(row.get("segment")) == "large_home":
            for key in ("best_blended_mape", "base_mape", "segment_mape"):
                if row.get(key) is not None and np.isfinite(_f(row.get(key))):
                    return _f(row.get(key))
    return float("nan")


def build_calibration_ablation_row(
    *,
    experiment: str,
    calibration_mode: str,
    ensemble_profile: str,
    metrics_summary: dict[str, Any],
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    ens = metrics_summary.get("ensemble") or {}
    cheap, rich = (float("nan"), float("nan"))
    lh_mape = float("nan")
    if reports_dir is not None:
        cheap, rich = _read_decile_bias(reports_dir)
        lh_mape = _read_large_home_mape(reports_dir, metrics_summary)
    return {
        "experiment": experiment,
        "calibration_mode": calibration_mode,
        "ensemble_profile": ensemble_profile,
        "target_profile": TARGET_PROFILE_FIXED,
        "rows": ens.get("rows") if ens.get("rows") is not None else metrics_summary.get("rows"),
        "r2": metrics_summary.get("r2") if metrics_summary.get("r2") is not None else ens.get("r2"),
        "log_r2": ens.get("log_r2"),
        "mape": metrics_summary.get("mape") if metrics_summary.get("mape") is not None else ens.get("mape"),
        "mae": ens.get("mae_tl_per_m2"),
        "median_ape": ens.get("median_ape"),
        "variance_ratio": metrics_summary.get("variance_ratio")
        if metrics_summary.get("variance_ratio") is not None
        else ens.get("basiskele_variance_ratio"),
        "large_home_r2": metrics_summary.get("large_home_r2"),
        "large_home_mape": lh_mape,
        "cheap_decile_bias": cheap,
        "expensive_decile_bias": rich,
        "selected": False,
        "notes": "",
    }


def leakage_guard_pass(metrics_summary: dict[str, Any]) -> bool:
    guard = metrics_summary.get("calibration_leakage_guard")
    if guard is None:
        ens = metrics_summary.get("ensemble") or {}
        guard = ens.get("calibration_guard") or {}
    if not guard:
        # none-mode still writes a pass=true guard; missing guard => fail closed
        return False
    return bool(guard.get("pass", False))


def calibration_ablation_eligible(row: dict[str, Any], control: dict[str, Any], metrics_by_exp: dict[str, dict[str, Any]]) -> bool:
    """Eligible if beats control on R²/MAPE, leakage passes, variance_ratio not much worse."""
    name = str(row.get("experiment") or "")
    if name == CONTROL_EXPERIMENT:
        return False
    ms = metrics_by_exp.get(name) or {}
    if not leakage_guard_pass(ms):
        return False
    try:
        r2 = _f(row.get("r2"))
        mape = _f(row.get("mape"))
        vr = _f(row.get("variance_ratio"))
        c_r2 = _f(control.get("r2"))
        c_mape = _f(control.get("mape"))
        c_vr = _f(control.get("variance_ratio"))
        if not (np.isfinite(r2) and np.isfinite(mape) and np.isfinite(c_r2) and np.isfinite(c_mape)):
            return False
        if not (r2 > c_r2 and mape <= c_mape + 0.005 + 1e-12):
            return False
        if np.isfinite(vr) and np.isfinite(c_vr) and vr < c_vr - VARIANCE_RATIO_MAX_DROP:
            return False
        return True
    except Exception:
        return False


def select_calibration_ablation(
    rows: list[dict[str, Any]],
    metrics_by_exp: dict[str, dict[str, Any]],
) -> str:
    by_exp = {str(r.get("experiment")): r for r in rows}
    control = by_exp.get(CONTROL_EXPERIMENT) or {}
    candidates = [r for r in rows if calibration_ablation_eligible(r, control, metrics_by_exp)]
    if candidates:
        pick = max(
            candidates,
            key=lambda r: (
                _f(r.get("r2"), -9.0),
                _f(r.get("variance_ratio"), -9.0),
                -_f(r.get("mape"), 9.0),
            ),
        )
        return str(pick.get("experiment"))
    return CONTROL_EXPERIMENT


def apply_selection_flags(rows: list[dict[str, Any]], pick: str) -> list[dict[str, Any]]:
    for r in rows:
        is_sel = str(r.get("experiment")) == pick
        r["selected"] = bool(is_sel)
        if is_sel:
            if pick == CONTROL_EXPERIMENT:
                r["notes"] = (r.get("notes") or "").strip()
                if r["notes"]:
                    r["notes"] += "; "
                r["notes"] += "no candidate beat control; keep control_none_balanced (calibration_mode=none)"
            elif not r.get("notes"):
                r["notes"] = "selected vs control_none_balanced (R2 lift + MAPE/variance/leakage guards)"
            else:
                r["notes"] = str(r["notes"]) + "; selected"
    return rows


def write_ablation_csv(rows: list[dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(rows)
    for c in ABLATION_CSV_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan if c not in {"experiment", "notes", "selected"} else (False if c == "selected" else "")
    df = df[ABLATION_CSV_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_experiment_mini_summary(reports_root: Path, experiment: str, metrics_summary: dict[str, Any]) -> Path:
    dest_dir = reports_root / "ablation_runs" / experiment
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "metrics_summary.json"
    dest.write_text(json.dumps(metrics_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return dest


def promote_selected_bundle(exp_artifacts_dir: Path, top_artifacts_dir: Path) -> Path:
    src = exp_artifacts_dir / "model_bundle_v19_basiskele.joblib"
    top_artifacts_dir.mkdir(parents=True, exist_ok=True)
    dest = top_artifacts_dir / "model_bundle_v19_basiskele.joblib"
    if not src.is_file():
        raise FileNotFoundError(f"Selected experiment bundle missing: {src}")
    shutil.copy2(src, dest)
    return dest
