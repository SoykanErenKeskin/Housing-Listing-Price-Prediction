"""Join satellite environment features onto listing frames by classified_id."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from satellite_feature_builder import (
    ALL_SATELLITE_NUMERIC_FEATURES,
    TEMPLATE_COLUMNS,
    USABLE_SAT_COVERAGE_COLS,
    get_model_satellite_feature_names,
    get_satellite_feature_names,
    normalize_satellite_feature_mode,
    usable_sat_feature_coverage,
)

# Minimum usable NDVI/NDBI/NDWI coverage to allow satellite training smoke/ablation arms
SAT_COVERAGE_GATE = 0.65


def estimate_satellite_csv_coverage(csv_path: str | Path | None) -> float:
    p = Path(csv_path) if csv_path else None
    if p is None or not p.exists():
        return 0.0
    want = ["classified_id", *USABLE_SAT_COVERAGE_COLS, "sat_has_features"]
    try:
        df = pd.read_csv(p, encoding="utf-8-sig", usecols=lambda c: c in want, low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
        except Exception:
            return 0.0
    return usable_sat_feature_coverage(df)


def resolve_satellite_csv_path(
    path: str | Path | None,
    *,
    min_coverage: float | None = None,
) -> Path | None:
    """Return path if CSV is a usable Sentinel feature table (not template).

    If min_coverage is set (e.g. 0.65), also require sat_has_features rate.
    """
    if path is None or str(path).strip() == "":
        return None
    p = Path(path)
    if not p.exists() or p.stat().st_size < 50:
        return None
    try:
        head = pd.read_csv(p, encoding="utf-8-sig", nrows=50)
    except Exception:
        return None
    if "classified_id" not in head.columns:
        return None
    if "sat_missing_reason" in head.columns:
        reasons = head["sat_missing_reason"].astype(str).str.lower()
        if len(head) <= 2 and reasons.str.contains("template").all():
            return None
    if "sat_has_features" in head.columns:
        has = pd.to_numeric(head["sat_has_features"], errors="coerce").fillna(0)
        if float(has.max()) <= 0 and len(head) <= 5:
            return None
    if min_coverage is not None:
        cov = estimate_satellite_csv_coverage(p)
        if cov < float(min_coverage):
            return None
    return p


def load_satellite_feature_table(csv_path: str | Path | None) -> pd.DataFrame:
    p = resolve_satellite_csv_path(csv_path)
    if p is None:
        return pd.DataFrame(columns=TEMPLATE_COLUMNS)
    df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
    if "classified_id" not in df.columns:
        raise ValueError(f"Satellite CSV missing classified_id: {p}")
    df = df.copy()
    df["classified_id"] = df["classified_id"].astype(str).str.strip()
    df = df.drop_duplicates("classified_id", keep="last")
    for col in ALL_SATELLITE_NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    if "sat_has_features" not in df.columns:
        probe = [c for c in ALL_SATELLITE_NUMERIC_FEATURES if c != "sat_has_features" and c in df.columns]
        if probe:
            df["sat_has_features"] = df[probe].notna().any(axis=1).astype(float)
        else:
            df["sat_has_features"] = 0.0
    else:
        df["sat_has_features"] = pd.to_numeric(df["sat_has_features"], errors="coerce").fillna(0.0)
    return df


def join_satellite_features(
    sales: pd.DataFrame,
    satellite_csv: str | Path | None,
    satellite_feature_mode: str = "full",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Left-join satellite features; always create sat_* columns for requested mode."""
    mode = normalize_satellite_feature_mode(satellite_feature_mode)
    out = sales.copy()
    if "classified_id" in out.columns:
        out["classified_id"] = out["classified_id"].astype(str).str.strip()

    report: dict[str, Any] = {
        "satellite_feature_mode": mode,
        "satellite_csv": str(satellite_csv) if satellite_csv else None,
        "satellite_csv_exists": resolve_satellite_csv_path(satellite_csv) is not None,
        "rows": int(len(out)),
        "sat_feature_coverage": 0.0,
        "coverage_gate": SAT_COVERAGE_GATE,
        "coverage_gate_pass": False,
        "zero_variance_excluded": [],
        "used_in_model": [],
        "warning": None,
    }

    wanted = get_satellite_feature_names(mode if mode != "none" else "full")
    ensure_cols = list(dict.fromkeys(wanted + ["sat_has_features", "sat_missing_reason"]))

    sat = load_satellite_feature_table(satellite_csv)
    if sat.empty or not report["satellite_csv_exists"]:
        for col in ensure_cols:
            if col == "sat_has_features":
                out[col] = 0.0
            elif col == "sat_missing_reason":
                out[col] = "satellite_csv_missing"
            else:
                out[col] = np.nan
        report["warning"] = "satellite_csv_missing_or_empty"
        report["sat_feature_coverage"] = 0.0
        return out, report

    if "classified_id" not in out.columns:
        for col in ensure_cols:
            if col == "sat_has_features":
                out[col] = 0.0
            elif col == "sat_missing_reason":
                out[col] = "no_classified_id"
            else:
                out[col] = np.nan
        report["warning"] = "sales_missing_classified_id"
        return out, report

    keep_sat_cols = ["classified_id"] + [c for c in sat.columns if c != "classified_id"]
    merged = out.merge(sat[keep_sat_cols], on="classified_id", how="left", suffixes=("", "_sat"))

    for col in ensure_cols:
        if col not in merged.columns:
            if col == "sat_has_features":
                merged[col] = 0.0
            elif col == "sat_missing_reason":
                merged[col] = "no_match"
            else:
                merged[col] = np.nan
        elif col.endswith("_sat"):
            continue
        else:
            alt = f"{col}_sat"
            if alt in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[alt])
                merged = merged.drop(columns=[alt])

    has = pd.to_numeric(merged.get("sat_has_features"), errors="coerce").fillna(0.0)
    # Probe usable sat signals only — sat_has_features itself is always non-null and must not
    # inflate has/coverage to 1.0.
    probe_cols = [c for c in USABLE_SAT_COVERAGE_COLS if c in merged.columns]
    if not probe_cols:
        probe_cols = [c for c in wanted if c in merged.columns and c not in {"sat_has_features", "sat_missing_reason"}]
    if probe_cols:
        any_feat = merged[probe_cols].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
        has = has.where(has > 0, any_feat.astype(float))
        # If CSV flagged has_features=1 but usable indices are all null, downgrade
        has = has.where(any_feat, 0.0)
    merged["sat_has_features"] = has.astype(float)
    missing = merged.get("sat_missing_reason")
    if missing is None:
        merged["sat_missing_reason"] = np.where(merged["sat_has_features"] > 0, "", "no_match_or_coords")
    else:
        merged["sat_missing_reason"] = missing.where(merged["sat_has_features"] > 0, missing.fillna("no_match_or_coords"))

    coverage = usable_sat_feature_coverage(merged)
    report["sat_feature_coverage"] = coverage
    report["sat_has_features_rate"] = float((merged["sat_has_features"] > 0).mean()) if len(merged) else 0.0
    _probe = [c for c in USABLE_SAT_COVERAGE_COLS if c in merged.columns]
    report["sat_rows_with_features"] = (
        int(merged[_probe].apply(pd.to_numeric, errors="coerce").notna().any(axis=1).sum()) if _probe else 0
    )
    report["coverage_definition"] = "non_null_any_of_sat_ndvi_mean_250m|sat_ndbi_mean_250m|sat_ndwi_mean_250m"
    report["coverage_gate_pass"] = bool(coverage >= SAT_COVERAGE_GATE)

    # Zero-variance QA relative to the requested training mode
    used, zero, qa = get_model_satellite_feature_names(mode, merged)
    report["used_in_model"] = list(used)
    report["zero_variance_excluded"] = list(zero)
    report["feature_qa"] = qa
    if zero:
        report["warning"] = (
            (report["warning"] + "; " if report["warning"] else "")
            + "zero_variance_excluded:"
            + ",".join(zero)
        )
    return merged, report
