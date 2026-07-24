"""V22 satellite / environment feature name lists and mode resolution.

Free Sentinel-style environmental proxies (NDVI/NDWI/NDBI/brightness/texture).
No paid map APIs / RGB crops / CNN. No target leakage — external rasters only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SATELLITE_FEATURE_MODES = ("none", "basic", "radii", "full")

# Core 250m pack
SAT_BASIC_250M = [
    "sat_ndvi_mean_250m",
    "sat_ndwi_mean_250m",
    "sat_ndbi_mean_250m",
    "sat_brightness_mean_250m",
]

# Multi-radius indices
SAT_RADII = [
    "sat_ndvi_mean_100m",
    "sat_ndvi_mean_250m",
    "sat_ndvi_mean_500m",
    "sat_ndwi_mean_100m",
    "sat_ndwi_mean_250m",
    "sat_ndwi_mean_500m",
    "sat_ndbi_mean_100m",
    "sat_ndbi_mean_250m",
    "sat_ndbi_mean_500m",
]

# Full pack (all radii + shares + texture + meta)
SAT_FULL_EXTRA = [
    "sat_ndvi_std_250m",
    "sat_brightness_mean_100m",
    "sat_brightness_mean_250m",
    "sat_brightness_mean_500m",
    "sat_green_share_250m",
    "sat_water_share_250m",
    "sat_builtup_share_250m",
    "sat_texture_proxy_250m",
    "sat_cloud_coverage_proxy",
    "sat_feature_year",
    "sat_feature_month",
]

SAT_META = [
    "sat_has_features",
]

# Usable environmental coverage probes (before imputation)
USABLE_SAT_COVERAGE_COLS = [
    "sat_ndvi_mean_250m",
    "sat_ndbi_mean_250m",
    "sat_ndwi_mean_250m",
]

# All numeric satellite columns that may appear in CSV / training matrix
ALL_SATELLITE_NUMERIC_FEATURES: list[str] = sorted(
    set(SAT_BASIC_250M + SAT_RADII + SAT_FULL_EXTRA + SAT_META)
)

# Template columns for empty / fetch-required CSV
TEMPLATE_COLUMNS: list[str] = [
    "classified_id",
    "latitude",
    "longitude",
    "location_precision",
    "location_source",
    "county",
    "district",
    "sat_missing_reason",
    *ALL_SATELLITE_NUMERIC_FEATURES,
]


def normalize_satellite_feature_mode(mode: str | None) -> str:
    m = str(mode or "none").strip().lower()
    if m not in SATELLITE_FEATURE_MODES:
        raise ValueError(f"Unsupported satellite_feature_mode={mode!r}; expected one of {SATELLITE_FEATURE_MODES}")
    return m


def get_satellite_feature_names(mode: str | None) -> list[str]:
    """Numeric satellite columns for a mode (before zero-variance filtering)."""
    m = normalize_satellite_feature_mode(mode)
    if m == "none":
        return []
    if m == "basic":
        return list(SAT_BASIC_250M) + list(SAT_META)
    if m == "radii":
        return list(SAT_RADII) + list(SAT_META)
    return list(dict.fromkeys(SAT_RADII + SAT_FULL_EXTRA + SAT_META))


def detect_zero_variance_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> list[str]:
    """Return columns with no usable variance (all-null or nunique<=1 among non-null)."""
    cols = list(columns or [])
    zero: list[str] = []
    for c in cols:
        if c not in df.columns:
            zero.append(c)
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        nn = s.dropna()
        if len(nn) == 0:
            zero.append(c)
            continue
        if int(nn.nunique(dropna=True)) <= 1:
            zero.append(c)
            continue
        std = float(nn.std(ddof=0))
        if not np.isfinite(std) or std == 0.0:
            zero.append(c)
    return zero


def usable_sat_feature_coverage(df: pd.DataFrame) -> float:
    """Share of rows with any non-null NDVI/NDBI/NDWI 250m (pre-imputation usable signal)."""
    if df is None or df.empty:
        return 0.0
    cols = [c for c in USABLE_SAT_COVERAGE_COLS if c in df.columns]
    if not cols:
        return 0.0
    block = df[cols].apply(pd.to_numeric, errors="coerce")
    return float(block.notna().any(axis=1).mean())


def get_model_satellite_feature_names(
    mode: str | None,
    df: pd.DataFrame | None = None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Mode feature names for the model, excluding zero-variance columns.

    Returns (used_in_model, zero_variance_excluded, qa_info).
    """
    names = get_satellite_feature_names(mode)
    if not names:
        return [], [], {"mode": "none", "zero_variance_excluded": [], "used_in_model": []}
    if df is None or df.empty:
        return list(names), [], {
            "mode": normalize_satellite_feature_mode(mode),
            "zero_variance_excluded": [],
            "used_in_model": list(names),
            "note": "no_frame_for_variance_check",
        }
    zero = detect_zero_variance_features(df, names)
    used = [n for n in names if n not in zero]
    return used, zero, {
        "mode": normalize_satellite_feature_mode(mode),
        "zero_variance_excluded": list(zero),
        "used_in_model": list(used),
        "n_mode_features": len(names),
        "n_used_in_model": len(used),
        "n_zero_variance_excluded": len(zero),
    }


def satellite_ablation_experiments() -> list[tuple[str, str]]:
    """(experiment_name, satellite_feature_mode) for V22 ablation."""
    return [
        ("control_v21", "none"),
        ("sat_basic_250m", "basic"),
        ("sat_radii", "radii"),
        ("sat_full", "full"),
    ]


def empty_template_row() -> dict[str, object]:
    row = {c: None for c in TEMPLATE_COLUMNS}
    row["sat_has_features"] = 0
    row["sat_missing_reason"] = "template_only"
    return row
