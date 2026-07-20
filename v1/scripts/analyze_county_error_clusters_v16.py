#!/usr/bin/env python
"""V16 Phase-0 county error-cluster diagnostics (OOF only, no model changes).

Answers:
1) Which Başiskele segments hurt R² most?
2) Which price deciles show mean-pulling bias?
3) Is Başiskele large_home a primary culprit?
4) Is Karamürsel error concentrated in districts?
5) Karamürsel: sparsity vs age vs district?
6) Top 3 V16 experiments to try next.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for cand in [here, *here.parents]:
        if (cand / "MANIFEST.json").is_file() and (cand / "data").is_dir():
            return cand
    return here.parents[1]


ROOT = _repo_root()
V15 = ROOT / "v1" / "source_versions" / "v15"
DEFAULT_OOF = V15 / "outputs" / "v15_full" / "data" / "output" / "oof_predictions_v15.csv"
DEFAULT_SALES = V15 / "outputs" / "v15_full" / "data" / "input" / "sales_cleaned_v15.csv"
DEFAULT_METRICS = V15 / "outputs" / "v15_full" / "reports" / "metrics_summary_v15.json"
DEFAULT_OUT = ROOT / "v1" / "outputs" / "v16_diagnostics"

# Optional V15 helpers for deterministic quality / premium proxies
sys.path.insert(0, str(V15))
try:
    from attribute_features import add_attribute_quality_features
except Exception:  # pragma: no cover
    add_attribute_quality_features = None  # type: ignore

try:
    from county_specialist_features import BasiskelePremiumSpecialistAdder, LargeHomeFeatureAdder
except Exception:  # pragma: no cover
    BasiskelePremiumSpecialistAdder = None  # type: ignore
    LargeHomeFeatureAdder = None  # type: ignore


DETAIL_GROUPS = {
    "outside": "out_",
    "view": "view_",
    "inside": "in_",
    "front": "front_",
    "nearby": "near_",
    "subtype": "subtype_",
}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_r2(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) < 5:
        return float("nan")
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _mape(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p) & (y > 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y[mask] - p[mask]) / y[mask]))


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y[mask] - p[mask])))


def _var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    return float(np.var(x, ddof=1))


def parse_rooms(v: Any) -> float:
    if pd.isna(v):
        return np.nan
    s = str(v).replace(" ", "")
    import re

    m = re.search(r"(\d+)\+(\d+)", s)
    if m:
        return float(m.group(1)) + float(m.group(2)) * 0.5
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else np.nan


def enrich_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add diagnostic-only proxies (not training features)."""
    out = df.copy()
    notes: dict[str, Any] = {"proxies": [], "missing_requested": []}

    actual = _num(out["actual_unit_price_gross"])
    pred = _num(out["pred_ensemble"])
    out["residual"] = actual - pred
    out["abs_residual"] = out["residual"].abs()
    out["log_residual"] = np.log(np.clip(actual, 1.0, None)) - np.log(np.clip(pred, 1.0, None))
    out["ape"] = out["abs_residual"] / np.clip(actual, 1.0, None)

    gross = _num(out["gross_m2"]) if "gross_m2" in out.columns else pd.Series(np.nan, index=out.index)
    rooms = out["room_count"].map(parse_rooms) if "room_count" in out.columns else pd.Series(np.nan, index=out.index)
    out["rooms_numeric"] = rooms
    out["is_large_flat"] = ((gross >= 151) | (rooms >= 4)).fillna(False).astype(int)

    # Attribute quality (deterministic) if helper available
    if add_attribute_quality_features is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tmp = add_attribute_quality_features(out)
            if "attr_total_quality_score" in tmp.columns:
                out["attr_total_quality_score"] = _num(tmp["attr_total_quality_score"])
                notes["proxies"].append("attr_total_quality_score via attribute_features")
        except Exception as exc:
            notes["attr_quality_error"] = str(exc)

    if "attr_total_quality_score" not in out.columns:
        # lightweight fallback from amenity-ish fields
        elev = out["elevator"].astype(str).str.lower().isin(["var", "evet", "1", "true"]).astype(float) if "elevator" in out.columns else 0.0
        park = out["parking"].astype(str).str.lower().isin(["var", "evet", "1", "true"]).astype(float) if "parking" in out.columns else 0.0
        site = out["site_inside"].astype(str).str.lower().isin(["evet", "var", "1", "true"]).astype(float) if "site_inside" in out.columns else 0.0
        age = _num(out["building_age"]) if "building_age" in out.columns else pd.Series(20.0, index=out.index)
        out["attr_total_quality_score"] = (
            elev * 2 + park * 1.5 + site * 2 + np.clip((40 - age) / 10.0, 0, 4)
        )
        notes["proxies"].append("attr_total_quality_score lightweight fallback")

    # detail_effect_* proxies from binary group sums (NOT fold-safe OOF effects)
    for group, prefix in DETAIL_GROUPS.items():
        cols = [c for c in out.columns if str(c).startswith(prefix)]
        if cols:
            out[f"detail_effect_{group}_sum"] = out[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        else:
            out[f"detail_effect_{group}_sum"] = 0.0
            notes["missing_requested"].append(f"detail_effect_{group}_sum binaries")
    out["detail_effect_total_sum"] = (
        out["detail_effect_outside_sum"]
        + out["detail_effect_view_sum"]
        + out["detail_effect_inside_sum"]
        + out["detail_effect_front_sum"]
        + out["detail_effect_nearby_sum"]
        + out["detail_effect_subtype_sum"]
    )
    notes["proxies"].append(
        "detail_effect_*_sum = sum of present detail binaries (coverage proxy, not fold-safe residual effects)"
    )

    # location proxies from OOF actuals (diagnostic only)
    if "district" in out.columns and "county" in out.columns:
        out["district_target_median"] = out.groupby(["county", "district"], dropna=False)[
            "actual_unit_price_gross"
        ].transform("median")
        out["location_baseline_m2"] = out.groupby(["county", "district", "m2_group"], dropna=False)[
            "actual_unit_price_gross"
        ].transform("median")
        # fallback
        out["location_baseline_m2"] = out["location_baseline_m2"].fillna(out["district_target_median"])
        notes["proxies"].append(
            "location_baseline_m2 / district_target_median = OOF actual group medians (diagnostic proxies only)"
        )
    else:
        out["district_target_median"] = np.nan
        out["location_baseline_m2"] = np.nan
        notes["missing_requested"].extend(["district_target_median", "location_baseline_m2"])

    # large_home redesign-style proxies
    out["large_home_m2_excess"] = np.maximum(gross.fillna(0) - 150.0, 0.0) * out["is_large_flat"]
    out["large_home_log_m2"] = np.log1p(np.maximum(gross.fillna(0), 0.0)) * out["is_large_flat"]
    net = _num(out["net_m2"]) if "net_m2" in out.columns else pd.Series(np.nan, index=out.index)
    ratio = (net / np.maximum(gross, 1.0)).clip(0, 1.5).fillna(0.8)
    out["large_home_net_gross_quality"] = ratio * out["is_large_flat"]
    out["large_home_room_density"] = (rooms / np.maximum(gross, 1.0)).fillna(0) * out["is_large_flat"]
    out["large_home_quality_x_m2"] = _num(out["attr_total_quality_score"]).fillna(0) * gross.fillna(0) * out["is_large_flat"]
    out["large_home_detail_premium_x_m2"] = out["detail_effect_total_sum"] * gross.fillna(0) * out["is_large_flat"]
    site_bin = (
        out["site_inside"].astype(str).str.lower().isin(["evet", "var", "1", "true"]).astype(float)
        if "site_inside" in out.columns
        else 0.0
    )
    out["large_home_site_x_m2"] = site_bin * gross.fillna(0) * out["is_large_flat"]

    # Başiskele premium proxies via specialist adder if available (needs detail_effect cols we just created)
    if BasiskelePremiumSpecialistAdder is not None:
        try:
            adder = BasiskelePremiumSpecialistAdder(mode="premium")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adder.fit(out)
                tmp = adder.transform(out)
            for c in tmp.columns:
                if str(c).startswith("basiskele_"):
                    out[c] = tmp[c]
            notes["proxies"].append("basiskele_* via BasiskelePremiumSpecialistAdder on binary-sum proxies")
        except Exception as exc:
            notes["basiskele_premium_error"] = str(exc)

    if "basiskele_premium_score" not in out.columns:
        mask = out["county"].astype(str).eq("Başiskele") if "county" in out.columns else False
        out["basiskele_premium_score"] = np.where(
            mask,
            (
                0.3 * np.tanh(out["detail_effect_outside_sum"] / 5.0)
                + 0.25 * np.tanh(out["detail_effect_view_sum"] / 4.0)
                + 0.2 * np.tanh(out["detail_effect_inside_sum"] / 5.0)
                + 0.25 * np.tanh(_num(out["attr_total_quality_score"]).fillna(0) / 10.0)
            ),
            0.0,
        )
        notes["proxies"].append("basiskele_premium_score simple fallback")

    out["large_home_basiskele_premium"] = out["is_large_flat"] * _num(out["basiskele_premium_score"]).fillna(0)

    # Buckets
    out["attr_total_quality_bucket"] = pd.qcut(
        _num(out["attr_total_quality_score"]).rank(method="first"),
        q=5,
        labels=["q1_low", "q2", "q3", "q4", "q5_high"],
        duplicates="drop",
    ).astype(str)
    out["detail_effect_total_bucket"] = pd.qcut(
        out["detail_effect_total_sum"].rank(method="first"),
        q=5,
        labels=["q1_low", "q2", "q3", "q4", "q5_high"],
        duplicates="drop",
    ).astype(str)

    if "basiskele_premium_bucket" not in out.columns:
        # from score quantiles on Başiskele only, else none
        score = _num(out["basiskele_premium_score"]).fillna(0)
        bas = out["county"].astype(str).eq("Başiskele") if "county" in out.columns else pd.Series(False, index=out.index)
        low = high = np.nan
        if bas.sum() >= 30:
            low = float(score.loc[bas].quantile(0.30))
            high = float(score.loc[bas].quantile(0.70))
        bucket = np.where(~bas, "none", np.where(score >= high, "high", np.where(score <= low, "low", "mid")))
        out["basiskele_premium_bucket"] = bucket
        notes["proxies"].append("basiskele_premium_bucket from score quantiles on Başiskele")

    return out, notes


def decile_bias_table(df: pd.DataFrame, county: str) -> pd.DataFrame:
    sub = df[df["county"].astype(str).eq(county)].copy()
    if sub.empty:
        return pd.DataFrame()
    y = _num(sub["actual_unit_price_gross"])
    p = _num(sub["pred_ensemble"])
    sub = sub.assign(actual=y, pred=p).dropna(subset=["actual", "pred"])
    sub["decile"] = pd.qcut(sub["actual"].rank(method="first"), 10, labels=False, duplicates="drop") + 1
    rows = []
    for d, g in sub.groupby("decile"):
        ya, yp = g["actual"].to_numpy(), g["pred"].to_numpy()
        bias = yp - ya
        rows.append(
            {
                "decile": int(d),
                "rows": int(len(g)),
                "actual_mean": float(np.mean(ya)),
                "pred_mean": float(np.mean(yp)),
                "mean_bias": float(np.mean(bias)),
                "median_bias": float(np.median(bias)),
                "mape": _mape(ya, yp),
                "mae": _mae(ya, yp),
                "r2_if_possible": _safe_r2(ya, yp),
                "actual_min": float(np.min(ya)),
                "actual_max": float(np.max(ya)),
                "pred_min": float(np.min(yp)),
                "pred_max": float(np.max(yp)),
                "bias_pct_of_actual": float(np.mean(bias) / np.mean(ya)) if np.mean(ya) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("decile")


def segment_error_table(df: pd.DataFrame, county: str, group_cols: list[str], min_rows: int = 8) -> pd.DataFrame:
    sub = df[df["county"].astype(str).eq(county)].copy()
    rows: list[dict[str, Any]] = []
    for col in group_cols:
        if col not in sub.columns:
            continue
        for key, g in sub.groupby(sub[col].fillna("missing").astype(str), dropna=False):
            if len(g) < min_rows:
                continue
            ya = _num(g["actual_unit_price_gross"]).to_numpy()
            yp = _num(g["pred_ensemble"]).to_numpy()
            mask = np.isfinite(ya) & np.isfinite(yp)
            ya, yp = ya[mask], yp[mask]
            if len(ya) < min_rows:
                continue
            av, pv = _var(ya), _var(yp)
            rows.append(
                {
                    "segment_col": col,
                    "segment_value": key,
                    "rows": int(len(ya)),
                    "actual_mean": float(np.mean(ya)),
                    "pred_mean": float(np.mean(yp)),
                    "mean_bias": float(np.mean(yp - ya)),
                    "abs_bias": float(np.mean(np.abs(yp - ya))),
                    "mape": _mape(ya, yp),
                    "r2": _safe_r2(ya, yp),
                    "pred_variance": pv,
                    "actual_variance": av,
                    "variance_ratio": (pv / av) if av and np.isfinite(av) and av > 0 else np.nan,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["r2", "mape"], ascending=[True, False])


def large_home_error_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[(df["county"].astype(str).eq("Başiskele")) & (df["is_large_flat"] == 1)].copy()
    dims = [
        "district",
        "m2_group",
        "building_age_group",
        "room_count",
        "site_inside",
        "detail_effect_total_bucket",
    ]
    rows = []
    # overall
    ya = _num(sub["actual_unit_price_gross"]).to_numpy()
    yp = _num(sub["pred_ensemble"]).to_numpy()
    av, pv = _var(ya), _var(yp)
    rows.append(
        {
            "slice": "ALL_BASISKELE_LARGE_HOME",
            "district": "ALL",
            "m2_group": "ALL",
            "building_age_group": "ALL",
            "room_count": "ALL",
            "site_inside": "ALL",
            "detail_effect_total_bucket": "ALL",
            "rows": int(len(sub)),
            "r2": _safe_r2(ya, yp),
            "mape": _mape(ya, yp),
            "mean_bias": float(np.nanmean(yp - ya)),
            "actual_mean": float(np.nanmean(ya)),
            "pred_mean": float(np.nanmean(yp)),
            "variance_ratio": (pv / av) if av and av > 0 else np.nan,
        }
    )
    for col in dims:
        if col not in sub.columns:
            continue
        for key, g in sub.groupby(sub[col].fillna("missing").astype(str)):
            if len(g) < 8:
                continue
            ya = _num(g["actual_unit_price_gross"]).to_numpy()
            yp = _num(g["pred_ensemble"]).to_numpy()
            av, pv = _var(ya), _var(yp)
            row = {
                "slice": col,
                "district": key if col == "district" else "",
                "m2_group": key if col == "m2_group" else "",
                "building_age_group": key if col == "building_age_group" else "",
                "room_count": key if col == "room_count" else "",
                "site_inside": key if col == "site_inside" else "",
                "detail_effect_total_bucket": key if col == "detail_effect_total_bucket" else "",
                "rows": int(len(g)),
                "r2": _safe_r2(ya, yp),
                "mape": _mape(ya, yp),
                "mean_bias": float(np.nanmean(yp - ya)),
                "actual_mean": float(np.nanmean(ya)),
                "pred_mean": float(np.nanmean(yp)),
                "variance_ratio": (pv / av) if av and av > 0 else np.nan,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def residual_correlations(df: pd.DataFrame, county: str) -> pd.DataFrame:
    sub = df[df["county"].astype(str).eq(county)].copy()
    features = [
        "gross_m2",
        "net_m2",
        "building_age",
        "attr_total_quality_score",
        "detail_effect_total_sum",
        "detail_effect_outside_sum",
        "detail_effect_view_sum",
        "detail_effect_inside_sum",
        "location_baseline_m2",
        "district_target_median",
        "dues",
        "large_home_m2_excess",
        "large_home_log_m2",
        "large_home_net_gross_quality",
        "large_home_room_density",
        "large_home_quality_x_m2",
        "large_home_detail_premium_x_m2",
        "large_home_site_x_m2",
        "large_home_basiskele_premium",
        "basiskele_premium_score",
        "basiskele_detail_total_premium_signal",
        "basiskele_detail_outside_premium_signal",
        "basiskele_detail_view_premium_signal",
        "basiskele_detail_inside_premium_signal",
        "basiskele_has_pool_signal",
        "basiskele_has_view_signal",
        "basiskele_site_premium_signal",
        "is_large_flat",
        "detail_quality_score",
        "detail_selected_count",
    ]
    rows = []
    for feat in features:
        if feat not in sub.columns:
            rows.append(
                {
                    "feature": feat,
                    "pearson_corr_residual": np.nan,
                    "spearman_corr_residual": np.nan,
                    "pearson_corr_log_residual": np.nan,
                    "spearman_corr_log_residual": np.nan,
                    "abs_spearman": np.nan,
                    "missing_rate": 1.0,
                    "present": False,
                }
            )
            continue
        x = _num(sub[feat])
        miss = float(x.isna().mean())
        pair = pd.DataFrame(
            {
                "x": x,
                "r": _num(sub["residual"]),
                "lr": _num(sub["log_residual"]),
            }
        ).dropna()
        if len(pair) < 20:
            pear = spear = pear_l = spear_l = np.nan
        else:
            pear = float(pair["x"].corr(pair["r"], method="pearson"))
            spear = float(pair["x"].corr(pair["r"], method="spearman"))
            pear_l = float(pair["x"].corr(pair["lr"], method="pearson"))
            spear_l = float(pair["x"].corr(pair["lr"], method="spearman"))
        rows.append(
            {
                "feature": feat,
                "pearson_corr_residual": pear,
                "spearman_corr_residual": spear,
                "pearson_corr_log_residual": pear_l,
                "spearman_corr_log_residual": spear_l,
                "abs_spearman": abs(spear) if pd.notna(spear) else np.nan,
                "missing_rate": miss,
                "present": True,
                "n_used": int(len(pair)),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False)


def county_heatmap(df: pd.DataFrame, min_rows: int = 10) -> pd.DataFrame:
    rows = []
    for (county, district, m2g), g in df.groupby(
        [df["county"].astype(str), df["district"].fillna("missing").astype(str), df["m2_group"].fillna("missing").astype(str)],
        dropna=False,
    ):
        if len(g) < min_rows:
            continue
        ya = _num(g["actual_unit_price_gross"]).to_numpy()
        yp = _num(g["pred_ensemble"]).to_numpy()
        av, pv = _var(ya), _var(yp)
        rows.append(
            {
                "county": county,
                "district": district,
                "m2_group": m2g,
                "rows": int(len(g)),
                "r2": _safe_r2(ya, yp),
                "mape": _mape(ya, yp),
                "mean_bias": float(np.nanmean(yp - ya)),
                "variance_ratio": (pv / av) if av and av > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["county", "mape"], ascending=[True, False])


def build_summary(
    df: pd.DataFrame,
    bas_dec: pd.DataFrame,
    bas_seg: pd.DataFrame,
    bas_lh: pd.DataFrame,
    bas_corr: pd.DataFrame,
    kar_dec: pd.DataFrame,
    kar_seg: pd.DataFrame,
    metrics: dict[str, Any],
    proxy_notes: dict[str, Any],
) -> dict[str, Any]:
    # County-level stats
    def county_stats(name: str) -> dict[str, Any]:
        s = df[df["county"].astype(str).eq(name)]
        ya = _num(s["actual_unit_price_gross"]).to_numpy()
        yp = _num(s["pred_ensemble"]).to_numpy()
        av, pv = _var(ya), _var(yp)
        return {
            "rows": int(len(s)),
            "r2": _safe_r2(ya, yp),
            "mape": _mape(ya, yp),
            "variance_ratio": (pv / av) if av and av > 0 else np.nan,
            "actual_std": float(np.nanstd(ya)),
            "pred_std": float(np.nanstd(yp)),
        }

    bas = county_stats("Başiskele")
    kar = county_stats("Karamürsel")

    # Mean-pulling: low variance ratio + cheap decile overpred + rich underpred
    mean_pulling = False
    decile_notes = []
    if not bas_dec.empty and len(bas_dec) >= 8:
        d1 = bas_dec.iloc[0]
        d10 = bas_dec.iloc[-1]
        cheap_over = float(d1["mean_bias"]) > 0
        rich_under = float(d10["mean_bias"]) < 0
        mean_pulling = bool(
            (bas["variance_ratio"] is not None and bas["variance_ratio"] < 0.55)
            and cheap_over
            and rich_under
        )
        decile_notes = [
            {
                "decile": int(d1["decile"]),
                "mean_bias": float(d1["mean_bias"]),
                "bias_pct_of_actual": float(d1.get("bias_pct_of_actual", np.nan)),
                "role": "cheapest",
            },
            {
                "decile": int(d10["decile"]),
                "mean_bias": float(d10["mean_bias"]),
                "bias_pct_of_actual": float(d10.get("bias_pct_of_actual", np.nan)),
                "role": "most_expensive",
            },
        ]

    # Top segments hurting R² (low r2, material rows)
    def top_hurt(seg: pd.DataFrame, n: int = 8) -> list[dict[str, Any]]:
        if seg.empty:
            return []
        s = seg[(seg["rows"] >= 20) & seg["r2"].notna()].copy()
        s = s.sort_values(["r2", "rows"], ascending=[True, False]).head(n)
        return s.to_dict(orient="records")

    # Large home culpability
    lh_all = bas_lh[bas_lh["slice"] == "ALL_BASISKELE_LARGE_HOME"] if not bas_lh.empty else pd.DataFrame()
    lh_r2 = float(lh_all.iloc[0]["r2"]) if not lh_all.empty else float("nan")
    bas_non_lh = df[(df["county"].astype(str).eq("Başiskele")) & (df["is_large_flat"] == 0)]
    non_lh_r2 = _safe_r2(
        _num(bas_non_lh["actual_unit_price_gross"]).to_numpy(),
        _num(bas_non_lh["pred_ensemble"]).to_numpy(),
    )
    lh_share = float((df["county"].astype(str).eq("Başiskele") & (df["is_large_flat"] == 1)).mean()) if len(df) else 0.0
    # among Başiskele only
    bas_mask = df["county"].astype(str).eq("Başiskele")
    lh_share_bas = float((df.loc[bas_mask, "is_large_flat"] == 1).mean()) if bas_mask.any() else 0.0
    large_home_primary = bool(pd.notna(lh_r2) and pd.notna(non_lh_r2) and lh_r2 < non_lh_r2 - 0.08 and lh_share_bas >= 0.15)

    # Karamürsel district concentration: Herfindahl of abs error mass
    kar_df = df[df["county"].astype(str).eq("Karamürsel")].copy()
    sparsity = bool(len(kar_df) < 350)
    district_conc = False
    age_driven = False
    top_kar_districts: list[dict[str, Any]] = []
    if not kar_df.empty and "district" in kar_df.columns:
        kar_df["abs_err"] = _num(kar_df["abs_residual"])
        by_d = (
            kar_df.groupby(kar_df["district"].fillna("missing").astype(str))
            .agg(rows=("abs_err", "size"), mape=("ape", "mean"), mean_abs_err=("abs_err", "mean"), total_abs=("abs_err", "sum"))
            .reset_index()
            .rename(columns={"district": "district"})
        )
        by_d = by_d.sort_values("total_abs", ascending=False)
        total = float(by_d["total_abs"].sum()) or 1.0
        by_d["error_share"] = by_d["total_abs"] / total
        top3_share = float(by_d.head(3)["error_share"].sum())
        district_conc = top3_share >= 0.45 and len(by_d) >= 5
        top_kar_districts = by_d.head(8).to_dict(orient="records")

    # age signal from correlations / segment r2
    if not kar_seg.empty:
        age_rows = kar_seg[kar_seg["segment_col"] == "building_age_group"]
        if not age_rows.empty:
            age_driven = bool(age_rows["r2"].min() < 0.35 or age_rows["mape"].max() > 0.20)

    # Correlations top
    top_corr = []
    if not bas_corr.empty:
        top_corr = (
            bas_corr[bas_corr["present"] & bas_corr["abs_spearman"].notna()]
            .head(10)[["feature", "spearman_corr_residual", "spearman_corr_log_residual", "abs_spearman"]]
            .to_dict(orient="records")
        )

    # Recommended experiments
    experiments = []
    if mean_pulling:
        experiments.append(
            {
                "id": "V16-E1",
                "name": "basiskele_spread_residual_layer",
                "why": "Mean-pulling confirmed (var_ratio<0.55 + cheap over / rich under). Need OOF-safe delta that expands prediction variance without raising MAPE.",
                "guardrails": ["global_mape<=0.131", "basiskele_mape<=0.115", "izmit_r2>=0.70"],
                "success": "basiskele_r2>=0.50 and variance_ratio>=0.55",
            }
        )
    if large_home_primary or (pd.notna(lh_r2) and lh_r2 < 0.35):
        experiments.append(
            {
                "id": "V16-E2",
                "name": "basiskele_large_home_regime",
                "why": f"Başiskele large_home R2={lh_r2:.3f} vs non-large R2={non_lh_r2:.3f}; share={lh_share_bas:.1%}. Separate simple regime better than more global features.",
                "guardrails": ["no global mape worsen >0.002", "non-large basiskele r2 must not drop >0.01"],
                "success": "basiskele_large_home_r2 lift >=0.08",
            }
        )
    experiments.append(
        {
            "id": "V16-E3",
            "name": "karamursel_location_baseline_strengthen" if not sparsity else "karamursel_data_plus_strong_baseline",
            "why": (
                f"Karamürsel n={len(kar_df)}, MAPE high; "
                + ("district error concentrated. " if district_conc else "error fairly diffuse. ")
                + ("building_age segments weak. " if age_driven else "")
                + "Prefer stronger fold-safe location/age baseline over aggressive county expert."
            ),
            "guardrails": ["karamursel_mape<=0.17", "global_r2>=0.675"],
            "success": "karamursel_r2>=0.60",
        }
    )
    # keep exactly top 3 primary recommendations
    experiments = experiments[:3]

    # Answers block
    answers = {
        "q1_basiskele_segments_hurting_r2": top_hurt(bas_seg, 8),
        "q2_basiskele_decile_bias": decile_notes,
        "q3_basiskele_large_home_primary_culprit": {
            "yes": large_home_primary,
            "large_home_r2": lh_r2,
            "non_large_home_r2": non_lh_r2,
            "large_home_share_in_basiskele": lh_share_bas,
            "large_home_rows": int(lh_all.iloc[0]["rows"]) if not lh_all.empty else 0,
        },
        "q4_karamursel_error_concentrated_in_districts": {
            "yes": district_conc,
            "top_districts_by_abs_error_mass": top_kar_districts[:5],
        },
        "q5_karamursel_root_cause": {
            "data_sparsity": sparsity,
            "district_concentration": district_conc,
            "building_age_weak_segments": age_driven,
            "primary": (
                "data_sparsity"
                if sparsity and not district_conc
                else "district_hotspots_plus_sparsity"
                if sparsity and district_conc
                else "building_age"
                if age_driven
                else "mixed"
            ),
            "rows": int(len(kar_df)),
        },
        "q6_top3_v16_experiments": experiments,
    }

    return {
        "source_metrics_global": {
            "r2": (metrics.get("ensemble") or {}).get("r2"),
            "mape": (metrics.get("ensemble") or {}).get("mape"),
            "overall": (metrics.get("decision") or {}).get("overall"),
            "ship_ready": (metrics.get("decision") or {}).get("ship_ready_all_counties_r2_ge_0_65"),
        },
        "basiskele": bas,
        "karamursel": kar,
        "basiskele_mean_pulling_confirmed": mean_pulling,
        "karamursel_data_sparsity_confirmed": sparsity,
        "top_basiskele_error_segments": top_hurt(bas_seg, 10),
        "top_karamursel_error_segments": top_hurt(kar_seg, 10),
        "top_basiskele_residual_correlations": top_corr,
        "recommended_v16_experiments": experiments,
        "answers": answers,
        "proxy_notes": proxy_notes,
        "caveats": [
            "detail_effect_* here are binary coverage proxies, not V15 fold-safe residual effects.",
            "location_baseline_m2 / district_target_median are OOF actual group medians for diagnosis only.",
            "No model was trained or modified in this sprint.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V16 Phase-0 county error cluster diagnostics (OOF only).")
    ap.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    ap.add_argument("--sales", type=Path, default=DEFAULT_SALES, help="Optional; used only for metadata if present.")
    ap.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.oof.exists():
        raise FileNotFoundError(f"OOF not found: {args.oof}")

    print(f"Loading OOF: {args.oof}")
    oof = pd.read_csv(args.oof)
    if "pred_ensemble" not in oof.columns or "actual_unit_price_gross" not in oof.columns:
        raise ValueError("OOF must contain pred_ensemble and actual_unit_price_gross")
    if "county" not in oof.columns:
        raise ValueError("OOF must contain county")

    metrics: dict[str, Any] = {}
    if args.metrics.exists():
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
        print(f"Loaded metrics: {args.metrics}")
    else:
        print(f"WARNING: metrics not found at {args.metrics}")

    if args.sales.exists():
        print(f"Sales file present (not required for join): {args.sales}")
    else:
        print(f"WARNING: sales not found at {args.sales} (continuing with OOF only)")

    print("Enriching diagnostic proxies...")
    df, proxy_notes = enrich_frame(oof)

    print("Building Başiskele tables...")
    bas_dec = decile_bias_table(df, "Başiskele")
    bas_seg = segment_error_table(
        df,
        "Başiskele",
        [
            "m2_group",
            "building_age_group",
            "room_count",
            "district",
            "is_large_flat",
            "site_inside",
            "attr_total_quality_bucket",
            "detail_effect_total_bucket",
            "basiskele_premium_bucket",
        ],
    )
    bas_lh = large_home_error_table(df)
    bas_corr = residual_correlations(df, "Başiskele")

    print("Building Karamürsel tables...")
    kar_dec = decile_bias_table(df, "Karamürsel")
    kar_seg = segment_error_table(
        df,
        "Karamürsel",
        [
            "district",
            "m2_group",
            "building_age_group",
            "room_count",
            "site_inside",
            "heating",
            "is_large_flat",
            "detail_effect_total_bucket",
        ],
        min_rows=5,
    )

    print("Building county heatmap...")
    heat = county_heatmap(df)

    summary = build_summary(df, bas_dec, bas_seg, bas_lh, bas_corr, kar_dec, kar_seg, metrics, proxy_notes)

    # Write outputs
    bas_dec.to_csv(out_dir / "basiskele_decile_bias_v16.csv", index=False, encoding="utf-8-sig")
    bas_seg.to_csv(out_dir / "basiskele_error_by_segment_v16.csv", index=False, encoding="utf-8-sig")
    bas_lh.to_csv(out_dir / "basiskele_large_home_error_v16.csv", index=False, encoding="utf-8-sig")
    # user-requested residual correlation schema (+ log residual extras kept)
    corr_out = bas_corr.rename(
        columns={
            "spearman_corr_residual": "spearman_corr",
            "pearson_corr_residual": "pearson_corr",
        }
    )
    keep = [
        "feature",
        "pearson_corr",
        "spearman_corr",
        "abs_spearman",
        "missing_rate",
        "pearson_corr_log_residual",
        "spearman_corr_log_residual",
        "n_used",
        "present",
    ]
    corr_out[[c for c in keep if c in corr_out.columns]].to_csv(
        out_dir / "basiskele_residual_correlations_v16.csv", index=False, encoding="utf-8-sig"
    )
    kar_dec.to_csv(out_dir / "karamursel_decile_bias_v16.csv", index=False, encoding="utf-8-sig")
    kar_seg.to_csv(out_dir / "karamursel_error_by_segment_v16.csv", index=False, encoding="utf-8-sig")
    heat.to_csv(out_dir / "county_error_heatmap_v16.csv", index=False, encoding="utf-8-sig")
    (out_dir / "v16_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )

    # Human-readable answers
    a = summary["answers"]
    print("\n========== V16 DIAGNOSTIC ANSWERS ==========")
    print(f"1) Başiskele mean-pulling confirmed: {summary['basiskele_mean_pulling_confirmed']}")
    print(f"   Başiskele R2={summary['basiskele']['r2']:.4f} MAPE={summary['basiskele']['mape']:.4f} var_ratio={summary['basiskele']['variance_ratio']:.4f}")
    print("   Worst segments (low R2):")
    for row in a["q1_basiskele_segments_hurting_r2"][:5]:
        print(f"   - {row['segment_col']}={row['segment_value']}: rows={row['rows']} r2={row['r2']:.3f} mape={row['mape']:.3f}")
    print("2) Decile bias:")
    for d in a["q2_basiskele_decile_bias"]:
        print(f"   - {d}")
    q3 = a["q3_basiskele_large_home_primary_culprit"]
    print(f"3) Large_home primary culprit? {q3['yes']} (lh_r2={q3['large_home_r2']:.3f}, non_lh_r2={q3['non_large_home_r2']:.3f}, share={q3['large_home_share_in_basiskele']:.1%})")
    q4 = a["q4_karamursel_error_concentrated_in_districts"]
    print(f"4) Karamürsel district-concentrated? {q4['yes']}")
    q5 = a["q5_karamursel_root_cause"]
    print(f"5) Karamürsel primary={q5['primary']} (n={q5['rows']}, sparsity={q5['data_sparsity']}, age_weak={q5['building_age_weak_segments']})")
    print("6) Top 3 V16 experiments:")
    for e in a["q6_top3_v16_experiments"]:
        print(f"   - {e['id']} {e['name']}: {e['why'][:120]}...")
    print(f"\nWrote reports to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
