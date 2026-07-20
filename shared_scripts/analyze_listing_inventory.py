#!/usr/bin/env python
"""Standalone Listing inventory analysis (no model training).

Examples:
  python shared_scripts/analyze_listing_inventory.py --city Kocaeli
  python shared_scripts/analyze_listing_inventory.py --city Kocaeli --county Başiskele
  python shared_scripts/analyze_listing_inventory.py --city Kocaeli --purpose sale
  python shared_scripts/analyze_listing_inventory.py --city Kocaeli --export-samples
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from db_utils import (  # noqa: E402
    RENTAL_PRICE_CANDIDATES,
    RENTAL_UNIT_PRICE_CANDIDATES,
    SALE_PRICE_CANDIDATES,
    SALE_UNIT_PRICE_CANDIDATES,
    create_engine,
    fetch_listings,
    get_database_url,
    resolve_first_present_column,
)
from env_loader import find_project_root, load_root_env  # noqa: E402

LARGE_HOME_M2 = 180.0
CATEGORICAL_COLS = [
    "county",
    "district",
    "room_count",
    "heating",
    "kitchen",
    "balcony",
    "elevator",
    "parking",
    "furnished",
    "usage_status",
    "site_inside",
    "credit_eligible",
    "deed_status",
    "seller_type",
    "location_precision",
    "location_source",
    "location_backfill_status",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    root = find_project_root(HERE)
    if root is None:
        raise RuntimeError("Could not locate project root (MANIFEST.json + data/).")
    return root


def _ts_folder() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _safe_series(df: pd.DataFrame, *names: str) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([np.nan] * len(df), index=df.index)


def _num(df: pd.DataFrame, *names: str) -> pd.Series:
    return pd.to_numeric(_safe_series(df, *names), errors="coerce")


def _normalize_frame(df: pd.DataFrame, purpose: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize columns; resolve price aliases. Returns (frame, price_meta)."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    # unify lat/lon
    if "lat" not in out.columns and "latitude" in out.columns:
        out["lat"] = pd.to_numeric(out["latitude"], errors="coerce")
    elif "lat" in out.columns:
        out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    else:
        out["lat"] = np.nan

    if "lon" not in out.columns and "longitude" in out.columns:
        out["lon"] = pd.to_numeric(out["longitude"], errors="coerce")
    elif "lon" in out.columns:
        out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    else:
        out["lon"] = np.nan

    out["listing_purpose"] = purpose
    out["gross_m2"] = _num(out, "gross_m2")
    out["net_m2"] = _num(out, "net_m2")
    out["building_age"] = _num(out, "building_age")

    price_meta: dict[str, Any] = {
        "purpose": purpose,
        "price_column_used": None,
        "unit_price_column_used": None,
        "price_derived_from_m2": False,
        "price_available": False,
    }

    if purpose == "sale":
        price_col = resolve_first_present_column(out, SALE_PRICE_CANDIDATES)
        unit_col = resolve_first_present_column(out, SALE_UNIT_PRICE_CANDIDATES)
    else:
        price_col = resolve_first_present_column(out, RENTAL_PRICE_CANDIDATES)
        unit_col = resolve_first_present_column(out, RENTAL_UNIT_PRICE_CANDIDATES)

    price_meta["price_column_used"] = price_col
    price_meta["unit_price_column_used"] = unit_col

    if price_col is not None:
        out["price"] = pd.to_numeric(out[price_col], errors="coerce")
        price_meta["price_available"] = True
    else:
        out["price"] = np.nan
        price_meta["price_available"] = False

    if unit_col is not None:
        unit_vals = pd.to_numeric(out[unit_col], errors="coerce")
    else:
        unit_vals = pd.Series(np.nan, index=out.index, dtype=float)

    if purpose == "sale":
        out["unit_price_gross"] = unit_vals
        need = out["unit_price_gross"].isna() & out["price"].notna() & (out["gross_m2"] > 0)
        out.loc[need, "unit_price_gross"] = out.loc[need, "price"] / out.loc[need, "gross_m2"]
        if bool(need.any()):
            price_meta["price_derived_from_m2"] = True
        out["rent_m2"] = np.nan
    else:
        # rental unit price = rent per m2
        out["rent_m2"] = unit_vals
        need = out["rent_m2"].isna() & out["price"].notna() & (out["gross_m2"] > 0)
        out.loc[need, "rent_m2"] = out.loc[need, "price"] / out.loc[need, "gross_m2"]
        if bool(need.any()):
            price_meta["price_derived_from_m2"] = True
        # keep unit_price_gross empty for rentals (sale metric)
        out["unit_price_gross"] = np.nan

    out["has_lat_lon"] = out["lat"].notna() & out["lon"].notna()
    # invalid coords
    out["coord_invalid"] = False
    ok = out["has_lat_lon"]
    out.loc[ok, "coord_invalid"] = ~(
        out.loc[ok, "lat"].between(-90, 90) & out.loc[ok, "lon"].between(-180, 180)
    )
    out["is_large_home"] = (out["gross_m2"] >= LARGE_HOME_M2).fillna(False).astype(bool)

    for c in ("city", "county", "district", "classified_id", "source_url"):
        if c not in out.columns:
            out[c] = np.nan
        else:
            out[c] = out[c].astype("string")

    if "location_precision" not in out.columns:
        out["location_precision"] = pd.NA
    if "location_source" not in out.columns:
        out["location_source"] = pd.NA
    if "location_backfill_status" not in out.columns:
        out["location_backfill_status"] = pd.NA

    # exact map heuristic
    prec = out["location_precision"].astype("string").str.lower()
    out["is_exact_map"] = prec.isin(["exact", "exact_map", "map", "pin", "precise"]) | (
        out["has_lat_lon"] & prec.isna()
    )
    # if precision explicitly district_only / missing → not exact
    out.loc[prec.isin(["district_only", "district", "missing", "approx", "approximate"]), "is_exact_map"] = False
    out.loc[~out["has_lat_lon"], "is_exact_map"] = False

    return out, price_meta


def _basic_filter_mask(
    df: pd.DataFrame,
    purpose: str,
    *,
    min_sale_unit_price: float,
    max_sale_unit_price: float,
    min_rent_m2: float,
    max_rent_m2: float,
    price_available: bool = True,
) -> pd.Series | None:
    """Return filter mask, or None when price column is unavailable (skip numeric filter)."""
    if not price_available:
        return None
    m = pd.Series(True, index=df.index)
    m &= df["price"].notna() & (df["price"] > 0)
    m &= df["gross_m2"].notna() & (df["gross_m2"] > 20) & (df["gross_m2"] < 600)
    if purpose == "sale":
        m &= df["unit_price_gross"].notna()
        m &= df["unit_price_gross"].between(min_sale_unit_price, max_sale_unit_price)
    else:
        m &= df["rent_m2"].notna()
        m &= df["rent_m2"].between(min_rent_m2, max_rent_m2)
    return m.fillna(False)


def _coverage(s: pd.Series) -> float:
    if len(s) == 0:
        return float("nan")
    return float(s.fillna(False).mean())


def _pct(series: pd.Series, q: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float(np.nanpercentile(s.to_numpy(dtype=float), q))


def _iqr(series: pd.Series) -> float:
    return _pct(series, 75) - _pct(series, 25)


def _dist_stats(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "min": np.nan,
            "max": np.nan,
            "std": np.nan,
            "iqr": np.nan,
            "cv": np.nan,
            "outlier_low_count": 0,
            "outlier_high_count": 0,
        }
    q1, q3 = float(np.nanpercentile(s, 25)), float(np.nanpercentile(s, 75))
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    return {
        "count": int(len(s)),
        "mean": mean,
        "median": float(s.median()),
        "p10": float(np.nanpercentile(s, 10)),
        "p25": q1,
        "p75": q3,
        "p90": float(np.nanpercentile(s, 90)),
        "min": float(s.min()),
        "max": float(s.max()),
        "std": std,
        "iqr": float(iqr),
        "cv": float(std / mean) if mean and not math.isclose(mean, 0.0) else np.nan,
        "outlier_low_count": int((s < low).sum()),
        "outlier_high_count": int((s > high).sum()),
    }


def _county_centroid_distance(df: pd.DataFrame) -> pd.Series:
    """Approx haversine distance (m) to county mean lat/lon; NaN if unavailable."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if df.empty or "county" not in df.columns:
        return out
    for county, g in df.groupby("county", dropna=False):
        mask = g["has_lat_lon"].fillna(False) & ~g["coord_invalid"].fillna(False)
        sub = g.loc[mask]
        if len(sub) < 3:
            continue
        clat, clon = float(sub["lat"].mean()), float(sub["lon"].mean())
        lat = sub["lat"].to_numpy(dtype=float)
        lon = sub["lon"].to_numpy(dtype=float)
        # haversine meters
        r = 6371000.0
        p1, p2 = np.radians(lat), np.radians(clat)
        dphi = np.radians(clat - lat)
        dlmb = np.radians(clon - lon)
        a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
        dist = 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        out.loc[sub.index] = dist
    return out


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def build_county_distribution(sale: pd.DataFrame, rental: pd.DataFrame) -> pd.DataFrame:
    counties = sorted(
        set(sale["county"].dropna().astype(str)) | set(rental["county"].dropna().astype(str))
    )
    rows = []
    sale_n = max(len(sale), 1)
    rent_n = max(len(rental), 1)
    for county in counties:
        s = sale[sale["county"].astype(str) == county]
        r = rental[rental["county"].astype(str) == county]
        city = (
            str(s["city"].dropna().iloc[0])
            if len(s) and s["city"].notna().any()
            else (str(r["city"].dropna().iloc[0]) if len(r) and r["city"].notna().any() else "")
        )
        warn = []
        if len(s) < 200:
            warn.append("low_sale_count")
        if _coverage(s["has_lat_lon"]) < 0.4 if len(s) else True:
            warn.append("low_sale_coord_coverage")
        rows.append(
            {
                "city": city,
                "county": county,
                "sale_count": int(len(s)),
                "rental_count": int(len(r)),
                "total_count": int(len(s) + len(r)),
                "sale_share": float(len(s) / sale_n) if len(sale) else 0.0,
                "rental_share": float(len(r) / rent_n) if len(rental) else 0.0,
                "sale_coord_coverage": _coverage(s["has_lat_lon"]) if len(s) else np.nan,
                "rental_coord_coverage": _coverage(r["has_lat_lon"]) if len(r) else np.nan,
                "sale_exact_map_coverage": _coverage(s["is_exact_map"]) if len(s) else np.nan,
                "rental_exact_map_coverage": _coverage(r["is_exact_map"]) if len(r) else np.nan,
                "median_sale_unit_price": float(s["unit_price_gross"].median()) if len(s) else np.nan,
                "median_rent_m2": float(r["rent_m2"].median()) if len(r) else np.nan,
                "median_gross_m2_sale": float(s["gross_m2"].median()) if len(s) else np.nan,
                "median_gross_m2_rental": float(r["gross_m2"].median()) if len(r) else np.nan,
                "district_count": int(
                    pd.concat([s["district"], r["district"]], ignore_index=True).dropna().nunique()
                ),
                "warning": "|".join(warn),
            }
        )
    return pd.DataFrame(rows).sort_values("total_count", ascending=False)


def build_district_distribution(sale: pd.DataFrame, rental: pd.DataFrame) -> pd.DataFrame:
    keys = set()
    for df in (sale, rental):
        for _, row in df[["city", "county", "district"]].drop_duplicates().iterrows():
            keys.add((str(row.get("city") or ""), str(row.get("county") or ""), str(row.get("district") or "")))
    rows = []
    for city, county, district in sorted(keys):
        s = sale[
            (sale["city"].astype(str) == city)
            & (sale["county"].astype(str) == county)
            & (sale["district"].astype(str) == district)
        ]
        r = rental[
            (rental["city"].astype(str) == city)
            & (rental["county"].astype(str) == county)
            & (rental["district"].astype(str) == district)
        ]
        warn = []
        if len(s) < 15:
            warn.append("sparse_sale")
        if len(s) and _coverage(s["has_lat_lon"]) < 0.4:
            warn.append("low_coord")
        rows.append(
            {
                "city": city,
                "county": county,
                "district": district,
                "sale_count": int(len(s)),
                "rental_count": int(len(r)),
                "total_count": int(len(s) + len(r)),
                "sale_coord_coverage": _coverage(s["has_lat_lon"]) if len(s) else np.nan,
                "rental_coord_coverage": _coverage(r["has_lat_lon"]) if len(r) else np.nan,
                "median_sale_unit_price": float(s["unit_price_gross"].median()) if len(s) else np.nan,
                "median_rent_m2": float(r["rent_m2"].median()) if len(r) else np.nan,
                "median_gross_m2_sale": float(s["gross_m2"].median()) if len(s) else np.nan,
                "median_building_age_sale": float(s["building_age"].median()) if len(s) else np.nan,
                "large_home_count": int(s["is_large_home"].sum()) if len(s) else 0,
                "large_home_share": float(s["is_large_home"].mean()) if len(s) else np.nan,
                "warning": "|".join(warn),
            }
        )
    return pd.DataFrame(rows).sort_values("total_count", ascending=False)


def build_price_distribution(df: pd.DataFrame, value_col: str, level: str) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    if level == "city":
        groups = df.groupby(["city"], dropna=False)
    elif level == "county":
        groups = df.groupby(["city", "county"], dropna=False)
    else:
        groups = df.groupby(["city", "county", "district"], dropna=False)
    for key, g in groups:
        if not isinstance(key, tuple):
            key = (key,)
        stats = _dist_stats(g[value_col])
        rec: dict[str, Any] = {"level": level}
        names = ["city", "county", "district"][: len(key)]
        for n, v in zip(names, key):
            rec[n] = v
        for n in ["city", "county", "district"]:
            rec.setdefault(n, "")
        rec.update(stats)
        rows.append(rec)
    return pd.DataFrame(rows)


def build_feature_missingness(sale: pd.DataFrame, rental: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for purpose, df in (("sale", sale), ("rental", rental), ("both", pd.concat([sale, rental], ignore_index=True))):
        if df.empty:
            continue
        for col in df.columns:
            miss = int(df[col].isna().sum())
            n = len(df)
            frames.append(
                {
                    "column": col,
                    "missing_count": miss,
                    "missing_rate": float(miss / n) if n else np.nan,
                    "available_count": int(n - miss),
                    "purpose": purpose,
                }
            )
    return pd.DataFrame(frames)


def build_categorical_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        s = df[col].astype("string")
        vc = s.value_counts(dropna=True)
        if vc.empty:
            rows.append(
                {
                    "column": col,
                    "unique_count": 0,
                    "top_value": "",
                    "top_value_count": 0,
                    "top_value_share": np.nan,
                    "rare_value_count": 0,
                }
            )
            continue
        top_v, top_c = str(vc.index[0]), int(vc.iloc[0])
        rare = int((vc <= max(2, int(0.005 * len(s)))).sum())
        rows.append(
            {
                "column": col,
                "unique_count": int(s.nunique(dropna=True)),
                "top_value": top_v,
                "top_value_count": top_c,
                "top_value_share": float(top_c / len(s)) if len(s) else np.nan,
                "rare_value_count": rare,
            }
        )
    return pd.DataFrame(rows)


def build_location_quality(sale: pd.DataFrame, rental: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for purpose, df in (("sale", sale), ("rental", rental)):
        if df.empty:
            continue
        dist = _county_centroid_distance(df)
        df = df.copy()
        df["_dist_centroid"] = dist
        for (city, county, district), g in df.groupby(["city", "county", "district"], dropna=False):
            prec = g["location_precision"].astype("string").str.lower()
            src = g["location_source"].astype("string").str.lower()
            back = g["location_backfill_status"].astype("string").str.lower()
            rows.append(
                {
                    "purpose": purpose,
                    "city": city,
                    "county": county,
                    "district": district,
                    "rows": int(len(g)),
                    "has_lat_lon_count": int(g["has_lat_lon"].sum()),
                    "has_lat_lon_rate": _coverage(g["has_lat_lon"]),
                    "exact_map_count": int(g["is_exact_map"].sum()),
                    "exact_map_rate": _coverage(g["is_exact_map"]),
                    "approx_count": int(prec.isin(["approx", "approximate"]).sum()),
                    "district_only_count": int(prec.isin(["district_only", "district"]).sum()),
                    "missing_count": int((prec.isin(["missing", ""]) | prec.isna()).sum())
                    if len(g)
                    else 0,
                    "listing_removed_count": int(back.str.contains("removed", na=False).sum()),
                    "data_attr_map_count": int(src.str.contains("data_attr|map", na=False, regex=True).sum()),
                    "median_distance_to_county_centroid": float(g["_dist_centroid"].median())
                    if g["_dist_centroid"].notna().any()
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_model_readiness(
    sale: pd.DataFrame,
    rental: pd.DataFrame,
    *,
    location_coverage_min: float,
) -> pd.DataFrame:
    counties = sorted(
        set(sale["county"].dropna().astype(str)) | set(rental["county"].dropna().astype(str))
    )
    rows = []
    for county in counties:
        s = sale[sale["county"].astype(str) == county]
        r = rental[rental["county"].astype(str) == county]
        coord = _coverage(s["has_lat_lon"]) if len(s) else 0.0
        dist_counts = s.groupby(s["district"].astype(str)).size() if len(s) else pd.Series(dtype=int)
        reasons = []
        if len(s) < 400:
            reasons.append("low_sale_count")
        if len(r) < 150:
            reasons.append("low_rental_count")
        if not (isinstance(coord, float) and coord >= location_coverage_min):
            reasons.append("low_location_coverage")
        if len(dist_counts) and float(dist_counts.median()) < 10:
            reasons.append("sparse_districts")
        stats = _dist_stats(s["unit_price_gross"]) if len(s) else {"iqr": np.nan, "outlier_high_count": 0, "count": 0}
        if stats["count"] and (stats["outlier_high_count"] + stats.get("outlier_low_count", 0)) / stats["count"] > 0.08:
            reasons.append("high_outlier_rate")

        if len(s) >= 800 and len(r) >= 300 and coord >= 0.65:
            status = "GOOD"
        elif len(s) >= 400 and len(r) >= 150 and coord >= 0.4:
            status = "OK"
        else:
            status = "WEAK"

        rows.append(
            {
                "county": county,
                "sale_count": int(len(s)),
                "rental_count": int(len(r)),
                "coord_coverage": float(coord) if coord == coord else np.nan,
                "district_count": int(s["district"].nunique()) if len(s) else 0,
                "min_district_sale_count": int(dist_counts.min()) if len(dist_counts) else 0,
                "median_district_sale_count": float(dist_counts.median()) if len(dist_counts) else np.nan,
                "large_home_count": int(s["is_large_home"].sum()) if len(s) else 0,
                "large_home_share": float(s["is_large_home"].mean()) if len(s) else np.nan,
                "sale_price_iqr": stats["iqr"],
                "readiness_status": status,
                "reason": "|".join(reasons),
            }
        )
    return pd.DataFrame(rows).sort_values(["readiness_status", "sale_count"])


def build_basiskele_special(sale: pd.DataFrame, rental: pd.DataFrame) -> pd.DataFrame:
    s = sale[sale["county"].astype(str) == "Başiskele"]
    r = rental[rental["county"].astype(str) == "Başiskele"]
    districts = sorted(
        set(s["district"].dropna().astype(str)) | set(r["district"].dropna().astype(str))
    )
    rows = []
    for d in districts:
        sd = s[s["district"].astype(str) == d]
        rd = r[r["district"].astype(str) == d]
        p10 = _pct(sd["unit_price_gross"], 10)
        p90 = _pct(sd["unit_price_gross"], 90)
        spread = (p90 / p10) if p10 and p10 == p10 and p10 > 0 else np.nan
        lh = sd[sd["is_large_home"]]
        nl = sd[~sd["is_large_home"]]
        warn = []
        if len(sd) < 20:
            warn.append("sparse")
        if spread == spread and spread > 2.5:
            warn.append("wide_price_spread")
        if len(sd) and _coverage(sd["has_lat_lon"]) < 0.5:
            warn.append("low_coord")
        rows.append(
            {
                "district": d,
                "sale_count": int(len(sd)),
                "rental_count": int(len(rd)),
                "coord_coverage": _coverage(sd["has_lat_lon"]) if len(sd) else np.nan,
                "median_unit_price": float(sd["unit_price_gross"].median()) if len(sd) else np.nan,
                "p10_unit_price": p10,
                "p90_unit_price": p90,
                "large_home_count": int(len(lh)),
                "large_home_share": float(len(lh) / len(sd)) if len(sd) else np.nan,
                "large_home_median_unit_price": float(lh["unit_price_gross"].median()) if len(lh) else np.nan,
                "non_large_median_unit_price": float(nl["unit_price_gross"].median()) if len(nl) else np.nan,
                "price_spread_ratio": float(spread) if spread == spread else np.nan,
                "warning": "|".join(warn),
            }
        )
    return pd.DataFrame(rows).sort_values("sale_count", ascending=False)


def build_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for key in ("classified_id", "source_url"):
        if key not in df.columns:
            continue
        vc = df[key].astype("string").value_counts(dropna=True)
        dups = vc[vc > 1]
        for val, cnt in dups.items():
            sub = df[df[key].astype("string") == val]
            rows.append(
                {
                    "duplicate_key": key,
                    "key_value": str(val)[:300],
                    "count": int(cnt),
                    "example_ids": "|".join(sub["classified_id"].astype(str).head(5).tolist()),
                    "purposes": "|".join(sorted(sub["listing_purpose"].dropna().astype(str).unique())),
                    "counties": "|".join(sorted(sub["county"].dropna().astype(str).unique())[:8]),
                    "districts": "|".join(sorted(sub["district"].dropna().astype(str).unique())[:8]),
                }
            )
    return pd.DataFrame(rows)


def build_suspicious(
    sale: pd.DataFrame,
    rental: pd.DataFrame,
    *,
    min_sale_unit_price: float,
    max_sale_unit_price: float,
    min_rent_m2: float,
    max_rent_m2: float,
    sale_price_available: bool = True,
    rental_price_available: bool = True,
) -> pd.DataFrame:
    rows = []

    def add(df: pd.DataFrame, purpose: str, mask: pd.Series, reason: str) -> None:
        sub = df.loc[mask.fillna(False)]
        for _, r in sub.iterrows():
            rows.append(
                {
                    "reason": reason,
                    "classified_id": r.get("classified_id"),
                    "source_url": r.get("source_url"),
                    "purpose": purpose,
                    "city": r.get("city"),
                    "county": r.get("county"),
                    "district": r.get("district"),
                    "price": r.get("price"),
                    "unit_price_gross": r.get("unit_price_gross"),
                    "gross_m2": r.get("gross_m2"),
                    "room_count": r.get("room_count"),
                    "lat": r.get("lat"),
                    "lon": r.get("lon"),
                }
            )

    for purpose, df, price_ok, price_meta in (
        ("sale", sale, sale_price_available, {"has_total_price": sale_price_available}),
        ("rental", rental, rental_price_available, {"has_total_price": rental_price_available}),
    ):
        if df.empty:
            continue
        # Flood-guard: only flag row-level missing_price when a total-price column exists.
        has_total = bool(df["price"].notna().any()) if "price" in df.columns else False
        if price_ok and has_total:
            add(df, purpose, df["price"].isna(), "missing_price")
            add(df, purpose, df["price"].notna() & (df["price"] <= 0), "price_le_0")
        add(df, purpose, df["gross_m2"].isna(), "missing_gross_m2")
        add(df, purpose, df["gross_m2"].notna() & (df["gross_m2"] <= 20), "gross_m2_le_20")
        add(df, purpose, df["gross_m2"].notna() & (df["gross_m2"] >= 600), "gross_m2_ge_600")
        add(df, purpose, df["county"].isna() | (df["county"].astype(str).str.len() == 0), "county_missing")
        add(df, purpose, df["district"].isna() | (df["district"].astype(str).str.len() == 0), "district_missing")
        add(df, purpose, df["coord_invalid"].fillna(False), "lat_lon_invalid")
        if purpose == "sale" and price_ok:
            add(df, purpose, df["unit_price_gross"] < min_sale_unit_price, "unit_price_gross_lt_min")
            add(df, purpose, df["unit_price_gross"] > max_sale_unit_price, "unit_price_gross_gt_max")
        elif purpose == "rental" and price_ok:
            add(df, purpose, df["rent_m2"].notna() & (df["rent_m2"] < min_rent_m2), "rent_m2_lt_min")
            add(df, purpose, df["rent_m2"].notna() & (df["rent_m2"] > max_rent_m2), "rent_m2_gt_max")
    return pd.DataFrame(rows)


def collect_warnings(
    summary: dict[str, Any],
    readiness: pd.DataFrame,
    county_df: pd.DataFrame,
) -> list[str]:
    warns: list[str] = []
    if summary.get("sale_rows_raw", 0) == 0 and summary.get("rental_rows_raw", 0) == 0:
        warns.append("no_rows_fetched")
    if summary.get("duplicate_classified_id_count", 0):
        warns.append(f"duplicate_classified_id={summary['duplicate_classified_id_count']}")
    if summary.get("missing_location_count", 0):
        warns.append(f"missing_location={summary['missing_location_count']}")
    if summary.get("coordinate_coverage_sale") == summary.get("coordinate_coverage_sale"):
        cov = summary.get("coordinate_coverage_sale")
        if isinstance(cov, (int, float)) and cov < summary.get("location_coverage_min", 0.4):
            warns.append(f"sale_coord_coverage_below_min={cov:.3f}")
    if not readiness.empty:
        weak = readiness[readiness["readiness_status"] == "WEAK"]["county"].astype(str).tolist()
        if weak:
            warns.append("weak_readiness_counties=" + ",".join(weak[:8]))
    if not county_df.empty:
        low = county_df[county_df["sale_count"] < 200]["county"].astype(str).tolist()
        if low:
            warns.append("low_sale_counties=" + ",".join(low[:8]))
    if summary.get("missing_columns"):
        warns.append("missing_db_columns=" + ",".join(summary["missing_columns"][:12]))
    return warns


def write_summary_md(
    path: Path,
    *,
    summary: dict[str, Any],
    county_df: pd.DataFrame,
    readiness: pd.DataFrame,
    basiskele: pd.DataFrame,
    warnings_list: list[str],
) -> None:
    lines = [
        "# Listing Inventory Summary",
        "",
        f"- run_timestamp: `{summary.get('run_timestamp')}`",
        f"- filters: city=`{summary.get('filters', {}).get('city')}`, "
        f"county=`{summary.get('filters', {}).get('county')}`, "
        f"purpose=`{summary.get('filters', {}).get('purpose')}`",
        "",
        "## Counts",
        f"- sale raw: **{summary.get('sale_rows_raw', 0)}**",
        f"- rental raw: **{summary.get('rental_rows_raw', 0)}**",
        f"- sale after basic filter: **{summary.get('sale_rows_after_basic_filter', 0)}** "
        f"({summary.get('sale_basic_filter_status')})",
        f"- rental after basic filter: **{summary.get('rental_rows_after_basic_filter', 0)}** "
        f"({summary.get('rental_basic_filter_status')})",
        f"- price columns: sale=`{summary.get('price_column_used', {}).get('sale')}`, "
        f"rental=`{summary.get('price_column_used', {}).get('rental')}`",
        f"- unit price columns: sale=`{summary.get('unit_price_column_used', {}).get('sale')}`, "
        f"rental=`{summary.get('unit_price_column_used', {}).get('rental')}`",
        "",
        "## Location coverage",
        f"- sale coord: `{summary.get('coordinate_coverage_sale')}`",
        f"- rental coord: `{summary.get('coordinate_coverage_rental')}`",
        f"- sale exact_map: `{summary.get('exact_map_coverage_sale')}`",
        f"- rental exact_map: `{summary.get('exact_map_coverage_rental')}`",
        "",
        "## County distribution (top)",
    ]
    if county_df.empty:
        lines.append("- (empty)")
    else:
        for _, r in county_df.head(12).iterrows():
            lines.append(
                f"- {r['county']}: sale={int(r['sale_count'])}, rental={int(r['rental_count'])}, "
                f"sale_coord={r['sale_coord_coverage']}"
            )
    lines += ["", "## Model readiness"]
    if readiness.empty:
        lines.append("- (empty)")
    else:
        for _, r in readiness.iterrows():
            lines.append(
                f"- **{r['county']}**: `{r['readiness_status']}` "
                f"(sale={int(r['sale_count'])}, rental={int(r['rental_count'])}, "
                f"coord={r['coord_coverage']}, reason={r['reason']})"
            )
    lines += ["", "## Başiskele special"]
    if basiskele.empty:
        lines.append("- (not produced / no rows)")
    else:
        lines.append(f"- districts: {len(basiskele)}")
        top = basiskele.head(8)
        for _, r in top.iterrows():
            lines.append(
                f"- {r['district']}: sale={int(r['sale_count'])}, "
                f"median_up={r['median_unit_price']}, spread={r['price_spread_ratio']}, "
                f"large_share={r['large_home_share']}"
            )
    lines += ["", "## Top warnings"]
    if not warnings_list:
        lines.append("- none")
    else:
        for w in warnings_list[:15]:
            lines.append(f"- {w}")
    lines += [
        "",
        "## Suggested next actions",
        "- Focus modeling on counties with readiness GOOD/OK.",
        "- Investigate WEAK counties: location backfill, sparse districts, outliers.",
        "- For Başiskele: inspect wide price_spread districts and large_home share.",
        "- Re-run this script after each major scrape to track inventory drift.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def make_plots(
    out_dir: Path,
    *,
    county_df: pd.DataFrame,
    sale: pd.DataFrame,
    rental: pd.DataFrame,
    basiskele: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = _ensure_dir(out_dir / "plots")

    def save_bar(names, values, title, fname, ylabel="count"):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(names)), values)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(plots / fname, dpi=120)
        plt.close(fig)

    if not county_df.empty:
        c = county_df.sort_values("sale_count", ascending=False).head(15)
        save_bar(c["county"].astype(str).tolist(), c["sale_count"].tolist(), "County sale count", "county_sale_count_bar.png")
        c2 = county_df.sort_values("rental_count", ascending=False).head(15)
        save_bar(c2["county"].astype(str).tolist(), c2["rental_count"].tolist(), "County rental count", "county_rental_count_bar.png")
        c3 = county_df.sort_values("sale_coord_coverage", ascending=True).head(15)
        save_bar(
            c3["county"].astype(str).tolist(),
            c3["sale_coord_coverage"].fillna(0).tolist(),
            "County sale location coverage",
            "county_location_coverage_bar.png",
            ylabel="coverage",
        )

    if not sale.empty and sale["unit_price_gross"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(sale["unit_price_gross"].dropna().clip(upper=_pct(sale["unit_price_gross"], 99)), bins=40)
        ax.set_title("Sale unit_price_gross histogram")
        fig.tight_layout()
        fig.savefig(plots / "sale_unit_price_hist.png", dpi=120)
        plt.close(fig)

    if not rental.empty and rental["rent_m2"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(rental["rent_m2"].dropna().clip(upper=_pct(rental["rent_m2"], 99)), bins=40)
        ax.set_title("Rental rent_m2 histogram")
        fig.tight_layout()
        fig.savefig(plots / "rental_m2_hist.png", dpi=120)
        plt.close(fig)

    if not basiskele.empty:
        b = basiskele.sort_values("sale_count", ascending=False).head(20)
        save_bar(
            b["district"].astype(str).tolist(),
            b["sale_count"].tolist(),
            "Başiskele district sale count",
            "basiskele_district_sale_count_bar.png",
        )
        b2 = basiskele.dropna(subset=["price_spread_ratio"]).sort_values("price_spread_ratio", ascending=False).head(20)
        if not b2.empty:
            save_bar(
                b2["district"].astype(str).tolist(),
                b2["price_spread_ratio"].tolist(),
                "Başiskele district price spread (p90/p10)",
                "basiskele_district_price_spread_bar.png",
                ylabel="p90/p10",
            )


def export_samples(out_dir: Path, sale: pd.DataFrame, rental: pd.DataFrame, suspicious: pd.DataFrame, n: int) -> None:
    samples = _ensure_dir(out_dir / "samples")
    _write_csv(sale.head(n), samples / "sample_sale_rows.csv")
    _write_csv(rental.head(n), samples / "sample_rental_rows.csv")
    _write_csv(suspicious.head(n), samples / "suspicious_sample_rows.csv")
    miss = pd.concat(
        [
            sale[~sale["has_lat_lon"].fillna(False)],
            rental[~rental["has_lat_lon"].fillna(False)],
        ],
        ignore_index=True,
    )
    _write_csv(miss.head(n), samples / "missing_location_sample_rows.csv")
    if not sale.empty:
        _write_csv(
            sale.nlargest(min(n, len(sale)), "unit_price_gross"),
            samples / "high_price_sample_rows.csv",
        )
        _write_csv(
            sale.nsmallest(min(n, len(sale)), "unit_price_gross"),
            samples / "low_price_sample_rows.csv",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze Listing inventory from DB (no training).")
    ap.add_argument("--city", default="Kocaeli")
    ap.add_argument("--county", default=None)
    ap.add_argument("--district", default=None)
    ap.add_argument("--purpose", choices=["sale", "rental", "both"], default="both")
    ap.add_argument("--sale-table", default="sale_listings")
    ap.add_argument("--rental-table", default="rental_listings")
    ap.add_argument("--out", default=None, help="Output directory (default: analysis_outputs/listing_inventory/<ts>)")
    ap.add_argument("--export-samples", action="store_true")
    ap.add_argument("--sample-size", type=int, default=50)
    ap.add_argument("--min-sale-unit-price", type=float, default=8000.0)
    ap.add_argument("--max-sale-unit-price", type=float, default=200000.0)
    ap.add_argument("--min-rent-m2", type=float, default=50.0)
    ap.add_argument("--max-rent-m2", type=float, default=2500.0)
    ap.add_argument("--location-coverage-min", type=float, default=0.4)
    ap.add_argument("--no-plots", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        load_root_env(start=HERE)
    except Exception as exc:
        print(f"ERROR: failed to load root .env — {exc}")
        return 2

    root = _repo_root()
    out_dir = Path(args.out) if args.out else root / "analysis_outputs" / "listing_inventory" / _ts_folder()
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    _ensure_dir(out_dir)

    try:
        engine = create_engine(get_database_url())
        # light connectivity check
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        print("DB connected")
    except Exception as exc:
        print(f"ERROR: DB connection failed — {exc}")
        print("Hint: set DATABASE_URL in project-root .env (value not printed).")
        return 3

    sale_raw = pd.DataFrame()
    rental_raw = pd.DataFrame()
    fetch_meta: dict[str, Any] = {}
    missing_cols: list[str] = []

    try:
        if args.purpose in ("sale", "both"):
            sale_raw, meta_s = fetch_listings(
                engine,
                table=args.sale_table,
                purpose="sale",
                city=args.city,
                county=args.county,
                district=args.district,
            )
            fetch_meta["sale"] = meta_s
            missing_cols.extend(meta_s.get("missing_columns") or [])
            print(f"rows fetched sale={len(sale_raw)}")
        if args.purpose in ("rental", "both"):
            rental_raw, meta_r = fetch_listings(
                engine,
                table=args.rental_table,
                purpose="rental",
                city=args.city,
                county=args.county,
                district=args.district,
            )
            fetch_meta["rental"] = meta_r
            missing_cols.extend(meta_r.get("missing_columns") or [])
            print(f"rows fetched rental={len(rental_raw)}")
    except Exception as exc:
        print(f"ERROR: fetch failed — {exc}")
        return 4

    sale, sale_price_meta = (
        _normalize_frame(sale_raw, "sale")
        if len(sale_raw)
        else _normalize_frame(pd.DataFrame(), "sale")
    )
    rental, rental_price_meta = (
        _normalize_frame(rental_raw, "rental")
        if len(rental_raw)
        else _normalize_frame(pd.DataFrame(), "rental")
    )

    sale_price_ok = bool(sale_price_meta.get("price_available"))
    rental_price_ok = bool(rental_price_meta.get("price_available"))
    # unit-only rental (rent_per_m2_*) still counts as price-available for filters
    if not rental_price_ok and rental_price_meta.get("unit_price_column_used"):
        rental_price_ok = True
        rental_price_meta["price_available"] = True
        rental_price_meta["note"] = (
            "total rent column missing; using unit rent column "
            f"{rental_price_meta.get('unit_price_column_used')}"
        )

    sale_mask = _basic_filter_mask(
        sale,
        "sale",
        min_sale_unit_price=args.min_sale_unit_price,
        max_sale_unit_price=args.max_sale_unit_price,
        min_rent_m2=args.min_rent_m2,
        max_rent_m2=args.max_rent_m2,
        price_available=sale_price_ok,
    )
    rental_mask = _basic_filter_mask(
        rental,
        "rental",
        min_sale_unit_price=args.min_sale_unit_price,
        max_sale_unit_price=args.max_sale_unit_price,
        min_rent_m2=args.min_rent_m2,
        max_rent_m2=args.max_rent_m2,
        price_available=rental_price_ok,
    )

    if sale_mask is None:
        sale_f = sale.copy()
        sale_filter_note = "skipped_basic_price_filter:sale_price_column_missing"
        sale_rows_after: Any = None
    else:
        sale_f = sale.loc[sale_mask] if len(sale) else sale
        sale_filter_note = "applied"
        sale_rows_after = int(len(sale_f))

    if rental_mask is None:
        rental_f = rental.copy()
        rental_filter_note = "skipped_basic_price_filter:rental_price_column_missing"
        rental_rows_after: Any = None
    else:
        rental_f = rental.loc[rental_mask] if len(rental) else rental
        rental_filter_note = "applied"
        rental_rows_after = int(len(rental_f))

    both = pd.concat([sale, rental], ignore_index=True)

    # date range
    date_min = date_max = None
    for col in ("scraped_at", "listing_date", "saved_at", "updated_at", "created_at"):
        if col in both.columns and both[col].notna().any():
            dt = pd.to_datetime(both[col], errors="coerce")
            if dt.notna().any():
                date_min = str(dt.min())
                date_max = str(dt.max())
                break

    dup_id = 0
    dup_url = 0
    if "classified_id" in both.columns:
        dup_id = int((both["classified_id"].astype("string").value_counts() > 1).sum())
    if "source_url" in both.columns:
        dup_url = int((both["source_url"].astype("string").value_counts() > 1).sum())

    missing_unique = sorted(set(missing_cols))
    # Don't treat resolved price aliases as missing in top-level warning
    if sale_price_ok:
        missing_unique = [c for c in missing_unique if c not in SALE_PRICE_CANDIDATES and c != "unit_price_gross"]
    if rental_price_ok:
        missing_unique = [
            c
            for c in missing_unique
            if c not in RENTAL_PRICE_CANDIDATES
            and c not in RENTAL_UNIT_PRICE_CANDIDATES
            and c != "unit_price_gross"
        ]

    summary: dict[str, Any] = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "filters": {
            "city": args.city,
            "county": args.county,
            "district": args.district,
            "purpose": args.purpose,
            "sale_table": args.sale_table,
            "rental_table": args.rental_table,
            "min_sale_unit_price": args.min_sale_unit_price,
            "max_sale_unit_price": args.max_sale_unit_price,
            "min_rent_m2": args.min_rent_m2,
            "max_rent_m2": args.max_rent_m2,
            "location_coverage_min": args.location_coverage_min,
        },
        "price_column_used": {
            "sale": sale_price_meta.get("price_column_used"),
            "rental": rental_price_meta.get("price_column_used"),
        },
        "unit_price_column_used": {
            "sale": sale_price_meta.get("unit_price_column_used"),
            "rental": rental_price_meta.get("unit_price_column_used"),
        },
        "price_meta": {
            "sale": sale_price_meta,
            "rental": rental_price_meta,
        },
        "sale_rows_raw": int(len(sale)),
        "rental_rows_raw": int(len(rental)),
        "total_rows_raw": int(len(sale) + len(rental)),
        "sale_rows_after_basic_filter": sale_rows_after,
        "rental_rows_after_basic_filter": rental_rows_after,
        "sale_basic_filter_status": sale_filter_note,
        "rental_basic_filter_status": rental_filter_note,
        "city_count": int(both["city"].nunique()) if len(both) else 0,
        "county_count": int(both["county"].nunique()) if len(both) else 0,
        "district_count": int(both["district"].nunique()) if len(both) else 0,
        "duplicate_classified_id_count": dup_id,
        "duplicate_source_url_count": dup_url,
        "missing_price_count": int(
            (sale["price"].isna().sum() if sale_price_ok and len(sale) else 0)
            + (rental["price"].isna().sum() if rental_price_ok and len(rental) and rental_price_meta.get("price_column_used") else 0)
        ),
        "missing_gross_m2_count": int(both["gross_m2"].isna().sum()) if len(both) else 0,
        "missing_unit_price_count": int(sale["unit_price_gross"].isna().sum()) if len(sale) else 0,
        "missing_county_count": int(both["county"].isna().sum()) if len(both) else 0,
        "missing_district_count": int(both["district"].isna().sum()) if len(both) else 0,
        "missing_location_count": int((~both["has_lat_lon"]).sum()) if len(both) else 0,
        "coordinate_coverage_sale": _coverage(sale["has_lat_lon"]) if len(sale) else np.nan,
        "coordinate_coverage_rental": _coverage(rental["has_lat_lon"]) if len(rental) else np.nan,
        "exact_map_coverage_sale": _coverage(sale["is_exact_map"]) if len(sale) else np.nan,
        "exact_map_coverage_rental": _coverage(rental["is_exact_map"]) if len(rental) else np.nan,
        "date_min": date_min,
        "date_max": date_max,
        "missing_columns": missing_unique,
        "fetch_meta": fetch_meta,
        "location_coverage_min": args.location_coverage_min,
        "output_dir": str(out_dir),
    }

    county_df = build_county_distribution(sale, rental)
    district_df = build_district_distribution(sale, rental)
    sale_price = pd.concat(
        [
            build_price_distribution(sale, "unit_price_gross", "city"),
            build_price_distribution(sale, "unit_price_gross", "county"),
            build_price_distribution(sale, "unit_price_gross", "district"),
        ],
        ignore_index=True,
    )
    if rental_price_ok:
        rental_price = pd.concat(
            [
                build_price_distribution(rental, "rent_m2", "city"),
                build_price_distribution(rental, "rent_m2", "county"),
                build_price_distribution(rental, "rent_m2", "district"),
            ],
            ignore_index=True,
        )
    else:
        rental_price = pd.DataFrame(
            [
                {
                    "level": "warning",
                    "city": args.city,
                    "county": "",
                    "district": "",
                    "count": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "p10": np.nan,
                    "p25": np.nan,
                    "p75": np.nan,
                    "p90": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "std": np.nan,
                    "iqr": np.nan,
                    "cv": np.nan,
                    "outlier_low_count": 0,
                    "outlier_high_count": 0,
                    "warning": "rental_price_column_missing",
                }
            ]
        )
    missingness = build_feature_missingness(sale, rental)
    cardinality = build_categorical_cardinality(both)
    location_q = build_location_quality(sale, rental)
    readiness = build_model_readiness(sale, rental, location_coverage_min=args.location_coverage_min)
    basiskele = pd.DataFrame()
    if str(args.city).lower() == "kocaeli" or (args.county and "başiskele" in str(args.county).lower()):
        basiskele = build_basiskele_special(sale, rental)
    duplicates = build_duplicates(both)
    suspicious = build_suspicious(
        sale,
        rental,
        min_sale_unit_price=args.min_sale_unit_price,
        max_sale_unit_price=args.max_sale_unit_price,
        min_rent_m2=args.min_rent_m2,
        max_rent_m2=args.max_rent_m2,
        sale_price_available=sale_price_ok,
        rental_price_available=rental_price_ok,
    )

    warns = collect_warnings(summary, readiness, county_df)
    if not rental_price_ok:
        warns.insert(0, "rental_price_column_missing")
    if not sale_price_ok:
        warns.insert(0, "sale_price_column_missing")
    summary["top_warnings"] = warns

    # write reports
    (out_dir / "inventory_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    _write_csv(county_df, out_dir / "county_distribution.csv")
    _write_csv(district_df, out_dir / "district_distribution.csv")
    _write_csv(sale_price, out_dir / "sale_price_distribution.csv")
    _write_csv(rental_price, out_dir / "rental_price_distribution.csv")
    _write_csv(missingness, out_dir / "feature_missingness.csv")
    _write_csv(cardinality, out_dir / "categorical_cardinality.csv")
    _write_csv(location_q, out_dir / "location_quality_report.csv")
    _write_csv(readiness, out_dir / "model_readiness_report.csv")
    if not basiskele.empty:
        _write_csv(basiskele, out_dir / "basiskele_special_report.csv")
    _write_csv(duplicates, out_dir / "duplicates_report.csv")
    _write_csv(suspicious, out_dir / "suspicious_rows.csv")
    write_summary_md(
        out_dir / "summary.md",
        summary=summary,
        county_df=county_df,
        readiness=readiness,
        basiskele=basiskele,
        warnings_list=warns,
    )

    if args.export_samples:
        export_samples(out_dir, sale, rental, suspicious, args.sample_size)

    if not args.no_plots:
        try:
            make_plots(out_dir, county_df=county_df, sale=sale, rental=rental, basiskele=basiskele)
        except Exception as exc:
            print(f"WARNING: plots failed — {exc}")
            warns.append(f"plots_failed={exc}")

    print(f"reports written -> {out_dir}")
    print("top warnings:")
    for w in (warns[:5] or ["none"]):
        print(f"  - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
