#!/usr/bin/env python
"""Ankara listing inventory + statistical model-readiness audit (no training).

Data-quality / model-readiness audit only. This script never trains a model,
never writes model artifacts, and never touches app/production files.

Reuses the existing DB connection stack (``env_loader`` / ``db_utils`` /
``canonical_db``) and the Kocaeli inventory + feature-reactivation audit
logic (``analyze_listing_inventory.py``, ``audit_feature_reactivation.py``)
so Ankara is scored with the same yardstick as the Kocaeli V25 checkpoint.

Examples:
  python shared_scripts/audit_ankara_inventory_and_stats.py --city Ankara --compare-city Kocaeli \
      --fast --limit-sale-per-county 300 --limit-rental-per-county 150

  python shared_scripts/audit_ankara_inventory_and_stats.py --city Ankara --compare-city Kocaeli \
      --out analysis_outputs/ankara_inventory_statistical_audit_2026-09-11_1200
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
    PREFERRED_COLUMNS,
    RENTAL_PRICE_CANDIDATES,
    RENTAL_UNIT_PRICE_CANDIDATES,
    SALE_PRICE_CANDIDATES,
    SALE_UNIT_PRICE_CANDIDATES,
    create_engine,
    fetch_listings,
    list_table_columns,
    pick_available_columns,
    resolve_first_present_column,
)
from env_loader import find_project_root, load_root_env  # noqa: E402

# Reuse Kocaeli inventory-audit normalization / stats helpers instead of
# re-deriving them (keeps Ankara scored the same way as Kocaeli V25).
from analyze_listing_inventory import (  # noqa: E402
    LARGE_HOME_M2,
    _basic_filter_mask,
    _coverage,
    _dist_stats,
    _normalize_frame,
    _pct,
    attach_inventory_signals,
)

# Reuse feature-reactivation audit helpers (Yerden Isıtma detection, Turkish
# folding, pipe-token splitting) instead of re-implementing them.
from audit_feature_reactivation import (  # noqa: E402
    _fold,
    _is_yerden,
    _split_pipe_tokens,
)

LARGE_HOME_250_M2 = 250.0

DUPLEX_ROOF_RE = re.compile(r"(çat[iı]\s*dubleks|teras\s*dubleks|penthouse|roof\s*duplex)", re.IGNORECASE)
DUPLEX_GARDEN_RE = re.compile(r"(bah[cç]e\s*dubleks|garden\s*duplex)", re.IGNORECASE)
DUPLEX_MIDDLE_RE = re.compile(r"(ara\s*kat\s*dubleks|orta\s*kat\s*dubleks|middle\s*floor\s*duplex)", re.IGNORECASE)
DUPLEX_ANY_RE = re.compile(
    r"(dubleks|dubleksi|dublex|duplex|penthouse)",
    re.IGNORECASE,
)
SITE_TEXT_RE = re.compile(
    r"(\bsite\b|sitelerde|site\s*i[cç]inde|\bproje\b|konut\s*projesi|yeni\s*proje|marka\s*proje)",
    re.IGNORECASE,
)
_EMPTY_SITE_NAME = {
    "", "nan", "none", "null", "<na>", "yok", "belirtilmemis", "belirtilmemiş", "-", "--",
}
_SITE_INSIDE_TRUE = {
    "1", "true", "yes", "evet", "var", "site içinde", "site icinde", "içinde", "icinde",
}

# Feature groups audited in feature_value_counts_ankara.csv.
FEATURE_GROUP_COLUMNS = (
    "heating",
    "room_count",
    "floor_segment",
    "bathroom_count",
    "kitchen",
    "balcony",
    "usage_status",
    "site_inside",
)

MISSINGNESS_FIELDS = (
    "price",
    "gross_m2",
    "net_m2",
    "room_count",
    "building_age",
    "floor",
    "total_floors",
    "heating",
    "bathroom_count",
    "kitchen",
    "balcony",
    "usage_status",
    "site_name",
    "site_inside",
    "latitude",
    "longitude",
    "county",
    "district",
)

PUBLIC_ROW_COLUMNS = {
    "classified_id", "source_url", "seller_type", "title", "address_text",
    "street_name", "raw", "location_raw",
}


# ---------------------------------------------------------------------------
# Small local helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    root = find_project_root(HERE)
    if root is None:
        raise RuntimeError("Could not locate project root (MANIFEST.json + data/).")
    return root


def _ts_folder() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pd.DataFrame()
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _mad(series: pd.Series) -> float:
    """Median absolute deviation, scaled to be std-comparable (x1.4826)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    med = float(s.median())
    return float((s - med).abs().median()) * 1.4826


def _drift_level_numeric(effect: float) -> str:
    if effect != effect:
        return "unknown"
    a = abs(effect)
    if a < 0.2:
        return "low"
    if a < 0.5:
        return "medium"
    return "high"


def _drift_level_categorical(cramers_v: float) -> str:
    if cramers_v != cramers_v:
        return "unknown"
    if cramers_v < 0.1:
        return "low"
    if cramers_v < 0.3:
        return "medium"
    return "high"


def _room_count_numeric(series: pd.Series) -> pd.Series:
    """Turn '3+1' / '4.5+1' / 'Stüdyo' style room_count text into a numeric room total."""
    s = series.astype("string").fillna("")

    def _parse(v: str) -> float:
        t = v.strip().lower()
        if not t:
            return np.nan
        if "stüdyo" in t or "studio" in t:
            return 1.0
        m = re.match(r"^\s*([\d.]+)\s*\+\s*([\d.]+)\s*$", t)
        if m:
            try:
                return float(m.group(1)) + float(m.group(2))
            except ValueError:
                return np.nan
        m2 = re.match(r"^\s*([\d.]+)\s*$", t)
        if m2:
            try:
                return float(m2.group(1))
            except ValueError:
                return np.nan
        return np.nan

    return s.map(_parse)


def _meaningful_site_name(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower().fillna("")
    return ~s.isin(_EMPTY_SITE_NAME) & (s.str.len() >= 2)


def _truthy_site_inside(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower().fillna("")
    return s.isin(_SITE_INSIDE_TRUE)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def list_unique_cities(engine, table: str) -> pd.DataFrame:
    """Report distinct raw ``province`` values (+ row counts) for the city-normalization check."""
    from sqlalchemy import text

    from canonical_db import resolve_relation, sql_relation

    relation = resolve_relation(table)
    rel_sql = sql_relation(relation)
    available = {c.lower() for c in list_table_columns(engine, relation)}
    city_col = "province" if "province" in available else "city"
    sql = text(
        f"""
        SELECT {city_col} AS raw_city_value, count(*) AS row_count
        FROM {rel_sql}
        GROUP BY {city_col}
        ORDER BY row_count DESC
        """
    )
    df = pd.read_sql(sql, engine)
    df["table"] = relation
    return df


def fetch_city_listings_capped(
    engine,
    *,
    table: str,
    purpose: str,
    city: str,
    source_site: str | None,
    per_county_limit: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch listings for a city, optionally capped to N rows per county (ilçe).

    Falls back to the shared ``fetch_listings`` helper when no cap is requested,
    so full-mode runs go through the exact same code path as the Kocaeli audits.
    """
    if not per_county_limit:
        return fetch_listings(
            engine,
            table=table,
            purpose=purpose,
            city=city,
            source_site=source_site or "",
        )

    from sqlalchemy import text

    from canonical_db import alias_listing_frame, resolve_relation, sql_relation

    relation = resolve_relation(table)
    rel_sql = sql_relation(relation)
    available = list_table_columns(engine, relation)
    selected, missing = pick_available_columns(available, preferred=PREFERRED_COLUMNS)
    select_sql = ", ".join(selected) if selected else "*"
    avail_l = {a.lower() for a in available}
    province_col = "province" if "province" in avail_l else "city"
    district_col = "district" if "district" in avail_l else "county"

    where = [
        f"lower(coalesce({province_col}, '')) = lower(:city)",
        "lower(coalesce(listing_purpose, '')) = lower(:purpose)",
    ]
    params: dict[str, Any] = {"city": city, "purpose": purpose, "lim": int(per_county_limit)}
    if source_site:
        where.append("lower(coalesce(source_site, 'listing_portal')) = lower(:source_site)")
        params["source_site"] = source_site
    where_sql = " AND ".join(where)

    sql = text(
        f"""
        WITH ranked AS (
            SELECT {select_sql},
                   ROW_NUMBER() OVER (
                       PARTITION BY {district_col}
                       ORDER BY saved_at DESC NULLS LAST, updated_at DESC NULLS LAST
                   ) AS __rn
            FROM {rel_sql}
            WHERE {where_sql}
        )
        SELECT * FROM ranked WHERE __rn <= :lim
        """
    )
    try:
        df = pd.read_sql(sql, engine, params=params)
    except Exception as exc:
        raise RuntimeError(f"Failed to query table '{relation}' (capped fetch). Error: {exc}") from exc
    if "__rn" in df.columns:
        df = df.drop(columns=["__rn"])
    df = alias_listing_frame(df)
    meta = {
        "table": relation,
        "selected_columns": selected,
        "missing_columns": missing,
        "rows": int(len(df)),
        "per_county_limit": int(per_county_limit),
    }
    return df, meta


# ---------------------------------------------------------------------------
# Normalization / signal attachment
# ---------------------------------------------------------------------------


def normalize_city_frame(df_raw: pd.DataFrame, purpose: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize a fetched frame using the shared Kocaeli-audit normalizer, then
    attach Ankara-audit-specific derived columns (duplex subtype, large_home_250,
    room_count_numeric, building_age_bucket, floor number-only view)."""
    out, price_meta = _normalize_frame(df_raw, purpose)

    title = out["title"] if "title" in out.columns else pd.Series("", index=out.index, dtype="string")
    site_name = out["site_name"] if "site_name" in out.columns else pd.Series("", index=out.index, dtype="string")
    detail_konut = (
        out["detail_konut_tipi"] if "detail_konut_tipi" in out.columns else pd.Series("", index=out.index, dtype="string")
    )
    blob = title.fillna("").astype(str) + " | " + detail_konut.fillna("").astype(str) + " | " + site_name.fillna("").astype(str)

    out["is_duplex_any"] = blob.str.contains(DUPLEX_ANY_RE.pattern, case=False, regex=True, na=False)
    is_roof = blob.str.contains(DUPLEX_ROOF_RE.pattern, case=False, regex=True, na=False)
    is_garden = blob.str.contains(DUPLEX_GARDEN_RE.pattern, case=False, regex=True, na=False)
    is_middle = blob.str.contains(DUPLEX_MIDDLE_RE.pattern, case=False, regex=True, na=False)
    duplex_type = pd.Series("none", index=out.index, dtype="object")
    duplex_type[out["is_duplex_any"]] = "unknown"
    duplex_type[out["is_duplex_any"] & is_roof] = "roof"
    duplex_type[out["is_duplex_any"] & is_garden & ~is_roof] = "garden"
    duplex_type[out["is_duplex_any"] & is_middle & ~is_roof & ~is_garden] = "middle"
    duplex_type[out["is_duplex_any"] & ~is_roof & ~is_garden & ~is_middle] = "standard"
    out["duplex_type"] = duplex_type

    out["is_large_home_180"] = out["is_large_home"]
    out["is_large_home_250"] = (out["gross_m2"] >= LARGE_HOME_250_M2).fillna(False)

    out["is_yerden"] = out["heating"].map(_is_yerden) if "heating" in out.columns else False
    out["has_site_name"] = _meaningful_site_name(site_name)
    out["site_inside_flag"] = (
        _truthy_site_inside(out["site_inside"]) if "site_inside" in out.columns else pd.Series(False, index=out.index)
    )
    out["text_site_project_signal"] = blob.str.contains(SITE_TEXT_RE.pattern, case=False, regex=True, na=False)
    out["has_site_project_signal"] = out["has_site_name"] | out["site_inside_flag"] | out["text_site_project_signal"]

    out["room_count_numeric"] = _room_count_numeric(out["room_count"]) if "room_count" in out.columns else np.nan

    age = pd.to_numeric(out.get("building_age"), errors="coerce")
    out["building_age_bucket"] = pd.cut(
        age, bins=[-0.1, 0, 5, 10, 20, 40, 200], labels=["0", "1-5", "6-10", "11-20", "21-40", "40+"]
    ).astype("string")

    return out, price_meta


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------


def readiness_label(sale_n: int, rental_n: int, neigh_n: int, sale_coord_cov: float) -> tuple[str, str]:
    cov = sale_coord_cov if sale_coord_cov == sale_coord_cov else 0.0
    reasons_good = [sale_n >= 800, rental_n >= 250, neigh_n >= 10, cov >= 0.75]
    reasons_ok = [sale_n >= 300, rental_n >= 100, neigh_n >= 5, cov >= 0.60]
    if all(reasons_good):
        return "GOOD", "meets GOOD thresholds (sale>=800, rental>=250, neighborhoods>=10, sale_coord>=0.75)"
    if all(reasons_ok):
        return "OK", "meets OK but not GOOD thresholds"
    missing = []
    if sale_n < 300:
        missing.append(f"sale_after_filter={sale_n}<300")
    if rental_n < 100:
        missing.append(f"rental_after_filter={rental_n}<100")
    if neigh_n < 5:
        missing.append(f"neighborhood_count={neigh_n}<5")
    if cov < 0.60:
        missing.append(f"sale_coord_coverage={cov:.3f}<0.60")
    return "WEAK", "below OK: " + ", ".join(missing) if missing else "below OK thresholds"


def build_county_counts(sale_raw: pd.DataFrame, rental_raw: pd.DataFrame, sale_f: pd.DataFrame, rental_f: pd.DataFrame) -> pd.DataFrame:
    counties = sorted(
        set(sale_raw["county"].dropna().astype(str)) | set(rental_raw["county"].dropna().astype(str))
    )
    rows = []
    for county in counties:
        sr = sale_raw[sale_raw["county"].astype(str) == county]
        rr = rental_raw[rental_raw["county"].astype(str) == county]
        sf = sale_f[sale_f["county"].astype(str) == county]
        rf = rental_f[rental_f["county"].astype(str) == county]
        neigh = pd.concat([sf["district"], rf["district"]], ignore_index=True).dropna().astype(str)
        neigh_count = int(neigh.nunique())
        sale_cov = _coverage(sf["has_lat_lon"]) if len(sf) else np.nan
        rental_cov = _coverage(rf["has_lat_lon"]) if len(rf) else np.nan
        label, reason = readiness_label(len(sf), len(rf), neigh_count, sale_cov if sale_cov == sale_cov else 0.0)
        rows.append(
            {
                "county": county,
                "sale_raw": int(len(sr)),
                "rental_raw": int(len(rr)),
                "sale_after_filter": int(len(sf)),
                "rental_after_filter": int(len(rf)),
                "total_after_filter": int(len(sf) + len(rf)),
                "neighborhood_count": neigh_count,
                "sale_coord_coverage": sale_cov,
                "rental_coord_coverage": rental_cov,
                "median_sale_unit_price": float(sf["unit_price_gross"].median()) if len(sf) else np.nan,
                "median_rent_per_m2": float(rf["rent_m2"].median()) if len(rf) else np.nan,
                "readiness_label": label,
                "readiness_reason": reason,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["readiness_label", "sale_after_filter"], ascending=[True, False])


def build_neighborhood_counts(sale_f: pd.DataFrame, rental_f: pd.DataFrame) -> pd.DataFrame:
    keys = set()
    for df in (sale_f, rental_f):
        if df.empty:
            continue
        for _, row in df[["county", "district"]].drop_duplicates().iterrows():
            keys.add((str(row.get("county") or ""), str(row.get("district") or "")))
    rows = []
    for county, neighborhood in sorted(keys):
        if not neighborhood or neighborhood.lower() in {"nan", "none", "<na>"}:
            continue
        s = sale_f[(sale_f["county"].astype(str) == county) & (sale_f["district"].astype(str) == neighborhood)]
        r = rental_f[(rental_f["county"].astype(str) == county) & (rental_f["district"].astype(str) == neighborhood)]
        density = "sparse" if (len(s) + len(r)) < 10 else ("moderate" if (len(s) + len(r)) < 40 else "dense")
        rows.append(
            {
                "county": county,
                "neighborhood": neighborhood,
                "sale_count": int(len(s)),
                "rental_count": int(len(r)),
                "sale_coord_coverage": _coverage(s["has_lat_lon"]) if len(s) else np.nan,
                "rental_coord_coverage": _coverage(r["has_lat_lon"]) if len(r) else np.nan,
                "median_sale_unit_price": float(s["unit_price_gross"].median()) if len(s) else np.nan,
                "median_rent_per_m2": float(r["rent_m2"].median()) if len(r) else np.nan,
                "listing_density_flag": density,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["county", "sale_count"], ascending=[True, False])


def build_price_distribution_by_county(sale_f: pd.DataFrame, rental_f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for purpose, df, col in (("sale", sale_f, "unit_price_gross"), ("rental", rental_f, "rent_m2")):
        if df.empty:
            continue
        for county, g in df.groupby(df["county"].astype(str)):
            stats = _dist_stats(g[col])
            if stats["count"] == 0:
                continue
            p95 = _pct(g[col], 95)
            low_thr, high_thr = (8000.0, 250000.0) if purpose == "sale" else (50.0, 3000.0)
            vals = pd.to_numeric(g[col], errors="coerce")
            rows.append(
                {
                    "purpose": purpose,
                    "county": county,
                    "n": stats["count"],
                    "mean": stats["mean"],
                    "median": stats["median"],
                    "p10": stats["p10"],
                    "p25": stats["p25"],
                    "p75": stats["p75"],
                    "p90": stats["p90"],
                    "p95": p95,
                    "iqr": stats["iqr"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "suspicious_low_count": int((vals < low_thr).sum()),
                    "suspicious_high_count": int((vals > high_thr).sum()),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["purpose", "n"], ascending=[True, False])


def build_suspicious(sale: pd.DataFrame, rental: pd.DataFrame) -> pd.DataFrame:
    """Row-level suspicious flags per the audit spec's rule set (private-debug only)."""
    rows = []

    def add(df: pd.DataFrame, purpose: str, mask: pd.Series, reason: str) -> None:
        sub = df.loc[mask.fillna(False)]
        for idx, r in sub.iterrows():
            rows.append(
                {
                    "reason": reason,
                    "purpose": purpose,
                    "classified_id": r.get("classified_id"),
                    "source_url": r.get("source_url"),
                    "county": r.get("county"),
                    "district": r.get("district"),
                    "price": r.get("price"),
                    "unit_price_gross": r.get("unit_price_gross"),
                    "rent_m2": r.get("rent_m2"),
                    "gross_m2": r.get("gross_m2"),
                    "lat": r.get("lat"),
                    "lon": r.get("lon"),
                }
            )

    for purpose, df in (("sale", sale), ("rental", rental)):
        if df.empty:
            continue
        add(df, purpose, df["price"].notna() & (df["price"] <= 0), "price_le_0")
        add(df, purpose, df["gross_m2"].notna() & (df["gross_m2"] <= 20), "gross_m2_le_20")
        add(df, purpose, df["gross_m2"].notna() & (df["gross_m2"] >= 600), "gross_m2_ge_600")
        add(df, purpose, df["gross_m2"].isna(), "missing_gross_m2")
        add(df, purpose, df["county"].isna() | (df["county"].astype(str).str.len() == 0), "missing_county")
        add(df, purpose, df["district"].isna() | (df["district"].astype(str).str.len() == 0), "missing_neighborhood")
        add(df, purpose, ~df["has_lat_lon"].fillna(False), "missing_coordinates")
        if purpose == "sale":
            add(df, purpose, df["price"].isna(), "missing_target")
            add(df, purpose, df["unit_price_gross"].notna() & (df["unit_price_gross"] < 8000), "unit_price_gross_lt_8000")
            add(df, purpose, df["unit_price_gross"].notna() & (df["unit_price_gross"] > 250000), "unit_price_gross_gt_250000")
        else:
            add(df, purpose, df["rent_m2"].isna() & df["price"].isna(), "missing_target")
            add(df, purpose, df["rent_m2"].notna() & (df["rent_m2"] < 50), "rent_per_m2_lt_50")
            add(df, purpose, df["rent_m2"].notna() & (df["rent_m2"] > 3000), "rent_per_m2_gt_3000")

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_missingness(sale: pd.DataFrame, rental: pd.DataFrame, county_df: pd.DataFrame) -> pd.DataFrame:
    both = pd.concat([sale, rental], ignore_index=True)
    rows = []
    for field in MISSINGNESS_FIELDS:
        col = "lat" if field == "latitude" else ("lon" if field == "longitude" else field)
        if col not in both.columns:
            rows.append(
                {
                    "field": field,
                    "missing_count": len(both),
                    "missing_rate": 1.0 if len(both) else np.nan,
                    "sale_missing_rate": 1.0 if len(sale) else np.nan,
                    "rental_missing_rate": 1.0 if len(rental) else np.nan,
                    "top_counties_missing": "",
                    "warning": "column_not_present_in_db",
                }
            )
            continue
        miss_both = both[col].isna()
        miss_sale = sale[col].isna() if col in sale.columns and len(sale) else pd.Series(dtype=bool)
        miss_rent = rental[col].isna() if col in rental.columns and len(rental) else pd.Series(dtype=bool)
        top_counties = ""
        if "county" in both.columns and miss_both.any():
            vc = both.loc[miss_both, "county"].astype(str).value_counts().head(5)
            top_counties = ", ".join(f"{k}:{int(v)}" for k, v in vc.items())
        rows.append(
            {
                "field": field,
                "missing_count": int(miss_both.sum()),
                "missing_rate": float(miss_both.mean()) if len(both) else np.nan,
                "sale_missing_rate": float(miss_sale.mean()) if len(miss_sale) else np.nan,
                "rental_missing_rate": float(miss_rent.mean()) if len(miss_rent) else np.nan,
                "top_counties_missing": top_counties,
                "warning": "",
            }
        )
    return pd.DataFrame(rows)


def build_feature_value_counts(sale: pd.DataFrame, rental: pd.DataFrame, city_median_sale: float) -> pd.DataFrame:
    frames = [("sale", sale), ("rental", rental)]
    rows: list[dict[str, Any]] = []
    for group in FEATURE_GROUP_COLUMNS:
        if group not in sale.columns and group not in rental.columns:
            rows.append(
                {
                    "feature_group": group,
                    "value": "__COLUMN_MISSING__",
                    "sale_count": 0,
                    "rental_count": 0,
                    "total_count": 0,
                    "county_count": 0,
                    "top_counties": "",
                    "median_sale_unit_price": np.nan,
                    "median_lift_vs_city_median": np.nan,
                    "candidate_for_model": False,
                    "notes": "column_not_present_in_db",
                }
            )
            continue
        values: set[str] = set()
        for _, df in frames:
            if group in df.columns:
                values.update(df[group].dropna().astype(str).unique().tolist())
        for value in sorted(values):
            mask_s = sale[group].astype(str) == value if group in sale.columns else pd.Series(False, index=sale.index)
            mask_r = rental[group].astype(str) == value if group in rental.columns else pd.Series(False, index=rental.index)
            s_n = int(mask_s.sum())
            r_n = int(mask_r.sum())
            cc = sale.loc[mask_s, "county"].astype(str).value_counts() if s_n else pd.Series(dtype=int)
            med = float(sale.loc[mask_s, "unit_price_gross"].median()) if s_n else np.nan
            lift = (
                float(med / city_median_sale - 1.0)
                if med == med and city_median_sale and city_median_sale == city_median_sale and city_median_sale != 0
                else np.nan
            )
            candidate = bool(s_n + r_n >= 100 and int((cc >= 5).sum()) >= 3)
            rows.append(
                {
                    "feature_group": group,
                    "value": value,
                    "sale_count": s_n,
                    "rental_count": r_n,
                    "total_count": s_n + r_n,
                    "county_count": int((cc > 0).sum()),
                    "top_counties": ", ".join(f"{k}:{int(v)}" for k, v in cc.head(5).items()),
                    "median_sale_unit_price": med,
                    "median_lift_vs_city_median": lift,
                    "candidate_for_model": candidate,
                    "notes": "",
                }
            )

    # Derived binary signals (duplex/large-home/site) reported the same way.
    for flag, label, group in (
        ("is_yerden", "Yerden Isıtma (heating contains 'yerden')", "heating_yerden_isitma"),
        ("is_duplex_any", "duplex_text_signal", "duplex_type"),
        ("is_large_home_180", "large_home_180", "large_home_180"),
        ("is_large_home_250", "large_home_250", "large_home_250"),
        ("has_site_name", "site_name_present", "site_name_availability"),
        ("site_inside_flag", "site_inside_true", "site_inside"),
    ):
        if flag not in sale.columns and flag not in rental.columns:
            continue
        mask_s = sale[flag].fillna(False) if flag in sale.columns else pd.Series(False, index=sale.index)
        mask_r = rental[flag].fillna(False) if flag in rental.columns else pd.Series(False, index=rental.index)
        s_n = int(mask_s.sum())
        r_n = int(mask_r.sum())
        cc = sale.loc[mask_s, "county"].astype(str).value_counts() if s_n else pd.Series(dtype=int)
        med = float(sale.loc[mask_s, "unit_price_gross"].median()) if s_n else np.nan
        lift = (
            float(med / city_median_sale - 1.0)
            if med == med and city_median_sale and city_median_sale == city_median_sale and city_median_sale != 0
            else np.nan
        )
        candidate = bool(s_n + r_n >= 100 and int((cc >= 5).sum()) >= 3)
        note = ""
        if flag == "is_yerden":
            note = "candidate_for_multi_city_model: keep as heating leaf if county_count>=3 and volume holds"
        rows.append(
            {
                "feature_group": group,
                "value": label,
                "sale_count": s_n,
                "rental_count": r_n,
                "total_count": s_n + r_n,
                "county_count": int((cc > 0).sum()),
                "top_counties": ", ".join(f"{k}:{int(v)}" for k, v in cc.head(5).items()),
                "median_sale_unit_price": med,
                "median_lift_vs_city_median": lift,
                "candidate_for_model": candidate,
                "notes": note,
            }
        )
    return pd.DataFrame(rows)


def yerden_isitma_special(sale: pd.DataFrame, rental: pd.DataFrame) -> dict[str, Any]:
    ys = sale["is_yerden"].fillna(False) if "is_yerden" in sale.columns else pd.Series(dtype=bool)
    yr = rental["is_yerden"].fillna(False) if "is_yerden" in rental.columns else pd.Series(dtype=bool)
    other_med = float(sale.loc[~ys, "unit_price_gross"].median()) if len(sale) and (~ys).any() else np.nan
    yerden_med = float(sale.loc[ys, "unit_price_gross"].median()) if ys.any() else np.nan
    lift = (yerden_med / other_med - 1.0) if other_med and other_med == other_med and other_med != 0 else np.nan
    by_county = (
        sale.groupby(sale["county"].astype(str)).agg(sale_total=("county", "size"), yerden_sale=("is_yerden", "sum")).reset_index()
        if len(sale)
        else pd.DataFrame(columns=["county", "sale_total", "yerden_sale"])
    )
    if not by_county.empty:
        by_county["yerden_share"] = by_county["yerden_sale"] / by_county["sale_total"].clip(lower=1)
    county_ge30 = int((by_county["yerden_sale"] >= 30).sum()) if not by_county.empty else 0
    candidate = bool(int(ys.sum()) + int(yr.sum()) >= 100 and county_ge30 >= 3)
    return {
        "sale_count": int(ys.sum()) if len(sale) else 0,
        "rental_count": int(yr.sum()) if len(rental) else 0,
        "sale_share": float(ys.mean()) if len(sale) else np.nan,
        "rental_share": float(yr.mean()) if len(rental) else np.nan,
        "median_unit_price_yerden": yerden_med,
        "median_unit_price_other": other_med,
        "city_level_lift": lift,
        "counties_with_ge30_yerden_sale": county_ge30,
        "candidate_for_model": candidate,
        "note": (
            "Yerden Isıtma has sufficient volume/county spread for multi-city inclusion."
            if candidate
            else "Yerden Isıtma volume/county spread below multi-city inclusion threshold; monitor, don't force it in yet."
        ),
        "by_county": by_county,
    }


def build_site_project_coverage(sale: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if sale.empty:
        return pd.DataFrame()
    for county, g in sale.groupby(sale["county"].astype(str)):
        n = len(g)
        non_missing = g["has_site_name"].sum()
        uniq = g.loc[g["has_site_name"], "site_name"].astype(str).str.strip().nunique()
        top_names = (
            g.loc[g["has_site_name"], "site_name"]
            .astype(str)
            .str.strip()
            .value_counts()
            .head(5)
        )
        signal_share = float(g["has_site_project_signal"].mean()) if n else np.nan
        strength = "strong" if signal_share >= 0.25 else ("moderate" if signal_share >= 0.10 else "weak")
        rows.append(
            {
                "county": county,
                "listing_count": int(n),
                "site_name_non_missing": int(non_missing),
                "site_name_coverage": float(non_missing / n) if n else np.nan,
                "unique_site_name_count": int(uniq),
                "top_site_names_public_safe": "; ".join(f"{k} ({int(v)})" for k, v in top_names.items()),
                "possible_site_project_signal_strength": strength,
                "notes": "site names noisy; no canonical site/project ID assigned by this audit",
            }
        )
    return pd.DataFrame(rows).sort_values("listing_count", ascending=False)


def build_duplex_largehome_coverage(sale: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if sale.empty:
        return pd.DataFrame()
    for county, g in sale.groupby(sale["county"].astype(str)):
        n = len(g)
        dt = g["duplex_type"].astype(str)
        dcount = int((dt != "none").sum())
        lh180 = g["is_large_home_180"].fillna(False)
        lh250 = g["is_large_home_250"].fillna(False)
        med_duplex = float(g.loc[dt != "none", "unit_price_gross"].median()) if dcount else np.nan
        med_lh = float(g.loc[lh180, "unit_price_gross"].median()) if lh180.any() else np.nan
        rows.append(
            {
                "county": county,
                "total_sale_count": int(n),
                "duplex_count": dcount,
                "roof_duplex_count": int((dt == "roof").sum()),
                "garden_duplex_count": int((dt == "garden").sum()),
                "middle_floor_duplex_count": int((dt == "middle").sum()),
                "standard_duplex_count": int((dt == "standard").sum()),
                "unknown_duplex_count": int((dt == "unknown").sum()),
                "large_home_180_count": int(lh180.sum()),
                "large_home_250_count": int(lh250.sum()),
                "median_unit_price_duplex": med_duplex,
                "median_unit_price_large_home": med_lh,
                "notes": "",
            }
        )
    return pd.DataFrame(rows).sort_values("total_sale_count", ascending=False)


# ---------------------------------------------------------------------------
# Ankara vs Kocaeli comparison + statistical tests
# ---------------------------------------------------------------------------


def build_distribution_comparison(
    ank_sale: pd.DataFrame,
    ank_rental: pd.DataFrame,
    koc_sale: pd.DataFrame,
    koc_rental: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add_numeric(metric: str, a_series: pd.Series, k_series: pd.Series) -> None:
        a = pd.to_numeric(a_series, errors="coerce").dropna()
        k = pd.to_numeric(k_series, errors="coerce").dropna()
        a_med = float(a.median()) if len(a) else np.nan
        k_med = float(k.median()) if len(k) else np.nan
        abs_diff = a_med - k_med if a_med == a_med and k_med == k_med else np.nan
        rel_diff = (abs_diff / k_med) if abs_diff == abs_diff and k_med not in (0, np.nan) and k_med == k_med and k_med != 0 else np.nan
        pooled_mad = _mad(pd.concat([a, k], ignore_index=True)) if len(a) and len(k) else np.nan
        effect = (abs_diff / pooled_mad) if pooled_mad and pooled_mad == pooled_mad and pooled_mad != 0 else np.nan
        rows.append(
            {
                "metric": metric,
                "kocaeli_value": k_med,
                "ankara_value": a_med,
                "absolute_diff": abs_diff,
                "relative_diff": rel_diff,
                "drift_level": _drift_level_numeric(effect),
                "notes": f"median comparison, n_ankara={len(a)}, n_kocaeli={len(k)}",
            }
        )

    def add_ratio(metric: str, a_val: float, k_val: float, note: str) -> None:
        abs_diff = a_val - k_val if a_val == a_val and k_val == k_val else np.nan
        rel_diff = (abs_diff / k_val) if abs_diff == abs_diff and k_val not in (0, np.nan) and k_val == k_val and k_val != 0 else np.nan
        level = "low" if (abs_diff == abs_diff and abs(abs_diff) < 0.05) else ("medium" if abs_diff == abs_diff and abs(abs_diff) < 0.15 else ("high" if abs_diff == abs_diff else "unknown"))
        rows.append(
            {
                "metric": metric,
                "kocaeli_value": k_val,
                "ankara_value": a_val,
                "absolute_diff": abs_diff,
                "relative_diff": rel_diff,
                "drift_level": level,
                "notes": note,
            }
        )

    add_numeric("sale_unit_price_gross_median", ank_sale.get("unit_price_gross", pd.Series(dtype=float)), koc_sale.get("unit_price_gross", pd.Series(dtype=float)))
    add_numeric("rent_per_m2_median", ank_rental.get("rent_m2", pd.Series(dtype=float)), koc_rental.get("rent_m2", pd.Series(dtype=float)))
    add_numeric("sale_gross_m2_median", ank_sale.get("gross_m2", pd.Series(dtype=float)), koc_sale.get("gross_m2", pd.Series(dtype=float)))
    add_numeric("sale_building_age_median", ank_sale.get("building_age", pd.Series(dtype=float)), koc_sale.get("building_age", pd.Series(dtype=float)))
    add_numeric("sale_room_count_numeric_median", ank_sale.get("room_count_numeric", pd.Series(dtype=float)), koc_sale.get("room_count_numeric", pd.Series(dtype=float)))

    for col in ("floor_segment", "heating"):
        if col in ank_sale.columns and col in koc_sale.columns and len(ank_sale) and len(koc_sale):
            a_top = ank_sale[col].astype(str).value_counts(normalize=True)
            k_top = koc_sale[col].astype(str).value_counts(normalize=True)
            top_cat = k_top.index[0] if len(k_top) else None
            if top_cat is not None:
                add_ratio(
                    f"{col}_top_category_share ({top_cat})",
                    float(a_top.get(top_cat, 0.0)),
                    float(k_top.get(top_cat, 0.0)),
                    "share of Kocaeli's most common category, compared in Ankara",
                )

    add_ratio(
        "site_name_coverage_sale",
        float(ank_sale["has_site_name"].mean()) if len(ank_sale) else np.nan,
        float(koc_sale["has_site_name"].mean()) if len(koc_sale) else np.nan,
        "share of sale rows with a non-empty site_name",
    )
    add_ratio(
        "duplex_share_sale",
        float((ank_sale["duplex_type"] != "none").mean()) if len(ank_sale) else np.nan,
        float((koc_sale["duplex_type"] != "none").mean()) if len(koc_sale) else np.nan,
        "share of sale rows with a duplex text signal",
    )
    add_ratio(
        "large_home_180_share_sale",
        float(ank_sale["is_large_home_180"].mean()) if len(ank_sale) else np.nan,
        float(koc_sale["is_large_home_180"].mean()) if len(koc_sale) else np.nan,
        "share of sale rows with gross_m2>=180",
    )
    add_ratio(
        "coordinate_coverage_sale",
        _coverage(ank_sale["has_lat_lon"]) if len(ank_sale) else np.nan,
        _coverage(koc_sale["has_lat_lon"]) if len(koc_sale) else np.nan,
        "share of sale rows with lat/lon present",
    )

    ank_both = pd.concat([ank_sale, ank_rental], ignore_index=True) if len(ank_sale) or len(ank_rental) else pd.DataFrame()
    koc_both = pd.concat([koc_sale, koc_rental], ignore_index=True) if len(koc_sale) or len(koc_rental) else pd.DataFrame()
    common_fields = [f for f in MISSINGNESS_FIELDS if f in ank_both.columns and f in koc_both.columns]
    if common_fields and len(ank_both) and len(koc_both):
        a_missrate = float(np.mean([ank_both[f].isna().mean() for f in common_fields]))
        k_missrate = float(np.mean([koc_both[f].isna().mean() for f in common_fields]))
        add_ratio(
            "avg_missingness_across_checked_fields",
            a_missrate,
            k_missrate,
            f"averaged over {len(common_fields)} fields: {', '.join(common_fields)}",
        )

    return pd.DataFrame(rows)


def build_statistical_tests(
    ank_sale: pd.DataFrame,
    ank_rental: pd.DataFrame,
    koc_sale: pd.DataFrame,
    koc_rental: pd.DataFrame,
) -> dict[str, Any]:
    from scipy import stats as sstats

    out: dict[str, Any] = {"caveat": "p-value alone is not interpreted; large-N samples make even small differences 'significant'. Use effect_size for magnitude.", "tests": {}}

    def ks_test(name: str, a: pd.Series, k: pd.Series) -> None:
        a_v = pd.to_numeric(a, errors="coerce").dropna()
        k_v = pd.to_numeric(k, errors="coerce").dropna()
        if len(a_v) < 5 or len(k_v) < 5:
            out["tests"][name] = {"statistic": None, "p_value": None, "effect_size": None, "drift_level": "unknown", "interpretation": "insufficient sample size"}
            return
        res = sstats.ks_2samp(a_v.to_numpy(dtype=float), k_v.to_numpy(dtype=float))
        pooled_mad = _mad(pd.concat([a_v, k_v], ignore_index=True))
        effect = float((a_v.median() - k_v.median()) / pooled_mad) if pooled_mad and pooled_mad == pooled_mad and pooled_mad != 0 else None
        level = _drift_level_numeric(effect if effect is not None else float("nan"))
        out["tests"][name] = {
            "test": "ks_2samp",
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            "effect_size": effect,
            "effect_size_definition": "robust standardized median diff = (median_ankara - median_kocaeli) / pooled_MAD",
            "n_ankara": int(len(a_v)),
            "n_kocaeli": int(len(k_v)),
            "drift_level": level,
            "interpretation": (
                f"KS statistic {res.statistic:.3f} (p={res.pvalue:.2e}); effect_size magnitude classifies drift as '{level}'. "
                "Do not read p_value alone at this sample size."
            ),
        }

    def chi_square_test(name: str, a: pd.Series, k: pd.Series) -> None:
        a_v = a.astype("string").dropna()
        k_v = k.astype("string").dropna()
        if len(a_v) < 5 or len(k_v) < 5:
            out["tests"][name] = {"statistic": None, "p_value": None, "effect_size": None, "drift_level": "unknown", "interpretation": "insufficient sample size"}
            return
        cats = sorted(set(a_v.unique().tolist()) | set(k_v.unique().tolist()))
        if len(cats) < 2:
            out["tests"][name] = {"statistic": None, "p_value": None, "effect_size": None, "drift_level": "unknown", "interpretation": "fewer than 2 categories observed"}
            return
        a_counts = a_v.value_counts().reindex(cats, fill_value=0)
        k_counts = k_v.value_counts().reindex(cats, fill_value=0)
        table = np.array([a_counts.to_numpy(), k_counts.to_numpy()], dtype=float)
        try:
            chi2, p, dof, _exp = sstats.chi2_contingency(table)
        except ValueError as exc:
            out["tests"][name] = {"statistic": None, "p_value": None, "effect_size": None, "drift_level": "unknown", "interpretation": f"chi2 failed: {exc}"}
            return
        n = table.sum()
        min_dim = min(table.shape) - 1
        cramers_v = float(np.sqrt((chi2 / n) / max(min_dim, 1))) if n else None
        level = _drift_level_categorical(cramers_v if cramers_v is not None else float("nan"))
        out["tests"][name] = {
            "test": "chi2_contingency",
            "statistic": float(chi2),
            "p_value": float(p),
            "effect_size": cramers_v,
            "effect_size_definition": "Cramer's V",
            "n_ankara": int(len(a_v)),
            "n_kocaeli": int(len(k_v)),
            "categories_compared": cats[:30],
            "drift_level": level,
            "interpretation": (
                f"chi2={chi2:.1f} (p={p:.2e}), Cramer's V={cramers_v:.3f} -> drift '{level}'. "
                "Do not read p_value alone at this sample size."
            ),
        }

    ks_test("unit_price_gross_sale", ank_sale.get("unit_price_gross", pd.Series(dtype=float)), koc_sale.get("unit_price_gross", pd.Series(dtype=float)))
    ks_test("gross_m2_sale", ank_sale.get("gross_m2", pd.Series(dtype=float)), koc_sale.get("gross_m2", pd.Series(dtype=float)))
    ks_test("building_age_sale", ank_sale.get("building_age", pd.Series(dtype=float)), koc_sale.get("building_age", pd.Series(dtype=float)))
    ks_test("rent_per_m2_rental", ank_rental.get("rent_m2", pd.Series(dtype=float)), koc_rental.get("rent_m2", pd.Series(dtype=float)))

    if "heating" in ank_sale.columns and "heating" in koc_sale.columns:
        chi_square_test("heating_sale", ank_sale["heating"], koc_sale["heating"])
    if "room_count" in ank_sale.columns and "room_count" in koc_sale.columns:
        chi_square_test("room_count_sale", ank_sale["room_count"], koc_sale["room_count"])
    if "site_inside" in ank_sale.columns and "site_inside" in koc_sale.columns:
        chi_square_test("site_inside_sale", ank_sale["site_inside"], koc_sale["site_inside"])
    if "duplex_type" in ank_sale.columns and "duplex_type" in koc_sale.columns:
        chi_square_test("duplex_type_sale", ank_sale["duplex_type"], koc_sale["duplex_type"])
    if "floor_segment" in ank_sale.columns and "floor_segment" in koc_sale.columns:
        chi_square_test("floor_segment_sale", ank_sale["floor_segment"], koc_sale["floor_segment"])

    return out


# ---------------------------------------------------------------------------
# Markdown / JSON summaries
# ---------------------------------------------------------------------------


def write_model_readiness_md(
    path: Path,
    *,
    summary: dict[str, Any],
    county_df: pd.DataFrame,
    yerden: dict[str, Any],
    site_cov: pd.DataFrame,
    duplex_cov: pd.DataFrame,
    comparison: pd.DataFrame,
    training_rec: dict[str, Any],
) -> None:
    lines = [
        "# Ankara Model Readiness — Audit Only (No Training)",
        "",
        f"- run_timestamp: `{summary.get('run_timestamp')}`",
        f"- city: `{summary.get('city')}` vs compare_city: `{summary.get('compare_city')}`",
        "",
        "## Ankara dataset overview",
        f"- sale raw: **{summary.get('sale_raw')}**, sale after filter: **{summary.get('sale_after_filter')}**",
        f"- rental raw: **{summary.get('rental_raw')}**, rental after filter: **{summary.get('rental_after_filter')}**",
        f"- counties observed: **{summary.get('county_count')}**, neighborhoods observed: **{summary.get('neighborhood_count')}**",
        "",
        "## İlçe (county) readiness table",
        "",
        "| county | sale | rental | neighborhoods | sale_coord | label |",
        "|---|---:|---:|---:|---:|---|",
    ]
    if county_df is not None and not county_df.empty:
        for _, r in county_df.iterrows():
            lines.append(
                f"| {r['county']} | {int(r['sale_after_filter'])} | {int(r['rental_after_filter'])} | "
                f"{int(r['neighborhood_count'])} | {r['sale_coord_coverage']:.2f} | {r['readiness_label']} |"
                if r["sale_coord_coverage"] == r["sale_coord_coverage"]
                else f"| {r['county']} | {int(r['sale_after_filter'])} | {int(r['rental_after_filter'])} | {int(r['neighborhood_count'])} | n/a | {r['readiness_label']} |"
            )
    else:
        lines.append("| (no data) | | | | | |")

    good = county_df[county_df["readiness_label"] == "GOOD"]["county"].tolist() if county_df is not None and not county_df.empty else []
    ok = county_df[county_df["readiness_label"] == "OK"]["county"].tolist() if county_df is not None and not county_df.empty else []
    weak = county_df[county_df["readiness_label"] == "WEAK"]["county"].tolist() if county_df is not None and not county_df.empty else []
    lines += [
        "",
        "## Güçlü ilçeler (GOOD)",
        ("- " + ", ".join(good)) if good else "- (none)",
        "",
        "## OK ilçeler",
        ("- " + ", ".join(ok)) if ok else "- (none)",
        "",
        "## Zayıf ilçeler (WEAK)",
        ("- " + ", ".join(weak)) if weak else "- (none)",
        "",
        "## Koordinat coverage",
        f"- sale: `{summary.get('coordinate_coverage_sale')}`",
        f"- rental: `{summary.get('coordinate_coverage_rental')}`",
        "",
        "## Feature readiness",
        "- See `feature_value_counts_ankara.csv` for per-value county spread and `candidate_for_model` flags.",
        "",
        "## Site/project readiness",
    ]
    if site_cov is not None and not site_cov.empty:
        for _, r in site_cov.head(10).iterrows():
            if r["site_name_coverage"] == r["site_name_coverage"]:
                lines.append(f"- {r['county']}: coverage={r['site_name_coverage']:.2f}, signal={r['possible_site_project_signal_strength']}")
            else:
                lines.append(f"- {r['county']}: coverage=n/a")
    else:
        lines.append("- (no sale rows)")
    lines += [
        "",
        "## Yerden Isıtma durumu",
        f"- sale_count={yerden.get('sale_count')}, rental_count={yerden.get('rental_count')}",
        f"- sale_share={yerden.get('sale_share')}, rental_share={yerden.get('rental_share')}",
        f"- city_level_lift vs other heating: {yerden.get('city_level_lift')}",
        f"- counties_with_ge30_yerden_sale: {yerden.get('counties_with_ge30_yerden_sale')}",
        f"- candidate_for_model: **{yerden.get('candidate_for_model')}** — {yerden.get('note')}",
        "",
        "## Duplex/large-home durumu",
    ]
    if duplex_cov is not None and not duplex_cov.empty:
        tot_duplex = int(duplex_cov["duplex_count"].sum())
        tot_lh180 = int(duplex_cov["large_home_180_count"].sum())
        tot_sale = int(duplex_cov["total_sale_count"].sum())
        lines.append(f"- city-wide duplex_count={tot_duplex} ({tot_duplex / max(tot_sale,1):.1%} of sale), large_home_180_count={tot_lh180} ({tot_lh180 / max(tot_sale,1):.1%})")
    else:
        lines.append("- (no sale rows)")
    lines += ["", "## Ankara vs Kocaeli distribution drift", ""]
    if comparison is not None and not comparison.empty:
        for _, r in comparison.iterrows():
            lines.append(f"- **{r['metric']}**: kocaeli={r['kocaeli_value']}, ankara={r['ankara_value']}, drift=`{r['drift_level']}`")
    else:
        lines.append("- (comparison unavailable)")
    lines += [
        "",
        "## Model eğitimi öncesi riskler",
        "- Coordinate coverage varies sharply by ilçe; low-coverage counties will weaken any geo feature.",
        "- Suspicious-row rules (price/m2/unit-price bounds) can hide genuine luxury/budget listings — review before hard-filtering.",
        "- Site/project names are noisy free text; no canonical site ID exists yet, so site-level effects must stay county-scoped.",
        "- Ankara-Kocaeli distribution drift (see table above) means a naive pooled model risks biasing toward whichever city has more rows.",
        "- This audit does not validate label correctness (scrape errors, stale listings) — only structural/statistical readiness.",
        "",
        "## Önerilen ilk eğitim stratejisi",
    ]
    for note in training_rec.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_training_recommendation(
    county_df: pd.DataFrame,
    comparison: pd.DataFrame,
    stat_tests: dict[str, Any],
    ankara_summary: dict[str, Any],
) -> dict[str, Any]:
    good = county_df[county_df["readiness_label"] == "GOOD"]["county"].tolist() if county_df is not None and not county_df.empty else []
    ok = county_df[county_df["readiness_label"] == "OK"]["county"].tolist() if county_df is not None and not county_df.empty else []
    weak = county_df[county_df["readiness_label"] == "WEAK"]["county"].tolist() if county_df is not None and not county_df.empty else []

    high_drift_metrics = []
    if comparison is not None and not comparison.empty:
        high_drift_metrics = comparison[comparison["drift_level"] == "high"]["metric"].tolist()
    for name, t in (stat_tests.get("tests") or {}).items():
        if t.get("drift_level") == "high":
            high_drift_metrics.append(name)

    ankara_only_ready = bool(len(good) + len(ok) >= 3 and int(ankara_summary.get("sale_after_filter") or 0) >= 1000)
    multi_city_ready = bool(ankara_only_ready and len(high_drift_metrics) <= 6)

    notes = [
        "Do not train yet on this audit alone — it is a readiness/statistics report, not a train run.",
        (
            f"Ankara-only baseline first, scoped to GOOD+OK counties ({', '.join(good + ok) if (good or ok) else '(none yet)'})."
            if ankara_only_ready
            else "Ankara is not yet dense enough for a standalone baseline; densify WEAK counties first: "
            + (", ".join(weak) if weak else "(see county_counts_ankara.csv)")
        ),
        (
            "Ankara/Kocaeli distribution drift is expected, not disqualifying — it argues for city-aware modeling "
            "(a `city` feature, or separate-then-blend models), not for silently pooling the two."
            if high_drift_metrics
            else "Ankara/Kocaeli distributions look broadly comparable; a pooled multi-city model is a reasonable next step."
        ),
        "If/when a Kocaeli+Ankara multi-city model is attempted, `city` must be a mandatory categorical feature.",
        "County/district/neighborhood values must stay scoped by city (Çankaya in Ankara is not the same key as any Kocaeli ilçe).",
        "Site/project canonical IDs must be city+county scoped (e.g. `ankara::cankaya::<slug>`, `kocaeli::izmit::<slug>`), never merged across cities automatically.",
    ]

    return {
        "recommended_first_model": "ankara_only_baseline" if ankara_only_ready else "continue_ankara_data_collection",
        "recommended_county_policy": "core_good_ok" if (good or ok) else "insufficient_data",
        "included_counties_first_run": good + ok,
        "excluded_or_diagnostic_counties": weak,
        "feature_stack_recommendation": [
            "gross_m2", "net_m2", "room_count", "building_age", "floor_segment", "heating",
            "bathroom_count", "site_inside", "lat", "lon", "county", "district",
            "duplex_type (if volume supports it — see duplex_largehome_coverage_ankara.csv)",
            "site_project signal (county-scoped only; no cross-city canonical ID yet)",
        ],
        "target_transform_recommendation": "log1p(unit_price_gross) for sale, log1p(rent_per_m2) for rental — matches Kocaeli V25 practice; verify residuals before committing.",
        "validation_strategy_recommendation": "county-stratified holdout (mirror Kocaeli V25's core_good_ok / WEAK split) plus a held-out time slice if scrape dates allow.",
        "multi_city_ready": multi_city_ready,
        "ankara_only_baseline_ready": ankara_only_ready,
        "high_drift_metrics_vs_kocaeli": high_drift_metrics,
        "notes": notes,
    }


def write_readme(
    path: Path,
    *,
    summary: dict[str, Any],
    county_df: pd.DataFrame,
    suspicious_count: int,
    yerden: dict[str, Any],
    comparison: pd.DataFrame,
    training_rec: dict[str, Any],
) -> None:
    good = county_df[county_df["readiness_label"] == "GOOD"]["county"].tolist() if county_df is not None and not county_df.empty else []
    weak = county_df[county_df["readiness_label"] == "WEAK"]["county"].tolist() if county_df is not None and not county_df.empty else []
    lines = [
        "# Ankara Inventory & Statistical Audit — README",
        "",
        "**This run did not train a model, write model artifacts, or modify the app.** "
        "It is a data-quality / model-readiness audit only.",
        "",
        f"- run_timestamp: `{summary.get('run_timestamp')}`",
        f"- script: `shared_scripts/audit_ankara_inventory_and_stats.py`",
        f"- city filter: `province = '{summary.get('city')}'` (compare_city: `{summary.get('compare_city')}`)",
        f"- DB tables: sale=`{summary.get('sale_table')}`, rental=`{summary.get('rental_table')}`",
        f"- fast mode: `{summary.get('fast_mode')}` (per_county_limit sale={summary.get('limit_sale_per_county')}, rental={summary.get('limit_rental_per_county')})",
        "",
        "## Totals",
        f"- sale: raw={summary.get('sale_raw')}, after_filter={summary.get('sale_after_filter')}",
        f"- rental: raw={summary.get('rental_raw')}, after_filter={summary.get('rental_after_filter')}",
        f"- counties: {summary.get('county_count')}, neighborhoods: {summary.get('neighborhood_count')}",
        "",
        "## Readiness result",
        f"- GOOD counties: {', '.join(good) if good else '(none)'}",
        f"- WEAK counties: {', '.join(weak) if weak else '(none)'}",
        f"- suspicious rows flagged: {suspicious_count}",
        "",
        "## Most important feature signals",
        f"- Yerden Isıtma: sale={yerden.get('sale_count')}, candidate_for_model={yerden.get('candidate_for_model')}",
        "- Full feature signal table: `feature_value_counts_ankara.csv`",
        "",
        "## Ankara vs Kocaeli drift",
    ]
    if comparison is not None and not comparison.empty:
        high = comparison[comparison["drift_level"] == "high"]["metric"].tolist()
        lines.append(f"- high-drift metrics: {', '.join(high) if high else '(none)'}")
        lines.append("- full table: `ankara_vs_kocaeli_distribution_comparison.csv`, `statistical_tests_ankara_vs_kocaeli.json`")
    else:
        lines.append("- comparison unavailable (see console warnings)")
    lines += [
        "",
        "## Recommended next step",
        f"- {training_rec.get('recommended_first_model')}",
        f"- ankara_only_baseline_ready: **{training_rec.get('ankara_only_baseline_ready')}**",
        f"- multi_city_ready: **{training_rec.get('multi_city_ready')}**",
        "- See `model_readiness_ankara.md` and `ankara_model_training_recommendation.json` for full detail.",
        "",
        "## Security / privacy notes",
        "- Public CSVs/JSON in this folder never contain classified_id, source_url, seller/contact info, exact address, or exact coordinates.",
        "- `suspicious_rows_ankara_private_debug.csv` is a **private/debug** file (contains classified_id/source_url for row-level triage). "
        "Do not include it in portfolio or app exports.",
        "",
        "## Train status",
        "**No training was run. No model artifacts were produced or modified. V24.1/V25 checkpoints and the app were not touched.**",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ankara listing inventory + statistical model-readiness audit (no training).")
    ap.add_argument("--city", default="Ankara")
    ap.add_argument("--compare-city", default="Kocaeli")
    ap.add_argument("--sale-table", default=os.getenv("SALE_TABLE", "market.sale_listings"))
    ap.add_argument("--rental-table", default=os.getenv("RENTAL_TABLE", "market.rental_listings"))
    ap.add_argument("--source-site", default="", help="Optional source_site filter; empty = no filter (default).")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fast", action="store_true", help="Smoke-test mode: implies per-county row caps if not otherwise set.")
    ap.add_argument("--limit-sale-per-county", type=int, default=None)
    ap.add_argument("--limit-rental-per-county", type=int, default=None)
    return ap.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    try:
        load_root_env(start=HERE)
    except Exception as exc:
        print(f"ERROR: failed to load root .env — {exc}")
        return 2

    args = parse_args()
    limit_sale = args.limit_sale_per_county or (300 if args.fast else None)
    limit_rental = args.limit_rental_per_county or (150 if args.fast else None)

    root = _repo_root()
    out_dir = Path(args.out) if args.out else root / "analysis_outputs" / f"ankara_inventory_statistical_audit_{_ts_folder()}"
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    _ensure_dir(out_dir)

    try:
        from db_url import safe_database_label

        engine = create_engine()
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        print(f"DB connected ({safe_database_label()})")
    except Exception as exc:
        try:
            from db_url import sanitize_db_error

            msg = sanitize_db_error(exc)
        except Exception:
            msg = "connection failed (details redacted)"
        print(f"ERROR: DB connection failed — {msg}")
        return 3

    # --- unique-city normalization check -----------------------------------
    data_source_notes: list[str] = []
    try:
        uniq_cities = list_unique_cities(engine, args.sale_table)
        _write_csv(uniq_cities, out_dir / "unique_city_values.csv")
        match = uniq_cities[uniq_cities["raw_city_value"].astype(str).str.lower() == args.city.lower()]
        if match.empty:
            data_source_notes.append(f"WARNING: no exact case-insensitive match for city='{args.city}' in raw province values.")
            print(f"WARNING: no exact match for --city {args.city!r} in DB province values; see unique_city_values.csv")
        else:
            data_source_notes.append(f"city='{args.city}' matched raw province value {match.iloc[0]['raw_city_value']!r} ({int(match.iloc[0]['row_count'])} sale rows).")
    except Exception as exc:
        data_source_notes.append(f"unique-city check failed: {exc}")
        print(f"WARNING: unique-city check failed — {exc}")

    # --- fetch Ankara --------------------------------------------------------
    try:
        ank_sale_raw, meta_s = fetch_city_listings_capped(
            engine, table=args.sale_table, purpose="sale", city=args.city,
            source_site=args.source_site or None, per_county_limit=limit_sale,
        )
        ank_rental_raw, meta_r = fetch_city_listings_capped(
            engine, table=args.rental_table, purpose="rental", city=args.city,
            source_site=args.source_site or None, per_county_limit=limit_rental,
        )
        print(f"Ankara rows fetched: sale={len(ank_sale_raw)} rental={len(ank_rental_raw)}")
    except Exception as exc:
        print(f"ERROR: Ankara fetch failed — {exc}")
        return 4

    ank_sale, sale_price_meta = normalize_city_frame(ank_sale_raw, "sale")
    ank_rental, rental_price_meta = normalize_city_frame(ank_rental_raw, "rental")

    sale_price_ok = bool(sale_price_meta.get("price_available"))
    rental_price_ok = bool(rental_price_meta.get("price_available")) or bool(rental_price_meta.get("unit_price_column_used"))

    sale_mask = _basic_filter_mask(
        ank_sale, "sale", min_sale_unit_price=8000.0, max_sale_unit_price=250000.0,
        min_rent_m2=50.0, max_rent_m2=3000.0, price_available=sale_price_ok,
    )
    rental_mask = _basic_filter_mask(
        ank_rental, "rental", min_sale_unit_price=8000.0, max_sale_unit_price=250000.0,
        min_rent_m2=50.0, max_rent_m2=3000.0, price_available=rental_price_ok,
    )
    ank_sale_f = ank_sale.loc[sale_mask] if sale_mask is not None else ank_sale.copy()
    ank_rental_f = ank_rental.loc[rental_mask] if rental_mask is not None else ank_rental.copy()

    # --- fetch Kocaeli comparison (best-effort; comparison reports degrade gracefully) --
    koc_sale, koc_rental = pd.DataFrame(), pd.DataFrame()
    koc_sale_f, koc_rental_f = pd.DataFrame(), pd.DataFrame()
    try:
        koc_sale_raw, _ = fetch_city_listings_capped(
            engine, table=args.sale_table, purpose="sale", city=args.compare_city,
            source_site=args.source_site or None, per_county_limit=limit_sale,
        )
        koc_rental_raw, _ = fetch_city_listings_capped(
            engine, table=args.rental_table, purpose="rental", city=args.compare_city,
            source_site=args.source_site or None, per_county_limit=limit_rental,
        )
        koc_sale, _ = normalize_city_frame(koc_sale_raw, "sale")
        koc_rental, _ = normalize_city_frame(koc_rental_raw, "rental")
        koc_sale_mask = _basic_filter_mask(koc_sale, "sale", min_sale_unit_price=8000.0, max_sale_unit_price=250000.0, min_rent_m2=50.0, max_rent_m2=3000.0, price_available=True)
        koc_rental_mask = _basic_filter_mask(koc_rental, "rental", min_sale_unit_price=8000.0, max_sale_unit_price=250000.0, min_rent_m2=50.0, max_rent_m2=3000.0, price_available=True)
        koc_sale_f = koc_sale.loc[koc_sale_mask] if koc_sale_mask is not None else koc_sale.copy()
        koc_rental_f = koc_rental.loc[koc_rental_mask] if koc_rental_mask is not None else koc_rental.copy()
        print(f"{args.compare_city} rows fetched (comparison basis, live DB): sale={len(koc_sale)} rental={len(koc_rental)}")
        data_source_notes.append(
            f"Comparison basis for {args.compare_city} is the *live DB inventory* fetched the same way as Ankara "
            "(not the V25 core_good_ok filtered training subset)."
        )
    except Exception as exc:
        print(f"WARNING: {args.compare_city} comparison fetch failed — {exc}")
        data_source_notes.append(f"{args.compare_city} comparison fetch failed: {exc}")

    # --- reports --------------------------------------------------------------
    county_df = build_county_counts(ank_sale, ank_rental, ank_sale_f, ank_rental_f)
    neigh_df = build_neighborhood_counts(ank_sale_f, ank_rental_f)
    price_dist_df = build_price_distribution_by_county(ank_sale_f, ank_rental_f)
    suspicious_df = build_suspicious(ank_sale, ank_rental)
    missingness_df = build_missingness(ank_sale, ank_rental, county_df)
    city_median_sale = float(ank_sale_f["unit_price_gross"].median()) if len(ank_sale_f) else np.nan
    feature_vc_df = build_feature_value_counts(ank_sale_f, ank_rental_f, city_median_sale)
    yerden = yerden_isitma_special(ank_sale_f, ank_rental_f)
    site_cov_df = build_site_project_coverage(ank_sale_f)
    duplex_cov_df = build_duplex_largehome_coverage(ank_sale_f)

    comparison_df = pd.DataFrame()
    stat_tests: dict[str, Any] = {}
    if len(koc_sale) or len(koc_rental):
        comparison_df = build_distribution_comparison(ank_sale_f, ank_rental_f, koc_sale_f, koc_rental_f)
        try:
            stat_tests = build_statistical_tests(ank_sale_f, ank_rental_f, koc_sale_f, koc_rental_f)
        except Exception as exc:
            stat_tests = {"error": str(exc)}
            print(f"WARNING: statistical tests failed — {exc}")

    both_raw = pd.concat([ank_sale, ank_rental], ignore_index=True)
    dup_id = int((both_raw["classified_id"].astype("string").value_counts() > 1).sum()) if "classified_id" in both_raw.columns and len(both_raw) else 0
    dup_url = int((both_raw["source_url"].astype("string").value_counts() > 1).sum()) if "source_url" in both_raw.columns and len(both_raw) else 0

    summary: dict[str, Any] = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "city": args.city,
        "compare_city": args.compare_city,
        "sale_table": meta_s.get("table"),
        "rental_table": meta_r.get("table"),
        "fast_mode": bool(args.fast),
        "limit_sale_per_county": limit_sale,
        "limit_rental_per_county": limit_rental,
        "total_raw": int(len(ank_sale) + len(ank_rental)),
        "sale_raw": int(len(ank_sale)),
        "rental_raw": int(len(ank_rental)),
        "sale_after_basic_filter": int(len(ank_sale_f)),
        "rental_after_basic_filter": int(len(ank_rental_f)),
        "county_count": int(both_raw["county"].nunique()) if len(both_raw) else 0,
        "neighborhood_count": int(both_raw["district"].nunique()) if len(both_raw) else 0,
        "coordinate_coverage_sale": _coverage(ank_sale["has_lat_lon"]) if len(ank_sale) else np.nan,
        "coordinate_coverage_rental": _coverage(ank_rental["has_lat_lon"]) if len(ank_rental) else np.nan,
        "duplicate_classified_id": dup_id,
        "duplicate_source_url": dup_url,
        "suspicious_row_count": int(len(suspicious_df)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_source_notes": data_source_notes,
        "sale_price_meta": sale_price_meta,
        "rental_price_meta": rental_price_meta,
        "output_dir": str(out_dir),
    }
    # convenience aliases used by markdown builders
    summary["sale_after_filter"] = summary["sale_after_basic_filter"]
    summary["rental_after_filter"] = summary["rental_after_basic_filter"]

    training_rec = build_training_recommendation(county_df, comparison_df, stat_tests, summary)

    # --- write everything -------------------------------------------------
    _write_json(summary, out_dir / "inventory_summary_ankara.json")
    _write_csv(county_df, out_dir / "county_counts_ankara.csv")
    _write_csv(neigh_df, out_dir / "neighborhood_counts_ankara.csv")
    _write_csv(price_dist_df, out_dir / "price_distribution_ankara_by_county.csv")

    suspicious_public_summary = (
        suspicious_df.groupby(["purpose", "reason"]).size().reset_index(name="count").sort_values("count", ascending=False)
        if not suspicious_df.empty
        else pd.DataFrame(columns=["purpose", "reason", "count"])
    )
    _write_json(
        {
            "total_suspicious_rows": int(len(suspicious_df)),
            "by_reason": suspicious_public_summary.to_dict(orient="records"),
            "note": "Row-level detail (classified_id/source_url) lives only in the private debug CSV, not here.",
        },
        out_dir / "suspicious_rows_ankara_summary.json",
    )
    if not suspicious_df.empty:
        _write_csv(suspicious_df, out_dir / "suspicious_rows_ankara_private_debug.csv")

    _write_csv(missingness_df, out_dir / "missingness_report_ankara.csv")
    _write_csv(feature_vc_df, out_dir / "feature_value_counts_ankara.csv")
    _write_csv(site_cov_df, out_dir / "site_project_coverage_ankara.csv")
    _write_csv(duplex_cov_df, out_dir / "duplex_largehome_coverage_ankara.csv")
    if not comparison_df.empty:
        _write_csv(comparison_df, out_dir / "ankara_vs_kocaeli_distribution_comparison.csv")
    if stat_tests:
        _write_json(stat_tests, out_dir / "statistical_tests_ankara_vs_kocaeli.json")
    _write_json(training_rec, out_dir / "ankara_model_training_recommendation.json")

    write_model_readiness_md(
        out_dir / "model_readiness_ankara.md",
        summary=summary, county_df=county_df, yerden=yerden, site_cov=site_cov_df,
        duplex_cov=duplex_cov_df, comparison=comparison_df, training_rec=training_rec,
    )
    write_readme(
        out_dir / "README_ankara_inventory_statistical_audit.md",
        summary=summary, county_df=county_df, suspicious_count=int(len(suspicious_df)),
        yerden=yerden, comparison=comparison_df, training_rec=training_rec,
    )

    # --- console summary ----------------------------------------------------
    print(f"reports written -> {out_dir}")
    print(f"Ankara sale after filter={len(ank_sale_f)}, rental after filter={len(ank_rental_f)}")
    print(f"county_count={summary['county_count']}")
    if not county_df.empty:
        vc = county_df["readiness_label"].value_counts()
        print(f"readiness: GOOD={int(vc.get('GOOD', 0))} OK={int(vc.get('OK', 0))} WEAK={int(vc.get('WEAK', 0))}")
        top5 = county_df.sort_values("sale_after_filter", ascending=False).head(5)
        print("top 5 counties by sale count: " + ", ".join(f"{r['county']}={int(r['sale_after_filter'])}" for _, r in top5.iterrows()))
    print(f"coord coverage sale={summary['coordinate_coverage_sale']}, rental={summary['coordinate_coverage_rental']}")
    print(f"city median sale unit price={city_median_sale}")
    print(f"Yerden Isitma: sale={yerden.get('sale_count')} share={yerden.get('sale_share')} lift={yerden.get('city_level_lift')} candidate={yerden.get('candidate_for_model')}")
    if not site_cov_df.empty:
        print(f"site coverage (mean across counties)={site_cov_df['site_name_coverage'].mean():.3f}")
    if not duplex_cov_df.empty:
        tot_sale = int(duplex_cov_df["total_sale_count"].sum()) or 1
        print(f"duplex share={int(duplex_cov_df['duplex_count'].sum())/tot_sale:.3f}, large_home_180 share={int(duplex_cov_df['large_home_180_count'].sum())/tot_sale:.3f}")
    if not comparison_df.empty:
        high = comparison_df[comparison_df["drift_level"] == "high"]["metric"].tolist()
        print(f"strongest drift vs {args.compare_city}: {', '.join(high) if high else '(none flagged high)'}")
    print(f"recommended next training path: {training_rec.get('recommended_first_model')} (ankara_only_baseline_ready={training_rec.get('ankara_only_baseline_ready')}, multi_city_ready={training_rec.get('multi_city_ready')})")
    print("NOTE: no training was run; no model artifacts were produced or modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
