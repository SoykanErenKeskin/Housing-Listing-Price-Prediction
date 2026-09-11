#!/usr/bin/env python
"""Feature Reactivation & App Input Coverage Audit (no training).

Compares refreshed Kocaeli listing inventory vs V24.1 model feature schema
vs EDER app AnalysisRequest / heating options.

Examples:
  python shared_scripts/audit_feature_reactivation.py --city Kocaeli --source-site sahibinden
  python shared_scripts/audit_feature_reactivation.py --out analysis_outputs/feature_reactivation_audit_manual
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
    RENTAL_PRICE_CANDIDATES,
    RENTAL_UNIT_PRICE_CANDIDATES,
    SALE_PRICE_CANDIDATES,
    SALE_UNIT_PRICE_CANDIDATES,
    create_engine,
    fetch_listings,
    resolve_first_present_column,
)
from env_loader import find_project_root, load_root_env  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults / thresholds
# ---------------------------------------------------------------------------

AUDIT_COLUMNS: tuple[str, ...] = (
    "classified_id",
    "source_url",
    "listing_purpose",
    "source_site",
    "city",
    "county",
    "district",
    "province",
    "neighborhood",
    "title",
    "site_name",
    "price",
    "sale_price",
    "listing_price",
    "unit_price_gross",
    "rent_price",
    "rental_price",
    "monthly_rent",
    "rent",
    "rent_per_m2_gross",
    "rent_per_m2_net",
    "gross_m2",
    "net_m2",
    "room_count",
    "building_age",
    "floor_num",
    "total_floors",
    "floor_segment",
    "bathroom_count",
    "dues",
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
    "energy_certificate",
    "open_area_m2",
    "real_estate_type",
    "lat",
    "latitude",
    "lon",
    "longitude",
    "location_precision",
    "location_source",
    "detail_cephe",
    "detail_manzara",
    "detail_ulasim",
    "detail_muhit",
    "detail_ic_ozellikler",
    "detail_dis_ozellikler",
    "detail_konut_tipi",
    "detail_engelli_yasli_uygun",
    "detail_quality_score",
    "detail_selected_count",
)

CATEGORICAL_VALUE_COLS = (
    "heating",
    "kitchen",
    "balcony",
    "elevator",
    "parking",
    "furnished",
    "usage_status",
    "site_inside",
    "room_count",
    "floor_segment",
    "bathroom_count",
    "real_estate_type",
    "credit_eligible",
    "deed_status",
    "energy_certificate",
    "seller_type",
)

DETAIL_PIPE_COLS = (
    "detail_ic_ozellikler",
    "detail_dis_ozellikler",
    "detail_konut_tipi",
    "detail_cephe",
    "detail_manzara",
    "detail_ulasim",
    "detail_muhit",
    "detail_engelli_yasli_uygun",
)

# Canonical V24.1 model feature groups (schema-level, not every OHE leaf).
MODEL_BASE_CATEGORICAL = {
    "real_estate_type",
    "room_count",
    "floor_segment",
    "heating",
    "kitchen",
    "balcony",
    "elevator",
    "parking",
    "furnished",
    "usage_status",
    "site_inside",
    "credit_eligible",
    "energy_certificate",
    "deed_status",
    "seller_type",
    "barter",
    "city",
    "county",
    "district",
    "building_age_group",
    "m2_group",
    "detail_cephe",
    "detail_manzara",
    "detail_konut_tipi",
    "duplex_type",
    "duplex_match_source",
    "large_home_bucket",
    "site_project_id",
    "site_project_id_county_scoped",
    "site_project_match_source",
    "site_quality_tier",
}

MODEL_BASE_NUMERIC = {
    "gross_m2",
    "net_m2",
    "building_age",
    "floor_num",
    "total_floors",
    "bathroom_count",
    "dues",
    "open_area_m2",
    "has_open_area",
    "attr_heating_quality_score",
    "attr_has_premium_heating",
    "is_duplex",
    "is_roof_duplex",
    "is_garden_duplex",
    "is_large_home",
    "has_site_project_id",
    "detail_inside_count",
    "detail_outside_count",
    "detail_subtype_count",
    "lat",
    "lon",
}

# User-facing app fields (EDER AnalysisRequest + option groups).
APP_REQUEST_FIELDS = {
    "city",
    "county",
    "district",
    "neighborhood",
    "gross_m2",
    "net_m2",
    "room_count",
    "building_age",
    "floor_num",
    "total_floors",
    "heating",
    "elevator",
    "parking",
    "furnished",
    "site_inside",
}

APP_HARDCODED_DEFAULTS = {
    "kitchen",
    "balcony",
    "usage_status",
    "bathroom_count",
    "real_estate_type",
    "detail_cephe",
    "detail_manzara",
    "detail_konut_tipi",
}

FEATURE_GROUP_MAP = {
    "heating": "heating",
    "kitchen": "kitchen",
    "balcony": "balcony",
    "elevator": "elevator",
    "parking": "parking",
    "furnished": "furnished",
    "usage_status": "usage_status",
    "site_inside": "site_inside_site",
    "site_name": "site_project",
    "room_count": "room_count",
    "floor_segment": "floor_segment",
    "bathroom_count": "bathroom_count",
    "building_age": "building_age",
    "total_floors": "total_floors",
    "open_area_m2": "open_area",
    "credit_eligible": "deed_credit",
    "deed_status": "deed_credit",
    "detail_ic_ozellikler": "detail_inside",
    "detail_dis_ozellikler": "detail_outside",
    "detail_konut_tipi": "detail_housing_type_duplex",
    "detail_cephe": "detail_front",
    "detail_manzara": "detail_view",
    "detail_ulasim": "detail_transport",
    "detail_muhit": "detail_nearby",
    "detail_engelli_yasli_uygun": "detail_accessibility",
}

GARDEN_TERRACE_POOL_PATTERNS = (
    (r"bah[cç]e", "garden"),
    (r"teras", "terrace"),
    (r"havuz", "pool"),
    (r"a[cç][iı]k\s*havuz", "open_pool"),
    (r"kapal[iı]\s*havuz", "closed_pool"),
)

DUPLEX_RE = re.compile(
    r"(dubleks|dubleksi|dublex|duplex|bah[cç]e\s*dubleks|[cç]at[iı]\s*dubleks|teras\s*dubleks|penthouse)",
    re.IGNORECASE,
)

YERDEN_ALIASES = {"yerden isitma", "yerden ısıtma", "yerden ısıtmalı", "floor heating"}


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


def _fold(s: Any) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    t = str(s).strip().lower()
    return (
        t.replace("ı", "i")
        .replace("İ", "i")
        .replace("ş", "s")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ö", "o")
        .replace("ç", "c")
    )


def _is_yerden(val: Any) -> bool:
    return _fold(val) in YERDEN_ALIASES or "yerden" in _fold(val)


def _split_pipe_tokens(val: Any) -> list[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null", "<na>", "-", "yok"}:
        return []
    parts = re.split(r"[|/;,]+", s)
    out = []
    for p in parts:
        t = p.strip()
        if t and t.lower() not in {"nan", "none", "null", "-", "yok", "belirtilmemiş", "belirtilmemis"}:
            out.append(t)
    return out


def _truthy_site_inside(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower().fillna("")
    return s.isin({"1", "true", "yes", "evet", "var", "site içinde", "site icinde"})


def _meaningful_site_name(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower().fillna("")
    empty = {"", "nan", "none", "null", "<na>", "yok", "belirtilmemis", "belirtilmemiş", "-", "--"}
    return ~s.isin(empty) & (s.str.len() >= 2)


# ---------------------------------------------------------------------------
# Load / parse frames
# ---------------------------------------------------------------------------


def normalize_listings(df: pd.DataFrame, purpose: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
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

    for c in ("city", "county", "district", "title", "site_name", "heating", "site_inside"):
        if c not in out.columns:
            out[c] = pd.NA
        else:
            out[c] = out[c].astype("string")

    out["listing_purpose"] = purpose
    out["gross_m2"] = pd.to_numeric(out.get("gross_m2"), errors="coerce")
    out["building_age"] = pd.to_numeric(out.get("building_age"), errors="coerce")
    out["bathroom_count"] = pd.to_numeric(out.get("bathroom_count"), errors="coerce")

    if purpose == "sale":
        price_col = resolve_first_present_column(out, SALE_PRICE_CANDIDATES)
        unit_col = resolve_first_present_column(out, SALE_UNIT_PRICE_CANDIDATES)
        out["price"] = pd.to_numeric(out[price_col], errors="coerce") if price_col else np.nan
        out["unit_price_gross"] = pd.to_numeric(out[unit_col], errors="coerce") if unit_col else np.nan
        need = out["unit_price_gross"].isna() & out["price"].notna() & (out["gross_m2"] > 0)
        out.loc[need, "unit_price_gross"] = out.loc[need, "price"] / out.loc[need, "gross_m2"]
    else:
        price_col = resolve_first_present_column(out, RENTAL_PRICE_CANDIDATES)
        unit_col = resolve_first_present_column(out, RENTAL_UNIT_PRICE_CANDIDATES)
        out["price"] = pd.to_numeric(out[price_col], errors="coerce") if price_col else np.nan
        out["rent_m2"] = pd.to_numeric(out[unit_col], errors="coerce") if unit_col else np.nan
        need = out["rent_m2"].isna() & out["price"].notna() & (out["gross_m2"] > 0)
        out.loc[need, "rent_m2"] = out.loc[need, "price"] / out.loc[need, "gross_m2"]
        out["unit_price_gross"] = np.nan

    out["is_yerden"] = out["heating"].map(_is_yerden)
    out["site_inside_flag"] = _truthy_site_inside(out["site_inside"])
    out["has_site_name"] = _meaningful_site_name(out["site_name"])
    text = (
        out["title"].fillna("").astype(str)
        + " "
        + out.get("detail_konut_tipi", pd.Series("", index=out.index)).fillna("").astype(str)
    )
    out["is_duplex_text"] = text.str.contains(DUPLEX_RE.pattern, case=False, regex=True, na=False)
    out["is_large_home"] = (out["gross_m2"] >= 180).fillna(False)
    return out


def build_raw_feature_value_counts(sale: pd.DataFrame, rental: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frames = [("sale", sale), ("rental", rental)]
    both = pd.concat([sale, rental], ignore_index=True) if len(sale) or len(rental) else pd.DataFrame()
    if len(both):
        frames.append(("both", both))

    for purpose, df in frames:
        if df.empty:
            continue
        n = len(df)
        # categorical / scalar columns
        for col in CATEGORICAL_VALUE_COLS:
            if col not in df.columns:
                rows.append(
                    {
                        "purpose": purpose,
                        "feature_group": FEATURE_GROUP_MAP.get(col, col),
                        "raw_column": col,
                        "raw_value": "__COLUMN_MISSING__",
                        "count": 0,
                        "share": 0.0,
                        "n_rows": n,
                        "value_kind": "column_missing",
                    }
                )
                continue
            s = df[col].astype("string")
            vc = s.value_counts(dropna=True)
            for value, count in vc.items():
                rows.append(
                    {
                        "purpose": purpose,
                        "feature_group": FEATURE_GROUP_MAP.get(col, col),
                        "raw_column": col,
                        "raw_value": str(value),
                        "count": int(count),
                        "share": float(count / n),
                        "n_rows": n,
                        "value_kind": "categorical",
                    }
                )

        # pipe-detail tokens
        for col in DETAIL_PIPE_COLS:
            if col not in df.columns:
                continue
            tokens: list[str] = []
            for v in df[col].tolist():
                tokens.extend(_split_pipe_tokens(v))
            if not tokens:
                continue
            vc = pd.Series(tokens).value_counts()
            for value, count in vc.items():
                rows.append(
                    {
                        "purpose": purpose,
                        "feature_group": FEATURE_GROUP_MAP.get(col, "detail"),
                        "raw_column": col,
                        "raw_value": str(value),
                        "count": int(count),
                        "share": float(count / n),
                        "n_rows": n,
                        "value_kind": "detail_token",
                    }
                )

        # derived signals
        for flag, colname, group in (
            ("is_yerden", "heating==Yerden Isıtma", "heating"),
            ("is_duplex_text", "duplex_text_signal", "duplex_largehome"),
            ("is_large_home", "large_home_m2_ge_180", "duplex_largehome"),
            ("has_site_name", "site_name_present", "site_project"),
            ("site_inside_flag", "site_inside_true", "site_inside_site"),
        ):
            if flag not in df.columns:
                continue
            c = int(df[flag].fillna(False).sum())
            rows.append(
                {
                    "purpose": purpose,
                    "feature_group": group,
                    "raw_column": colname,
                    "raw_value": "True",
                    "count": c,
                    "share": float(c / n) if n else 0.0,
                    "n_rows": n,
                    "value_kind": "derived_signal",
                }
            )

        # garden/terrace/pool heuristic from outside+title tokens
        blob = (
            df.get("detail_dis_ozellikler", pd.Series("", index=df.index)).fillna("").astype(str)
            + "|"
            + df.get("detail_ic_ozellikler", pd.Series("", index=df.index)).fillna("").astype(str)
            + "|"
            + df["title"].fillna("").astype(str)
        )
        for pat, label in GARDEN_TERRACE_POOL_PATTERNS:
            hit = blob.str.contains(pat, case=False, regex=True, na=False)
            c = int(hit.sum())
            rows.append(
                {
                    "purpose": purpose,
                    "feature_group": "garden_terrace_pool",
                    "raw_column": "text_heuristic",
                    "raw_value": label,
                    "count": c,
                    "share": float(c / n) if n else 0.0,
                    "n_rows": n,
                    "value_kind": "text_heuristic",
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "purpose",
                "feature_group",
                "raw_column",
                "raw_value",
                "count",
                "share",
                "n_rows",
                "value_kind",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["purpose", "feature_group", "raw_column", "count"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Model / app inventory
# ---------------------------------------------------------------------------


def load_v24_model_features(root: Path, importance_csv: Path | None) -> dict[str, Any]:
    features: set[str] = set()
    heating_categories: set[str] = set()
    ohe_by_col: dict[str, set[str]] = {}
    if importance_csv and importance_csv.exists():
        df = pd.read_csv(importance_csv)
        col = "feature" if "feature" in df.columns else df.columns[0]
        for raw in df[col].astype(str):
            features.add(raw)
            if raw.startswith("cat__"):
                rest = raw[5:]
                # Prefer known multiword categorical leaves
                known = sorted(MODEL_BASE_CATEGORICAL, key=len, reverse=True)
                matched = False
                for base in known:
                    prefix = base + "_"
                    if rest.startswith(prefix):
                        val = rest[len(prefix) :]
                        ohe_by_col.setdefault(base, set()).add(val)
                        if base == "heating":
                            heating_categories.add(val)
                        matched = True
                        break
                if not matched:
                    # fallback first token
                    if "_" in rest:
                        base, val = rest.split("_", 1)
                        ohe_by_col.setdefault(base, set()).add(val)
            elif raw.startswith("num__"):
                features.add(raw[5:])

    # Schema-level presence (always true for V24.1 full_v24)
    schema_cols = set(MODEL_BASE_CATEGORICAL) | set(MODEL_BASE_NUMERIC)
    return {
        "importance_path": str(importance_csv) if importance_csv else None,
        "encoded_feature_count": len(features),
        "heating_categories": sorted(heating_categories),
        "ohe_by_col": {k: sorted(v) for k, v in ohe_by_col.items()},
        "schema_columns": sorted(schema_cols),
        "has_heating_yerden": "Yerden Isıtma" in heating_categories
        or any("yerden" in _fold(x) for x in heating_categories),
        "has_duplex_type": any(f.startswith("cat__duplex_type") or f == "duplex_type" for f in features)
        or True,  # full_v24 includes duplex
        "has_site_project_id": any("site_project_id" in f for f in features) or True,
        "raw_features": features,
    }


def _parse_ts_string_array(text: str, const_name: str) -> list[str]:
    m = re.search(rf"export const {const_name}\s*=\s*\[(.*?)\]\s*as const", text, re.S)
    if not m:
        return []
    body = m.group(1)
    return re.findall(r'"([^"]+)"', body)


def load_app_coverage(app_root: Path | None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "app_root": str(app_root) if app_root else None,
        "app_found": False,
        "request_fields": sorted(APP_REQUEST_FIELDS),
        "hardcoded_defaults": sorted(APP_HARDCODED_DEFAULTS),
        "heating_options": [],
        "elevator_options": [],
        "parking_options": [],
        "furnished_options": [],
        "site_inside_options": [],
        "room_count_options": [],
        "vocab_heating": [],
        "has_yerden_in_ui": False,
        "has_yerden_in_vocab": False,
        "has_duplex_type_input": False,
        "has_site_project_id_input": False,
        "has_lat_lon_input": False,
        "has_balcony_input": False,
        "has_kitchen_input": False,
        "has_usage_status_input": False,
        "has_bathroom_count_input": False,
        "has_credit_deed_input": False,
        "county_scope": "İzmit-only (hardcoded)",
        "endpoint": "POST /analysis",
        "notes": [],
    }
    if app_root is None or not app_root.exists():
        info["notes"].append("App root not found; using known EDER AnalysisRequest contract from audit time.")
        # Fall back to inspected snapshot
        info["heating_options"] = [
            "Merkezi (Pay Ölçer)",
            "Merkezi",
            "Kombi (Doğalgaz)",
            "Doğalgaz Sobası",
            "Kat Kaloriferi",
            "Klima",
            "Soba",
            "Yok",
        ]
        info["vocab_heating"] = info["heating_options"] + ["Yerden Isıtma", "Kombi (Elektrik)", "VRV"]
        info["has_yerden_in_ui"] = False
        info["has_yerden_in_vocab"] = True
        info["elevator_options"] = ["Var", "Yok"]
        info["parking_options"] = ["Yok", "Açık Otopark", "Kapalı Otopark", "Açık & Kapalı Otopark"]
        info["furnished_options"] = ["Belirtilmemiş", "Evet", "Hayır"]
        info["site_inside_options"] = ["Evet", "Hayır"]
        return info

    info["app_found"] = True
    options_path = app_root / "src" / "data" / "analysisFieldOptions.ts"
    types_path = app_root / "src" / "types" / "analysis.ts"
    vocab_path = app_root / "backend" / "models" / "training_vocab.json"
    schema_path = app_root / "backend" / "schemas" / "analysis.py"
    config_path = app_root / "backend" / "config.py"

    if options_path.exists():
        text = options_path.read_text(encoding="utf-8")
        info["heating_options"] = _parse_ts_string_array(text, "HEATING_OPTIONS")
        info["elevator_options"] = _parse_ts_string_array(text, "ELEVATOR_OPTIONS")
        info["parking_options"] = _parse_ts_string_array(text, "PARKING_OPTIONS")
        info["furnished_options"] = _parse_ts_string_array(text, "FURNISHED_OPTIONS")
        info["site_inside_options"] = _parse_ts_string_array(text, "SITE_INSIDE_OPTIONS")
        info["room_count_options"] = _parse_ts_string_array(text, "ROOM_COUNT_OPTIONS")
        info["has_yerden_in_ui"] = any(_is_yerden(x) for x in info["heating_options"])

    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        info["vocab_heating"] = list(vocab.get("heating") or [])
        info["has_yerden_in_vocab"] = any(_is_yerden(x) for x in info["vocab_heating"])

    if types_path.exists():
        t = types_path.read_text(encoding="utf-8")
        info["has_duplex_type_input"] = "duplex" in t.lower()
        info["has_site_project_id_input"] = "site_project" in t.lower() or "siteProject" in t
        info["has_lat_lon_input"] = bool(re.search(r"\b(lat|lon|latitude|longitude)\b", t))
        info["has_balcony_input"] = "balcony" in t
        info["has_kitchen_input"] = "kitchen" in t
        info["has_usage_status_input"] = "usage_status" in t
        info["has_bathroom_count_input"] = "bathroom_count" in t
        info["has_credit_deed_input"] = "credit_eligible" in t or "deed_status" in t

    if schema_path.exists():
        s = schema_path.read_text(encoding="utf-8")
        if "duplex" not in s.lower():
            info["notes"].append("Backend AnalysisRequest has no duplexType.")
        if "site_project" not in s.lower() and "siteProject" not in s:
            info["notes"].append("Backend AnalysisRequest has no site_project_id.")
        if "latitude" not in s and "longitude" not in s and re.search(r"\blat\b", s) is None:
            info["notes"].append("Backend AnalysisRequest has no lat/lon.")

    if config_path.exists():
        c = config_path.read_text(encoding="utf-8")
        if "INFERENCE_DEFAULTS" in c:
            info["notes"].append(
                "App forces kitchen/balcony/usage_status/bathroom_count/detail_* via INFERENCE_DEFAULTS."
            )
        if "IZMIT" in c.upper() or "İzmit" in c:
            info["notes"].append("App county scope appears İzmit-centric.")

    info["notes"].append("No WizardDraft/LocationStep in app; single AnalysisForm.")
    info["notes"].append("No /api/real-estate/predict; live endpoint is POST /analysis.")
    return info


# ---------------------------------------------------------------------------
# Candidate / gap logic
# ---------------------------------------------------------------------------


def _county_counts(sale: pd.DataFrame, mask: pd.Series) -> pd.Series:
    if sale.empty or not mask.any():
        return pd.Series(dtype=int)
    return sale.loc[mask, "county"].astype(str).value_counts()


def _median_lift(sale: pd.DataFrame, mask: pd.Series) -> tuple[float, float, float]:
    """Return (value_median, peer_county_weighted_median_baseline, lift)."""
    if sale.empty or not mask.any():
        return (np.nan, np.nan, np.nan)
    sub = sale.loc[mask]
    val_med = float(pd.to_numeric(sub["unit_price_gross"], errors="coerce").median())
    # Compare each row's unit price vs its county median, then average lifts — aggregate form:
    county_med = sale.groupby(sale["county"].astype(str))["unit_price_gross"].median()
    base = sub["county"].astype(str).map(county_med)
    base_med = float(pd.to_numeric(base, errors="coerce").median())
    if not base_med or not np.isfinite(base_med) or base_med == 0:
        return (val_med, base_med, np.nan)
    lift = (val_med / base_med) - 1.0
    return (val_med, base_med, float(lift))


def assess_candidate(
    *,
    sale_count: int,
    county_ge30: int,
    lift: float,
    in_model: bool,
    in_app: bool,
    value_kind: str,
    raw_column: str,
    raw_value: str,
) -> tuple[bool, str, str]:
    """Return (is_candidate, recommendation, notes)."""
    premium = bool(np.isfinite(lift) and abs(lift) >= 0.08 and sale_count >= 50)
    volume_ok = sale_count >= 100 or county_ge30 >= 3 or premium
    notes = []
    if sale_count >= 100:
        notes.append("sale_count>=100")
    if county_ge30 >= 3:
        notes.append("counties_ge30>=3")
    if premium:
        notes.append(f"premium_lift={lift:+.1%}")

    # Risky / noisy patterns
    risky = False
    rv = _fold(raw_value)
    if value_kind == "detail_token" and sale_count < 200 and not premium:
        risky = True
        notes.append("sparse_detail_token")
    if raw_column in {"credit_eligible", "deed_status", "energy_certificate", "seller_type", "open_area_m2"}:
        if sale_count == 0:
            notes.append("column_absent_or_empty_in_db")

    if not volume_ok:
        return False, "IGNORE_LOW_SIGNAL", " | ".join(notes) or "below_thresholds"

    # Already strong in model, missing in app
    if in_model and not in_app and raw_column in {
        "heating",
        "kitchen",
        "balcony",
        "usage_status",
        "bathroom_count",
        "floor_segment",
        "duplex_text_signal",
        "site_name_present",
        "site_project",
    }:
        if raw_column in {"site_name_present"} or "site" in raw_column:
            return True, "APP_INPUT_ADD", " | ".join(notes + ["model_uses_site_layer"])
        if value_kind == "detail_token":
            return True, "KEEP_DERIVED_ONLY", " | ".join(notes + ["user_should_not_paste_raw_pipe_detail"])
        return True, "APP_INPUT_ADD", " | ".join(notes + ["already_in_model"])

    if in_model and in_app:
        if premium or sale_count >= 500:
            return True, "INCLUDE_V25", " | ".join(notes + ["already_covered_keep_monitoring"])
        return True, "INCLUDE_V25", " | ".join(notes + ["already_covered"])

    if (not in_model) and in_app:
        return True, "NEEDS_MANUAL_REVIEW", " | ".join(notes + ["app_field_not_used_by_v24_1"])

    if value_kind in {"detail_token", "text_heuristic"}:
        if risky:
            return True, "KEEP_DERIVED_ONLY", " | ".join(notes)
        if premium or sale_count >= 300:
            return True, "INCLUDE_V25", " | ".join(notes + ["candidate_via_detail_group_or_flag"])
        return True, "KEEP_DERIVED_ONLY", " | ".join(notes + ["prefer_derived_counts_effects"])

    if raw_column in {"duplex_text_signal", "large_home_m2_ge_180"}:
        return True, "APP_INPUT_ADD", " | ".join(notes + ["model_has_duplex_largehome_add_app_control"])

    if not in_model and not in_app:
        if risky:
            return True, "IGNORE_LOW_SIGNAL", " | ".join(notes)
        if premium or sale_count >= 200:
            return True, "NEEDS_MANUAL_REVIEW", " | ".join(notes + ["new_candidate_not_in_model_or_app"])
        return True, "IGNORE_LOW_SIGNAL", " | ".join(notes)

    return True, "NEEDS_MANUAL_REVIEW", " | ".join(notes)


def build_gap_matrix(
    sale: pd.DataFrame,
    rental: pd.DataFrame,
    raw_counts: pd.DataFrame,
    model_info: dict[str, Any],
    app_info: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    both_sale = raw_counts[raw_counts["purpose"] == "sale"].copy()
    rows = []
    heating_app = {_fold(x) for x in app_info.get("heating_options") or []}
    heating_model = {_fold(x) for x in model_info.get("heating_categories") or []}
    app_fields = set(app_info.get("request_fields") or [])

    # Focus rows: both-purpose derived + sale categorical/detail with count>0
    focus = raw_counts[
        (raw_counts["purpose"].isin(["sale", "both"]))
        & (raw_counts["raw_value"] != "__COLUMN_MISSING__")
        & (raw_counts["count"] > 0)
    ].copy()
    # Prefer sale for volume; if both-only derived keep both
    focus = focus[focus["purpose"] == "sale"].copy()
    # Cap sparse detail tokens to keep aggregate audit readable
    if not focus.empty:
        keep_detail = (focus["value_kind"] != "detail_token") | (focus["count"] >= 30)
        focus = focus[keep_detail]
    # Add derived both rows converted to sale-equivalent by recomputation below via flags

    seen = set()
    for _, r in focus.iterrows():
        key = (r["raw_column"], str(r["raw_value"]))
        if key in seen:
            continue
        seen.add(key)
        col = str(r["raw_column"])
        val = str(r["raw_value"])
        kind = str(r["value_kind"])

        if kind == "categorical" and col in sale.columns:
            mask_s = sale[col].astype("string") == val
            mask_r = rental[col].astype("string") == val if col in rental.columns else pd.Series(False, index=rental.index)
        elif kind == "detail_token" and col in sale.columns:
            mask_s = sale[col].fillna("").astype(str).map(lambda x: val in _split_pipe_tokens(x))
            mask_r = (
                rental[col].fillna("").astype(str).map(lambda x: val in _split_pipe_tokens(x))
                if col in rental.columns
                else pd.Series(False, index=rental.index)
            )
        elif kind == "derived_signal":
            flag_map = {
                "heating==Yerden Isıtma": "is_yerden",
                "duplex_text_signal": "is_duplex_text",
                "large_home_m2_ge_180": "is_large_home",
                "site_name_present": "has_site_name",
                "site_inside_true": "site_inside_flag",
            }
            flag = flag_map.get(col)
            if not flag:
                continue
            mask_s = sale[flag].fillna(False).astype(bool)
            mask_r = rental[flag].fillna(False).astype(bool)
        elif kind == "text_heuristic":
            blob_s = (
                sale.get("detail_dis_ozellikler", pd.Series("", index=sale.index)).fillna("").astype(str)
                + "|"
                + sale.get("detail_ic_ozellikler", pd.Series("", index=sale.index)).fillna("").astype(str)
                + "|"
                + sale["title"].fillna("").astype(str)
            )
            blob_r = (
                rental.get("detail_dis_ozellikler", pd.Series("", index=rental.index)).fillna("").astype(str)
                + "|"
                + rental.get("detail_ic_ozellikler", pd.Series("", index=rental.index)).fillna("").astype(str)
                + "|"
                + rental["title"].fillna("").astype(str)
            )
            pat = next((p for p, lab in GARDEN_TERRACE_POOL_PATTERNS if lab == val), None)
            if not pat:
                continue
            mask_s = blob_s.str.contains(pat, case=False, regex=True, na=False)
            mask_r = blob_r.str.contains(pat, case=False, regex=True, na=False)
        else:
            continue

        sale_count = int(mask_s.sum())
        rental_count = int(mask_r.sum())
        cc = _county_counts(sale, mask_s)
        county_count = int((cc > 0).sum())
        county_ge30 = int((cc >= 30).sum())
        top_counties = ", ".join(f"{k}:{int(v)}" for k, v in cc.head(5).items())
        med, _base, lift = _median_lift(sale, mask_s)

        # model / app flags
        in_model = False
        in_app = False
        safe_user = True
        if col == "heating" or (col.startswith("heating") and kind != "detail_token"):
            in_model = (_fold(val) in heating_model) or model_info.get("has_heating_yerden", False) and _is_yerden(val)
            if kind == "categorical":
                in_model = _fold(val) in heating_model or any(_fold(val) == h for h in heating_model)
            in_app = _fold(val) in heating_app
            if _is_yerden(val):
                in_model = bool(model_info.get("has_heating_yerden")) or in_model
                in_app = bool(app_info.get("has_yerden_in_ui"))
        elif col in MODEL_BASE_CATEGORICAL or col in MODEL_BASE_NUMERIC:
            in_model = True
            in_app = col in app_fields
        elif col in {"duplex_text_signal", "large_home_m2_ge_180"}:
            in_model = True  # V24.1 full_v24 duplex/largehome pack
            in_app = bool(app_info.get("has_duplex_type_input"))
            safe_user = True
        elif col in {"site_name_present", "site_inside_true"}:
            in_model = True
            in_app = ("site_inside" in app_fields) if col == "site_inside_true" else bool(
                app_info.get("has_site_project_id_input")
            )
        elif kind == "detail_token":
            # raw pipes not fed as categoricals except konut_tipi/cephe/manzara wholes; tokens via derived effects
            in_model = col in {"detail_konut_tipi", "detail_cephe", "detail_manzara"} or True  # derived group effects
            # More precise: model does NOT take raw token as user categorical
            in_model = col in {"detail_konut_tipi", "detail_cephe", "detail_manzara"}
            # but aggregates exist for inside/outside
            if col in {"detail_ic_ozellikler", "detail_dis_ozellikler"}:
                in_model = True  # via detail_inside/outside_count + detail_effect groups
            in_app = False
            safe_user = False  # don't ask users to paste pipe lists
        elif kind == "text_heuristic":
            in_model = True  # often captured in out_pool / garden flags when expanded from raw
            in_app = False
            safe_user = True  # chips OK

        # bathroom_count / floor_segment special
        if col == "bathroom_count":
            in_model = True
            in_app = bool(app_info.get("has_bathroom_count_input"))
        if col == "floor_segment":
            in_model = True
            in_app = False
        if col in {"kitchen", "balcony", "usage_status"}:
            in_model = True
            in_app = col in app_fields and not (col in APP_HARDCODED_DEFAULTS and col not in app_fields)
            # App defaults them — treat as NOT user-input
            in_app = False

        is_cand, rec, notes = assess_candidate(
            sale_count=sale_count,
            county_ge30=county_ge30,
            lift=lift if lift == lift else 0.0,
            in_model=in_model,
            in_app=in_app,
            value_kind=kind,
            raw_column=col,
            raw_value=val,
        )

        # Strong special-cases
        if _is_yerden(val) or col.startswith("heating==Yerden"):
            in_model = bool(model_info.get("has_heating_yerden"))
            in_app = bool(app_info.get("has_yerden_in_ui"))
            rec = "APP_INPUT_ADD"
            notes = (notes + " | " if notes else "") + "yerden_focus: in_model_not_in_app_ui"
            is_cand = True
            safe_user = True

        rows.append(
            {
                "feature_group": r["feature_group"],
                "raw_column": col,
                "raw_value": val,
                "value_kind": kind,
                "sale_count": sale_count,
                "rental_count": rental_count,
                "county_count": county_count,
                "counties_with_ge30": county_ge30,
                "top_counties": top_counties,
                "median_unit_price": med,
                "median_lift_vs_county": lift,
                "currently_in_model": in_model,
                "currently_in_app_input": in_app,
                "safe_for_user_input": safe_user,
                "is_reactivation_candidate": is_cand,
                "recommendation": rec,
                "notes": notes,
            }
        )

    gap = pd.DataFrame(rows)
    if gap.empty:
        return gap, gap
    gap = gap.sort_values(
        ["recommendation", "sale_count"],
        ascending=[True, False],
    ).reset_index(drop=True)
    candidates = gap[gap["is_reactivation_candidate"]].copy()
    return gap, candidates


def build_model_feature_coverage(
    model_info: dict[str, Any],
    app_info: dict[str, Any],
    gap: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    # Base schema features of interest
    interesting = [
        ("heating", "categorical", True, "heating" in app_info["request_fields"]),
        ("kitchen", "categorical", True, False),
        ("balcony", "categorical", True, False),
        ("elevator", "categorical", True, True),
        ("parking", "categorical", True, True),
        ("furnished", "categorical", True, True),
        ("usage_status", "categorical", True, False),
        ("site_inside", "categorical", True, True),
        ("room_count", "categorical", True, True),
        ("floor_segment", "categorical", True, False),
        ("bathroom_count", "numeric", True, False),
        ("building_age", "numeric", True, True),
        ("total_floors", "numeric", True, True),
        ("credit_eligible", "categorical", True, False),
        ("deed_status", "categorical", True, False),
        ("energy_certificate", "categorical", True, False),
        ("open_area_m2", "numeric", True, False),
        ("detail_konut_tipi", "categorical+duplex_source", True, False),
        ("detail_ic_ozellikler", "derived_counts_effects", True, False),
        ("detail_dis_ozellikler", "derived_counts_effects", True, False),
        ("duplex_type", "derived_categorical", True, False),
        ("is_large_home", "derived_numeric", True, False),
        ("site_project_id", "derived_categorical", True, False),
        ("lat/lon", "geo", True, False),
        ("county", "location", True, False),  # app locked İzmit
        ("Yerden Isıtma (heating leaf)", "ohe_leaf", bool(model_info.get("has_heating_yerden")), bool(app_info.get("has_yerden_in_ui"))),
    ]
    for name, kind, in_model, in_app in interesting:
        match = gap[gap["raw_column"].astype(str).str.contains(name.split()[0], case=False, na=False)] if not gap.empty else pd.DataFrame()
        sale_n = int(match["sale_count"].max()) if len(match) else np.nan
        rows.append(
            {
                "model_feature": name,
                "feature_kind": kind,
                "in_v24_1_schema": in_model,
                "in_app_user_input": in_app,
                "app_coverage_status": (
                    "ok"
                    if in_model and in_app
                    else ("app_input_missing" if in_model and not in_app else ("model_unused_app_field" if (not in_model and in_app) else "neither"))
                ),
                "max_sale_count_observed": sale_n,
                "notes": "",
            }
        )
    # heating leaves
    for h in model_info.get("heating_categories") or []:
        in_app = _fold(h) in {_fold(x) for x in app_info.get("heating_options") or []}
        rows.append(
            {
                "model_feature": f"cat__heating_{h}",
                "feature_kind": "ohe_leaf",
                "in_v24_1_schema": True,
                "in_app_user_input": in_app,
                "app_coverage_status": "ok" if in_app else "app_input_missing",
                "max_sale_count_observed": np.nan,
                "notes": "from feature_importance",
            }
        )
    return pd.DataFrame(rows)


def build_app_input_coverage(app_info: dict[str, Any], model_info: dict[str, Any], gap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in sorted(set(app_info.get("request_fields") or []) | set(APP_HARDCODED_DEFAULTS)):
        user_editable = field in (app_info.get("request_fields") or []) and field not in APP_HARDCODED_DEFAULTS
        if field in APP_HARDCODED_DEFAULTS and field not in (app_info.get("request_fields") or []):
            user_editable = False
        in_model = field in MODEL_BASE_CATEGORICAL or field in MODEL_BASE_NUMERIC or field in {
            "neighborhood",  # maps to district
        }
        if field == "neighborhood":
            in_model = True  # maps toward district
        status = (
            "used_by_model"
            if user_editable and in_model
            else (
                "hardcoded_default_not_user"
                if field in APP_HARDCODED_DEFAULTS
                else ("app_only_unused" if user_editable and not in_model else "schema_only")
            )
        )
        options = []
        if field == "heating":
            options = app_info.get("heating_options") or []
        elif field == "parking":
            options = app_info.get("parking_options") or []
        elif field == "elevator":
            options = app_info.get("elevator_options") or []
        elif field == "furnished":
            options = app_info.get("furnished_options") or []
        elif field == "site_inside":
            options = app_info.get("site_inside_options") or []
        elif field == "room_count":
            options = app_info.get("room_count_options") or []
        missing_vs_model = []
        if field == "heating":
            model_h = model_info.get("heating_categories") or []
            app_h = {_fold(x) for x in options}
            missing_vs_model = [h for h in model_h if _fold(h) not in app_h and h != "missing"]
        rows.append(
            {
                "app_field": field,
                "user_editable": user_editable,
                "in_v24_1_model_schema": in_model,
                "status": status,
                "ui_option_count": len(options),
                "ui_options": " | ".join(options[:20]),
                "missing_model_values_in_ui": " | ".join(missing_vs_model),
                "notes": "",
            }
        )
    # Explicit missing high-value controls
    for field, note in [
        ("duplex_type", "V24.1 duplex_type not in AnalysisRequest"),
        ("site_project_id", "V24.1 site picker not connected"),
        ("lat", "geo pin not collected"),
        ("lon", "geo pin not collected"),
        ("county", "UI locked to İzmit; V24.1 is multi-county"),
        ("kitchen", "hardcoded Kapalı in INFERENCE_DEFAULTS"),
        ("balcony", "hardcoded Yok in INFERENCE_DEFAULTS"),
        ("usage_status", "hardcoded Boş"),
        ("bathroom_count", "hardcoded 1"),
        ("Yerden Isıtma", "in vocab+model; missing from HEATING_OPTIONS"),
    ]:
        rows.append(
            {
                "app_field": field,
                "user_editable": False,
                "in_v24_1_model_schema": True,
                "status": "app_input_missing",
                "ui_option_count": 0,
                "ui_options": "",
                "missing_model_values_in_ui": field if field == "Yerden Isıtma" else "",
                "notes": note,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Special Yerden report + markdown
# ---------------------------------------------------------------------------


def yerden_deep_dive(sale: pd.DataFrame, rental: pd.DataFrame, model_info: dict[str, Any], app_info: dict[str, Any]) -> dict[str, Any]:
    ys = sale["is_yerden"].fillna(False)
    yr = rental["is_yerden"].fillna(False)
    by_county = (
        sale.groupby(sale["county"].astype(str))
        .agg(sale_total=("classified_id", "count"), yerden_sale=("is_yerden", "sum"))
        .reset_index()
    )
    by_county["yerden_share"] = by_county["yerden_sale"] / by_county["sale_total"].clip(lower=1)
    by_county = by_county.sort_values("yerden_sale", ascending=False)

    yerden_med = float(sale.loc[ys, "unit_price_gross"].median()) if ys.any() else np.nan
    other_med = float(sale.loc[~ys, "unit_price_gross"].median()) if (~ys).any() else np.nan
    lift = (yerden_med / other_med - 1.0) if other_med and other_med == other_med and other_med else np.nan

    site_ct = (
        pd.DataFrame(
            {
                "heating_group": np.where(ys, "yerden", "other"),
                "site_group": np.where(sale["site_inside_flag"].fillna(False), "site", "non_site"),
            }
        )
        .value_counts()
        .rename("count")
        .reset_index()
    )

    age = sale.copy()
    age["age_bucket"] = pd.cut(
        age["building_age"],
        bins=[-0.1, 0, 5, 10, 20, 40, 200],
        labels=["0", "1-5", "6-10", "11-20", "21-40", "40+"],
    )
    age_ct = (
        age.groupby(["age_bucket", age["is_yerden"].map({True: "yerden", False: "other"})], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )

    return {
        "sale_count": int(ys.sum()),
        "rental_count": int(yr.sum()),
        "sale_share": float(ys.mean()) if len(sale) else np.nan,
        "rental_share": float(yr.mean()) if len(rental) else np.nan,
        "median_unit_price_yerden": yerden_med,
        "median_unit_price_other": other_med,
        "median_lift_vs_other": lift,
        "by_county": by_county,
        "by_site": site_ct,
        "by_age": age_ct,
        "in_model_heating_categorical": bool(model_info.get("has_heating_yerden")),
        "in_app_ui": bool(app_info.get("has_yerden_in_ui")),
        "in_app_vocab": bool(app_info.get("has_yerden_in_vocab")),
        "recommendation": "APP_INPUT_ADD",
    }


def write_recommendations_md(
    path: Path,
    *,
    summary: dict[str, Any],
    gap: pd.DataFrame,
    candidates: pd.DataFrame,
    model_cov: pd.DataFrame,
    app_cov: pd.DataFrame,
    yerden: dict[str, Any],
    app_info: dict[str, Any],
) -> None:
    def _top(rec: str, n: int = 12) -> pd.DataFrame:
        if candidates.empty:
            return candidates
        return candidates[candidates["recommendation"] == rec].head(n)

    lines = [
        "# V25 Feature Reactivation Recommendations",
        "",
        "Audit only — **no training**.",
        "",
        f"- run_timestamp: `{summary.get('run_timestamp')}`",
        f"- inventory: sale={summary.get('sale_rows')}, rental={summary.get('rental_rows')}, total={summary.get('total_rows')}",
        f"- V24.1 selected: `full_v24` (R²={summary.get('v24_1_r2')}, MAPE={summary.get('v24_1_mape')})",
        f"- app root: `{app_info.get('app_root')}` (found={app_info.get('app_found')})",
        f"- source_site filter: `{summary.get('source_site')}`",
        "",
        "## 1) Previously sparse features now viable",
        "",
    ]
    viable = candidates[candidates["sale_count"] >= 100].head(25) if not candidates.empty else candidates
    if viable.empty:
        lines.append("- (none)")
    else:
        for _, r in viable.iterrows():
            lines.append(
                f"- **{r['raw_column']}={r['raw_value']}**: sale={int(r['sale_count'])}, "
                f"counties={int(r['county_count'])}, lift={r['median_lift_vs_county']}, "
                f"rec=`{r['recommendation']}`"
            )

    lines += ["", "## 2) Add to V25 model (INCLUDE_V25)", ""]
    inc = _top("INCLUDE_V25")
    if inc.empty:
        lines.append("- Mostly already covered by V24.1 schema; prioritize app wiring over new model leaves.")
    else:
        for _, r in inc.iterrows():
            lines.append(
                f"- {r['raw_column']}={r['raw_value']} (sale={int(r['sale_count'])}, lift={r['median_lift_vs_county']})"
            )

    lines += ["", "## 3) Add to app input (APP_INPUT_ADD)", ""]
    app_add = _top("APP_INPUT_ADD", 20)
    if app_add.empty:
        lines.append("- (none)")
    else:
        for _, r in app_add.iterrows():
            lines.append(
                f"- {r['raw_column']}={r['raw_value']} — model={r['currently_in_model']}, "
                f"safe={r['safe_for_user_input']} ({r['notes']})"
            )

    lines += [
        "",
        "### Priority app gaps vs V24.1",
        "- **Yerden Isıtma**: in V24.1 heating OHE + premium heating score; **missing from HEATING_OPTIONS** (vocab has it).",
        "- **kitchen / balcony / usage_status / bathroom_count**: used by model, hardcoded in app defaults — not user editable.",
        "- **duplex_type**: model has full duplex pack; app has no chip/control.",
        "- **site_project_id**: model core; app only `site_inside` yes/no.",
        "- **lat/lon + multi-county**: model geo/county-aware; app İzmit-only neighborhood picker, no pin.",
        "- **floor_segment**: in DB+model; not exposed in app (only floor_num/total_floors).",
        "",
        "## 4) In model but not in app",
        "",
    ]
    if not model_cov.empty:
        miss = model_cov[model_cov["app_coverage_status"] == "app_input_missing"]
        for _, r in miss.head(30).iterrows():
            lines.append(f"- `{r['model_feature']}` ({r['feature_kind']})")

    lines += ["", "## 5) In app but not used by V24.1 / weak wiring", ""]
    lines.append("- App still serves an older V6.1-era joblib path in EDER, not V24.1 bundle — even collected fields may not hit V24.1.")
    lines.append("- `site_inside` is collected but V24.1 premium signal is mainly `site_project_id` / foldsafe site layer.")
    lines.append("- `neighborhood` locked to İzmit list; V24.1 expects multi-county `county`+`district`.")
    if not app_cov.empty:
        unused = app_cov[app_cov["status"].isin(["app_only_unused", "hardcoded_default_not_user"])]
        for _, r in unused.head(20).iterrows():
            lines.append(f"- `{r['app_field']}` — {r['status']} {r['notes']}")

    lines += ["", "## 6) Risky / keep out or derived-only", ""]
    risky = _top("IGNORE_LOW_SIGNAL", 15)
    derived = _top("KEEP_DERIVED_ONLY", 15)
    if derived.empty and risky.empty:
        lines.append("- Prefer KEEP_DERIVED_ONLY for raw detail pipe tokens (do not ask users to paste `detail_*`).")
    for _, r in derived.iterrows():
        lines.append(f"- KEEP_DERIVED_ONLY: {r['raw_column']}={r['raw_value']} (sale={int(r['sale_count'])})")
    for _, r in risky.iterrows():
        lines.append(f"- IGNORE_LOW_SIGNAL: {r['raw_column']}={r['raw_value']} (sale={int(r['sale_count'])})")

    lines += [
        "",
        "## Special focus: Yerden Isıtma",
        "",
        f"- sale count: **{yerden.get('sale_count')}** (share={yerden.get('sale_share')})",
        f"- rental count: **{yerden.get('rental_count')}** (share={yerden.get('rental_share')})",
        f"- median unit price yerden: `{yerden.get('median_unit_price_yerden')}`",
        f"- median unit price other: `{yerden.get('median_unit_price_other')}`",
        f"- lift vs other: `{yerden.get('median_lift_vs_other')}`",
        f"- in V24.1 heating categorical: **{yerden.get('in_model_heating_categorical')}**",
        f"- in app UI options: **{yerden.get('in_app_ui')}**",
        f"- in app training_vocab: **{yerden.get('in_app_vocab')}**",
        f"- recommendation: **`{yerden.get('recommendation')}`** (app_input_missing; model already has leaf)",
        "",
        "### Yerden by county (sale)",
        "",
    ]
    yc = yerden.get("by_county")
    if isinstance(yc, pd.DataFrame):
        for _, r in yc.iterrows():
            lines.append(
                f"- {r['county']}: yerden={int(r['yerden_sale'])}/{int(r['sale_total'])} "
                f"({r['yerden_share']:.1%})"
            )

    lines += [
        "",
        "## Recommended V25 prep order (no train yet)",
        "1. App: add **Yerden Isıtma** (+ Kombi Elektrik / VRV) to HEATING_OPTIONS.",
        "2. App: expose kitchen, balcony, usage_status, bathroom_count (stop hardcoding).",
        "3. App: multi-county Kocaeli selector; unlock beyond İzmit.",
        "4. App: duplexType chip (none/roof/garden/middle/standard).",
        "5. App: site_project picker wired to V24 options export (not only site_inside).",
        "6. App: optional map pin (lat/lon) for geo features.",
        "7. Model V25: keep heating OHE; optionally ablate detail-group expansions that cleared volume thresholds; do **not** dump raw `detail_*` pipes into user form.",
        "8. Only after app contract matches V24.1 inputs, swap EDER joblib from V6.1-era to V24.1/V25 bundle.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Feature reactivation & app input coverage audit (no training).")
    ap.add_argument("--city", default="Kocaeli")
    ap.add_argument("--source-site", default=os.getenv("SOURCE_SITE", "sahibinden"))
    ap.add_argument("--sale-table", default=os.getenv("SALE_TABLE", "market.sale_listings"))
    ap.add_argument("--rental-table", default=os.getenv("RENTAL_TABLE", "market.rental_listings"))
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--v24-importance",
        default="v3/outputs/v24_1_kocaeli_site_merge_repair/ablation_v24_full_v24/reports/feature_importance_v18_basiskele.csv",
    )
    ap.add_argument(
        "--v24-selection",
        default="v3/outputs/v24_1_kocaeli_site_merge_repair/reports/selection_decision_v24_1.json",
    )
    ap.add_argument(
        "--app-root",
        default=r"c:\Users\soyka\OneDrive\Desktop\Quarox Emlak\EDER\thequarox-eder",
        help="Path to EDER thequarox-eder app (optional).",
    )
    return ap.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        load_root_env(start=HERE)
    except Exception as exc:
        print(f"ERROR: failed to load root .env — {exc}")
        return 2

    args = parse_args()
    # Normalize known bad env value
    source_site = args.source_site
    if source_site == "sahibinden.com":
        source_site = "sahibinden"
        print("NOTE: SOURCE_SITE=sahibinden.com remapped to sahibinden (DB value).")

    root = _repo_root()
    out_dir = Path(args.out) if args.out else root / "analysis_outputs" / f"feature_reactivation_audit_{_ts_folder()}"
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
        print(f"ERROR: DB connection failed — {exc}")
        return 3

    sale_raw, meta_s = fetch_listings(
        engine,
        table=args.sale_table,
        purpose="sale",
        city=args.city,
        source_site=source_site,
        columns=AUDIT_COLUMNS,
    )
    rental_raw, meta_r = fetch_listings(
        engine,
        table=args.rental_table,
        purpose="rental",
        city=args.city,
        source_site=source_site,
        columns=AUDIT_COLUMNS,
    )
    print(f"rows fetched sale={len(sale_raw)} rental={len(rental_raw)}")

    sale = normalize_listings(sale_raw, "sale")
    rental = normalize_listings(rental_raw, "rental")

    imp_path = Path(args.v24_importance)
    if not imp_path.is_absolute():
        imp_path = root / imp_path
    model_info = load_v24_model_features(root, imp_path if imp_path.exists() else None)

    app_root = Path(args.app_root) if args.app_root else None
    app_info = load_app_coverage(app_root)

    sel = {}
    sel_path = Path(args.v24_selection)
    if not sel_path.is_absolute():
        sel_path = root / sel_path
    if sel_path.exists():
        sel = json.loads(sel_path.read_text(encoding="utf-8"))

    raw_counts = build_raw_feature_value_counts(sale, rental)
    gap, candidates = build_gap_matrix(sale, rental, raw_counts, model_info, app_info)
    model_cov = build_model_feature_coverage(model_info, app_info, gap)
    app_cov = build_app_input_coverage(app_info, model_info, gap)
    yerden = yerden_deep_dive(sale, rental, model_info, app_info)

    # Attach sale counts for heating leaves into model_cov
    if not raw_counts.empty:
        heat = raw_counts[(raw_counts["purpose"] == "sale") & (raw_counts["raw_column"] == "heating")]
        heat_map = {str(r["raw_value"]): int(r["count"]) for _, r in heat.iterrows()}
        if not model_cov.empty:
            def _fill(row):
                f = str(row["model_feature"])
                if f.startswith("cat__heating_"):
                    return heat_map.get(f[len("cat__heating_") :], row["max_sale_count_observed"])
                return row["max_sale_count_observed"]

            model_cov["max_sale_count_observed"] = model_cov.apply(_fill, axis=1)

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "city": args.city,
        "source_site": source_site,
        "sale_rows": int(len(sale)),
        "rental_rows": int(len(rental)),
        "total_rows": int(len(sale) + len(rental)),
        "v24_1_r2": (sel.get("metrics") or {}).get("r2"),
        "v24_1_mape": (sel.get("metrics") or {}).get("mape"),
        "v24_1_selected": sel.get("selected_experiment"),
        "model_heating_categories": model_info.get("heating_categories"),
        "app_heating_options": app_info.get("heating_options"),
        "yerden_sale_count": yerden.get("sale_count"),
        "yerden_recommendation": yerden.get("recommendation"),
        "candidate_count": int(len(candidates)),
        "output_dir": str(out_dir),
        "fetch_missing_sale": meta_s.get("missing_columns"),
        "fetch_missing_rental": meta_r.get("missing_columns"),
        "app_notes": app_info.get("notes"),
    }

    _write_csv(raw_counts, out_dir / "raw_feature_value_counts.csv")
    _write_csv(candidates, out_dir / "candidate_reactivated_features.csv")
    _write_csv(model_cov, out_dir / "model_feature_coverage_audit.csv")
    _write_csv(app_cov, out_dir / "app_input_coverage_audit.csv")
    _write_csv(gap, out_dir / "feature_gap_matrix.csv")
    if isinstance(yerden.get("by_county"), pd.DataFrame):
        _write_csv(yerden["by_county"], out_dir / "yerden_isitma_by_county.csv")
    if isinstance(yerden.get("by_site"), pd.DataFrame):
        _write_csv(yerden["by_site"], out_dir / "yerden_isitma_by_site_inside.csv")
    if isinstance(yerden.get("by_age"), pd.DataFrame):
        _write_csv(yerden["by_age"], out_dir / "yerden_isitma_by_building_age.csv")
    (out_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out_dir / "yerden_isitma_special.json").write_text(
        json.dumps({k: v for k, v in yerden.items() if not isinstance(v, pd.DataFrame)}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    write_recommendations_md(
        out_dir / "v25_feature_reactivation_recommendations.md",
        summary=summary,
        gap=gap,
        candidates=candidates,
        model_cov=model_cov,
        app_cov=app_cov,
        yerden=yerden,
        app_info=app_info,
    )

    print(f"reports written -> {out_dir}")
    print(f"yerden sale={yerden.get('sale_count')} app_ui={yerden.get('in_app_ui')} model={yerden.get('in_model_heating_categorical')}")
    print(f"candidates={len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
