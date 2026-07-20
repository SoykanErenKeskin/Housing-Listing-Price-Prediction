from __future__ import annotations

"""
V8 DB -> clean -> residual-target train -> report pipeline for Kocaeli housing unit price model.

What it does:
  1) Pulls sale and rental listings directly from PostgreSQL/Neon DB.
  2) Expands helper fields from raw JSON.
  3) Cleans sale/rental data and saves raw + cleaned CSV files.
  4) Builds rental-market and optional trend-market features.
  5) Trains multiple tabular ML models with leak-safe target-stat features inside CV.
  6) Creates OOF predictions, metrics, group-error reports, plots, and joblib artifacts.

Install:
  pip install pandas numpy scikit-learn joblib matplotlib sqlalchemy psycopg2-binary python-dotenv

Recommended DB usage:
  Put this in .env, or pass --db-url:
    DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/DB?sslmode=require

Example:
  python train_v8_db_residual_pipeline.py ^
    --out outputs/v8_kocaeli ^
    --city Kocaeli ^
    --counties "İzmit,Başiskele,Gölcük,Karamürsel" ^
    --sale-table sale_listings ^
    --rental-table rental_listings ^
    --trend-table trend_observed

Local JSON test example:
  python train_v8_db_residual_pipeline.py ^
    --sale-json "sale_listings (2).json" ^
    --rental-json "rental_listings (1).json" ^
    --out outputs/v8_local_test
"""

import argparse
import json
import math
import os
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from pandas.errors import PerformanceWarning
    warnings.filterwarnings("ignore", category=PerformanceWarning)
except Exception:
    pass

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

try:
    from sqlalchemy import create_engine, text
except Exception:  # pragma: no cover - handled at runtime if DB mode is used
    create_engine = None
    text = None


# =========================
# Easy-to-edit constants
# =========================

# Prefer .env / env var. For quick local tests only, you can paste a DB URL here.
# Do NOT commit a real DB URL to GitHub.
DB_URL = os.getenv("DATABASE_URL") or ""

DEFAULT_SALE_TABLE = os.getenv("SALE_TABLE", "sale_listings")
DEFAULT_RENTAL_TABLE = os.getenv("RENTAL_TABLE", "rental_listings")
DEFAULT_SOURCE_SITE = os.getenv("SOURCE_SITE", "listing_portal")
DEFAULT_TREND_TABLE = "trend_observed"
DEFAULT_CITY = "Kocaeli"
DEFAULT_COUNTIES = ["İzmit", "Başiskele", "Gölcük", "Karamürsel"]
TARGET = "unit_price_gross"
RANDOM_STATE = 42

DETAIL_PREFIXES = ("front_", "view_", "transport_", "near_", "out_", "in_", "subtype_")
DETAIL_EXACT = {
    "building_age_raw",
    "building_age_group",
    "detail_cephe",
    "detail_manzara",
    "detail_konut_tipi",
    "detail_ic_ozellikler",
    "detail_dis_ozellikler",
    "detail_muhit",
    "detail_ulasim",
    "detail_engelli_yasli_uygun",
    "detail_selected_count",
    "detail_quality_score",
}

DETAIL_RAW_COLUMNS = {
    "detail_cephe": "detail_front_count",
    "detail_manzara": "detail_view_count",
    "detail_ulasim": "detail_transport_count",
    "detail_muhit": "detail_near_count",
    "detail_ic_ozellikler": "detail_inside_count",
    "detail_dis_ozellikler": "detail_outside_count",
    "detail_konut_tipi": "detail_subtype_count",
}

SCORE_GROUPS = {
    "front_score": ["front_west", "front_east", "front_south", "front_north"],
    "view_score": [
        "view_bosphorus",
        "view_sea",
        "view_nature",
        "view_lake",
        "view_pool",
        "view_river",
        "view_park_green",
        "view_city",
    ],
    "transport_score": [
        "transport_main_road",
        "transport_e5",
        "transport_tem",
        "transport_tram",
        "transport_train",
        "transport_bus_stop",
        "transport_minibus",
        "transport_metro",
        "transport_airport",
    ],
    "nearby_score": [
        "near_mall",
        "near_mosque",
        "near_pharmacy",
        "near_hospital",
        "near_school",
        "near_university",
        "near_market",
        "near_park",
        "near_city_center",
        "near_sea_zero",
        "near_lake_zero",
    ],
    "outside_quality_score": [
        "out_security",
        "out_camera",
        "out_pool",
        "out_open_pool",
        "out_closed_pool",
        "out_heat_insulation",
        "out_sound_insulation",
        "out_generator",
        "out_hydrofor",
        "out_children_playground",
        "out_sports_area",
        "out_sauna_hamam",
    ],
    "inside_quality_score": [
        "in_builtin_kitchen",
        "in_parent_bathroom",
        "in_glass_balcony",
        "in_terrace",
        "in_air_conditioner",
        "in_smart_home",
        "in_steel_door",
        "in_fiber",
        "in_intercom",
        "in_dressing_room",
        "in_pantry",
        "in_laminate_floor",
        "in_pvc",
        "in_heat_glass",
    ],
    "site_security_score": [
        "out_security",
        "out_camera",
        "out_generator",
        "out_hydrofor",
        "out_heat_insulation",
        "out_sound_insulation",
    ],
    "accessibility_score": [
        "transport_tram",
        "transport_bus_stop",
        "transport_minibus",
        "transport_main_road",
        "transport_tem",
        "transport_e5",
        "near_market",
        "near_hospital",
        "near_school",
    ],
}

NUMERIC_FEATURES_BASE = [
    "gross_m2",
    "net_m2",
    "building_age",
    "floor_num",
    "total_floors",
    "bathroom_count",
    "dues",
    "open_area_m2",
    "has_open_area",
    "net_gross_ratio",
    "floor_ratio",
    "remaining_floors",
    "is_ground_floor",
    "is_basement",
    "is_top_floor",
    "is_middle_floor",
    "rooms",
    "living_rooms",
    "total_room_score",
    "is_new_building",
    "is_old_building",
    "is_small_flat",
    "is_large_flat",
    "detail_selected_count",
    "detail_quality_score",
    "detail_front_count",
    "detail_view_count",
    "detail_transport_count",
    "detail_near_count",
    "detail_inside_count",
    "detail_outside_count",
    "detail_subtype_count",
    "front_score",
    "view_score",
    "transport_score",
    "nearby_score",
    "inside_quality_score",
    "outside_quality_score",
    "site_security_score",
    "accessibility_score",
    "district_rent_m2_median",
    "district_rent_m2_mean",
    "district_rent_m2_count",
    "district_rent_m2_iqr",
    "district_rent_m2_cv",
    "district_room_rent_m2_median",
    "district_room_rent_m2_count",
    "district_m2_group_rent_m2_median",
    "district_m2_group_rent_m2_count",
    "county_rent_m2_median",
    "county_rent_m2_mean",
    "county_rent_m2_count",
    "estimated_rent_m2_gross",
    "estimated_monthly_rent_gross",
    "rent_feature_confidence",
    "trend_sale_m2",
    "trend_rent_m2",
    "trend_count_sale",
    "trend_count_rent",
    "trend_listing_period_sale",
    "trend_yield",
    "trend_price_change_sale",
    "trend_sale_annual_change",
    "location_baseline_m2",
    "location_baseline_log",
    "location_baseline_count",
    "location_baseline_level_code",
    "location_baseline_vs_trend",
]

TARGET_STAT_COLS = [
    "district_target_median",
    "district_target_mean",
    "county_target_median",
    "county_target_mean",
    "district_m2_group_target_median",
    "district_room_count_target_median",
    "county_room_count_target_median",
]

NUMERIC_FEATURES = NUMERIC_FEATURES_BASE + TARGET_STAT_COLS

CATEGORICAL_FEATURES = [
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
    "rent_feature_level",
]

LEAKAGE_OR_UNUSED_COLUMNS = {
    TARGET,
    "unit_price_net",
    "price",
    "monthly_rent",
    "rent_per_m2_gross",
    "rent_per_m2_net",
    "deposit",
    "classified_id",
    "source_site",
    "source_url",
    "image_url",
    "raw",
    "title",
    "saved_at",
    "updated_at",
    "listing_date",
    "site_name",
    "neighborhood",
}


@dataclass
class RunConfig:
    city: str
    counties: list[str]
    target_mode: str
    n_splits: int
    random_state: int
    sale_table: str
    rental_table: str
    trend_table: str
    use_trend: bool
    selected_models: list[str]
    fast_mode: bool
    min_sale_unit_price: float
    max_sale_unit_price: float
    min_rent_m2: float
    max_rent_m2: float
    use_location_outlier_filter: bool
    min_location_ratio: float
    max_location_ratio: float
    location_mad_threshold: float
    location_min_group_size: int


class ModelBundle:
    """Small prediction wrapper saved with the final ensemble."""

    def __init__(self, models: dict[str, Any], weights: dict[str, float], feature_columns: list[str], metrics: dict[str, Any]):
        self.models = models
        self.weights = weights
        self.feature_columns = feature_columns
        self.metrics = metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self.feature_columns]
        pred = np.zeros(len(X), dtype=float)
        total_w = sum(self.weights.values())
        for name, model in self.models.items():
            pred += np.asarray(model.predict(X), dtype=float) * (self.weights[name] / total_w)
        return pred


# =========================
# General helpers
# =========================


def safe_json_loads(x: Any) -> dict[str, Any]:
    if pd.isna(x):
        return {}
    if isinstance(x, dict):
        return x
    if not isinstance(x, str) or not x.strip():
        return {}
    try:
        obj = json.loads(x)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).replace("\u00a0", " ").strip().lower()
    repl = {
        "ı": "i",
        "ğ": "g",
        "ü": "u",
        "ş": "s",
        "ö": "o",
        "ç": "c",
        "İ": "i",
        "Ğ": "g",
        "Ü": "u",
        "Ş": "s",
        "Ö": "o",
        "Ç": "c",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    for token in [" mahallesi", " mah.", " mh.", " mah", " mh"]:
        s = s.replace(token, "")
    return " ".join(s.split())


def clean_str(x: Any) -> Any:
    if pd.isna(x):
        return np.nan
    s = str(x).replace("\u00a0", " ").strip()
    return np.nan if not s or s.lower() in {"nan", "none", "null", "belirtilmemiş"} else s


def to_num(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip()
    s = s.replace("TL", "").replace("₺", "").replace("m²", "").replace("m2", "")
    s = "".join(ch for ch in s if ch.isdigit() or ch in ".,-")
    if not s or s in {"-", ".", ","}:
        return np.nan

    # Handles both Turkish formatted numbers (6.500.000, 54.166,67)
    # and JSON/API decimal strings (54166.67).
    if "," in s and "." in s:
        # Last separator is accepted as decimal separator.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "." in s:
        if s.count(".") > 1:
            s = s.replace(".", "")
        else:
            left, right = s.split(".", 1)
            # 6.500 is more likely thousands; 54166.67 is decimal.
            if len(right) == 3 and len(left) <= 3:
                s = left + right

    try:
        return float(s)
    except Exception:
        return np.nan


def validate_table_name(name: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    if not name or any(ch not in allowed for ch in name):
        raise ValueError(f"Invalid table name: {name}")
    return name


def parse_counties(s: str | None) -> list[str]:
    if not s:
        return DEFAULT_COUNTIES.copy()
    return [p.strip() for p in s.split(",") if p.strip()]


def m2_group_from_value(x: Any) -> Any:
    x = to_num(x)
    if not np.isfinite(x):
        return np.nan
    if x <= 75:
        return "0-75"
    if x <= 100:
        return "76-100"
    if x <= 125:
        return "101-125"
    if x <= 150:
        return "126-150"
    if x <= 200:
        return "151-200"
    return "200+"


def parse_building_age(value: Any) -> tuple[float, str]:
    raw = "" if pd.isna(value) else str(value).strip()
    low = normalize_text(raw)
    if not low:
        return np.nan, "missing"
    m = re.search(r"(\d+)\s*-\s*(\d+)", low)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return round((a + b) / 2, 1), raw
    m = re.search(r"(\d+)\s*(ve)?\s*(uzeri|ustu|\+)", low)
    if m:
        return float(m.group(1)) + 4, raw
    m = re.search(r"(\d+)", low)
    if m:
        return float(m.group(1)), raw
    return np.nan, raw or "missing"


def parse_room(v: Any) -> tuple[float, float, float]:
    if pd.isna(v):
        return np.nan, np.nan, np.nan
    s = str(v).replace(" ", "").lower()
    m = re.search(r"(\d+)\+(\d+)", s)
    if m:
        r, l = float(m.group(1)), float(m.group(2))
        return r, l, r + l
    m = re.search(r"(\d+)", s)
    if m:
        r = float(m.group(1))
        return r, np.nan, r
    return np.nan, np.nan, np.nan


def count_pipe_values(x: Any) -> int:
    if pd.isna(x):
        return 0
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return 0
    return len([p for p in s.split("|") if p.strip()])


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def metric_dict(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp) & (yt > 0)
    yt = yt[mask]
    yp = yp[mask]
    if len(yt) == 0:
        return {
            "rows": 0,
            "r2": np.nan,
            "log_r2": np.nan,
            "mape": np.nan,
            "median_ape": np.nan,
            "mae_tl_per_m2": np.nan,
            "median_ae_tl_per_m2": np.nan,
        }
    ape = np.abs((yt - yp) / yt)
    log_true = np.log1p(yt)
    log_pred = np.log1p(np.maximum(yp, 0))
    return {
        "rows": int(len(yt)),
        "r2": float(r2_score(yt, yp)) if len(yt) > 1 else np.nan,
        "log_r2": float(r2_score(log_true, log_pred)) if len(yt) > 1 else np.nan,
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "mae_tl_per_m2": float(mean_absolute_error(yt, yp)),
        "median_ae_tl_per_m2": float(median_absolute_error(yt, yp)),
    }


# =========================
# Loading from DB / JSON
# =========================


def create_db_engine(db_url: str):
    if create_engine is None:
        raise RuntimeError("sqlalchemy is not installed. Install sqlalchemy and psycopg2-binary for DB mode.")
    if not db_url:
        raise ValueError("DB URL is empty. Set DATABASE_URL, DB_URL, or pass --db-url.")
    return create_engine(db_url, pool_pre_ping=True)


def fetch_listing_table(engine, table: str, purpose: str, city: str, limit: int | None = None) -> pd.DataFrame:
    table = validate_table_name(table)
    limit_clause = f" LIMIT {int(limit)}" if limit else ""
    sql = text(
        f"""
        SELECT *
        FROM {table}
        WHERE lower(coalesce(city, '')) = lower(:city)
          AND lower(coalesce(listing_purpose, '')) = lower(:purpose)
        ORDER BY saved_at DESC NULLS LAST, updated_at DESC NULLS LAST
        {limit_clause}
        """
    )
    return pd.read_sql(sql, engine, params={"city": city, "purpose": purpose})


def fetch_latest_trend_table(engine, table: str, city: str, max_date: str | None = None) -> pd.DataFrame:
    table = validate_table_name(table)
    date_filter = "AND property_date <= :max_date" if max_date else ""
    params = {"city": city}
    if max_date:
        params["max_date"] = max_date
    sql = text(
        f"""
        WITH filtered AS (
            SELECT
                id,
                property_date,
                property_year,
                property_month,
                city_name,
                county_name,
                district_name,
                district_id,
                unit_price_for_sale,
                unit_price_for_rent,
                count_for_sale,
                count_for_rent,
                listing_period_for_sale,
                yield,
                price_change_sale,
                unit_price_sale_annual_change,
                projection_like,
                ROW_NUMBER() OVER (
                    PARTITION BY county_name, district_name
                    ORDER BY property_date DESC
                ) AS rn
            FROM {table}
            WHERE lower(coalesce(city_name, '')) = lower(:city)
              AND coalesce(projection_like, false) = false
              {date_filter}
        )
        SELECT *
        FROM filtered
        WHERE rn = 1
        ORDER BY county_name, district_name
        """
    )
    return pd.read_sql(sql, engine, params=params)


def read_json_records(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"JSON file must contain a list of records: {path}")
    return pd.DataFrame(data)


def load_raw_data(args: argparse.Namespace, out_dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    db_url = args.db_url or DB_URL or os.getenv("DATABASE_URL")

    if args.sale_json and args.rental_json:
        sales_raw = read_json_records(args.sale_json)
        rentals_raw = read_json_records(args.rental_json)
        trend_raw = pd.DataFrame()
    else:
        engine = create_db_engine(db_url)
        print("DB sale listings are being fetched...")
        sales_raw = fetch_listing_table(engine, args.sale_table, "sale", args.city, args.limit_sale)
        print(f"Raw sale rows: {len(sales_raw)}")

        print("DB rental listings are being fetched...")
        rentals_raw = fetch_listing_table(engine, args.rental_table, "rental", args.city, args.limit_rental)
        print(f"Raw rental rows: {len(rentals_raw)}")

        trend_raw = pd.DataFrame()
        if args.use_trend:
            try:
                print("DB trend rows are being fetched...")
                trend_raw = fetch_latest_trend_table(engine, args.trend_table, args.city, args.trend_max_date)
                print(f"Trend rows: {len(trend_raw)}")
            except Exception as exc:
                warnings.warn(f"Trend table could not be fetched; continuing without trend features. Error: {exc}")
                trend_raw = pd.DataFrame()

    sales_raw.to_csv(out_dirs["raw"] / "sales_raw_from_source.csv", index=False, encoding="utf-8-sig")
    rentals_raw.to_csv(out_dirs["raw"] / "rentals_raw_from_source.csv", index=False, encoding="utf-8-sig")
    if not trend_raw.empty:
        trend_raw.to_csv(out_dirs["raw"] / "trend_latest_from_source.csv", index=False, encoding="utf-8-sig")
    return sales_raw, rentals_raw, trend_raw


# =========================
# Cleaning and feature setup
# =========================


def expand_raw_json_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "raw" not in df.columns:
        return df.copy()
    out = df.copy()
    raw_objects = out["raw"].map(safe_json_loads)

    keys = set()
    for obj in raw_objects:
        if not isinstance(obj, dict):
            continue
        for k in obj:
            if k in DETAIL_EXACT or k.startswith(DETAIL_PREFIXES):
                keys.add(k)

    for key in sorted(keys):
        raw_values = raw_objects.map(lambda obj, kk=key: obj.get(kk, np.nan) if isinstance(obj, dict) else np.nan)
        raw_values = raw_values.replace("", np.nan)
        if key.startswith(DETAIL_PREFIXES) or key in {"detail_selected_count", "detail_quality_score"}:
            raw_values = pd.to_numeric(raw_values, errors="coerce")
            if key not in out.columns:
                out[key] = raw_values
            else:
                current = pd.to_numeric(out[key].replace("", np.nan), errors="coerce")
                out[key] = current.where(current.notna(), raw_values)
        else:
            raw_values = raw_values.astype("object")
            if key not in out.columns:
                out[key] = raw_values
            else:
                current_text = out[key].fillna("").astype(str).str.strip()
                mask = current_text.eq("") & raw_values.notna()
                out.loc[mask, key] = raw_values.loc[mask]
    return out


def dedupe_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "classified_id" in out.columns:
        sort_cols = [c for c in ["saved_at", "updated_at"] if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols)
        out = out.drop_duplicates("classified_id", keep="last")
    return out.reset_index(drop=True)


def base_clean(df: pd.DataFrame, purpose: str, cfg: RunConfig) -> pd.DataFrame:
    out = expand_raw_json_columns(df)
    out = dedupe_rows(out)
    out = out.loc[:, ~out.columns.duplicated()].copy()

    for c in out.columns:
        if out[c].dtype == "object" and c != "raw":
            out[c] = out[c].map(clean_str)

    numeric_cols = [
        "price",
        "monthly_rent",
        "unit_price_gross",
        "unit_price_net",
        "rent_per_m2_gross",
        "rent_per_m2_net",
        "gross_m2",
        "net_m2",
        "building_age",
        "floor_num",
        "total_floors",
        "bathroom_count",
        "dues",
        "deposit",
        "detail_selected_count",
        "detail_quality_score",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = out[col].map(to_num)

    if "building_age_raw" not in out.columns:
        out["building_age_raw"] = np.nan
    if "building_age_group" not in out.columns:
        out["building_age_group"] = np.nan
    age_source = out["building_age_raw"].where(out["building_age_raw"].notna(), out.get("building_age", np.nan))
    parsed = age_source.map(parse_building_age)
    parsed_age = parsed.map(lambda x: x[0])
    parsed_group = parsed.map(lambda x: x[1])
    if "building_age" not in out.columns:
        out["building_age"] = parsed_age
    else:
        out["building_age"] = out["building_age"].where(out["building_age"].notna(), parsed_age)
    out["building_age_group"] = out["building_age_group"].where(out["building_age_group"].notna(), parsed_group)

    if "listing_purpose" not in out.columns:
        out["listing_purpose"] = purpose
    out["listing_purpose"] = out["listing_purpose"].fillna(purpose).astype(str).str.lower()

    if "city" not in out.columns:
        out["city"] = cfg.city
    if "county" not in out.columns:
        out["county"] = np.nan
    if "district" not in out.columns and "neighborhood" in out.columns:
        out["district"] = out["neighborhood"]
    if "district" not in out.columns:
        out["district"] = np.nan

    out = out[out["listing_purpose"].eq(purpose)].copy()
    out["city_key"] = out["city"].map(normalize_text)
    out["county_key"] = out["county"].map(normalize_text)
    out["district_key"] = out["district"].map(normalize_text)

    city_key = normalize_text(cfg.city)
    county_keys = {normalize_text(c) for c in cfg.counties}
    out = out[out["city_key"].eq(city_key)].copy()
    out = out[out["county_key"].isin(county_keys)].copy()

    if "gross_m2" in out.columns:
        out["gross_m2"] = pd.to_numeric(out["gross_m2"], errors="coerce")
    else:
        out["gross_m2"] = np.nan

    if "net_m2" not in out.columns:
        out["net_m2"] = np.nan
    out["net_m2"] = pd.to_numeric(out["net_m2"], errors="coerce")

    if purpose == "sale":
        if TARGET not in out.columns:
            out[TARGET] = np.nan
        if "price" in out.columns:
            mask = out[TARGET].isna() & out["price"].notna() & out["gross_m2"].notna() & (out["gross_m2"] > 0)
            out.loc[mask, TARGET] = out.loc[mask, "price"] / out.loc[mask, "gross_m2"]
    else:
        if "rent_per_m2_gross" not in out.columns:
            out["rent_per_m2_gross"] = np.nan
        if "monthly_rent" in out.columns:
            mask = out["rent_per_m2_gross"].isna() & out["monthly_rent"].notna() & out["gross_m2"].notna() & (out["gross_m2"] > 0)
            out.loc[mask, "rent_per_m2_gross"] = out.loc[mask, "monthly_rent"] / out.loc[mask, "gross_m2"]

    out["m2_group"] = out["gross_m2"].map(m2_group_from_value)

    return out.reset_index(drop=True)


def apply_basic_filters(df: pd.DataFrame, purpose: str, cfg: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    keep = pd.Series(True, index=out.index)

    keep &= out["gross_m2"].between(25, 500, inclusive="both")
    if purpose == "sale":
        keep &= out[TARGET].between(cfg.min_sale_unit_price, cfg.max_sale_unit_price, inclusive="both")
        if "price" in out.columns:
            keep &= out["price"].isna() | out["price"].between(500_000, 150_000_000, inclusive="both")
    else:
        keep &= out["rent_per_m2_gross"].between(cfg.min_rent_m2, cfg.max_rent_m2, inclusive="both")
        if "monthly_rent" in out.columns:
            keep &= out["monthly_rent"].isna() | out["monthly_rent"].between(1_000, 1_000_000, inclusive="both")

    removed = out.loc[~keep].copy()
    clean = out.loc[keep].copy()
    return clean.reset_index(drop=True), removed.reset_index(drop=True)


def remove_iqr_outliers(
    df: pd.DataFrame,
    target_col: str,
    group_cols: list[str],
    min_group_size: int = 18,
    multiplier: float = 2.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    keep = pd.Series(True, index=out.index)

    existing_group_cols = [c for c in group_cols if c in out.columns]
    if existing_group_cols:
        for _, idx in out.groupby(existing_group_cols, dropna=False).groups.items():
            idx = list(idx)
            if len(idx) < min_group_size:
                continue
            values = pd.to_numeric(out.loc[idx, target_col], errors="coerce")
            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr <= 0:
                continue
            low, high = q1 - multiplier * iqr, q3 + multiplier * iqr
            keep.loc[idx] &= values.between(low, high, inclusive="both")

    # Gentle global fallback.
    values = pd.to_numeric(out[target_col], errors="coerce")
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    if np.isfinite(iqr) and iqr > 0:
        low, high = q1 - 4.0 * iqr, q3 + 4.0 * iqr
        keep &= values.between(low, high, inclusive="both")

    removed = out.loc[~keep].copy()
    clean = out.loc[keep].copy()
    return clean.reset_index(drop=True), removed.reset_index(drop=True)


def clean_sales_and_rentals(
    sales_raw: pd.DataFrame,
    rentals_raw: pd.DataFrame,
    cfg: RunConfig,
    out_dirs: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sales = base_clean(sales_raw, "sale", cfg)
    rentals = base_clean(rentals_raw, "rental", cfg)

    sales_basic, sales_removed_basic = apply_basic_filters(sales, "sale", cfg)
    rentals_basic, rentals_removed_basic = apply_basic_filters(rentals, "rental", cfg)

    sales_clean, sales_removed_iqr = remove_iqr_outliers(
        sales_basic,
        TARGET,
        group_cols=["county", "room_count", "m2_group"],
        min_group_size=18,
        multiplier=2.75,
    )
    rentals_clean, rentals_removed_iqr = remove_iqr_outliers(
        rentals_basic,
        "rent_per_m2_gross",
        group_cols=["county", "room_count", "m2_group"],
        min_group_size=18,
        multiplier=2.75,
    )

    sales_clean.to_csv(out_dirs["input"] / "sales_cleaned_v8.csv", index=False, encoding="utf-8-sig")
    rentals_clean.to_csv(out_dirs["input"] / "rentals_cleaned_v8.csv", index=False, encoding="utf-8-sig")
    pd.concat([sales_removed_basic, sales_removed_iqr], ignore_index=True).to_csv(
        out_dirs["input"] / "sales_removed_v8.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat([rentals_removed_basic, rentals_removed_iqr], ignore_index=True).to_csv(
        out_dirs["input"] / "rentals_removed_v8.csv", index=False, encoding="utf-8-sig"
    )

    report = {
        "sales_raw_rows": int(len(sales_raw)),
        "sales_after_base_clean_rows": int(len(sales)),
        "sales_after_basic_filter_rows": int(len(sales_basic)),
        "sales_final_rows": int(len(sales_clean)),
        "sales_removed_basic_rows": int(len(sales_removed_basic)),
        "sales_removed_iqr_rows": int(len(sales_removed_iqr)),
        "rentals_raw_rows": int(len(rentals_raw)),
        "rentals_after_base_clean_rows": int(len(rentals)),
        "rentals_after_basic_filter_rows": int(len(rentals_basic)),
        "rentals_final_rows": int(len(rentals_clean)),
        "rentals_removed_basic_rows": int(len(rentals_removed_basic)),
        "rentals_removed_iqr_rows": int(len(rentals_removed_iqr)),
    }
    return sales_clean, rentals_clean, report


# =========================
# Market feature attachment
# =========================


def coefficient_of_variation(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 2:
        return 0.0
    mean = x.mean()
    return float(x.std() / mean) if mean else 0.0


def iqr_value(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 2:
        return 0.0
    return float(x.quantile(0.75) - x.quantile(0.25))


def rental_group_stats(rentals: pd.DataFrame, group_cols: list[str], prefix: str) -> pd.DataFrame:
    cols = [c for c in group_cols if c in rentals.columns]
    if not cols or rentals.empty:
        return pd.DataFrame(columns=cols)
    g = rentals.groupby(cols, dropna=False)["rent_per_m2_gross"]
    rep = g.agg(
        **{
            f"{prefix}_median": "median",
            f"{prefix}_mean": "mean",
            f"{prefix}_count": "count",
            f"{prefix}_iqr": iqr_value,
            f"{prefix}_cv": coefficient_of_variation,
        }
    ).reset_index()
    return rep


def attach_rental_features(sales: pd.DataFrame, rentals: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    s = sales.copy()
    r = rentals.copy()

    if r.empty:
        for col in [
            "district_rent_m2_median",
            "district_rent_m2_mean",
            "district_rent_m2_count",
            "district_rent_m2_iqr",
            "district_rent_m2_cv",
            "district_room_rent_m2_median",
            "district_room_rent_m2_count",
            "district_m2_group_rent_m2_median",
            "district_m2_group_rent_m2_count",
            "county_rent_m2_median",
            "county_rent_m2_mean",
            "county_rent_m2_count",
            "estimated_rent_m2_gross",
            "estimated_monthly_rent_gross",
            "rent_feature_confidence",
        ]:
            s[col] = np.nan if "count" not in col and "confidence" not in col else 0
        s["rent_feature_level"] = "missing"
        return s, {"rental_feature_note": "No rental rows were available."}

    for df in [s, r]:
        if "county_key" not in df.columns:
            df["county_key"] = df["county"].map(normalize_text)
        if "district_key" not in df.columns:
            df["district_key"] = df["district"].map(normalize_text)
        if "m2_group" not in df.columns:
            df["m2_group"] = df["gross_m2"].map(m2_group_from_value)

    district_stats = rental_group_stats(r, ["county_key", "district_key"], "district_rent_m2")
    county_stats = rental_group_stats(r, ["county_key"], "county_rent_m2")
    district_room_stats = rental_group_stats(r, ["county_key", "district_key", "room_count"], "district_room_rent_m2")
    district_m2_stats = rental_group_stats(r, ["county_key", "district_key", "m2_group"], "district_m2_group_rent_m2")

    # The downstream code expects count columns with shorter names.
    district_room_stats = district_room_stats.rename(
        columns={
            "district_room_rent_m2_iqr": "_drop_room_iqr",
            "district_room_rent_m2_cv": "_drop_room_cv",
            "district_room_rent_m2_mean": "_drop_room_mean",
        }
    ).drop(columns=["_drop_room_iqr", "_drop_room_cv", "_drop_room_mean"], errors="ignore")
    district_m2_stats = district_m2_stats.rename(
        columns={
            "district_m2_group_rent_m2_iqr": "_drop_m2_iqr",
            "district_m2_group_rent_m2_cv": "_drop_m2_cv",
            "district_m2_group_rent_m2_mean": "_drop_m2_mean",
        }
    ).drop(columns=["_drop_m2_iqr", "_drop_m2_cv", "_drop_m2_mean"], errors="ignore")

    s = s.merge(district_stats, on=["county_key", "district_key"], how="left")
    s = s.merge(county_stats, on=["county_key"], how="left")
    s = s.merge(district_room_stats, on=["county_key", "district_key", "room_count"], how="left")
    s = s.merge(district_m2_stats, on=["county_key", "district_key", "m2_group"], how="left")

    global_rent = float(r["rent_per_m2_gross"].median()) if r["rent_per_m2_gross"].notna().any() else np.nan
    global_count = int(r["rent_per_m2_gross"].notna().sum())

    # Fill from most specific to broader fallback.
    estimated = s["district_room_rent_m2_median"]
    estimated = estimated.where(estimated.notna(), s["district_m2_group_rent_m2_median"])
    estimated = estimated.where(estimated.notna(), s["district_rent_m2_median"])
    estimated = estimated.where(estimated.notna(), s["county_rent_m2_median"])
    estimated = estimated.fillna(global_rent)

    s["estimated_rent_m2_gross"] = estimated
    s["estimated_monthly_rent_gross"] = estimated * pd.to_numeric(s["gross_m2"], errors="coerce")

    for col in [
        "district_rent_m2_median",
        "district_rent_m2_mean",
        "district_rent_m2_iqr",
        "district_rent_m2_cv",
        "county_rent_m2_median",
        "county_rent_m2_mean",
    ]:
        if col in s.columns:
            s[col] = s[col].fillna(global_rent if "median" in col or "mean" in col else 0)
        else:
            s[col] = global_rent if "median" in col or "mean" in col else 0
    for col in [
        "district_rent_m2_count",
        "district_room_rent_m2_count",
        "district_m2_group_rent_m2_count",
        "county_rent_m2_count",
    ]:
        if col in s.columns:
            s[col] = pd.to_numeric(s[col], errors="coerce").fillna(0)
        else:
            s[col] = 0

    s["district_room_rent_m2_median"] = s["district_room_rent_m2_median"].fillna(s["district_rent_m2_median"])
    s["district_m2_group_rent_m2_median"] = s["district_m2_group_rent_m2_median"].fillna(s["district_rent_m2_median"])
    confidence_count = np.maximum.reduce(
        [
            s["district_room_rent_m2_count"].to_numpy(dtype=float),
            s["district_m2_group_rent_m2_count"].to_numpy(dtype=float),
            s["district_rent_m2_count"].to_numpy(dtype=float),
            s["county_rent_m2_count"].to_numpy(dtype=float),
        ]
    )
    s["rent_feature_confidence"] = np.minimum(confidence_count, 30) / 30
    s["rent_feature_level"] = np.select(
        [
            s["district_room_rent_m2_count"] > 0,
            s["district_m2_group_rent_m2_count"] > 0,
            s["district_rent_m2_count"] > 0,
            s["county_rent_m2_count"] > 0,
        ],
        ["district_room", "district_m2_group", "district", "county"],
        default="global",
    )

    report = {
        "rental_rows_used": int(len(r)),
        "global_rent_m2_median": global_rent,
        "global_rent_row_count": global_count,
        "rent_feature_level_counts": s["rent_feature_level"].value_counts(dropna=False).to_dict(),
    }
    return s, report


def attach_trend_features(sales: pd.DataFrame, trend: pd.DataFrame, cfg: RunConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    s = sales.copy()
    if trend is None or trend.empty:
        for col in [
            "trend_sale_m2",
            "trend_rent_m2",
            "trend_count_sale",
            "trend_count_rent",
            "trend_listing_period_sale",
            "trend_yield",
            "trend_price_change_sale",
            "trend_sale_annual_change",
        ]:
            s[col] = 0.0
        return s, {"trend_rows_used": 0, "trend_note": "No trend rows were available."}

    t = trend.copy()
    if "city_name" in t.columns:
        t = t[t["city_name"].map(normalize_text).eq(normalize_text(cfg.city))].copy()
    if "county_name" in t.columns:
        county_keys = {normalize_text(c) for c in cfg.counties}
        t = t[t["county_name"].map(normalize_text).isin(county_keys)].copy()

    if t.empty:
        return attach_trend_features(sales, pd.DataFrame(), cfg)

    t["county_key"] = t["county_name"].map(normalize_text)
    t["district_key"] = t["district_name"].map(normalize_text)
    for col in [
        "unit_price_for_sale",
        "unit_price_for_rent",
        "count_for_sale",
        "count_for_rent",
        "listing_period_for_sale",
        "yield",
        "price_change_sale",
        "unit_price_sale_annual_change",
    ]:
        if col in t.columns:
            t[col] = pd.to_numeric(t[col], errors="coerce")

    t = t.drop_duplicates(["county_key", "district_key"], keep="last")
    keep_cols = [
        "county_key",
        "district_key",
        "unit_price_for_sale",
        "unit_price_for_rent",
        "count_for_sale",
        "count_for_rent",
        "listing_period_for_sale",
        "yield",
        "price_change_sale",
        "unit_price_sale_annual_change",
        "property_date",
    ]
    keep_cols = [c for c in keep_cols if c in t.columns]
    s = s.merge(t[keep_cols], on=["county_key", "district_key"], how="left")

    rename = {
        "unit_price_for_sale": "trend_sale_m2",
        "unit_price_for_rent": "trend_rent_m2",
        "count_for_sale": "trend_count_sale",
        "count_for_rent": "trend_count_rent",
        "listing_period_for_sale": "trend_listing_period_sale",
        "yield": "trend_yield",
        "price_change_sale": "trend_price_change_sale",
        "unit_price_sale_annual_change": "trend_sale_annual_change",
    }
    s = s.rename(columns=rename)

    # County/global fallback for trend values.
    county_fallback = (
        t.groupby("county_key")[["unit_price_for_sale", "unit_price_for_rent", "count_for_sale", "count_for_rent"]]
        .median(numeric_only=True)
        .rename(
            columns={
                "unit_price_for_sale": "trend_sale_m2_county_fallback",
                "unit_price_for_rent": "trend_rent_m2_county_fallback",
                "count_for_sale": "trend_count_sale_county_fallback",
                "count_for_rent": "trend_count_rent_county_fallback",
            }
        )
        .reset_index()
    )
    s = s.merge(county_fallback, on="county_key", how="left")

    fallback_map = {
        "trend_sale_m2": "trend_sale_m2_county_fallback",
        "trend_rent_m2": "trend_rent_m2_county_fallback",
        "trend_count_sale": "trend_count_sale_county_fallback",
        "trend_count_rent": "trend_count_rent_county_fallback",
    }
    for c, fb in fallback_map.items():
        if c not in s.columns:
            s[c] = np.nan
        s[c] = s[c].fillna(s[fb])

    for c in [
        "trend_listing_period_sale",
        "trend_yield",
        "trend_price_change_sale",
        "trend_sale_annual_change",
    ]:
        if c not in s.columns:
            s[c] = np.nan

    drop_cols = [c for c in s.columns if c.endswith("_county_fallback")]
    s = s.drop(columns=drop_cols, errors="ignore")

    report = {
        "trend_rows_used": int(len(t)),
        "trend_district_matched_rows": int(s["trend_sale_m2"].notna().sum()) if "trend_sale_m2" in s.columns else 0,
        "trend_date_min": str(pd.to_datetime(t["property_date"]).min().date()) if "property_date" in t.columns and t["property_date"].notna().any() else None,
        "trend_date_max": str(pd.to_datetime(t["property_date"]).max().date()) if "property_date" in t.columns and t["property_date"].notna().any() else None,
    }
    return s, report


# =========================
# Sklearn transformers
# =========================


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: Any = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = expand_raw_json_columns(df)

        for col in NUMERIC_FEATURES_BASE + ["price", "monthly_rent", TARGET, "unit_price_net", "rent_per_m2_gross", "rent_per_m2_net"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].map(to_num) if df[col].dtype == "object" else df[col], errors="coerce")

        for col in CATEGORICAL_FEATURES + ["county", "district", "room_count"]:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = df[col].astype("object").where(df[col].notna(), "missing")
            df[col] = df[col].astype(str).str.strip().replace({"": "missing", "nan": "missing", "None": "missing"})

        if "m2_group" not in df.columns or df["m2_group"].isna().all():
            df["m2_group"] = df["gross_m2"].map(m2_group_from_value)
        df["m2_group"] = df["m2_group"].astype("object").where(df["m2_group"].notna(), "missing")

        if "building_age_group" not in df.columns or df["building_age_group"].isna().all():
            df["building_age_group"] = df["building_age"].map(lambda x: parse_building_age(x)[1])
        df["building_age_group"] = df["building_age_group"].astype("object").where(df["building_age_group"].notna(), "missing")

        if "net_gross_ratio" not in df.columns:
            df["net_gross_ratio"] = np.nan
        ratio = pd.to_numeric(df.get("net_m2", np.nan), errors="coerce") / pd.to_numeric(df.get("gross_m2", np.nan), errors="coerce").replace(0, np.nan)
        df["net_gross_ratio"] = pd.to_numeric(df["net_gross_ratio"], errors="coerce").where(df["net_gross_ratio"].notna(), ratio)
        df["net_gross_ratio"] = df["net_gross_ratio"].clip(0.2, 1.2)

        if "floor_num" not in df.columns:
            df["floor_num"] = np.nan
        if "total_floors" not in df.columns:
            df["total_floors"] = np.nan
        floor = pd.to_numeric(df["floor_num"], errors="coerce")
        total = pd.to_numeric(df["total_floors"], errors="coerce")
        df["floor_ratio"] = floor / total.replace(0, np.nan)
        df["remaining_floors"] = total - floor
        df["is_ground_floor"] = (floor == 0).astype(int)
        df["is_basement"] = (floor < 0).astype(int)
        df["is_top_floor"] = ((total.notna()) & (floor.notna()) & (floor >= total)).astype(int)
        df["is_middle_floor"] = ((floor > 0) & (total.notna()) & (floor < total)).astype(int)

        parsed_rooms = df["room_count"].map(parse_room)
        df["rooms"] = parsed_rooms.map(lambda x: x[0])
        df["living_rooms"] = parsed_rooms.map(lambda x: x[1])
        df["total_room_score"] = parsed_rooms.map(lambda x: x[2])

        age = pd.to_numeric(df.get("building_age", np.nan), errors="coerce")
        df["is_new_building"] = (age <= 3).astype(int)
        df["is_old_building"] = (age >= 20).astype(int)
        gross = pd.to_numeric(df.get("gross_m2", np.nan), errors="coerce")
        df["is_small_flat"] = (gross <= 75).astype(int)
        df["is_large_flat"] = (gross >= 180).astype(int)

        for raw_col, count_col in DETAIL_RAW_COLUMNS.items():
            if raw_col in df.columns:
                df[count_col] = df[raw_col].map(count_pipe_values).astype(int)
            elif count_col not in df.columns:
                df[count_col] = 0

        for col in set(sum(SCORE_GROUPS.values(), [])):
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 1).astype(int)

        for score_col, members in SCORE_GROUPS.items():
            df[score_col] = df[members].sum(axis=1)

        detail_count_cols = list(DETAIL_RAW_COLUMNS.values())
        if "detail_selected_count" not in df.columns:
            df["detail_selected_count"] = df[detail_count_cols].sum(axis=1)
        else:
            df["detail_selected_count"] = pd.to_numeric(df["detail_selected_count"], errors="coerce").fillna(
                df[detail_count_cols].sum(axis=1)
            )
        if "detail_quality_score" not in df.columns:
            df["detail_quality_score"] = df["inside_quality_score"] + df["outside_quality_score"]
        else:
            df["detail_quality_score"] = pd.to_numeric(df["detail_quality_score"], errors="coerce").fillna(0)

        if "open_area_m2" not in df.columns:
            df["open_area_m2"] = 0
        if "has_open_area" not in df.columns:
            df["has_open_area"] = (pd.to_numeric(df["open_area_m2"], errors="coerce").fillna(0) > 0).astype(int)

        for col in NUMERIC_FEATURES:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                df[col] = "missing"
            df[col] = df[col].astype("object").where(df[col].notna(), "missing")
            df[col] = df[col].astype(str).str.strip().replace({"": "missing", "nan": "missing", "None": "missing"})

        return df


class TargetStatsAdder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing: float = 20.0):
        self.smoothing = smoothing
        self.groups = {
            "district_target": ["county", "district"],
            "county_target": ["county"],
            "district_m2_group_target": ["county", "district", "m2_group"],
            "district_room_count_target": ["county", "district", "room_count"],
            "county_room_count_target": ["county", "room_count"],
        }

    def fit(self, X: pd.DataFrame, y: Any):
        df = X.copy()
        y_series = pd.Series(np.asarray(y, dtype=float), index=df.index, name="__target__")
        self.global_median_ = float(np.nanmedian(y_series))
        self.global_mean_ = float(np.nanmean(y_series))
        self.maps_: dict[str, dict[str, float]] = {}

        work = df.join(y_series)
        for name, cols in self.groups.items():
            cols = [c for c in cols if c in work.columns]
            if not cols:
                continue
            stats = work.groupby(cols, dropna=False)["__target__"].agg(["median", "mean", "count"]).reset_index()
            # Smooth mean toward global mean; keep median unsmoothed.
            stats["smooth_mean"] = (
                stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing
            ) / (stats["count"] + self.smoothing)
            stats["__key__"] = stats[cols].astype(str).agg("||".join, axis=1)
            self.maps_[f"{name}_median"] = dict(zip(stats["__key__"], stats["median"]))
            self.maps_[f"{name}_mean"] = dict(zip(stats["__key__"], stats["smooth_mean"]))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for name, cols in self.groups.items():
            cols = [c for c in cols if c in df.columns]
            if not cols:
                continue
            key = df[cols].astype(str).agg("||".join, axis=1)
            med_col = f"{name}_median"
            mean_col = f"{name}_mean"
            if med_col in TARGET_STAT_COLS:
                df[med_col] = key.map(self.maps_.get(med_col, {})).fillna(self.global_median_)
            if mean_col in TARGET_STAT_COLS:
                df[mean_col] = key.map(self.maps_.get(mean_col, {})).fillna(self.global_mean_)

        for col in TARGET_STAT_COLS:
            if col not in df.columns:
                df[col] = self.global_median_ if col.endswith("median") else self.global_mean_
        return df


class FeatureColumnKeeper(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_features: list[str], categorical_features: list[str]):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def fit(self, X: pd.DataFrame, y: Any = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in self.numeric_features:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in self.categorical_features:
            if col not in df.columns:
                df[col] = "missing"
            df[col] = df[col].astype("object").where(df[col].notna(), "missing")
            df[col] = df[col].astype(str).str.strip().replace({"": "missing", "nan": "missing", "None": "missing"})
        return df[self.numeric_features + self.categorical_features]




class LocationBaselineEncoder(BaseEstimator, TransformerMixin):
    """Learns a leak-safe local market baseline from the training fold.

    The baseline is intentionally limited to fields that can exist at prediction time:
    county, district, room_count, gross_m2-derived m2_group, and optional trend_sale_m2.
    It does not use title text, listing photos, or subjective premium labels.
    """

    def __init__(
        self,
        min_specific_count: int = 8,
        min_medium_count: int = 12,
        min_broad_count: int = 20,
        trend_weight_low_count: float = 0.65,
        trend_weight_ok_count: float = 0.25,
    ):
        self.min_specific_count = min_specific_count
        self.min_medium_count = min_medium_count
        self.min_broad_count = min_broad_count
        self.trend_weight_low_count = trend_weight_low_count
        self.trend_weight_ok_count = trend_weight_ok_count
        self.group_defs = [
            ("district_m2_room", ["county", "district", "m2_group", "room_count"], min_specific_count, 7),
            ("district_m2", ["county", "district", "m2_group"], min_medium_count, 6),
            ("district_room", ["county", "district", "room_count"], min_medium_count, 5),
            ("district", ["county", "district"], min_specific_count, 4),
            ("county_m2_room", ["county", "m2_group", "room_count"], min_medium_count, 3),
            ("county_m2", ["county", "m2_group"], min_broad_count, 2),
            ("county", ["county"], min_broad_count, 1),
        ]

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in ["county", "district", "room_count"]:
            if col not in df.columns:
                df[col] = "missing"
            df[col] = df[col].astype("object").where(df[col].notna(), "missing")
            df[col] = df[col].astype(str).str.strip().replace({"": "missing", "nan": "missing", "None": "missing"})
        if "m2_group" not in df.columns or df["m2_group"].isna().all():
            if "gross_m2" in df.columns:
                df["m2_group"] = df["gross_m2"].map(m2_group_from_value)
            else:
                df["m2_group"] = "missing"
        df["m2_group"] = df["m2_group"].astype("object").where(df["m2_group"].notna(), "missing")
        if "trend_sale_m2" not in df.columns:
            df["trend_sale_m2"] = np.nan
        df["trend_sale_m2"] = pd.to_numeric(df["trend_sale_m2"], errors="coerce")
        return df

    @staticmethod
    def _make_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
        return df[cols].astype(str).agg("||".join, axis=1)

    def fit(self, X: pd.DataFrame, y: Any):
        df = self._prepare(X)
        y_series = pd.Series(np.asarray(y, dtype=float), index=df.index, name="__target__")
        valid = y_series.notna() & np.isfinite(y_series) & (y_series > 0)
        df = df.loc[valid].copy()
        y_series = y_series.loc[valid]
        self.global_median_ = float(np.nanmedian(y_series)) if len(y_series) else 1.0
        self.maps_: dict[str, dict[str, tuple[float, int]]] = {}

        work = df.join(y_series)
        for name, cols, min_count, _code in self.group_defs:
            cols = [c for c in cols if c in work.columns]
            if not cols:
                continue
            stats = work.groupby(cols, dropna=False)["__target__"].agg(["median", "count"]).reset_index()
            stats = stats[stats["count"] >= int(min_count)].copy()
            if stats.empty:
                self.maps_[name] = {}
                continue
            stats["__key__"] = self._make_key(stats, cols)
            self.maps_[name] = {
                str(row["__key__"]): (float(row["median"]), int(row["count"]))
                for _, row in stats.iterrows()
            }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = self._prepare(X)
        baseline = pd.Series(self.global_median_, index=df.index, dtype=float)
        count = pd.Series(0, index=df.index, dtype=float)
        level_code = pd.Series(0, index=df.index, dtype=float)

        unresolved = pd.Series(True, index=df.index)
        for name, cols, _min_count, code in self.group_defs:
            cols = [c for c in cols if c in df.columns]
            if not cols:
                continue
            mapping = self.maps_.get(name, {})
            if not mapping:
                continue
            keys = self._make_key(df, cols)
            mapped = keys.map(mapping)
            has = unresolved & mapped.notna()
            if has.any():
                vals = mapped.loc[has].map(lambda x: x[0])
                cnts = mapped.loc[has].map(lambda x: x[1])
                baseline.loc[has] = pd.to_numeric(vals, errors="coerce").fillna(self.global_median_)
                count.loc[has] = pd.to_numeric(cnts, errors="coerce").fillna(0)
                level_code.loc[has] = float(code)
                unresolved.loc[has] = False

        trend = pd.to_numeric(df.get("trend_sale_m2", np.nan), errors="coerce")
        trend_valid = trend.notna() & np.isfinite(trend) & (trend > 0)
        base_valid = baseline.notna() & np.isfinite(baseline) & (baseline > 0)

        # Use trend as an external stabilizer. If local sample count is low, trend gets stronger weight.
        low_count = count < self.min_medium_count
        w = pd.Series(0.0, index=df.index)
        w.loc[trend_valid & low_count] = float(self.trend_weight_low_count)
        w.loc[trend_valid & ~low_count] = float(self.trend_weight_ok_count)
        combined = baseline.copy()
        both = trend_valid & base_valid & (w > 0)
        combined.loc[both] = np.exp((1 - w.loc[both]) * np.log(baseline.loc[both]) + w.loc[both] * np.log(trend.loc[both]))
        trend_only = trend_valid & ~base_valid
        combined.loc[trend_only] = trend.loc[trend_only]
        combined = pd.to_numeric(combined, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(self.global_median_)
        combined = combined.clip(lower=1.0)

        out = X.copy()
        out["location_baseline_m2"] = combined.astype(float)
        out["location_baseline_log"] = np.log(out["location_baseline_m2"])
        out["location_baseline_count"] = count.astype(float)
        out["location_baseline_level_code"] = level_code.astype(float)
        out["location_baseline_vs_trend"] = np.where(
            trend_valid,
            out["location_baseline_m2"] / trend.replace(0, np.nan),
            1.0,
        )
        out["location_baseline_vs_trend"] = pd.to_numeric(out["location_baseline_vs_trend"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
        return out


class LocationResidualRegressor(BaseEstimator, RegressorMixin):
    """Predicts final unit price by learning residual over a local market baseline."""

    def __init__(self, estimator: Any, baseline_encoder: Any | None = None):
        self.estimator = estimator
        self.baseline_encoder = baseline_encoder

    def fit(self, X: pd.DataFrame, y: Any):
        y_arr = np.asarray(y, dtype=float)
        self.baseline_encoder_ = clone(self.baseline_encoder) if self.baseline_encoder is not None else LocationBaselineEncoder()
        self.baseline_encoder_.fit(X, y_arr)
        Xb = self.baseline_encoder_.transform(X)
        baseline = pd.to_numeric(Xb["location_baseline_m2"], errors="coerce").to_numpy(dtype=float)
        baseline = np.clip(baseline, 1.0, None)
        residual_target = np.log(np.clip(y_arr, 1.0, None)) - np.log(baseline)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(Xb, residual_target)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xb = self.baseline_encoder_.transform(X)
        baseline = pd.to_numeric(Xb["location_baseline_m2"], errors="coerce").to_numpy(dtype=float)
        baseline = np.clip(baseline, 1.0, None)
        residual_pred = np.asarray(self.estimator_.predict(Xb), dtype=float)
        pred = np.exp(np.log(baseline) + residual_pred)
        return np.clip(pred, 1.0, None)


def apply_location_outlier_filter(sales: pd.DataFrame, cfg: RunConfig, out_dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Removes listings whose price is too far from the local market baseline.

    This is intentionally conservative and app-safe: it uses only location/room/m2/trend signals,
    not title text or premium user-unavailable labels.
    """
    if not cfg.use_location_outlier_filter or sales.empty:
        return sales.copy(), pd.DataFrame(), {"location_outlier_filter_enabled": False}

    y = pd.to_numeric(sales[TARGET], errors="coerce")
    X = sales.drop(columns=[TARGET], errors="ignore").copy()
    enc = LocationBaselineEncoder(
        min_specific_count=max(5, min(cfg.location_min_group_size, 12)),
        min_medium_count=max(8, cfg.location_min_group_size),
        min_broad_count=max(15, cfg.location_min_group_size + 5),
    ).fit(X, y)
    enriched = enc.transform(X)
    baseline = pd.to_numeric(enriched["location_baseline_m2"], errors="coerce")
    ratio = y / baseline.replace(0, np.nan)
    log_ratio = np.log(ratio.replace(0, np.nan))

    keep = y.notna() & baseline.notna() & ratio.notna() & np.isfinite(ratio) & ratio.between(
        cfg.min_location_ratio,
        cfg.max_location_ratio,
        inclusive="both",
    )

    # Additional group-wise robust MAD filter on log(price / baseline).
    group_cols = [c for c in ["county", "room_count", "m2_group"] if c in sales.columns]
    if group_cols:
        robust_keep = pd.Series(True, index=sales.index)
        work = sales[group_cols].copy()
        work["__log_ratio__"] = log_ratio
        for _, idx in work.groupby(group_cols, dropna=False).groups.items():
            idx = list(idx)
            if len(idx) < int(cfg.location_min_group_size):
                continue
            vals = pd.to_numeric(work.loc[idx, "__log_ratio__"], errors="coerce")
            med = vals.median()
            mad = np.median(np.abs(vals.dropna() - med)) if vals.notna().any() else np.nan
            if not np.isfinite(mad) or mad <= 1e-9:
                continue
            z = 0.6745 * (vals - med) / mad
            robust_keep.loc[idx] &= z.abs() <= float(cfg.location_mad_threshold)
        keep &= robust_keep

    out = sales.copy()
    out["location_baseline_m2_for_filter"] = baseline
    out["target_to_location_baseline_ratio"] = ratio
    out["location_log_ratio_for_filter"] = log_ratio
    removed = out.loc[~keep].copy()
    clean = sales.loc[keep].copy()

    removed.to_csv(out_dirs["input"] / "sales_removed_location_outliers_v8.csv", index=False, encoding="utf-8-sig")
    clean.to_csv(out_dirs["input"] / "sales_after_location_outlier_filter_v8.csv", index=False, encoding="utf-8-sig")

    report = {
        "location_outlier_filter_enabled": True,
        "rows_before_location_filter": int(len(sales)),
        "rows_after_location_filter": int(len(clean)),
        "rows_removed_location_filter": int(len(removed)),
        "min_location_ratio": float(cfg.min_location_ratio),
        "max_location_ratio": float(cfg.max_location_ratio),
        "location_mad_threshold": float(cfg.location_mad_threshold),
        "location_min_group_size": int(cfg.location_min_group_size),
        "removed_ratio_summary": {
            "min": float(ratio.loc[~keep].min()) if (~keep).any() and ratio.loc[~keep].notna().any() else None,
            "median": float(ratio.loc[~keep].median()) if (~keep).any() and ratio.loc[~keep].notna().any() else None,
            "max": float(ratio.loc[~keep].max()) if (~keep).any() and ratio.loc[~keep].notna().any() else None,
        },
    }
    return clean.reset_index(drop=True), removed.reset_index(drop=True), report

# =========================
# Model training
# =========================


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_ohe()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def build_pipeline(model: Any, scale_numeric: bool, target_mode: str) -> Any:
    pipe = Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("target_stats", TargetStatsAdder(smoothing=20.0)),
            ("feature_columns", FeatureColumnKeeper(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
            ("preprocess", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )
    if target_mode == "residual":
        return LocationResidualRegressor(pipe, baseline_encoder=LocationBaselineEncoder())
    if target_mode == "log":
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1, check_inverse=False)
    if target_mode == "raw":
        return pipe
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def model_specs(target_mode: str, selected_models: list[str] | None = None, fast_mode: bool = False) -> dict[str, Any]:
    if selected_models is None:
        selected_models = ["ridge", "gradient_boosting", "extra_trees", "random_forest"]

    # Moderate defaults: strong enough for ~5k rows, not painfully slow on a laptop.
    gb_estimators = 120 if fast_mode else 300
    hgb_iters = 150 if fast_mode else 320
    et_estimators = 160 if fast_mode else 350
    rf_estimators = 140 if fast_mode else 300

    registry = {
        "ridge": build_pipeline(
            RidgeCV(alphas=np.logspace(-2, 3, 16)),
            scale_numeric=True,
            target_mode=target_mode,
        ),
        "gradient_boosting": build_pipeline(
            GradientBoostingRegressor(
                n_estimators=gb_estimators,
                learning_rate=0.04,
                max_depth=3,
                min_samples_leaf=8,
                subsample=0.85,
                random_state=RANDOM_STATE,
            ),
            scale_numeric=False,
            target_mode=target_mode,
        ),
        "hist_gradient_boosting": build_pipeline(
            HistGradientBoostingRegressor(
                max_iter=hgb_iters,
                learning_rate=0.04,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                min_samples_leaf=15,
                random_state=RANDOM_STATE,
            ),
            scale_numeric=False,
            target_mode=target_mode,
        ),
        "extra_trees": build_pipeline(
            ExtraTreesRegressor(
                n_estimators=et_estimators,
                min_samples_leaf=2,
                max_features=0.70,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            scale_numeric=False,
            target_mode=target_mode,
        ),
        "random_forest": build_pipeline(
            RandomForestRegressor(
                n_estimators=rf_estimators,
                min_samples_leaf=2,
                max_features=0.75,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            scale_numeric=False,
            target_mode=target_mode,
        ),
    }

    unknown = [m for m in selected_models if m not in registry]
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}. Available: {sorted(registry)}")
    return {name: registry[name] for name in selected_models}

def make_training_matrix(sales: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = pd.to_numeric(sales[TARGET], errors="coerce")
    X = sales.drop(columns=[c for c in LEAKAGE_OR_UNUSED_COLUMNS if c in sales.columns], errors="ignore").copy()
    valid = y.notna() & np.isfinite(y) & (y > 0)
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)


def choose_ensemble_weights(model_metrics: pd.DataFrame, max_models: int = 3) -> dict[str, float]:
    valid = model_metrics.dropna(subset=["mape"]).sort_values(["mape", "mae_tl_per_m2"], ascending=True)
    if valid.empty:
        raise ValueError("No valid model metrics were produced.")
    top = valid.head(max_models).copy()
    inv = 1 / np.maximum(top["mape"].to_numpy(dtype=float), 1e-6)
    weights = inv / inv.sum()
    return dict(zip(top["model"], weights.astype(float)))


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, cfg: RunConfig, out_dirs: dict[str, Path]) -> tuple[ModelBundle, dict[str, Any]]:
    if len(X) < 10:
        raise ValueError(f"Too few sale rows to train: {len(X)}")

    n_splits = min(cfg.n_splits, max(2, len(X) // 20))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
    specs = model_specs(cfg.target_mode, selected_models=cfg.selected_models, fast_mode=cfg.fast_mode)

    oof = pd.DataFrame({"actual_unit_price_gross": y})
    comparison_rows = []
    fitted_models: dict[str, Any] = {}

    for name, estimator in specs.items():
        print(f"Training OOF model: {name}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = cross_val_predict(clone(estimator), X, y, cv=cv, n_jobs=None, method="predict")
        pred = np.maximum(np.asarray(pred, dtype=float), 0)
        oof[f"pred_{name}"] = pred
        metrics = metric_dict(y, pred)
        metrics["model"] = name
        comparison_rows.append(metrics)

        print(f"Fitting final model: {name}")
        final_estimator = clone(estimator)
        final_estimator.fit(X, y)
        fitted_models[name] = final_estimator
        joblib.dump(final_estimator, out_dirs["artifacts"] / f"model_{name}_v8.joblib")

    model_comparison = pd.DataFrame(comparison_rows).sort_values(["mape", "mae_tl_per_m2"])
    model_comparison.to_csv(out_dirs["reports"] / "model_comparison_v8.csv", index=False, encoding="utf-8-sig")

    weights = choose_ensemble_weights(model_comparison, max_models=3)
    ensemble_pred = np.zeros(len(oof), dtype=float)
    total_w = sum(weights.values())
    for name, w in weights.items():
        ensemble_pred += oof[f"pred_{name}"].to_numpy(dtype=float) * (w / total_w)
    oof["pred_ensemble"] = ensemble_pred

    ensemble_metrics = metric_dict(y, ensemble_pred)
    ensemble_metrics["model"] = "ensemble_top_models_v8"
    ensemble_metrics["weights"] = weights
    ensemble_metrics["target_mode"] = cfg.target_mode
    ensemble_metrics["n_splits"] = n_splits

    selected_models = {name: fitted_models[name] for name in weights}
    bundle = ModelBundle(selected_models, weights, list(X.columns), ensemble_metrics)
    joblib.dump(bundle, out_dirs["artifacts"] / "ensemble_model_bundle_v8.joblib")

    oof["error"] = oof["pred_ensemble"] - oof["actual_unit_price_gross"]
    oof["abs_error"] = oof["error"].abs()
    oof["abs_pct_error"] = oof["abs_error"] / oof["actual_unit_price_gross"]
    oof = pd.concat([oof, X.reset_index(drop=True)], axis=1)
    oof.to_csv(out_dirs["output"] / "oof_predictions_v8.csv", index=False, encoding="utf-8-sig")

    metrics_summary = {
        "ensemble": ensemble_metrics,
        "model_comparison": model_comparison.to_dict(orient="records"),
    }
    (out_dirs["reports"] / "metrics_summary_v8.json").write_text(
        json.dumps(metrics_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    save_group_error_reports(oof, out_dirs["reports"])
    save_plots(oof, out_dirs["reports"])
    save_feature_importance(selected_models, out_dirs["reports"])

    return bundle, metrics_summary


# =========================
# Reporting
# =========================


def save_group_error_reports(oof: pd.DataFrame, reports_dir: Path) -> None:
    group_cols = ["county", "district", "room_count", "m2_group", "building_age_group", "floor_segment", "heating", "site_inside"]
    for col in group_cols:
        if col not in oof.columns:
            continue
        rep = (
            oof.groupby(col, dropna=False)
            .agg(
                n=("actual_unit_price_gross", "size"),
                mape=("abs_pct_error", "mean"),
                median_ape=("abs_pct_error", "median"),
                mae_tl_per_m2=("abs_error", "mean"),
                median_ae_tl_per_m2=("abs_error", "median"),
                mean_actual_unit_price=("actual_unit_price_gross", "mean"),
                mean_predicted_unit_price=("pred_ensemble", "mean"),
            )
            .reset_index()
            .sort_values(["mape", "n"], ascending=[False, False])
        )
        rep.to_csv(reports_dir / f"error_by_{col}_v8.csv", index=False, encoding="utf-8-sig")


def save_plots(oof: pd.DataFrame, reports_dir: Path) -> None:
    x = oof["actual_unit_price_gross"].astype(float)
    y = oof["pred_ensemble"].astype(float)
    min_val = float(min(x.min(), y.min()))
    max_val = float(max(x.max(), y.max()))

    plt.figure(figsize=(7.5, 7.5))
    plt.scatter(x, y, s=18, alpha=0.60)
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1.5)
    plt.xlabel("Gerçek Birim Fiyat (TL/m²)")
    plt.ylabel("Tahmin Edilen Birim Fiyat (TL/m²)")
    plt.title("V8 OOF: Gerçek ve Tahmin Edilen Birim Fiyat")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig(reports_dir / "actual_vs_predicted_v8.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(oof["error"].dropna(), bins=35, alpha=0.85)
    plt.xlabel("Tahmin Hatası (TL/m²)")
    plt.ylabel("İlan Sayısı")
    plt.title("V8 OOF: Hata Dağılımı")
    plt.grid(True, axis="y", linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig(reports_dir / "residual_distribution_v8.png", dpi=300, bbox_inches="tight")
    plt.close()

    if "district" in oof.columns:
        rep = oof.groupby("district").agg(n=("abs_pct_error", "size"), mape=("abs_pct_error", "mean")).reset_index()
        rep = rep[rep["n"] >= 5].sort_values("mape", ascending=True).tail(20)
        if not rep.empty:
            plt.figure(figsize=(9, 7))
            plt.barh(rep["district"].astype(str), rep["mape"] * 100)
            plt.xlabel("MAPE (%)")
            plt.title("V8 OOF: Mahalle Bazlı En Yüksek MAPE")
            plt.grid(True, axis="x", linestyle="--", alpha=0.45)
            plt.tight_layout()
            plt.savefig(reports_dir / "mape_by_district_v8.png", dpi=300, bbox_inches="tight")
            plt.close()


def unwrap_pipeline(model: Any) -> Any:
    if hasattr(model, "estimator_") and isinstance(model, LocationResidualRegressor):
        return model.estimator_
    if hasattr(model, "regressor_"):
        return model.regressor_
    return model


def get_preprocess_feature_names(model: Any) -> list[str] | None:
    try:
        pipe = unwrap_pipeline(model)
        preprocess = pipe.named_steps["preprocess"]
        names = preprocess.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        return None


def get_inner_model(model: Any) -> Any | None:
    try:
        pipe = unwrap_pipeline(model)
        return pipe.named_steps["model"]
    except Exception:
        return None


def save_feature_importance(models: dict[str, Any], reports_dir: Path) -> None:
    rows = []
    for name, model in models.items():
        inner = get_inner_model(model)
        if inner is None or not hasattr(inner, "feature_importances_"):
            continue
        feat_names = get_preprocess_feature_names(model)
        if feat_names is None:
            continue
        importances = np.asarray(inner.feature_importances_, dtype=float)
        n = min(len(feat_names), len(importances))
        for feat, imp in zip(feat_names[:n], importances[:n]):
            rows.append({"model": name, "feature": feat, "importance": float(imp)})
    if not rows:
        return
    rep = pd.DataFrame(rows).sort_values(["model", "importance"], ascending=[True, False])
    rep.to_csv(reports_dir / "feature_importance_v8.csv", index=False, encoding="utf-8-sig")
    top = rep.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False).head(50)
    top.to_csv(reports_dir / "feature_importance_top50_v8.csv", index=False, encoding="utf-8-sig")


# =========================
# Main orchestration
# =========================


def build_output_dirs(base_out: str | Path) -> dict[str, Path]:
    base = Path(base_out)
    dirs = {
        "base": base,
        "raw": base / "data" / "raw",
        "input": base / "data" / "input",
        "output": base / "data" / "output",
        "reports": base / "reports",
        "artifacts": base / "artifacts",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def write_run_readme(out_dirs: dict[str, Path], cfg: RunConfig, metrics_summary: dict[str, Any], cleaning_report: dict[str, Any], feature_reports: dict[str, Any]) -> None:
    ensemble = metrics_summary.get("ensemble", {})
    lines = [
        "# V8 Model Run",
        "",
        "## Config",
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False),
        "",
        "## Cleaning report",
        json.dumps(cleaning_report, indent=2, ensure_ascii=False),
        "",
        "## Feature reports",
        json.dumps(feature_reports, indent=2, ensure_ascii=False),
        "",
        "## Ensemble metrics",
        json.dumps(ensemble, indent=2, ensure_ascii=False),
        "",
        "## Main outputs",
        "- data/raw/sales_raw_from_source.csv",
        "- data/raw/rentals_raw_from_source.csv",
        "- data/input/sales_cleaned_v8.csv",
        "- data/input/rentals_cleaned_v8.csv",
        "- data/output/oof_predictions_v8.csv",
        "- reports/model_comparison_v8.csv",
        "- reports/metrics_summary_v8.json",
        "- reports/error_by_*_v8.csv",
        "- reports/*.png",
        "- artifacts/model_*_v8.joblib",
        "- artifacts/ensemble_model_bundle_v8.joblib",
        "",
    ]
    (out_dirs["base"] / "README_v8_run.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", default=None, help="PostgreSQL/Neon connection string. Empty reads DATABASE_URL or DB_URL constant.")
    ap.add_argument("--sale-table", default=DEFAULT_SALE_TABLE)
    ap.add_argument("--rental-table", default=DEFAULT_RENTAL_TABLE)
    ap.add_argument("--trend-table", default=DEFAULT_TREND_TABLE)
    ap.add_argument("--city", default=DEFAULT_CITY)
    ap.add_argument("--counties", default=",".join(DEFAULT_COUNTIES), help="Comma-separated county list.")
    ap.add_argument("--out", default="outputs/v8_db_residual_pipeline")
    ap.add_argument("--target-mode", choices=["residual", "log", "raw"], default="residual")
    ap.add_argument("--models", default="ridge,gradient_boosting,extra_trees,random_forest", help="Comma-separated model names. Available: ridge, gradient_boosting, hist_gradient_boosting, extra_trees, random_forest.")
    ap.add_argument("--fast", action="store_true", help="Use lighter model settings for quick experiments.")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--use-trend", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--trend-max-date", default=None, help="Optional upper date for trend rows. Example: 2026-06-01")
    ap.add_argument("--limit-sale", type=int, default=None)
    ap.add_argument("--limit-rental", type=int, default=None)
    ap.add_argument("--sale-json", default=None, help="Optional local sale JSON for testing instead of DB.")
    ap.add_argument("--rental-json", default=None, help="Optional local rental JSON for testing instead of DB.")
    ap.add_argument("--min-sale-unit-price", type=float, default=8_000)
    ap.add_argument("--max-sale-unit-price", type=float, default=200_000)
    ap.add_argument("--min-rent-m2", type=float, default=50)
    ap.add_argument("--max-rent-m2", type=float, default=2_500)
    ap.add_argument("--location-outlier-filter", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min-location-ratio", type=float, default=0.50)
    ap.add_argument("--max-location-ratio", type=float, default=1.90)
    ap.add_argument("--location-mad-threshold", type=float, default=3.50)
    ap.add_argument("--location-min-group-size", type=int, default=12)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(
        city=args.city,
        counties=parse_counties(args.counties),
        target_mode=args.target_mode,
        n_splits=args.n_splits,
        random_state=RANDOM_STATE,
        sale_table=args.sale_table,
        rental_table=args.rental_table,
        trend_table=args.trend_table,
        use_trend=bool(args.use_trend),
        selected_models=[m.strip() for m in args.models.split(",") if m.strip()],
        fast_mode=bool(args.fast),
        min_sale_unit_price=float(args.min_sale_unit_price),
        max_sale_unit_price=float(args.max_sale_unit_price),
        min_rent_m2=float(args.min_rent_m2),
        max_rent_m2=float(args.max_rent_m2),
        use_location_outlier_filter=bool(args.location_outlier_filter),
        min_location_ratio=float(args.min_location_ratio),
        max_location_ratio=float(args.max_location_ratio),
        location_mad_threshold=float(args.location_mad_threshold),
        location_min_group_size=int(args.location_min_group_size),
    )

    out_dirs = build_output_dirs(args.out)
    (out_dirs["base"] / "run_config_v8.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")

    print("Loading raw data...")
    sales_raw, rentals_raw, trend_raw = load_raw_data(args, out_dirs)

    print("Cleaning sales and rentals...")
    sales_clean, rentals_clean, cleaning_report = clean_sales_and_rentals(sales_raw, rentals_raw, cfg, out_dirs)
    print(json.dumps(cleaning_report, indent=2, ensure_ascii=False))

    print("Attaching rental market features...")
    sales_with_rent, rental_feature_report = attach_rental_features(sales_clean, rentals_clean)

    print("Attaching trend market features...")
    sales_final, trend_feature_report = attach_trend_features(sales_with_rent, trend_raw, cfg)

    print("Applying location-baseline outlier filter...")
    sales_final, sales_removed_location, location_outlier_report = apply_location_outlier_filter(sales_final, cfg, out_dirs)
    cleaning_report.update(location_outlier_report)
    sales_final.to_csv(out_dirs["input"] / "sales_training_table_v8.csv", index=False, encoding="utf-8-sig")

    X, y = make_training_matrix(sales_final)
    print(f"Training rows: {len(X)} | Features before pipeline: {X.shape[1]}")
    if len(X) == 0:
        raise ValueError("No valid sale rows left after cleaning.")

    bundle, metrics_summary = train_and_evaluate(X, y, cfg, out_dirs)

    feature_reports = {
        "rental_features": rental_feature_report,
        "trend_features": trend_feature_report,
    }
    write_run_readme(out_dirs, cfg, metrics_summary, cleaning_report, feature_reports)

    print("\n=== V8 ENSEMBLE METRICS ===")
    print(json.dumps(metrics_summary["ensemble"], indent=2, ensure_ascii=False))
    print(f"\nOutputs written to: {out_dirs['base'].resolve()}")


if __name__ == "__main__":
    main()
