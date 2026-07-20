from __future__ import annotations

"""
V11.3 DB -> clean -> residual-target train -> anomaly/county-expert + demographics pipeline for Kocaeli housing unit price model.

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
  python train_v11_demographics_pipeline.py ^
    --out outputs/v11_kocaeli ^
    --city Kocaeli ^
    --counties "İzmit,Başiskele,Gölcük,Karamürsel" ^
    --sale-table sale_listings ^
    --rental-table rental_listings ^
    --trend-table trend_observed

Local JSON test example:
  python train_v11_demographics_pipeline.py ^
    --sale-json "sale_listings (2).json" ^
    --rental-json "rental_listings (1).json" ^
    --out outputs/v11_local_test
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
import matplotlib
matplotlib.use("Agg")
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
from sklearn.linear_model import Ridge, RidgeCV
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

DEMOGRAPHIC_SAFE_NUMERIC_FEATURES = [
    "demo_population_total", "demo_population_male", "demo_population_female", "demo_population_density",
    "demo_young_ratio", "demo_middle_ratio", "demo_old_ratio", "demo_young_count", "demo_middle_count", "demo_old_count",
    "demo_female_ratio", "demo_male_ratio",
    "demo_married_ratio", "demo_single_ratio", "demo_divorced_ratio", "demo_widow_ratio",
    "demo_never_married_count", "demo_married_count", "demo_divorced_count", "demo_widow_count",
    "demo_age_0_14", "demo_age_15_24", "demo_age_25_34", "demo_age_35_44", "demo_age_45_54", "demo_age_55_64", "demo_age_65_plus",
    "demo_age_0_14_count", "demo_age_15_24_count", "demo_age_25_34_count", "demo_age_35_44_count", "demo_age_45_54_count", "demo_age_55_64_count", "demo_age_65_plus_count",
    "demo_education_total", "demo_education_university_ratio", "demo_education_university_count",
    "demo_education_high_school_ratio", "demo_education_high_school_count",
    "demo_education_middle_school_ratio", "demo_education_middle_school_count",
    "demo_education_primary_school_ratio", "demo_education_primary_school_count",
    "demo_education_primary_education_ratio", "demo_education_graduate_ratio", "demo_education_graduate_count",
    "demo_education_doctorate_ratio", "demo_education_doctorate_count",
    "demo_education_non_literate_ratio", "demo_education_unknown_ratio",
    "demo_ses_a_plus_count", "demo_ses_a_count", "demo_ses_b_count", "demo_ses_c_count", "demo_ses_d_count",
    "demo_ses_a_plus_ratio", "demo_ses_a_ratio", "demo_ses_b_ratio", "demo_ses_c_ratio", "demo_ses_d_ratio", "demo_ses_ab_ratio", "demo_ses_cd_ratio",
    "demo_household_count", "demo_household_size", "demo_per_capita_income_try", "demo_household_income_try",
    "demo_residential_count", "demo_workplace_count", "demo_summer_house_count",
    "demo_vehicle_count", "demo_car_count", "demo_atm_count", "demo_pharmacy_count", "demo_bank_count",
    "county_demo_population_total_sum", "county_demo_population_density_median", "county_demo_population_density_mean",
    "county_demo_per_capita_income_median", "county_demo_household_income_median", "county_demo_household_size_mean",
    "county_demo_education_university_ratio_mean", "county_demo_education_high_school_ratio_mean",
    "county_demo_ses_ab_ratio_mean", "county_demo_ses_cd_ratio_mean",
    "county_demo_old_ratio_mean", "county_demo_young_ratio_mean", "county_demo_married_ratio_mean", "county_demo_single_ratio_mean",
    "county_demo_residential_count_sum", "county_demo_workplace_count_sum",
    "county_demo_vehicle_count_sum", "county_demo_car_count_sum", "county_demo_atm_count_sum", "county_demo_pharmacy_count_sum", "county_demo_bank_count_sum",
    "demo_income_vs_county", "demo_household_income_vs_county", "demo_density_vs_county",
    "demo_university_diff_county", "demo_high_school_diff_county", "demo_ses_ab_diff_county", "demo_ses_cd_diff_county",
    "demo_old_ratio_diff_county", "demo_young_ratio_diff_county",
    "demo_residential_density", "demo_workplace_density", "demo_vehicle_per_capita", "demo_car_per_capita",
    "demo_pharmacy_per_10k", "demo_bank_per_10k", "demo_atm_per_10k",
    "demo_has_demographics", "demo_has_county_demographics", "demo_age_coverage", "demo_education_coverage", "demo_income_available",
    "demo_ses_available", "demo_market_activity_available", "demo_infrastructure_available", "demo_coverage_score",
]

DEMOGRAPHIC_FULL_EXTRA_NUMERIC_FEATURES = [
    "demo_real_estate_agent_count", "demo_agent_listing_count", "demo_owner_listing_count",
    "demo_sale_count", "demo_mortgage_count", "demo_turnover_ratio", "demo_computed_turnover_ratio",
    "demo_listing_count_2024", "demo_bb_sale_count_2024", "demo_bb_mortgaged_sale_count_2024",
    "county_demo_sale_count_sum", "county_demo_turnover_ratio_median", "demo_turnover_vs_county",
    "demo_saving_total", "demo_expense_total", "demo_expense_food", "demo_expense_shelter",
    "demo_expense_transportation", "demo_expense_education", "demo_ecommerce_count", "demo_ecommerce_density", "demo_online_retail",
]

DEMOGRAPHIC_CATEGORICAL_FEATURES = [
    "demo_dominant_age_group", "demo_dominant_marital_status", "demo_dominant_education", "demo_ses_group",
]

# Kept globally so saved bundles can accept any V11 demographics mode at prediction time.
NUMERIC_FEATURES = NUMERIC_FEATURES + DEMOGRAPHIC_SAFE_NUMERIC_FEATURES + DEMOGRAPHIC_FULL_EXTRA_NUMERIC_FEATURES
CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + DEMOGRAPHIC_CATEGORICAL_FEATURES

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
    enable_county_experts: bool
    county_expert_min_rows: int
    enable_anomaly_reports: bool
    demographics_mode: str
    demographics_table: str
    exclude_anomalies_threshold: float


class ModelBundle:
    """Prediction wrapper saved with the final ensemble.

    V11 supports both segment-aware and county-expert layers. The base ensemble
    predicts first, then validated segment blends are applied, and finally
    validated county-specific blends are applied. County experts are only used
    for counties whose OOF blend improved during training.
    """

    def __init__(
        self,
        models: dict[str, Any],
        weights: dict[str, float],
        feature_columns: list[str],
        metrics: dict[str, Any],
        segment_models: dict[str, dict[str, Any]] | None = None,
        segment_weights: dict[str, dict[str, float]] | None = None,
        segment_blend_weights: dict[str, float] | None = None,
        county_models: dict[str, dict[str, Any]] | None = None,
        county_weights: dict[str, dict[str, float]] | None = None,
        county_blend_weights: dict[str, float] | None = None,
    ):
        self.models = models
        self.weights = weights
        self.feature_columns = feature_columns
        self.metrics = metrics
        self.segment_models = segment_models or {}
        self.segment_weights = segment_weights or {}
        self.segment_blend_weights = segment_blend_weights or {}
        self.county_models = county_models or {}
        self.county_weights = county_weights or {}
        self.county_blend_weights = county_blend_weights or {}

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = np.nan
        return X[self.feature_columns]

    def _weighted_predict(self, X: pd.DataFrame, models: dict[str, Any], weights: dict[str, float]) -> np.ndarray:
        pred = np.zeros(len(X), dtype=float)
        total_w = sum(weights.values())
        for name, model in models.items():
            pred += np.asarray(model.predict(X), dtype=float) * (weights[name] / total_w)
        return pred

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw_X = X.copy()
        aligned_X = self._align(raw_X)
        pred = self._weighted_predict(aligned_X, self.models, self.weights)

        for segment_name, models in self.segment_models.items():
            mask = segment_mask(raw_X, segment_name)
            if not mask.any():
                continue
            weights = self.segment_weights.get(segment_name, {})
            if not weights:
                continue
            seg_pred = self._weighted_predict(aligned_X.loc[mask], models, weights)
            blend = float(self.segment_blend_weights.get(segment_name, 0.35))
            pred[mask.to_numpy()] = (1.0 - blend) * pred[mask.to_numpy()] + blend * seg_pred

        if "county" in raw_X.columns:
            county_values = raw_X["county"].fillna("missing").astype(str)
            for county_name, models in self.county_models.items():
                mask = county_values.eq(str(county_name))
                if not mask.any():
                    continue
                weights = self.county_weights.get(county_name, {})
                if not weights:
                    continue
                county_pred = self._weighted_predict(aligned_X.loc[mask], models, weights)
                blend = float(self.county_blend_weights.get(county_name, 0.35))
                pred[mask.to_numpy()] = (1.0 - blend) * pred[mask.to_numpy()] + blend * county_pred

        return np.maximum(pred, 0)


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



def fetch_demographics_table(engine, table: str, city: str | None = None) -> pd.DataFrame:
    """Fetch district-level demographic data from PostgreSQL.

    The project keeps external reference IDs directly in ref/demographic id columns, so
    city_id/county_id/district_id are used as stable join keys.
    """
    table = validate_table_name(table)
    city_filter = "WHERE lower(coalesce(city_name, '')) = lower(:city)" if city else ""
    params = {"city": city} if city else {}
    sql = text(
        f"""
        SELECT *
        FROM {table}
        {city_filter}
        """
    )
    try:
        return pd.read_sql(sql, engine, params=params)
    except Exception as exc:
        warnings.warn(f"Demographics table could not be fetched; continuing without demographics. Error: {exc}")
        return pd.DataFrame()


def safe_divide_series(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    out = num / den.replace({0: np.nan})
    return out.replace([np.inf, -np.inf], np.nan)


def normalize_location_text(value: Any) -> str:
    """Robust Turkish location normalizer used only for demographic fallback joins."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text_value = str(value).strip().casefold()
    tr_map = str.maketrans({
        "ç": "c", "ğ": "g", "ı": "i", "i": "i", "ö": "o", "ş": "s", "ü": "u",
        "â": "a", "î": "i", "û": "u",
    })
    text_value = text_value.translate(tr_map)
    text_value = re.sub(r"\b(mahallesi|mahalle|mah\.?|mh\.?)\b", " ", text_value)
    text_value = re.sub(r"[^0-9a-z]+", " ", text_value)
    text_value = re.sub(r"\s+", " ", text_value).strip()
    return text_value


def add_location_norm_keys(df: pd.DataFrame, *, prefix: str = "") -> pd.DataFrame:
    out = df.copy()
    candidates = {
        "city": ["city", "city_name"],
        "county": ["county", "county_name"],
        "district": ["district", "district_name"],
    }
    for canonical, names in candidates.items():
        src = next((c for c in names if c in out.columns), None)
        col_name = f"{prefix}{canonical}_norm"
        if src is None:
            out[col_name] = ""
        else:
            out[col_name] = out[src].map(normalize_location_text)
    return out


def _to_numeric_columns(df: pd.DataFrame, skip: set[str]) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in skip:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_demographic_features(demo_raw: pd.DataFrame, mode: str = "safe") -> pd.DataFrame:
    """Create neighborhood + county demographic features.

    mode='safe' uses slower-moving demographic/infrastructure fields.
    mode='full' also keeps market-activity, spending and e-commerce proxies.
    mode='none' returns only join keys.
    """
    if demo_raw is None or demo_raw.empty:
        return pd.DataFrame()
    mode = (mode or "safe").lower().strip()
    if mode == "none":
        return pd.DataFrame()

    df = demo_raw.copy()
    for key in ["city_id", "county_id", "district_id"]:
        if key not in df.columns:
            warnings.warn(f"Demographics table is missing join key: {key}; demographics disabled.")
            return pd.DataFrame()
        df[key] = pd.to_numeric(df[key], errors="coerce").astype("Int64")

    # Columns copied from district_demographics to demo_* names.
    safe_base = [
        "population_total", "population_male", "population_female", "population_density",
        "young_ratio", "middle_ratio", "old_ratio", "young_count", "middle_count", "old_count",
        "female_ratio", "male_ratio",
        "married_ratio", "single_ratio", "divorced_ratio", "widow_ratio",
        "never_married_count", "married_count", "divorced_count", "widow_count",
        "age_0_14", "age_15_24", "age_25_34", "age_35_44", "age_45_54", "age_55_64", "age_65_plus",
        "age_0_14_count", "age_15_24_count", "age_25_34_count", "age_35_44_count", "age_45_54_count", "age_55_64_count", "age_65_plus_count",
        "education_total", "education_university_ratio", "education_university_count",
        "education_high_school_ratio", "education_high_school_count", "education_middle_school_ratio", "education_middle_school_count",
        "education_primary_school_ratio", "education_primary_school_count", "education_primary_education_ratio",
        "education_graduate_ratio", "education_graduate_count", "education_doctorate_ratio", "education_doctorate_count",
        "education_non_literate_ratio", "education_unknown_ratio",
        "ses_a_plus_count", "ses_a_count", "ses_b_count", "ses_c_count", "ses_d_count",
        "ses_a_plus_ratio", "ses_a_ratio", "ses_b_ratio", "ses_c_ratio", "ses_d_ratio", "ses_ab_ratio", "ses_cd_ratio",
        "household_count", "household_size", "per_capita_income_try", "household_income_try",
        "residential_count", "workplace_count", "summer_house_count",
        "vehicle_count", "car_count", "atm_count", "pharmacy_count", "bank_count",
    ]
    full_extra = [
        "real_estate_agent_count", "agent_listing_count", "owner_listing_count",
        "sale_count", "mortgage_count", "turnover_ratio", "computed_turnover_ratio",
        "listing_count_2024", "bb_sale_count_2024", "bb_mortgaged_sale_count_2024",
        "saving_total", "expense_total", "expense_food", "expense_shelter", "expense_transportation", "expense_education",
        "ecommerce_count", "ecommerce_density", "online_retail",
    ]
    categorical = ["dominant_age_group", "dominant_marital_status", "dominant_education", "ses_group"]
    use_cols = safe_base + (full_extra if mode == "full" else [])

    out = df[["city_id", "county_id", "district_id"]].copy()
    # Keep source names for fallback merge when listing tables do not contain ref IDs.
    for src_name, out_name in [
        ("city_name", "demo_city_name"),
        ("county_name", "demo_county_name"),
        ("district_name", "demo_district_name"),
    ]:
        if src_name in df.columns:
            out[out_name] = df[src_name].astype("object").where(df[src_name].notna(), "").astype(str)
        else:
            out[out_name] = ""
    out = add_location_norm_keys(out.rename(columns={
        "demo_city_name": "city_name",
        "demo_county_name": "county_name",
        "demo_district_name": "district_name",
    }), prefix="demo_").rename(columns={
        "city_name": "demo_city_name",
        "county_name": "demo_county_name",
        "district_name": "demo_district_name",
    })

    for c in use_cols:
        out[f"demo_{c}"] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    for c in categorical:
        out[f"demo_{c}"] = df[c].astype("object") if c in df.columns else "missing"
        out[f"demo_{c}"] = out[f"demo_{c}"].where(pd.notna(out[f"demo_{c}"]), "missing").astype(str).str.strip().replace({"": "missing", "nan": "missing", "None": "missing"})

    # Coverage flags.
    age_cols = ["demo_age_0_14", "demo_age_15_24", "demo_age_25_34", "demo_age_35_44", "demo_age_45_54", "demo_age_55_64", "demo_age_65_plus"]
    edu_cols = ["demo_education_university_ratio", "demo_education_high_school_ratio", "demo_education_middle_school_ratio", "demo_education_primary_school_ratio", "demo_education_graduate_ratio", "demo_education_doctorate_ratio", "demo_education_non_literate_ratio"]
    out["demo_has_demographics"] = out[["demo_population_total", "demo_population_density", "demo_per_capita_income_try"]].notna().any(axis=1).astype(int)
    out["demo_age_coverage"] = out[age_cols].sum(axis=1, min_count=1).between(80, 120).fillna(False).astype(int)
    out["demo_education_coverage"] = out[edu_cols].sum(axis=1, min_count=1).between(70, 120).fillna(False).astype(int)
    out["demo_income_available"] = out[["demo_per_capita_income_try", "demo_household_income_try"]].notna().any(axis=1).astype(int)
    out["demo_ses_available"] = out[["demo_ses_ab_ratio", "demo_ses_cd_ratio"]].notna().any(axis=1).astype(int)
    market_cols = [c for c in ["demo_sale_count", "demo_listing_count_2024", "demo_turnover_ratio"] if c in out.columns]
    out["demo_market_activity_available"] = out[market_cols].notna().any(axis=1).astype(int) if market_cols else 0
    out["demo_infrastructure_available"] = out[["demo_atm_count", "demo_pharmacy_count", "demo_bank_count"]].notna().any(axis=1).astype(int)
    # District-level coverage. County-level coverage is computed below after
    # county aggregate columns are created. Do not include
    # demo_has_county_demographics here yet; otherwise the function fails before
    # county features exist.
    base_flag_cols = [
        "demo_has_demographics",
        "demo_age_coverage",
        "demo_education_coverage",
        "demo_income_available",
        "demo_ses_available",
        "demo_market_activity_available",
        "demo_infrastructure_available",
    ]
    out["demo_has_county_demographics"] = 0
    out["demo_coverage_score"] = out[base_flag_cols].mean(axis=1)

    # County aggregate features from the same demographics table.
    agg_map = {
        "demo_population_total": ["sum"],
        "demo_population_density": ["median", "mean"],
        "demo_per_capita_income_try": ["median"],
        "demo_household_income_try": ["median"],
        "demo_household_size": ["mean"],
        "demo_education_university_ratio": ["mean"],
        "demo_education_high_school_ratio": ["mean"],
        "demo_ses_ab_ratio": ["mean"],
        "demo_ses_cd_ratio": ["mean"],
        "demo_old_ratio": ["mean"],
        "demo_young_ratio": ["mean"],
        "demo_married_ratio": ["mean"],
        "demo_single_ratio": ["mean"],
        "demo_residential_count": ["sum"],
        "demo_workplace_count": ["sum"],
        "demo_vehicle_count": ["sum"],
        "demo_car_count": ["sum"],
        "demo_atm_count": ["sum"],
        "demo_pharmacy_count": ["sum"],
        "demo_bank_count": ["sum"],
    }
    if mode == "full":
        agg_map.update({"demo_sale_count": ["sum"], "demo_turnover_ratio": ["median"]})
    present_agg = {k: v for k, v in agg_map.items() if k in out.columns}
    if present_agg:
        county = out.groupby("county_id", dropna=False).agg(present_agg)
        county.columns = [f"county_{col}_{stat}" for col, stat in county.columns]
        county = county.reset_index()
        out = out.merge(county, on="county_id", how="left")

    # Relative-to-county features.
    def div(a: str, b: str, out_col: str):
        if a in out.columns and b in out.columns:
            out[out_col] = safe_divide_series(out[a], out[b])
        else:
            out[out_col] = np.nan
    def diff(a: str, b: str, out_col: str):
        if a in out.columns and b in out.columns:
            out[out_col] = pd.to_numeric(out[a], errors="coerce") - pd.to_numeric(out[b], errors="coerce")
        else:
            out[out_col] = np.nan
    div("demo_per_capita_income_try", "county_demo_per_capita_income_try_median", "demo_income_vs_county")
    div("demo_household_income_try", "county_demo_household_income_try_median", "demo_household_income_vs_county")
    div("demo_population_density", "county_demo_population_density_median", "demo_density_vs_county")
    diff("demo_education_university_ratio", "county_demo_education_university_ratio_mean", "demo_university_diff_county")
    diff("demo_education_high_school_ratio", "county_demo_education_high_school_ratio_mean", "demo_high_school_diff_county")
    diff("demo_ses_ab_ratio", "county_demo_ses_ab_ratio_mean", "demo_ses_ab_diff_county")
    diff("demo_ses_cd_ratio", "county_demo_ses_cd_ratio_mean", "demo_ses_cd_diff_county")
    diff("demo_old_ratio", "county_demo_old_ratio_mean", "demo_old_ratio_diff_county")
    diff("demo_young_ratio", "county_demo_young_ratio_mean", "demo_young_ratio_diff_county")
    if mode == "full":
        div("demo_turnover_ratio", "county_demo_turnover_ratio_median", "demo_turnover_vs_county")
    else:
        out["demo_turnover_vs_county"] = np.nan
    div("demo_residential_count", "demo_population_total", "demo_residential_density")
    div("demo_workplace_count", "demo_population_total", "demo_workplace_density")
    div("demo_vehicle_count", "demo_population_total", "demo_vehicle_per_capita")
    div("demo_car_count", "demo_population_total", "demo_car_per_capita")
    out["demo_pharmacy_per_10k"] = safe_divide_series(out.get("demo_pharmacy_count", pd.Series(np.nan, index=out.index)) * 10000, out.get("demo_population_total", pd.Series(np.nan, index=out.index)))
    out["demo_bank_per_10k"] = safe_divide_series(out.get("demo_bank_count", pd.Series(np.nan, index=out.index)) * 10000, out.get("demo_population_total", pd.Series(np.nan, index=out.index)))
    out["demo_atm_per_10k"] = safe_divide_series(out.get("demo_atm_count", pd.Series(np.nan, index=out.index)) * 10000, out.get("demo_population_total", pd.Series(np.nan, index=out.index)))

    # County-level demographic availability is based on county_demo_* columns.
    # These are computed by aggregating neighborhood rows from district_demographics,
    # so they can exist even when an exact district-level join later fails.
    county_indicator_cols = [c for c in out.columns if c.startswith("county_demo_")]
    if county_indicator_cols:
        out["demo_has_county_demographics"] = out[county_indicator_cols].notna().any(axis=1).astype(int)
    else:
        out["demo_has_county_demographics"] = 0

    final_flag_cols = [
        "demo_has_demographics",
        "demo_has_county_demographics",
        "demo_age_coverage",
        "demo_education_coverage",
        "demo_income_available",
        "demo_ses_available",
        "demo_market_activity_available",
        "demo_infrastructure_available",
    ]
    final_flag_cols = [c for c in final_flag_cols if c in out.columns]
    out["demo_coverage_score"] = out[final_flag_cols].mean(axis=1)

    return out.drop_duplicates(["city_id", "county_id", "district_id"]).reset_index(drop=True)



def build_county_demo_fallback_features(demo_features: pd.DataFrame) -> pd.DataFrame:
    """Return one row per county with county-level demographic aggregate features.

    Important: district_demographics is a neighborhood-level table. County-level
    demographics are intentionally computed from that table by aggregating all
    available neighborhood rows. These features should be attached by county even
    when a listing cannot be matched to an exact neighborhood demographic row.
    """
    if demo_features is None or demo_features.empty:
        return pd.DataFrame()
    demo = demo_features.copy()
    if "county_id" not in demo.columns:
        return pd.DataFrame()
    for key in ["city_id", "county_id"]:
        if key in demo.columns:
            demo[key] = pd.to_numeric(demo[key], errors="coerce").astype("Int64")

    county_cols = [c for c in demo.columns if c.startswith("county_demo_")]
    keep_cols = [c for c in ["city_id", "county_id", "demo_city_name", "demo_county_name", "demo_city_norm", "demo_county_norm"] if c in demo.columns]
    if not county_cols:
        return pd.DataFrame()
    county = demo[keep_cols + county_cols].copy()

    # For each county, aggregate columns are repeated on every district row; keep
    # the first non-null value per column. Name helper columns are only used for
    # fallback joins and are dropped before model training.
    def first_non_null(series: pd.Series):
        valid = series.dropna()
        if valid.empty:
            return np.nan
        return valid.iloc[0]

    group_cols = ["county_id"]
    if "city_id" in county.columns:
        group_cols = ["city_id", "county_id"]
    agg_spec = {c: first_non_null for c in county.columns if c not in group_cols}
    county = county.groupby(group_cols, dropna=False).agg(agg_spec).reset_index()
    county["demo_has_county_demographics"] = county[county_cols].notna().any(axis=1).astype(int)
    return county


def merge_county_demographic_fallback(out: pd.DataFrame, demo_features: pd.DataFrame) -> tuple[pd.DataFrame, int, str]:
    """Attach county aggregate demographics independently from district match.

    District-level demographic features require exact district matching. County
    aggregate features only require county matching, because they are computed by
    aggregating all neighborhood demographic rows in the county. This prevents all
    county_demo_* fields from becoming NaN when listing rows miss district IDs or
    district names differ slightly.
    """
    county = build_county_demo_fallback_features(demo_features)
    if county.empty:
        out["demo_has_county_demographics"] = 0
        return out, 0, "county_disabled"

    result = out.copy()
    county_cols = [c for c in county.columns if c.startswith("county_demo_")]
    helper_cols = [c for c in ["demo_city_name", "demo_county_name", "demo_city_norm", "demo_county_norm"] if c in county.columns]
    county_feature_cols = county_cols + ["demo_has_county_demographics"]

    # Remove county aggregate columns created by a failed/partial district merge;
    # re-attach them from a clean one-row-per-county table. This is deliberate:
    # county aggregates are county-level signals and should not depend on exact
    # neighborhood matching.
    drop_existing = [c for c in county_feature_cols if c in result.columns]
    if drop_existing:
        result = result.drop(columns=drop_existing)

    join_method = "county_id"
    id_join_usable = all(k in result.columns for k in ["city_id", "county_id"]) and all(k in county.columns for k in ["city_id", "county_id"])
    if id_join_usable:
        for k in ["city_id", "county_id"]:
            result[k] = pd.to_numeric(result[k], errors="coerce").astype("Int64")
            county[k] = pd.to_numeric(county[k], errors="coerce").astype("Int64")
        if result["county_id"].notna().sum() == 0:
            id_join_usable = False

    if id_join_usable:
        merge_cols = ["city_id", "county_id"] if "city_id" in result.columns and "city_id" in county.columns else ["county_id"]
        result = result.merge(county[merge_cols + county_feature_cols], on=merge_cols, how="left", validate="m:1")
        matched = int(pd.to_numeric(result.get("demo_has_county_demographics", 0), errors="coerce").fillna(0).sum())
        if len(result) > 0 and matched == 0:
            # Fall back to name-based county join if ID columns exist but are not aligned.
            result = result.drop(columns=[c for c in county_feature_cols if c in result.columns], errors="ignore")
            id_join_usable = False
            join_method = "county_name_fallback_after_zero_id_match"
    else:
        join_method = "county_name_fallback"

    if not id_join_usable:
        result = add_location_norm_keys(result, prefix="")
        county_name = county.copy()
        if not all(c in county_name.columns for c in ["demo_city_norm", "demo_county_norm"]):
            county_name = add_location_norm_keys(county_name.rename(columns={
                "demo_city_name": "city_name",
                "demo_county_name": "county_name",
            }), prefix="demo_").rename(columns={
                "city_name": "demo_city_name",
                "county_name": "demo_county_name",
            })
        use_cols = [c for c in ["demo_city_norm", "demo_county_norm"] + county_feature_cols if c in county_name.columns]
        county_name = county_name[use_cols].drop_duplicates(["demo_city_norm", "demo_county_norm"])
        result = result.merge(
            county_name,
            left_on=["city_norm", "county_norm"],
            right_on=["demo_city_norm", "demo_county_norm"],
            how="left",
            validate="m:1",
            suffixes=("", "_countyfill"),
        )
        result = result.drop(columns=[c for c in ["city_norm", "county_norm", "district_norm", "demo_city_norm", "demo_county_norm"] if c in result.columns], errors="ignore")

    if "demo_has_county_demographics" not in result.columns:
        result["demo_has_county_demographics"] = 0
    result["demo_has_county_demographics"] = pd.to_numeric(result["demo_has_county_demographics"], errors="coerce").fillna(0).astype(int)
    matched = int(result["demo_has_county_demographics"].sum())
    return result, matched, join_method

def attach_demographic_features(listings: pd.DataFrame, demo_features: pd.DataFrame, mode: str, out_dirs: dict[str, Path] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach demographic features.

    Primary join is city_id/county_id/district_id. If the listing table does not
    carry those IDs, fall back to normalized city/county/district names that are
    kept in the demographic feature table. This prevents the model from training
    with all demographic columns missing.
    """
    if mode == "none" or demo_features is None or demo_features.empty:
        out = listings.copy()
        out["demo_has_demographics"] = 0
        out["demo_has_county_demographics"] = 0
        out["demo_coverage_score"] = 0.0
        return out, {"mode": mode, "demo_rows": 0, "matched_listing_rows": 0, "match_rate": 0.0, "join_method": "disabled"}

    out = listings.copy()
    demo = demo_features.copy()
    before = len(out)
    join_method = "id"

    # Decide whether ID join is usable. Some listing tables only have city/county/district names.
    id_join_usable = True
    for key in ["city_id", "county_id", "district_id"]:
        if key not in out.columns:
            id_join_usable = False
            break
        out[key] = pd.to_numeric(out[key], errors="coerce").astype("Int64")
        if out[key].notna().sum() == 0:
            id_join_usable = False
    for key in ["city_id", "county_id", "district_id"]:
        if key not in demo.columns:
            id_join_usable = False
            break
        demo[key] = pd.to_numeric(demo[key], errors="coerce").astype("Int64")

    if id_join_usable:
        out = out.merge(demo, on=["city_id", "county_id", "district_id"], how="left", validate="m:1")
        matched = int(pd.to_numeric(out.get("demo_has_demographics", 0), errors="coerce").fillna(0).sum())
        # If ID join matched nothing, fall back to names rather than silently keeping all NaNs.
        if before > 0 and matched == 0:
            base_cols = [c for c in out.columns if not c.startswith("demo_") and not c.startswith("county_demo_")]
            out = out[base_cols].copy()
            join_method = "name_fallback_after_zero_id_match"
            id_join_usable = False
    else:
        join_method = "name_fallback"

    if not id_join_usable:
        out = add_location_norm_keys(out, prefix="")
        # demo was already augmented with demo_*_norm keys in build_demographic_features.
        needed = ["demo_city_norm", "demo_county_norm", "demo_district_norm"]
        if not all(c in demo.columns for c in needed):
            demo = add_location_norm_keys(demo.rename(columns={
                "demo_city_name": "city_name",
                "demo_county_name": "county_name",
                "demo_district_name": "district_name",
            }), prefix="demo_").rename(columns={
                "city_name": "demo_city_name",
                "county_name": "demo_county_name",
                "district_name": "demo_district_name",
            })
        demo_name = demo.drop_duplicates(["demo_city_norm", "demo_county_norm", "demo_district_norm"]).copy()
        out = out.merge(
            demo_name,
            left_on=["city_norm", "county_norm", "district_norm"],
            right_on=["demo_city_norm", "demo_county_norm", "demo_district_norm"],
            how="left",
            validate="m:1",
            suffixes=("", "_demo"),
        )
        # If source listing lacked IDs, fill them from demographics after successful name join.
        for key in ["city_id", "county_id", "district_id"]:
            demo_key = f"{key}_demo"
            if key not in out.columns and demo_key in out.columns:
                out[key] = out[demo_key]
            elif demo_key in out.columns:
                out[key] = out[key].where(out[key].notna(), out[demo_key])
        out = out.drop(columns=[c for c in ["city_norm", "county_norm", "district_norm", "city_id_demo", "county_id_demo", "district_id_demo"] if c in out.columns], errors="ignore")

    # Attach county-level demographic aggregates independently. These aggregates
    # are computed from all neighborhood demographic rows by county, so they only
    # require a county match and should be present even if the exact neighborhood
    # demographic row cannot be matched.
    out, county_matched, county_join_method = merge_county_demographic_fallback(out, demo_features)

    # Drop fallback helper/name columns from model input; they are not features.
    out = out.drop(columns=[c for c in out.columns if c.startswith("demo_") and c.endswith("_norm")], errors="ignore")
    out = out.drop(columns=[c for c in ["demo_city_name", "demo_county_name", "demo_district_name"] if c in out.columns], errors="ignore")

    if "demo_has_demographics" not in out.columns:
        out["demo_has_demographics"] = 0
    out["demo_has_demographics"] = pd.to_numeric(out["demo_has_demographics"], errors="coerce").fillna(0).astype(int)
    if "demo_has_county_demographics" not in out.columns:
        out["demo_has_county_demographics"] = 0
    out["demo_has_county_demographics"] = pd.to_numeric(out["demo_has_county_demographics"], errors="coerce").fillna(0).astype(int)
    if "demo_coverage_score" not in out.columns:
        out["demo_coverage_score"] = 0.0
    out["demo_coverage_score"] = pd.to_numeric(out["demo_coverage_score"], errors="coerce").fillna(0)

    matched = int(out["demo_has_demographics"].sum())
    report = {
        "mode": mode,
        "demo_rows": int(len(demo_features)),
        "listing_rows": int(before),
        "matched_listing_rows": matched,
        "match_rate": float(matched / before) if before else 0.0,
        "county_matched_listing_rows": int(county_matched) if 'county_matched' in locals() else 0,
        "county_match_rate": float(county_matched / before) if before and 'county_matched' in locals() else 0.0,
        "join_method": join_method,
        "county_join_method": county_join_method if 'county_join_method' in locals() else "disabled",
    }
    if out_dirs is not None:
        cov_cols = ["county", "district", "demo_has_demographics", "demo_coverage_score", "demo_income_available", "demo_age_coverage", "demo_education_coverage"]
        present = [c for c in cov_cols if c in out.columns]
        if present and "county" in out.columns and "district" in out.columns:
            cov = out.groupby(["county", "district"], dropna=False).agg(
                rows=("demo_has_demographics", "size"),
                matched_demo_rows=("demo_has_demographics", "sum"),
                matched_county_demo_rows=("demo_has_county_demographics", "sum") if "demo_has_county_demographics" in out.columns else ("demo_has_demographics", "sum"),
                avg_demo_coverage_score=("demo_coverage_score", "mean"),
            ).reset_index()
            cov["match_rate"] = cov["matched_demo_rows"] / cov["rows"].replace({0: np.nan})
            cov["county_match_rate"] = cov["matched_county_demo_rows"] / cov["rows"].replace({0: np.nan})
            cov.to_csv(out_dirs["reports"] / f"demographic_feature_coverage_{mode}_v11.csv", index=False, encoding="utf-8-sig")
    return out, report


def get_demographic_mode_feature_names(mode: str) -> tuple[list[str], list[str]]:
    mode = (mode or "none").lower().strip()
    if mode == "none":
        return ["demo_has_demographics", "demo_coverage_score"], []
    if mode == "safe":
        return DEMOGRAPHIC_SAFE_NUMERIC_FEATURES, DEMOGRAPHIC_CATEGORICAL_FEATURES
    if mode == "full":
        return DEMOGRAPHIC_SAFE_NUMERIC_FEATURES + DEMOGRAPHIC_FULL_EXTRA_NUMERIC_FEATURES, DEMOGRAPHIC_CATEGORICAL_FEATURES
    raise ValueError(f"Unsupported demographics mode: {mode}")

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

    sales_clean.to_csv(out_dirs["input"] / "sales_cleaned_v11.csv", index=False, encoding="utf-8-sig")
    rentals_clean.to_csv(out_dirs["input"] / "rentals_cleaned_v11.csv", index=False, encoding="utf-8-sig")
    pd.concat([sales_removed_basic, sales_removed_iqr], ignore_index=True).to_csv(
        out_dirs["input"] / "sales_removed_v11.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat([rentals_removed_basic, rentals_removed_iqr], ignore_index=True).to_csv(
        out_dirs["input"] / "rentals_removed_v11.csv", index=False, encoding="utf-8-sig"
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
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

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
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
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

    removed.to_csv(out_dirs["input"] / "sales_removed_location_outliers_v11.csv", index=False, encoding="utf-8-sig")
    clean.to_csv(out_dirs["input"] / "sales_after_location_outlier_filter_v11.csv", index=False, encoding="utf-8-sig")

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
    # keep_empty_features prevents fold/segment-specific all-missing columns from being dropped.
    numeric_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
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
            # RidgeCV may use an SVD path that can fail on wide one-hot matrices in small county folds.
            # Fixed-alpha Ridge with LSQR is much more stable for segment/county expert fits.
            Ridge(alpha=10.0, solver="lsqr"),
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


def segment_mask(X: pd.DataFrame, segment_name: str) -> pd.Series:
    """App-safe segment rules. They only use fields the application already asks from the user."""
    idx = X.index

    gross_m2 = pd.to_numeric(X.get("gross_m2", pd.Series(np.nan, index=idx)), errors="coerce")
    building_age = pd.to_numeric(X.get("building_age", pd.Series(np.nan, index=idx)), errors="coerce")

    if "rooms" in X.columns:
        rooms = pd.to_numeric(X["rooms"], errors="coerce")
    elif "room_count" in X.columns:
        rooms = X["room_count"].map(lambda v: parse_room(v)[0])
    else:
        rooms = pd.Series(np.nan, index=idx)

    if segment_name == "large_home":
        return ((gross_m2 >= 151) | (rooms >= 4)).fillna(False)
    if segment_name == "compact_home":
        return ((gross_m2 <= 85) | (rooms <= 1)).fillna(False)
    if segment_name == "old_building":
        return (building_age >= 26).fillna(False)
    if segment_name == "mainstream_home":
        return ((gross_m2 > 85) & (gross_m2 < 151) & (rooms >= 2) & (rooms <= 3) & (building_age < 26)).fillna(False)

    return pd.Series(False, index=idx)


def county_metric_report(oof: pd.DataFrame, pred_col: str = "pred_ensemble") -> pd.DataFrame:
    if "county" not in oof.columns:
        return pd.DataFrame()
    rows = []
    for county, g in oof.groupby("county", dropna=False):
        m = metric_dict(g["actual_unit_price_gross"], g[pred_col])
        m["county"] = county
        rows.append(m)
    if not rows:
        return pd.DataFrame()
    cols = ["county", "rows", "r2", "log_r2", "mape", "median_ape", "mae_tl_per_m2", "median_ae_tl_per_m2"]
    return pd.DataFrame(rows)[cols].sort_values("mape", ascending=True)


def train_segment_layer(
    X: pd.DataFrame,
    y: pd.Series,
    base_oof: pd.DataFrame,
    cfg: RunConfig,
    out_dirs: dict[str, Path],
    min_rows: int = 180,
    blend_weight: float = 0.35,
) -> tuple[pd.Series, dict[str, dict[str, Any]], dict[str, dict[str, float]], dict[str, float], pd.DataFrame]:
    """Train optional specialist models for weak/structurally different segments.

    V11 note: the previous V9 only used the segment layer when the specialist segment
    ensemble alone beat the base ensemble. In practice, the blend often improved MAPE/R2
    even when the specialist alone was weaker. This function now tests multiple blend
    weights and applies the best blend when it improves the segment OOF MAPE.
    """
    segment_names = ["large_home", "compact_home", "old_building", "mainstream_home"]
    blend_candidates = [0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.00]
    final_pred = base_oof["pred_ensemble_base"].copy()
    segment_models: dict[str, dict[str, Any]] = {}
    segment_weights: dict[str, dict[str, float]] = {}
    segment_blend_weights: dict[str, float] = {}
    report_rows: list[dict[str, Any]] = []

    specs = model_specs(cfg.target_mode, selected_models=cfg.selected_models, fast_mode=cfg.fast_mode)

    for segment_name in segment_names:
        mask = segment_mask(X, segment_name)
        n = int(mask.sum())
        if n < min_rows:
            report_rows.append({"segment": segment_name, "rows": n, "status": "skipped_too_few_rows"})
            continue

        Xs = X.loc[mask].reset_index(drop=True)
        ys = y.loc[mask].reset_index(drop=True)
        split_count = min(cfg.n_splits, max(2, n // 35))
        cv = KFold(n_splits=split_count, shuffle=True, random_state=cfg.random_state)

        seg_oof = pd.DataFrame({"actual_unit_price_gross": ys})
        seg_rows = []
        fitted = {}

        for model_name, estimator in specs.items():
            print(f"Training segment OOF model: {segment_name}/{model_name}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = cross_val_predict(clone(estimator), Xs, ys, cv=cv, n_jobs=None, method="predict")
            pred = np.maximum(np.asarray(pred, dtype=float), 0)
            seg_oof[f"pred_{model_name}"] = pred
            m = metric_dict(ys, pred)
            m["model"] = model_name
            seg_rows.append(m)

            final_estimator = clone(estimator)
            final_estimator.fit(Xs, ys)
            fitted[model_name] = final_estimator

        seg_comparison = pd.DataFrame(seg_rows).sort_values(["mape", "mae_tl_per_m2"])
        weights = choose_ensemble_weights(seg_comparison, max_models=2)
        seg_pred = np.zeros(n, dtype=float)
        total_w = sum(weights.values())
        for model_name, w in weights.items():
            seg_pred += seg_oof[f"pred_{model_name}"].to_numpy(dtype=float) * (w / total_w)

        base_values = base_oof.loc[mask, "pred_ensemble_base"].to_numpy(dtype=float)
        actual_values = y.loc[mask]
        base_metrics = metric_dict(actual_values, base_values)
        segment_metrics = metric_dict(ys, seg_pred)

        candidate_rows = []
        for bw in blend_candidates:
            pred_candidate = (1.0 - bw) * base_values + bw * seg_pred
            metrics_candidate = metric_dict(actual_values, pred_candidate)
            candidate_rows.append({
                "blend_weight": bw,
                "r2": metrics_candidate["r2"],
                "log_r2": metrics_candidate["log_r2"],
                "mape": metrics_candidate["mape"],
                "mae_tl_per_m2": metrics_candidate["mae_tl_per_m2"],
                "pred": pred_candidate,
            })

        best_candidate = min(candidate_rows, key=lambda row: (row["mape"], row["mae_tl_per_m2"], -row["r2"]))
        blended_metrics = {
            "r2": best_candidate["r2"],
            "log_r2": best_candidate["log_r2"],
            "mape": best_candidate["mape"],
            "mae_tl_per_m2": best_candidate["mae_tl_per_m2"],
        }
        best_blend_weight = float(best_candidate["blend_weight"])

        # Use the blend only if it improves segment MAPE. A tiny epsilon prevents false positives
        # from floating-point noise while still allowing small real improvements.
        use_segment = bool(best_candidate["mape"] < base_metrics["mape"] - 1e-6)

        if use_segment:
            final_pred.loc[mask] = best_candidate["pred"]
            selected = {name: fitted[name] for name in weights}
            segment_models[segment_name] = selected
            segment_weights[segment_name] = weights
            segment_blend_weights[segment_name] = best_blend_weight
            for model_name, model_obj in selected.items():
                joblib.dump(model_obj, out_dirs["artifacts"] / f"model_segment_{segment_name}_{model_name}_v11.joblib")

        row = {
            "segment": segment_name,
            "rows": n,
            "status": "used_blend" if use_segment else "kept_base",
            "blend_weight": best_blend_weight if use_segment else 0.0,
            "base_r2": base_metrics["r2"],
            "base_log_r2": base_metrics["log_r2"],
            "base_mape": base_metrics["mape"],
            "base_mae_tl_per_m2": base_metrics["mae_tl_per_m2"],
            "segment_r2": segment_metrics["r2"],
            "segment_log_r2": segment_metrics["log_r2"],
            "segment_mape": segment_metrics["mape"],
            "segment_mae_tl_per_m2": segment_metrics["mae_tl_per_m2"],
            "best_blended_r2": blended_metrics["r2"],
            "best_blended_log_r2": blended_metrics["log_r2"],
            "best_blended_mape": blended_metrics["mape"],
            "best_blended_mae_tl_per_m2": blended_metrics["mae_tl_per_m2"],
            "blend_candidates": json.dumps([{k: v for k, v in c.items() if k != "pred"} for c in candidate_rows], ensure_ascii=False),
            "weights": json.dumps(weights, ensure_ascii=False),
        }
        report_rows.append(row)

        seg_comparison.insert(0, "segment", segment_name)
        seg_comparison.to_csv(out_dirs["reports"] / f"model_comparison_segment_{segment_name}_v11.csv", index=False, encoding="utf-8-sig")

    report = pd.DataFrame(report_rows)
    report.to_csv(out_dirs["reports"] / "segment_layer_report_v11.csv", index=False, encoding="utf-8-sig")
    return final_pred, segment_models, segment_weights, segment_blend_weights, report



# =========================
# V11 anomaly + county expert helpers
# =========================

def parse_rooms_number_for_anomaly(v: Any) -> float:
    if pd.isna(v):
        return np.nan
    s = str(v).replace(" ", "").lower()
    m = re.search(r"(\d+)\+(\d+)", s)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else np.nan


def safe_ratio(a: Any, b: Any) -> pd.Series:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce").replace(0, np.nan)
    out = aa / bb
    return out.replace([np.inf, -np.inf], np.nan)


def compute_anomaly_scores(X: pd.DataFrame, y: pd.Series, cfg: RunConfig) -> pd.DataFrame:
    """Create a diagnostic listing-anomaly score.

    This does not add app-unavailable model features. It uses only listing fields, rental/trend
    market aggregates, and leak-safe local baseline diagnostics. The goal is to identify rows that
    may be wrong, suspicious, or hard for the app-safe feature set to explain.
    """
    df = X.copy().reset_index(drop=True)
    y = pd.to_numeric(y, errors="coerce").reset_index(drop=True)
    df["actual_unit_price_gross"] = y

    try:
        enc = LocationBaselineEncoder(
            min_specific_count=max(5, min(cfg.location_min_group_size, 12)),
            min_medium_count=max(8, cfg.location_min_group_size),
            min_broad_count=max(15, cfg.location_min_group_size + 5),
        ).fit(X, y)
        enriched = enc.transform(X)
        baseline = pd.to_numeric(enriched.get("location_baseline_m2"), errors="coerce").reset_index(drop=True)
        baseline_count = pd.to_numeric(enriched.get("location_baseline_count"), errors="coerce").reset_index(drop=True)
        baseline_level = pd.to_numeric(enriched.get("location_baseline_level_code"), errors="coerce").reset_index(drop=True)
    except Exception:
        baseline = pd.Series(np.nan, index=df.index)
        baseline_count = pd.Series(np.nan, index=df.index)
        baseline_level = pd.Series(np.nan, index=df.index)

    ratio = y / baseline.replace(0, np.nan)
    log_ratio = np.log(ratio.replace(0, np.nan))
    df["anomaly_location_baseline_m2"] = baseline
    df["anomaly_location_baseline_count"] = baseline_count
    df["anomaly_location_baseline_level_code"] = baseline_level
    df["anomaly_price_to_location_baseline"] = ratio
    df["anomaly_log_price_to_location_baseline"] = log_ratio

    flags: dict[str, pd.Series] = {}
    weights: dict[str, float] = {}

    def add_flag(name: str, cond: Any, weight: float) -> None:
        c = pd.Series(cond, index=df.index).fillna(False).astype(bool)
        flags[name] = c
        weights[name] = float(weight)
        df[name] = c.astype(int)

    add_flag("anom_baseline_ratio_very_low", ratio < 0.60, 18)
    add_flag("anom_baseline_ratio_low", ratio.between(0.60, 0.72, inclusive="left"), 8)
    add_flag("anom_baseline_ratio_high", ratio.between(1.55, 1.90, inclusive="right"), 8)
    add_flag("anom_baseline_ratio_very_high", ratio > 1.90, 18)
    add_flag("anom_low_baseline_support", baseline_count.notna() & (baseline_count < 5), 8)

    if "net_m2" in df.columns and "gross_m2" in df.columns:
        ng = safe_ratio(df["net_m2"], df["gross_m2"])
        df["anomaly_net_gross_ratio"] = ng
        add_flag("anom_net_gross_too_low", ng < 0.50, 12)
        add_flag("anom_net_gross_too_high", ng > 1.03, 18)
    else:
        df["anomaly_net_gross_ratio"] = np.nan

    if "price" in df.columns and "gross_m2" in df.columns:
        calc_unit = safe_ratio(df["price"], df["gross_m2"])
        unit_diff = (calc_unit - y).abs() / y.replace(0, np.nan)
        df["anomaly_unit_price_recalc_diff_ratio"] = unit_diff
        add_flag("anom_price_unit_mismatch", unit_diff > 0.03, 25)
    else:
        df["anomaly_unit_price_recalc_diff_ratio"] = np.nan

    gross = pd.to_numeric(df.get("gross_m2", np.nan), errors="coerce")
    rooms = df.get("room_count", pd.Series(np.nan, index=df.index)).map(parse_rooms_number_for_anomaly)
    df["anomaly_rooms_numeric"] = rooms
    add_flag("anom_large_room_tiny_m2", (rooms >= 4) & (gross < 95), 12)
    add_flag("anom_small_room_huge_m2", (rooms <= 1) & (gross > 135), 10)
    add_flag("anom_very_large_property", gross > 230, 7)
    add_flag("anom_very_small_property", gross < 45, 7)

    floor_num = pd.to_numeric(df.get("floor_num", np.nan), errors="coerce")
    total_floors = pd.to_numeric(df.get("total_floors", np.nan), errors="coerce")
    add_flag("anom_floor_above_total", floor_num.notna() & total_floors.notna() & (total_floors > 0) & (floor_num > total_floors), 20)
    add_flag("anom_total_floors_extreme", total_floors > 45, 8)

    building_age = pd.to_numeric(df.get("building_age", np.nan), errors="coerce")
    add_flag("anom_building_age_extreme", building_age > 80, 8)

    rent_m2 = pd.to_numeric(df.get("estimated_rent_m2_gross", np.nan), errors="coerce")
    amort_years = y / rent_m2.replace(0, np.nan) / 12.0
    df["anomaly_estimated_amortization_years"] = amort_years
    add_flag("anom_amortization_too_low", amort_years.notna() & (amort_years < 8), 10)
    add_flag("anom_amortization_too_high", amort_years.notna() & (amort_years > 45), 10)

    detail_count = pd.to_numeric(df.get("detail_selected_count", np.nan), errors="coerce")
    add_flag("anom_low_detail", detail_count.notna() & (detail_count <= 3), 5)

    # Group robust z-score over log(price/location_baseline). This catches rows that are odd
    # relative to their county + room + m2 group without deleting them blindly.
    df["anomaly_group_mad_z"] = np.nan
    group_cols = [c for c in ["county", "room_count", "m2_group"] if c in df.columns]
    if group_cols:
        work = df[group_cols].copy()
        work["__log_ratio__"] = log_ratio
        for _, idx in work.groupby(group_cols, dropna=False).groups.items():
            idx = list(idx)
            if len(idx) < max(10, cfg.location_min_group_size):
                continue
            vals = pd.to_numeric(work.loc[idx, "__log_ratio__"], errors="coerce")
            med = vals.median()
            mad = np.median(np.abs(vals.dropna() - med)) if vals.notna().any() else np.nan
            if not np.isfinite(mad) or mad <= 1e-9:
                continue
            z = 0.6745 * (vals - med) / mad
            df.loc[idx, "anomaly_group_mad_z"] = z
    add_flag("anom_group_mad_extreme", pd.to_numeric(df["anomaly_group_mad_z"], errors="coerce").abs() > 3.5, 15)

    score = pd.Series(0.0, index=df.index)
    reasons = []
    for i in df.index:
        active = []
        total = 0.0
        for name, cond in flags.items():
            if bool(cond.loc[i]):
                active.append(name)
                total += weights[name]
        reasons.append("|".join(active))
        score.loc[i] = min(100.0, total)
    df["anomaly_score"] = score
    df["anomaly_reasons"] = reasons
    df["anomaly_severity"] = pd.cut(
        df["anomaly_score"],
        bins=[-0.1, 9.9, 24.9, 44.9, 100.0],
        labels=["normal", "review", "high_risk", "exclude_candidate"],
    ).astype(str)
    return df


def save_anomaly_reports(anom: pd.DataFrame, reports_dir: Path, output_dir: Path) -> None:
    if anom.empty:
        return
    anom.sort_values("anomaly_score", ascending=False).to_csv(output_dir / "listing_anomaly_scores_v11.csv", index=False, encoding="utf-8-sig")
    anom.sort_values("anomaly_score", ascending=False).head(250).to_csv(reports_dir / "top_listing_anomalies_v11.csv", index=False, encoding="utf-8-sig")

    def group_anomaly(col: str) -> None:
        if col not in anom.columns:
            return
        rep = (
            anom.groupby(col, dropna=False)
            .agg(
                n=("anomaly_score", "size"),
                mean_anomaly_score=("anomaly_score", "mean"),
                median_anomaly_score=("anomaly_score", "median"),
                high_risk_or_worse=("anomaly_severity", lambda s: int(s.isin(["high_risk", "exclude_candidate"]).sum())),
                exclude_candidate=("anomaly_severity", lambda s: int(s.eq("exclude_candidate").sum())),
                median_price_to_baseline=("anomaly_price_to_location_baseline", "median"),
                mean_unit_price=("actual_unit_price_gross", "mean"),
            )
            .reset_index()
            .sort_values(["mean_anomaly_score", "n"], ascending=[False, False])
        )
        rep.to_csv(reports_dir / f"anomaly_by_{col}_v11.csv", index=False, encoding="utf-8-sig")

    for col in ["county", "district", "room_count", "m2_group", "building_age_group"]:
        group_anomaly(col)


def anomaly_metric_diagnostics(oof: pd.DataFrame) -> pd.DataFrame:
    if "anomaly_score" not in oof.columns:
        return pd.DataFrame()
    rows = []
    for label, mask in {
        "all_rows": pd.Series(True, index=oof.index),
        "score_lt_25": oof["anomaly_score"] < 25,
        "score_lt_45": oof["anomaly_score"] < 45,
        "score_gte_25": oof["anomaly_score"] >= 25,
        "score_gte_45": oof["anomaly_score"] >= 45,
    }.items():
        sub = oof.loc[mask]
        if len(sub) < 5:
            continue
        m = metric_dict(sub["actual_unit_price_gross"], sub["pred_ensemble"])
        m["slice"] = label
        m["rows"] = int(len(sub))
        rows.append(m)
    return pd.DataFrame(rows)


def train_county_expert_layer(
    X: pd.DataFrame,
    y: pd.Series,
    current_pred: pd.Series,
    cfg: RunConfig,
    out_dirs: dict[str, Path],
) -> tuple[pd.Series, dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, float]], dict[str, dict[str, float]], pd.DataFrame]:
    """Train county-specific expert models and use them only when an OOF blend improves.

    Returns final prediction, used county models, all county models for diagnostics/feature importance,
    used county weights, used county blend weights, and a report.
    """
    final_pred = pd.Series(current_pred, index=X.index, dtype=float).copy()
    used_county_models: dict[str, dict[str, Any]] = {}
    all_county_models: dict[str, dict[str, Any]] = {}
    county_weights: dict[str, dict[str, float]] = {}
    county_blend_weights: dict[str, float] = {}
    report_rows: list[dict[str, Any]] = []

    if not cfg.enable_county_experts or "county" not in X.columns:
        report = pd.DataFrame([{"status": "disabled"}])
        report.to_csv(out_dirs["reports"] / "county_expert_report_v11.csv", index=False, encoding="utf-8-sig")
        return final_pred, used_county_models, all_county_models, county_weights, county_blend_weights, report

    counties = sorted([c for c in X["county"].dropna().astype(str).unique() if c and c != "missing"])
    for county_name in counties:
        mask = X["county"].fillna("missing").astype(str).eq(county_name)
        n = int(mask.sum())
        if n < int(cfg.county_expert_min_rows):
            report_rows.append({"county": county_name, "rows": n, "status": "skipped_too_few_rows"})
            continue

        Xc = X.loc[mask].reset_index(drop=True)
        yc = y.loc[mask].reset_index(drop=True)
        base_pred_c = pd.Series(current_pred.loc[mask].to_numpy(dtype=float), index=Xc.index)
        n_splits = min(cfg.n_splits, max(2, min(5, n // 80)))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
        specs = model_specs(cfg.target_mode, selected_models=cfg.selected_models, fast_mode=cfg.fast_mode)

        oof_c = pd.DataFrame({"actual_unit_price_gross": yc})
        comparison_rows = []
        fitted: dict[str, Any] = {}
        for model_name, estimator in specs.items():
            print(f"Training county expert OOF model: {county_name}/{model_name}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = cross_val_predict(clone(estimator), Xc, yc, cv=cv, n_jobs=None, method="predict")
            pred = np.maximum(np.asarray(pred, dtype=float), 0)
            oof_c[f"pred_{model_name}"] = pred
            metrics = metric_dict(yc, pred)
            metrics["model"] = model_name
            comparison_rows.append(metrics)

            final_estimator = clone(estimator)
            final_estimator.fit(Xc, yc)
            fitted[model_name] = final_estimator

        comp = pd.DataFrame(comparison_rows).sort_values(["mape", "mae_tl_per_m2"])
        comp.insert(0, "county", county_name)
        safe_county = re.sub(r"[^0-9A-Za-z_]+", "_", county_name)
        comp.to_csv(out_dirs["reports"] / f"model_comparison_county_{safe_county}_v11.csv", index=False, encoding="utf-8-sig")

        weights = choose_ensemble_weights(comp, max_models=2)
        expert_pred = np.zeros(n, dtype=float)
        total_w = sum(weights.values())
        for model_name, w in weights.items():
            expert_pred += oof_c[f"pred_{model_name}"].to_numpy(dtype=float) * (w / total_w)

        base_metrics = metric_dict(yc, base_pred_c)
        expert_metrics = metric_dict(yc, expert_pred)
        candidates = []
        for blend in [0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00]:
            pred_candidate = (1.0 - blend) * base_pred_c.to_numpy(dtype=float) + blend * expert_pred
            m = metric_dict(yc, pred_candidate)
            candidates.append({
                "blend_weight": float(blend),
                "r2": m["r2"],
                "log_r2": m["log_r2"],
                "mape": m["mape"],
                "mae_tl_per_m2": m["mae_tl_per_m2"],
                "pred": pred_candidate,
            })
        best = min(candidates, key=lambda row: (row["mape"], row["mae_tl_per_m2"], -row["r2"]))
        use_county = bool(best["mape"] < base_metrics["mape"] - 1e-6)

        selected = {name: fitted[name] for name in weights}
        all_county_models[county_name] = selected
        if use_county:
            final_pred.loc[mask] = best["pred"]
            used_county_models[county_name] = selected
            county_weights[county_name] = weights
            county_blend_weights[county_name] = float(best["blend_weight"])
            for model_name, model_obj in selected.items():
                joblib.dump(model_obj, out_dirs["artifacts"] / f"model_county_{safe_county}_{model_name}_v11.joblib")

        report_rows.append({
            "county": county_name,
            "rows": n,
            "status": "used_blend" if use_county else "kept_current",
            "blend_weight": float(best["blend_weight"]) if use_county else 0.0,
            "current_r2": base_metrics["r2"],
            "current_log_r2": base_metrics["log_r2"],
            "current_mape": base_metrics["mape"],
            "current_mae_tl_per_m2": base_metrics["mae_tl_per_m2"],
            "expert_r2": expert_metrics["r2"],
            "expert_log_r2": expert_metrics["log_r2"],
            "expert_mape": expert_metrics["mape"],
            "expert_mae_tl_per_m2": expert_metrics["mae_tl_per_m2"],
            "best_blended_r2": best["r2"],
            "best_blended_log_r2": best["log_r2"],
            "best_blended_mape": best["mape"],
            "best_blended_mae_tl_per_m2": best["mae_tl_per_m2"],
            "blend_candidates": json.dumps([{k: v for k, v in c.items() if k != "pred"} for c in candidates], ensure_ascii=False),
            "weights": json.dumps(weights, ensure_ascii=False),
        })

    report = pd.DataFrame(report_rows)
    report.to_csv(out_dirs["reports"] / "county_expert_report_v11.csv", index=False, encoding="utf-8-sig")
    return final_pred, used_county_models, all_county_models, county_weights, county_blend_weights, report


def save_feature_importance_by_county(county_models: dict[str, dict[str, Any]], reports_dir: Path) -> None:
    rows = []
    for county_name, models in county_models.items():
        for model_name, model in models.items():
            inner = get_inner_model(model)
            if inner is None or not hasattr(inner, "feature_importances_"):
                continue
            feat_names = get_preprocess_feature_names(model)
            if feat_names is None:
                continue
            importances = np.asarray(inner.feature_importances_, dtype=float)
            n = min(len(feat_names), len(importances))
            for feat, imp in zip(feat_names[:n], importances[:n]):
                rows.append({"county": county_name, "model": model_name, "feature": feat, "importance": float(imp)})
    if not rows:
        return
    rep = pd.DataFrame(rows).sort_values(["county", "model", "importance"], ascending=[True, True, False])
    rep.to_csv(reports_dir / "feature_importance_by_county_v11.csv", index=False, encoding="utf-8-sig")
    top = (
        rep.groupby(["county", "feature"], as_index=False)["importance"]
        .mean()
        .sort_values(["county", "importance"], ascending=[True, False])
    )
    top.groupby("county", group_keys=False).head(40).to_csv(reports_dir / "feature_importance_by_county_top40_v11.csv", index=False, encoding="utf-8-sig")


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, cfg: RunConfig, out_dirs: dict[str, Path]) -> tuple[ModelBundle, dict[str, Any]]:
    if len(X) < 10:
        raise ValueError(f"Too few sale rows to train: {len(X)}")

    anomaly_df = pd.DataFrame()
    if cfg.enable_anomaly_reports:
        print("Computing listing anomaly diagnostics...")
        anomaly_df = compute_anomaly_scores(X, y, cfg)
        save_anomaly_reports(anomaly_df, out_dirs["reports"], out_dirs["output"])

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
        joblib.dump(final_estimator, out_dirs["artifacts"] / f"model_{name}_v11.joblib")

    model_comparison = pd.DataFrame(comparison_rows).sort_values(["mape", "mae_tl_per_m2"])
    model_comparison.to_csv(out_dirs["reports"] / "model_comparison_v11.csv", index=False, encoding="utf-8-sig")

    weights = choose_ensemble_weights(model_comparison, max_models=3)
    base_pred = np.zeros(len(oof), dtype=float)
    total_w = sum(weights.values())
    for name, w in weights.items():
        base_pred += oof[f"pred_{name}"].to_numpy(dtype=float) * (w / total_w)
    oof["pred_ensemble_base"] = base_pred

    base_metrics = metric_dict(y, base_pred)
    base_metrics["model"] = "ensemble_base_top_models_v11"
    base_metrics["weights"] = weights
    base_metrics["target_mode"] = cfg.target_mode
    base_metrics["n_splits"] = n_splits

    # Segment-aware layer from V9.1 remains as the first specialist correction.
    segment_pred, segment_models, segment_weights, segment_blend_weights, segment_report = train_segment_layer(
        X=X,
        y=y,
        base_oof=oof,
        cfg=cfg,
        out_dirs=out_dirs,
        min_rows=180,
        blend_weight=0.35,
    )
    segment_pred = pd.Series(np.maximum(segment_pred.to_numpy(dtype=float), 0), index=X.index)
    oof["pred_after_segment"] = segment_pred
    segment_metrics_final = metric_dict(y, segment_pred)
    segment_metrics_final["model"] = "segment_aware_before_county_v11"

    # County-expert layer. It is applied after segment correction and only where OOF blend improves.
    county_pred, county_models, county_models_all, county_weights, county_blend_weights, county_report = train_county_expert_layer(
        X=X,
        y=y,
        current_pred=segment_pred,
        cfg=cfg,
        out_dirs=out_dirs,
    )
    oof["pred_ensemble"] = np.maximum(county_pred.to_numpy(dtype=float), 0)

    ensemble_metrics = metric_dict(y, oof["pred_ensemble"])
    ensemble_metrics["model"] = "county_expert_segment_aware_ensemble_v11"
    ensemble_metrics["base_weights"] = weights
    ensemble_metrics["segment_weights"] = segment_weights
    ensemble_metrics["segment_blend_weights"] = segment_blend_weights
    ensemble_metrics["county_weights"] = county_weights
    ensemble_metrics["county_blend_weights"] = county_blend_weights
    ensemble_metrics["target_mode"] = cfg.target_mode
    ensemble_metrics["n_splits"] = n_splits

    selected_models = {name: fitted_models[name] for name in weights}
    bundle = ModelBundle(
        selected_models,
        weights,
        list(X.columns),
        ensemble_metrics,
        segment_models=segment_models,
        segment_weights=segment_weights,
        segment_blend_weights=segment_blend_weights,
        county_models=county_models,
        county_weights=county_weights,
        county_blend_weights=county_blend_weights,
    )
    joblib.dump(bundle, out_dirs["artifacts"] / "model_bundle_v11.joblib")

    oof["error"] = oof["pred_ensemble"] - oof["actual_unit_price_gross"]
    oof["abs_error"] = oof["error"].abs()
    oof["abs_pct_error"] = oof["abs_error"] / oof["actual_unit_price_gross"]
    oof = pd.concat([oof, X.reset_index(drop=True)], axis=1)

    if not anomaly_df.empty:
        anomaly_cols = [c for c in anomaly_df.columns if c.startswith("anom_") or c.startswith("anomaly_")]
        anomaly_cols = ["anomaly_score", "anomaly_severity", "anomaly_reasons"] + [c for c in anomaly_cols if c not in {"anomaly_score", "anomaly_severity", "anomaly_reasons"}]
        anomaly_cols = [c for c in anomaly_cols if c in anomaly_df.columns]
        oof = pd.concat([oof, anomaly_df[anomaly_cols].reset_index(drop=True)], axis=1)

    oof.to_csv(out_dirs["output"] / "oof_predictions_v11.csv", index=False, encoding="utf-8-sig")

    county_metrics = county_metric_report(oof, pred_col="pred_ensemble")
    if not county_metrics.empty:
        county_metrics.to_csv(out_dirs["reports"] / "county_metrics_v11.csv", index=False, encoding="utf-8-sig")

    anomaly_metric_report = anomaly_metric_diagnostics(oof)
    if not anomaly_metric_report.empty:
        anomaly_metric_report.to_csv(out_dirs["reports"] / "anomaly_metric_diagnostics_v11.csv", index=False, encoding="utf-8-sig")

    metrics_summary = {
        "ensemble": ensemble_metrics,
        "base_ensemble": base_metrics,
        "segment_before_county": segment_metrics_final,
        "county_metrics": county_metrics.to_dict(orient="records") if not county_metrics.empty else [],
        "segment_layer": segment_report.to_dict(orient="records"),
        "county_expert_layer": county_report.to_dict(orient="records"),
        "anomaly_metric_diagnostics": anomaly_metric_report.to_dict(orient="records") if not anomaly_metric_report.empty else [],
        "model_comparison": model_comparison.to_dict(orient="records"),
    }
    (out_dirs["reports"] / "metrics_summary_v11.json").write_text(
        json.dumps(metrics_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    save_group_error_reports(oof, out_dirs["reports"])
    save_plots(oof, out_dirs["reports"])
    save_feature_importance(selected_models, out_dirs["reports"])
    save_feature_importance_by_county(county_models_all, out_dirs["reports"])

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
        rep.to_csv(reports_dir / f"error_by_{col}_v11.csv", index=False, encoding="utf-8-sig")


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
    plt.title("V11 OOF: Gerçek ve Tahmin Edilen Birim Fiyat")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig(reports_dir / "actual_vs_predicted_v11.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(oof["error"].dropna(), bins=35, alpha=0.85)
    plt.xlabel("Tahmin Hatası (TL/m²)")
    plt.ylabel("İlan Sayısı")
    plt.title("V11 OOF: Hata Dağılımı")
    plt.grid(True, axis="y", linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig(reports_dir / "residual_distribution_v11.png", dpi=300, bbox_inches="tight")
    plt.close()

    if "district" in oof.columns:
        rep = oof.groupby("district").agg(n=("abs_pct_error", "size"), mape=("abs_pct_error", "mean")).reset_index()
        rep = rep[rep["n"] >= 5].sort_values("mape", ascending=True).tail(20)
        if not rep.empty:
            plt.figure(figsize=(9, 7))
            plt.barh(rep["district"].astype(str), rep["mape"] * 100)
            plt.xlabel("MAPE (%)")
            plt.title("V11 OOF: Mahalle Bazlı En Yüksek MAPE")
            plt.grid(True, axis="x", linestyle="--", alpha=0.45)
            plt.tight_layout()
            plt.savefig(reports_dir / "mape_by_district_v11.png", dpi=300, bbox_inches="tight")
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
    rep.to_csv(reports_dir / "feature_importance_v11.csv", index=False, encoding="utf-8-sig")
    top = rep.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False).head(50)
    top.to_csv(reports_dir / "feature_importance_top50_v11.csv", index=False, encoding="utf-8-sig")


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
        "# V11 Model Run",
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
        "- data/input/sales_cleaned_v11.csv",
        "- data/input/rentals_cleaned_v11.csv",
        "- data/output/oof_predictions_v11.csv",
        "- reports/model_comparison_v11.csv",
        "- reports/metrics_summary_v11.json",
        "- reports/error_by_*_v11.csv",
        "- reports/*.png",
        "- artifacts/model_*_v11.joblib",
        "- artifacts/model_bundle_v11.joblib",
        "",
    ]
    (out_dirs["base"] / "README_v11_run.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", default=None, help="PostgreSQL/Neon connection string. Empty reads DATABASE_URL or DB_URL constant.")
    ap.add_argument("--sale-table", default=DEFAULT_SALE_TABLE)
    ap.add_argument("--rental-table", default=DEFAULT_RENTAL_TABLE)
    ap.add_argument("--trend-table", default=DEFAULT_TREND_TABLE)
    ap.add_argument("--city", default=DEFAULT_CITY)
    ap.add_argument("--counties", default=",".join(DEFAULT_COUNTIES), help="Comma-separated county list.")
    ap.add_argument("--out", default="outputs/v11_demographics_pipeline")
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
    ap.add_argument("--county-experts", action=argparse.BooleanOptionalAction, default=True, help="Train and validate county-specific expert blend layer.")
    ap.add_argument("--county-expert-min-rows", type=int, default=250, help="Minimum rows needed to train a county expert model.")
    ap.add_argument("--anomaly-reports", action=argparse.BooleanOptionalAction, default=True, help="Create listing anomaly score reports without adding app-unavailable features.")
    ap.add_argument("--demographics-table", default="district_demographics", help="PostgreSQL table containing district-level demographic features.")
    ap.add_argument("--demographics-mode", choices=["none", "safe", "full"], default="safe", help="Demographic feature mode for final training run.")
    ap.add_argument("--run-demographics-ablation", action=argparse.BooleanOptionalAction, default=True, help="Run none/safe/full demographic ablation before final selected run.")
    ap.add_argument("--exclude-anomalies-threshold", type=float, default=25.0, help="Exclude rows with anomaly_score >= threshold before training. Set 0 to disable.")
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
        enable_county_experts=bool(args.county_experts),
        county_expert_min_rows=int(args.county_expert_min_rows),
        enable_anomaly_reports=bool(args.anomaly_reports),
        demographics_mode=str(args.demographics_mode),
        demographics_table=str(args.demographics_table),
        exclude_anomalies_threshold=float(args.exclude_anomalies_threshold),
    )

    out_dirs = build_output_dirs(args.out)
    (out_dirs["base"] / "run_config_v11.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")

    print("Loading raw data...")
    sales_raw, rentals_raw, trend_raw = load_raw_data(args, out_dirs)

    print("Cleaning sales and rentals...")
    sales_clean, rentals_clean, cleaning_report = clean_sales_and_rentals(sales_raw, rentals_raw, cfg, out_dirs)
    print(json.dumps(cleaning_report, indent=2, ensure_ascii=False))

    print("Attaching rental market features...")
    sales_with_rent, rental_feature_report = attach_rental_features(sales_clean, rentals_clean)

    print("Attaching trend market features...")
    sales_final_base, trend_feature_report = attach_trend_features(sales_with_rent, trend_raw, cfg)

    print("Applying location-baseline outlier filter...")
    sales_final_base, sales_removed_location, location_outlier_report = apply_location_outlier_filter(sales_final_base, cfg, out_dirs)
    cleaning_report.update(location_outlier_report)

    # Demographics are loaded from DB, not CSV. If unavailable, safe/full fall back to no demo data with a report.
    demographic_raw = pd.DataFrame()
    db_url = args.db_url or DB_URL or os.getenv("DATABASE_URL")
    if str(args.demographics_mode).lower() != "none" or bool(args.run_demographics_ablation):
        try:
            engine = create_db_engine(db_url)
            print(f"DB demographics are being fetched from {args.demographics_table}...")
            demographic_raw = fetch_demographics_table(engine, args.demographics_table, args.city)
            if not demographic_raw.empty:
                demographic_raw.to_csv(out_dirs["raw"] / "district_demographics_from_db.csv", index=False, encoding="utf-8-sig")
                print(f"Demographic rows: {len(demographic_raw)}")
        except Exception as exc:
            warnings.warn(f"Demographics could not be loaded; continuing without demographic features. Error: {exc}")
            demographic_raw = pd.DataFrame()

    def run_one_demographics_mode(mode: str, mode_out_dirs: dict[str, Path], final_run: bool = False):
        mode_cfg = RunConfig(**{**asdict(cfg), "demographics_mode": mode})
        demo_features = build_demographic_features(demographic_raw, mode=mode)
        sales_with_demo, demo_feature_report = attach_demographic_features(sales_final_base, demo_features, mode, mode_out_dirs)
        sales_with_demo.to_csv(mode_out_dirs["input"] / f"sales_training_table_v11_{mode}.csv", index=False, encoding="utf-8-sig")

        X, y = make_training_matrix(sales_with_demo)
        print(f"[{mode}] Training rows before anomaly exclusion: {len(X)} | Features before pipeline: {X.shape[1]}")
        if len(X) == 0:
            raise ValueError(f"No valid sale rows left after cleaning for demographics mode={mode}.")

        anomaly_filter_report = {
            "training_rows_before_anomaly_filter": int(len(X)),
            "training_rows_after_anomaly_filter": int(len(X)),
            "excluded_anomaly_rows": 0,
            "exclude_anomalies_threshold": float(mode_cfg.exclude_anomalies_threshold),
        }
        if mode_cfg.exclude_anomalies_threshold and mode_cfg.exclude_anomalies_threshold > 0:
            pre_anom = compute_anomaly_scores(X, y, mode_cfg)
            keep_mask = pd.to_numeric(pre_anom["anomaly_score"], errors="coerce").fillna(0) < float(mode_cfg.exclude_anomalies_threshold)
            excluded = int((~keep_mask).sum())
            pre_anom.assign(training_excluded_by_threshold=(~keep_mask).astype(int)).to_csv(
                mode_out_dirs["reports"] / f"anomaly_training_filter_{mode}_v11.csv", index=False, encoding="utf-8-sig"
            )
            if excluded > 0:
                X = X.loc[keep_mask].reset_index(drop=True)
                y = y.loc[keep_mask].reset_index(drop=True)
            anomaly_filter_report.update({
                "training_rows_after_anomaly_filter": int(len(X)),
                "excluded_anomaly_rows": excluded,
            })
            print(f"[{mode}] Excluded anomaly rows: {excluded} | after filter: {len(X)}")

        bundle, metrics_summary = train_and_evaluate(X, y, mode_cfg, mode_out_dirs)
        metrics_summary["ensemble"]["demographics_mode"] = mode
        metrics_summary["ensemble"].update(anomaly_filter_report)
        metrics_summary["demographic_features"] = demo_feature_report
        metrics_summary["anomaly_training_filter"] = anomaly_filter_report
        (mode_out_dirs["reports"] / "metrics_summary_v11.json").write_text(
            json.dumps(metrics_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return bundle, metrics_summary, demo_feature_report, anomaly_filter_report

    ablation_rows: list[dict[str, Any]] = []
    ablation_summaries: dict[str, Any] = {}
    modes_to_run = ["none", "safe", "full"] if bool(args.run_demographics_ablation) else [str(args.demographics_mode)]
    last_bundle = None
    last_metrics_summary: dict[str, Any] | None = None
    last_demo_feature_report: dict[str, Any] = {}
    last_anomaly_filter_report: dict[str, Any] = {}
    for mode in modes_to_run:
        print(f"\n========== V11 DEMOGRAPHICS MODE: {mode} ==========")
        mode_dirs = build_output_dirs(out_dirs["base"] / f"ablation_{mode}") if bool(args.run_demographics_ablation) else out_dirs
        last_bundle, ms, demo_rep, filt_rep = run_one_demographics_mode(mode, mode_dirs)
        last_metrics_summary = ms
        last_demo_feature_report = demo_rep
        last_anomaly_filter_report = filt_rep
        row = {"demographics_mode": mode, **ms["ensemble"]}
        ablation_rows.append(row)
        ablation_summaries[mode] = {"metrics": ms["ensemble"], "demographic_features": demo_rep, "anomaly_filter": filt_rep}

    ablation_df = pd.DataFrame(ablation_rows)
    if not ablation_df.empty:
        keep_cols = [c for c in ["demographics_mode", "rows", "r2", "log_r2", "mape", "median_ape", "mae_tl_per_m2", "median_ae_tl_per_m2", "excluded_anomaly_rows", "training_rows_after_anomaly_filter"] if c in ablation_df.columns]
        ablation_df[keep_cols].to_csv(out_dirs["reports"] / "metrics_demographics_ablation_v11.csv", index=False, encoding="utf-8-sig")

    final_mode = str(args.demographics_mode)
    if bool(args.run_demographics_ablation):
        print(f"\n========== V11 FINAL SELECTED MODE: {final_mode} ==========")
        bundle, metrics_summary, demo_feature_report, anomaly_filter_report = run_one_demographics_mode(final_mode, out_dirs, final_run=True)
    else:
        # run_one_demographics_mode already used the main output dirs.
        if last_metrics_summary is None:
            raise RuntimeError("Internal error: no metrics summary produced.")
        bundle, metrics_summary, demo_feature_report, anomaly_filter_report = last_bundle, last_metrics_summary, last_demo_feature_report, last_anomaly_filter_report

    feature_reports = {
        "rental_features": rental_feature_report,
        "trend_features": trend_feature_report,
        "demographic_features": demo_feature_report,
        "demographics_ablation": ablation_summaries,
    }
    metrics_summary["demographics_ablation"] = ablation_summaries
    metrics_summary["final_demographics_mode"] = final_mode
    (out_dirs["reports"] / "metrics_summary_v11.json").write_text(
        json.dumps(metrics_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_run_readme(out_dirs, cfg, metrics_summary, cleaning_report, feature_reports)

    print("\n=== V11 ENSEMBLE METRICS ===")
    print(json.dumps(metrics_summary["ensemble"], indent=2, ensure_ascii=False))
    print(f"\nOutputs written to: {out_dirs['base'].resolve()}")


if __name__ == "__main__":
    main()
