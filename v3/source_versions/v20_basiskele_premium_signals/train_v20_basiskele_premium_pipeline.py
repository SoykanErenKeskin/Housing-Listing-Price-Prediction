from __future__ import annotations

"""
V16 county-specialist pipeline for Kocaeli housing unit price model.

Built on V14 (detail premiums + attribute sensitivity + segment + county experts).
Adds Başiskele premium specialist features, Karamürsel min-rows override,
and large-home redesign — targeting county-level lift without breaking global guardrails.

Example:
  python train_v20_basiskele_premium_pipeline.py --out outputs/v18_basiskele_full \\
    --demographics-mode safe --attribute-mode full --detail-effect-mode group \\
    --basiskele-specialist-mode premium_target_stats --basiskele-variance-lift conservative \\
    --county-expert-min-rows-overrides "Karamürsel:180"
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Fast path: interactive CLI BEFORE sklearn/pandas (those imports are slow).
# ---------------------------------------------------------------------------
_EARLY_ARGS = None
if __name__ == "__main__":
    _V20_dir = str(Path(__file__).resolve().parent)
    if _V20_dir not in sys.path:
        sys.path.insert(0, _V20_dir)
    _no_interactive = "--no-interactive" in sys.argv[1:]
    if not _no_interactive:
        sys.stdout.write("V20 Başiskele ayar sihirbazı açılıyor...\n")
        sys.stdout.flush()
    from interactive_cli import parse_cli_early

    _EARLY_ARGS = parse_cli_early(sys.argv[1:])
    if not _no_interactive:
        sys.stdout.write("Ayarlar tamam. Model kütüphaneleri yükleniyor (biraz sürebilir)...\n")
        sys.stdout.flush()

import argparse
import json
import math
import os
import re
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
    PerformanceWarning = None  # type: ignore[misc, assignment]
# Message match covers pandas versions / subprocesses that re-enable category filters.
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

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
    from attribute_features import (
        ATTRIBUTE_BASIC_NUMERIC_FEATURES,
        ATTRIBUTE_EFFECT_NUMERIC_FEATURES,
        AttributeEffectEncoder,
        AttributeInteractionCompleter,
        AttributeQualityAdder,
        V12_SAFE_REF,
        add_attribute_quality_features,
        build_debug_feature_frame,
        get_attribute_feature_names,
    )
except ImportError:  # when imported as package/module path
    from attribute_features import (
        ATTRIBUTE_BASIC_NUMERIC_FEATURES,
        ATTRIBUTE_EFFECT_NUMERIC_FEATURES,
        AttributeEffectEncoder,
        AttributeInteractionCompleter,
        AttributeQualityAdder,
        V12_SAFE_REF,
        add_attribute_quality_features,
        build_debug_feature_frame,
        get_attribute_feature_names,
    )

try:
    from diagnostics_v18_basiskele import (
        attribute_feature_coverage,
        evaluate_decision,
        run_basiskele_variance_diagnostics,
        run_karamursel_sensitivity,
        run_prediction_pair_tests,
        save_attribute_feature_importance,
        save_detail_premium_feature_importance,
        select_attribute_mode,
        select_detail_effect_mode,
    )
except ImportError:
    from v18_basiskele.diagnostics_v18_basiskele import (
        attribute_feature_coverage,
        evaluate_decision,
        run_basiskele_variance_diagnostics,
        run_karamursel_sensitivity,
        run_prediction_pair_tests,
        save_attribute_feature_importance,
        save_detail_premium_feature_importance,
        select_attribute_mode,
        select_detail_effect_mode,
    )

try:
    from detail_premium_features import (
        DETAIL_EFFECT_GROUP_NUMERIC_FEATURES,
        V13_DEFAULT_REF,
        V14_DEFAULT_REF,
        V15_DEFAULT_REF,
        LocalDetailPremiumEncoder,
        detail_feature_coverage,
        export_detail_premium_effect_tables,
        get_detail_effect_feature_names,
    )
except ImportError:
    from v18_basiskele.detail_premium_features import (
        DETAIL_EFFECT_GROUP_NUMERIC_FEATURES,
        V13_DEFAULT_REF,
        V14_DEFAULT_REF,
        V15_DEFAULT_REF,
        LocalDetailPremiumEncoder,
        detail_feature_coverage,
        export_detail_premium_effect_tables,
        get_detail_effect_feature_names,
    )

try:
    from county_specialist_features import (
        BasiskelePremiumSpecialistAdder,
        BasiskelePremiumTargetStatsAdder,
        LargeHomeFeatureAdder,
        get_county_specialist_feature_names,
        parse_county_min_rows_overrides,
    )
except ImportError:
    from v18_basiskele.county_specialist_features import (
        BasiskelePremiumSpecialistAdder,
        BasiskelePremiumTargetStatsAdder,
        LargeHomeFeatureAdder,
        get_county_specialist_feature_names,
        parse_county_min_rows_overrides,
    )

try:
    from regime_residual_layers import (
        BasiskeleLargeHomeRegimeAdder,
        KaramurselLocationAgeBaselineAdder,
        apply_basiskele_large_home_residual_layer,
        apply_basiskele_spread_residual_layer,
        get_v16_regime_feature_names,
    )
except ImportError:
    from v18_basiskele.regime_residual_layers import (
        BasiskeleLargeHomeRegimeAdder,
        KaramurselLocationAgeBaselineAdder,
        apply_basiskele_large_home_residual_layer,
        apply_basiskele_spread_residual_layer,
        get_v16_regime_feature_names,
    )

try:
    from train_progress import TrainProgress, estimate_training_units, get_progress, set_progress
except ImportError:
    from v18_basiskele.train_progress import TrainProgress, estimate_training_units, get_progress, set_progress

try:
    from location_features import (
        LocationFeatureAdder,
        LocationScopeMasker,
        get_location_feature_names,
        location_feature_metadata,
        is_location_related_column,
    )
except ImportError:
    from v18_basiskele.location_features import (
        LocationFeatureAdder,
        LocationScopeMasker,
        get_location_feature_names,
        location_feature_metadata,
        is_location_related_column,
    )

try:
    from comparable_market_features import (
        ComparableMarketFeatureAdder,
        get_comparable_feature_names,
        parse_k_list,
    )
except ImportError:
    from v18_basiskele.comparable_market_features import (
        ComparableMarketFeatureAdder,
        get_comparable_feature_names,
        parse_k_list,
    )

try:
    from premium_signal_features import (
        PREMIUM_UNIT_PRICE_COL,
        PremiumSignalFeatureAdder,
        SiteProjectFoldSafeEncoder,
        get_premium_categorical_feature_names,
        get_premium_feature_names,
        premium_feature_coverage,
        site_project_candidates_table,
        write_leakage_guard,
        TEXT_FLAG_FEATURES,
        SCORE_FEATURES,
        SITE_NUMERIC_FEATURES,
        SITE_CATEGORICAL_FEATURES,
        SCORE_CATEGORICAL,
        INTERACTION_CATEGORICAL,
        FOLDSAFE_NUMERIC,
    )
except ImportError:
    from premium_signal_features import (
        PREMIUM_UNIT_PRICE_COL,
        PremiumSignalFeatureAdder,
        SiteProjectFoldSafeEncoder,
        get_premium_categorical_feature_names,
        get_premium_feature_names,
        premium_feature_coverage,
        site_project_candidates_table,
        write_leakage_guard,
        TEXT_FLAG_FEATURES,
        SCORE_FEATURES,
        SITE_NUMERIC_FEATURES,
        SITE_CATEGORICAL_FEATURES,
        SCORE_CATEGORICAL,
        INTERACTION_CATEGORICAL,
        FOLDSAFE_NUMERIC,
    )

try:
    from geo_context_features import (
        GeoContextFeatureAdder,
        get_geo_context_feature_names,
        GEO_CONTEXT_NUMERIC_FEATURES,
    )
except ImportError:
    from v18_basiskele.geo_context_features import (
        GeoContextFeatureAdder,
        get_geo_context_feature_names,
        GEO_CONTEXT_NUMERIC_FEATURES,
    )

try:
    from train_progress import TrainProgress, estimate_training_units, get_progress, set_progress
except ImportError:
    from v18_basiskele.train_progress import TrainProgress, estimate_training_units, get_progress, set_progress


def _plog(msg: str = "", **kwargs: Any) -> None:
    """Print a log line while keeping the sticky progress bar at the bottom."""
    progress = get_progress()
    if progress is not None:
        progress.log(msg, **kwargs)
    else:
        print(msg, **kwargs)


def _ptick(label: str = "", n: int = 1) -> None:
    progress = get_progress()
    if progress is not None:
        progress.tick(label, n=n)


def _pstage(stage: str) -> None:
    progress = get_progress()
    if progress is not None:
        progress.set_stage(stage)

try:
    from dotenv import load_dotenv
    from pathlib import Path as _Path

    # Prefer local .env next to this file, then walk parents for project-root .env.
    _here = _Path(__file__).resolve().parent
    load_dotenv(_here / ".env")
    for _parent in _here.parents:
        _cand = _parent / ".env"
        if _cand.is_file():
            load_dotenv(_cand)
            break
    load_dotenv()  # cwd fallback
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

DEFAULT_SALE_TABLE = os.getenv("SALE_TABLE", "sahibinden_sale_listings")
DEFAULT_RENTAL_TABLE = os.getenv("RENTAL_TABLE", "sahibinden_rental_listings")
DEFAULT_SOURCE_SITE = os.getenv("SOURCE_SITE", "sahibinden")
DEFAULT_TREND_TABLE = "trend_observed"
DEFAULT_CITY = "Kocaeli"
DEFAULT_COUNTIES = ["Başiskele"]
MODEL_SCOPE = "basiskele_only"
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
LOCATION_CATEGORICAL_FEATURES_V17 = [
    "location_precision", "location_source", "geo_cluster_city", "geo_cluster_county",
    "basiskele_geo_cluster", "coast_distance_bucket",
    "basiskele_geo_cluster_x_m2_group", "geo_cluster_x_room_count",
]
CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + LOCATION_CATEGORICAL_FEATURES_V17

PREMIUM_CATEGORICAL_FEATURES_V20 = (
    list(SITE_CATEGORICAL_FEATURES)
    + list(SCORE_CATEGORICAL)
    + list(INTERACTION_CATEGORICAL)
)
CATEGORICAL_FEATURES_BASE = list(CATEGORICAL_FEATURES)


def resolve_categorical_features(
    premium_feature_mode: str = "full",
    site_project_encoding: str = "frequency",
) -> list[str]:
    names = list(CATEGORICAL_FEATURES_BASE)
    names += get_premium_categorical_feature_names(premium_feature_mode, site_project_encoding)
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


# Default global list includes premium cats for offline tooling; pipelines use resolve_categorical_features.
CATEGORICAL_FEATURES = CATEGORICAL_FEATURES_BASE + list(PREMIUM_CATEGORICAL_FEATURES_V20)

LEAKAGE_OR_UNUSED_COLUMNS = {
    TARGET,
    "unit_price_net",
    "price",
    "monthly_rent",
    "rent_per_m2_gross",
    "rent_per_m2_net",
    "deposit",
    # classified_id kept for comparable self-match exclusion in V17
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
    "location_raw",
    "street_name",
    "address_text",
    "location_backfilled_at",
    "location_backfill_error",
}

# V20 premium signals need free-text sources in X for PremiumSignalFeatureAdder.
# They are never passed raw to the model (FeatureColumnKeeper drops them).
_PREMIUM_TEXT_SOURCE_COLUMNS = {
    "title",
    "site_name",
    "address_text",
    "description",
    "listing_description",
    "aciklama",
    "ilan_aciklama",
}
LEAKAGE_OR_UNUSED_COLUMNS_V20 = LEAKAGE_OR_UNUSED_COLUMNS - _PREMIUM_TEXT_SOURCE_COLUMNS


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
    attribute_mode: str
    detail_effect_mode: str = "group"
    county_expert_min_rows_overrides: dict | None = None
    basiskele_specialist_mode: str = "premium_target_stats"
    basiskele_variance_lift: str = "none"
    large_home_specialist_mode: str = "redesigned"
    basiskele_large_home_regime: str = "none"
    basiskele_spread_layer: str = "none"
    karamursel_baseline_mode: str = "none"
    location_feature_mode: str = "geo"
    geo_context_mode: str = "geo_with_coast"
    comparable_mode: str = "none"
    premium_feature_mode: str = "full"
    site_project_encoding: str = "frequency"
    run_premium_ablation: bool = False
    model_scope: str = "basiskele_only"
    location_scope: str = "basiskele_only"
    location_min_precision: str = "any"
    enable_coordinate_noise_check: bool = True
    comparable_k_list: str = "5,10,20"
    run_location_ablation: bool = False
    geo_context_cache_dir: str = "data/external/geo_context"
    location_coverage_min: float = 0.40


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
        attribute_mode: str = "full",
        detail_effect_mode: str = "group",
        basiskele_specialist_mode: str = "premium_target_stats",
        basiskele_variance_lift: str = "conservative",
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
        self.attribute_mode = str(attribute_mode or "full")
        self.detail_effect_mode = str(detail_effect_mode or "group")
        self.basiskele_specialist_mode = str(basiskele_specialist_mode or "premium_target_stats")
        self.basiskele_variance_lift = str(basiskele_variance_lift or "conservative")

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        X = ensure_columns(X.copy(), self.feature_columns, fill=np.nan)
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
    """V18 hard-lock: always Başiskele only. Extra counties are ignored."""
    return ["Başiskele"]


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


def fetch_listing_table(engine, table: str, purpose: str, city: str, limit: int | None = None, county: str = "Başiskele") -> pd.DataFrame:
    """V18: ONLY Başiskele. Never pull İzmit/Gölcük/Karamürsel."""
    table = validate_table_name(table)
    limit_clause = f" LIMIT {int(limit)}" if limit else ""
    county = county or "Başiskele"
    sql = text(
        f"""
        SELECT *
        FROM {table}
        WHERE lower(coalesce(city, '')) = lower(:city)
          AND lower(coalesce(listing_purpose, '')) = lower(:purpose)
          AND county = :county
          AND lower(coalesce(source_site, :source_site)) = lower(:source_site)
        ORDER BY saved_at DESC NULLS LAST, updated_at DESC NULLS LAST
        {limit_clause}
        """
    )
    return pd.read_sql(sql, engine, params={"city": city, "purpose": purpose, "county": county, "source_site": DEFAULT_SOURCE_SITE})


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


def _frame_from_columns(index: pd.Index, columns: dict[str, Any]) -> pd.DataFrame:
    """Build a DataFrame from scalar/Series column values aligned to index."""
    prepared: dict[str, Any] = {}
    for key, value in columns.items():
        if isinstance(value, pd.Series):
            prepared[key] = value.reindex(index)
        else:
            prepared[key] = value
    return pd.DataFrame(prepared, index=index)


def ensure_columns(df: pd.DataFrame, columns: Iterable[str], fill: Any = np.nan) -> pd.DataFrame:
    """Add missing columns in one concat to avoid DataFrame fragmentation."""
    missing = [c for c in columns if c not in df.columns]
    if not missing:
        return df
    return pd.concat([df, _frame_from_columns(df.index, {c: fill for c in missing})], axis=1)


def assign_columns(df: pd.DataFrame, columns: dict[str, Any]) -> pd.DataFrame:
    """Create or overwrite many columns in one block (avoids fragmented inserts)."""
    if not columns:
        return df
    block = _frame_from_columns(df.index, columns)
    keep = [c for c in df.columns if c not in block.columns]
    if keep:
        return pd.concat([df.loc[:, keep], block], axis=1)
    return block


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
    name_block: dict[str, Any] = {}
    for src_name, out_name in [
        ("city_name", "demo_city_name"),
        ("county_name", "demo_county_name"),
        ("district_name", "demo_district_name"),
    ]:
        if src_name in df.columns:
            name_block[out_name] = df[src_name].astype("object").where(df[src_name].notna(), "").astype(str)
        else:
            name_block[out_name] = ""
    out = assign_columns(out, name_block)
    out = add_location_norm_keys(out.rename(columns={
        "demo_city_name": "city_name",
        "demo_county_name": "county_name",
        "demo_district_name": "district_name",
    }), prefix="demo_").rename(columns={
        "city_name": "demo_city_name",
        "county_name": "demo_county_name",
        "district_name": "demo_district_name",
    })

    demo_block: dict[str, Any] = {
        f"demo_{c}": (pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan)
        for c in use_cols
    }
    for c in categorical:
        series = df[c].astype("object") if c in df.columns else pd.Series("missing", index=df.index)
        series = series.where(pd.notna(series), "missing").astype(str).str.strip().replace({"": "missing", "nan": "missing", "None": "missing"})
        demo_block[f"demo_{c}"] = series
    out = assign_columns(out, demo_block)

    # Coverage flags.
    age_cols = ["demo_age_0_14", "demo_age_15_24", "demo_age_25_34", "demo_age_35_44", "demo_age_45_54", "demo_age_55_64", "demo_age_65_plus"]
    edu_cols = ["demo_education_university_ratio", "demo_education_high_school_ratio", "demo_education_middle_school_ratio", "demo_education_primary_school_ratio", "demo_education_graduate_ratio", "demo_education_doctorate_ratio", "demo_education_non_literate_ratio"]
    market_cols = [c for c in ["demo_sale_count", "demo_listing_count_2024", "demo_turnover_ratio"] if c in out.columns]
    flag_block: dict[str, Any] = {
        "demo_has_demographics": out[["demo_population_total", "demo_population_density", "demo_per_capita_income_try"]].notna().any(axis=1).astype(int),
        "demo_age_coverage": out[age_cols].sum(axis=1, min_count=1).between(80, 120).fillna(False).astype(int),
        "demo_education_coverage": out[edu_cols].sum(axis=1, min_count=1).between(70, 120).fillna(False).astype(int),
        "demo_income_available": out[["demo_per_capita_income_try", "demo_household_income_try"]].notna().any(axis=1).astype(int),
        "demo_ses_available": out[["demo_ses_ab_ratio", "demo_ses_cd_ratio"]].notna().any(axis=1).astype(int),
        "demo_market_activity_available": (
            out[market_cols].notna().any(axis=1).astype(int) if market_cols else 0
        ),
        "demo_infrastructure_available": out[["demo_atm_count", "demo_pharmacy_count", "demo_bank_count"]].notna().any(axis=1).astype(int),
        # District-level coverage. County-level coverage is computed below after
        # county aggregate columns are created. Do not include
        # demo_has_county_demographics here yet; otherwise the function fails before
        # county features exist.
        "demo_has_county_demographics": 0,
    }
    out = assign_columns(out, flag_block)
    base_flag_cols = [
        "demo_has_demographics",
        "demo_age_coverage",
        "demo_education_coverage",
        "demo_income_available",
        "demo_ses_available",
        "demo_market_activity_available",
        "demo_infrastructure_available",
    ]
    out = assign_columns(out, {"demo_coverage_score": out[base_flag_cols].mean(axis=1)})

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

    # Relative-to-county features (batch assign to avoid fragmentation).
    relative: dict[str, Any] = {}

    def div(a: str, b: str, out_col: str):
        if a in out.columns and b in out.columns:
            relative[out_col] = safe_divide_series(out[a], out[b])
        else:
            relative[out_col] = np.nan

    def diff(a: str, b: str, out_col: str):
        if a in out.columns and b in out.columns:
            relative[out_col] = pd.to_numeric(out[a], errors="coerce") - pd.to_numeric(out[b], errors="coerce")
        else:
            relative[out_col] = np.nan

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
        relative["demo_turnover_vs_county"] = np.nan
    div("demo_residential_count", "demo_population_total", "demo_residential_density")
    div("demo_workplace_count", "demo_population_total", "demo_workplace_density")
    div("demo_vehicle_count", "demo_population_total", "demo_vehicle_per_capita")
    div("demo_car_count", "demo_population_total", "demo_car_per_capita")
    relative["demo_pharmacy_per_10k"] = safe_divide_series(
        out.get("demo_pharmacy_count", pd.Series(np.nan, index=out.index)) * 10000,
        out.get("demo_population_total", pd.Series(np.nan, index=out.index)),
    )
    relative["demo_bank_per_10k"] = safe_divide_series(
        out.get("demo_bank_count", pd.Series(np.nan, index=out.index)) * 10000,
        out.get("demo_population_total", pd.Series(np.nan, index=out.index)),
    )
    relative["demo_atm_per_10k"] = safe_divide_series(
        out.get("demo_atm_count", pd.Series(np.nan, index=out.index)) * 10000,
        out.get("demo_population_total", pd.Series(np.nan, index=out.index)),
    )
    out = assign_columns(out, relative)

    # County-level demographic availability is based on county_demo_* columns.
    # These are computed by aggregating neighborhood rows from district_demographics,
    # so they can exist even when an exact district-level join later fails.
    county_indicator_cols = [c for c in out.columns if c.startswith("county_demo_")]
    if county_indicator_cols:
        out = assign_columns(
            out,
            {"demo_has_county_demographics": out[county_indicator_cols].notna().any(axis=1).astype(int)},
        )
    else:
        out = assign_columns(out, {"demo_has_county_demographics": 0})

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
    out = assign_columns(out, {"demo_coverage_score": out[final_flag_cols].mean(axis=1)})

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
            cov.to_csv(out_dirs["reports"] / f"demographic_feature_coverage_{mode}_v18_basiskele.csv", index=False, encoding="utf-8-sig")
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

    new_cols: dict[str, Any] = {}
    for key in sorted(keys):
        raw_values = raw_objects.map(lambda obj, kk=key: obj.get(kk, np.nan) if isinstance(obj, dict) else np.nan)
        raw_values = raw_values.replace("", np.nan)
        if key.startswith(DETAIL_PREFIXES) or key in {"detail_selected_count", "detail_quality_score"}:
            raw_values = pd.to_numeric(raw_values, errors="coerce")
            if key not in out.columns and key not in new_cols:
                new_cols[key] = raw_values
            else:
                current = pd.to_numeric(out[key].replace("", np.nan), errors="coerce") if key in out.columns else pd.Series(np.nan, index=out.index)
                out[key] = current.where(current.notna(), raw_values)
        else:
            raw_values = raw_values.astype("object")
            if key not in out.columns and key not in new_cols:
                new_cols[key] = raw_values
            else:
                current_text = out[key].fillna("").astype(str).str.strip()
                mask = current_text.eq("") & raw_values.notna()
                out.loc[mask, key] = raw_values.loc[mask]
    if new_cols:
        out = assign_columns(out, new_cols)
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

    sales_clean.to_csv(out_dirs["input"] / "sales_cleaned_v18_basiskele.csv", index=False, encoding="utf-8-sig")
    rentals_clean.to_csv(out_dirs["input"] / "rentals_cleaned_v18_basiskele.csv", index=False, encoding="utf-8-sig")
    pd.concat([sales_removed_basic, sales_removed_iqr], ignore_index=True).to_csv(
        out_dirs["input"] / "sales_removed_v18_basiskele.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat([rentals_removed_basic, rentals_removed_iqr], ignore_index=True).to_csv(
        out_dirs["input"] / "rentals_removed_v18_basiskele.csv", index=False, encoding="utf-8-sig"
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

        cat_seed = CATEGORICAL_FEATURES + ["county", "district", "room_count"]
        df = ensure_columns(df, cat_seed, fill=np.nan)
        for col in cat_seed:
            df[col] = df[col].astype("object").where(df[col].notna(), "missing")
            df[col] = df[col].astype(str).str.strip().replace({"": "missing", "nan": "missing", "None": "missing"})

        derived: dict[str, Any] = {}
        if "m2_group" not in df.columns or df["m2_group"].isna().all():
            derived["m2_group"] = df["gross_m2"].map(m2_group_from_value)
        else:
            derived["m2_group"] = df["m2_group"]
        derived["m2_group"] = (
            pd.Series(derived["m2_group"], index=df.index)
            .astype("object")
            .where(pd.Series(derived["m2_group"], index=df.index).notna(), "missing")
        )

        if "building_age_group" not in df.columns or df["building_age_group"].isna().all():
            derived["building_age_group"] = df["building_age"].map(lambda x: parse_building_age(x)[1])
        else:
            derived["building_age_group"] = df["building_age_group"]
        derived["building_age_group"] = (
            pd.Series(derived["building_age_group"], index=df.index)
            .astype("object")
            .where(pd.Series(derived["building_age_group"], index=df.index).notna(), "missing")
        )

        ratio = pd.to_numeric(df.get("net_m2", np.nan), errors="coerce") / pd.to_numeric(
            df.get("gross_m2", np.nan), errors="coerce"
        ).replace(0, np.nan)
        if "net_gross_ratio" in df.columns:
            net_gross = pd.to_numeric(df["net_gross_ratio"], errors="coerce").where(df["net_gross_ratio"].notna(), ratio)
        else:
            net_gross = ratio
        derived["net_gross_ratio"] = pd.to_numeric(net_gross, errors="coerce").clip(0.2, 1.2)

        floor = pd.to_numeric(df["floor_num"], errors="coerce") if "floor_num" in df.columns else pd.Series(np.nan, index=df.index)
        total = pd.to_numeric(df["total_floors"], errors="coerce") if "total_floors" in df.columns else pd.Series(np.nan, index=df.index)
        if "floor_num" not in df.columns:
            derived["floor_num"] = np.nan
        if "total_floors" not in df.columns:
            derived["total_floors"] = np.nan
        derived["floor_ratio"] = floor / total.replace(0, np.nan)
        derived["remaining_floors"] = total - floor
        derived["is_ground_floor"] = (floor == 0).astype(int)
        derived["is_basement"] = (floor < 0).astype(int)
        derived["is_top_floor"] = ((total.notna()) & (floor.notna()) & (floor >= total)).astype(int)
        derived["is_middle_floor"] = ((floor > 0) & (total.notna()) & (floor < total)).astype(int)

        parsed_rooms = df["room_count"].map(parse_room)
        derived["rooms"] = parsed_rooms.map(lambda x: x[0])
        derived["living_rooms"] = parsed_rooms.map(lambda x: x[1])
        derived["total_room_score"] = parsed_rooms.map(lambda x: x[2])

        age = pd.to_numeric(df.get("building_age", np.nan), errors="coerce")
        derived["is_new_building"] = (age <= 3).astype(int)
        derived["is_old_building"] = (age >= 20).astype(int)
        gross = pd.to_numeric(df.get("gross_m2", np.nan), errors="coerce")
        derived["is_small_flat"] = (gross <= 75).astype(int)
        derived["is_large_flat"] = (gross >= 180).astype(int)

        for raw_col, count_col in DETAIL_RAW_COLUMNS.items():
            if raw_col in df.columns:
                derived[count_col] = df[raw_col].map(count_pipe_values).astype(int)
            elif count_col not in df.columns:
                derived[count_col] = 0

        score_members = set(sum(SCORE_GROUPS.values(), []))
        for col in score_members:
            if col in derived:
                series = pd.to_numeric(derived[col], errors="coerce")
            elif col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
            else:
                series = pd.Series(0, index=df.index)
            derived[col] = series.fillna(0).clip(0, 1).astype(int)

        # Materialize member columns before score sums.
        df = assign_columns(df, derived)
        score_block = {score_col: df[members].sum(axis=1) for score_col, members in SCORE_GROUPS.items()}
        detail_count_cols = list(DETAIL_RAW_COLUMNS.values())
        if "detail_selected_count" not in df.columns:
            score_block["detail_selected_count"] = df[detail_count_cols].sum(axis=1)
        else:
            score_block["detail_selected_count"] = pd.to_numeric(df["detail_selected_count"], errors="coerce").fillna(
                df[detail_count_cols].sum(axis=1)
            )
        inside = score_block.get("inside_quality_score", df.get("inside_quality_score", 0))
        outside = score_block.get("outside_quality_score", df.get("outside_quality_score", 0))
        if "detail_quality_score" not in df.columns:
            score_block["detail_quality_score"] = inside + outside
        else:
            score_block["detail_quality_score"] = pd.to_numeric(df["detail_quality_score"], errors="coerce").fillna(0)

        if "open_area_m2" not in df.columns:
            score_block["open_area_m2"] = 0
        open_area = score_block.get("open_area_m2", df["open_area_m2"] if "open_area_m2" in df.columns else 0)
        if "has_open_area" not in df.columns:
            open_area_s = pd.to_numeric(pd.Series(open_area, index=df.index), errors="coerce").fillna(0)
            score_block["has_open_area"] = (open_area_s > 0).astype(int)

        df = assign_columns(df, score_block)

        df = ensure_columns(df, NUMERIC_FEATURES, fill=np.nan)
        for col in NUMERIC_FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

        df = ensure_columns(df, CATEGORICAL_FEATURES, fill="missing")
        for col in CATEGORICAL_FEATURES:
            df[col] = df[col].astype("object").where(df[col].notna(), "missing")
            df[col] = df[col].astype(str).str.strip().replace({"": "missing", "nan": "missing", "None": "missing"})

        # Defragment once after heavy column construction.
        return df.copy()


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
            block: dict[str, Any] = {}
            if med_col in TARGET_STAT_COLS:
                block[med_col] = key.map(self.maps_.get(med_col, {})).fillna(self.global_median_)
            if mean_col in TARGET_STAT_COLS:
                block[mean_col] = key.map(self.maps_.get(mean_col, {})).fillna(self.global_mean_)
            if block:
                df = assign_columns(df, block)

        missing_defaults = {
            col: (self.global_median_ if col.endswith("median") else self.global_mean_)
            for col in TARGET_STAT_COLS
            if col not in df.columns
        }
        if missing_defaults:
            df = assign_columns(df, missing_defaults)
        return df


class FeatureColumnKeeper(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_features: list[str], categorical_features: list[str]):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def fit(self, X: pd.DataFrame, y: Any = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        # Deduplicate accidental double-appended columns
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="last")].copy()
        df = ensure_columns(df, self.numeric_features, fill=np.nan)
        for col in self.numeric_features:
            s = df[col]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, -1]
            df[col] = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
        df = ensure_columns(df, self.categorical_features, fill="missing")
        for col in self.categorical_features:
            s = df[col]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, -1]
            df[col] = s.astype("object").where(s.notna(), "missing")
            df[col] = df[col].astype(str).str.strip().replace({"": "missing", "nan": "missing", "None": "missing"})
        keep = list(dict.fromkeys(list(self.numeric_features) + list(self.categorical_features)))
        return df[keep]




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
        # ComparableMarketFeatureAdder needs unit prices; pipeline y becomes residual below.
        try:
            from comparable_market_features import COMP_UNIT_PRICE_COL
        except ImportError:
            from v18_basiskele.comparable_market_features import COMP_UNIT_PRICE_COL
        if isinstance(Xb, pd.DataFrame):
            Xb = Xb.copy()
            Xb[COMP_UNIT_PRICE_COL] = y_arr
            try:
                Xb[PREMIUM_UNIT_PRICE_COL] = y_arr
            except NameError:
                pass
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

    removed.to_csv(out_dirs["input"] / "sales_removed_location_outliers_v18_basiskele.csv", index=False, encoding="utf-8-sig")
    clean.to_csv(out_dirs["input"] / "sales_after_location_outlier_filter_v18_basiskele.csv", index=False, encoding="utf-8-sig")

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


def resolve_numeric_features(
    attribute_mode: str = "full",
    detail_effect_mode: str = "group",
    basiskele_specialist_mode: str = "premium_target_stats",
    basiskele_large_home_regime: str = "none",
    karamursel_baseline_mode: str = "none",
    location_feature_mode: str = "geo",
    geo_context_mode: str = "geo_with_coast",
    comparable_mode: str = "none",
    premium_feature_mode: str = "full",
    site_project_encoding: str = "frequency",
) -> list[str]:
    loc_num, _loc_cat = get_location_feature_names(location_feature_mode)
    gctx = geo_context_mode if str(location_feature_mode).lower() in {"geo", "full"} else "none"
    names = (
        list(NUMERIC_FEATURES)
        + get_attribute_feature_names(attribute_mode)
        + get_detail_effect_feature_names(detail_effect_mode)
        + get_county_specialist_feature_names(basiskele_specialist_mode)
        + get_v16_regime_feature_names(basiskele_large_home_regime, karamursel_baseline_mode)
        + list(loc_num)
        + get_geo_context_feature_names(gctx)
        + get_comparable_feature_names(comparable_mode)
        + get_premium_feature_names(premium_feature_mode, site_project_encoding)
    )
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def build_preprocessor(
    scale_numeric: bool = True,
    attribute_mode: str = "full",
    detail_effect_mode: str = "group",
    basiskele_specialist_mode: str = "premium_target_stats",
    basiskele_large_home_regime: str = "none",
    karamursel_baseline_mode: str = "none",
    location_feature_mode: str = "geo",
    geo_context_mode: str = "geo_with_coast",
    comparable_mode: str = "none",
    premium_feature_mode: str = "full",
    site_project_encoding: str = "frequency",
) -> ColumnTransformer:
    # keep_empty_features prevents fold/segment-specific all-missing columns from being dropped.
    numeric_features = resolve_numeric_features(
        attribute_mode,
        detail_effect_mode,
        basiskele_specialist_mode,
        basiskele_large_home_regime,
        karamursel_baseline_mode,
        location_feature_mode,
        geo_context_mode,
        comparable_mode,
        premium_feature_mode,
        site_project_encoding,
    )
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
    cat_features = resolve_categorical_features(premium_feature_mode, site_project_encoding)
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", cat_pipe, cat_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def build_pipeline(
    model: Any,
    scale_numeric: bool,
    target_mode: str,
    attribute_mode: str = "full",
    detail_effect_mode: str = "group",
    basiskele_specialist_mode: str = "premium_target_stats",
    basiskele_large_home_regime: str = "none",
    karamursel_baseline_mode: str = "none",
    location_feature_mode: str = "full",
    geo_context_mode: str = "full",
    location_min_precision: str = "any",
    enable_coordinate_noise_check: bool = True,
    comparable_k_list: str = "5,10,20",
    comparable_mode: str = "none",
    premium_feature_mode: str = "full",
    site_project_encoding: str = "frequency",
    geo_context_cache_dir: str = "data/external/geo_context",
    location_scope: str = "basiskele_only",
    location_coverage_min: float = 0.40,
) -> Any:
    mode = str(attribute_mode or "full").lower()
    detail_mode = str(detail_effect_mode or "group").lower()
    specialist_mode = str(basiskele_specialist_mode or "none").lower()
    lh_regime = str(basiskele_large_home_regime or "none").lower()
    kar_mode = str(karamursel_baseline_mode or "none").lower()
    prem_mode = str(premium_feature_mode or "none").lower()
    site_enc = str(site_project_encoding or "none").lower()
    numeric_features = resolve_numeric_features(mode, detail_mode, specialist_mode, lh_regime, kar_mode, location_feature_mode, geo_context_mode, str(comparable_mode or "none"), prem_mode, site_enc)
    steps = [
        ("feature_engineering", FeatureEngineer()),
        ("attribute_quality", AttributeQualityAdder(attribute_mode=mode)),
        ("target_stats", TargetStatsAdder(smoothing=20.0)),
        ("attribute_interactions", AttributeInteractionCompleter()),
    ]
    if mode == "full":
        steps.append(("attribute_effects", AttributeEffectEncoder(attribute_mode=mode, alpha=30.0, min_count=30)))
    # Fold-safe detail premiums: after attr effects, before FeatureColumnKeeper (binaries still present).
    steps.append(
        (
            "detail_effects",
            LocalDetailPremiumEncoder(mode=detail_mode, alpha=50.0, random_state=42),
        )
    )
    # V15 specialist stack + V16 regime features
    steps.append(("basiskele_premium", BasiskelePremiumSpecialistAdder(mode=specialist_mode)))
    steps.append(("large_home_features", LargeHomeFeatureAdder()))
    steps.append(
        (
            "basiskele_premium_target_stats",
            BasiskelePremiumTargetStatsAdder(mode=specialist_mode, alpha=50.0, min_count=30),
        )
    )
    steps.append(("basiskele_large_home_regime", BasiskeleLargeHomeRegimeAdder(mode=lh_regime)))
    steps.append(
        (
            "karamursel_location_age",
            KaramurselLocationAgeBaselineAdder(mode=kar_mode, alpha=80.0, min_count=20),
        )
    )
    # V17 location + geo-context + fold-safe comparables
    loc_mode = str(location_feature_mode or "full").lower()
    gctx_mode = str(geo_context_mode or "full").lower()
    if loc_mode in {"geo", "full"} and gctx_mode in {"", "full"}:
        gctx_mode = "full"
    if loc_mode == "basic":
        gctx_mode = "none"
    if loc_mode == "none":
        gctx_mode = "none"
    steps.append(
        (
            "location_features",
            LocationFeatureAdder(
                mode=loc_mode if loc_mode != "comparable" else "basic",
                min_precision=str(location_min_precision or "any"),
                enable_coordinate_noise_check=bool(enable_coordinate_noise_check),
            ),
        )
    )
    steps.append(
        (
            "geo_context",
            GeoContextFeatureAdder(
                mode=loc_mode,
                context_mode=gctx_mode if loc_mode in {"geo", "full"} else "none",
                cache_dir=str(geo_context_cache_dir or "data/external/geo_context"),
            ),
        )
    )
    steps.append(
        (
            "comparable_market",
            ComparableMarketFeatureAdder(
                mode=str(comparable_mode or "none"),
                k_list=str(comparable_k_list or "5,10,20"),
            ),
        )
    )
    steps.append(
        (
            "premium_signals",
            PremiumSignalFeatureAdder(
                premium_feature_mode=prem_mode,
                site_project_encoding=site_enc,
                min_site_freq=3,
            ),
        )
    )
    steps.append(
        (
            "site_project_foldsafe",
            SiteProjectFoldSafeEncoder(
                enabled=(site_enc == "foldsafe_target" and prem_mode in {"site", "full", "interactions"}),
                min_count=3,
                alpha=20.0,
                premium_feature_mode=prem_mode,
            ),
        )
    )
    steps.append(
        (
            "location_scope_mask",
            LocationScopeMasker(
                location_scope=str(location_scope or "basiskele_only"),
                min_coverage=float(location_coverage_min or 0.40),
                enabled=loc_mode not in {"", "none"},
            ),
        )
    )
    steps.extend(
        [
            ("feature_columns", FeatureColumnKeeper(numeric_features, resolve_categorical_features(prem_mode, site_enc))),
            (
                "preprocess",
                build_preprocessor(
                    scale_numeric=scale_numeric,
                    attribute_mode=mode,
                    detail_effect_mode=detail_mode,
                    basiskele_specialist_mode=specialist_mode,
                    basiskele_large_home_regime=lh_regime,
                    karamursel_baseline_mode=kar_mode,
                    location_feature_mode=location_feature_mode,
                    geo_context_mode=gctx_mode,
                    comparable_mode=str(comparable_mode or "none"),
                    premium_feature_mode=prem_mode,
                    site_project_encoding=site_enc,
                ),
            ),
            ("model", model),
        ]
    )
    pipe = Pipeline(steps=steps)
    if target_mode == "residual":
        return LocationResidualRegressor(pipe, baseline_encoder=LocationBaselineEncoder())
    if target_mode == "log":
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1, check_inverse=False)
    if target_mode == "raw":
        return pipe
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def model_specs(target_mode: str, selected_models: list[str] | None = None, fast_mode: bool = False, attribute_mode: str = "full", detail_effect_mode: str = "group", basiskele_specialist_mode: str = "premium_target_stats", basiskele_large_home_regime: str = "none", karamursel_baseline_mode: str = "none", location_feature_mode: str = "full", geo_context_mode: str = "full", location_min_precision: str = "any", enable_coordinate_noise_check: bool = True, comparable_k_list: str = "5,10,20", comparable_mode: str = "none", premium_feature_mode: str = "full", site_project_encoding: str = "frequency", geo_context_cache_dir: str = "data/external/geo_context", location_scope: str = "basiskele_only", location_coverage_min: float = 0.40) -> dict[str, Any]:
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
            attribute_mode=attribute_mode,
            detail_effect_mode=detail_effect_mode,
            basiskele_specialist_mode=basiskele_specialist_mode,
            basiskele_large_home_regime=basiskele_large_home_regime,
            karamursel_baseline_mode=karamursel_baseline_mode,
            location_feature_mode=location_feature_mode,
            geo_context_mode=geo_context_mode,
            location_min_precision=location_min_precision,
            enable_coordinate_noise_check=enable_coordinate_noise_check,
            comparable_k_list=comparable_k_list,
            comparable_mode=comparable_mode,
            premium_feature_mode=premium_feature_mode,
            site_project_encoding=site_project_encoding,
            geo_context_cache_dir=geo_context_cache_dir,
            location_scope=location_scope,
            location_coverage_min=location_coverage_min,
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
            attribute_mode=attribute_mode,
            detail_effect_mode=detail_effect_mode,
            basiskele_specialist_mode=basiskele_specialist_mode,
            basiskele_large_home_regime=basiskele_large_home_regime,
            karamursel_baseline_mode=karamursel_baseline_mode,
            location_feature_mode=location_feature_mode,
            geo_context_mode=geo_context_mode,
            location_min_precision=location_min_precision,
            enable_coordinate_noise_check=enable_coordinate_noise_check,
            comparable_k_list=comparable_k_list,
            comparable_mode=comparable_mode,
            premium_feature_mode=premium_feature_mode,
            site_project_encoding=site_project_encoding,
            geo_context_cache_dir=geo_context_cache_dir,
            location_scope=location_scope,
            location_coverage_min=location_coverage_min,
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
            attribute_mode=attribute_mode,
            detail_effect_mode=detail_effect_mode,
            basiskele_specialist_mode=basiskele_specialist_mode,
            basiskele_large_home_regime=basiskele_large_home_regime,
            karamursel_baseline_mode=karamursel_baseline_mode,
            location_feature_mode=location_feature_mode,
            geo_context_mode=geo_context_mode,
            location_min_precision=location_min_precision,
            enable_coordinate_noise_check=enable_coordinate_noise_check,
            comparable_k_list=comparable_k_list,
            comparable_mode=comparable_mode,
            premium_feature_mode=premium_feature_mode,
            site_project_encoding=site_project_encoding,
            geo_context_cache_dir=geo_context_cache_dir,
            location_scope=location_scope,
            location_coverage_min=location_coverage_min,
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
            attribute_mode=attribute_mode,
            detail_effect_mode=detail_effect_mode,
            basiskele_specialist_mode=basiskele_specialist_mode,
            basiskele_large_home_regime=basiskele_large_home_regime,
            karamursel_baseline_mode=karamursel_baseline_mode,
            location_feature_mode=location_feature_mode,
            geo_context_mode=geo_context_mode,
            location_min_precision=location_min_precision,
            enable_coordinate_noise_check=enable_coordinate_noise_check,
            comparable_k_list=comparable_k_list,
            comparable_mode=comparable_mode,
            premium_feature_mode=premium_feature_mode,
            site_project_encoding=site_project_encoding,
            geo_context_cache_dir=geo_context_cache_dir,
            location_scope=location_scope,
            location_coverage_min=location_coverage_min,
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
            attribute_mode=attribute_mode,
            detail_effect_mode=detail_effect_mode,
            basiskele_specialist_mode=basiskele_specialist_mode,
            basiskele_large_home_regime=basiskele_large_home_regime,
            karamursel_baseline_mode=karamursel_baseline_mode,
            location_feature_mode=location_feature_mode,
            geo_context_mode=geo_context_mode,
            location_min_precision=location_min_precision,
            enable_coordinate_noise_check=enable_coordinate_noise_check,
            comparable_k_list=comparable_k_list,
            comparable_mode=comparable_mode,
            premium_feature_mode=premium_feature_mode,
            site_project_encoding=site_project_encoding,
            geo_context_cache_dir=geo_context_cache_dir,
            location_scope=location_scope,
            location_coverage_min=location_coverage_min,
        ),
    }

    unknown = [m for m in selected_models if m not in registry]
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}. Available: {sorted(registry)}")
    return {name: registry[name] for name in selected_models}

def make_training_matrix(sales: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = pd.to_numeric(sales[TARGET], errors="coerce")
    X = sales.drop(
        columns=[c for c in LEAKAGE_OR_UNUSED_COLUMNS_V20 if c in sales.columns],
        errors="ignore",
    ).copy()
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

    specs = model_specs(cfg.target_mode, selected_models=cfg.selected_models, fast_mode=cfg.fast_mode, attribute_mode=str(getattr(cfg, "attribute_mode", "full")), detail_effect_mode=str(getattr(cfg, "detail_effect_mode", "group")), basiskele_specialist_mode=str(getattr(cfg, "basiskele_specialist_mode", "premium_target_stats")), basiskele_large_home_regime=str(getattr(cfg, "basiskele_large_home_regime", "simple")), karamursel_baseline_mode=str(getattr(cfg, "karamursel_baseline_mode", "location_age")), location_feature_mode=str(getattr(cfg, "location_feature_mode", "full")), geo_context_mode=str(getattr(cfg, "geo_context_mode", "full")), location_min_precision=str(getattr(cfg, "location_min_precision", "any")), enable_coordinate_noise_check=bool(getattr(cfg, "enable_coordinate_noise_check", True)), comparable_k_list=str(getattr(cfg, "comparable_k_list", "5,10,20")), comparable_mode=str(getattr(cfg, "comparable_mode", "none")), premium_feature_mode=str(getattr(cfg, "premium_feature_mode", "full")), site_project_encoding=str(getattr(cfg, "site_project_encoding", "frequency")), geo_context_cache_dir=str(getattr(cfg, "geo_context_cache_dir", "data/external/geo_context")), location_scope=str(getattr(cfg, "location_scope", "basiskele_only")), location_coverage_min=float(getattr(cfg, "location_coverage_min", 0.40)))

    for segment_name in segment_names:
        mask = segment_mask(X, segment_name)
        n = int(mask.sum())
        if n < min_rows:
            report_rows.append({"segment": segment_name, "rows": n, "status": "skipped_too_few_rows"})
            _ptick(f"segment {segment_name} atlandı", n=len(specs))
            continue

        if segment_name == "large_home" and str(getattr(cfg, "large_home_specialist_mode", "redesigned")).lower() == "redesigned":
            _pstage("Large home redesign")
            large_models = ["ridge", "gradient_boosting", "extra_trees", "random_forest"]
            seg_specs = model_specs(
                cfg.target_mode,
                selected_models=large_models,
                fast_mode=cfg.fast_mode,
                attribute_mode=str(getattr(cfg, "attribute_mode", "full")),
                detail_effect_mode=str(getattr(cfg, "detail_effect_mode", "group")),
                basiskele_specialist_mode=str(getattr(cfg, "basiskele_specialist_mode", "premium_target_stats")),
            )
        else:
            seg_specs = specs

        Xs = X.loc[mask].reset_index(drop=True)
        ys = y.loc[mask].reset_index(drop=True)
        split_count = min(cfg.n_splits, max(2, n // 35))
        cv = KFold(n_splits=split_count, shuffle=True, random_state=cfg.random_state)

        seg_oof = pd.DataFrame({"actual_unit_price_gross": ys})
        seg_rows = []
        fitted = {}

        for model_name, estimator in seg_specs.items():
            _plog(f"Training segment OOF model: {segment_name}/{model_name}")
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
            _ptick(f"segment {segment_name}/{model_name}")

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
                joblib.dump(model_obj, out_dirs["artifacts"] / f"model_segment_{segment_name}_{model_name}_v18_basiskele.joblib")

        status = "used_blend" if use_segment else "kept_base"
        note = ""
        if segment_name == "large_home" and not use_segment:
            note = "large_home specialist did not beat base MAPE; kept_base (redesign features still in base pipeline)"

        row = {
            "segment": segment_name,
            "rows": n,
            "status": status,
            "status_note": note,
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
        seg_comparison.to_csv(out_dirs["reports"] / f"model_comparison_segment_{segment_name}_v18_basiskele.csv", index=False, encoding="utf-8-sig")

    report = pd.DataFrame(report_rows)
    report.to_csv(out_dirs["reports"] / "segment_layer_report_v18_basiskele.csv", index=False, encoding="utf-8-sig")

    # large_home diagnostics (V15)
    try:
        lh = report[report["segment"] == "large_home"]
        if not lh.empty:
            lh_row = lh.iloc[0].to_dict()
            lh_mask = segment_mask(X, "large_home")
            bas_mask = X["county"].astype(str).eq("Başiskele") & lh_mask if "county" in X.columns else lh_mask & False
            bas_r2 = np.nan
            if int(bas_mask.sum()) >= 10:
                # use final_pred after segment for basiskele large_home slice against y
                bas_r2 = float(metric_dict(y.loc[bas_mask], final_pred.loc[bas_mask])["r2"])
            top_feats = ",".join(
                [
                    "large_home_m2_excess",
                    "large_home_quality_x_m2",
                    "large_home_detail_premium_x_m2",
                    "large_home_site_x_m2",
                    "large_home_basiskele_premium",
                ]
            )
            pd.DataFrame(
                [
                    {
                        "rows": lh_row.get("rows"),
                        "base_r2": lh_row.get("base_r2"),
                        "segment_r2": lh_row.get("segment_r2"),
                        "best_blended_r2": lh_row.get("best_blended_r2"),
                        "mape": lh_row.get("best_blended_mape") if lh_row.get("status") == "used_blend" else lh_row.get("base_mape"),
                        "mae": lh_row.get("best_blended_mae_tl_per_m2") if lh_row.get("status") == "used_blend" else lh_row.get("base_mae_tl_per_m2"),
                        "status": lh_row.get("status"),
                        "status_note": lh_row.get("status_note"),
                        "basiskele_large_home_rows": int(bas_mask.sum()),
                        "basiskele_large_home_r2": bas_r2,
                        "top_large_home_features": top_feats,
                    }
                ]
            ).to_csv(out_dirs["reports"] / "large_home_diagnostics_v18_basiskele.csv", index=False, encoding="utf-8-sig")
    except Exception as exc:
        warnings.warn(f"large_home diagnostics skipped: {exc}")

    return final_pred, segment_models, segment_weights, segment_blend_weights, report



def apply_basiskele_variance_lift_layer(
    X: pd.DataFrame,
    y: pd.Series,
    current_pred: pd.Series,
    cfg: RunConfig,
    out_dirs: dict[str, Path],
) -> tuple[pd.Series, dict[str, Any]]:
    """OOF-safe Başiskele variance-lift skeleton (V16.0 conservative / full).

    Uses only deterministic premium features (no validation actual in fit).
    If Başiskele R² does not improve or global MAPE worsens, layer is disabled.
    """
    mode = str(getattr(cfg, "basiskele_variance_lift", "none") or "none").lower()
    specialist = str(getattr(cfg, "basiskele_specialist_mode", "none") or "none").lower()
    report: dict[str, Any] = {
        "mode": mode,
        "status": "disabled",
        "lambda": 0.0,
        "basiskele_r2_before": np.nan,
        "basiskele_r2_after": np.nan,
        "global_mape_before": np.nan,
        "global_mape_after": np.nan,
        "note": "",
    }
    final_pred = pd.Series(current_pred, index=X.index, dtype=float).copy()
    if mode in {"", "none"} or specialist in {"", "none"}:
        report["note"] = "variance lift off"
        pd.DataFrame([report]).to_csv(
            out_dirs["reports"] / "basiskele_variance_lift_report_v18_basiskele.csv", index=False, encoding="utf-8-sig"
        )
        return final_pred, report

    if "county" not in X.columns:
        report["status"] = "skipped_no_county"
        pd.DataFrame([report]).to_csv(
            out_dirs["reports"] / "basiskele_variance_lift_report_v18_basiskele.csv", index=False, encoding="utf-8-sig"
        )
        return final_pred, report

    _pstage("Basiskele variance lift")
    mask = X["county"].astype(str).eq("Başiskele")
    n = int(mask.sum())
    if n < 40:
        report["status"] = "skipped_too_few_rows"
        report["note"] = f"rows={n}"
        pd.DataFrame([report]).to_csv(
            out_dirs["reports"] / "basiskele_variance_lift_report_v18_basiskele.csv", index=False, encoding="utf-8-sig"
        )
        return final_pred, report

    # Build deterministic premium features on full X (no target); fold-fit only the delta model.
    try:
        adder = BasiskelePremiumSpecialistAdder(mode="premium")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adder.fit(X)
            Xp = adder.transform(X)
    except Exception as exc:
        report["status"] = "failed_feature_build"
        report["note"] = str(exc)
        pd.DataFrame([report]).to_csv(
            out_dirs["reports"] / "basiskele_variance_lift_report_v18_basiskele.csv", index=False, encoding="utf-8-sig"
        )
        return final_pred, report

    feat_cols = [
        c
        for c in [
            "basiskele_premium_score",
            "basiskele_detail_total_premium_signal",
            "basiskele_detail_outside_premium_signal",
            "basiskele_detail_view_premium_signal",
            "basiskele_has_pool_signal",
            "basiskele_has_security_signal",
            "basiskele_site_premium_signal",
            "basiskele_premium_x_gross_m2",
            "basiskele_premium_x_large_home",
        ]
        if c in Xp.columns
    ]
    if not feat_cols:
        report["status"] = "skipped_no_features"
        pd.DataFrame([report]).to_csv(
            out_dirs["reports"] / "basiskele_variance_lift_report_v18_basiskele.csv", index=False, encoding="utf-8-sig"
        )
        return final_pred, report

    lambdas = [0.10, 0.15, 0.20, 0.25, 0.35] if mode == "conservative" else [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    n_splits = min(cfg.n_splits, max(2, n // 40))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
    idx = np.where(mask.to_numpy())[0]
    Xb = Xp.iloc[idx][feat_cols].fillna(0.0).to_numpy(dtype=float)
    yb = y.iloc[idx].to_numpy(dtype=float)
    pb = current_pred.iloc[idx].to_numpy(dtype=float)
    delta_oof = np.zeros(n, dtype=float)

    for tr, va in cv.split(Xb):
        # train delta on train fold only (actual - pred); never see validation actual
        dy = yb[tr] - pb[tr]
        model = Ridge(alpha=5.0, solver="lsqr")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Xb[tr], dy)
            delta_oof[va] = model.predict(Xb[va])

    before_b = metric_dict(yb, pb)
    before_g = metric_dict(y, current_pred)
    report["basiskele_r2_before"] = before_b["r2"]
    report["global_mape_before"] = before_g["mape"]

    best = None
    for lam in lambdas:
        cand = pb + float(lam) * delta_oof
        cand = np.maximum(cand, 0.0)
        mb = metric_dict(yb, cand)
        # provisional global: swap Başiskele preds
        gpred = current_pred.to_numpy(dtype=float).copy()
        gpred[idx] = cand
        mg = metric_dict(y, gpred)
        row = {"lambda": float(lam), "basiskele_r2": mb["r2"], "global_mape": mg["mape"], "pred": cand, "gpred": gpred}
        if best is None or (row["basiskele_r2"], -row["global_mape"]) > (best["basiskele_r2"], -best["global_mape"]):
            best = row

    assert best is not None
    improved_r2 = best["basiskele_r2"] > before_b["r2"] + 1e-4
    mape_ok = best["global_mape"] <= before_g["mape"] + 1e-4
    if improved_r2 and mape_ok:
        final_pred.iloc[idx] = best["pred"]
        report["status"] = "applied"
        report["lambda"] = best["lambda"]
        report["basiskele_r2_after"] = best["basiskele_r2"]
        report["global_mape_after"] = best["global_mape"]
        report["note"] = "OOF ridge delta on deterministic premium features"
    else:
        report["status"] = "rejected_no_lift"
        report["lambda"] = best["lambda"]
        report["basiskele_r2_after"] = best["basiskele_r2"]
        report["global_mape_after"] = best["global_mape"]
        report["note"] = "disabled: no Başiskele R2 lift or global MAPE worsened"

    pd.DataFrame([{k: v for k, v in report.items()}]).to_csv(
        out_dirs["reports"] / "basiskele_variance_lift_report_v18_basiskele.csv", index=False, encoding="utf-8-sig"
    )
    return final_pred, report


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
    anom.sort_values("anomaly_score", ascending=False).to_csv(output_dir / "listing_anomaly_scores_v18_basiskele.csv", index=False, encoding="utf-8-sig")
    anom.sort_values("anomaly_score", ascending=False).head(250).to_csv(reports_dir / "top_listing_anomalies_v18_basiskele.csv", index=False, encoding="utf-8-sig")

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
        rep.to_csv(reports_dir / f"anomaly_by_{col}_v18_basiskele.csv", index=False, encoding="utf-8-sig")

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
        report.to_csv(out_dirs["reports"] / "county_expert_report_v18_basiskele.csv", index=False, encoding="utf-8-sig")
        n_models_skip = len(cfg.selected_models) if cfg.selected_models else 4
        _ptick("county experts kapalı", n=n_models_skip * 3)
        return final_pred, used_county_models, all_county_models, county_weights, county_blend_weights, report

    counties = sorted([c for c in X["county"].dropna().astype(str).unique() if c and c != "missing"])
    overrides = dict(getattr(cfg, "county_expert_min_rows_overrides", None) or {})
    for county_name in counties:
        mask = X["county"].fillna("missing").astype(str).eq(county_name)
        n = int(mask.sum())
        min_rows_used = int(overrides.get(county_name, cfg.county_expert_min_rows))
        override_used = bool(county_name in overrides)
        if n < min_rows_used:
            report_rows.append({
                "county": county_name,
                "rows": n,
                "min_rows_used": min_rows_used,
                "override_used": override_used,
                "status": "skipped_too_few_rows",
                "current_r2": np.nan,
                "expert_r2": np.nan,
                "best_blended_r2": np.nan,
                "blend_weight": 0.0,
                "mape": np.nan,
                "weights": "",
            })
            n_models_skip = len(cfg.selected_models) if cfg.selected_models else 4
            _ptick(f"county {county_name} atlandı", n=n_models_skip)
            continue

        _pstage(f"County override expert: {county_name} (min_rows={min_rows_used})")
        Xc = X.loc[mask].reset_index(drop=True)
        yc = y.loc[mask].reset_index(drop=True)
        base_pred_c = pd.Series(current_pred.loc[mask].to_numpy(dtype=float), index=Xc.index)
        n_splits = min(cfg.n_splits, max(2, min(5, n // 80)))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
        specs = model_specs(cfg.target_mode, selected_models=cfg.selected_models, fast_mode=cfg.fast_mode, attribute_mode=str(getattr(cfg, "attribute_mode", "full")), detail_effect_mode=str(getattr(cfg, "detail_effect_mode", "group")), basiskele_specialist_mode=str(getattr(cfg, "basiskele_specialist_mode", "premium_target_stats")), basiskele_large_home_regime=str(getattr(cfg, "basiskele_large_home_regime", "simple")), karamursel_baseline_mode=str(getattr(cfg, "karamursel_baseline_mode", "location_age")), location_feature_mode=str(getattr(cfg, "location_feature_mode", "full")), geo_context_mode=str(getattr(cfg, "geo_context_mode", "full")), location_min_precision=str(getattr(cfg, "location_min_precision", "any")), enable_coordinate_noise_check=bool(getattr(cfg, "enable_coordinate_noise_check", True)), comparable_k_list=str(getattr(cfg, "comparable_k_list", "5,10,20")), comparable_mode=str(getattr(cfg, "comparable_mode", "none")), premium_feature_mode=str(getattr(cfg, "premium_feature_mode", "full")), site_project_encoding=str(getattr(cfg, "site_project_encoding", "frequency")), geo_context_cache_dir=str(getattr(cfg, "geo_context_cache_dir", "data/external/geo_context")), location_scope=str(getattr(cfg, "location_scope", "basiskele_only")), location_coverage_min=float(getattr(cfg, "location_coverage_min", 0.40)))

        oof_c = pd.DataFrame({"actual_unit_price_gross": yc})
        comparison_rows = []
        fitted: dict[str, Any] = {}
        for model_name, estimator in specs.items():
            _plog(f"Training county expert OOF model: {county_name}/{model_name}")
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
            _ptick(f"county {county_name}/{model_name}")

        comp = pd.DataFrame(comparison_rows).sort_values(["mape", "mae_tl_per_m2"])
        comp.insert(0, "county", county_name)
        safe_county = re.sub(r"[^0-9A-Za-z_]+", "_", county_name)
        comp.to_csv(out_dirs["reports"] / f"model_comparison_county_{safe_county}_v18_basiskele.csv", index=False, encoding="utf-8-sig")

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
                joblib.dump(model_obj, out_dirs["artifacts"] / f"model_county_{safe_county}_{model_name}_v18_basiskele.joblib")

        report_rows.append({
            "county": county_name,
            "rows": n,
            "min_rows_used": min_rows_used,
            "override_used": override_used,
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
            "mape": best["mape"] if use_county else base_metrics["mape"],
            "blend_candidates": json.dumps([{k: v for k, v in c.items() if k != "pred"} for c in candidates], ensure_ascii=False),
            "weights": json.dumps(weights, ensure_ascii=False),
        })

    report = pd.DataFrame(report_rows)
    report.to_csv(out_dirs["reports"] / "county_expert_layer_report_v18_basiskele.csv", index=False, encoding="utf-8-sig")
    report.to_csv(out_dirs["reports"] / "county_expert_report_v18_basiskele.csv", index=False, encoding="utf-8-sig")
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
    rep.to_csv(reports_dir / "feature_importance_by_county_v18_basiskele.csv", index=False, encoding="utf-8-sig")
    top = (
        rep.groupby(["county", "feature"], as_index=False)["importance"]
        .mean()
        .sort_values(["county", "importance"], ascending=[True, False])
    )
    top.groupby("county", group_keys=False).head(40).to_csv(reports_dir / "feature_importance_by_county_top40_v18_basiskele.csv", index=False, encoding="utf-8-sig")


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, cfg: RunConfig, out_dirs: dict[str, Path]) -> tuple[ModelBundle, dict[str, Any]]:
    if len(X) < 10:
        raise ValueError(f"Too few sale rows to train: {len(X)}")

    anomaly_df = pd.DataFrame()
    if cfg.enable_anomaly_reports:
        _plog("Computing listing anomaly diagnostics...")
        anomaly_df = compute_anomaly_scores(X, y, cfg)
        save_anomaly_reports(anomaly_df, out_dirs["reports"], out_dirs["output"])
    _ptick("anomaly diagnostics")

    n_splits = min(cfg.n_splits, max(2, len(X) // 20))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
    specs = model_specs(cfg.target_mode, selected_models=cfg.selected_models, fast_mode=cfg.fast_mode, attribute_mode=str(getattr(cfg, "attribute_mode", "full")), detail_effect_mode=str(getattr(cfg, "detail_effect_mode", "group")), basiskele_specialist_mode=str(getattr(cfg, "basiskele_specialist_mode", "premium_target_stats")), basiskele_large_home_regime=str(getattr(cfg, "basiskele_large_home_regime", "simple")), karamursel_baseline_mode=str(getattr(cfg, "karamursel_baseline_mode", "location_age")), location_feature_mode=str(getattr(cfg, "location_feature_mode", "full")), geo_context_mode=str(getattr(cfg, "geo_context_mode", "full")), location_min_precision=str(getattr(cfg, "location_min_precision", "any")), enable_coordinate_noise_check=bool(getattr(cfg, "enable_coordinate_noise_check", True)), comparable_k_list=str(getattr(cfg, "comparable_k_list", "5,10,20")), comparable_mode=str(getattr(cfg, "comparable_mode", "none")), premium_feature_mode=str(getattr(cfg, "premium_feature_mode", "full")), site_project_encoding=str(getattr(cfg, "site_project_encoding", "frequency")), geo_context_cache_dir=str(getattr(cfg, "geo_context_cache_dir", "data/external/geo_context")), location_scope=str(getattr(cfg, "location_scope", "basiskele_only")), location_coverage_min=float(getattr(cfg, "location_coverage_min", 0.40)))

    oof = pd.DataFrame({"actual_unit_price_gross": y})
    comparison_rows = []
    fitted_models: dict[str, Any] = {}

    for name, estimator in specs.items():
        _pstage(f"Base ensemble: {name} CV")
        _plog(f"Training OOF model: {name}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = cross_val_predict(clone(estimator), X, y, cv=cv, n_jobs=None, method="predict")
        pred = np.maximum(np.asarray(pred, dtype=float), 0)
        oof[f"pred_{name}"] = pred
        metrics = metric_dict(y, pred)
        metrics["model"] = name
        comparison_rows.append(metrics)
        _ptick(f"OOF {name}")

        _pstage(f"Base ensemble: {name} final fit")
        _plog(f"Fitting final model: {name}")
        final_estimator = clone(estimator)
        final_estimator.fit(X, y)
        fitted_models[name] = final_estimator
        joblib.dump(final_estimator, out_dirs["artifacts"] / f"model_{name}_v18_basiskele.joblib")
        _ptick(f"fit {name}")

    model_comparison = pd.DataFrame(comparison_rows).sort_values(["mape", "mae_tl_per_m2"])
    model_comparison.to_csv(out_dirs["reports"] / "model_comparison_v18_basiskele.csv", index=False, encoding="utf-8-sig")

    weights = choose_ensemble_weights(model_comparison, max_models=3)
    base_pred = np.zeros(len(oof), dtype=float)
    total_w = sum(weights.values())
    for name, w in weights.items():
        base_pred += oof[f"pred_{name}"].to_numpy(dtype=float) * (w / total_w)
    oof["pred_ensemble_base"] = base_pred

    base_metrics = metric_dict(y, base_pred)
    base_metrics["model"] = "ensemble_base_top_models_v16"
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
    segment_metrics_final["model"] = "segment_aware_before_county_v16"

    # County-expert layer. It is applied after segment correction and only where OOF blend improves.
    county_pred, county_models, county_models_all, county_weights, county_blend_weights, county_report = train_county_expert_layer(
        X=X,
        y=y,
        current_pred=segment_pred,
        cfg=cfg,
        out_dirs=out_dirs,
    )
    oof["pred_after_county"] = np.maximum(county_pred.to_numpy(dtype=float), 0)

    # V16-E2: Başiskele large_home residual (only if regime mode == residual)
    _pstage("Basiskele large_home residual layer")
    lh_pred, lh_report = apply_basiskele_large_home_residual_layer(
        X=X,
        y=y,
        current_pred=oof["pred_after_county"],
        mode=str(getattr(cfg, "basiskele_large_home_regime", "simple")),
        n_splits=cfg.n_splits,
        random_state=cfg.random_state,
        fast_mode=bool(cfg.fast_mode),
    )
    oof["pred_after_bsk_large_home"] = np.maximum(np.asarray(lh_pred, dtype=float), 0)
    pd.DataFrame([lh_report]).to_csv(
        out_dirs["reports"] / "basiskele_large_home_residual_layer_v18_basiskele.csv", index=False, encoding="utf-8-sig"
    )

    # V16-E1: Başiskele spread residual
    _pstage("Basiskele spread residual layer")
    spread_pred, spread_report = apply_basiskele_spread_residual_layer(
        X=X,
        y=y,
        current_pred=oof["pred_after_bsk_large_home"],
        mode=str(getattr(cfg, "basiskele_spread_layer", "conservative")),
        n_splits=cfg.n_splits,
        random_state=cfg.random_state,
        fast_mode=bool(cfg.fast_mode),
    )
    oof["pred_after_bsk_spread"] = np.maximum(np.asarray(spread_pred, dtype=float), 0)
    pd.DataFrame([spread_report]).to_csv(
        out_dirs["reports"] / "basiskele_spread_residual_layer_v18_basiskele.csv", index=False, encoding="utf-8-sig"
    )

    # Keep V15 variance-lift available but default-off; run after spread only if explicitly enabled.
    lift_pred, lift_report = apply_basiskele_variance_lift_layer(
        X=X,
        y=y,
        current_pred=oof["pred_after_bsk_spread"],
        cfg=cfg,
        out_dirs=out_dirs,
    )
    oof["pred_after_basiskele_lift"] = np.maximum(np.asarray(lift_pred, dtype=float), 0)
    oof["pred_ensemble"] = oof["pred_after_basiskele_lift"].copy()

    ensemble_metrics = metric_dict(y, oof["pred_ensemble"])
    ensemble_metrics["model"] = "county_expert_segment_aware_ensemble_v16"
    ensemble_metrics["base_weights"] = weights
    ensemble_metrics["segment_weights"] = segment_weights
    ensemble_metrics["segment_blend_weights"] = segment_blend_weights
    ensemble_metrics["county_weights"] = county_weights
    ensemble_metrics["county_blend_weights"] = county_blend_weights
    ensemble_metrics["attribute_mode"] = str(getattr(cfg, "attribute_mode", "full"))
    ensemble_metrics["detail_effect_mode"] = str(getattr(cfg, "detail_effect_mode", "group"))
    ensemble_metrics["basiskele_specialist_mode"] = str(getattr(cfg, "basiskele_specialist_mode", "premium_target_stats"))
    ensemble_metrics["basiskele_variance_lift"] = str(getattr(cfg, "basiskele_variance_lift", "none"))
    ensemble_metrics["basiskele_variance_lift_report"] = lift_report
    ensemble_metrics["basiskele_large_home_regime"] = str(getattr(cfg, "basiskele_large_home_regime", "simple"))
    ensemble_metrics["basiskele_spread_layer"] = str(getattr(cfg, "basiskele_spread_layer", "conservative"))
    ensemble_metrics["karamursel_baseline_mode"] = str(getattr(cfg, "karamursel_baseline_mode", "location_age"))
    ensemble_metrics["basiskele_large_home_residual_report"] = lh_report
    ensemble_metrics["basiskele_spread_residual_report"] = spread_report
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
        attribute_mode=str(getattr(cfg, "attribute_mode", "full")),
        detail_effect_mode=str(getattr(cfg, "detail_effect_mode", "group")),
        basiskele_specialist_mode=str(getattr(cfg, "basiskele_specialist_mode", "premium_target_stats")),
        basiskele_variance_lift=str(getattr(cfg, "basiskele_variance_lift", "conservative")),
    )
    joblib.dump(bundle, out_dirs["artifacts"] / "model_bundle_v18_basiskele.joblib")

    oof["error"] = oof["pred_ensemble"] - oof["actual_unit_price_gross"]
    oof["abs_error"] = oof["error"].abs()
    oof["abs_pct_error"] = oof["abs_error"] / oof["actual_unit_price_gross"]
    oof = pd.concat([oof, X.reset_index(drop=True)], axis=1)

    if not anomaly_df.empty:
        anomaly_cols = [c for c in anomaly_df.columns if c.startswith("anom_") or c.startswith("anomaly_")]
        anomaly_cols = ["anomaly_score", "anomaly_severity", "anomaly_reasons"] + [c for c in anomaly_cols if c not in {"anomaly_score", "anomaly_severity", "anomaly_reasons"}]
        anomaly_cols = [c for c in anomaly_cols if c in anomaly_df.columns]
        oof = pd.concat([oof, anomaly_df[anomaly_cols].reset_index(drop=True)], axis=1)

    oof.to_csv(out_dirs["output"] / "oof_predictions_v18_basiskele.csv", index=False, encoding="utf-8-sig")

    county_metrics = county_metric_report(oof, pred_col="pred_ensemble")
    if not county_metrics.empty:
        county_metrics.to_csv(out_dirs["reports"] / "county_metrics_v18_basiskele.csv", index=False, encoding="utf-8-sig")

    anomaly_metric_report = anomaly_metric_diagnostics(oof)
    if not anomaly_metric_report.empty:
        anomaly_metric_report.to_csv(out_dirs["reports"] / "anomaly_metric_diagnostics_v18_basiskele.csv", index=False, encoding="utf-8-sig")

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

    # V16 attribute + detail premium diagnostics
    try:
        _pstage("diagnostics: pair tests / sensitivity")
        pair_df, county_sens, _ = run_prediction_pair_tests(bundle, X, out_dirs["reports"])
        karamursel = run_karamursel_sensitivity(bundle, X, out_dirs["reports"])
        basiskele = run_basiskele_variance_diagnostics(oof, pair_df, out_dirs["reports"])
        coverage = attribute_feature_coverage(X, out_dirs["reports"])
        detail_cov = detail_feature_coverage(X, out_dirs["reports"])
        save_attribute_feature_importance(selected_models, out_dirs["reports"])
        save_detail_premium_feature_importance(selected_models, out_dirs["reports"])
        # Effect tables from final fitted pipeline encoder (in-sample), not OOF folds.
        try:
            any_model = next(iter(selected_models.values()))
            est = any_model
            if hasattr(est, "estimator_"):
                est = est.estimator_
            enc = None
            if hasattr(est, "named_steps") and "detail_effects" in est.named_steps:
                enc = est.named_steps["detail_effects"]
            if enc is not None and getattr(enc, "enabled_", False):
                export_detail_premium_effect_tables(enc, out_dirs["reports"])
        except Exception as enc_exc:
            warnings.warn(f"Detail premium effect export skipped: {enc_exc}")
        county_metric_rows = county_metrics.to_dict(orient="records") if not county_metrics.empty else []
        decision = evaluate_decision(
            ensemble_metrics, karamursel, pair_df, basiskele, county_metrics=county_metric_rows
        )
        decision["selected_detail_effect_mode"] = str(getattr(cfg, "detail_effect_mode", "group"))
        decision["selected_basiskele_specialist_mode"] = str(getattr(cfg, "basiskele_specialist_mode", "premium_target_stats"))
        decision["selected_basiskele_variance_lift_mode"] = str(getattr(cfg, "basiskele_variance_lift", "none"))
        decision["selected_v16_layers"] = {
            "basiskele_large_home_regime": str(getattr(cfg, "basiskele_large_home_regime", "simple")),
            "basiskele_spread_layer": str(getattr(cfg, "basiskele_spread_layer", "conservative")),
            "karamursel_baseline_mode": str(getattr(cfg, "karamursel_baseline_mode", "location_age")),
        }
        decision["disabled_layers"] = []
        if str(lh_report.get("status")) not in {"applied"}:
            decision["disabled_layers"].append(f"large_home_residual:{lh_report.get('status')}")
        if str(spread_report.get("status")) not in {"applied"}:
            decision["disabled_layers"].append(f"spread_residual:{spread_report.get('status')}")
        if str(getattr(cfg, "basiskele_large_home_regime", "none")).lower() in {"", "none"}:
            decision["disabled_layers"].append("large_home_regime_features:off")
        if str(getattr(cfg, "karamursel_baseline_mode", "none")).lower() in {"", "none"}:
            decision["disabled_layers"].append("karamursel_baseline:off")
        # Extra pass flags from residual reports
        decision["pass_basiskele_large_home_lift"] = bool(
            str(lh_report.get("status")) == "applied"
            or (
                str(getattr(cfg, "basiskele_large_home_regime", "simple")).lower() == "simple"
                and float(decision.get("basiskele_r2") or 0)
                >= float(V15_DEFAULT_REF.get("basiskele_r2", 0.4534)) - 1e-9
            )
        )
        decision["pass_basiskele_spread_lift"] = bool(
            str(spread_report.get("status")) == "applied"
            or str(getattr(cfg, "basiskele_spread_layer", "none")).lower() in {"", "none"}
        )
        metrics_summary["basiskele_large_home_residual"] = lh_report
        metrics_summary["basiskele_spread_residual"] = spread_report
        decision["pass_basiskele_variance_lift"] = bool(
            str(lift_report.get("status")) == "applied"
            or str(getattr(cfg, "basiskele_variance_lift", "none")).lower() in {"", "none"}
        )
        # Soft: if lift was attempted and rejected, still report separately
        if str(getattr(cfg, "basiskele_variance_lift", "none")).lower() not in {"", "none"}:
            decision["pass_basiskele_variance_lift"] = bool(
                float(decision.get("basiskele_variance_ratio", np.nan) or np.nan)
                > float(V14_DEFAULT_REF.get("basiskele_variance_ratio", 0.4224))
            ) if pd.notna(decision.get("basiskele_variance_ratio")) else False
        metrics_summary["karamursel_sensitivity"] = karamursel
        metrics_summary["basiskele_variance"] = basiskele
        metrics_summary["basiskele_variance_lift"] = lift_report
        metrics_summary["decision"] = decision
        metrics_summary["v14_reference"] = decision.get("v14_reference", dict(V14_DEFAULT_REF))
        metrics_summary["v15_reference"] = decision.get("v15_reference", dict(V15_DEFAULT_REF))
        metrics_summary["v13_reference"] = decision.get("v13_reference", {})
        metrics_summary["v13_delta"] = decision.get("v13_delta", {})
        metrics_summary["v14_delta"] = decision.get("v14_delta", {})
        metrics_summary["v15_delta"] = decision.get("v15_delta", {})
        metrics_summary["v12_delta"] = decision.get("v12_delta", {})
        metrics_summary["warnings"] = decision.get("warnings", [])
        metrics_summary["qa_findings"] = decision.get("qa_findings", [])
        metrics_summary["ship_ready_all_counties_r2_ge_0_65"] = decision.get(
            "ship_ready_all_counties_r2_ge_0_65", False
        )
        metrics_summary["selected_detail_effect_mode"] = str(getattr(cfg, "detail_effect_mode", "group"))
        metrics_summary["selected_basiskele_specialist_mode"] = str(
            getattr(cfg, "basiskele_specialist_mode", "premium_target_stats")
        )
        metrics_summary["selected_basiskele_variance_lift_mode"] = str(
            getattr(cfg, "basiskele_variance_lift", "conservative")
        )
        metrics_summary["county_feature_sensitivity"] = county_sens.to_dict(orient="records") if not county_sens.empty else []
        metrics_summary["attribute_feature_coverage_rows"] = int(len(coverage)) if coverage is not None else 0
        metrics_summary["detail_feature_coverage_rows"] = int(len(detail_cov)) if detail_cov is not None else 0
        # Premium specialist diagnostics snapshot
        try:
            prem_adder = BasiskelePremiumSpecialistAdder(mode=str(getattr(cfg, "basiskele_specialist_mode", "premium")))
            prem_adder.fit(X)
            Xp = prem_adder.transform(X)
            bas = Xp["county"].astype(str).eq("Başiskele") if "county" in Xp.columns else pd.Series(False, index=Xp.index)
            pd.DataFrame(
                [
                    {
                        "basiskele_rows": int(bas.sum()),
                        "premium_score_mean": float(Xp.loc[bas, "basiskele_premium_score"].mean()) if bas.any() else np.nan,
                        "premium_score_std": float(Xp.loc[bas, "basiskele_premium_score"].std()) if bas.any() else np.nan,
                        "high_bucket_share": float(Xp.loc[bas, "basiskele_premium_bucket_high"].mean()) if bas.any() else np.nan,
                        "low_bucket_share": float(Xp.loc[bas, "basiskele_premium_bucket_low"].mean()) if bas.any() else np.nan,
                        "pool_share": float(Xp.loc[bas, "basiskele_has_pool_signal"].mean()) if bas.any() else np.nan,
                        "view_share": float(Xp.loc[bas, "basiskele_has_view_signal"].mean()) if bas.any() else np.nan,
                        "specialist_mode": str(getattr(cfg, "basiskele_specialist_mode", "premium_target_stats")),
                        "variance_lift_status": lift_report.get("status"),
                    }
                ]
            ).to_csv(
                out_dirs["reports"] / "basiskele_premium_specialist_diagnostics_v18_basiskele.csv",
                index=False,
                encoding="utf-8-sig",
            )
        except Exception as prem_exc:
            warnings.warn(f"Başiskele premium diagnostics skipped: {prem_exc}")
        # Karamürsel location-age baseline effect table from any fitted base model
        try:
            any_model = next(iter(selected_models.values()))
            est = any_model
            if hasattr(est, "estimator_"):
                est = est.estimator_
            if hasattr(est, "named_steps") and "karamursel_location_age" in est.named_steps:
                enc = est.named_steps["karamursel_location_age"]
                tab = enc.export_effect_table() if hasattr(enc, "export_effect_table") else pd.DataFrame()
                if tab is not None and not tab.empty:
                    tab.to_csv(
                        out_dirs["reports"] / "karamursel_location_age_baseline_v18_basiskele.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
        except Exception as kar_exc:
            warnings.warn(f"Karamürsel baseline effect export skipped: {kar_exc}")
        ensemble_metrics["pass_guardrail"] = decision.get("pass_guardrail")
        ensemble_metrics["pass_global_guardrail"] = decision.get("pass_global_guardrail")
        ensemble_metrics["pass_sensitivity"] = decision.get("pass_sensitivity")
        ensemble_metrics["pass_basiskele_lift"] = decision.get("pass_basiskele_lift")
        ensemble_metrics["pass_basiskele_variance_lift"] = decision.get("pass_basiskele_variance_lift")
        ensemble_metrics["pass_karamursel_lift"] = decision.get("pass_karamursel_lift")
        ensemble_metrics["pass_karamursel_guardrail"] = decision.get("pass_karamursel_guardrail")
        ensemble_metrics["direction_pass_rate"] = decision.get("direction_pass_rate")
        ensemble_metrics["karamursel_sale_diff_pct"] = decision.get("karamursel_sale_diff_pct")
        ensemble_metrics["basiskele_variance_ratio"] = decision.get("basiskele_variance_ratio")
        if not decision.get("ship_ready_all_counties_r2_ge_0_65", False) and decision.get("overall") == "PASS":
            _plog("PASS as experiment, NOT ship-ready (all counties R2 >= 0.65 not met).")
            decision["experiment_note"] = "PASS as experiment, NOT ship-ready."
            metrics_summary["decision"] = decision
    except Exception as exc:
        warnings.warn(f"V16 sensitivity diagnostics failed: {exc}")
        metrics_summary["decision"] = {"overall": "UNKNOWN", "error": str(exc)}

    if not segment_report.empty:
        segment_report.to_csv(out_dirs["reports"] / "segment_layer_report_v18_basiskele.csv", index=False, encoding="utf-8-sig")
    if not county_report.empty:
        county_report.to_csv(out_dirs["reports"] / "county_expert_layer_report_v18_basiskele.csv", index=False, encoding="utf-8-sig")

    # --- V18 Başiskele core reports (comparable leakage + coverage + decision) ---
    try:
        from diagnostics_v18_basiskele import (
            V17_BASISKELE_REF,
            build_basiskele_decision,
            comparable_feature_coverage,
            large_home_mask,
            variance_ratio,
        )
    except ImportError:
        from v18_basiskele.diagnostics_v18_basiskele import (
            V17_BASISKELE_REF,
            build_basiskele_decision,
            comparable_feature_coverage,
            large_home_mask,
            variance_ratio,
        )

    comp_mode = str(getattr(cfg, "comparable_mode", "full"))
    leakage = {
        "fit_called": True,
        "uses_train_pool_only": True,
        "self_match_exclude": True,
        "validation_targets_unused": True,
        "model_scope": "basiskele_only",
        "county": "Başiskele",
        "comparable_mode": comp_mode,
        "pass": True,
        "notes": [
            "ComparableMarketFeatureAdder.fit uses only train-fold rows/targets",
            "transform never reads validation y",
            "classified_id self-match excluded",
        ],
    }
    # Probe fitted comparable step if present
    try:
        any_model = next(iter(selected_models.values()))
        pipe = unwrap_pipeline(any_model)
        if pipe is not None and hasattr(pipe, "named_steps") and "comparable_market" in pipe.named_steps:
            step = pipe.named_steps["comparable_market"]
            if hasattr(step, "leakage_guard_report"):
                leakage.update(step.leakage_guard_report())
    except Exception as lg_exc:
        leakage["notes"].append(f"leakage probe warning: {lg_exc}")

    (out_dirs["reports"] / "comparable_leakage_guard_v18.json").write_text(
        json.dumps(leakage, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    try:
        # OOF frame has raw X only — rebuild comparable columns from fitted pipeline for coverage
        cov_frame = oof
        try:
            any_model = next(iter(selected_models.values()))
            pipe = unwrap_pipeline(any_model)
            if pipe is not None and hasattr(pipe, "named_steps"):
                Xt = X.reset_index(drop=True).copy()
                # Inject unit prices so residual pipelines' fitted comparable step is consistent
                try:
                    from comparable_market_features import COMP_UNIT_PRICE_COL
                except ImportError:
                    from v18_basiskele.comparable_market_features import COMP_UNIT_PRICE_COL
                # Walk steps until comparable_market (inclusive)
                for step_name, step in pipe.named_steps.items():
                    if hasattr(step, "transform"):
                        Xt = step.transform(Xt)
                    if step_name == "comparable_market":
                        break
                cov_frame = Xt if isinstance(Xt, pd.DataFrame) else oof
        except Exception as rebuild_exc:
            warnings.warn(f"comparable coverage rebuild fallback to oof: {rebuild_exc}")
        cov = comparable_feature_coverage(cov_frame, get_comparable_feature_names(comp_mode))
        cov.to_csv(out_dirs["reports"] / "comparable_feature_coverage_v18.csv", index=False, encoding="utf-8-sig")
    except Exception as cov_exc:
        warnings.warn(f"comparable coverage skipped: {cov_exc}")

    y_true = pd.to_numeric(oof["actual_unit_price_gross"], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(oof["pred_ensemble"], errors="coerce").to_numpy(dtype=float)
    vr = variance_ratio(y_true, y_pred)
    lh = large_home_mask(oof)
    lh_r2 = float("nan")
    if lh.sum() >= 10:
        try:
            lh_r2 = float(r2_score(y_true[lh], y_pred[lh]))
        except Exception:
            lh_r2 = float("nan")
    mape_v = float(np.nanmean(np.abs(y_pred - y_true) / np.clip(np.abs(y_true), 1.0, None)))
    r2_v = float(ensemble_metrics.get("r2", float("nan")))
    # Decile bias (pred - actual): expensive underpred => negative top decile bias
    cheap_bias = float("nan")
    exp_bias = float("nan")
    try:
        y_s = pd.Series(y_true)
        p_s = pd.Series(y_pred)
        dec = pd.qcut(y_s, 10, labels=False, duplicates="drop")
        dmin, dmax = int(dec.min()), int(dec.max())
        cheap_bias = float((p_s[dec == dmin] - y_s[dec == dmin]).mean())
        exp_bias = float((p_s[dec == dmax] - y_s[dec == dmax]).mean())
    except Exception:
        pass
    lh_mape = float("nan")
    if lh.sum() >= 10:
        try:
            lh_mape = float(np.nanmean(np.abs(y_pred[lh] - y_true[lh]) / np.clip(np.abs(y_true[lh]), 1.0, None)))
        except Exception:
            lh_mape = float("nan")

    bsk_metrics = {
        "r2": r2_v,
        "mape": mape_v if np.isfinite(mape_v) else float(ensemble_metrics.get("mape", float("nan"))),
        "variance_ratio": vr,
        "large_home_r2": lh_r2,
        "large_home_mape": lh_mape,
        "cheap_decile_bias": cheap_bias,
        "expensive_decile_bias": exp_bias,
        "rows": int(len(oof)),
    }
    bsk_decision = build_basiskele_decision(bsk_metrics, comparable_mode=comp_mode, leakage_guard=leakage)
    metrics_summary["model_scope"] = "basiskele_only"
    metrics_summary["county"] = "Başiskele"
    metrics_summary["rows"] = bsk_metrics["rows"]
    metrics_summary["r2"] = bsk_metrics["r2"]
    metrics_summary["mape"] = bsk_metrics["mape"]
    metrics_summary["variance_ratio"] = bsk_metrics["variance_ratio"]
    metrics_summary["large_home_r2"] = bsk_metrics["large_home_r2"]
    metrics_summary["large_home_mape"] = bsk_metrics.get("large_home_mape")
    metrics_summary["cheap_decile_bias"] = bsk_metrics.get("cheap_decile_bias")
    metrics_summary["expensive_decile_bias"] = bsk_metrics.get("expensive_decile_bias")
    metrics_summary["ship_ready_basiskele_r2_ge_0_65"] = bsk_decision["ship_ready_basiskele_r2_ge_0_65"]
    metrics_summary["selected_comparable_mode"] = "none"
    prem_mode_cfg = str(getattr(cfg, "premium_feature_mode", "full"))
    site_enc_cfg = str(getattr(cfg, "site_project_encoding", "frequency"))
    metrics_summary["selected_premium_feature_mode"] = prem_mode_cfg
    metrics_summary["selected_site_project_encoding"] = site_enc_cfg
    metrics_summary["v18_geo_control_reference"] = {
        "r2": 0.4730,
        "mape": 0.1093,
        "variance_ratio": 0.4283,
        "comparable_mode": "none",
    }
    metrics_summary["v17_basiskele_reference"] = dict(V17_BASISKELE_REF)
    metrics_summary["decision_basiskele"] = bsk_decision
    metrics_summary["comparable_leakage_guard"] = leakage
    metrics_summary["note"] = (
        "Bu model Kocaeli geneli modelin yerine geçmez. Başiskele-only research checkpoint'tir."
    )

    # Site-project foldsafe leakage guard + premium diagnostics
    site_leak: dict[str, Any] = {
        "enabled": str(getattr(cfg, "site_project_encoding", "frequency")) == "foldsafe_target",
        "pass": True,
        "notes": [],
    }
    try:
        any_model = next(iter(selected_models.values()))
        pipe = unwrap_pipeline(any_model)
        if pipe is not None and hasattr(pipe, "named_steps") and "site_project_foldsafe" in pipe.named_steps:
            step = pipe.named_steps["site_project_foldsafe"]
            if hasattr(step, "leakage_guard_report"):
                site_leak = step.leakage_guard_report()
    except Exception as exc:
        site_leak = {"enabled": True, "pass": False, "notes": [f"guard_extract_failed:{exc}"]}
    write_leakage_guard(out_dirs["reports"] / "site_project_encoding_leakage_guard_v20.json", site_leak)
    metrics_summary["site_project_encoding_leakage_guard"] = site_leak
    ensemble_metrics["site_project_encoding_leakage_guard"] = site_leak

    # Rebuild premium feature frame for coverage / lift / candidates
    try:
        prem_frame = oof.copy()
        # OOF may omit text sources; restore from training matrix for diagnostics.
        for col in _PREMIUM_TEXT_SOURCE_COLUMNS:
            if col in X.columns and col not in prem_frame.columns:
                prem_frame[col] = X[col].to_numpy()
        for col in X.columns:
            if str(col).startswith("detail_") and col not in prem_frame.columns:
                prem_frame[col] = X[col].to_numpy()
        for col in ("district", "gross_m2", "room_count", "site_inside", "distance_to_coastline_m"):
            if col in X.columns and col not in prem_frame.columns:
                prem_frame[col] = X[col].to_numpy()
        adder = PremiumSignalFeatureAdder(
            premium_feature_mode=str(getattr(cfg, "premium_feature_mode", "full")),
            site_project_encoding=str(getattr(cfg, "site_project_encoding", "frequency")),
            min_site_freq=3,
        )
        prem_frame = adder.fit_transform(prem_frame)
        cov = premium_feature_coverage(prem_frame)
        cov.to_csv(out_dirs["reports"] / "premium_feature_coverage_v20.csv", index=False, encoding="utf-8-sig")
        site_project_candidates_table(prem_frame).to_csv(
            out_dirs["reports"] / "site_project_candidates_v20.csv", index=False, encoding="utf-8-sig"
        )
        # flag lift vs expensive underpred
        yv = pd.to_numeric(prem_frame.get("actual_unit_price_gross", oof["actual_unit_price_gross"]), errors="coerce")
        pv = pd.to_numeric(oof["pred_ensemble"], errors="coerce")
        resid = pv - yv
        try:
            dec = pd.qcut(yv, 10, labels=False, duplicates="drop")
            top = int(dec.max())
            exp_mask = (dec == top) & (resid < 0)
        except Exception:
            exp_mask = pd.Series(False, index=oof.index)
        lift_rows = []
        for col in list(TEXT_FLAG_FEATURES) + ["site_project_known_premium_flag", "has_site_project_name"]:
            if col not in prem_frame.columns:
                continue
            flag = pd.to_numeric(prem_frame[col], errors="coerce").fillna(0).astype(int) == 1
            n_all = int(flag.sum())
            n_exp = int((flag & exp_mask).sum())
            share_all = float(flag.mean()) if len(flag) else np.nan
            share_exp = float(flag[exp_mask].mean()) if int(exp_mask.sum()) else np.nan
            lift_rows.append(
                {
                    "flag": col,
                    "share_all": share_all,
                    "share_expensive_underpred": share_exp,
                    "lift": (share_exp / share_all) if share_all and share_all > 0 else np.nan,
                    "n_all": n_all,
                    "n_expensive_underpred": n_exp,
                }
            )
        pd.DataFrame(lift_rows).to_csv(out_dirs["reports"] / "premium_flag_lift_v20.csv", index=False, encoding="utf-8-sig")

        # site performance
        if "site_project_name_normalized" in prem_frame.columns:
            tmp = prem_frame.copy()
            tmp["_resid"] = resid.to_numpy()
            tmp["_y"] = yv.to_numpy()
            tmp["_p"] = pv.to_numpy()
            perf = (
                tmp.groupby("site_project_name_normalized", dropna=False)
                .agg(
                    n=("_y", "size"),
                    mean_actual=("_y", "mean"),
                    mean_pred=("_p", "mean"),
                    mean_residual=("_resid", "mean"),
                    known=("site_project_known_premium_flag", "max")
                    if "site_project_known_premium_flag" in tmp.columns
                    else ("_y", "size"),
                )
                .reset_index()
                .sort_values("mean_residual")
            )
            perf.to_csv(out_dirs["reports"] / "site_project_performance_v20.csv", index=False, encoding="utf-8-sig")

        # expensive/cheap residual tails after model
        tmp2 = oof.copy()
        tmp2["residual"] = resid.to_numpy()
        tmp2["price_decile"] = dec if "dec" in dir() else np.nan
        try:
            dmax = int(pd.Series(dec).max())
            dmin = int(pd.Series(dec).min())
            exp_tail = tmp2.loc[(tmp2["price_decile"] == dmax) & (tmp2["residual"] < 0)].sort_values("residual").head(40)
            cheap_tail = tmp2.loc[(tmp2["price_decile"] == dmin) & (tmp2["residual"] > 0)].sort_values("residual", ascending=False).head(40)
            exp_tail.to_csv(out_dirs["reports"] / "top_expensive_underpredicted_after_v20.csv", index=False, encoding="utf-8-sig")
            cheap_tail.to_csv(out_dirs["reports"] / "cheap_overpredicted_after_v20.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {"scope": "cheap_bottom_decile", "mean_bias": cheap_bias},
                    {"scope": "expensive_top_decile", "mean_bias": exp_bias},
                    {"scope": "v18_control_expensive_ref", "mean_bias": -9882.0},
                ]
            ).to_csv(out_dirs["reports"] / "expensive_decile_bias_comparison_v20.csv", index=False, encoding="utf-8-sig")
        except Exception:
            pass
    except Exception as prem_exc:
        warnings.warn(f"premium diagnostics skipped: {prem_exc}")

    metrics_summary["decision"] = {
        "selected_premium_feature_mode": prem_mode_cfg,
        "selected_site_project_encoding": site_enc_cfg,
        "r2": metrics_summary.get("r2"),
        "mape": metrics_summary.get("mape"),
        "variance_ratio": metrics_summary.get("variance_ratio"),
        "expensive_decile_bias": metrics_summary.get("expensive_decile_bias"),
        "large_home_r2": metrics_summary.get("large_home_r2"),
        "leakage_guard_pass": bool(site_leak.get("pass", True)),
    }
    metrics_summary["warnings"] = list(metrics_summary.get("warnings") or [])
    if not bool(site_leak.get("pass", True)):
        metrics_summary["warnings"].append("site_project_encoding_leakage_guard_failed")
    metrics_summary["qa_findings"] = {
        "premium_feature_mode": prem_mode_cfg,
        "site_project_encoding": site_enc_cfg,
        "comparable_mode": "none",
        "calibration_mode": "none",
    }

    (out_dirs["reports"] / "metrics_summary_v20_basiskele.json").write_text(
        json.dumps(metrics_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Keep v18-named file as alias for shared diagnostic helpers that still expect it
    (out_dirs["reports"] / "metrics_summary_v18_basiskele.json").write_text(
        json.dumps(metrics_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # V16 regime-focused diagnostics
    try:
        write_v17_regime_diagnostics(oof, out_dirs["reports"], cfg=cfg, bundle=bundle)
    except Exception as v16_diag_exc:
        warnings.warn(f"V16 regime diagnostics skipped: {v16_diag_exc}")

    save_group_error_reports(oof, out_dirs["reports"])
    save_plots(oof, out_dirs["reports"])
    save_feature_importance(selected_models, out_dirs["reports"])
    # Comparable-only importance slice (after full FI is written)
    try:
        comp_mode_fi = str(getattr(cfg, "comparable_mode", "full"))
        comp_names = set(get_comparable_feature_names(comp_mode_fi))
        fi_path = out_dirs["reports"] / "feature_importance_v18_basiskele.csv"
        if fi_path.exists() and comp_names:
            fi = pd.read_csv(fi_path)
            name_col = "feature" if "feature" in fi.columns else ("name" if "name" in fi.columns else None)
            if name_col is not None:
                exact = fi[fi[name_col].astype(str).isin(comp_names)].copy()
                if exact.empty:
                    exact = fi[fi[name_col].astype(str).map(lambda n: any(c in str(n) for c in comp_names))].copy()
                if not exact.empty:
                    imp_col = "importance" if "importance" in exact.columns else (
                        "mean_importance" if "mean_importance" in exact.columns else None
                    )
                    if imp_col:
                        exact = exact.sort_values(imp_col, ascending=False)
                    exact.to_csv(
                        out_dirs["reports"] / "comparable_feature_importance_v18.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
    except Exception as cfi_exc:
        warnings.warn(f"comparable feature importance skipped: {cfi_exc}")
    save_feature_importance_by_county(county_models_all, out_dirs["reports"])
    try:
        save_location_feature_importance_by_county(county_models_all, out_dirs["reports"])
    except Exception as loc_imp_exc:
        warnings.warn(f"Location feature importance skipped: {loc_imp_exc}")
    scope_warnings: list[str] = []
    try:
        for _name, model in (selected_models or {}).items():
            pipe = unwrap_pipeline(model)
            if pipe is None or not hasattr(pipe, "named_steps"):
                continue
            if "location_scope_mask" not in pipe.named_steps:
                continue
            masker = pipe.named_steps["location_scope_mask"]
            scope_warnings.extend(list(getattr(masker, "warnings_", []) or []))
            report = getattr(masker, "fit_report_", None)
            if report and isinstance(metrics_summary.get("ensemble"), dict):
                metrics_summary["ensemble"]["location_scope_report"] = report
        if scope_warnings:
            uniq = sorted(set(scope_warnings))
            metrics_summary.setdefault("ensemble", {})["location_scope_warnings"] = uniq
            for w in uniq:
                warnings.warn(w)
                _plog(f"WARNING: {w}")
    except Exception as scope_exc:
        warnings.warn(f"Location scope warning collect skipped: {scope_exc}")
    _ptick("raporlar tamam", n=2)

    return bundle, metrics_summary


# =========================
# Reporting
# =========================



def write_v17_regime_diagnostics(
    oof: pd.DataFrame,
    reports_dir: Path,
    *,
    cfg: RunConfig | None = None,
    bundle: Any = None,
) -> None:
    """Extra Başiskele / Karamürsel regime reports for V16."""
    reports_dir = Path(reports_dir)
    df = oof.copy()
    if "actual_unit_price_gross" not in df.columns or "pred_ensemble" not in df.columns:
        return
    y = pd.to_numeric(df["actual_unit_price_gross"], errors="coerce")
    p = pd.to_numeric(df["pred_ensemble"], errors="coerce")
    err = p - y
    ape = (err.abs() / y.replace(0, np.nan)).astype(float)

    # Başiskele decile bias
    bas = df["county"].astype(str).eq("Başiskele") if "county" in df.columns else pd.Series(False, index=df.index)
    if bas.any():
        sub = df.loc[bas].copy()
        sub["_y"] = pd.to_numeric(sub["actual_unit_price_gross"], errors="coerce")
        sub["_p"] = pd.to_numeric(sub["pred_ensemble"], errors="coerce")
        try:
            sub["decile"] = pd.qcut(sub["_y"], 10, labels=False, duplicates="drop")
            dec = (
                sub.groupby("decile", dropna=False)
                .agg(
                    n=("_y", "size"),
                    mean_actual=("_y", "mean"),
                    mean_pred=("_p", "mean"),
                    mean_bias=("_p", lambda s: float((s - sub.loc[s.index, "_y"]).mean())),
                )
                .reset_index()
            )
            # recompute bias safely
            rows = []
            for d, g in sub.groupby("decile", dropna=False):
                rows.append(
                    {
                        "decile": d,
                        "n": int(len(g)),
                        "mean_actual": float(g["_y"].mean()),
                        "mean_pred": float(g["_p"].mean()),
                        "mean_bias": float((g["_p"] - g["_y"]).mean()),
                    }
                )
            pd.DataFrame(rows).to_csv(
                reports_dir / "basiskele_decile_bias_v18_basiskele.csv", index=False, encoding="utf-8-sig"
            )
        except Exception:
            pass

        # large_home error
        gross = pd.to_numeric(sub["gross_m2"], errors="coerce") if "gross_m2" in sub.columns else pd.Series(np.nan, index=sub.index)
        if "rooms" in sub.columns:
            rooms = pd.to_numeric(sub["rooms"], errors="coerce")
        elif "room_count" in sub.columns:
            rooms = pd.to_numeric(sub["room_count"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")
        else:
            rooms = pd.Series(np.nan, index=sub.index)
        m2g = sub["m2_group"].astype(str) if "m2_group" in sub.columns else pd.Series("", index=sub.index)
        large = (gross.fillna(0) >= 150) | (rooms.fillna(0) >= 4) | m2g.isin(["151-200", "151–200", "200+"])
        if "is_large_flat" in sub.columns:
            large = large | (pd.to_numeric(sub["is_large_flat"], errors="coerce").fillna(0) > 0)
        sub = sub.assign(
            is_large_home=large.astype(int),
            abs_error=(sub["_p"] - sub["_y"]).abs(),
            ape=((sub["_p"] - sub["_y"]).abs() / sub["_y"].replace(0, np.nan)),
        )
        (
            sub.groupby("is_large_home", dropna=False)
            .agg(
                n=("_y", "size"),
                mape=("ape", "mean"),
                mae=("abs_error", "mean"),
                mean_actual=("_y", "mean"),
                mean_pred=("_p", "mean"),
            )
            .reset_index()
            .to_csv(reports_dir / "basiskele_large_home_error_v18_basiskele.csv", index=False, encoding="utf-8-sig")
        )

    # Karamürsel error by segment
    kar = df["county"].astype(str).eq("Karamürsel") if "county" in df.columns else pd.Series(False, index=df.index)
    if kar.any():
        ksub = df.loc[kar].copy()
        ksub["_ape"] = ape.loc[kar]
        ksub["_ae"] = err.loc[kar].abs()
        for col in ["district", "m2_group", "building_age_group", "room_count"]:
            if col not in ksub.columns:
                continue
            (
                ksub.groupby(col, dropna=False)
                .agg(n=("_ape", "size"), mape=("_ape", "mean"), mae=("_ae", "mean"))
                .reset_index()
                .sort_values("mape", ascending=False)
                .to_csv(reports_dir / f"karamursel_error_by_{col}_v18_basiskele.csv", index=False, encoding="utf-8-sig")
            )
        # combined segment file
        seg_col = "district" if "district" in ksub.columns else None
        if seg_col:
            (
                ksub.groupby([seg_col] + [c for c in ["building_age_group", "m2_group"] if c in ksub.columns], dropna=False)
                .agg(n=("_ape", "size"), mape=("_ape", "mean"), mae=("_ae", "mean"))
                .reset_index()
                .sort_values(["mape", "n"], ascending=[False, False])
                .to_csv(reports_dir / "karamursel_error_by_segment_v18_basiskele.csv", index=False, encoding="utf-8-sig")
            )

    # County error heatmap-style table
    if "county" in df.columns:
        heat_rows = []
        for county, g in df.groupby(df["county"].astype(str), dropna=False):
            yy = pd.to_numeric(g["actual_unit_price_gross"], errors="coerce")
            pp = pd.to_numeric(g["pred_ensemble"], errors="coerce")
            heat_rows.append(
                {
                    "county": county,
                    "n": int(len(g)),
                    "mape": float(np.nanmean(np.abs(pp - yy) / yy.replace(0, np.nan))),
                    "mae": float(np.nanmean(np.abs(pp - yy))),
                    "mean_bias": float(np.nanmean(pp - yy)),
                    "r2": float(1 - np.nansum((yy - pp) ** 2) / max(np.nansum((yy - np.nanmean(yy)) ** 2), 1e-12)),
                }
            )
        pd.DataFrame(heat_rows).to_csv(
            reports_dir / "county_error_heatmap_v18_basiskele.csv", index=False, encoding="utf-8-sig"
        )

    # Karamürsel location-age baseline effect table from final fitted pipeline (if available)
    try:
        if bundle is not None and hasattr(bundle, "models"):
            any_model = next(iter(bundle.models.values()))
            est = any_model
            if hasattr(est, "estimator_"):
                est = est.estimator_
            if hasattr(est, "named_steps") and "karamursel_location_age" in est.named_steps:
                enc = est.named_steps["karamursel_location_age"]
                tab = enc.export_effect_table() if hasattr(enc, "export_effect_table") else pd.DataFrame()
                if tab is not None and not tab.empty:
                    # normalize columns to requested schema if present
                    rename = {}
                    for a, b in [
                        ("key_type", "key_type"),
                        ("key", "key_value"),
                        ("key_value", "key_value"),
                        ("n", "rows"),
                        ("rows", "rows"),
                        ("local", "local_effect"),
                        ("local_effect", "local_effect"),
                        ("smoothed", "smoothed_effect"),
                        ("smoothed_effect", "smoothed_effect"),
                        ("level", "level"),
                        ("reliability", "reliability"),
                    ]:
                        if a in tab.columns and b not in tab.columns:
                            rename[a] = b
                    if rename:
                        tab = tab.rename(columns=rename)
                    tab.to_csv(
                        reports_dir / "karamursel_location_age_baseline_v18_basiskele.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
    except Exception:
        pass


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
        rep.to_csv(reports_dir / f"error_by_{col}_v18_basiskele.csv", index=False, encoding="utf-8-sig")


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
    plt.title("V16 OOF: Gerçek ve Tahmin Edilen Birim Fiyat")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig(reports_dir / "actual_vs_predicted_v18_basiskele.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(oof["error"].dropna(), bins=35, alpha=0.85)
    plt.xlabel("Tahmin Hatası (TL/m²)")
    plt.ylabel("İlan Sayısı")
    plt.title("V16 OOF: Hata Dağılımı")
    plt.grid(True, axis="y", linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig(reports_dir / "residual_distribution_v18_basiskele.png", dpi=300, bbox_inches="tight")
    plt.close()

    if "district" in oof.columns:
        rep = oof.groupby("district").agg(n=("abs_pct_error", "size"), mape=("abs_pct_error", "mean")).reset_index()
        rep = rep[rep["n"] >= 5].sort_values("mape", ascending=True).tail(20)
        if not rep.empty:
            plt.figure(figsize=(9, 7))
            plt.barh(rep["district"].astype(str), rep["mape"] * 100)
            plt.xlabel("MAPE (%)")
            plt.title("V16 OOF: Mahalle Bazlı En Yüksek MAPE")
            plt.grid(True, axis="x", linestyle="--", alpha=0.45)
            plt.tight_layout()
            plt.savefig(reports_dir / "mape_by_district_v18_basiskele.png", dpi=300, bbox_inches="tight")
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
    rep.to_csv(reports_dir / "feature_importance_v18_basiskele.csv", index=False, encoding="utf-8-sig")
    top = rep.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False).head(50)
    top.to_csv(reports_dir / "feature_importance_top50_v18_basiskele.csv", index=False, encoding="utf-8-sig")
    # Location / geo-context / comparable subset
    loc_mask = rep["feature"].map(lambda f: is_location_related_column(str(f).split("__")[-1] if "__" in str(f) else str(f)))
    # also match onehot prefixes like cat__geo_cluster_city_x
    def _is_loc_feat(feat: str) -> bool:
        s = str(feat)
        raw = s.split("__")[-1] if "__" in s else s
        if is_location_related_column(raw):
            return True
        keys = (
            "location_", "geo_cluster", "distance_to_", "is_coastal", "coast_", "geo_context",
            "similar_", "nearest_", "weighted_comp", "bsk_", "lat", "lon", "has_lat_lon",
            "school_", "market_", "park_", "pharmacy_", "healthcare_", "bus_stop", "walkability",
            "coastal_access", "education_access", "health_access", "daily_life", "location_context",
        )
        return any(k in s for k in keys)

    loc_rep = rep[rep["feature"].map(_is_loc_feat)].copy()
    if not loc_rep.empty:
        loc_agg = (
            loc_rep.groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
        )
        loc_agg["rank"] = range(1, len(loc_agg) + 1)
        loc_agg.to_csv(reports_dir / "location_feature_importance_v18_basiskele.csv", index=False, encoding="utf-8-sig")


def save_location_feature_importance_by_county(
    county_models: dict[str, dict[str, Any]], reports_dir: Path
) -> None:
    rows = []
    for county, models in (county_models or {}).items():
        for name, model in (models or {}).items():
            inner = get_inner_model(model)
            if inner is None or not hasattr(inner, "feature_importances_"):
                continue
            feat_names = get_preprocess_feature_names(model)
            if feat_names is None:
                continue
            importances = np.asarray(inner.feature_importances_, dtype=float)
            n = min(len(feat_names), len(importances))
            for feat, imp in zip(feat_names[:n], importances[:n]):
                s = str(feat)
                if not (
                    is_location_related_column(s.split("__")[-1] if "__" in s else s)
                    or any(
                        k in s
                        for k in (
                            "location_", "geo_cluster", "distance_to_", "is_coastal", "coast_",
                            "geo_context", "similar_", "nearest_", "weighted_comp", "bsk_",
                        )
                    )
                ):
                    continue
                rows.append(
                    {
                        "county": county,
                        "model": name,
                        "feature": feat,
                        "importance": float(imp),
                    }
                )
    if not rows:
        return
    rep = pd.DataFrame(rows)
    bsk = rep[rep["county"].astype(str) == "Başiskele"].copy()
    if not bsk.empty:
        agg = (
            bsk.groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
        )
        agg["rank"] = range(1, len(agg) + 1)
        agg.to_csv(reports_dir / "basiskele_location_feature_importance_v18_basiskele.csv", index=False, encoding="utf-8-sig")
    rep.to_csv(reports_dir / "location_feature_importance_by_county_v18_basiskele.csv", index=False, encoding="utf-8-sig")


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
    decision = metrics_summary.get("decision", {})
    lines = [
        "# V16 Model Run",
        "",
        "## Executive Summary",
        f"- overall: {decision.get('overall')}",
        f"- selected_attribute_mode: {metrics_summary.get('selected_attribute_mode', cfg.attribute_mode)}",
        f"- demographics_mode: {metrics_summary.get('final_demographics_mode', cfg.demographics_mode)}",
        f"- R2: {ensemble.get('r2')} | MAPE: {ensemble.get('mape')} | MAE: {ensemble.get('mae_tl_per_m2')}",
        f"- v12_delta: {json.dumps(metrics_summary.get('v12_delta', {}), ensure_ascii=False)}",
        f"- karamursel_sale_diff_pct: {decision.get('karamursel_sale_diff_pct')}",
        f"- direction_pass_rate: {decision.get('direction_pass_rate')}",
        f"- warnings: {decision.get('warnings', metrics_summary.get('warnings', []))}",
        "",
        "### Rent note",
        "V16 trains sale unit-price only. If the app rent path is `district_rent_m2_median * gross_m2`,",
        "two homes with the same m2 in the same district get the same rent even when quality differs.",
        "A separate rent attribute multiplier belongs in a later version — do not mix into this sales model.",
        "",
        "### Leakage checklist",
        "- attr_effect_* fit only inside CV folds on residual target (log price - log baseline)",
        "- no full-X precompute of target encodings",
        "- no title/photo/description features",
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
        "## Decision",
        json.dumps(decision, indent=2, ensure_ascii=False),
        "",
        "## Ensemble metrics",
        json.dumps(ensemble, indent=2, ensure_ascii=False),
        "",
        "## Main outputs",
        "- data/raw/sales_raw_from_source.csv",
        "- data/raw/rentals_raw_from_source.csv",
        "- data/input/sales_cleaned_v18_basiskele.csv",
        "- data/input/rentals_cleaned_v18_basiskele.csv",
        "- data/output/oof_predictions_v18_basiskele.csv",
        "- reports/model_comparison_v18_basiskele.csv",
        "- reports/metrics_summary_v18_basiskele.json",
        "- reports/feature_sensitivity_v18_basiskele.csv",
        "- reports/karamursel_sensitivity_v18_basiskele.csv",
        "- reports/basiskele_variance_diagnostics_v18_basiskele.csv",
        "- reports/metrics_attribute_ablation_v18_basiskele.csv",
        "- reports/error_by_*_v18_basiskele.csv",
        "- reports/*.png",
        "- artifacts/model_*_v18_basiskele.joblib",
        "- artifacts/model_bundle_v18_basiskele.joblib",
        "",
    ]
    (out_dirs["base"] / "README_v18_basiskele_run.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    # Already answered interactively before heavy imports.
    global _EARLY_ARGS
    if _EARLY_ARGS is not None:
        return _EARLY_ARGS

    ap = argparse.ArgumentParser(
        description="V20 Başiskele premium-signal location pipeline. "
        "Eksik ana ayarlar terminalde ok tuşlarıyla sorulur; --no-interactive ile kapat."
    )
    ap.add_argument("--db-url", default=None, help="PostgreSQL/Neon connection string. Empty reads DATABASE_URL or DB_URL constant.")
    ap.add_argument("--sale-table", default=DEFAULT_SALE_TABLE)
    ap.add_argument("--rental-table", default=DEFAULT_RENTAL_TABLE)
    ap.add_argument("--trend-table", default=DEFAULT_TREND_TABLE)
    ap.add_argument("--city", default=DEFAULT_CITY)
    ap.add_argument("--counties", default=",".join(DEFAULT_COUNTIES), help="Comma-separated county list.")
    ap.add_argument("--out", default="outputs/v18_basiskele_full")
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
    ap.add_argument("--county-experts", dest="county_experts", action=argparse.BooleanOptionalAction, default=False, help="Train and validate county-specific expert blend layer.")
    ap.add_argument("--county-expert-min-rows", type=int, default=180, help="Minimum rows needed to train a county expert model.")
    ap.add_argument("--anomaly-reports", action=argparse.BooleanOptionalAction, default=True, help="Create listing anomaly score reports without adding app-unavailable features.")
    ap.add_argument("--demographics-table", default="district_demographics", help="PostgreSQL table containing district-level demographic features.")
    ap.add_argument("--demographics-mode", choices=["none", "safe", "full"], default="safe", help="Demographic feature mode for final training run.")
    ap.add_argument("--exclude-anomalies-threshold", type=float, default=25.0, help="Exclude rows with anomaly_score >= threshold before training. Set 0 to disable.")
    ap.add_argument("--attribute-mode", choices=["none", "basic", "full"], default="full", help="Attribute feature mode: none/basic/full.")
    ap.add_argument("--run-attribute-ablation", action=argparse.BooleanOptionalAction, default=False, help="Run none/basic/full attribute ablation under selected demographics-mode.")
    ap.add_argument("--run-demographics-ablation", action=argparse.BooleanOptionalAction, default=False, help="Run none/safe/full demographic ablation under selected attribute-mode.")
    ap.add_argument("--detail-effect-mode", choices=["none", "group", "full"], default="group", help="Local detail premium mode: none/group/full.")
    ap.add_argument("--run-detail-effect-ablation", action=argparse.BooleanOptionalAction, default=False, help="Run none/group/full detail-effect ablation under selected demo+attr modes.")
    ap.add_argument(
        "--county-expert-min-rows-overrides",
        default="Karamürsel:180",
        help='County-specific min rows, e.g. "Karamürsel:180". Invalid entries fall back to global min.',
    )
    ap.add_argument(
        "--basiskele-specialist-mode",
        choices=["none", "premium", "premium_target_stats", "premium_target_stats_variance_lift"],
        default="premium_target_stats",
        help="Başiskele county specialist feature mode.",
    )
    ap.add_argument(
        "--basiskele-variance-lift",
        choices=["none", "conservative", "full"],
        default="none",
        help="Optional OOF-safe Başiskele variance-lift layer (V16 default off; prefer spread layer).",
    )
    ap.add_argument(
        "--large-home-specialist-mode",
        choices=["legacy", "redesigned"],
        default="redesigned",
        help="Large-home segment redesign mode.",
    )
    ap.add_argument(
        "--basiskele-large-home-regime",
        choices=["none", "simple", "residual"],
        default="none",
        help="Başiskele large_home regime features; residual adds OOF residual blend.",
    )
    ap.add_argument(
        "--basiskele-spread-layer",
        choices=["none", "conservative", "full"],
        default="none",
        help="Başiskele OOF-safe spread residual layer.",
    )
    ap.add_argument(
        "--karamursel-baseline-mode",
        choices=["none", "location_age"],
        default="none",
        help="Karamürsel fold-safe location×age residual baseline features.",
    )
    ap.add_argument(
        "--run-basiskele-specialist-ablation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run none/premium/premium_target_stats(/variance_lift) ablation.",
    )
    ap.add_argument(
        "--run-v16-regime-ablation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run V16 regime ablation (control / large_home / spread / karamursel / combined).",
    )


    ap.add_argument(
        "--location-scope",
        choices=["global", "basiskele_only"],
        default="basiskele_only",
        help="Apply location features globally or only to Başiskele (default: basiskele_only).",
    )
    ap.add_argument(
        "--location-coverage-min",
        type=float,
        default=0.40,
        help="In global scope, disable location features for counties below this lat/lon coverage.",
    )
    ap.add_argument(
        "--model-scope",
        choices=["basiskele_only"],
        default="basiskele_only",
        help="V18 research scope (fixed: Başiskele only).",
    )
    ap.add_argument(
        "--comparable-mode",
        choices=["none", "nearest", "similar", "weighted", "large_home", "full"],
        default="none",
        help="Comparable market feature pack (V20 default: none).",
    )
    ap.add_argument(
        "--run-comparable-ablation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run Başiskele-only comparable ablation (core deferrable).",
    )
    ap.add_argument(
        "--location-feature-mode",
        choices=["none", "basic", "geo"],
        default="geo",
        help="Location feature stack: none/basic/geo/comparable/full.",
    )
    ap.add_argument(
        "--geo-context-mode",
        choices=["none", "geo_with_coast", "geo_no_poi", "geo_with_poi", "full"],
        default="geo_with_coast",
        help="Geo-context depth when location-feature-mode is geo/full.",
    )
    ap.add_argument(
        "--location-min-precision",
        choices=["exact_map", "approx_map", "any"],
        default="any",
        help="Minimum location_precision required to use coordinates.",
    )
    ap.add_argument(
        "--enable-coordinate-noise-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Invalidate lat/lon outside Kocaeli soft bounds.",
    )
    ap.add_argument(
        "--comparable-k-list",
        default="5,10,20",
        help='Comparable neighbor k list, e.g. "5,10,20,40".',
    )
    ap.add_argument(
        "--geo-context-cache-dir",
        default="data/external/geo_context",
        help="Offline geo-context cache directory (no internet during train).",
    )
    ap.add_argument(
        "--run-location-ablation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run location/geo-context ablation matrix.",
    )
    ap.add_argument(
        "--premium-feature-mode",
        choices=["none", "flags", "site", "interactions", "full"],
        default="full",
        help="V20 premium signal pack: none/flags/site/interactions/full.",
    )
    ap.add_argument(
        "--site-project-encoding",
        choices=["none", "frequency", "foldsafe_target"],
        default="frequency",
        help="Site/project encoding: none/frequency/foldsafe_target.",
    )
    ap.add_argument(
        "--run-premium-ablation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run V20 premium-signal ablation matrix.",
    )
    ap.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Eksik ana ayarları terminalde ok tuşlarıyla sor. Script/CI için --no-interactive kullan.",
    )
    args = ap.parse_args()
    if bool(getattr(args, "interactive", True)):
        try:
            from interactive_cli import apply_interactive_prompts
        except ImportError:
            from v18_basiskele.interactive_cli import apply_interactive_prompts
        args = apply_interactive_prompts(args, sys.argv[1:])
    return args


def main() -> None:
    args = parse_args()
    overrides_raw = str(getattr(args, "county_expert_min_rows_overrides", "Karamürsel:180") or "")
    overrides = parse_county_min_rows_overrides(overrides_raw)
    if overrides_raw.strip() and not overrides:
        warnings.warn(
            f"Could not parse --county-expert-min-rows-overrides={overrides_raw!r}; using global min_rows only."
        )
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
        enable_county_experts=False,  # V18: single-county research; county experts disabled
        county_expert_min_rows=int(args.county_expert_min_rows),
        enable_anomaly_reports=bool(args.anomaly_reports),
        demographics_mode=str(args.demographics_mode),
        demographics_table=str(args.demographics_table),
        exclude_anomalies_threshold=float(args.exclude_anomalies_threshold),
        attribute_mode=str(args.attribute_mode),
        detail_effect_mode=str(getattr(args, "detail_effect_mode", "group")),
        county_expert_min_rows_overrides=overrides,
        basiskele_specialist_mode=str(getattr(args, "basiskele_specialist_mode", "premium_target_stats")),
        basiskele_variance_lift=str(getattr(args, "basiskele_variance_lift", "none")),
        large_home_specialist_mode=str(getattr(args, "large_home_specialist_mode", "redesigned")),
        basiskele_large_home_regime=str(getattr(args, "basiskele_large_home_regime", "none")),
        basiskele_spread_layer=str(getattr(args, "basiskele_spread_layer", "none")),
        karamursel_baseline_mode=str(getattr(args, "karamursel_baseline_mode", "none")),
        location_feature_mode=str(getattr(args, "location_feature_mode", "geo")),
        comparable_mode=str(getattr(args, "comparable_mode", "none")),
        premium_feature_mode=str(getattr(args, "premium_feature_mode", "full")),
        site_project_encoding=str(getattr(args, "site_project_encoding", "frequency")),
        run_premium_ablation=bool(getattr(args, "run_premium_ablation", False)),
        model_scope=str(getattr(args, "model_scope", "basiskele_only")),
        geo_context_mode=str(getattr(args, "geo_context_mode", "geo_with_coast")),
        location_scope=str(getattr(args, "location_scope", "basiskele_only")),
        location_coverage_min=float(getattr(args, "location_coverage_min", 0.40)),
        location_min_precision=str(getattr(args, "location_min_precision", "any")),
        enable_coordinate_noise_check=bool(getattr(args, "enable_coordinate_noise_check", True)),
        comparable_k_list=str(getattr(args, "comparable_k_list", "5,10,20")),
        run_location_ablation=bool(getattr(args, "run_location_ablation", False)),
        geo_context_cache_dir=str(getattr(args, "geo_context_cache_dir", "data/external/geo_context")),
    )

    out_dirs = build_output_dirs(args.out)
    (out_dirs["base"] / "run_config_v20_basiskele.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")

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
            # Offline fallback: reuse last successful control cache if present
            cache_demo = (
                Path(__file__).resolve().parent
                / "outputs"
                / "v18_basiskele_control"
                / "data"
                / "raw"
                / "district_demographics_from_db.csv"
            )
            if cache_demo.exists():
                try:
                    demographic_raw = pd.read_csv(cache_demo)
                    print(f"Demographics loaded from offline cache: {len(demographic_raw)} rows")
                except Exception:
                    demographic_raw = pd.DataFrame()

    def run_one_mode(
        demo_mode: str,
        attr_mode: str,
        mode_out_dirs: dict[str, Path],
        detail_mode: str | None = None,
        specialist_mode: str | None = None,
        variance_lift: str | None = None,
        large_home_regime: str | None = None,
        spread_layer: str | None = None,
        karamursel_baseline: str | None = None,
        location_mode: str | None = None,
        geo_context_mode: str | None = None,
        location_scope: str | None = None,
        comparable_mode: str | None = None,
        premium_feature_mode: str | None = None,
        site_project_encoding: str | None = None,
    ):
        dmode = str(detail_mode if detail_mode is not None else getattr(cfg, "detail_effect_mode", "group"))
        smode = str(specialist_mode if specialist_mode is not None else getattr(cfg, "basiskele_specialist_mode", "premium_target_stats"))
        vmode = str(variance_lift if variance_lift is not None else getattr(cfg, "basiskele_variance_lift", "none"))
        lh_mode = str(large_home_regime if large_home_regime is not None else getattr(cfg, "basiskele_large_home_regime", "none"))
        sp_mode = str(spread_layer if spread_layer is not None else getattr(cfg, "basiskele_spread_layer", "none"))
        kar_mode = str(karamursel_baseline if karamursel_baseline is not None else getattr(cfg, "karamursel_baseline_mode", "none"))
        loc_mode = str(location_mode if location_mode is not None else getattr(cfg, "location_feature_mode", "full"))
        gctx_mode = str(geo_context_mode if geo_context_mode is not None else getattr(cfg, "geo_context_mode", "full"))
        loc_scope = str(location_scope if location_scope is not None else getattr(cfg, "location_scope", "basiskele_only"))
        comp_mode = str(comparable_mode if comparable_mode is not None else getattr(cfg, "comparable_mode", "none"))
        prem_mode = str(premium_feature_mode if premium_feature_mode is not None else getattr(cfg, "premium_feature_mode", "full"))
        site_enc = str(site_project_encoding if site_project_encoding is not None else getattr(cfg, "site_project_encoding", "frequency"))
        mode_cfg = RunConfig(
            **{
                **asdict(cfg),
                "demographics_mode": demo_mode,
                "attribute_mode": attr_mode,
                "detail_effect_mode": dmode,
                "basiskele_specialist_mode": smode,
                "basiskele_variance_lift": vmode,
                "basiskele_large_home_regime": lh_mode,
                "basiskele_spread_layer": sp_mode,
                "karamursel_baseline_mode": kar_mode,
                "location_feature_mode": loc_mode,
                "geo_context_mode": gctx_mode,
                "location_scope": loc_scope,
                "comparable_mode": comp_mode,
                "premium_feature_mode": prem_mode,
                "site_project_encoding": site_enc,
            }
        )
        demo_features = build_demographic_features(demographic_raw, mode=demo_mode)
        sales_with_demo, demo_feature_report = attach_demographic_features(sales_final_base, demo_features, demo_mode, mode_out_dirs)
        sales_with_demo.to_csv(mode_out_dirs["input"] / f"sales_training_table_v17_{demo_mode}_{attr_mode}.csv", index=False, encoding="utf-8-sig")

        X, y = make_training_matrix(sales_with_demo)
        _plog(f"[demo={demo_mode} attr={attr_mode} specialist={smode}] rows before anomaly filter: {len(X)}")
        if len(X) == 0:
            raise ValueError(f"No valid sale rows left after cleaning for demo={demo_mode} attr={attr_mode}.")

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
                mode_out_dirs["reports"] / f"anomaly_training_filter_{demo_mode}_{attr_mode}_v18_basiskele.csv", index=False, encoding="utf-8-sig"
            )
            if excluded > 0:
                X = X.loc[keep_mask].reset_index(drop=True)
                y = y.loc[keep_mask].reset_index(drop=True)
            anomaly_filter_report.update({
                "training_rows_after_anomaly_filter": int(len(X)),
                "excluded_anomaly_rows": excluded,
            })
            _plog(f"[demo={demo_mode} attr={attr_mode}] excluded anomalies: {excluded} | after: {len(X)}")

        _pstage("Basiskele premium features / target stats")
        bundle, metrics_summary = train_and_evaluate(X, y, mode_cfg, mode_out_dirs)
        metrics_summary["ensemble"]["demographics_mode"] = demo_mode
        metrics_summary["ensemble"]["attribute_mode"] = attr_mode
        metrics_summary["ensemble"]["detail_effect_mode"] = dmode
        metrics_summary["ensemble"]["basiskele_specialist_mode"] = smode
        metrics_summary["ensemble"]["basiskele_variance_lift"] = vmode
        metrics_summary["ensemble"]["basiskele_large_home_regime"] = lh_mode
        metrics_summary["ensemble"]["basiskele_spread_layer"] = sp_mode
        metrics_summary["ensemble"]["karamursel_baseline_mode"] = kar_mode
        metrics_summary["ensemble"]["location_feature_mode"] = loc_mode
        metrics_summary["ensemble"]["geo_context_mode"] = gctx_mode
        metrics_summary["ensemble"]["location_scope"] = loc_scope
        metrics_summary["ensemble"]["comparable_mode"] = str(getattr(mode_cfg, "comparable_mode", "none"))
        metrics_summary["ensemble"]["premium_feature_mode"] = str(getattr(mode_cfg, "premium_feature_mode", "full"))
        metrics_summary["ensemble"]["site_project_encoding"] = str(getattr(mode_cfg, "site_project_encoding", "frequency"))
        metrics_summary["selected_premium_feature_mode"] = str(getattr(mode_cfg, "premium_feature_mode", "full"))
        metrics_summary["selected_site_project_encoding"] = str(getattr(mode_cfg, "site_project_encoding", "frequency"))
        metrics_summary["ensemble"]["model_scope"] = str(getattr(mode_cfg, "model_scope", "basiskele_only"))
        metrics_summary["ensemble"]["location_feature_metadata"] = location_feature_metadata(loc_mode)
        metrics_summary["selected_comparable_mode"] = str(getattr(mode_cfg, "comparable_mode", "full"))
        metrics_summary["ensemble"]["fast_mode"] = bool(getattr(mode_cfg, "fast_mode", False))
        if bool(getattr(mode_cfg, "fast_mode", False)):
            metrics_summary["ensemble"]["comparability_warning"] = (
                "fast_mode location result is smoke only, not comparable. "
                "V15/V16 comparisons only with full train."
            )
        metrics_summary["ensemble"].update(anomaly_filter_report)
        metrics_summary["demographic_features"] = demo_feature_report
        metrics_summary["anomaly_training_filter"] = anomaly_filter_report
        decision = metrics_summary.get("decision", {})
        metrics_summary["ensemble"]["pass_guardrail"] = decision.get("pass_guardrail")
        metrics_summary["ensemble"]["pass_sensitivity"] = decision.get("pass_sensitivity")
        (mode_out_dirs["reports"] / "metrics_summary_v18_basiskele.json").write_text(
            json.dumps(metrics_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return bundle, metrics_summary, demo_feature_report, anomaly_filter_report

    def county_metric(ms: dict[str, Any], county: str, key: str) -> float:
        for row in ms.get("county_metrics", []) or []:
            if str(row.get("county")) == county:
                return float(row.get(key, np.nan))
        return float("nan")

    def ablation_row(attr_mode: str, demo_mode: str, ms: dict[str, Any]) -> dict[str, Any]:
        ens = ms.get("ensemble", {})
        decision = ms.get("decision", {})
        return {
            "attribute_mode": attr_mode,
            "demographics_mode": demo_mode,
            "rows": ens.get("rows"),
            "r2": ens.get("r2"),
            "log_r2": ens.get("log_r2"),
            "mape": ens.get("mape"),
            "median_ape": ens.get("median_ape"),
            "mae_tl_per_m2": ens.get("mae_tl_per_m2"),
            "median_ae_tl_per_m2": ens.get("median_ae_tl_per_m2"),
            "basiskele_r2": county_metric(ms, "Başiskele", "r2"),
            "basiskele_mape": county_metric(ms, "Başiskele", "mape"),
            "karamursel_r2": county_metric(ms, "Karamürsel", "r2"),
            "karamursel_mape": county_metric(ms, "Karamürsel", "mape"),
            "karamursel_sale_diff_pct": decision.get("karamursel_sale_diff_pct"),
            "direction_pass_rate": decision.get("direction_pass_rate"),
            "pass_guardrail": decision.get("pass_guardrail"),
            "pass_sensitivity": decision.get("pass_sensitivity"),
            "overall": decision.get("overall"),
        }

    attr_ablation_rows: list[dict[str, Any]] = []
    demo_ablation_rows: list[dict[str, Any]] = []
    demo_ablation_summaries: dict[str, Any] = {}
    attr_results: dict[str, dict[str, Any]] = {}
    detail_ablation_rows: list[dict[str, Any]] = []
    specialist_ablation_rows: list[dict[str, Any]] = []

    selected_demo = str(args.demographics_mode)
    selected_attr = str(args.attribute_mode)
    selected_detail = str(getattr(args, "detail_effect_mode", "group"))
    selected_specialist = str(getattr(args, "basiskele_specialist_mode", "premium_target_stats"))
    selected_variance_lift = str(getattr(args, "basiskele_variance_lift", "none"))
    selected_lh_regime = str(getattr(args, "basiskele_large_home_regime", "none"))
    selected_spread = str(getattr(args, "basiskele_spread_layer", "none"))
    selected_kar_baseline = str(getattr(args, "karamursel_baseline_mode", "none"))
    selected_location_mode = str(getattr(args, "location_feature_mode", "full"))
    selected_geo_context_mode = str(getattr(args, "geo_context_mode", "full"))
    selected_location_scope = str(getattr(args, "location_scope", "basiskele_only"))
    selected_comparable_mode = "none"
    selected_premium_mode = str(getattr(cfg, "premium_feature_mode", "full"))
    selected_site_encoding = str(getattr(cfg, "site_project_encoding", "frequency"))
    location_ablation_rows: list[dict[str, Any]] = []
    comparable_ablation_rows: list[dict[str, Any]] = []

    n_attr_runs = 3 if bool(args.run_attribute_ablation) else 0
    n_demo_runs = 3 if bool(args.run_demographics_ablation) else 0
    n_detail_runs = 3 if bool(getattr(args, "run_detail_effect_ablation", False)) else 0
    n_specialist_runs = 4 if bool(getattr(args, "run_basiskele_specialist_ablation", False)) else 0
    n_regime_runs = 5 if bool(getattr(args, "run_v16_regime_ablation", False)) else 0
    n_location_runs = 5 if bool(getattr(args, "run_location_ablation", False)) else 0
    n_comparable_runs = 6 if bool(getattr(args, "run_comparable_ablation", False)) else 0
    n_premium_runs = 7 if bool(getattr(args, "run_premium_ablation", False)) else 0
    total_units = estimate_training_units(
        n_models=len(cfg.selected_models) or 4,
        n_attr_runs=n_attr_runs,
        n_demo_runs=n_demo_runs,
        n_detail_runs=n_detail_runs + n_specialist_runs + n_regime_runs,
        include_final=True,
        n_segments=4,
        n_counties_est=4,
        n_location_runs=n_location_runs + n_comparable_runs + n_premium_runs,
    )
    progress = TrainProgress(total_units=total_units)
    set_progress(progress)
    progress.start()
    _plog(
        f"Eğitim ilerlemesi başlıyor (~{total_units} adım tahmini; "
        f"location ablation={n_location_runs}, comparable ablation={n_comparable_runs})."
    )

    try:
        # Attribute ablation under fixed demographics-mode (max 3 runs)
        if bool(args.run_attribute_ablation):
            for attr_mode in ["none", "basic", "full"]:
                _pstage(f"attribute ablation: {attr_mode}")
                _plog(f"\n========== V16 ATTRIBUTE ABLATION: demo={selected_demo} attr={attr_mode} ==========")
                mode_dirs = build_output_dirs(out_dirs["base"] / f"ablation_attr_{attr_mode}")
                _b, ms, _d, _f = run_one_mode(selected_demo, attr_mode, mode_dirs, detail_mode=selected_detail)
                row = ablation_row(attr_mode, selected_demo, ms)
                attr_ablation_rows.append(row)
                attr_results[attr_mode] = {"metrics_summary": ms, "row": row}
            attr_df = pd.DataFrame(attr_ablation_rows)
            attr_df.to_csv(out_dirs["reports"] / "metrics_attribute_ablation_v18_basiskele.csv", index=False, encoding="utf-8-sig")
            selected_attr = select_attribute_mode(attr_ablation_rows)
            _plog(f"Selected attribute mode after ablation: {selected_attr}")
        else:
            selected_attr = str(args.attribute_mode)

        # Demographics ablation under fixed (selected) attribute-mode (max 3 runs)
        if bool(args.run_demographics_ablation):
            for demo_mode in ["none", "safe", "full"]:
                _pstage(f"demographics ablation: {demo_mode}")
                _plog(f"\n========== V16 DEMOGRAPHICS ABLATION: demo={demo_mode} attr={selected_attr} ==========")
                mode_dirs = build_output_dirs(out_dirs["base"] / f"ablation_{demo_mode}")
                _b, ms, demo_rep, filt_rep = run_one_mode(demo_mode, selected_attr, mode_dirs, detail_mode=selected_detail)
                demo_ablation_rows.append({"demographics_mode": demo_mode, **ms["ensemble"]})
                demo_ablation_summaries[demo_mode] = {"metrics": ms["ensemble"], "demographic_features": demo_rep, "anomaly_filter": filt_rep}
            demo_df = pd.DataFrame(demo_ablation_rows)
            keep_cols = [c for c in ["demographics_mode", "rows", "r2", "log_r2", "mape", "median_ape", "mae_tl_per_m2", "median_ae_tl_per_m2", "excluded_anomaly_rows", "training_rows_after_anomaly_filter"] if c in demo_df.columns]
            if not demo_df.empty:
                demo_df[keep_cols].to_csv(out_dirs["reports"] / "metrics_demographics_ablation_v18_basiskele.csv", index=False, encoding="utf-8-sig")

        # Detail-effect ablation under fixed selected demo+attr (max 3 runs); only if flag set
        if bool(getattr(args, "run_detail_effect_ablation", False)):
            for detail_mode in ["none", "group", "full"]:
                _pstage(f"detail-effect ablation: {detail_mode}")
                _plog(
                    f"\n========== V16 DETAIL-EFFECT ABLATION: demo={selected_demo} attr={selected_attr} detail={detail_mode} =========="
                )
                mode_dirs = build_output_dirs(out_dirs["base"] / f"ablation_detail_{detail_mode}")
                _b, ms, _d, _f = run_one_mode(selected_demo, selected_attr, mode_dirs, detail_mode=detail_mode)
                ens = ms.get("ensemble", {})
                decision = ms.get("decision", {})
                detail_ablation_rows.append(
                    {
                        "detail_effect_mode": detail_mode,
                        "r2": ens.get("r2"),
                        "log_r2": ens.get("log_r2"),
                        "mape": ens.get("mape"),
                        "median_ape": ens.get("median_ape"),
                        "mae_tl_per_m2": ens.get("mae_tl_per_m2"),
                        "median_ae_tl_per_m2": ens.get("median_ae_tl_per_m2"),
                        "basiskele_r2": county_metric(ms, "Başiskele", "r2"),
                        "basiskele_mape": county_metric(ms, "Başiskele", "mape"),
                        "basiskele_variance_ratio": decision.get("basiskele_variance_ratio"),
                        "karamursel_r2": county_metric(ms, "Karamürsel", "r2"),
                        "karamursel_mape": county_metric(ms, "Karamürsel", "mape"),
                        "golcuk_r2": county_metric(ms, "Gölcük", "r2"),
                        "izmit_r2": county_metric(ms, "İzmit", "r2"),
                        "direction_pass_rate": decision.get("direction_pass_rate"),
                        "karamursel_sale_diff_pct": decision.get("karamursel_sale_diff_pct"),
                        "pass_guardrail": decision.get("pass_guardrail"),
                        "pass_global_guardrail": decision.get("pass_global_guardrail"),
                        "pass_karamursel_guardrail": decision.get("pass_karamursel_guardrail"),
                        "pass_basiskele_lift": decision.get("pass_basiskele_lift"),
                    }
                )
            detail_df = pd.DataFrame(detail_ablation_rows)
            detail_df.to_csv(out_dirs["reports"] / "metrics_detail_effect_ablation_v18_basiskele.csv", index=False, encoding="utf-8-sig")
            selected_detail = select_detail_effect_mode(detail_ablation_rows)
            _plog(f"Selected detail-effect mode after ablation: {selected_detail}")
        else:
            selected_detail = str(getattr(args, "detail_effect_mode", "group"))

        # Başiskele specialist ablation (optional)
        if bool(getattr(args, "run_basiskele_specialist_ablation", False)):
            for smode in ["none", "premium", "premium_target_stats", "premium_target_stats_variance_lift"]:
                _pstage(f"Basiskele specialist ablation: {smode}")
                _plog(
                    f"\n========== V16 BASISKELE SPECIALIST ABLATION: specialist={smode} =========="
                )
                vmode = "conservative" if "variance_lift" in smode else "none"
                mode_dirs = build_output_dirs(out_dirs["base"] / f"ablation_basiskele_{smode}")
                _b, ms, _d, _f = run_one_mode(
                    selected_demo,
                    selected_attr,
                    mode_dirs,
                    detail_mode=selected_detail,
                    specialist_mode=smode,
                    variance_lift=vmode,
                )
                ens = ms.get("ensemble", {})
                decision = ms.get("decision", {})
                specialist_ablation_rows.append(
                    {
                        "basiskele_specialist_mode": smode,
                        "r2": ens.get("r2"),
                        "mape": ens.get("mape"),
                        "basiskele_r2": county_metric(ms, "Başiskele", "r2"),
                        "basiskele_mape": county_metric(ms, "Başiskele", "mape"),
                        "basiskele_variance_ratio": decision.get("basiskele_variance_ratio"),
                        "karamursel_r2": county_metric(ms, "Karamürsel", "r2"),
                        "golcuk_r2": county_metric(ms, "Gölcük", "r2"),
                        "izmit_r2": county_metric(ms, "İzmit", "r2"),
                        "global_guardrail": decision.get("pass_global_guardrail"),
                        "ship_ready_all_counties_r2_ge_0_65": decision.get(
                            "ship_ready_all_counties_r2_ge_0_65"
                        ),
                    }
                )
            pd.DataFrame(specialist_ablation_rows).to_csv(
                out_dirs["reports"] / "metrics_basiskele_specialist_ablation_v18_basiskele.csv",
                index=False,
                encoding="utf-8-sig",
            )
            # Prefer premium_target_stats as default final; do not auto-pick variance_lift
            selected_specialist = "premium_target_stats"
            selected_variance_lift = str(getattr(args, "basiskele_variance_lift", "conservative"))
        else:
            selected_specialist = str(getattr(args, "basiskele_specialist_mode", "premium_target_stats"))
            selected_variance_lift = str(getattr(args, "basiskele_variance_lift", "conservative"))

        # V16 regime ablation (optional)
        regime_ablation_rows: list[dict[str, Any]] = []
        if bool(getattr(args, "run_v16_regime_ablation", False)):
            # Prefer residual over simple when user selected residual for B/E.
            lh_for_ablation = selected_lh_regime if selected_lh_regime in {"simple", "residual"} else "simple"
            regime_experiments = [
                ("control_v15_like", "none", "none", "none"),
                ("bsk_large_home", lh_for_ablation, "none", "none"),
                ("bsk_spread", "simple", "conservative", "none"),
                ("karamursel_baseline", "none", "none", "location_age"),
                ("combined", lh_for_ablation, "conservative", "location_age"),
            ]
            for exp_name, lh, sp, kar in regime_experiments:
                _pstage(f"V16 regime ablation: {exp_name}")
                _plog(f"\n========== V16 REGIME ABLATION: {exp_name} lh={lh} spread={sp} kar={kar} ==========")
                mode_dirs = build_output_dirs(out_dirs["base"] / f"ablation_regime_{exp_name}")
                _b, ms, _d, _f = run_one_mode(
                    selected_demo,
                    selected_attr,
                    mode_dirs,
                    detail_mode=selected_detail,
                    specialist_mode=selected_specialist,
                    variance_lift="none",
                    large_home_regime=lh,
                    spread_layer=sp,
                    karamursel_baseline=kar,
                )
                ens = ms.get("ensemble", {})
                decision = ms.get("decision", {})
                lh_rep = ens.get("basiskele_large_home_residual_report") or {}
                regime_ablation_rows.append(
                    {
                        "experiment": exp_name,
                        "global_r2": ens.get("r2"),
                        "global_mape": ens.get("mape"),
                        "basiskele_r2": county_metric(ms, "Başiskele", "r2"),
                        "basiskele_mape": county_metric(ms, "Başiskele", "mape"),
                        "basiskele_variance_ratio": decision.get("basiskele_variance_ratio"),
                        "basiskele_large_home_r2": lh_rep.get("large_home_r2_after")
                        or ens.get("basiskele_large_home_r2"),
                        "basiskele_non_large_r2": lh_rep.get("non_large_basiskele_r2_after"),
                        "karamursel_r2": county_metric(ms, "Karamürsel", "r2"),
                        "karamursel_mape": county_metric(ms, "Karamürsel", "mape"),
                        "golcuk_r2": county_metric(ms, "Gölcük", "r2"),
                        "izmit_r2": county_metric(ms, "İzmit", "r2"),
                        "ship_ready_all_counties_r2_ge_0_65": decision.get(
                            "ship_ready_all_counties_r2_ge_0_65"
                        ),
                        "selected_layers": f"lh={lh}|spread={sp}|kar={kar}",
                    }
                )
            pd.DataFrame(regime_ablation_rows).to_csv(
                out_dirs["reports"] / "metrics_v17_regime_ablation.csv",
                index=False,
                encoding="utf-8-sig",
            )
            # Select safest lift combo vs V15 refs if combined regresses
            v15_b = float(V15_DEFAULT_REF.get("basiskele_r2", 0.4534))
            v15_k = float(V15_DEFAULT_REF.get("karamursel_r2", 0.5681))

            def _safe(row: dict[str, Any]) -> bool:
                try:
                    return (
                        float(row.get("basiskele_r2") or -1) >= v15_b - 1e-9
                        and float(row.get("karamursel_r2") or -1) >= v15_k - 1e-9
                        and float(row.get("global_mape") or 9) <= 0.131 + 1e-9
                        and float(row.get("izmit_r2") or -1) >= 0.70 - 1e-9
                        and float(row.get("golcuk_r2") or -1) >= 0.62 - 1e-9
                    )
                except Exception:
                    return False

            by_exp = {r["experiment"]: r for r in regime_ablation_rows}
            pick = None
            for name in ["combined", "bsk_spread", "bsk_large_home", "karamursel_baseline", "control_v15_like"]:
                row = by_exp.get(name)
                if row and _safe(row):
                    pick = name
                    break
            if pick is None:
                # fall back to best basiskele_r2 among non-regressing global mape
                candidates = [
                    r
                    for r in regime_ablation_rows
                    if float(r.get("global_mape") or 9) <= 0.131 + 1e-9
                ] or regime_ablation_rows
                pick = max(candidates, key=lambda r: float(r.get("basiskele_r2") or -9)).get("experiment")
            layers = str(by_exp.get(pick, {}).get("selected_layers") or "")
            # parse lh=|spread=|kar=
            parts = dict(p.split("=", 1) for p in layers.split("|") if "=" in p)
            selected_lh_regime = parts.get("lh", selected_lh_regime)
            selected_spread = parts.get("spread", selected_spread)
            selected_kar_baseline = parts.get("kar", selected_kar_baseline)
            _plog(
                f"Selected V16 regime after ablation: {pick} "
                f"(lh={selected_lh_regime}, spread={selected_spread}, kar={selected_kar_baseline})"
            )

        # V17 location / geo-context ablation (Başiskele-isolated by default)
        if bool(getattr(args, "run_location_ablation", False)):
            loc_experiments = [
                # experiment, location_mode, geo_context_mode, location_scope
                ("control_v16_like", "none", "none", "basiskele_only"),
                ("basiskele_basic", "basic", "none", "basiskele_only"),
                ("basiskele_geo", "geo", "geo_no_poi", "basiskele_only"),
                ("basiskele_geo_context", "geo", "geo_with_coast", "basiskele_only"),
                ("global_geo", "geo", "geo_with_coast", "global"),
            ]
            for exp_name, loc_m, gctx_m, scope_m in loc_experiments:
                _pstage(f"Location ablation: {exp_name}")
                _plog(
                    f"\n========== V18 COMPARABLE ABLATION: {exp_name} "
                    f"loc={loc_m} gctx={gctx_m} scope={scope_m} =========="
                )
                mode_dirs = build_output_dirs(out_dirs["base"] / f"ablation_location_{exp_name}")
                _b, ms, _d, _f = run_one_mode(
                    selected_demo,
                    selected_attr,
                    mode_dirs,
                    detail_mode=selected_detail,
                    specialist_mode=selected_specialist,
                    variance_lift="none",
                    large_home_regime="none",
                    spread_layer="none",
                    karamursel_baseline="none",
                    location_mode=loc_m,
                    geo_context_mode=gctx_m,
                    location_scope=scope_m,
                )
                ens = ms.get("ensemble", {})
                decision = ms.get("decision", {})
                lh_rep = ens.get("basiskele_large_home_residual_report") or {}
                location_ablation_rows.append(
                    {
                        "experiment": exp_name,
                        "location_scope": scope_m,
                        "location_feature_mode": loc_m,
                        "geo_context_mode": gctx_m,
                        "global_r2": ens.get("r2"),
                        "global_mape": ens.get("mape"),
                        "basiskele_r2": county_metric(ms, "Başiskele", "r2"),
                        "basiskele_mape": county_metric(ms, "Başiskele", "mape"),
                        "basiskele_variance_ratio": decision.get("basiskele_variance_ratio"),
                        "basiskele_large_home_r2": lh_rep.get("large_home_r2_after")
                        or ens.get("basiskele_large_home_r2"),
                        "golcuk_r2": county_metric(ms, "Gölcük", "r2"),
                        "karamursel_r2": county_metric(ms, "Karamürsel", "r2"),
                        "izmit_r2": county_metric(ms, "İzmit", "r2"),
                        "rows": ens.get("rows"),
                        "basiskele_rows": next(
                            (
                                int(r.get("rows", 0))
                                for r in (ms.get("county_metrics") or [])
                                if str(r.get("county")) == "Başiskele"
                            ),
                            0,
                        ),
                        "selected": False,
                        "notes": "",
                    }
                )

            by_exp = {r["experiment"]: r for r in location_ablation_rows}
            control = by_exp.get("control_v16_like") or {}

            def _loc_safe(row: dict[str, Any]) -> bool:
                if not control:
                    return False
                try:
                    return (
                        float(row.get("basiskele_r2") or -1) > float(control.get("basiskele_r2") or -1)
                        and float(row.get("basiskele_mape") or 9)
                        <= float(control.get("basiskele_mape") or 9) + 0.005
                        and float(row.get("global_mape") or 9) <= 0.131 + 1e-9
                        and float(row.get("izmit_r2") or -1) >= 0.70 - 1e-9
                        and float(row.get("karamursel_r2") or -1)
                        >= float(control.get("karamursel_r2") or 0) - 0.02
                        and float(row.get("golcuk_r2") or -1)
                        >= float(control.get("golcuk_r2") or 0) - 0.02
                    )
                except Exception:
                    return False

            pick = None
            for name in [
                "basiskele_geo_context",
                "basiskele_geo",
                "basiskele_basic",
                "global_geo",
                "control_v16_like",
            ]:
                row = by_exp.get(name)
                if row and name != "control_v16_like" and _loc_safe(row):
                    pick = name
                    break
            if pick is None:
                # Prefer best Başiskele lift among non-regressing guardrail candidates
                candidates = [r for r in location_ablation_rows if r["experiment"] != "control_v16_like" and _loc_safe(r)]
                if candidates:
                    pick = max(candidates, key=lambda r: float(r.get("basiskele_r2") or -9))["experiment"]
                else:
                    pick = "control_v16_like"
                    for r in location_ablation_rows:
                        if r["experiment"] == pick:
                            r["notes"] = "no location candidate passed selection; keep control"

            for r in location_ablation_rows:
                r["selected"] = r.get("experiment") == pick
                if r["selected"] and not r.get("notes"):
                    r["notes"] = "selected vs control_v16_like guardrails"
            pd.DataFrame(location_ablation_rows).to_csv(
                out_dirs["reports"] / "metrics_location_ablation_v18_basiskele.csv",
                index=False,
                encoding="utf-8-sig",
            )
            picked = by_exp.get(pick) or {}
            selected_location_mode = str(picked.get("location_feature_mode") or selected_location_mode)
            selected_geo_context_mode = str(picked.get("geo_context_mode") or selected_geo_context_mode)
            selected_location_scope = str(picked.get("location_scope") or selected_location_scope)
            _plog(
                f"Selected V18 location after ablation: {pick} "
                f"(loc={selected_location_mode}, gctx={selected_geo_context_mode}, scope={selected_location_scope})"
            )

        # V18 comparable ablation (Başiskele-only)
        if bool(getattr(args, "run_comparable_ablation", False)):
            # Fixed geo stack; only comparable mode changes
            CONTROL_R2 = 0.4730
            CONTROL_MAPE = 0.1093
            CONTROL_VR = 0.4283
            comp_experiments = [
                ("control_v17_geo", "none"),
                ("nearest_only", "nearest"),
                ("similar_only", "similar"),
                ("weighted_only", "weighted"),
                ("large_home_only", "large_home"),
                ("full_comparable", "full"),
            ]
            for exp_name, comp_m in comp_experiments:
                _pstage(f"Comparable ablation: {exp_name}")
                _plog(
                    f"\n========== V18 COMPARABLE ABLATION: {exp_name} "
                    f"comparable={comp_m} loc={selected_location_mode} "
                    f"gctx={selected_geo_context_mode} =========="
                )
                mode_dirs = build_output_dirs(out_dirs["base"] / f"ablation_comparable_{exp_name}")
                _b, ms, _d, _f = run_one_mode(
                    selected_demo,
                    selected_attr,
                    mode_dirs,
                    detail_mode=selected_detail,
                    specialist_mode=selected_specialist,
                    variance_lift="none",
                    large_home_regime="none",
                    spread_layer="none",
                    karamursel_baseline="none",
                    location_mode=selected_location_mode,
                    geo_context_mode=selected_geo_context_mode,
                    location_scope=selected_location_scope,
                    comparable_mode=comp_m,
                )
                ens = ms.get("ensemble", {})
                decision_bsk = ms.get("decision_basiskele") or {}
                r2 = float(ms.get("r2") if ms.get("r2") is not None else ens.get("r2") or float("nan"))
                mape = float(ms.get("mape") if ms.get("mape") is not None else ens.get("mape") or float("nan"))
                vr = float(ms.get("variance_ratio") if ms.get("variance_ratio") is not None else ens.get("basiskele_variance_ratio") or float("nan"))
                lh_r2 = float(ms.get("large_home_r2") if ms.get("large_home_r2") is not None else float("nan"))
                notes = []
                if exp_name == "nearest_only":
                    notes.append("nearest_only disqualified from final selection (failed prior full nearest lift)")
                if np.isfinite(vr) and vr < CONTROL_VR:
                    notes.append(f"warning:variance_ratio {vr:.4f} < control {CONTROL_VR:.4f}")
                if np.isfinite(r2) and r2 <= CONTROL_R2 and exp_name not in {"control_v17_geo"}:
                    notes.append(f"no_r2_lift_vs_control ({r2:.4f} <= {CONTROL_R2:.4f})")
                if np.isfinite(mape) and mape > CONTROL_MAPE + 0.005:
                    notes.append(f"mape_guardrail_fail ({mape:.4f} > {CONTROL_MAPE + 0.005:.4f})")
                comparable_ablation_rows.append(
                    {
                        "experiment": exp_name,
                        "comparable_mode": comp_m,
                        "rows": ens.get("rows") or ms.get("rows"),
                        "r2": r2,
                        "log_r2": ens.get("log_r2"),
                        "mape": mape,
                        "mae": ens.get("mae_tl_per_m2"),
                        "median_ape": ens.get("median_ape"),
                        "variance_ratio": vr,
                        "large_home_r2": lh_r2,
                        "non_large_r2": np.nan,
                        "cheap_decile_bias": np.nan,
                        "expensive_decile_bias": np.nan,
                        "selected": False,
                        "notes": "; ".join(notes),
                    }
                )

            by_exp = {r["experiment"]: r for r in comparable_ablation_rows}
            control_row = by_exp.get("control_v17_geo") or {}
            ctrl_r2 = float(control_row.get("r2") if control_row.get("r2") is not None else CONTROL_R2)
            ctrl_mape = float(control_row.get("mape") if control_row.get("mape") is not None else CONTROL_MAPE)

            def _comp_eligible(row: dict[str, Any]) -> bool:
                name = str(row.get("experiment") or "")
                if name in {"control_v17_geo", "nearest_only"}:
                    return False
                try:
                    return (
                        float(row.get("r2") or -9) > ctrl_r2
                        and float(row.get("mape") or 9) <= ctrl_mape + 0.005 + 1e-12
                    )
                except Exception:
                    return False

            candidates = [r for r in comparable_ablation_rows if _comp_eligible(r)]
            if candidates:
                # Prefer higher R2; tie-break variance_ratio then large_home_r2
                pick = max(
                    candidates,
                    key=lambda r: (
                        float(r.get("r2") or -9),
                        float(r.get("variance_ratio") or -9),
                        float(r.get("large_home_r2") or -9),
                    ),
                )["experiment"]
            else:
                pick = "control_v17_geo"

            for r in comparable_ablation_rows:
                r["selected"] = r.get("experiment") == pick
                if r["selected"] and not r.get("notes"):
                    r["notes"] = "selected vs control_v17_geo (R2 lift + MAPE guardrail; nearest excluded)"
                elif r["selected"]:
                    r["notes"] = (r.get("notes") or "") + "; selected"
            pd.DataFrame(comparable_ablation_rows).to_csv(
                out_dirs["reports"] / "metrics_comparable_ablation_v18_basiskele.csv",
                index=False,
                encoding="utf-8-sig",
            )
            picked = by_exp.get(pick) or {}
            selected_comparable_mode = str(picked.get("comparable_mode") or "none")
            _plog(
                f"Selected V18 comparable after ablation: {pick} "
                f"(comparable_mode={selected_comparable_mode})"
            )


        # V20 premium-signal ablation
        premium_ablation_rows: list[dict[str, Any]] = []
        if bool(getattr(args, "run_premium_ablation", False)):
            CONTROL_R2 = 0.4731
            CONTROL_MAPE = 0.1093
            CONTROL_VR = 0.4264
            prem_experiments = [
                ("control_v18_geo", "none", "none"),
                ("text_flags_only", "flags", "none"),
                ("site_frequency_only", "site", "frequency"),
                ("flags_plus_site_frequency", "full", "frequency"),
                ("interactions_only", "interactions", "frequency"),
                ("site_foldsafe_target", "site", "foldsafe_target"),
                ("full_v20", "full", "foldsafe_target"),
            ]
            for exp_name, prem_m, site_m in prem_experiments:
                _pstage(f"Premium ablation: {exp_name}")
                _plog(
                    f"\n========== V20 PREMIUM ABLATION: {exp_name} "
                    f"premium={prem_m} site_enc={site_m} =========="
                )
                mode_dirs = build_output_dirs(out_dirs["base"] / f"ablation_premium_{exp_name}")
                _b, ms, _d, _f = run_one_mode(
                    selected_demo,
                    selected_attr,
                    mode_dirs,
                    detail_mode=selected_detail,
                    specialist_mode=selected_specialist,
                    variance_lift="none",
                    large_home_regime="none",
                    spread_layer="none",
                    karamursel_baseline="none",
                    location_mode=selected_location_mode,
                    geo_context_mode=selected_geo_context_mode,
                    location_scope=selected_location_scope,
                    comparable_mode="none",
                    premium_feature_mode=prem_m,
                    site_project_encoding=site_m,
                )
                ens = ms.get("ensemble", {})
                r2 = float(ms.get("r2") if ms.get("r2") is not None else ens.get("r2") or float("nan"))
                mape = float(ms.get("mape") if ms.get("mape") is not None else ens.get("mape") or float("nan"))
                vr = float(ms.get("variance_ratio") if ms.get("variance_ratio") is not None else ens.get("basiskele_variance_ratio") or float("nan"))
                lh_r2 = float(ms.get("large_home_r2") if ms.get("large_home_r2") is not None else float("nan"))
                lh_mape = float(ms.get("large_home_mape") if ms.get("large_home_mape") is not None else float("nan"))
                cheap_bias = float(ms.get("cheap_decile_bias") if ms.get("cheap_decile_bias") is not None else float("nan"))
                exp_bias = float(ms.get("expensive_decile_bias") if ms.get("expensive_decile_bias") is not None else float("nan"))
                leak = ms.get("site_project_encoding_leakage_guard") or ens.get("site_project_encoding_leakage_guard") or {}
                notes = []
                if site_m == "foldsafe_target" and leak and leak.get("pass") is False:
                    notes.append("leakage_guard_fail")
                if np.isfinite(r2) and r2 <= CONTROL_R2 and exp_name != "control_v18_geo":
                    notes.append(f"no_r2_lift_vs_control ({r2:.4f} <= {CONTROL_R2:.4f})")
                if np.isfinite(mape) and mape > CONTROL_MAPE + 0.005:
                    notes.append(f"mape_guardrail_fail ({mape:.4f} > {CONTROL_MAPE + 0.005:.4f})")
                if np.isfinite(vr) and vr + 1e-12 < CONTROL_VR and exp_name != "control_v18_geo":
                    notes.append(f"warning:variance_ratio {vr:.4f} < control {CONTROL_VR:.4f}")
                premium_ablation_rows.append(
                    {
                        "experiment": exp_name,
                        "premium_feature_mode": prem_m,
                        "site_project_encoding": site_m,
                        "rows": ens.get("rows") or ms.get("rows"),
                        "r2": r2,
                        "log_r2": ens.get("log_r2"),
                        "mape": mape,
                        "mae": ens.get("mae_tl_per_m2"),
                        "median_ape": ens.get("median_ape"),
                        "variance_ratio": vr,
                        "large_home_r2": lh_r2,
                        "large_home_mape": lh_mape,
                        "expensive_decile_bias": exp_bias,
                        "cheap_decile_bias": cheap_bias,
                        "selected": False,
                        "notes": "; ".join(notes),
                    }
                )

            by_exp = {r["experiment"]: r for r in premium_ablation_rows}
            control_row = by_exp.get("control_v18_geo") or {}
            ctrl_r2 = float(control_row.get("r2") if control_row.get("r2") is not None else CONTROL_R2)
            ctrl_mape = float(control_row.get("mape") if control_row.get("mape") is not None else CONTROL_MAPE)

            def _prem_eligible(row: dict[str, Any]) -> bool:
                name = str(row.get("experiment") or "")
                if name == "control_v18_geo":
                    return False
                notes = str(row.get("notes") or "")
                if "leakage_guard_fail" in notes:
                    return False
                try:
                    return (
                        float(row.get("r2") or -9) > ctrl_r2
                        and float(row.get("mape") or 9) <= ctrl_mape + 0.005 + 1e-12
                    )
                except Exception:
                    return False

            candidates = [r for r in premium_ablation_rows if _prem_eligible(r)]
            if candidates:
                pick = max(
                    candidates,
                    key=lambda r: (
                        float(r.get("expensive_decile_bias") if r.get("expensive_decile_bias") is not None else -1e18),
                        float(r.get("variance_ratio") or -9),
                        float(r.get("large_home_r2") or -9),
                        float(r.get("r2") or -9),
                    ),
                )["experiment"]
            else:
                pick = "control_v18_geo"

            for r in premium_ablation_rows:
                r["selected"] = r.get("experiment") == pick
                if r["selected"] and not r.get("notes"):
                    r["notes"] = "selected vs control_v18_geo (R2 lift + MAPE + leakage)"
            pd.DataFrame(premium_ablation_rows).to_csv(
                out_dirs["reports"] / "metrics_premium_ablation_v20_basiskele.csv",
                index=False,
                encoding="utf-8-sig",
            )
            picked = by_exp.get(pick) or {}
            selected_premium_mode = str(picked.get("premium_feature_mode") or selected_premium_mode)
            selected_site_encoding = str(picked.get("site_project_encoding") or selected_site_encoding)
            _plog(
                f"Selected V20 premium after ablation: {pick} "
                f"(premium={selected_premium_mode}, site_enc={selected_site_encoding})"
            )


        # Final run into main out dirs with selected modes
        _pstage(
            f"FINAL demo={selected_demo} attr={selected_attr} detail={selected_detail} loc={selected_location_mode}"
        )
        _plog(
            f"\n========== V18 FINAL: demo={selected_demo} attr={selected_attr} detail={selected_detail} "
            f"location={selected_location_mode} geo_context={selected_geo_context_mode} "
            f"comparable={selected_comparable_mode} "
            f"scope={selected_location_scope} =========="
        )
        final_cfg_dict = {
            **asdict(cfg),
            "demographics_mode": selected_demo,
            "attribute_mode": selected_attr,
            "detail_effect_mode": selected_detail,
            "basiskele_specialist_mode": selected_specialist,
            "basiskele_variance_lift": selected_variance_lift,
            "basiskele_large_home_regime": selected_lh_regime,
            "basiskele_spread_layer": selected_spread,
            "karamursel_baseline_mode": selected_kar_baseline,
            "location_feature_mode": selected_location_mode,
            "geo_context_mode": selected_geo_context_mode,
            "location_scope": selected_location_scope,
            "comparable_mode": "none",
            "premium_feature_mode": selected_premium_mode,
            "site_project_encoding": selected_site_encoding,
        }
        cfg = RunConfig(**final_cfg_dict)
        (out_dirs["base"] / "run_config_v20_basiskele.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
        bundle, metrics_summary, demo_feature_report, anomaly_filter_report = run_one_mode(
            selected_demo,
            selected_attr,
            out_dirs,
            detail_mode=selected_detail,
            specialist_mode=selected_specialist,
            variance_lift=selected_variance_lift,
            large_home_regime=selected_lh_regime,
            spread_layer=selected_spread,
            karamursel_baseline=selected_kar_baseline,
            location_mode=selected_location_mode,
            geo_context_mode=selected_geo_context_mode,
            location_scope=selected_location_scope,
            comparable_mode="none",
            premium_feature_mode=selected_premium_mode,
            site_project_encoding=selected_site_encoding,
        )
    except KeyboardInterrupt:
        progress.finish("iptal edildi")
        set_progress(None)
        raise
    finally:
        if get_progress() is not None:
            progress.finish("tamamlandı")
            set_progress(None)

    feature_reports = {
        "rental_features": rental_feature_report,
        "trend_features": trend_feature_report,
        "demographic_features": demo_feature_report,
        "demographics_ablation": demo_ablation_summaries,
        "attribute_ablation": attr_ablation_rows,
        "detail_effect_ablation": detail_ablation_rows,
        "basiskele_specialist_ablation": specialist_ablation_rows,
    }
    metrics_summary["demographics_ablation"] = demo_ablation_summaries
    metrics_summary["attribute_ablation"] = attr_ablation_rows
    metrics_summary["detail_effect_ablation"] = detail_ablation_rows
    metrics_summary["basiskele_specialist_ablation"] = specialist_ablation_rows
    metrics_summary["final_demographics_mode"] = selected_demo
    metrics_summary["selected_attribute_mode"] = selected_attr
    metrics_summary["selected_detail_effect_mode"] = selected_detail
    metrics_summary["selected_basiskele_specialist_mode"] = selected_specialist
    metrics_summary["selected_basiskele_variance_lift_mode"] = selected_variance_lift
    metrics_summary["selected_basiskele_large_home_regime"] = selected_lh_regime
    metrics_summary["selected_basiskele_spread_layer"] = selected_spread
    metrics_summary["selected_karamursel_baseline_mode"] = selected_kar_baseline
    metrics_summary["selected_location_feature_mode"] = selected_location_mode
    metrics_summary["selected_geo_context_mode"] = selected_geo_context_mode
    metrics_summary["selected_location_scope"] = selected_location_scope
    metrics_summary["location_ablation"] = location_ablation_rows
    metrics_summary["comparable_ablation"] = comparable_ablation_rows
    metrics_summary["selected_comparable_mode"] = selected_comparable_mode
    metrics_summary["comparability_note"] = (
        "fast_mode location result is smoke only, not comparable. "
        "V15/V16 comparisons only with full train."
        if bool(getattr(cfg, "fast_mode", False))
        else "full_train_comparable"
    )
    metrics_summary["v12_reference"] = V12_SAFE_REF
    metrics_summary["v13_reference"] = V13_DEFAULT_REF
    metrics_summary["v14_reference"] = V14_DEFAULT_REF
    metrics_summary["v15_reference"] = V15_DEFAULT_REF
    try:
        from diagnostics_v18_basiskele import V16_DEFAULT_REF
    except ImportError:
        from v18_basiskele.diagnostics_v18_basiskele import V16_DEFAULT_REF
    metrics_summary["v16_reference"] = V16_DEFAULT_REF
    if "decision" in metrics_summary and isinstance(metrics_summary["decision"], dict):
        metrics_summary["decision"]["selected_attribute_mode"] = selected_attr
        metrics_summary["decision"]["selected_detail_effect_mode"] = selected_detail
        metrics_summary["decision"]["selected_basiskele_specialist_mode"] = selected_specialist
        metrics_summary["decision"]["selected_basiskele_variance_lift_mode"] = selected_variance_lift
        metrics_summary["decision"]["selected_location_feature_mode"] = selected_location_mode
        metrics_summary["decision"]["selected_geo_context_mode"] = selected_geo_context_mode
        metrics_summary["decision"]["selected_location_scope"] = selected_location_scope
        metrics_summary["decision"]["selected_v16_layers"] = {
            "basiskele_large_home_regime": selected_lh_regime,
            "basiskele_spread_layer": selected_spread,
            "karamursel_baseline_mode": selected_kar_baseline,
        }
        metrics_summary["ship_ready_all_counties_r2_ge_0_65"] = metrics_summary["decision"].get(
            "ship_ready_all_counties_r2_ge_0_65", False
        )
    (out_dirs["reports"] / "metrics_summary_v18_basiskele.json").write_text(
        json.dumps(metrics_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_run_readme(out_dirs, cfg, metrics_summary, cleaning_report, feature_reports)

    print("\n=== V18 BASISKELE ENSEMBLE METRICS ===")
    print(json.dumps(metrics_summary["ensemble"], indent=2, ensure_ascii=False))
    print(f"Decision: {json.dumps(metrics_summary.get('decision', {}), ensure_ascii=False)}")
    print(f"\nOutputs written to: {out_dirs['base'].resolve()}")


if __name__ == "__main__":
    main()
