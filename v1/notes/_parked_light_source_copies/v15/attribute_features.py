"""V15 attribute quality + fold-safe effect helpers (app-safe only)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# V12 safe final reference (guardrail)
V12_SAFE_REF = {
    "r2": 0.6739332530409017,
    "log_r2": 0.697737799337622,
    "mape": 0.12937468889809503,
    "mae_tl_per_m2": 4879.044334479972,
}

ATTRIBUTE_BASIC_NUMERIC_FEATURES = [
    "attr_is_new_building",
    "attr_is_young_building",
    "attr_is_mid_age_building",
    "attr_is_old_building",
    "attr_building_age_score",
    "attr_building_age_inverse",
    "attr_floor_ratio",
    "attr_is_ground_or_below",
    "attr_is_top_floor",
    "attr_is_middle_floor",
    "attr_floor_quality_score",
    "attr_net_gross_ratio",
    "attr_net_gross_good",
    "attr_net_gross_bad",
    "attr_m2_per_room",
    "attr_compactness_score",
    "attr_has_elevator",
    "attr_has_parking",
    "attr_has_balcony",
    "attr_is_furnished",
    "attr_is_site_inside",
    "attr_credit_eligible",
    "attr_amenity_score",
    "attr_heating_quality_score",
    "attr_has_natural_gas_heating",
    "attr_has_premium_heating",
    "attr_has_poor_heating",
    "attr_total_quality_score",
    "attr_quality_x_location_baseline",
    "attr_quality_x_demo_income",
    "attr_quality_x_county_income",
    "attr_quality_x_district_median",
    "attr_new_building_x_site_inside",
    "attr_old_building_x_no_elevator",
    "attr_large_home_x_quality",
    "attr_compact_home_x_quality",
]

ATTRIBUTE_EFFECT_NUMERIC_FEATURES = [
    "attr_effect_building_age_bucket",
    "attr_effect_elevator",
    "attr_effect_site_inside",
    "attr_effect_parking",
    "attr_effect_heating_quality_bucket",
    "attr_effect_total_quality_bucket",
    "attr_county_new_building_premium",
    "attr_county_old_building_discount",
    "attr_county_elevator_premium",
    "attr_county_site_inside_premium",
    "attr_county_parking_premium",
    "attr_county_good_heating_premium",
    "attr_county_high_quality_premium",
]

AMENITY_WEIGHTS = {
    "attr_has_elevator": 0.25,
    "attr_has_parking": 0.25,
    "attr_has_balcony": 0.15,
    "attr_is_site_inside": 0.25,
    "attr_credit_eligible": 0.10,
}

HEATING_SCORE_MAP = {
    "yerden isitma": 1.00,
    "merkezi pay olcer": 0.92,
    "merkezi (pay olcer)": 0.92,
    "merkezi": 0.90,
    "kombi dogalgaz": 0.85,
    "kombi (dogalgaz)": 0.85,
    "kombi elektrik": 0.70,
    "kombi (elektrik)": 0.70,
    "kat kaloriferi": 0.65,
    "dogalgaz sobasi": 0.60,
    "klima": 0.45,
    "vrv": 0.55,
    "soba": 0.25,
    "yok": 0.05,
    "isitma yok": 0.05,
}


def normalize_text_simple(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).replace("\u00a0", " ").strip().lower()
    repl = {"ı": "i", "ğ": "g", "ü": "u", "ş": "s", "ö": "o", "ç": "c", "İ": "i", "Ğ": "g", "Ü": "u", "Ş": "s", "Ö": "o", "Ç": "c"}
    for a, b in repl.items():
        s = s.replace(a, b)
    return " ".join(s.split())


def parse_yes_no_flag(x: Any) -> float:
    """Return 1/0/NaN from Turkish amenity-like strings."""
    if pd.isna(x):
        return np.nan
    s = normalize_text_simple(x)
    if not s or s in {"missing", "nan", "none", "null", "belirtilmemis", "-"}:
        return np.nan
    yes_tokens = {"var", "evet", "true", "1", "yes", "mevcut", "bulunuyor"}
    no_tokens = {"yok", "hayir", "false", "0", "no", "degil"}
    # parking often "Açık Otopark" / "Kapalı Otopark"
    if "otopark" in s or "parking" in s:
        if any(t in s for t in no_tokens) or s in {"yok"}:
            return 0.0
        return 1.0
    if s in yes_tokens or any(s.startswith(t) for t in yes_tokens):
        return 1.0
    if s in no_tokens:
        return 0.0
    if "var" in s and "yok" not in s:
        return 1.0
    if "yok" in s or "hayir" in s:
        return 0.0
    return np.nan


def heating_quality_score(x: Any) -> float:
    s = normalize_text_simple(x)
    if not s or s in {"missing", "nan", "none", "null", "belirtilmemis"}:
        return 0.50
    if s in HEATING_SCORE_MAP:
        return float(HEATING_SCORE_MAP[s])
    for key, score in HEATING_SCORE_MAP.items():
        if key in s:
            return float(score)
    return 0.50


def building_age_score(age: float) -> float:
    if pd.isna(age):
        return 0.50
    a = float(age)
    if a <= 4:
        return 1.00
    if a <= 10:
        return 0.80
    if a <= 20:
        return 0.55
    if a <= 30:
        return 0.30
    return 0.10


def floor_quality_score(floor: float, total: float, ratio: float) -> float:
    if pd.isna(floor):
        return 0.50
    if float(floor) <= 0:
        return 0.20
    if pd.notna(total) and float(floor) >= float(total):
        return 0.45
    if pd.notna(ratio) and 0.25 <= float(ratio) <= 0.80:
        return 1.00
    if pd.notna(ratio) and float(ratio) < 0.25:
        return 0.70
    return 0.70


def compactness_score(m2_per_room: float) -> float:
    if pd.isna(m2_per_room):
        return 0.50
    v = float(m2_per_room)
    if v < 22:
        return 0.40
    if v <= 35:
        return 1.00
    if v <= 50:
        return 0.80
    return 0.55


def room_count_total(room_count: Any) -> float:
    if pd.isna(room_count):
        return np.nan
    s = str(room_count).replace(" ", "").lower()
    m = __import__("re").search(r"(\d+)\+(\d+)", s)
    if m:
        return float(m.group(1)) + float(m.group(2))
    m = __import__("re").search(r"(\d+)", s)
    if m:
        return float(m.group(1))
    return np.nan


def safe_divide_series(num: pd.Series, den: pd.Series) -> pd.Series:
    out = pd.to_numeric(num, errors="coerce") / pd.to_numeric(den, errors="coerce").replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def weighted_score(parts: dict[str, tuple[float, float]]) -> float:
    """parts: name -> (value, weight); skip NaN values; if all missing return 0.5."""
    num = 0.0
    den = 0.0
    for val, w in parts.values():
        if pd.isna(val):
            continue
        num += float(val) * float(w)
        den += float(w)
    if den <= 0:
        return 0.50
    return float(np.clip(num / den, 0.0, 1.0))


def total_quality_bucket(score: float) -> str:
    if pd.isna(score):
        return "mid"
    s = float(score)
    if s <= 0.35:
        return "low"
    if s <= 0.65:
        return "mid"
    return "high"


def heating_quality_bucket(score: float) -> str:
    if pd.isna(score) or abs(float(score) - 0.5) < 1e-9:
        return "unknown"
    s = float(score)
    if s >= 0.85:
        return "premium"
    if s >= 0.60:
        return "good"
    if s >= 0.40:
        return "mid"
    return "poor"


def building_age_bucket(age: float) -> str:
    if pd.isna(age):
        return "unknown"
    a = float(age)
    if a <= 4:
        return "new"
    if a <= 10:
        return "young"
    if a <= 25:
        return "mid"
    return "old"


def get_attribute_feature_names(attribute_mode: str) -> list[str]:
    mode = str(attribute_mode or "full").lower()
    if mode == "none":
        return []
    if mode == "basic":
        return list(ATTRIBUTE_BASIC_NUMERIC_FEATURES)
    return list(ATTRIBUTE_BASIC_NUMERIC_FEATURES) + list(ATTRIBUTE_EFFECT_NUMERIC_FEATURES)


def add_attribute_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic app-safe attribute quality features."""
    out = df.copy()
    age = pd.to_numeric(out.get("building_age", np.nan), errors="coerce")
    floor = pd.to_numeric(out.get("floor_num", np.nan), errors="coerce")
    total = pd.to_numeric(out.get("total_floors", np.nan), errors="coerce")
    gross = pd.to_numeric(out.get("gross_m2", np.nan), errors="coerce")
    net = pd.to_numeric(out.get("net_m2", np.nan), errors="coerce")

    out["attr_is_new_building"] = (age <= 4).astype(float).where(age.notna(), np.nan)
    out["attr_is_young_building"] = ((age >= 5) & (age <= 10)).astype(float).where(age.notna(), np.nan)
    out["attr_is_mid_age_building"] = ((age >= 11) & (age <= 25)).astype(float).where(age.notna(), np.nan)
    out["attr_is_old_building"] = (age >= 26).astype(float).where(age.notna(), np.nan)
    out["attr_building_age_score"] = age.map(building_age_score)
    out["attr_building_age_inverse"] = 1.0 / (1.0 + age.clip(lower=0))

    ratio = safe_divide_series(floor, total)
    out["attr_floor_ratio"] = ratio
    out["attr_is_ground_or_below"] = (floor <= 0).astype(float).where(floor.notna(), np.nan)
    out["attr_is_top_floor"] = ((total.notna()) & (floor.notna()) & (floor == total)).astype(float)
    out["attr_is_middle_floor"] = ((ratio >= 0.25) & (ratio <= 0.80)).astype(float).where(ratio.notna(), np.nan)
    out["attr_floor_quality_score"] = [
        floor_quality_score(f, t, r) for f, t, r in zip(floor, total, ratio)
    ]

    ngr = safe_divide_series(net, gross).clip(0.2, 1.2)
    out["attr_net_gross_ratio"] = ngr
    out["attr_net_gross_good"] = (ngr >= 0.80).astype(float).where(ngr.notna(), np.nan)
    out["attr_net_gross_bad"] = (ngr < 0.70).astype(float).where(ngr.notna(), np.nan)

    rooms_total = out.get("room_count", pd.Series(np.nan, index=out.index)).map(room_count_total)
    if "total_room_score" in out.columns:
        rooms_total = rooms_total.where(rooms_total.notna(), pd.to_numeric(out["total_room_score"], errors="coerce"))
    out["attr_m2_per_room"] = safe_divide_series(gross, rooms_total)
    out["attr_compactness_score"] = out["attr_m2_per_room"].map(compactness_score)

    out["attr_has_elevator"] = out.get("elevator", pd.Series(np.nan, index=out.index)).map(parse_yes_no_flag)
    out["attr_has_parking"] = out.get("parking", pd.Series(np.nan, index=out.index)).map(parse_yes_no_flag)
    out["attr_has_balcony"] = out.get("balcony", pd.Series(np.nan, index=out.index)).map(parse_yes_no_flag)
    out["attr_is_furnished"] = out.get("furnished", pd.Series(np.nan, index=out.index)).map(parse_yes_no_flag)
    out["attr_is_site_inside"] = out.get("site_inside", pd.Series(np.nan, index=out.index)).map(parse_yes_no_flag)
    out["attr_credit_eligible"] = out.get("credit_eligible", pd.Series(np.nan, index=out.index)).map(parse_yes_no_flag)

    amenity_rows = []
    for i in out.index:
        parts = {
            k: (out.at[i, k] if k in out.columns else np.nan, w)
            for k, w in AMENITY_WEIGHTS.items()
        }
        amenity_rows.append(weighted_score(parts))
    out["attr_amenity_score"] = amenity_rows

    heat = out.get("heating", pd.Series(np.nan, index=out.index))
    out["attr_heating_quality_score"] = heat.map(heating_quality_score)
    hs = out["attr_heating_quality_score"]
    out["attr_has_natural_gas_heating"] = (
        heat.map(normalize_text_simple).str.contains("dogalgaz|kombi|merkezi|yerden", regex=True).astype(float)
    )
    out["attr_has_premium_heating"] = (hs >= 0.85).astype(float)
    out["attr_has_poor_heating"] = (hs <= 0.30).astype(float)

    tq = []
    for i in out.index:
        parts = {
            "age": (out.at[i, "attr_building_age_score"], 0.30),
            "floor": (out.at[i, "attr_floor_quality_score"], 0.20),
            "amenity": (out.at[i, "attr_amenity_score"], 0.20),
            "heat": (out.at[i, "attr_heating_quality_score"], 0.15),
            "compact": (out.at[i, "attr_compactness_score"], 0.15),
        }
        tq.append(weighted_score(parts))
    out["attr_total_quality_score"] = np.clip(tq, 0.0, 1.0)

    # Interactions available at this stage
    baseline = pd.to_numeric(out.get("location_baseline_m2", np.nan), errors="coerce")
    demo_income = pd.to_numeric(out.get("demo_per_capita_income_try", np.nan), errors="coerce")
    county_income = pd.to_numeric(
        out.get("county_demo_per_capita_income_try_median", out.get("county_demo_per_capita_income_median", np.nan)),
        errors="coerce",
    )
    q = pd.to_numeric(out["attr_total_quality_score"], errors="coerce")
    out["attr_quality_x_location_baseline"] = q * baseline
    out["attr_quality_x_demo_income"] = q * demo_income
    out["attr_quality_x_county_income"] = q * county_income
    # district median filled later by AttributeInteractionCompleter
    if "attr_quality_x_district_median" not in out.columns:
        out["attr_quality_x_district_median"] = np.nan

    elev = pd.to_numeric(out["attr_has_elevator"], errors="coerce")
    site = pd.to_numeric(out["attr_is_site_inside"], errors="coerce")
    out["attr_new_building_x_site_inside"] = pd.to_numeric(out["attr_is_new_building"], errors="coerce") * site
    out["attr_old_building_x_no_elevator"] = pd.to_numeric(out["attr_is_old_building"], errors="coerce") * (1.0 - elev.fillna(0.5))
    out["attr_large_home_x_quality"] = (gross >= 150).astype(float).where(gross.notna(), np.nan) * q
    out["attr_compact_home_x_quality"] = (gross <= 85).astype(float).where(gross.notna(), np.nan) * q

    # helper buckets for effect encoder
    out["attr_building_age_bucket"] = age.map(building_age_bucket)
    out["attr_heating_quality_bucket"] = hs.map(heating_quality_bucket)
    out["attr_total_quality_bucket"] = q.map(total_quality_bucket)
    return out


class AttributeQualityAdder(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_mode: str = "full"):
        self.attribute_mode = attribute_mode

    def fit(self, X: pd.DataFrame, y: Any = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        mode = str(self.attribute_mode or "none").lower()
        if mode == "none":
            return X.copy()
        return add_attribute_quality_features(X)


class AttributeInteractionCompleter(BaseEstimator, TransformerMixin):
    """Fill interactions that depend on target-stat columns."""

    def fit(self, X: pd.DataFrame, y: Any = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "attr_total_quality_score" not in df.columns:
            return df
        q = pd.to_numeric(df["attr_total_quality_score"], errors="coerce")
        dist = pd.to_numeric(df.get("district_target_median", np.nan), errors="coerce")
        df["attr_quality_x_district_median"] = q * dist
        baseline = pd.to_numeric(df.get("location_baseline_m2", np.nan), errors="coerce")
        if "attr_quality_x_location_baseline" in df.columns:
            missing = pd.to_numeric(df["attr_quality_x_location_baseline"], errors="coerce").isna()
            df.loc[missing, "attr_quality_x_location_baseline"] = (q * baseline)[missing]
        return df


class AttributeEffectEncoder(BaseEstimator, TransformerMixin):
    """Fold-safe smoothed residual premiums by county × attribute group.

    In residual target_mode, y is already log(price)-log(baseline).
    """

    def __init__(self, attribute_mode: str = "full", alpha: float = 30.0, min_count: int = 30):
        self.attribute_mode = attribute_mode
        self.alpha = alpha
        self.min_count = min_count

    def fit(self, X: pd.DataFrame, y: Any):
        self.enabled_ = str(self.attribute_mode or "none").lower() == "full"
        self.global_effect_ = 0.0
        self.maps_: dict[str, dict[str, float]] = {}
        self.county_premiums_: dict[str, dict[str, float]] = {}
        if not self.enabled_:
            return self

        df = X.copy()
        y_series = pd.Series(np.asarray(y, dtype=float), index=df.index, name="__premium__")
        valid = y_series.notna() & np.isfinite(y_series)
        work = df.loc[valid].copy()
        yv = y_series.loc[valid]
        self.global_effect_ = float(np.nanmedian(yv)) if len(yv) else 0.0

        if "county" not in work.columns:
            work["county"] = "missing"
        work["county"] = work["county"].astype(str).fillna("missing")

        # ensure quality features exist
        if "attr_total_quality_score" not in work.columns:
            work = add_attribute_quality_features(work)

        group_defs = {
            "attr_effect_building_age_bucket": "attr_building_age_bucket",
            "attr_effect_elevator": "attr_has_elevator",
            "attr_effect_site_inside": "attr_is_site_inside",
            "attr_effect_parking": "attr_has_parking",
            "attr_effect_heating_quality_bucket": "attr_heating_quality_bucket",
            "attr_effect_total_quality_bucket": "attr_total_quality_bucket",
        }
        work = work.join(yv)
        for out_col, key_col in group_defs.items():
            if key_col not in work.columns:
                self.maps_[out_col] = {}
                continue
            keys = work["county"].astype(str) + "||" + work[key_col].astype(str)
            stats = (
                pd.DataFrame({"key": keys, "premium": work["__premium__"]})
                .groupby("key", dropna=False)["premium"]
                .agg(["median", "count"])
                .reset_index()
            )
            alpha = float(self.alpha)
            g = self.global_effect_
            mapping = {}
            for _, row in stats.iterrows():
                n = float(row["count"])
                med = float(row["median"])
                if n < float(self.min_count):
                    # still smooth toward global
                    eff = (n / (n + alpha)) * med + (alpha / (n + alpha)) * g
                else:
                    eff = (n / (n + alpha)) * med + (alpha / (n + alpha)) * g
                mapping[str(row["key"])] = float(eff)
            self.maps_[out_col] = mapping

        # county-level premiums for binary attributes
        def county_flag_premium(flag_col: str, positive: float = 1.0) -> dict[str, float]:
            if flag_col not in work.columns:
                return {}
            out_map: dict[str, float] = {}
            for county, gdf in work.groupby("county", dropna=False):
                pos = gdf[pd.to_numeric(gdf[flag_col], errors="coerce") == positive]
                if len(pos) < 5:
                    out_map[str(county)] = self.global_effect_
                    continue
                n = float(len(pos))
                med = float(np.nanmedian(pos["__premium__"]))
                out_map[str(county)] = float((n / (n + self.alpha)) * med + (self.alpha / (n + self.alpha)) * self.global_effect_)
            return out_map

        self.county_premiums_ = {
            "attr_county_new_building_premium": county_flag_premium("attr_is_new_building", 1.0),
            "attr_county_old_building_discount": county_flag_premium("attr_is_old_building", 1.0),
            "attr_county_elevator_premium": county_flag_premium("attr_has_elevator", 1.0),
            "attr_county_site_inside_premium": county_flag_premium("attr_is_site_inside", 1.0),
            "attr_county_parking_premium": county_flag_premium("attr_has_parking", 1.0),
            "attr_county_good_heating_premium": {},
            "attr_county_high_quality_premium": {},
        }
        # good heating / high quality via buckets
        if "attr_heating_quality_bucket" in work.columns:
            for county, gdf in work.groupby("county", dropna=False):
                pos = gdf[gdf["attr_heating_quality_bucket"].isin(["premium", "good"])]
                n = float(len(pos))
                med = float(np.nanmedian(pos["__premium__"])) if n else self.global_effect_
                self.county_premiums_["attr_county_good_heating_premium"][str(county)] = float(
                    (n / (n + self.alpha)) * med + (self.alpha / (n + self.alpha)) * self.global_effect_
                ) if n else self.global_effect_
        if "attr_total_quality_bucket" in work.columns:
            for county, gdf in work.groupby("county", dropna=False):
                pos = gdf[gdf["attr_total_quality_bucket"] == "high"]
                n = float(len(pos))
                med = float(np.nanmedian(pos["__premium__"])) if n else self.global_effect_
                self.county_premiums_["attr_county_high_quality_premium"][str(county)] = float(
                    (n / (n + self.alpha)) * med + (self.alpha / (n + self.alpha)) * self.global_effect_
                ) if n else self.global_effect_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if not getattr(self, "enabled_", False):
            for col in ATTRIBUTE_EFFECT_NUMERIC_FEATURES:
                if col not in df.columns:
                    df[col] = 0.0
            return df

        if "attr_total_quality_score" not in df.columns:
            df = add_attribute_quality_features(df)
        if "county" not in df.columns:
            df["county"] = "missing"
        county = df["county"].astype(str).fillna("missing")

        key_map = {
            "attr_effect_building_age_bucket": "attr_building_age_bucket",
            "attr_effect_elevator": "attr_has_elevator",
            "attr_effect_site_inside": "attr_is_site_inside",
            "attr_effect_parking": "attr_has_parking",
            "attr_effect_heating_quality_bucket": "attr_heating_quality_bucket",
            "attr_effect_total_quality_bucket": "attr_total_quality_bucket",
        }
        for out_col, key_col in key_map.items():
            if key_col not in df.columns:
                df[out_col] = self.global_effect_
                continue
            keys = county + "||" + df[key_col].astype(str)
            df[out_col] = keys.map(self.maps_.get(out_col, {})).fillna(self.global_effect_)

        for col, mapping in self.county_premiums_.items():
            df[col] = county.map(mapping).fillna(self.global_effect_)
        return df


def build_debug_feature_frame(df: pd.DataFrame, attribute_mode: str = "full") -> pd.DataFrame:
    """Shared FE+attribute path for debug script (no model / no fold stats)."""
    out = df.copy()
    out = add_attribute_quality_features(out) if str(attribute_mode).lower() != "none" else out
    return out
