"""V17 location / geo feature transformers (app-safe, no target leakage).

Modes:
  none        — no location features
  basic       — lat/lon + precision/coverage flags
  geo         — basic + distances / centroids / clusters / coast / interactions
  comparable  — handled by ComparableMarketFeatureAdder (basic only here if needed)
  full        — geo (+ comparable added separately in pipeline)

Location features do NOT use the target. Cluster/centroid stats are fit on
train-fold coordinates only.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans

# ---------------------------------------------------------------------------
# Kocaeli bounds (soft validation)
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 39.5, 41.5
LON_MIN, LON_MAX = 28.5, 31.5

# Approx static anchors (README: approximate public coordinates, not survey-grade)
STATIC_ANCHORS: dict[str, tuple[float, float]] = {
    "izmit_center": (40.7656, 29.9406),
    "basiskele_coast_center": (40.7120, 29.9200),
    "basiskele_yuvacik": (40.6780, 29.9750),
    "basiskele_bahcecik": (40.6950, 29.9550),
    "basiskele_kullar": (40.7350, 29.9800),
    "basiskele_sahil": (40.7200, 29.9050),
    "golcuk_center": (40.7167, 29.8167),
    "karamursel_center": (40.6914, 29.6153),
}

ANCHOR_FEATURE_MAP = {
    "izmit_center": "distance_to_izmit_center_m",
    "basiskele_coast_center": "distance_to_basiskele_coast_m",
    "basiskele_yuvacik": "distance_to_yuvacik_m",
    "basiskele_bahcecik": "distance_to_bahcecik_m",
    "basiskele_kullar": "distance_to_kullar_m",
    "basiskele_sahil": "distance_to_sahil_m",
    "golcuk_center": "distance_to_golcuk_center_m",
    "karamursel_center": "distance_to_karamursel_center_m",
}

# Approximate gulf coast polyline points (Karamürsel → Gölcük → İzmit → Başiskele)
COAST_POLYLINE: list[tuple[float, float]] = [
    (40.6914, 29.6153),  # Karamürsel
    (40.7000, 29.7000),
    (40.7100, 29.7800),
    (40.7167, 29.8167),  # Gölcük
    (40.7400, 29.8800),
    (40.7656, 29.9406),  # İzmit
    (40.7450, 29.9300),
    (40.7200, 29.9050),  # Başiskele sahil
    (40.7120, 29.9200),
    (40.7000, 29.9400),
]

COUNTY_CLUSTER_K = {
    "Başiskele": 6,
    "İzmit": 6,
    "Gölcük": 4,
    "Karamürsel": 3,
}

CITY_CLUSTER_K = 12

# Feature metadata: which features require exact_map precision to be trustworthy
EXACT_MAP_REQUIRED_FEATURES = frozenset(
    {
        "lat",
        "lon",
        "lat_centered_city",
        "lon_centered_city",
        "lat_centered_county",
        "lon_centered_county",
        "distance_to_county_centroid_m",
        "bearing_from_county_centroid_sin",
        "bearing_from_county_centroid_cos",
        "distance_to_district_centroid_m",
        "bearing_from_district_centroid_sin",
        "bearing_from_district_centroid_cos",
        "distance_to_coastline_m",
        "is_coastal_500m",
        "is_coastal_1000m",
        "is_coastal_2000m",
        "distance_to_geo_cluster_center_m",
        *ANCHOR_FEATURE_MAP.values(),
    }
)

BASIC_NUMERIC_FEATURES = [
    "has_lat_lon",
    "lat",
    "lon",
    "lat_centered_city",
    "lon_centered_city",
    "lat_centered_county",
    "lon_centered_county",
    "location_precision_exact",
    "location_precision_approx",
    "location_precision_district_only",
    "location_precision_missing",
    "location_source_data_attr_map",
    "location_backfill_ok",
    "location_backfill_listing_removed",
    "location_quality_score",
]

GEO_DISTANCE_NUMERIC = [
    "distance_to_county_centroid_m",
    "bearing_from_county_centroid_sin",
    "bearing_from_county_centroid_cos",
    "distance_to_district_centroid_m",
    "bearing_from_district_centroid_sin",
    "bearing_from_district_centroid_cos",
    *list(ANCHOR_FEATURE_MAP.values()),
    "distance_to_coastline_m",
    "is_coastal_500m",
    "is_coastal_1000m",
    "is_coastal_2000m",
    "distance_to_geo_cluster_center_m",
]

GEO_INTERACTION_NUMERIC = [
    "location_quality_x_detail_effect_total",
    "distance_to_coast_x_view_sea",
    "distance_to_coast_x_near_sea_zero",
    "distance_to_coast_x_site_inside",
    "distance_to_coast_x_large_home",
    "basiskele_lat_lon_interaction",
    "basiskele_distance_to_coast_x_large_home",
    "basiskele_distance_to_coast_x_quality",
]

LOCATION_CATEGORICAL_FEATURES = [
    "location_precision",
    "location_source",
    "geo_cluster_city",
    "geo_cluster_county",
    "basiskele_geo_cluster",
    "coast_distance_bucket",
    "basiskele_geo_cluster_x_m2_group",
    "geo_cluster_x_room_count",
]

LOCATION_NOT_USED = "location_not_used"
DEFAULT_COVERAGE_MIN = 0.40


def compute_county_lat_lon_coverage(df: pd.DataFrame) -> dict[str, float]:
    """lat/lon coverage rate by county."""
    if df is None or len(df) == 0:
        return {}
    n = len(df)
    county = df["county"].astype(str) if "county" in df.columns else pd.Series(["missing"] * n, index=df.index)
    if "latitude" in df.columns:
        lat = pd.to_numeric(df["latitude"], errors="coerce")
    elif "lat" in df.columns:
        lat = pd.to_numeric(df["lat"], errors="coerce")
    else:
        lat = pd.Series(np.nan, index=df.index, dtype=float)
    if "longitude" in df.columns:
        lon = pd.to_numeric(df["longitude"], errors="coerce")
    elif "lon" in df.columns:
        lon = pd.to_numeric(df["lon"], errors="coerce")
    else:
        lon = pd.Series(np.nan, index=df.index, dtype=float)
    has = lat.notna() & lon.notna()
    out: dict[str, float] = {}
    for c, idx in county.groupby(county).groups.items():
        n_c = len(idx)
        out[str(c)] = float(has.loc[idx].mean()) if n_c else 0.0
    return out


def resolve_enabled_counties(
    scope: str,
    coverage: dict[str, float],
    *,
    min_coverage: float = DEFAULT_COVERAGE_MIN,
) -> tuple[set[str], list[str]]:
    """Return (enabled_counties, warnings)."""
    scope = str(scope or "basiskele_only").lower()
    warnings: list[str] = []
    if scope in {"basiskele_only", "basiskele", "bsk_only"}:
        return {"Başiskele"}, warnings
    # global: enable counties with enough coverage
    enabled: set[str] = set()
    for county, rate in (coverage or {}).items():
        if float(rate) >= float(min_coverage):
            enabled.add(str(county))
        else:
            warnings.append(f"location_disabled_for_county_due_to_low_coverage:{county}:{rate:.3f}")
    if not enabled and coverage:
        # fall back to highest-coverage county rather than enabling all
        best = max(coverage.items(), key=lambda kv: float(kv[1]))
        if float(best[1]) > 0:
            enabled.add(str(best[0]))
            warnings.append(f"location_fallback_enabled_highest_coverage:{best[0]}:{best[1]:.3f}")
    return enabled, warnings


def is_location_related_column(col: str) -> bool:
    c = str(col)
    if c in LOCATION_CATEGORICAL_FEATURES:
        return True
    prefixes = (
        "has_lat_lon",
        "lat",
        "lon",
        "lat_centered_",
        "lon_centered_",
        "location_",
        "distance_to_",
        "bearing_from_",
        "is_coastal_",
        "coast_",
        "geo_cluster",
        "basiskele_geo",
        "basiskele_lat",
        "basiskele_distance",
        "geo_context_",
        "log_distance_to_sea",
        "walkability_",
        "coastal_access_",
        "education_access_",
        "health_access_",
        "daily_life_access_",
        "location_context_score",
        "bsk_",
        "similar_",
        "nearest_",
        "broad_",
        "weighted_comp_",
        "county_recent_",
        "district_recent_",
        "school_",
        "university_",
        "pharmacy_",
        "healthcare_",
        "market_",
        "park_",
        "daily_poi_",
        "bus_stop_",
        "transport_",
        "missing_county_centroid",
    )
    if c in {"lat", "lon"}:
        return True
    return any(c == p or c.startswith(p) for p in prefixes)


class LocationScopeMasker(BaseEstimator, TransformerMixin):
    """Zero / neutralize location features outside enabled counties.

    basiskele_only: only Başiskele keeps location signal.
    global: counties with lat/lon coverage < min_coverage are masked.
    """

    def __init__(
        self,
        location_scope: str = "basiskele_only",
        min_coverage: float = DEFAULT_COVERAGE_MIN,
        enabled: bool = True,
    ):
        self.location_scope = location_scope
        self.min_coverage = min_coverage
        self.enabled = enabled
        self.enabled_counties_: set[str] = set()
        self.coverage_: dict[str, float] = {}
        self.warnings_: list[str] = []
        self.numeric_cols_: list[str] = []
        self.categorical_cols_: list[str] = []
        self.fit_report_: dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: Any = None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.coverage_ = compute_county_lat_lon_coverage(df)
        if not self.enabled:
            self.enabled_counties_ = set()
            self.warnings_ = []
            self.numeric_cols_ = []
            self.categorical_cols_ = []
            self.fit_report_ = {"location_scope": self.location_scope, "enabled": False}
            return self
        self.enabled_counties_, self.warnings_ = resolve_enabled_counties(
            self.location_scope, self.coverage_, min_coverage=float(self.min_coverage)
        )
        self.numeric_cols_ = []
        self.categorical_cols_ = []
        for c in df.columns:
            if not is_location_related_column(c):
                continue
            if c in LOCATION_CATEGORICAL_FEATURES or df[c].dtype == object or str(df[c].dtype) == "string":
                # treat known cats + object location cols as categorical
                if c in LOCATION_CATEGORICAL_FEATURES or c.startswith("geo_cluster") or c.startswith("basiskele_geo") or c in {
                    "location_precision",
                    "location_source",
                    "coast_distance_bucket",
                }:
                    self.categorical_cols_.append(c)
                elif pd.api.types.is_numeric_dtype(df[c]):
                    self.numeric_cols_.append(c)
                else:
                    self.categorical_cols_.append(c)
            else:
                self.numeric_cols_.append(c)
        # ensure core flags are numeric-masked
        for c in ["has_lat_lon", "location_quality_score", "lat", "lon"]:
            if c in df.columns and c not in self.numeric_cols_:
                self.numeric_cols_.append(c)
        self.fit_report_ = {
            "location_scope": self.location_scope,
            "enabled_counties": sorted(self.enabled_counties_),
            "coverage": self.coverage_,
            "warnings": list(self.warnings_),
            "n_numeric_masked": len(self.numeric_cols_),
            "n_categorical_masked": len(self.categorical_cols_),
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if not self.enabled_counties_:
            # nothing enabled → mask all location features
            active = np.zeros(len(df), dtype=bool)
        else:
            county = df["county"].astype(str) if "county" in df.columns else pd.Series(["missing"] * len(df))
            active = county.isin(self.enabled_counties_).to_numpy()

        inactive = ~active
        if not inactive.any():
            return df

        for c in self.numeric_cols_:
            if c not in df.columns:
                continue
            vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float).copy()
            vals[inactive] = 0.0
            df[c] = vals

        for c in self.categorical_cols_:
            if c not in df.columns:
                continue
            s = df[c].astype(object).astype(str).copy()
            s.loc[inactive] = LOCATION_NOT_USED
            df[c] = s

        # Always force coverage flags off outside enabled counties
        for c in ["has_lat_lon", "geo_context_has_exact_location", "geo_context_has_approx_location"]:
            if c in df.columns:
                vals = pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(dtype=float).copy()
                vals[inactive] = 0.0
                df[c] = vals
        if "geo_context_missing" in df.columns:
            vals = pd.to_numeric(df["geo_context_missing"], errors="coerce").fillna(1).to_numpy(dtype=float).copy()
            vals[inactive] = 1.0
            df["geo_context_missing"] = vals
        return df


def haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance in meters."""
    r = 6_371_000.0
    phi1 = np.radians(lat1.astype(float))
    phi2 = np.radians(lat2.astype(float))
    dphi = np.radians(lat2.astype(float) - lat1.astype(float))
    dlambda = np.radians(lon2.astype(float) - lon1.astype(float))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_sin_cos(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phi1 = np.radians(lat1.astype(float))
    phi2 = np.radians(lat2.astype(float))
    dlon = np.radians(lon2.astype(float) - lon1.astype(float))
    x = np.sin(dlon) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlon)
    brng = np.arctan2(x, y)
    return np.sin(brng), np.cos(brng)


def distance_to_polyline_m(lat: np.ndarray, lon: np.ndarray, poly: list[tuple[float, float]]) -> np.ndarray:
    """Min haversine distance to any polyline vertex or segment midpoint (approx)."""
    n = len(lat)
    if not poly:
        return np.full(n, np.nan)
    lat_a = np.asarray(lat, dtype=float)
    lon_a = np.asarray(lon, dtype=float)
    valid = np.isfinite(lat_a) & np.isfinite(lon_a)
    out = np.full(n, np.nan, dtype=float)
    if not valid.any():
        return out

    pts = list(poly)
    # also sample midpoints of consecutive segments
    mids = [
        ((pts[i][0] + pts[i + 1][0]) / 2.0, (pts[i][1] + pts[i + 1][1]) / 2.0)
        for i in range(len(pts) - 1)
    ]
    all_pts = pts + mids
    lat_v = lat_a[valid]
    lon_v = lon_a[valid]
    dists = [
        haversine_m(
            lat_v,
            lon_v,
            np.full(lat_v.shape[0], plat, dtype=float),
            np.full(lon_v.shape[0], plon, dtype=float),
        )
        for plat, plon in all_pts
    ]
    out[valid] = np.min(np.vstack(dists), axis=0)
    return out


def _norm_precision(val: Any) -> str:
    s = str(val).strip().lower() if pd.notna(val) else ""
    if s in {"exact_map", "exact", "map_exact"}:
        return "exact_map"
    if s in {"approx_map", "approx", "approximate", "map_approx"}:
        return "approx_map"
    if s in {"district_only", "district", "neighborhood_only"}:
        return "district_only"
    if not s or s in {"nan", "none", "null", "missing"}:
        return "missing"
    return s


def _norm_source(val: Any) -> str:
    if pd.isna(val):
        return "missing"
    return str(val).strip() or "missing"


def _quality_score(precision: str, backfill_status: str, has_coords: bool) -> float:
    status = str(backfill_status or "").strip().lower()
    if status in {"listing_removed", "removed"} and has_coords:
        return 0.8
    if precision == "exact_map" and status in {"ok", "success", "done", ""}:
        return 1.0
    if precision == "exact_map":
        return 0.95
    if precision == "approx_map":
        return 0.7
    if precision == "district_only":
        return 0.3
    return 0.0


def get_location_feature_names(mode: str) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature names for a location mode."""
    m = str(mode or "none").lower()
    if m in {"", "none"}:
        return [], []
    numeric = list(BASIC_NUMERIC_FEATURES)
    categorical = ["location_precision", "location_source"]
    if m in {"geo", "full"}:
        numeric = numeric + list(GEO_DISTANCE_NUMERIC) + list(GEO_INTERACTION_NUMERIC)
        categorical = list(LOCATION_CATEGORICAL_FEATURES)
    elif m == "comparable":
        # comparable mode still gets basic flags for confidence gating
        pass
    elif m == "basic":
        pass
    else:
        return [], []
    # dedupe
    seen: set[str] = set()
    num_out: list[str] = []
    for n in numeric:
        if n not in seen:
            seen.add(n)
            num_out.append(n)
    seen_c: set[str] = set()
    cat_out: list[str] = []
    for n in categorical:
        if n not in seen_c:
            seen_c.add(n)
            cat_out.append(n)
    return num_out, cat_out


def location_feature_metadata(mode: str) -> dict[str, Any]:
    num, cat = get_location_feature_names(mode)
    return {
        "location_feature_mode": mode,
        "numeric_features": num,
        "categorical_features": cat,
        "exact_map_required_features": sorted(
            f for f in (num + cat) if f in EXACT_MAP_REQUIRED_FEATURES or f.startswith("distance_to_")
        ),
        "app_safe": True,
        "uses_target": False,
        "note": "Distance/cluster features are unreliable when location_precision is district_only or missing.",
    }


class LocationFeatureAdder(BaseEstimator, TransformerMixin):
    """Add basic + geo location features. Does not use y."""

    def __init__(
        self,
        mode: str = "full",
        min_precision: str = "any",
        enable_coordinate_noise_check: bool = True,
        city_cluster_k: int = CITY_CLUSTER_K,
        random_state: int = 42,
    ):
        self.mode = mode
        self.min_precision = min_precision
        self.enable_coordinate_noise_check = enable_coordinate_noise_check
        self.city_cluster_k = city_cluster_k
        self.random_state = random_state
        self.city_lat_mean_: float = 40.75
        self.city_lon_mean_: float = 29.90
        self.county_centroids_: dict[str, tuple[float, float]] = {}
        self.district_centroids_: dict[str, tuple[float, float]] = {}
        self.city_kmeans_: MiniBatchKMeans | None = None
        self.county_kmeans_: dict[str, MiniBatchKMeans] = {}
        self.invalid_coord_count_: int = 0
        self.fit_report_: dict[str, Any] = {}

    def _enabled(self) -> bool:
        return str(self.mode or "none").lower() not in {"", "none"}

    def _geo_enabled(self) -> bool:
        return str(self.mode or "none").lower() in {"geo", "full"}

    def _extract_coords(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lat = pd.to_numeric(df.get("latitude", df.get("lat", np.nan)), errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(df.get("longitude", df.get("lon", np.nan)), errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(lat) & np.isfinite(lon)
        if self.enable_coordinate_noise_check:
            in_bounds = (lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= LON_MAX)
            invalid = valid & ~in_bounds
            self.invalid_coord_count_ = int(invalid.sum())
            valid = valid & in_bounds
            lat = lat.copy()
            lon = lon.copy()
            lat[~valid] = np.nan
            lon[~valid] = np.nan
        else:
            self.invalid_coord_count_ = 0
        return lat, lon, valid

    def _precision_ok(self, precision: pd.Series) -> pd.Series:
        mp = str(self.min_precision or "any").lower()
        p = precision.map(_norm_precision)
        if mp == "exact_map":
            return p == "exact_map"
        if mp == "approx_map":
            return p.isin(["exact_map", "approx_map"])
        return pd.Series(True, index=precision.index)

    def fit(self, X: pd.DataFrame, y: Any = None):
        if not self._enabled():
            self.fit_report_ = {"enabled": False}
            return self

        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        lat, lon, valid = self._extract_coords(df)
        if valid.any():
            self.city_lat_mean_ = float(np.nanmean(lat[valid]))
            self.city_lon_mean_ = float(np.nanmean(lon[valid]))

        county = df["county"].astype(str) if "county" in df.columns else pd.Series(["missing"] * len(df))
        district = df["district"].astype(str) if "district" in df.columns else pd.Series(["missing"] * len(df))

        self.county_centroids_ = {}
        for c in county.unique():
            mask = (county == c).to_numpy() & valid
            if mask.sum() >= 3:
                self.county_centroids_[str(c)] = (float(np.nanmean(lat[mask])), float(np.nanmean(lon[mask])))
            else:
                self.county_centroids_[str(c)] = (self.city_lat_mean_, self.city_lon_mean_)

        self.district_centroids_ = {}
        for d in district.unique():
            mask = (district == d).to_numpy() & valid
            key = str(d)
            if mask.sum() >= 3:
                self.district_centroids_[key] = (float(np.nanmean(lat[mask])), float(np.nanmean(lon[mask])))
            else:
                # fallback to county centroid of first matching row
                idxs = np.where((district == d).to_numpy())[0]
                if len(idxs):
                    ck = str(county.iloc[int(idxs[0])])
                    self.district_centroids_[key] = self.county_centroids_.get(
                        ck, (self.city_lat_mean_, self.city_lon_mean_)
                    )
                else:
                    self.district_centroids_[key] = (self.city_lat_mean_, self.city_lon_mean_)

        self.city_kmeans_ = None
        self.county_kmeans_ = {}
        if self._geo_enabled() and valid.sum() >= max(self.city_cluster_k, 8):
            coords = np.column_stack([lat[valid], lon[valid]])
            k = min(int(self.city_cluster_k), int(valid.sum()))
            self.city_kmeans_ = MiniBatchKMeans(
                n_clusters=k, random_state=self.random_state, batch_size=min(1024, max(k * 10, 32)), n_init=3
            )
            self.city_kmeans_.fit(coords)

            for cname, k_c in COUNTY_CLUSTER_K.items():
                cmask = (county.astype(str) == cname).to_numpy() & valid
                n = int(cmask.sum())
                if n < max(k_c, 6):
                    continue
                kk = min(k_c, n)
                km = MiniBatchKMeans(
                    n_clusters=kk, random_state=self.random_state, batch_size=min(512, max(kk * 10, 32)), n_init=3
                )
                km.fit(np.column_stack([lat[cmask], lon[cmask]]))
                self.county_kmeans_[cname] = km

        self.fit_report_ = {
            "enabled": True,
            "mode": self.mode,
            "n_rows": int(len(df)),
            "n_valid_coords": int(valid.sum()),
            "invalid_coord_count": int(self.invalid_coord_count_),
            "city_cluster_k": int(getattr(self.city_kmeans_, "n_clusters", 0) or 0),
            "county_clusters": {k: int(v.n_clusters) for k, v in self.county_kmeans_.items()},
            "n_county_centroids": len(self.county_centroids_),
            "n_district_centroids": len(self.district_centroids_),
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if not self._enabled():
            return df

        n = len(df)
        lat, lon, valid = self._extract_coords(df)
        precision_raw = df["location_precision"] if "location_precision" in df.columns else pd.Series(["missing"] * n)
        precision = precision_raw.map(_norm_precision)
        prec_ok = self._precision_ok(precision).to_numpy()
        # gate coords by min_precision
        gated = valid & prec_ok
        lat_g = lat.copy()
        lon_g = lon.copy()
        lat_g[~gated] = np.nan
        lon_g[~gated] = np.nan
        has = gated.astype(float)

        source = (
            df["location_source"].map(_norm_source)
            if "location_source" in df.columns
            else pd.Series(["missing"] * n)
        )
        backfill = (
            df["location_backfill_status"].astype(str).str.strip().str.lower()
            if "location_backfill_status" in df.columns
            else pd.Series([""] * n)
        )

        quality = np.array(
            [_quality_score(precision.iloc[i], backfill.iloc[i], bool(gated[i])) for i in range(n)],
            dtype=float,
        )
        # invalid coords already zeroed via gated
        quality = np.where(valid & ~prec_ok, np.minimum(quality, 0.2), quality)
        quality = np.where(~valid, np.where(precision.eq("district_only"), 0.3, 0.0), quality)

        county = df["county"].astype(str) if "county" in df.columns else pd.Series(["missing"] * n)
        district = df["district"].astype(str) if "district" in df.columns else pd.Series(["missing"] * n)

        # county-centered means for centering features (from fit)
        c_lat = np.array(
            [self.county_centroids_.get(str(c), (self.city_lat_mean_, self.city_lon_mean_))[0] for c in county],
            dtype=float,
        )
        c_lon = np.array(
            [self.county_centroids_.get(str(c), (self.city_lat_mean_, self.city_lon_mean_))[1] for c in county],
            dtype=float,
        )

        df["has_lat_lon"] = has
        df["lat"] = np.where(gated, lat_g, np.nan)
        df["lon"] = np.where(gated, lon_g, np.nan)
        df["lat_centered_city"] = np.where(gated, lat_g - self.city_lat_mean_, 0.0)
        df["lon_centered_city"] = np.where(gated, lon_g - self.city_lon_mean_, 0.0)
        df["lat_centered_county"] = np.where(gated, lat_g - c_lat, 0.0)
        df["lon_centered_county"] = np.where(gated, lon_g - c_lon, 0.0)
        df["location_precision"] = precision.astype(str)
        df["location_source"] = source.astype(str)
        df["location_precision_exact"] = (precision == "exact_map").astype(float)
        df["location_precision_approx"] = (precision == "approx_map").astype(float)
        df["location_precision_district_only"] = (precision == "district_only").astype(float)
        df["location_precision_missing"] = (precision == "missing").astype(float)
        df["location_source_data_attr_map"] = source.str.lower().str.contains("data_attr|map|exact", regex=True).astype(float)
        df["location_backfill_ok"] = backfill.isin(["ok", "success", "done"]).astype(float)
        df["location_backfill_listing_removed"] = backfill.isin(["listing_removed", "removed"]).astype(float)
        df["location_quality_score"] = quality

        if not self._geo_enabled():
            return df

        # Centroid distances
        dist_c = haversine_m(lat_g, lon_g, c_lat, c_lon)
        bsin_c, bcos_c = bearing_sin_cos(c_lat, c_lon, lat_g, lon_g)
        miss_c = ~gated
        med_dist_c = float(np.nanmedian(dist_c[gated])) if gated.any() else 0.0
        df["distance_to_county_centroid_m"] = np.where(miss_c, med_dist_c, dist_c)
        df["bearing_from_county_centroid_sin"] = np.where(miss_c, 0.0, bsin_c)
        df["bearing_from_county_centroid_cos"] = np.where(miss_c, 0.0, bcos_c)
        df["missing_county_centroid_dist"] = miss_c.astype(float)

        d_lat = np.array(
            [
                self.district_centroids_.get(
                    str(d), self.county_centroids_.get(str(county.iloc[i]), (self.city_lat_mean_, self.city_lon_mean_))
                )[0]
                for i, d in enumerate(district)
            ],
            dtype=float,
        )
        d_lon = np.array(
            [
                self.district_centroids_.get(
                    str(d), self.county_centroids_.get(str(county.iloc[i]), (self.city_lat_mean_, self.city_lon_mean_))
                )[1]
                for i, d in enumerate(district)
            ],
            dtype=float,
        )
        dist_d = haversine_m(lat_g, lon_g, d_lat, d_lon)
        bsin_d, bcos_d = bearing_sin_cos(d_lat, d_lon, lat_g, lon_g)
        med_dist_d = float(np.nanmedian(dist_d[gated])) if gated.any() else 0.0
        df["distance_to_district_centroid_m"] = np.where(miss_c, med_dist_d, dist_d)
        df["bearing_from_district_centroid_sin"] = np.where(miss_c, 0.0, bsin_d)
        df["bearing_from_district_centroid_cos"] = np.where(miss_c, 0.0, bcos_d)

        # Anchor distances
        for key, feat in ANCHOR_FEATURE_MAP.items():
            alat, alon = STATIC_ANCHORS[key]
            d = haversine_m(lat_g, lon_g, np.full(n, alat), np.full(n, alon))
            med = float(np.nanmedian(d[gated])) if gated.any() else 0.0
            df[feat] = np.where(miss_c, med, d)

        # Coast
        coast_d = distance_to_polyline_m(lat_g, lon_g, COAST_POLYLINE)
        med_coast = float(np.nanmedian(coast_d[gated])) if gated.any() else 5000.0
        coast_d = np.where(miss_c, med_coast, coast_d)
        df["distance_to_coastline_m"] = coast_d
        df["is_coastal_500m"] = ((coast_d <= 500) & gated).astype(float)
        df["is_coastal_1000m"] = ((coast_d <= 1000) & gated).astype(float)
        df["is_coastal_2000m"] = ((coast_d <= 2000) & gated).astype(float)
        bucket = np.full(n, "3000_plus", dtype=object)
        bucket = np.where(coast_d <= 500, "0_500", bucket)
        bucket = np.where((coast_d > 500) & (coast_d <= 1000), "500_1000", bucket)
        bucket = np.where((coast_d > 1000) & (coast_d <= 3000), "1000_3000", bucket)
        bucket = np.where(miss_c, "missing", bucket)
        df["coast_distance_bucket"] = bucket

        # Clusters
        geo_city = np.full(n, "missing", dtype=object)
        dist_cluster = np.zeros(n, dtype=float)
        if self.city_kmeans_ is not None and gated.any():
            preds = self.city_kmeans_.predict(np.column_stack([lat_g[gated], lon_g[gated]]))
            centers = self.city_kmeans_.cluster_centers_
            geo_city[gated] = np.array([f"city_{int(p)}" for p in preds], dtype=object)
            dcc = haversine_m(
                lat_g[gated],
                lon_g[gated],
                centers[preds, 0],
                centers[preds, 1],
            )
            dist_cluster[gated] = dcc
        df["geo_cluster_city"] = geo_city
        df["distance_to_geo_cluster_center_m"] = dist_cluster

        geo_county = np.full(n, "other", dtype=object)
        bsk_cluster = np.full(n, "other", dtype=object)
        for cname, km in self.county_kmeans_.items():
            cmask = (county.astype(str) == cname).to_numpy() & gated
            if not cmask.any():
                continue
            preds = km.predict(np.column_stack([lat_g[cmask], lon_g[cmask]]))
            labels = np.array([f"{cname}_{int(p)}" for p in preds], dtype=object)
            geo_county[cmask] = labels
            if cname == "Başiskele":
                bsk_cluster[cmask] = labels
        df["geo_cluster_county"] = geo_county
        df["basiskele_geo_cluster"] = bsk_cluster

        # Interactions
        detail_total = pd.to_numeric(
            df["detail_effect_total"] if "detail_effect_total" in df.columns else df.get("detail_quality_score", 0.0),
            errors="coerce",
        )
        if not isinstance(detail_total, pd.Series):
            detail_total = pd.Series([float(detail_total)] * n)
        detail_total = detail_total.fillna(0.0).to_numpy(dtype=float)
        view_sea = pd.to_numeric(df["view_sea"] if "view_sea" in df.columns else 0.0, errors="coerce")
        if not isinstance(view_sea, pd.Series):
            view_sea = pd.Series([0.0] * n)
        view_sea = view_sea.fillna(0.0).to_numpy(dtype=float)
        near_sea = pd.to_numeric(df["near_sea_zero"] if "near_sea_zero" in df.columns else 0.0, errors="coerce")
        if not isinstance(near_sea, pd.Series):
            near_sea = pd.Series([0.0] * n)
        near_sea = near_sea.fillna(0.0).to_numpy(dtype=float)
        site = (
            df.get("site_inside", pd.Series(["Hayır"] * n))
            .astype(str)
            .str.lower()
            .isin(["evet", "var", "1", "true", "yes"])
            .astype(float)
            .to_numpy()
        )
        large = (
            (pd.to_numeric(df.get("gross_m2", 0.0), errors="coerce").fillna(0) >= 160).astype(float).to_numpy()
            if "is_large_home" not in df.columns
            else pd.to_numeric(df.get("is_large_home", 0.0), errors="coerce").fillna(0).to_numpy(dtype=float)
        )
        if "large_home_flag" in df.columns:
            large = np.maximum(large, pd.to_numeric(df["large_home_flag"], errors="coerce").fillna(0).to_numpy(dtype=float))

        is_bsk = (county.astype(str) == "Başiskele").to_numpy()
        log_coast = np.log1p(np.clip(coast_d, 0, None))

        df["location_quality_x_detail_effect_total"] = quality * detail_total
        df["distance_to_coast_x_view_sea"] = log_coast * view_sea
        df["distance_to_coast_x_near_sea_zero"] = log_coast * near_sea
        df["distance_to_coast_x_site_inside"] = log_coast * site
        df["distance_to_coast_x_large_home"] = log_coast * large
        df["basiskele_lat_lon_interaction"] = np.where(
            is_bsk & gated,
            (lat_g - self.city_lat_mean_) * (lon_g - self.city_lon_mean_),
            0.0,
        )
        df["basiskele_distance_to_coast_x_large_home"] = np.where(is_bsk, log_coast * large, 0.0)
        df["basiskele_distance_to_coast_x_quality"] = np.where(is_bsk, log_coast * quality, 0.0)

        m2g = df["m2_group"].astype(str) if "m2_group" in df.columns else pd.Series(["missing"] * n)
        rooms = df["room_count"].astype(str) if "room_count" in df.columns else pd.Series(["missing"] * n)
        df["basiskele_geo_cluster_x_m2_group"] = np.where(
            is_bsk, bsk_cluster.astype(str) + "||" + m2g.astype(str), "other"
        )
        df["geo_cluster_x_room_count"] = geo_city.astype(str) + "||" + rooms.astype(str)

        return df
