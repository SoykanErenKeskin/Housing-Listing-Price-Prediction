"""V17 geo-context features from offline OSM/POI/coast/road cache.

No target usage. No internet at transform time.
Distances computed in EPSG:32635 (UTM 35N) metric space.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "external" / "geo_context"
METRIC_CRS = "EPSG:32635"
WGS84 = "EPSG:4326"

GEO_CONTEXT_NUMERIC_FEATURES = [
    # flags
    "geo_context_missing",
    "geo_context_has_exact_location",
    "geo_context_has_approx_location",
    # sea/coast
    "distance_to_sea_m",
    "log_distance_to_sea",
    "is_coastal_250m",
    "is_coastal_500m",
    "is_coastal_1000m",
    "is_coastal_2000m",
    "coast_distance_bucket_code",
    # roads / transport
    "distance_to_major_road_m",
    "distance_to_primary_road_m",
    "distance_to_highway_m",
    "distance_to_nearest_bus_stop_m",
    "bus_stop_count_500m",
    "transport_count_1000m",
    # education
    "distance_to_nearest_school_m",
    "school_count_500m",
    "school_count_1000m",
    "distance_to_nearest_university_m",
    "university_count_3000m",
    # health
    "distance_to_nearest_hospital_m",
    "distance_to_nearest_healthcare_m",
    "pharmacy_count_1000m",
    "healthcare_count_2000m",
    # daily
    "distance_to_nearest_market_m",
    "market_count_500m",
    "market_count_1000m",
    "distance_to_nearest_park_m",
    "park_count_1000m",
    "daily_poi_count_500m",
    "daily_poi_count_1000m",
    # composite
    "walkability_proxy_score",
    "coastal_access_score",
    "education_access_score",
    "health_access_score",
    "daily_life_access_score",
    "location_context_score",
    # Başiskele interactions
    "bsk_distance_to_sea_x_view_sea",
    "bsk_distance_to_sea_x_near_sea_zero",
    "bsk_distance_to_sea_x_large_home",
    "bsk_distance_to_sea_x_site_inside",
    "bsk_distance_to_sea_x_quality",
    "bsk_school_access_x_family_home",
    "bsk_road_access_x_large_home",
    "bsk_coastal_x_large_home",
    "bsk_coastal_x_geo_cluster",
    "bsk_coastal_x_room_4p1",
]

GEO_CONTEXT_SUBMODES = {
    "none": [],
    "geo_no_poi": [
        "geo_context_missing",
        "geo_context_has_exact_location",
        "geo_context_has_approx_location",
    ],
    "geo_with_coast": [
        "geo_context_missing",
        "geo_context_has_exact_location",
        "geo_context_has_approx_location",
        "distance_to_sea_m",
        "log_distance_to_sea",
        "is_coastal_250m",
        "is_coastal_500m",
        "is_coastal_1000m",
        "is_coastal_2000m",
        "coast_distance_bucket_code",
        "coastal_access_score",
        "bsk_distance_to_sea_x_view_sea",
        "bsk_distance_to_sea_x_near_sea_zero",
        "bsk_distance_to_sea_x_large_home",
        "bsk_distance_to_sea_x_site_inside",
        "bsk_distance_to_sea_x_quality",
        "bsk_coastal_x_large_home",
        "bsk_coastal_x_geo_cluster",
        "bsk_coastal_x_room_4p1",
    ],
    "geo_with_poi": None,  # all except we still include coast in full_context
    "geo_full_context": None,  # all
}


def get_geo_context_feature_names(context_mode: str = "full") -> list[str]:
    m = str(context_mode or "full").lower()
    if m in {"", "none", "off"}:
        return []
    if m in {"geo_no_poi", "no_poi"}:
        return list(GEO_CONTEXT_SUBMODES["geo_no_poi"])
    if m in {"geo_with_coast", "coast", "coast_only"}:
        return list(GEO_CONTEXT_SUBMODES["geo_with_coast"])
    # geo_with_poi and full: all features
    return list(GEO_CONTEXT_NUMERIC_FEATURES)


def _load_table(path_stem: Path) -> pd.DataFrame:
    pq = path_stem.with_suffix(".parquet")
    csv = path_stem.with_suffix(".csv")
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    if csv.exists():
        return pd.read_csv(csv)
    # try direct path if stem already has suffix
    if path_stem.exists() and path_stem.suffix == ".parquet":
        return pd.read_parquet(path_stem)
    if path_stem.exists() and path_stem.suffix == ".csv":
        return pd.read_csv(path_stem)
    return pd.DataFrame()


def project_wgs84_to_utm35(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project WGS84 → EPSG:32635. Uses pyproj if available, else approximate UTM."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs(WGS84, METRIC_CRS, always_xy=True)
        x, y = transformer.transform(lon, lat)
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    except Exception:
        # Approximate UTM zone 35N (central meridian 27°)
        # Good enough for local Kocaeli relative distances.
        lon0 = 27.0
        k0 = 0.9996
        a = 6378137.0
        e2 = 0.00669437999014
        lat_r = np.radians(lat)
        lon_r = np.radians(lon)
        lon0_r = np.radians(lon0)
        N = a / np.sqrt(1 - e2 * np.sin(lat_r) ** 2)
        T = np.tan(lat_r) ** 2
        C = e2 / (1 - e2) * np.cos(lat_r) ** 2
        A = (lon_r - lon0_r) * np.cos(lat_r)
        M = a * (
            (1 - e2 / 4 - 3 * e2**2 / 64) * lat_r
            - (3 * e2 / 8 + 3 * e2**2 / 32) * np.sin(2 * lat_r)
            + (15 * e2**2 / 256) * np.sin(4 * lat_r)
        )
        x = k0 * N * (A + (1 - T + C) * A**3 / 6) + 500000.0
        y = k0 * (M + N * np.tan(lat_r) * (A**2 / 2 + (5 - T + 9 * C + 4 * C**2) * A**4 / 24))
        return x, y


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


def _access_score(distance_m: np.ndarray, scale_m: float) -> np.ndarray:
    d = np.clip(np.asarray(distance_m, dtype=float), 0, None)
    return np.exp(-d / float(scale_m))


def _count_within(xy: np.ndarray, points_xy: np.ndarray, radius_m: float) -> np.ndarray:
    if points_xy.size == 0 or len(xy) == 0:
        return np.zeros(len(xy), dtype=float)
    # chunked brute-force for modest POI counts; OK for Kocaeli scale
    n = len(xy)
    out = np.zeros(n, dtype=float)
    # if many points, use NN radius
    nn = NearestNeighbors(radius=radius_m, algorithm="kd_tree")
    nn.fit(points_xy)
    ind = nn.radius_neighbors(xy, return_distance=False)
    for i, nbrs in enumerate(ind):
        out[i] = float(len(nbrs))
    return out


def _nearest_dist(xy: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    if points_xy.size == 0 or len(xy) == 0:
        return np.full(len(xy), np.nan)
    k = min(1, len(points_xy))
    nn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree")
    nn.fit(points_xy)
    dist, _ = nn.kneighbors(xy)
    return dist[:, 0].astype(float)


def _family_home_mask(df: pd.DataFrame) -> np.ndarray:
    rooms = df["room_count"].astype(str) if "room_count" in df.columns else pd.Series(["missing"] * len(df))
    m2 = pd.to_numeric(df.get("gross_m2", 0.0), errors="coerce").fillna(0.0)
    family_rooms = rooms.isin(["3+1", "4+1", "5+1", "3 + 1", "4 + 1", "5 + 1"])
    return (family_rooms | (m2 >= 120)).to_numpy()


def _large_home_mask(df: pd.DataFrame) -> np.ndarray:
    rooms = df["room_count"].astype(str) if "room_count" in df.columns else pd.Series(["missing"] * len(df))
    m2 = pd.to_numeric(df.get("gross_m2", 0.0), errors="coerce").fillna(0.0)
    if "is_large_home" in df.columns:
        flag = pd.to_numeric(df["is_large_home"], errors="coerce").fillna(0).to_numpy(dtype=float) > 0.5
    else:
        flag = np.zeros(len(df), dtype=bool)
    large_rooms = rooms.isin(["4+1", "5+1", "4 + 1", "5 + 1"])
    return flag | large_rooms.to_numpy() | (m2 >= 150).to_numpy()


def _site_inside_mask(df: pd.DataFrame) -> np.ndarray:
    if "site_inside" not in df.columns:
        return np.zeros(len(df), dtype=float)
    return (
        df["site_inside"].astype(str).str.lower().isin(["evet", "var", "1", "true", "yes"]).astype(float).to_numpy()
    )


class GeoContextFeatureAdder(BaseEstimator, TransformerMixin):
    """Offline geo-context distances / counts / composite scores."""

    def __init__(
        self,
        mode: str = "full",
        context_mode: str = "full",
        cache_dir: str | Path | None = None,
        enabled: bool = True,
    ):
        self.mode = mode
        self.context_mode = context_mode
        self.cache_dir = cache_dir
        self.enabled = enabled
        self.cache_loaded_: bool = False
        self.cache_dir_: Path | None = None
        self.pois_: pd.DataFrame = pd.DataFrame()
        self.roads_: pd.DataFrame = pd.DataFrame()
        self.coast_: pd.DataFrame = pd.DataFrame()
        self.poi_groups_xy_: dict[str, np.ndarray] = {}
        self.road_groups_xy_: dict[str, np.ndarray] = {}
        self.coast_xy_: np.ndarray = np.zeros((0, 2))
        self.median_distances_: dict[str, float] = {}
        self.feature_names_: list[str] = []
        self.load_report_: dict[str, Any] = {}

    def _resolve_cache_dir(self) -> Path:
        if self.cache_dir:
            p = Path(self.cache_dir)
            if p.is_absolute():
                return p
            # relative: prefer CWD (repo root when launched from root), else walk to data/
            cwd_cand = Path.cwd() / p
            if (cwd_cand / "geo_context_metadata.json").exists() or cwd_cand.is_dir():
                return cwd_cand
            for parent in Path(__file__).resolve().parents:
                cand = parent / p
                if (cand / "geo_context_metadata.json").exists():
                    return cand
            return Path(__file__).resolve().parents[3] / p
        # search common locations
        here = Path(__file__).resolve()
        candidates = [
            DEFAULT_CACHE_DIR,
            here.parents[3] / "data" / "external" / "geo_context",
            Path.cwd() / "data" / "external" / "geo_context",
        ]
        for c in candidates:
            if (c / "geo_context_metadata.json").exists() or (c / "kocaeli_pois.parquet").exists() or (c / "kocaeli_pois.csv").exists():
                return c
        return candidates[0]

    def _active(self) -> bool:
        if not self.enabled:
            return False
        cm = str(self.context_mode or "full").lower()
        if cm in {"none", "off"}:
            return False
        m = str(self.mode or "none").lower()
        # Active for geo/full location modes, or when context_mode is explicitly set for ablation
        if m in {"geo", "full"}:
            return True
        if m == "comparable":
            return False
        if m in {"", "none", "basic"}:
            return False
        return True

    def _load_cache(self) -> None:
        if self.cache_loaded_:
            return
        cache_dir = self._resolve_cache_dir()
        self.cache_dir_ = cache_dir
        self.pois_ = _load_table(cache_dir / "kocaeli_pois")
        self.roads_ = _load_table(cache_dir / "kocaeli_roads")
        self.coast_ = _load_table(cache_dir / "kocaeli_coastline")
        if self.coast_.empty:
            # try anchors fallback
            anchors_path = cache_dir / "kocaeli_anchors.json"
            if anchors_path.exists():
                payload = json.loads(anchors_path.read_text(encoding="utf-8"))
                fb = payload.get("fallback_coastline") or []
                self.coast_ = pd.DataFrame(fb)

        self.poi_groups_xy_ = {}
        if not self.pois_.empty and {"lat", "lon"}.issubset(self.pois_.columns):
            lat = pd.to_numeric(self.pois_["lat"], errors="coerce").to_numpy(dtype=float)
            lon = pd.to_numeric(self.pois_["lon"], errors="coerce").to_numpy(dtype=float)
            x, y = project_wgs84_to_utm35(lat, lon)
            valid = np.isfinite(x) & np.isfinite(y)
            pois = self.pois_.loc[valid].copy()
            pois["_x"] = x[valid]
            pois["_y"] = y[valid]

            def _xy(mask: pd.Series) -> np.ndarray:
                sub = pois.loc[mask]
                if sub.empty:
                    return np.zeros((0, 2))
                return np.column_stack([sub["_x"].to_numpy(dtype=float), sub["_y"].to_numpy(dtype=float)])

            cat = pois["category"].astype(str) if "category" in pois.columns else pd.Series([""] * len(pois))
            subc = pois["subcategory"].astype(str) if "subcategory" in pois.columns else pd.Series([""] * len(pois))
            amenity = pois["amenity"].astype(str) if "amenity" in pois.columns else pd.Series([""] * len(pois))
            shop = pois["shop"].astype(str) if "shop" in pois.columns else pd.Series([""] * len(pois))
            leisure = pois["leisure"].astype(str) if "leisure" in pois.columns else pd.Series([""] * len(pois))
            highway = pois["highway"].astype(str) if "highway" in pois.columns else pd.Series([""] * len(pois))

            self.poi_groups_xy_["school"] = _xy(
                amenity.isin(["school", "kindergarten", "college"]) | subc.isin(["school", "kindergarten", "college"])
            )
            self.poi_groups_xy_["university"] = _xy(amenity.eq("university") | subc.eq("university"))
            self.poi_groups_xy_["hospital"] = _xy(amenity.eq("hospital") | subc.eq("hospital"))
            self.poi_groups_xy_["healthcare"] = _xy(amenity.isin(["hospital", "clinic", "doctors", "pharmacy"]))
            self.poi_groups_xy_["pharmacy"] = _xy(amenity.eq("pharmacy"))
            self.poi_groups_xy_["market"] = _xy(shop.isin(["supermarket", "convenience"]) | amenity.eq("marketplace"))
            self.poi_groups_xy_["park"] = _xy(leisure.eq("park"))
            self.poi_groups_xy_["daily"] = _xy(cat.eq("daily") | shop.isin(["supermarket", "convenience"]) | leisure.eq("park"))
            self.poi_groups_xy_["bus_stop"] = _xy(highway.eq("bus_stop") | subc.eq("bus_stop"))
            self.poi_groups_xy_["transport"] = _xy(cat.eq("transport"))

        self.road_groups_xy_ = {}
        if not self.roads_.empty and {"lat", "lon"}.issubset(self.roads_.columns):
            lat = pd.to_numeric(self.roads_["lat"], errors="coerce").to_numpy(dtype=float)
            lon = pd.to_numeric(self.roads_["lon"], errors="coerce").to_numpy(dtype=float)
            x, y = project_wgs84_to_utm35(lat, lon)
            valid = np.isfinite(x) & np.isfinite(y)
            roads = self.roads_.loc[valid].copy()
            roads["_x"] = x[valid]
            roads["_y"] = y[valid]
            rc = roads["road_class"].astype(str) if "road_class" in roads.columns else pd.Series(["major"] * len(roads))
            hwy = roads["highway"].astype(str) if "highway" in roads.columns else pd.Series([""] * len(roads))

            def _rxy(mask: pd.Series) -> np.ndarray:
                sub = roads.loc[mask]
                if sub.empty:
                    return np.zeros((0, 2))
                return np.column_stack([sub["_x"].to_numpy(dtype=float), sub["_y"].to_numpy(dtype=float)])

            self.road_groups_xy_["highway"] = _rxy(rc.eq("highway") | hwy.isin(["motorway", "motorway_link"]))
            self.road_groups_xy_["primary"] = _rxy(rc.isin(["primary", "highway"]) | hwy.isin(["trunk", "primary", "motorway"]))
            self.road_groups_xy_["major"] = _rxy(
                rc.isin(["major", "primary", "highway"])
                | hwy.isin(["motorway", "trunk", "primary", "secondary", "tertiary"])
            )

        if not self.coast_.empty and {"lat", "lon"}.issubset(self.coast_.columns):
            lat = pd.to_numeric(self.coast_["lat"], errors="coerce").to_numpy(dtype=float)
            lon = pd.to_numeric(self.coast_["lon"], errors="coerce").to_numpy(dtype=float)
            x, y = project_wgs84_to_utm35(lat, lon)
            valid = np.isfinite(x) & np.isfinite(y)
            self.coast_xy_ = np.column_stack([x[valid], y[valid]])
        else:
            self.coast_xy_ = np.zeros((0, 2))

        self.cache_loaded_ = True
        self.load_report_ = {
            "cache_dir": str(cache_dir),
            "n_pois": int(len(self.pois_)),
            "n_roads": int(len(self.roads_)),
            "n_coast": int(len(self.coast_)),
            "poi_groups": {k: int(len(v)) for k, v in self.poi_groups_xy_.items()},
            "road_groups": {k: int(len(v)) for k, v in self.road_groups_xy_.items()},
        }

    def fit(self, X: pd.DataFrame, y: Any = None):
        self.feature_names_ = get_geo_context_feature_names(self.context_mode if self._active() else "none")
        if not self._active():
            self.load_report_ = {"enabled": False}
            return self
        self._load_cache()
        # compute train medians for missing fallback from a dry transform pass on valid coords
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        feats = self._compute_raw(df)
        self.median_distances_ = {}
        for col in [
            "distance_to_sea_m",
            "distance_to_major_road_m",
            "distance_to_primary_road_m",
            "distance_to_highway_m",
            "distance_to_nearest_bus_stop_m",
            "distance_to_nearest_school_m",
            "distance_to_nearest_university_m",
            "distance_to_nearest_hospital_m",
            "distance_to_nearest_healthcare_m",
            "distance_to_nearest_market_m",
            "distance_to_nearest_park_m",
        ]:
            arr = feats.get(col)
            if arr is None:
                self.median_distances_[col] = 1500.0
                continue
            a = np.asarray(arr, dtype=float)
            finite = a[np.isfinite(a)]
            self.median_distances_[col] = float(np.median(finite)) if finite.size else 1500.0
        self.load_report_["enabled"] = True
        self.load_report_["median_distances"] = dict(self.median_distances_)
        return self

    def _listing_xy(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lat = pd.to_numeric(df.get("lat", df.get("latitude", np.nan)), errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(df.get("lon", df.get("longitude", np.nan)), errors="coerce").to_numpy(dtype=float)
        if "has_lat_lon" in df.columns:
            valid = pd.to_numeric(df["has_lat_lon"], errors="coerce").fillna(0).to_numpy(dtype=float) > 0.5
            valid = valid & np.isfinite(lat) & np.isfinite(lon)
        else:
            valid = np.isfinite(lat) & np.isfinite(lon)
        precision = (
            df["location_precision"].map(_norm_precision)
            if "location_precision" in df.columns
            else pd.Series(["missing"] * len(df))
        )
        exact = (precision == "exact_map").to_numpy()
        approx = precision.isin(["exact_map", "approx_map"]).to_numpy()
        x, y = project_wgs84_to_utm35(lat, lon)
        x = np.where(valid, x, np.nan)
        y = np.where(valid, y, np.nan)
        return x, y, valid, exact, approx

    def _compute_raw(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        n = len(df)
        x, y, valid, exact, approx = self._listing_xy(df)
        xy = np.column_stack([np.nan_to_num(x, nan=0.0), np.nan_to_num(y, nan=0.0)])

        out: dict[str, np.ndarray] = {
            "geo_context_missing": (~valid).astype(float),
            "geo_context_has_exact_location": (valid & exact).astype(float),
            "geo_context_has_approx_location": (valid & approx).astype(float),
        }

        cm = str(self.context_mode or "full").lower()
        want_coast = cm in {"full", "geo_full_context", "geo_with_coast", "coast", "coast_only", "geo_with_poi"}
        want_poi = cm in {"full", "geo_full_context", "geo_with_poi", "poi"}
        want_road = want_poi or cm in {"full", "geo_full_context"}

        # defaults
        for col in GEO_CONTEXT_NUMERIC_FEATURES:
            if col not in out:
                out[col] = np.zeros(n, dtype=float)

        if want_coast:
            d_sea = _nearest_dist(xy[valid], self.coast_xy_) if valid.any() else np.array([])
            sea = np.full(n, np.nan)
            if valid.any():
                sea[valid] = d_sea
            out["distance_to_sea_m"] = sea
            out["log_distance_to_sea"] = np.log1p(np.clip(sea, 0, None))
            out["is_coastal_250m"] = ((sea <= 250) & valid).astype(float)
            out["is_coastal_500m"] = ((sea <= 500) & valid).astype(float)
            out["is_coastal_1000m"] = ((sea <= 1000) & valid).astype(float)
            out["is_coastal_2000m"] = ((sea <= 2000) & valid).astype(float)
            bucket = np.full(n, 4.0)  # 3000_plus
            bucket = np.where(sea <= 250, 0.0, bucket)
            bucket = np.where((sea > 250) & (sea <= 500), 1.0, bucket)
            bucket = np.where((sea > 500) & (sea <= 1000), 2.0, bucket)
            bucket = np.where((sea > 1000) & (sea <= 3000), 3.0, bucket)
            bucket = np.where(~valid, -1.0, bucket)
            out["coast_distance_bucket_code"] = bucket
            out["coastal_access_score"] = np.where(valid, _access_score(sea, 1200.0), 0.0)

        if want_road:
            for key, col in [
                ("major", "distance_to_major_road_m"),
                ("primary", "distance_to_primary_road_m"),
                ("highway", "distance_to_highway_m"),
            ]:
                pts = self.road_groups_xy_.get(key, np.zeros((0, 2)))
                d = np.full(n, np.nan)
                if valid.any() and len(pts):
                    d[valid] = _nearest_dist(xy[valid], pts)
                out[col] = d

        if want_poi:
            # education
            schools = self.poi_groups_xy_.get("school", np.zeros((0, 2)))
            unis = self.poi_groups_xy_.get("university", np.zeros((0, 2)))
            d_school = np.full(n, np.nan)
            d_uni = np.full(n, np.nan)
            if valid.any():
                if len(schools):
                    d_school[valid] = _nearest_dist(xy[valid], schools)
                    out["school_count_500m"][valid] = _count_within(xy[valid], schools, 500)
                    out["school_count_1000m"][valid] = _count_within(xy[valid], schools, 1000)
                if len(unis):
                    d_uni[valid] = _nearest_dist(xy[valid], unis)
                    out["university_count_3000m"][valid] = _count_within(xy[valid], unis, 3000)
            out["distance_to_nearest_school_m"] = d_school
            out["distance_to_nearest_university_m"] = d_uni

            # health
            hospitals = self.poi_groups_xy_.get("hospital", np.zeros((0, 2)))
            healthcare = self.poi_groups_xy_.get("healthcare", np.zeros((0, 2)))
            pharmacy = self.poi_groups_xy_.get("pharmacy", np.zeros((0, 2)))
            d_hosp = np.full(n, np.nan)
            d_hc = np.full(n, np.nan)
            if valid.any():
                if len(hospitals):
                    d_hosp[valid] = _nearest_dist(xy[valid], hospitals)
                if len(healthcare):
                    d_hc[valid] = _nearest_dist(xy[valid], healthcare)
                    out["healthcare_count_2000m"][valid] = _count_within(xy[valid], healthcare, 2000)
                if len(pharmacy):
                    out["pharmacy_count_1000m"][valid] = _count_within(xy[valid], pharmacy, 1000)
            out["distance_to_nearest_hospital_m"] = d_hosp
            out["distance_to_nearest_healthcare_m"] = d_hc

            # daily
            markets = self.poi_groups_xy_.get("market", np.zeros((0, 2)))
            parks = self.poi_groups_xy_.get("park", np.zeros((0, 2)))
            daily = self.poi_groups_xy_.get("daily", np.zeros((0, 2)))
            d_mkt = np.full(n, np.nan)
            d_park = np.full(n, np.nan)
            if valid.any():
                if len(markets):
                    d_mkt[valid] = _nearest_dist(xy[valid], markets)
                    out["market_count_500m"][valid] = _count_within(xy[valid], markets, 500)
                    out["market_count_1000m"][valid] = _count_within(xy[valid], markets, 1000)
                if len(parks):
                    d_park[valid] = _nearest_dist(xy[valid], parks)
                    out["park_count_1000m"][valid] = _count_within(xy[valid], parks, 1000)
                if len(daily):
                    out["daily_poi_count_500m"][valid] = _count_within(xy[valid], daily, 500)
                    out["daily_poi_count_1000m"][valid] = _count_within(xy[valid], daily, 1000)
            out["distance_to_nearest_market_m"] = d_mkt
            out["distance_to_nearest_park_m"] = d_park

            # transport POI
            bus = self.poi_groups_xy_.get("bus_stop", np.zeros((0, 2)))
            transport = self.poi_groups_xy_.get("transport", np.zeros((0, 2)))
            d_bus = np.full(n, np.nan)
            if valid.any():
                if len(bus):
                    d_bus[valid] = _nearest_dist(xy[valid], bus)
                    out["bus_stop_count_500m"][valid] = _count_within(xy[valid], bus, 500)
                if len(transport):
                    out["transport_count_1000m"][valid] = _count_within(xy[valid], transport, 1000)
                    # distance_to_nearest_transport via min bus/transport
                    d_tr = _nearest_dist(xy[valid], transport) if len(transport) else d_bus[valid]
                    if "distance_to_nearest_transport_m" not in out:
                        out["distance_to_nearest_transport_m"] = np.full(n, np.nan)
                    out["distance_to_nearest_transport_m"][valid] = d_tr
            out["distance_to_nearest_bus_stop_m"] = d_bus

            # composites
            out["education_access_score"] = np.where(
                valid, 0.7 * _access_score(out["distance_to_nearest_school_m"], 800) + 0.3 * _access_score(out["distance_to_nearest_university_m"], 3000), 0.0
            )
            out["health_access_score"] = np.where(
                valid, 0.6 * _access_score(out["distance_to_nearest_healthcare_m"], 1500) + 0.4 * _access_score(out["distance_to_nearest_hospital_m"], 3000), 0.0
            )
            out["daily_life_access_score"] = np.where(
                valid, 0.6 * _access_score(out["distance_to_nearest_market_m"], 700) + 0.4 * _access_score(out["distance_to_nearest_park_m"], 1000), 0.0
            )
            out["walkability_proxy_score"] = np.where(
                valid,
                0.35 * out["daily_life_access_score"]
                + 0.25 * out["education_access_score"]
                + 0.20 * _access_score(out["distance_to_nearest_bus_stop_m"], 500)
                + 0.20 * np.clip(out["daily_poi_count_500m"] / 10.0, 0, 1),
                0.0,
            )

        if want_coast or want_poi:
            coast_s = out.get("coastal_access_score", np.zeros(n))
            walk = out.get("walkability_proxy_score", np.zeros(n))
            edu = out.get("education_access_score", np.zeros(n))
            health = out.get("health_access_score", np.zeros(n))
            daily_s = out.get("daily_life_access_score", np.zeros(n))
            road_s = _access_score(out.get("distance_to_major_road_m", np.full(n, np.nan)), 1200)
            out["location_context_score"] = np.where(
                valid,
                0.25 * coast_s + 0.20 * walk + 0.15 * edu + 0.15 * health + 0.15 * daily_s + 0.10 * np.nan_to_num(road_s, nan=0.0),
                0.0,
            )

        # Başiskele interactions
        county = df["county"].astype(str) if "county" in df.columns else pd.Series(["missing"] * n)
        is_bsk = (county == "Başiskele").to_numpy()
        log_sea = out.get("log_distance_to_sea", np.zeros(n))
        view_sea = pd.to_numeric(df.get("view_sea", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        near_sea = pd.to_numeric(df.get("near_sea_zero", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        quality = pd.to_numeric(df.get("location_quality_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        large = _large_home_mask(df).astype(float)
        family = _family_home_mask(df).astype(float)
        site = _site_inside_mask(df)
        rooms = df["room_count"].astype(str) if "room_count" in df.columns else pd.Series(["missing"] * n)
        room_4p1 = rooms.isin(["4+1", "5+1", "4 + 1", "5 + 1"]).astype(float).to_numpy()
        coastal_flag = out.get("is_coastal_1000m", np.zeros(n))
        geo_c = (
            pd.to_numeric(df.get("distance_to_geo_cluster_center_m", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if "geo_cluster_county" in df.columns or "distance_to_geo_cluster_center_m" in df.columns
            else np.zeros(n)
        )
        # encode cluster hash-ish via coastal * cluster distance signal
        school_acc = out.get("education_access_score", np.zeros(n))
        road_acc = _access_score(out.get("distance_to_major_road_m", np.full(n, np.nan)), 1200)

        out["bsk_distance_to_sea_x_view_sea"] = np.where(is_bsk, log_sea * view_sea, 0.0)
        out["bsk_distance_to_sea_x_near_sea_zero"] = np.where(is_bsk, log_sea * near_sea, 0.0)
        out["bsk_distance_to_sea_x_large_home"] = np.where(is_bsk, log_sea * large, 0.0)
        out["bsk_distance_to_sea_x_site_inside"] = np.where(is_bsk, log_sea * site, 0.0)
        out["bsk_distance_to_sea_x_quality"] = np.where(is_bsk, log_sea * quality, 0.0)
        out["bsk_school_access_x_family_home"] = np.where(is_bsk, school_acc * family, 0.0)
        out["bsk_road_access_x_large_home"] = np.where(is_bsk, np.nan_to_num(road_acc, nan=0.0) * large, 0.0)
        out["bsk_coastal_x_large_home"] = np.where(is_bsk, coastal_flag * large, 0.0)
        out["bsk_coastal_x_geo_cluster"] = np.where(is_bsk, coastal_flag * (1.0 / (1.0 + geo_c / 500.0)), 0.0)
        out["bsk_coastal_x_room_4p1"] = np.where(is_bsk, coastal_flag * room_4p1, 0.0)

        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        names = self.feature_names_ or get_geo_context_feature_names(self.context_mode if self._active() else "none")
        if not self._active() or not names:
            for c in GEO_CONTEXT_NUMERIC_FEATURES:
                if c not in df.columns:
                    df[c] = 0.0
            return df

        self._load_cache()
        raw = self._compute_raw(df)
        n = len(df)

        # fill missing distances with train medians
        for col, arr in raw.items():
            a = np.asarray(arr, dtype=float).copy()
            if col.startswith("distance_") or col == "log_distance_to_sea":
                med = self.median_distances_.get(col.replace("log_", ""), self.median_distances_.get(col, 1500.0))
                if col == "log_distance_to_sea":
                    med_log = np.log1p(self.median_distances_.get("distance_to_sea_m", 1500.0))
                    missing = ~np.isfinite(a)
                    a[missing] = med_log
                else:
                    missing = ~np.isfinite(a)
                    a[missing] = float(med)
            else:
                a = np.nan_to_num(a, nan=0.0)
            raw[col] = a

        for c in names:
            df[c] = raw.get(c, np.zeros(n, dtype=float))
        return df
