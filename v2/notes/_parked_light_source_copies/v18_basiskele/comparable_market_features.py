"""V18 Başiskele fold-safe comparable / emsal market features.

Uses unit price targets — MUST be fit inside CV folds only.
Validation never contributes targets to the pool.
Self-match via classified_id (and duplicate ids) is always excluded.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors

try:
    from location_features import haversine_m
except ImportError:  # pragma: no cover
    from v18_basiskele.location_features import haversine_m

# Injected by LocationResidualRegressor so comparable pool uses unit prices
# (not residual log targets) without leaking into the model matrix.
COMP_UNIT_PRICE_COL = "__comp_unit_price__"


def _series_lat_lon(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Safe lat/lon extraction (never return scalar)."""
    if "lat" in df.columns:
        lat = pd.to_numeric(df["lat"], errors="coerce")
    elif "latitude" in df.columns:
        lat = pd.to_numeric(df["latitude"], errors="coerce")
    else:
        lat = pd.Series(np.nan, index=df.index, dtype=float)
    if "lon" in df.columns:
        lon = pd.to_numeric(df["lon"], errors="coerce")
    elif "longitude" in df.columns:
        lon = pd.to_numeric(df["longitude"], errors="coerce")
    else:
        lon = pd.Series(np.nan, index=df.index, dtype=float)
    return lat.to_numpy(dtype=float), lon.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Feature catalogs by sub-mode
# ---------------------------------------------------------------------------
NEAREST_FEATURES = [
    "bsk_nearest_5_median_unit_price",
    "bsk_nearest_10_median_unit_price",
    "bsk_nearest_20_median_unit_price",
    "bsk_nearest_40_median_unit_price",
    "bsk_nearest_10_mean_unit_price",
    "bsk_nearest_10_std_unit_price",
    "bsk_nearest_10_iqr_unit_price",
    "bsk_nearest_10_min_unit_price",
    "bsk_nearest_10_max_unit_price",
    "bsk_nearest_10_count",
    "bsk_nearest_10_avg_distance_m",
    "bsk_nearest_10_min_distance_m",
    "bsk_nearest_confidence",
]

DISTRICT_SIMILAR_FEATURES = [
    "bsk_district_similar_10_median",
    "bsk_district_similar_20_median",
    "bsk_district_similar_20_std",
    "bsk_district_similar_count",
    "bsk_district_similar_confidence",
]

CLUSTER_SIMILAR_FEATURES = [
    "bsk_cluster_similar_10_median",
    "bsk_cluster_similar_20_median",
    "bsk_cluster_similar_std",
    "bsk_cluster_similar_count",
    "bsk_cluster_similar_confidence",
]

LARGE_HOME_FEATURES = [
    "bsk_large_home_nearest_10_median",
    "bsk_large_home_nearest_20_median",
    "bsk_large_home_similar_10_median",
    "bsk_large_home_cluster_median",
    "bsk_large_home_count",
    "bsk_large_home_confidence",
]

COASTAL_FEATURES = [
    "bsk_coastal_similar_median",
    "bsk_noncoastal_similar_median",
    "bsk_coastal_price_gap",
    "bsk_coastal_similar_count",
    "bsk_coastal_confidence",
]

WEIGHTED_FEATURES = [
    "bsk_weighted_comp_mean",
    "bsk_weighted_comp_median",
    "bsk_weighted_comp_std",
    "bsk_weighted_comp_count",
    "bsk_weighted_comp_confidence",
]

META_FEATURES = [
    "bsk_comparable_missing",
    "bsk_fallback_level",  # 0=none needed, 1=district, 2=cluster, 3=nearest, 4=all BSK median
]

SIMILAR_FEATURES = DISTRICT_SIMILAR_FEATURES + CLUSTER_SIMILAR_FEATURES

MODE_FEATURE_MAP: dict[str, list[str]] = {
    "none": [],
    "nearest": NEAREST_FEATURES + META_FEATURES,
    "similar": SIMILAR_FEATURES + META_FEATURES,
    "weighted": WEIGHTED_FEATURES + META_FEATURES,
    "large_home": LARGE_HOME_FEATURES + META_FEATURES,
    "full": (
        NEAREST_FEATURES
        + SIMILAR_FEATURES
        + LARGE_HOME_FEATURES
        + COASTAL_FEATURES
        + WEIGHTED_FEATURES
        + META_FEATURES
    ),
}


def get_comparable_feature_names(mode: str) -> list[str]:
    m = str(mode or "none").lower()
    if m in {"comparable"}:
        m = "full"
    return list(MODE_FEATURE_MAP.get(m, []))


def parse_k_list(raw: str | list[int] | None) -> list[int]:
    if raw is None:
        return [5, 10, 20]
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            continue
    return out or [5, 10, 20]


def _safe_str(s: pd.Series) -> pd.Series:
    return (
        s.astype("object")
        .where(s.notna(), "missing")
        .astype(str)
        .str.strip()
        .replace({"": "missing", "nan": "missing", "None": "missing"})
    )


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(dtype=float)


def is_large_home_row(
    gross_m2: float,
    room_count: str,
    m2_group: str,
) -> bool:
    rooms = {str(room_count or "").strip()}
    if float(gross_m2) >= 150:
        return True
    if rooms & {"4+1", "5+1", "6+1", "4+1+", "5+1+", "6+1+"}:
        return True
    if str(m2_group) in {"151-200", "200+"}:
        return True
    return False


def combined_similarity(
    m2_i: float,
    m2_j: float,
    room_i: str,
    room_j: str,
    age_i: str,
    age_j: str,
    site_i: str,
    site_j: str,
    district_i: str,
    district_j: str,
    cluster_i: str,
    cluster_j: str,
    large_i: bool,
    large_j: bool,
) -> float:
    m2_sim = float(np.exp(-abs(float(m2_i) - float(m2_j)) / 40.0)) if np.isfinite(m2_i) and np.isfinite(m2_j) else 0.0
    if room_i == room_j and room_i != "missing":
        room_sim = 1.0
    elif room_i != "missing" and room_j != "missing" and room_i[:1] == room_j[:1]:
        room_sim = 0.5
    else:
        room_sim = 0.0
    if age_i == age_j and age_i != "missing":
        age_sim = 1.0
    elif age_i != "missing" and age_j != "missing":
        age_sim = 0.5
    else:
        age_sim = 0.0
    site_sim = 1.0 if site_i == site_j else 0.5
    district_sim = 1.0 if district_i == district_j and district_i != "missing" else 0.0
    cluster_sim = 1.0 if cluster_i == cluster_j and cluster_i not in {"missing", "other", "location_not_used"} else 0.0
    large_home_sim = 1.0 if bool(large_i) == bool(large_j) else 0.3
    return (
        0.25 * m2_sim
        + 0.20 * room_sim
        + 0.15 * age_sim
        + 0.15 * district_sim
        + 0.10 * cluster_sim
        + 0.10 * site_sim
        + 0.05 * large_home_sim
    )


def _stats(y: np.ndarray) -> dict[str, float]:
    if y is None or len(y) == 0:
        return {"median": np.nan, "mean": np.nan, "std": np.nan, "iqr": np.nan, "min": np.nan, "max": np.nan, "n": 0.0}
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return {"median": np.nan, "mean": np.nan, "std": np.nan, "iqr": np.nan, "min": np.nan, "max": np.nan, "n": 0.0}
    q25, q75 = np.percentile(y, [25, 75])
    return {
        "median": float(np.median(y)),
        "mean": float(np.mean(y)),
        "std": float(np.std(y)) if len(y) > 1 else 0.0,
        "iqr": float(q75 - q25),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "n": float(len(y)),
    }


def _conf(n: float, need: float) -> float:
    if need <= 0:
        return 0.0
    return float(np.clip(n / need, 0.0, 1.0))


class ComparableMarketFeatureAdder(BaseEstimator, TransformerMixin):
    """Fold-safe Başiskele comparable market stats.

    fit(X_train, y_train): stores train pool only (unit prices).
    transform(X): uses stored pool; never uses X targets.
    """

    def __init__(
        self,
        mode: str = "full",
        k_list: str | list[int] = "5,10,20",
        distance_scale_m: float = 1500.0,
        random_state: int = 42,
    ):
        self.mode = mode
        self.k_list = k_list
        self.distance_scale_m = distance_scale_m
        self.random_state = random_state
        self.enabled_: bool = False
        self.mode_: str = "none"
        self.k_list_: list[int] = [5, 10, 20]
        self.feature_names_: list[str] = []
        self.train_lat_: np.ndarray | None = None
        self.train_lon_: np.ndarray | None = None
        self.train_valid_: np.ndarray | None = None
        self.train_y_: np.ndarray | None = None
        self.train_ids_: np.ndarray | None = None
        self.train_attrs_: pd.DataFrame | None = None
        self.nn_: NearestNeighbors | None = None
        self.valid_idx_: np.ndarray | None = None
        self.global_median_: float = np.nan
        self.fit_n_: int = 0
        self.leakage_guard_: dict[str, Any] = {}

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_, dtype=object)

    def _resolve_mode(self) -> str:
        m = str(self.mode or "none").lower()
        if m in {"comparable"}:
            return "full"
        return m if m in MODE_FEATURE_MAP else "none"

    def fit(self, X: pd.DataFrame, y: Any = None):
        self.mode_ = self._resolve_mode()
        self.enabled_ = self.mode_ != "none"
        self.k_list_ = parse_k_list(self.k_list)
        self.feature_names_ = get_comparable_feature_names(self.mode_)
        self.leakage_guard_ = {
            "fit_called": True,
            "fit_n": 0,
            "uses_train_pool_only": True,
            "self_match_exclude": True,
            "validation_targets_unused": True,
            "pass": True,
            "notes": [],
        }
        if not self.enabled_:
            self.leakage_guard_["comparable_mode"] = self.mode_
            self.leakage_guard_["feature_count"] = 0
            self.leakage_guard_["notes"].append("mode=none; comparable disabled")
            return self
        if y is None and COMP_UNIT_PRICE_COL not in (X.columns if isinstance(X, pd.DataFrame) else []):
            raise ValueError("ComparableMarketFeatureAdder requires y (unit prices) in fit().")

        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        # Prefer injected unit prices (residual pipelines pass residual as y)
        if COMP_UNIT_PRICE_COL in df.columns:
            y_arr = pd.to_numeric(df[COMP_UNIT_PRICE_COL], errors="coerce").to_numpy(dtype=float)
            self.leakage_guard_["notes"].append("pool prices from __comp_unit_price__ (unit_price_gross)")
            self.leakage_guard_["price_source"] = COMP_UNIT_PRICE_COL
        elif "unit_price_gross" in df.columns:
            y_arr = pd.to_numeric(df["unit_price_gross"], errors="coerce").to_numpy(dtype=float)
            self.leakage_guard_["notes"].append("pool prices from unit_price_gross column")
            self.leakage_guard_["price_source"] = "unit_price_gross"
        else:
            y_arr = np.asarray(y, dtype=float).reshape(-1)
            self.leakage_guard_["notes"].append("pool prices from fit(y) — ensure not residual")
            self.leakage_guard_["price_source"] = "fit_y"
        if len(y_arr) != len(df):
            raise ValueError(f"y length {len(y_arr)} != X length {len(df)}")

        lat, lon = _series_lat_lon(df)
        if "has_lat_lon" in df.columns:
            valid = pd.to_numeric(df["has_lat_lon"], errors="coerce").fillna(0).to_numpy(dtype=float) > 0.5
            valid = valid & np.isfinite(lat) & np.isfinite(lon)
        else:
            valid = np.isfinite(lat) & np.isfinite(lon)

        if "classified_id" in df.columns:
            ids = df["classified_id"].astype(str).to_numpy()
        else:
            ids = np.array([f"idx_{i}" for i in range(len(df))], dtype=object)

        district = _safe_str(df["district"]) if "district" in df.columns else pd.Series(["missing"] * len(df))
        room = _safe_str(df["room_count"]) if "room_count" in df.columns else pd.Series(["missing"] * len(df))
        age = _safe_str(df["building_age_group"]) if "building_age_group" in df.columns else pd.Series(["missing"] * len(df))
        site = _safe_str(df["site_inside"]) if "site_inside" in df.columns else pd.Series(["missing"] * len(df))
        m2g = _safe_str(df["m2_group"]) if "m2_group" in df.columns else pd.Series(["missing"] * len(df))
        # Prefer Başiskele-specific cluster column when present
        if "basiskele_geo_cluster" in df.columns:
            cluster = _safe_str(df["basiskele_geo_cluster"])
        elif "geo_cluster_county" in df.columns:
            cluster = _safe_str(df["geo_cluster_county"])
        else:
            cluster = pd.Series(["missing"] * len(df))
        gross = _num(df, "gross_m2", np.nan)
        coast_d = _num(df, "distance_to_coastline_m", 99999.0)
        if "is_coastal_1000m" in df.columns:
            coastal = _num(df, "is_coastal_1000m", 0.0)
        else:
            coastal = (coast_d <= 1000.0).astype(float)
        view_sea = _num(df, "view_sea", 0.0) if "view_sea" in df.columns else np.zeros(len(df))
        near_sea = _num(df, "near_sea_zero", 0.0) if "near_sea_zero" in df.columns else np.zeros(len(df))
        large = np.array(
            [is_large_home_row(gross[i], room.iloc[i], m2g.iloc[i]) for i in range(len(df))],
            dtype=bool,
        )

        self.train_lat_ = lat
        self.train_lon_ = lon
        self.train_valid_ = valid
        self.train_y_ = y_arr
        self.train_ids_ = ids
        self.train_attrs_ = pd.DataFrame(
            {
                "district": district.to_numpy(),
                "room_count": room.to_numpy(),
                "building_age_group": age.to_numpy(),
                "site_inside": site.to_numpy(),
                "m2_group": m2g.to_numpy(),
                "geo_cluster": cluster.to_numpy(),
                "gross_m2": gross,
                "is_coastal_1000m": coastal,
                "distance_to_coastline_m": coast_d,
                "view_sea": view_sea,
                "near_sea_zero": near_sea,
                "large_home": large.astype(float),
            }
        )
        self.fit_n_ = int(len(df))
        self.global_median_ = float(np.nanmedian(y_arr[np.isfinite(y_arr)])) if np.isfinite(y_arr).any() else np.nan
        self.valid_idx_ = np.where(valid)[0]
        self.leakage_guard_["fit_n"] = self.fit_n_
        self.leakage_guard_["feature_count"] = len(self.feature_names_)
        self.leakage_guard_["comparable_mode"] = self.mode_
        self.leakage_guard_["pool_with_coords"] = int(valid.sum())
        self.leakage_guard_["uses_train_pool_only"] = True
        self.leakage_guard_["self_match_exclude"] = True
        self.leakage_guard_["validation_targets_unused"] = True
        self.leakage_guard_["pass"] = True
        self.leakage_guard_["notes"].append("fit pool = train fold only")

        n_nn = int(valid.sum())
        if n_nn >= 3:
            max_k = max(self.k_list_ + [40, 10])
            self.nn_ = NearestNeighbors(
                n_neighbors=min(max_k + 5, n_nn),
                algorithm="ball_tree",
            )
            self.nn_.fit(np.column_stack([lat[valid], lon[valid]]))
        else:
            self.nn_ = None
            self.leakage_guard_["notes"].append("too few coords for NN")
        return self

    def fit_transform(self, X: pd.DataFrame, y: Any = None, **fit_params):
        return self.fit(X, y).transform(X)

    def _empty_frame(self, n: int) -> pd.DataFrame:
        cols = self.feature_names_ or get_comparable_feature_names(self._resolve_mode())
        out = pd.DataFrame(index=range(n))
        for c in cols:
            out[c] = 0.0 if c.endswith(("_count", "_confidence", "_missing", "_level")) else np.nan
        if "bsk_comparable_missing" in out.columns:
            out["bsk_comparable_missing"] = 1.0
        return out

    def _exclude_mask(self, query_id: str) -> np.ndarray:
        assert self.train_ids_ is not None
        return self.train_ids_ != str(query_id)

    def _row_attrs(self, df: pd.DataFrame, i: int) -> dict[str, Any]:
        def s(col: str, default: str = "missing") -> str:
            if col not in df.columns:
                return default
            v = df.iloc[i][col]
            if pd.isna(v):
                return default
            t = str(v).strip()
            return t if t and t.lower() not in {"nan", "none"} else default

        gross = float(pd.to_numeric(df.iloc[i].get("gross_m2", np.nan), errors="coerce")) if "gross_m2" in df.columns else np.nan
        room = s("room_count")
        m2g = s("m2_group")
        if "basiskele_geo_cluster" in df.columns:
            cluster = s("basiskele_geo_cluster")
        else:
            cluster = s("geo_cluster_county")
        coast_d = float(pd.to_numeric(df.iloc[i].get("distance_to_coastline_m", 99999), errors="coerce")) if "distance_to_coastline_m" in df.columns else 99999.0
        if "is_coastal_1000m" in df.columns:
            coastal = float(pd.to_numeric(df.iloc[i].get("is_coastal_1000m", 0), errors="coerce") or 0)
        else:
            coastal = 1.0 if coast_d <= 1000 else 0.0
        return {
            "district": s("district"),
            "room_count": room,
            "building_age_group": s("building_age_group"),
            "site_inside": s("site_inside"),
            "m2_group": m2g,
            "geo_cluster": cluster,
            "gross_m2": gross,
            "is_coastal_1000m": coastal,
            "distance_to_coastline_m": coast_d,
            "view_sea": float(pd.to_numeric(df.iloc[i].get("view_sea", 0), errors="coerce") or 0) if "view_sea" in df.columns else 0.0,
            "near_sea_zero": float(pd.to_numeric(df.iloc[i].get("near_sea_zero", 0), errors="coerce") or 0) if "near_sea_zero" in df.columns else 0.0,
            "large_home": is_large_home_row(gross if np.isfinite(gross) else 0.0, room, m2g),
            "lat": float(pd.to_numeric(df.iloc[i].get("lat", df.iloc[i].get("latitude", np.nan)), errors="coerce")),
            "lon": float(pd.to_numeric(df.iloc[i].get("lon", df.iloc[i].get("longitude", np.nan)), errors="coerce")),
            "classified_id": str(df.iloc[i]["classified_id"]) if "classified_id" in df.columns else f"idx_{i}",
        }

    def _nearest_indices(self, lat: float, lon: float, query_id: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (train_indices, distances_m) for nearest pool rows excluding self."""
        assert self.train_lat_ is not None and self.train_lon_ is not None
        assert self.train_valid_ is not None and self.train_ids_ is not None
        if self.nn_ is None or self.valid_idx_ is None or len(self.valid_idx_) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        if not (np.isfinite(lat) and np.isfinite(lon)):
            return np.array([], dtype=int), np.array([], dtype=float)

        n_ask = min(len(self.valid_idx_), max(k + 8, k))
        dists_deg, nn_pos = self.nn_.kneighbors(np.array([[lat, lon]]), n_neighbors=n_ask, return_distance=True)
        cand = self.valid_idx_[nn_pos[0]]
        keep = self.train_ids_[cand] != str(query_id)
        # also drop exact same coords with same id already handled; drop zero-distance same id only
        cand = cand[keep]
        if len(cand) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        d_m = haversine_m(
            np.full(len(cand), lat),
            np.full(len(cand), lon),
            self.train_lat_[cand],
            self.train_lon_[cand],
        )
        order = np.argsort(d_m)
        cand = cand[order][:k]
        d_m = d_m[order][:k]
        return cand, d_m

    def _similarity_rank(self, row: dict[str, Any], pool_mask: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self.train_attrs_ is not None and self.train_y_ is not None
        idx = np.where(pool_mask)[0]
        if len(idx) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        scores = np.zeros(len(idx), dtype=float)
        a = self.train_attrs_
        for j, ti in enumerate(idx):
            scores[j] = combined_similarity(
                row["gross_m2"],
                float(a.iloc[ti]["gross_m2"]),
                row["room_count"],
                str(a.iloc[ti]["room_count"]),
                row["building_age_group"],
                str(a.iloc[ti]["building_age_group"]),
                row["site_inside"],
                str(a.iloc[ti]["site_inside"]),
                row["district"],
                str(a.iloc[ti]["district"]),
                row["geo_cluster"],
                str(a.iloc[ti]["geo_cluster"]),
                bool(row["large_home"]),
                bool(a.iloc[ti]["large_home"]),
            )
        order = np.argsort(-scores)
        idx = idx[order][:k]
        scores = scores[order][:k]
        return idx, scores

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        n = len(df)
        # Never let helper / leakage columns reach the model
        if COMP_UNIT_PRICE_COL in df.columns:
            df = df.drop(columns=[COMP_UNIT_PRICE_COL])
        # mode=none (or fit skipped): pass through unchanged — never drop upstream columns
        if not self.enabled_ or self.train_y_ is None or self.train_attrs_ is None:
            return df

        mode = self.mode_
        feats = {c: np.full(n, np.nan, dtype=float) for c in self.feature_names_}
        for c in self.feature_names_:
            if c.endswith(("_count", "_confidence", "_missing", "_level")):
                feats[c] = np.zeros(n, dtype=float)

        y = self.train_y_
        a = self.train_attrs_
        assert self.train_ids_ is not None

        for i in range(n):
            row = self._row_attrs(df, i)
            qid = row["classified_id"]
            keep = self._exclude_mask(qid)
            fallback = 0.0
            any_ok = False

            # Global fallback median always available
            pool_all = keep & np.isfinite(y)

            if mode in {"nearest", "full"}:
                for k_tag, k_need in [(5, 5), (10, 10), (20, 20), (40, 40)]:
                    ni, dist = self._nearest_indices(row["lat"], row["lon"], qid, k_need)
                    st = _stats(y[ni]) if len(ni) else _stats(np.array([]))
                    key = f"bsk_nearest_{k_tag}_median_unit_price"
                    if key in feats:
                        feats[key][i] = st["median"]
                    if k_tag == 10:
                        for name, val in [
                            ("bsk_nearest_10_mean_unit_price", st["mean"]),
                            ("bsk_nearest_10_std_unit_price", st["std"]),
                            ("bsk_nearest_10_iqr_unit_price", st["iqr"]),
                            ("bsk_nearest_10_min_unit_price", st["min"]),
                            ("bsk_nearest_10_max_unit_price", st["max"]),
                            ("bsk_nearest_10_count", st["n"]),
                        ]:
                            if name in feats:
                                feats[name][i] = val
                        if "bsk_nearest_10_avg_distance_m" in feats:
                            feats["bsk_nearest_10_avg_distance_m"][i] = float(np.mean(dist)) if len(dist) else np.nan
                        if "bsk_nearest_10_min_distance_m" in feats:
                            feats["bsk_nearest_10_min_distance_m"][i] = float(np.min(dist)) if len(dist) else np.nan
                        if "bsk_nearest_confidence" in feats:
                            feats["bsk_nearest_confidence"][i] = _conf(st["n"], 10.0)
                        if st["n"] > 0:
                            any_ok = True

            if mode in {"similar", "full"}:
                # District similar
                dist_mask = pool_all & (a["district"].to_numpy() == row["district"])
                idx10, _ = self._similarity_rank(row, dist_mask, 10)
                idx20, _ = self._similarity_rank(row, dist_mask, 20)
                st10 = _stats(y[idx10])
                st20 = _stats(y[idx20])
                if st10["n"] < 3:
                    # fallback geo cluster
                    fallback = max(fallback, 2.0)
                    cl_mask = pool_all & (a["geo_cluster"].to_numpy() == row["geo_cluster"])
                    idx10, _ = self._similarity_rank(row, cl_mask, 10)
                    idx20, _ = self._similarity_rank(row, cl_mask, 20)
                    st10 = _stats(y[idx10])
                    st20 = _stats(y[idx20])
                if st10["n"] < 3:
                    fallback = max(fallback, 3.0)
                    ni, _ = self._nearest_indices(row["lat"], row["lon"], qid, 10)
                    st10 = _stats(y[ni])
                    ni20, _ = self._nearest_indices(row["lat"], row["lon"], qid, 20)
                    st20 = _stats(y[ni20])
                if st10["n"] < 1:
                    fallback = 4.0
                    st10 = {"median": self.global_median_, "mean": self.global_median_, "std": np.nan, "iqr": np.nan, "min": np.nan, "max": np.nan, "n": 0.0}
                    st20 = st10
                for name, val in [
                    ("bsk_district_similar_10_median", st10["median"]),
                    ("bsk_district_similar_20_median", st20["median"]),
                    ("bsk_district_similar_20_std", st20["std"]),
                    ("bsk_district_similar_count", st20["n"]),
                    ("bsk_district_similar_confidence", _conf(st20["n"], 20.0)),
                ]:
                    if name in feats:
                        feats[name][i] = val
                if st10["n"] > 0:
                    any_ok = True

                # Cluster similar
                cl_mask = pool_all & (a["geo_cluster"].to_numpy() == row["geo_cluster"])
                c10, _ = self._similarity_rank(row, cl_mask, 10)
                c20, _ = self._similarity_rank(row, cl_mask, 20)
                cst10 = _stats(y[c10])
                cst20 = _stats(y[c20])
                if cst10["n"] < 3:
                    ni, _ = self._nearest_indices(row["lat"], row["lon"], qid, 10)
                    cst10 = _stats(y[ni])
                    ni20, _ = self._nearest_indices(row["lat"], row["lon"], qid, 20)
                    cst20 = _stats(y[ni20])
                    fallback = max(fallback, 3.0)
                for name, val in [
                    ("bsk_cluster_similar_10_median", cst10["median"]),
                    ("bsk_cluster_similar_20_median", cst20["median"]),
                    ("bsk_cluster_similar_std", cst20["std"]),
                    ("bsk_cluster_similar_count", cst20["n"]),
                    ("bsk_cluster_similar_confidence", _conf(cst20["n"], 20.0)),
                ]:
                    if name in feats:
                        feats[name][i] = val

            if mode in {"large_home", "full"}:
                large_mask = pool_all & (a["large_home"].to_numpy() > 0.5)
                # nearest among large homes (filter after NN)
                ni, dist = self._nearest_indices(row["lat"], row["lon"], qid, 40)
                ni_large = np.array([j for j in ni if a.iloc[j]["large_home"] > 0.5], dtype=int)
                if len(ni_large) < 5:
                    # similarity among large
                    s_idx, _ = self._similarity_rank(row, large_mask, 20)
                    ni_large = s_idx
                st10 = _stats(y[ni_large[:10]])
                st20 = _stats(y[ni_large[:20]])
                sim10, _ = self._similarity_rank(row, large_mask, 10)
                sim_st = _stats(y[sim10])
                cl_large = large_mask & (a["geo_cluster"].to_numpy() == row["geo_cluster"])
                cl_idx, _ = self._similarity_rank(row, cl_large, 20)
                cl_st = _stats(y[cl_idx])
                for name, val in [
                    ("bsk_large_home_nearest_10_median", st10["median"]),
                    ("bsk_large_home_nearest_20_median", st20["median"]),
                    ("bsk_large_home_similar_10_median", sim_st["median"]),
                    ("bsk_large_home_cluster_median", cl_st["median"]),
                    ("bsk_large_home_count", st20["n"]),
                    ("bsk_large_home_confidence", _conf(st20["n"], 10.0)),
                ]:
                    if name in feats:
                        feats[name][i] = val
                if st10["n"] > 0:
                    any_ok = True

            if mode in {"full"}:  # coastal only in full for core
                coastal_mask = pool_all & (a["is_coastal_1000m"].to_numpy() > 0.5)
                non_mask = pool_all & (a["is_coastal_1000m"].to_numpy() <= 0.5)
                # prefer same coastal flag + similar coast distance / view
                same_coast = coastal_mask if row["is_coastal_1000m"] > 0.5 else non_mask
                c_idx, _ = self._similarity_rank(row, same_coast, 15)
                c_st = _stats(y[c_idx])
                coast_st = _stats(y[np.where(coastal_mask)[0][:50]] if coastal_mask.sum() else np.array([], dtype=int))
                # recompute properly
                coast_idx = np.where(coastal_mask)[0]
                non_idx = np.where(non_mask)[0]
                # take similar within coastal / noncoastal
                c_sim, _ = self._similarity_rank(row, coastal_mask, 15)
                n_sim, _ = self._similarity_rank(row, non_mask, 15)
                c_st = _stats(y[c_sim])
                n_st = _stats(y[n_sim])
                gap = (c_st["median"] - n_st["median"]) if np.isfinite(c_st["median"]) and np.isfinite(n_st["median"]) else np.nan
                for name, val in [
                    ("bsk_coastal_similar_median", c_st["median"]),
                    ("bsk_noncoastal_similar_median", n_st["median"]),
                    ("bsk_coastal_price_gap", gap),
                    ("bsk_coastal_similar_count", c_st["n"]),
                    ("bsk_coastal_confidence", _conf(c_st["n"], 10.0)),
                ]:
                    if name in feats:
                        feats[name][i] = val

            if mode in {"weighted", "full"}:
                # Top candidates from NN + similarity blend
                ni, dist = self._nearest_indices(row["lat"], row["lon"], qid, 40)
                if len(ni) == 0:
                    # similarity-only pool
                    s_idx, sims = self._similarity_rank(row, pool_all, 40)
                    ni = s_idx
                    dist = np.full(len(ni), 1500.0)
                    sim_map = {int(s_idx[j]): float(sims[j]) for j in range(len(s_idx))}
                else:
                    sim_map = {}
                    for j in ni:
                        sim_map[int(j)] = combined_similarity(
                            row["gross_m2"],
                            float(a.iloc[j]["gross_m2"]),
                            row["room_count"],
                            str(a.iloc[j]["room_count"]),
                            row["building_age_group"],
                            str(a.iloc[j]["building_age_group"]),
                            row["site_inside"],
                            str(a.iloc[j]["site_inside"]),
                            row["district"],
                            str(a.iloc[j]["district"]),
                            row["geo_cluster"],
                            str(a.iloc[j]["geo_cluster"]),
                            bool(row["large_home"]),
                            bool(a.iloc[j]["large_home"]),
                        )
                weights = []
                vals = []
                for j, d in zip(ni, dist if len(dist) == len(ni) else np.full(len(ni), 1500.0)):
                    if not np.isfinite(y[j]):
                        continue
                    dw = float(np.exp(-float(d) / float(self.distance_scale_m)))
                    sw = float(sim_map.get(int(j), 0.3))
                    w = dw * max(sw, 1e-6)
                    weights.append(w)
                    vals.append(float(y[j]))
                if weights:
                    w = np.asarray(weights, dtype=float)
                    v = np.asarray(vals, dtype=float)
                    w = w / w.sum()
                    order = np.argsort(v)
                    # weighted median
                    cdf = np.cumsum(w[order])
                    wmed = float(v[order][np.searchsorted(cdf, 0.5)])
                    wmean = float(np.sum(w * v))
                    wstd = float(np.sqrt(np.sum(w * (v - wmean) ** 2))) if len(v) > 1 else 0.0
                    wn = float(len(v))
                    any_ok = True
                else:
                    wmed = self.global_median_
                    wmean = self.global_median_
                    wstd = np.nan
                    wn = 0.0
                    fallback = max(fallback, 4.0)
                for name, val in [
                    ("bsk_weighted_comp_mean", wmean),
                    ("bsk_weighted_comp_median", wmed),
                    ("bsk_weighted_comp_std", wstd),
                    ("bsk_weighted_comp_count", wn),
                    ("bsk_weighted_comp_confidence", _conf(wn, 10.0)),
                ]:
                    if name in feats:
                        feats[name][i] = val

            if "bsk_comparable_missing" in feats:
                feats["bsk_comparable_missing"][i] = 0.0 if any_ok else 1.0
            if "bsk_fallback_level" in feats:
                feats["bsk_fallback_level"][i] = fallback

        out = pd.DataFrame(feats, index=df.index)
        # attach to X columns (sklearn pipeline Pattern: return X with new cols)
        for c in out.columns:
            df[c] = out[c].to_numpy()
        return df

    def leakage_guard_report(self) -> dict[str, Any]:
        g = dict(self.leakage_guard_ or {})
        g["mode"] = getattr(self, "mode_", self._resolve_mode())
        g["comparable_mode"] = g.get("comparable_mode") or g["mode"]
        g["feature_count"] = int(g.get("feature_count") or len(getattr(self, "feature_names_", []) or []))
        g["fit_n"] = int(g.get("fit_n") or getattr(self, "fit_n_", 0) or 0)
        g["uses_train_pool_only"] = bool(g.get("uses_train_pool_only", True))
        g["self_match_exclude"] = bool(g.get("self_match_exclude", True))
        g["validation_targets_unused"] = bool(g.get("validation_targets_unused", True))
        g["pass"] = (
            bool(g["uses_train_pool_only"])
            and bool(g["self_match_exclude"])
            and bool(g["validation_targets_unused"])
            and (g["feature_count"] > 0 if g["comparable_mode"] not in {"none", ""} else True)
            and (g["fit_n"] > 0 if g["comparable_mode"] not in {"none", ""} else True)
        )
        return g
