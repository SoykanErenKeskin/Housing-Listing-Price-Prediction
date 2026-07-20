"""V17 fold-safe comparable / emsal market features.

Uses target (unit price or residual) — MUST be fit inside CV folds only.
Self-match exclusion on transform of training rows.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors

from location_features import haversine_m

COMPARABLE_NUMERIC_FEATURES = [
    # Dar (similarity)
    "similar_5_median",
    "similar_10_median",
    "similar_20_median",
    "similar_10_mean",
    "similar_10_std",
    "similar_10_iqr",
    "similar_10_count",
    "similar_confidence_score",
    # Derin (nearest)
    "nearest_5_median",
    "nearest_10_median",
    "nearest_20_median",
    "nearest_10_mean",
    "nearest_10_std",
    "nearest_10_count",
    "nearest_10_avg_distance_m",
    "nearest_distance_confidence",
    # Yaygın (broad)
    "county_recent_similar_median",
    "district_recent_similar_median",
    "broad_similar_median",
    "broad_similar_count",
    "broad_similarity_confidence",
    # Weighted
    "weighted_comp_median",
    "weighted_comp_mean",
    "weighted_comp_std",
    "weighted_comp_count",
    "weighted_comp_confidence",
    # Başiskele special
    "bsk_nearest_10_median",
    "bsk_large_home_nearest_10_median",
    "bsk_200p_nearest_10_median",
    "bsk_same_geo_cluster_median",
    "bsk_same_geo_cluster_large_home_median",
    "bsk_coastal_similar_median",
    "bsk_noncoastal_similar_median",
    "bsk_geo_comp_confidence",
    "bsk_comparable_missing",
]


def get_comparable_feature_names(mode: str) -> list[str]:
    m = str(mode or "none").lower()
    if m in {"comparable", "full"}:
        return list(COMPARABLE_NUMERIC_FEATURES)
    return []


def parse_k_list(raw: str | list[int] | None) -> list[int]:
    if raw is None:
        return [5, 10, 20]
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            continue
    return out or [5, 10, 20]


def _safe_str(s: pd.Series) -> pd.Series:
    return s.astype("object").where(s.notna(), "missing").astype(str).str.strip().replace({"": "missing", "nan": "missing"})


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(dtype=float)


class ComparableMarketFeatureAdder(BaseEstimator, TransformerMixin):
    """Fold-safe comparable market stats.

    fit(X_train, y_train): stores train coords / attributes / targets.
    transform(X): computes stats from train pool only; excludes self-matches.
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
        self.k_list_: list[int] = [5, 10, 20]
        self.train_lat_: np.ndarray | None = None
        self.train_lon_: np.ndarray | None = None
        self.train_valid_: np.ndarray | None = None
        self.train_y_: np.ndarray | None = None
        self.train_ids_: np.ndarray | None = None
        self.train_attrs_: pd.DataFrame | None = None
        self.nn_: NearestNeighbors | None = None
        self.fit_n_: int = 0
        self._transforming_train_: bool = False

    def _enabled(self) -> bool:
        return str(self.mode or "none").lower() in {"comparable", "full"}

    def fit(self, X: pd.DataFrame, y: Any = None):
        self.enabled_ = self._enabled()
        self.k_list_ = parse_k_list(self.k_list)
        if not self.enabled_:
            return self
        if y is None:
            raise ValueError("ComparableMarketFeatureAdder requires y in fit().")

        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if len(y_arr) != len(df):
            raise ValueError(f"y length {len(y_arr)} != X length {len(df)}")

        lat = pd.to_numeric(df.get("lat", df.get("latitude", np.nan)), errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(df.get("lon", df.get("longitude", np.nan)), errors="coerce").to_numpy(dtype=float)
        # Prefer engineered has_lat_lon if present
        if "has_lat_lon" in df.columns:
            valid = pd.to_numeric(df["has_lat_lon"], errors="coerce").fillna(0).to_numpy(dtype=float) > 0.5
            valid = valid & np.isfinite(lat) & np.isfinite(lon)
        else:
            valid = np.isfinite(lat) & np.isfinite(lon)

        self.train_lat_ = lat
        self.train_lon_ = lon
        self.train_valid_ = valid
        self.train_y_ = y_arr
        if "classified_id" in df.columns:
            self.train_ids_ = df["classified_id"].astype(str).to_numpy()
        else:
            self.train_ids_ = np.array([f"idx_{i}" for i in range(len(df))], dtype=object)

        attrs = pd.DataFrame(
            {
                "county": _safe_str(df["county"]) if "county" in df.columns else "missing",
                "district": _safe_str(df["district"]) if "district" in df.columns else "missing",
                "room_count": _safe_str(df["room_count"]) if "room_count" in df.columns else "missing",
                "building_age_group": _safe_str(df["building_age_group"]) if "building_age_group" in df.columns else "missing",
                "site_inside": _safe_str(df["site_inside"]) if "site_inside" in df.columns else "missing",
                "real_estate_type": _safe_str(df["real_estate_type"]) if "real_estate_type" in df.columns else "missing",
                "m2_group": _safe_str(df["m2_group"]) if "m2_group" in df.columns else "missing",
                "geo_cluster_county": _safe_str(df["geo_cluster_county"]) if "geo_cluster_county" in df.columns else "other",
                "gross_m2": _num(df, "gross_m2", 0.0),
                "location_quality_score": _num(df, "location_quality_score", 0.0),
                "distance_to_coastline_m": _num(df, "distance_to_coastline_m", 99999.0),
                "is_coastal_1000m": _num(df, "is_coastal_1000m", 0.0),
                "large_home": (
                    _num(df, "is_large_home", 0.0)
                    if "is_large_home" in df.columns
                    else (_num(df, "gross_m2", 0.0) >= 160).astype(float)
                ),
            }
        )
        self.train_attrs_ = attrs
        self.fit_n_ = len(df)

        if valid.sum() >= 5:
            coords = np.column_stack([lat[valid], lon[valid]])
            # Use haversine via BallTree with haversine metric needs radians;
            # NearestNeighbors euclidean on lat/lon is OK for local Kocaeli scale as proxy,
            # but we re-rank with haversine for stats. Fit NN on (lat, lon) degrees.
            self.nn_ = NearestNeighbors(n_neighbors=min(int(max(self.k_list_) + 5), int(valid.sum())), algorithm="ball_tree")
            self.nn_.fit(coords)
        else:
            self.nn_ = None
        return self

    def fit_transform(self, X: pd.DataFrame, y: Any = None, **fit_params):
        self._transforming_train_ = True
        try:
            return self.fit(X, y).transform(X)
        finally:
            self._transforming_train_ = False

    def _similarity_scores(self, row: pd.Series) -> np.ndarray:
        assert self.train_attrs_ is not None
        ta = self.train_attrs_
        score = np.zeros(self.fit_n_, dtype=float)
        score += (ta["county"].to_numpy() == str(row.get("county", "missing"))).astype(float) * 3.0
        score += (ta["district"].to_numpy() == str(row.get("district", "missing"))).astype(float) * 2.0
        score += (ta["room_count"].to_numpy() == str(row.get("room_count", "missing"))).astype(float) * 2.0
        score += (ta["building_age_group"].to_numpy() == str(row.get("building_age_group", "missing"))).astype(float) * 1.0
        score += (ta["site_inside"].to_numpy() == str(row.get("site_inside", "missing"))).astype(float) * 1.0
        score += (ta["real_estate_type"].to_numpy() == str(row.get("real_estate_type", "missing"))).astype(float) * 1.0
        m2 = float(pd.to_numeric(row.get("gross_m2", 0), errors="coerce") or 0)
        dm2 = np.abs(ta["gross_m2"].to_numpy(dtype=float) - m2)
        score += np.clip(1.0 - dm2 / 40.0, 0.0, 1.0) * 1.5
        large = 1.0 if m2 >= 160 else 0.0
        if "is_large_home" in row.index:
            large = float(pd.to_numeric(row.get("is_large_home", large), errors="coerce") or large)
        score += (ta["large_home"].to_numpy(dtype=float) == large).astype(float) * 1.0
        return score

    def _exclude_mask(self, i_query: int | None, query_id: str | None, query_lat: float, query_lon: float, query_feats: np.ndarray | None) -> np.ndarray:
        """Boolean mask True = keep neighbor."""
        keep = np.ones(self.fit_n_, dtype=bool)
        if self.train_ids_ is not None and query_id is not None:
            keep &= self.train_ids_ != str(query_id)
        if self._transforming_train_ and i_query is not None and 0 <= i_query < self.fit_n_:
            keep[i_query] = False
        # distance≈0 self-match fallback
        if self.train_valid_ is not None and np.isfinite(query_lat) and np.isfinite(query_lon):
            d = haversine_m(
                np.full(self.fit_n_, query_lat),
                np.full(self.fit_n_, query_lon),
                self.train_lat_,
                self.train_lon_,
            )
            near_zero = self.train_valid_ & np.isfinite(d) & (d < 1.0)
            # if multiple exact same coords, still allow others unless same id already excluded
            if near_zero.sum() == 1 and self._transforming_train_:
                keep &= ~near_zero
        return keep

    @staticmethod
    def _stats(values: np.ndarray, k: int) -> dict[str, float]:
        if values.size == 0:
            return {"median": 0.0, "mean": 0.0, "std": 0.0, "iqr": 0.0, "count": 0.0}
        take = values[: min(k, values.size)]
        q75, q25 = np.percentile(take, [75, 25]) if take.size >= 2 else (take[0], take[0])
        return {
            "median": float(np.median(take)),
            "mean": float(np.mean(take)),
            "std": float(np.std(take)) if take.size >= 2 else 0.0,
            "iqr": float(q75 - q25),
            "count": float(take.size),
        }

    def _row_features(self, row: pd.Series, i_query: int | None) -> dict[str, float]:
        assert self.train_y_ is not None and self.train_attrs_ is not None
        out = {c: 0.0 for c in COMPARABLE_NUMERIC_FEATURES}
        out["bsk_comparable_missing"] = 1.0

        lat = float(pd.to_numeric(row.get("lat", row.get("latitude", np.nan)), errors="coerce"))
        lon = float(pd.to_numeric(row.get("lon", row.get("longitude", np.nan)), errors="coerce"))
        qid = str(row.get("classified_id", f"q_{i_query}")) if "classified_id" in row.index or i_query is not None else None
        quality = float(pd.to_numeric(row.get("location_quality_score", 0.0), errors="coerce") or 0.0)
        keep = self._exclude_mask(i_query, qid, lat, lon, None)

        # Similarity ranking
        sim = self._similarity_scores(row)
        sim = np.where(keep, sim, -1.0)
        order_sim = np.argsort(-sim)
        order_sim = order_sim[sim[order_sim] > 0]
        y_sim = self.train_y_[order_sim]

        for k in self.k_list_:
            st = self._stats(y_sim, k)
            if k == 5:
                out["similar_5_median"] = st["median"]
            if k == 10:
                out["similar_10_median"] = st["median"]
                out["similar_10_mean"] = st["mean"]
                out["similar_10_std"] = st["std"]
                out["similar_10_iqr"] = st["iqr"]
                out["similar_10_count"] = st["count"]
            if k == 20:
                out["similar_20_median"] = st["median"]
        out["similar_confidence_score"] = float(
            min(1.0, out["similar_10_count"] / 10.0) * (0.5 + 0.5 * quality)
        )

        # Nearest by distance (same county preferred)
        county = str(row.get("county", "missing"))
        dists = np.full(self.fit_n_, np.inf)
        if np.isfinite(lat) and np.isfinite(lon) and self.train_valid_ is not None:
            dists = haversine_m(
                np.full(self.fit_n_, lat),
                np.full(self.fit_n_, lon),
                self.train_lat_,
                self.train_lon_,
            )
            dists = np.where(self.train_valid_ & keep, dists, np.inf)
            # penalize other counties
            other = self.train_attrs_["county"].to_numpy() != county
            dists = np.where(other, dists + 50_000.0, dists)

        order_nn = np.argsort(dists)
        finite = np.isfinite(dists[order_nn]) & (dists[order_nn] < 1e12)
        order_nn = order_nn[finite]
        y_nn = self.train_y_[order_nn]
        d_nn = dists[order_nn]

        for k in self.k_list_:
            st = self._stats(y_nn, k)
            if k == 5:
                out["nearest_5_median"] = st["median"]
            if k == 10:
                out["nearest_10_median"] = st["median"]
                out["nearest_10_mean"] = st["mean"]
                out["nearest_10_std"] = st["std"]
                out["nearest_10_count"] = st["count"]
                out["nearest_10_avg_distance_m"] = float(np.mean(d_nn[: min(10, d_nn.size)])) if d_nn.size else 0.0
            if k == 20:
                out["nearest_20_median"] = st["median"]
        out["nearest_distance_confidence"] = float(
            min(1.0, out["nearest_10_count"] / 10.0)
            * (0.4 + 0.6 * quality)
            * (1.0 / (1.0 + out["nearest_10_avg_distance_m"] / 2000.0))
        )

        # Broad / county-district similar
        same_county = keep & (self.train_attrs_["county"].to_numpy() == county)
        same_dist = same_county & (self.train_attrs_["district"].to_numpy() == str(row.get("district", "missing")))
        room = str(row.get("room_count", "missing"))
        similar_county = same_county & (self.train_attrs_["room_count"].to_numpy() == room)
        y_c = self.train_y_[similar_county]
        y_d = self.train_y_[same_dist & (self.train_attrs_["room_count"].to_numpy() == room)]
        out["county_recent_similar_median"] = float(np.median(y_c)) if y_c.size else 0.0
        out["district_recent_similar_median"] = float(np.median(y_d)) if y_d.size else out["county_recent_similar_median"]
        out["broad_similar_median"] = out["similar_20_median"] or out["county_recent_similar_median"]
        out["broad_similar_count"] = float(min(y_c.size, 40))
        out["broad_similarity_confidence"] = float(min(1.0, y_c.size / 20.0) * (0.5 + 0.5 * quality))

        # Weighted comparable
        if order_nn.size:
            take_n = min(40, order_nn.size)
            idx = order_nn[:take_n]
            dist_w = np.exp(-d_nn[:take_n] / float(self.distance_scale_m))
            sim_w = sim[idx]
            sim_w = (sim_w - sim_w.min()) / (sim_w.max() - sim_w.min() + 1e-9) if sim_w.size else sim_w
            w = dist_w * (0.5 + 0.5 * sim_w)
            w = w / (w.sum() + 1e-12)
            yy = self.train_y_[idx]
            out["weighted_comp_median"] = float(yy[np.argsort(w)[len(w) // 2]]) if yy.size else 0.0
            # weighted median approx via sorting — use mean as primary + std
            out["weighted_comp_mean"] = float(np.sum(w * yy))
            out["weighted_comp_std"] = float(np.sqrt(np.sum(w * (yy - out["weighted_comp_mean"]) ** 2)))
            out["weighted_comp_count"] = float(take_n)
            out["weighted_comp_confidence"] = float(min(1.0, take_n / 20.0) * (0.5 + 0.5 * quality))
            # better weighted median
            order_w = np.argsort(yy)
            cdf = np.cumsum(w[order_w])
            med_i = int(np.searchsorted(cdf, 0.5))
            med_i = min(med_i, len(order_w) - 1)
            out["weighted_comp_median"] = float(yy[order_w[med_i]])

        # Başiskele specials
        if county == "Başiskele":
            out["bsk_comparable_missing"] = 0.0
            bsk_keep = keep & (self.train_attrs_["county"].to_numpy() == "Başiskele")
            if bsk_keep.any() and np.isfinite(lat) and np.isfinite(lon):
                bd = haversine_m(
                    np.full(self.fit_n_, lat),
                    np.full(self.fit_n_, lon),
                    self.train_lat_,
                    self.train_lon_,
                )
                bd = np.where(bsk_keep & self.train_valid_, bd, np.inf)
                bord = np.argsort(bd)
                bord = bord[np.isfinite(bd[bord])]
                yb = self.train_y_[bord]
                out["bsk_nearest_10_median"] = self._stats(yb, 10)["median"]

                large_mask = bsk_keep & (self.train_attrs_["large_home"].to_numpy() > 0.5)
                ld = np.where(large_mask & self.train_valid_, bd, np.inf)
                lord = np.argsort(ld)
                lord = lord[np.isfinite(ld[lord])]
                out["bsk_large_home_nearest_10_median"] = self._stats(self.train_y_[lord], 10)["median"]

                m200 = bsk_keep & (self.train_attrs_["gross_m2"].to_numpy() >= 200)
                d200 = np.where(m200 & self.train_valid_, bd, np.inf)
                o200 = np.argsort(d200)
                o200 = o200[np.isfinite(d200[o200])]
                out["bsk_200p_nearest_10_median"] = self._stats(self.train_y_[o200], 10)["median"]

                gc = str(row.get("geo_cluster_county", "other"))
                same_gc = bsk_keep & (self.train_attrs_["geo_cluster_county"].to_numpy() == gc)
                out["bsk_same_geo_cluster_median"] = (
                    float(np.median(self.train_y_[same_gc])) if same_gc.sum() >= 3 else out["bsk_nearest_10_median"]
                )
                same_gc_lh = same_gc & (self.train_attrs_["large_home"].to_numpy() > 0.5)
                out["bsk_same_geo_cluster_large_home_median"] = (
                    float(np.median(self.train_y_[same_gc_lh])) if same_gc_lh.sum() >= 3 else out["bsk_large_home_nearest_10_median"]
                )

                coastal = bsk_keep & (self.train_attrs_["is_coastal_1000m"].to_numpy() > 0.5)
                noncoastal = bsk_keep & ~coastal
                out["bsk_coastal_similar_median"] = (
                    float(np.median(self.train_y_[coastal])) if coastal.sum() >= 3 else 0.0
                )
                out["bsk_noncoastal_similar_median"] = (
                    float(np.median(self.train_y_[noncoastal])) if noncoastal.sum() >= 3 else 0.0
                )
                out["bsk_geo_comp_confidence"] = float(
                    min(1.0, bsk_keep.sum() / 30.0) * (0.4 + 0.6 * quality)
                )

        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if not self.enabled_:
            for c in COMPARABLE_NUMERIC_FEATURES:
                if c not in df.columns:
                    df[c] = 0.0
            return df

        rows = []
        # Detect if X is the same train frame (length match + transforming_train)
        same_train = self._transforming_train_ and len(df) == self.fit_n_
        for i in range(len(df)):
            row = df.iloc[i]
            i_query = i if same_train else None
            # also try id-based exclude always
            feats = self._row_features(row, i_query)
            rows.append(feats)

        feat_df = pd.DataFrame(rows, index=df.index)
        for c in COMPARABLE_NUMERIC_FEATURES:
            df[c] = feat_df[c].astype(float)
        return df
