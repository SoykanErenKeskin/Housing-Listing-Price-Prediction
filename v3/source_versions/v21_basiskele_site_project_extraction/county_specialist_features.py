"""V16 Başiskele premium specialist + large-home redesign features.

Deterministic transformers (no target) and fold-safe target-stats adder.
App-safe only: uses existing listing / attr / detail_effect columns.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

BASISKELE = "Başiskele"

BASISKELE_PREMIUM_NUMERIC_FEATURES = [
    "basiskele_detail_outside_premium_signal",
    "basiskele_detail_view_premium_signal",
    "basiskele_detail_inside_premium_signal",
    "basiskele_detail_subtype_premium_signal",
    "basiskele_detail_total_premium_signal",
    "basiskele_has_pool_signal",
    "basiskele_has_security_signal",
    "basiskele_has_social_luxury_signal",
    "basiskele_has_view_signal",
    "basiskele_has_inside_luxury_signal",
    "basiskele_site_premium_signal",
    "basiskele_premium_score",
    "basiskele_premium_bucket_low",
    "basiskele_premium_bucket_mid",
    "basiskele_premium_bucket_high",
    "basiskele_premium_x_location_baseline",
    "basiskele_premium_x_gross_m2",
    "basiskele_premium_x_large_home",
    "basiskele_premium_x_new_building",
    "basiskele_premium_x_site_inside",
    "basiskele_premium_x_district_target_median",
    "basiskele_premium_x_detail_outside",
    "basiskele_premium_x_detail_view",
]

BASISKELE_TARGET_STAT_FEATURES = [
    "basiskele_district_premium_bucket_median",
    "basiskele_district_premium_bucket_mean",
    "basiskele_district_premium_bucket_count",
    "basiskele_premium_bucket_county_median",
    "basiskele_premium_bucket_county_mean",
    "basiskele_detail_group_bucket_median",
    "basiskele_large_home_premium_bucket_median",
]

LARGE_HOME_NUMERIC_FEATURES = [
    "large_home_m2_excess",
    "large_home_log_m2",
    "large_home_net_gross_quality",
    "large_home_room_density",
    "large_home_quality_x_m2",
    "large_home_detail_premium_x_m2",
    "large_home_site_x_m2",
    "large_home_basiskele_premium",
]


def parse_county_min_rows_overrides(raw: str | None) -> dict[str, int]:
    """Parse 'Karamürsel:180,Gölcük:200' into dict. Invalid entries skipped with warning."""
    out: dict[str, int] = {}
    if not raw or not str(raw).strip():
        return out
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            import warnings

            warnings.warn(f"Ignoring invalid county override (no colon): {part!r}")
            continue
        county, val = part.rsplit(":", 1)
        county = county.strip()
        try:
            out[county] = int(float(val.strip()))
        except Exception:
            import warnings

            warnings.warn(f"Ignoring invalid county override value: {part!r}")
    return out


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _bin(df: pd.DataFrame, col: str) -> pd.Series:
    return _num(df, col, 0.0).clip(0, 1)


def _is_basiskele(df: pd.DataFrame) -> pd.Series:
    if "county" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["county"].astype(str).eq(BASISKELE)


def _rooms_numeric(df: pd.DataFrame) -> pd.Series:
    if "room_count" not in df.columns:
        return pd.Series(3.0, index=df.index)
    s = df["room_count"].astype(str).str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(s, errors="coerce").fillna(3.0)


class LargeHomeFeatureAdder(BaseEstimator, TransformerMixin):
    """Deterministic large-home redesign features (all counties)."""

    def fit(self, X: pd.DataFrame, y: Any = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        gross = _num(df, "gross_m2", 0.0)
        is_large = (gross >= 150).astype(float)
        if "is_large_flat" in df.columns:
            is_large = np.maximum(is_large, _num(df, "is_large_flat", 0.0).clip(0, 1))
        df["is_large_flat"] = is_large
        df["large_home_m2_excess"] = np.maximum(gross - 150.0, 0.0) * is_large
        df["large_home_log_m2"] = np.log1p(np.maximum(gross, 0.0)) * is_large
        net_ratio = _num(df, "net_gross_ratio", np.nan)
        if net_ratio.isna().all() and "net_m2" in df.columns:
            net_ratio = _num(df, "net_m2", 0.0) / np.maximum(gross, 1.0)
        net_ratio = net_ratio.fillna(0.8).clip(0, 1.5)
        df["large_home_net_gross_quality"] = net_ratio * is_large
        rooms = _rooms_numeric(df)
        df["large_home_room_density"] = (rooms / np.maximum(gross, 1.0)) * is_large
        attr_q = _num(df, "attr_total_quality_score", 0.0)
        df["large_home_quality_x_m2"] = attr_q * gross * is_large
        detail = _num(df, "detail_effect_total_sum", 0.0)
        df["large_home_detail_premium_x_m2"] = detail * gross * is_large
        site = _num(df, "attr_is_site_inside", 0.0)
        if site.abs().sum() == 0:
            site = (_num(df, "site_inside", 0.0) > 0).astype(float)
            # also accept string Evet
            if "site_inside" in df.columns and df["site_inside"].dtype == object:
                site = df["site_inside"].astype(str).str.lower().isin(["evet", "var", "1", "true"]).astype(float)
        df["large_home_site_x_m2"] = site.clip(0, 1) * gross * is_large
        prem = _num(df, "basiskele_premium_score", 0.0)
        df["large_home_basiskele_premium"] = is_large * prem
        missing = [c for c in LARGE_HOME_NUMERIC_FEATURES if c not in df.columns]
        if missing:
            df = pd.concat([df, pd.DataFrame({c: 0.0 for c in missing}, index=df.index)], axis=1)
        return df


class BasiskelePremiumSpecialistAdder(BaseEstimator, TransformerMixin):
    """Deterministic Başiskele premium signals (no target). Quantile buckets fit on X only."""

    def __init__(self, mode: str = "premium", low_q: float = 0.30, high_q: float = 0.70):
        self.mode = mode
        self.low_q = low_q
        self.high_q = high_q

    def fit(self, X: pd.DataFrame, y: Any = None):
        self.enabled_ = str(self.mode or "none").lower() not in {"", "none"}
        self.low_thr_ = 0.0
        self.high_thr_ = 1.0
        if not self.enabled_:
            return self
        df = self._compute_score_frame(X.copy(), fit_buckets=False)
        mask = _is_basiskele(df)
        scores = df.loc[mask, "basiskele_premium_score"]
        if len(scores) >= 20:
            self.low_thr_ = float(np.nanquantile(scores, self.low_q))
            self.high_thr_ = float(np.nanquantile(scores, self.high_q))
            if self.high_thr_ <= self.low_thr_:
                self.high_thr_ = self.low_thr_ + 1e-6
        else:
            self.low_thr_ = 0.33
            self.high_thr_ = 0.66
        return self

    def _compute_score_frame(self, df: pd.DataFrame, fit_buckets: bool) -> pd.DataFrame:
        mask = _is_basiskele(df).astype(float).to_numpy(dtype=float)
        outside = np.tanh(_num(df, "detail_effect_outside_sum", 0.0).to_numpy(dtype=float) * 3.0)
        view = np.tanh(_num(df, "detail_effect_view_sum", 0.0).to_numpy(dtype=float) * 3.0)
        inside = np.tanh(_num(df, "detail_effect_inside_sum", 0.0).to_numpy(dtype=float) * 3.0)
        subtype = np.tanh(_num(df, "detail_effect_subtype_sum", 0.0).to_numpy(dtype=float) * 3.0)
        total = np.tanh(_num(df, "detail_effect_total_sum", 0.0).to_numpy(dtype=float) * 3.0)

        df["basiskele_detail_outside_premium_signal"] = outside * mask
        df["basiskele_detail_view_premium_signal"] = view * mask
        df["basiskele_detail_inside_premium_signal"] = inside * mask
        df["basiskele_detail_subtype_premium_signal"] = subtype * mask
        df["basiskele_detail_total_premium_signal"] = total * mask

        pool = np.maximum.reduce(
            [
                _bin(df, "out_pool").to_numpy(),
                _bin(df, "out_open_pool").to_numpy(),
                _bin(df, "out_closed_pool").to_numpy(),
                _bin(df, "view_pool").to_numpy(),
            ]
        )
        security = np.maximum(_bin(df, "out_security").to_numpy(), _bin(df, "out_camera").to_numpy())
        social = np.maximum.reduce(
            [
                _bin(df, "out_sauna_hamam").to_numpy(),
                _bin(df, "out_sports_area").to_numpy(),
                _bin(df, "out_children_playground").to_numpy(),
                _bin(df, "out_generator").to_numpy(),
            ]
        )
        view_flag = np.maximum.reduce(
            [
                _bin(df, "view_sea").to_numpy(),
                _bin(df, "view_nature").to_numpy(),
                _bin(df, "view_lake").to_numpy(),
                _bin(df, "view_pool").to_numpy(),
                _bin(df, "view_city").to_numpy(),
            ]
        )
        inside_lux = np.maximum.reduce(
            [
                _bin(df, "in_builtin_kitchen").to_numpy(),
                _bin(df, "in_parent_bathroom").to_numpy(),
                _bin(df, "in_glass_balcony").to_numpy(),
                _bin(df, "in_air_conditioner").to_numpy(),
                _bin(df, "in_smart_home").to_numpy(),
            ]
        )
        site = _bin(df, "attr_is_site_inside").to_numpy()
        if float(np.nansum(site)) == 0 and "site_inside" in df.columns:
            site = df["site_inside"].astype(str).str.lower().isin(["evet", "var", "1", "true"]).astype(float).to_numpy()

        df["basiskele_has_pool_signal"] = pool * mask
        df["basiskele_has_security_signal"] = security * mask
        df["basiskele_has_social_luxury_signal"] = social * mask
        df["basiskele_has_view_signal"] = view_flag * mask
        df["basiskele_has_inside_luxury_signal"] = inside_lux * mask
        site_prem = np.clip(site + 0.5 * security + 0.5 * pool + 0.25 * social, 0, 2) / 2.0
        df["basiskele_site_premium_signal"] = site_prem * mask

        attr_q = _num(df, "attr_total_quality_score", 0.0).to_numpy(dtype=float)
        attr_q_n = np.tanh(attr_q / 10.0)
        flags = 0.25 * pool + 0.25 * security + 0.25 * social + 0.25 * view_flag
        score = (
            0.25 * outside
            + 0.20 * view
            + 0.15 * inside
            + 0.15 * subtype
            + 0.15 * attr_q_n
            + 0.10 * (0.5 * site_prem + 0.5 * flags)
        )
        # map tanh-ish mix into roughly [0,1]
        score = (np.tanh(score) + 1.0) / 2.0
        df["basiskele_premium_score"] = score * mask

        if fit_buckets:
            low = float(getattr(self, "low_thr_", 0.33))
            high = float(getattr(self, "high_thr_", 0.66))
            s = df["basiskele_premium_score"].to_numpy(dtype=float)
            df["basiskele_premium_bucket_low"] = ((s <= low) & (mask > 0)).astype(float)
            df["basiskele_premium_bucket_high"] = ((s >= high) & (mask > 0)).astype(float)
            df["basiskele_premium_bucket_mid"] = (
                (mask > 0)
                & (df["basiskele_premium_bucket_low"].to_numpy() == 0)
                & (df["basiskele_premium_bucket_high"].to_numpy() == 0)
            ).astype(float)
            bucket = np.where(
                df["basiskele_premium_bucket_high"].to_numpy() == 1,
                "high",
                np.where(df["basiskele_premium_bucket_low"].to_numpy() == 1, "low", "mid"),
            )
            df["basiskele_premium_bucket"] = np.where(mask > 0, bucket, "none")
        return df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if not getattr(self, "enabled_", False):
            missing = {c: 0.0 for c in BASISKELE_PREMIUM_NUMERIC_FEATURES if c not in df.columns}
            if "basiskele_premium_bucket" not in df.columns:
                missing["basiskele_premium_bucket"] = "none"
            if missing:
                df = pd.concat([df, pd.DataFrame(missing, index=df.index)], axis=1)
            return df

        df = self._compute_score_frame(df, fit_buckets=True)
        mask = _is_basiskele(df).astype(float)
        prem = _num(df, "basiskele_premium_score", 0.0)
        gross = _num(df, "gross_m2", 0.0)
        is_large = _num(df, "is_large_flat", 0.0)
        if is_large.abs().sum() == 0:
            is_large = (gross >= 150).astype(float)

        if "location_baseline_m2" in df.columns:
            base = np.log1p(_num(df, "location_baseline_m2", 0.0).clip(lower=0))
        else:
            base = _num(df, "county_target_median", 0.0)
        df["basiskele_premium_x_location_baseline"] = prem * base * mask
        df["basiskele_premium_x_gross_m2"] = prem * gross * mask
        df["basiskele_premium_x_large_home"] = prem * is_large * mask
        new_b = _num(df, "attr_is_new_building", 0.0)
        df["basiskele_premium_x_new_building"] = prem * new_b * mask
        site = _num(df, "attr_is_site_inside", 0.0)
        df["basiskele_premium_x_site_inside"] = prem * site * mask
        dist_med = _num(df, "district_target_median", 0.0)
        df["basiskele_premium_x_district_target_median"] = prem * dist_med * mask
        df["basiskele_premium_x_detail_outside"] = prem * _num(df, "basiskele_detail_outside_premium_signal", 0.0) * mask
        df["basiskele_premium_x_detail_view"] = prem * _num(df, "basiskele_detail_view_premium_signal", 0.0) * mask

        missing = [c for c in BASISKELE_PREMIUM_NUMERIC_FEATURES if c not in df.columns]
        if missing:
            df = pd.concat([df, pd.DataFrame({c: 0.0 for c in missing}, index=df.index)], axis=1)
        return df


class BasiskelePremiumTargetStatsAdder(BaseEstimator, TransformerMixin):
    """Fold-safe residual stats by Başiskele premium bucket × district (uses y)."""

    def __init__(self, mode: str = "premium_target_stats", alpha: float = 50.0, min_count: int = 30):
        self.mode = mode
        self.alpha = alpha
        self.min_count = min_count

    def fit(self, X: pd.DataFrame, y: Any):
        mode = str(self.mode or "none").lower()
        self.enabled_ = mode in {"premium_target_stats", "premium_target_stats_variance_lift"}
        self.global_median_ = 0.0
        self.maps_: dict[str, dict[str, float]] = {}
        if not self.enabled_:
            return self

        df = X.copy()
        if "basiskele_premium_bucket" not in df.columns:
            # ensure buckets exist
            adder = BasiskelePremiumSpecialistAdder(mode="premium")
            adder.fit(df)
            df = adder.transform(df)

        y_series = pd.Series(np.asarray(y, dtype=float), index=df.index, name="__premium__")
        valid = y_series.notna() & np.isfinite(y_series)
        work = df.loc[valid].copy()
        work["__premium__"] = y_series.loc[valid]
        mask = _is_basiskele(work)
        work = work.loc[mask].copy()
        if work.empty:
            self.global_median_ = float(np.nanmedian(y_series)) if len(y_series) else 0.0
            return self

        self.global_median_ = float(np.nanmedian(work["__premium__"]))
        alpha = float(self.alpha)

        def _fit_group(cols: list[str], out_name: str, use_mean: bool = False):
            cols = [c for c in cols if c in work.columns]
            if not cols:
                self.maps_[out_name] = {}
                return
            g = work.groupby(cols, dropna=False)["__premium__"].agg(["median", "mean", "count"]).reset_index()
            mapping = {}
            for _, row in g.iterrows():
                key = "||".join(str(row[c]) for c in cols)
                n = float(row["count"])
                local = float(row["mean"] if use_mean else row["median"])
                mapping[key] = float((n / (n + alpha)) * local + (alpha / (n + alpha)) * self.global_median_)
            self.maps_[out_name] = mapping

        if "district" not in work.columns:
            work["district"] = "missing"
        work["district"] = work["district"].astype(str).fillna("missing")
        work["basiskele_premium_bucket"] = work["basiskele_premium_bucket"].astype(str).fillna("mid")
        if "m2_group" not in work.columns:
            work["m2_group"] = "missing"
        if "is_large_flat" not in work.columns:
            work["is_large_flat"] = (_num(work, "gross_m2", 0.0) >= 150).astype(int)

        _fit_group(["district", "basiskele_premium_bucket"], "basiskele_district_premium_bucket_median")
        _fit_group(["district", "basiskele_premium_bucket"], "basiskele_district_premium_bucket_mean", use_mean=True)
        # count map (unsmoothed)
        gcnt = work.groupby(["district", "basiskele_premium_bucket"], dropna=False).size().reset_index(name="count")
        self.maps_["basiskele_district_premium_bucket_count"] = {
            f"{r['district']}||{r['basiskele_premium_bucket']}": float(r["count"]) for _, r in gcnt.iterrows()
        }
        _fit_group(["basiskele_premium_bucket"], "basiskele_premium_bucket_county_median")
        _fit_group(["basiskele_premium_bucket"], "basiskele_premium_bucket_county_mean", use_mean=True)
        _fit_group(["district", "m2_group", "basiskele_premium_bucket"], "basiskele_detail_group_bucket_median")
        _fit_group(["is_large_flat", "basiskele_premium_bucket"], "basiskele_large_home_premium_bucket_median")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if not getattr(self, "enabled_", False):
            missing = [c for c in BASISKELE_TARGET_STAT_FEATURES if c not in df.columns]
            if missing:
                df = pd.concat([df, pd.DataFrame({c: 0.0 for c in missing}, index=df.index)], axis=1)
            return df

        if "basiskele_premium_bucket" not in df.columns:
            adder = BasiskelePremiumSpecialistAdder(mode="premium")
            # use stored thresholds if available via a no-op; buckets mid/none without fit is ok for missing
            if not hasattr(adder, "low_thr_"):
                adder.low_thr_ = 0.33
                adder.high_thr_ = 0.66
                adder.enabled_ = True
            df = adder.transform(df)

        mask = _is_basiskele(df)
        if "district" not in df.columns:
            df["district"] = "missing"
        district = df["district"].astype(str).fillna("missing")
        bucket = df["basiskele_premium_bucket"].astype(str).fillna("none")
        m2g = df["m2_group"].astype(str).fillna("missing") if "m2_group" in df.columns else pd.Series("missing", index=df.index)
        is_large = _num(df, "is_large_flat", 0.0).fillna(0).astype(int).astype(str)

        def lookup(map_name: str, keys: pd.Series, fallback_keys: pd.Series | None = None) -> pd.Series:
            m = self.maps_.get(map_name, {})
            vals = keys.map(m)
            if fallback_keys is not None:
                vals = vals.fillna(fallback_keys.map(self.maps_.get("basiskele_premium_bucket_county_median", {})))
            return vals.fillna(self.global_median_)

        dkey = district + "||" + bucket
        df["basiskele_district_premium_bucket_median"] = np.where(
            mask, lookup("basiskele_district_premium_bucket_median", dkey, bucket), 0.0
        )
        df["basiskele_district_premium_bucket_mean"] = np.where(
            mask, lookup("basiskele_district_premium_bucket_mean", dkey, bucket), 0.0
        )
        cnt = dkey.map(self.maps_.get("basiskele_district_premium_bucket_count", {})).fillna(0.0)
        df["basiskele_district_premium_bucket_count"] = np.where(mask, cnt, 0.0)
        df["basiskele_premium_bucket_county_median"] = np.where(
            mask, bucket.map(self.maps_.get("basiskele_premium_bucket_county_median", {})).fillna(self.global_median_), 0.0
        )
        df["basiskele_premium_bucket_county_mean"] = np.where(
            mask, bucket.map(self.maps_.get("basiskele_premium_bucket_county_mean", {})).fillna(self.global_median_), 0.0
        )
        gkey = district + "||" + m2g + "||" + bucket
        df["basiskele_detail_group_bucket_median"] = np.where(
            mask, lookup("basiskele_detail_group_bucket_median", gkey, bucket), 0.0
        )
        lkey = is_large + "||" + bucket
        df["basiskele_large_home_premium_bucket_median"] = np.where(
            mask,
            lkey.map(self.maps_.get("basiskele_large_home_premium_bucket_median", {})).fillna(self.global_median_),
            0.0,
        )
        missing = [c for c in BASISKELE_TARGET_STAT_FEATURES if c not in df.columns]
        if missing:
            df = pd.concat([df, pd.DataFrame({c: 0.0 for c in missing}, index=df.index)], axis=1)
        return df


def get_county_specialist_feature_names(basiskele_mode: str = "premium_target_stats") -> list[str]:
    mode = str(basiskele_mode or "none").lower()
    names = list(LARGE_HOME_NUMERIC_FEATURES)
    if mode in {"premium", "premium_target_stats", "premium_target_stats_variance_lift"}:
        names = names + list(BASISKELE_PREMIUM_NUMERIC_FEATURES)
    if mode in {"premium_target_stats", "premium_target_stats_variance_lift"}:
        names = names + list(BASISKELE_TARGET_STAT_FEATURES)
    # unique preserve order
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out
