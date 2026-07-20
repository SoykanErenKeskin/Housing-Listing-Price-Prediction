"""V16 fold-safe local detail premium features.

Learns residual premiums for listing detail binaries (front_/view_/…)
in county/district context. Must only fit on fold-train residual y —
never precompute on the full dataframe outside the sklearn Pipeline.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DETAIL_BINARY_PREFIXES = ("front_", "view_", "transport_", "near_", "out_", "in_", "subtype_")

DETAIL_GROUPS: dict[str, str] = {
    "front": "front_",
    "view": "view_",
    "transport": "transport_",
    "nearby": "near_",
    "outside": "out_",
    "inside": "in_",
    "subtype": "subtype_",
}

# Known SCORE_GROUPS members (union with prefix discovery).
SCORE_GROUP_MEMBERS: list[str] = [
    "front_west", "front_east", "front_south", "front_north",
    "view_bosphorus", "view_sea", "view_nature", "view_lake", "view_pool",
    "view_river", "view_park_green", "view_city",
    "transport_main_road", "transport_e5", "transport_tem", "transport_tram",
    "transport_train", "transport_bus_stop", "transport_minibus", "transport_metro",
    "transport_airport",
    "near_mall", "near_mosque", "near_pharmacy", "near_hospital", "near_school",
    "near_university", "near_market", "near_park", "near_city_center",
    "near_sea_zero", "near_lake_zero",
    "out_security", "out_camera", "out_pool", "out_open_pool", "out_closed_pool",
    "out_heat_insulation", "out_sound_insulation", "out_generator", "out_hydrofor",
    "out_children_playground", "out_sports_area", "out_sauna_hamam",
    "in_builtin_kitchen", "in_parent_bathroom", "in_glass_balcony", "in_terrace",
    "in_air_conditioner", "in_smart_home", "in_steel_door", "in_fiber",
    "in_intercom", "in_dressing_room", "in_pantry", "in_laminate_floor",
    "in_pvc", "in_heat_glass",
]

GROUP_SUM_FEATURES = [
    "detail_effect_front_sum",
    "detail_effect_view_sum",
    "detail_effect_transport_sum",
    "detail_effect_nearby_sum",
    "detail_effect_outside_sum",
    "detail_effect_inside_sum",
    "detail_effect_subtype_sum",
]
GROUP_MEAN_FEATURES = [
    "detail_effect_front_mean",
    "detail_effect_view_mean",
    "detail_effect_transport_mean",
    "detail_effect_nearby_mean",
    "detail_effect_outside_mean",
    "detail_effect_inside_mean",
    "detail_effect_subtype_mean",
]
AGG_FEATURES = [
    "detail_effect_total_sum",
    "detail_effect_total_mean",
    "detail_effect_total_abs_sum",
    "detail_effect_positive_count",
    "detail_effect_negative_count",
    "detail_effect_used_count",
    "detail_effect_county_total_sum",
    "detail_effect_district_total_sum",
    "detail_effect_level_code",
    "detail_effect_total_x_attr_quality",
    "detail_effect_view_x_location_baseline",
    "detail_effect_outside_x_attr_quality",
    "detail_effect_inside_x_attr_quality",
]

DETAIL_EFFECT_GROUP_NUMERIC_FEATURES = GROUP_SUM_FEATURES + GROUP_MEAN_FEATURES + AGG_FEATURES

# V13 default reference (retained for compatibility)
V13_DEFAULT_REF = {
    "r2": 0.6800,
    "mape": 0.1279,
    "basiskele_r2": 0.4449,
    "karamursel_r2": 0.5468,
    "k180_karamursel_r2": 0.5768,
}

# V14 group-mode reference (retained for compatibility)
V14_DEFAULT_REF = {
    "r2": 0.6787,
    "mape": 0.1290,
    "basiskele_r2": 0.4553,
    "basiskele_mape": 0.1103,
    "basiskele_variance_ratio": 0.4224,
    "golcuk_r2": 0.6444,
    "karamursel_r2": 0.5582,
    "izmit_r2": 0.7107,
    "k180_karamursel_r2": 0.5768,
}

# V15 full reference (V16 primary lift baselines)
V15_DEFAULT_REF = {
    "r2": 0.6799,
    "mape": 0.1290,
    "basiskele_r2": 0.4534,
    "basiskele_mape": 0.1110,
    "basiskele_variance_ratio": 0.4516,
    "basiskele_large_home_r2": 0.2396,
    "golcuk_r2": 0.6481,
    "karamursel_r2": 0.5681,
    "izmit_r2": 0.7109,
    "ship_ready_all_counties_r2_ge_0_65": False,
}


def feature_group(name: str) -> str:
    for group, prefix in DETAIL_GROUPS.items():
        if str(name).startswith(prefix):
            return group
    return "other"


def discover_detail_binary_columns(X: pd.DataFrame) -> dict[str, Any]:
    """Discover detail binary columns via prefix ∪ SCORE_GROUP_MEMBERS."""
    cols = list(X.columns) if X is not None else []
    prefix_hits = [c for c in cols if str(c).startswith(DETAIL_BINARY_PREFIXES)]
    known = [c for c in SCORE_GROUP_MEMBERS if c in cols]
    union = sorted(set(prefix_hits) | set(known))
    known_set = set(SCORE_GROUP_MEMBERS)
    mapped = [c for c in union if c in known_set]
    unmapped_extra = [c for c in union if c not in known_set]
    return {
        "all": union,
        "mapped": mapped,
        "unmapped_extra": unmapped_extra,
        "by_group": {g: [c for c in union if c.startswith(p)] for g, p in DETAIL_GROUPS.items()},
    }


def get_detail_effect_feature_names(mode: str = "group", individual_features: list[str] | None = None) -> list[str]:
    mode = str(mode or "none").lower()
    if mode == "none":
        return []
    names = list(DETAIL_EFFECT_GROUP_NUMERIC_FEATURES)
    if mode == "full":
        # Column set must be known at pipeline build time; unused individuals stay 0 after fit.
        feats = list(individual_features) if individual_features is not None else list(SCORE_GROUP_MEMBERS)
        for feat in feats:
            col = f"detail_effect__{feat}"
            if col not in names:
                names.append(col)
    return names


def _smooth(local: float, n: float, global_eff: float, alpha: float) -> float:
    return float((n / (n + alpha)) * local + (alpha / (n + alpha)) * global_eff)


def _binary_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).clip(0, 1).astype(int)


class LocalDetailPremiumEncoder(BaseEstimator, TransformerMixin):
    """Fold-safe smoothed residual premiums for detail binaries.

    In residual target_mode, y is already log(price)-log(baseline).
    random_state is kept for API compatibility (deterministic; unused in math).
    """

    def __init__(
        self,
        mode: str = "group",
        alpha: float = 50.0,
        county_min_count: int = 30,
        district_min_count: int = 50,
        use_district_effects: bool = True,
        use_county_effects: bool = True,
        random_state: int = 42,
        full_min_global_pos: int = 20,
        full_min_county_pos: int = 10,
    ):
        self.mode = mode
        self.alpha = alpha
        self.county_min_count = county_min_count
        self.district_min_count = district_min_count
        self.use_district_effects = use_district_effects
        self.use_county_effects = use_county_effects
        self.random_state = random_state  # no-op; deterministic encoder
        self.full_min_global_pos = full_min_global_pos
        self.full_min_county_pos = full_min_county_pos

    def fit(self, X: pd.DataFrame, y: Any):
        mode = str(self.mode or "none").lower()
        self.enabled_ = mode in {"group", "full"}
        self.mode_ = mode
        self.global_effects_: dict[str, float] = {}
        self.county_effects_: dict[str, dict[str, float]] = {}
        self.district_effects_: dict[str, dict[str, float]] = {}
        self.feature_stats_: dict[str, dict[str, Any]] = {}
        self.individual_features_: list[str] = []
        self.discovered_features_: list[str] = []
        self.feature_names_out_: list[str] = get_detail_effect_feature_names(mode)

        if not self.enabled_:
            return self

        df = X.copy()
        y_series = pd.Series(np.asarray(y, dtype=float), index=df.index, name="__premium__")
        valid = y_series.notna() & np.isfinite(y_series)
        work = df.loc[valid].copy()
        work["__premium__"] = y_series.loc[valid]
        if "county" not in work.columns:
            work["county"] = "missing"
        work["county"] = work["county"].astype(str).fillna("missing")
        if "district" not in work.columns:
            work["district"] = "missing"
        work["district"] = work["district"].astype(str).fillna("missing")
        work["__district_key__"] = work["county"] + "||" + work["district"]

        discovered = discover_detail_binary_columns(work)
        features = discovered["all"]
        self.discovered_features_ = list(features)
        alpha = float(self.alpha)

        for feat in features:
            col = _binary_series(work[feat])
            pos = work.loc[col == 1, "__premium__"]
            neg = work.loc[col == 0, "__premium__"]
            if len(pos) < 3 or len(neg) < 3:
                g_eff = 0.0
            else:
                g_eff = float(np.nanmedian(pos) - np.nanmedian(neg))
            self.global_effects_[feat] = g_eff

            county_map: dict[str, float] = {}
            county_pos_counts: dict[str, int] = {}
            if self.use_county_effects:
                for county, gdf in work.groupby("county", dropna=False):
                    cbin = _binary_series(gdf[feat])
                    p = gdf.loc[cbin == 1, "__premium__"]
                    n = gdf.loc[cbin == 0, "__premium__"]
                    n_pos = int(len(p))
                    county_pos_counts[str(county)] = n_pos
                    if n_pos < 3 or len(n) < 3:
                        county_map[str(county)] = g_eff
                        continue
                    local = float(np.nanmedian(p) - np.nanmedian(n))
                    # n for smoothing: positive count (effect reliability)
                    county_map[str(county)] = _smooth(local, float(n_pos), g_eff, alpha)
            self.county_effects_[feat] = county_map

            district_map: dict[str, float] = {}
            if self.use_district_effects:
                for dkey, gdf in work.groupby("__district_key__", dropna=False):
                    cbin = _binary_series(gdf[feat])
                    p = gdf.loc[cbin == 1, "__premium__"]
                    n = gdf.loc[cbin == 0, "__premium__"]
                    n_pos = int(len(p))
                    county = str(gdf["county"].iloc[0]) if len(gdf) else "missing"
                    parent = county_map.get(county, g_eff)
                    if n_pos < 3 or len(n) < 3:
                        district_map[str(dkey)] = parent
                        continue
                    local = float(np.nanmedian(p) - np.nanmedian(n))
                    district_map[str(dkey)] = _smooth(local, float(n_pos), parent, alpha)
            self.district_effects_[feat] = district_map

            self.feature_stats_[feat] = {
                "group": feature_group(feat),
                "global_effect": g_eff,
                "global_positive_count": int((col == 1).sum()),
                "global_negative_count": int((col == 0).sum()),
                "county_positive_counts": county_pos_counts,
                "max_county_positive": int(max(county_pos_counts.values()) if county_pos_counts else 0),
            }

        if mode == "full":
            kept = []
            for feat, st in self.feature_stats_.items():
                if st["global_positive_count"] >= int(self.full_min_global_pos) or st["max_county_positive"] >= int(
                    self.full_min_county_pos
                ):
                    kept.append(feat)
            self.individual_features_ = sorted(kept)
            # Keep stable full column schema (SCORE_GROUP_MEMBERS + discovered)
            schema_feats = sorted(set(SCORE_GROUP_MEMBERS) | set(self.discovered_features_))
            self.feature_names_out_ = get_detail_effect_feature_names("full", schema_feats)
        else:
            self.individual_features_ = []
            self.feature_names_out_ = get_detail_effect_feature_names(mode)
        return self

    def _resolve_effect(self, feat: str, county: str, district_key: str) -> tuple[float, int]:
        """Return (effect, level_code) with district→county→global fallback."""
        g = float(self.global_effects_.get(feat, 0.0))
        st = self.feature_stats_.get(feat, {})
        county_pos = int((st.get("county_positive_counts") or {}).get(county, 0))

        if self.use_district_effects and feat in self.district_effects_:
            dmap = self.district_effects_[feat]
            if district_key in dmap:
                # reliability: need enough district positives approximated via county gate + key present
                if county_pos >= int(self.district_min_count) or (
                    county_pos >= int(self.county_min_count) and district_key in dmap
                ):
                    # Prefer district when county has enough mass overall
                    if county_pos >= int(self.district_min_count):
                        return float(dmap[district_key]), 2

        if self.use_county_effects and feat in self.county_effects_:
            cmap = self.county_effects_[feat]
            if county in cmap and county_pos >= int(self.county_min_count):
                return float(cmap[county]), 1
            if county in cmap:
                # still use smoothed county even if below min (already smoothed in fit)
                return float(cmap[county]), 1

        return g, 0

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        names = list(getattr(self, "feature_names_out_", DETAIL_EFFECT_GROUP_NUMERIC_FEATURES))
        if not getattr(self, "enabled_", False):
            for col in names:
                if col not in df.columns:
                    df[col] = 0.0
            return df

        if "county" not in df.columns:
            df["county"] = "missing"
        if "district" not in df.columns:
            df["district"] = "missing"
        county = df["county"].astype(str).fillna("missing")
        district = df["district"].astype(str).fillna("missing")
        dkey = county + "||" + district

        features = list(getattr(self, "discovered_features_", []) or [])
        n = len(df)
        group_sums = {g: np.zeros(n, dtype=float) for g in DETAIL_GROUPS}
        group_counts = {g: np.zeros(n, dtype=float) for g in DETAIL_GROUPS}
        total_sum = np.zeros(n, dtype=float)
        total_abs = np.zeros(n, dtype=float)
        pos_count = np.zeros(n, dtype=float)
        neg_count = np.zeros(n, dtype=float)
        used_count = np.zeros(n, dtype=float)
        county_total = np.zeros(n, dtype=float)
        district_total = np.zeros(n, dtype=float)
        level_sum = np.zeros(n, dtype=float)
        level_n = np.zeros(n, dtype=float)

        individual = set(getattr(self, "individual_features_", []) or [])
        for feat in features:
            if feat not in df.columns:
                vals = np.zeros(n, dtype=int)
            else:
                vals = _binary_series(df[feat]).to_numpy()
            effects = np.zeros(n, dtype=float)
            levels = np.zeros(n, dtype=float)
            for i in range(n):
                eff, lvl = self._resolve_effect(feat, str(county.iloc[i]), str(dkey.iloc[i]))
                effects[i] = eff
                levels[i] = lvl
            active = vals == 1
            contrib = np.where(active, effects, 0.0)
            grp = feature_group(feat)
            if grp in group_sums:
                group_sums[grp] += contrib
                group_counts[grp] += active.astype(float)
            total_sum += contrib
            total_abs += np.abs(contrib)
            pos_count += ((active) & (effects > 0)).astype(float)
            neg_count += ((active) & (effects < 0)).astype(float)
            used_count += active.astype(float)
            # county vs district contribution tracking
            county_total += np.where(active & (levels >= 1), effects, 0.0)
            district_total += np.where(active & (levels >= 2), effects, 0.0)
            level_sum += np.where(active, levels, 0.0)
            level_n += active.astype(float)

            if feat in individual:
                df[f"detail_effect__{feat}"] = contrib

        for g in DETAIL_GROUPS:
            df[f"detail_effect_{g}_sum"] = group_sums[g]
            with np.errstate(divide="ignore", invalid="ignore"):
                mean = np.where(group_counts[g] > 0, group_sums[g] / group_counts[g], 0.0)
            df[f"detail_effect_{g}_mean"] = mean

        df["detail_effect_total_sum"] = total_sum
        with np.errstate(divide="ignore", invalid="ignore"):
            df["detail_effect_total_mean"] = np.where(used_count > 0, total_sum / used_count, 0.0)
        df["detail_effect_total_abs_sum"] = total_abs
        df["detail_effect_positive_count"] = pos_count
        df["detail_effect_negative_count"] = neg_count
        df["detail_effect_used_count"] = used_count
        df["detail_effect_county_total_sum"] = county_total
        df["detail_effect_district_total_sum"] = district_total
        with np.errstate(divide="ignore", invalid="ignore"):
            df["detail_effect_level_code"] = np.where(level_n > 0, level_sum / level_n, 0.0)

        # Interactions
        attr_q = pd.to_numeric(df.get("attr_total_quality_score", 0), errors="coerce").fillna(0).to_numpy(dtype=float)
        df["detail_effect_total_x_attr_quality"] = total_sum * attr_q
        df["detail_effect_outside_x_attr_quality"] = group_sums["outside"] * attr_q
        df["detail_effect_inside_x_attr_quality"] = group_sums["inside"] * attr_q

        view_sum = group_sums["view"]
        if "location_baseline_m2" in df.columns:
            base = pd.to_numeric(df["location_baseline_m2"], errors="coerce").fillna(0).clip(lower=0)
            df["detail_effect_view_x_location_baseline"] = view_sum * np.log1p(base.to_numpy(dtype=float))
        elif "county_target_median" in df.columns:
            base = pd.to_numeric(df["county_target_median"], errors="coerce").fillna(0)
            df["detail_effect_view_x_location_baseline"] = view_sum * base.to_numpy(dtype=float)
        else:
            df["detail_effect_view_x_location_baseline"] = 0.0

        for col in names:
            if col not in df.columns:
                df[col] = 0.0
        return df


def detail_feature_coverage(X: pd.DataFrame, reports_dir=None, unreliable_min_pos: int = 10) -> pd.DataFrame:
    """Per-county coverage for detail binaries; low Başiskele positives → unreliable."""
    rows = []
    if X is None or X.empty:
        return pd.DataFrame()
    discovered = discover_detail_binary_columns(X)
    work = X.copy()
    if "county" not in work.columns:
        work["county"] = "missing"
    for county, gdf in work.groupby(work["county"].astype(str), dropna=False):
        n = len(gdf)
        for feat in discovered["all"]:
            if feat not in gdf.columns:
                non_null = 0
                pos = 0
                missing_rate = 1.0
            else:
                raw = pd.to_numeric(gdf[feat], errors="coerce")
                non_null = int(raw.notna().sum())
                pos = int((_binary_series(gdf[feat]) == 1).sum())
                missing_rate = float(1.0 - non_null / max(n, 1))
            status = "ok"
            if feat in discovered["unmapped_extra"]:
                status = "unmapped_extra"
            if str(county) == "Başiskele" and pos < int(unreliable_min_pos):
                status = "unreliable" if status == "ok" else status + "+unreliable"
            rows.append(
                {
                    "county": str(county),
                    "group": feature_group(feat),
                    "feature": feat,
                    "non_null_count": non_null,
                    "positive_count": pos,
                    "positive_rate": float(pos / max(n, 1)),
                    "missing_rate": missing_rate,
                    "reliability_status": status,
                }
            )
    rep = pd.DataFrame(rows)
    if reports_dir is not None and not rep.empty:
        rep.to_csv(reports_dir / "detail_feature_coverage_v16.csv", index=False, encoding="utf-8-sig")
    return rep


def export_detail_premium_effect_tables(encoder: LocalDetailPremiumEncoder, reports_dir) -> dict[str, pd.DataFrame]:
    """Export effect tables from a *final fitted* LocalDetailPremiumEncoder state.

    These are in-sample final-encoder effects (not OOF fold encoders).
    """
    reports_dir = reports_dir
    stats = getattr(encoder, "feature_stats_", {}) or {}
    county_effects = getattr(encoder, "county_effects_", {}) or {}
    global_effects = getattr(encoder, "global_effects_", {}) or {}

    by_county_rows = []
    for feat, cmap in county_effects.items():
        st = stats.get(feat, {})
        g_eff = float(global_effects.get(feat, 0.0))
        for county, eff in cmap.items():
            pos = int((st.get("county_positive_counts") or {}).get(county, 0))
            level = "county" if pos >= int(getattr(encoder, "county_min_count", 30)) else "global_fallback"
            by_county_rows.append(
                {
                    "county": county,
                    "feature": feat,
                    "group": st.get("group", feature_group(feat)),
                    "positive_count": pos,
                    "negative_count": np.nan,
                    "local_effect": float(eff),
                    "global_effect": g_eff,
                    "smoothed_effect": float(eff),
                    "reliability_level": level,
                }
            )
    by_county = pd.DataFrame(by_county_rows)
    if not by_county.empty:
        by_county.to_csv(reports_dir / "detail_premium_effects_by_county_v16.csv", index=False, encoding="utf-8-sig")

    # Başiskele focus
    bas_rows = []
    for feat, st in stats.items():
        g_eff = float(global_effects.get(feat, 0.0))
        pos = int((st.get("county_positive_counts") or {}).get("Başiskele", 0))
        neg = np.nan
        eff = float(county_effects.get(feat, {}).get("Başiskele", g_eff))
        unreliable = pos < 10
        bas_rows.append(
            {
                "feature": feat,
                "group": st.get("group", feature_group(feat)),
                "basiskele_positive_count": pos,
                "basiskele_negative_count": neg,
                "basiskele_effect": eff,
                "global_effect": g_eff,
                "smoothed_effect": eff,
                "abs_effect": abs(eff),
                "direction": "premium" if eff > 0 else ("discount" if eff < 0 else "neutral"),
                "reliability_level": "unreliable" if unreliable else "county",
            }
        )
    bas = pd.DataFrame(bas_rows).sort_values("abs_effect", ascending=False) if bas_rows else pd.DataFrame()
    if not bas.empty:
        bas.to_csv(reports_dir / "basiskele_detail_premium_diagnostics_v16.csv", index=False, encoding="utf-8-sig")

    # Group summary
    group_rows = []
    if not by_county.empty:
        for (county, group), gdf in by_county.groupby(["county", "group"]):
            top_pos = (
                gdf.sort_values("smoothed_effect", ascending=False).head(3)["feature"].tolist()
            )
            top_neg = (
                gdf.sort_values("smoothed_effect", ascending=True).head(3)["feature"].tolist()
            )
            group_rows.append(
                {
                    "county": county,
                    "group": group,
                    "mean_abs_effect": float(gdf["smoothed_effect"].abs().mean()),
                    "positive_feature_count": int((gdf["smoothed_effect"] > 0).sum()),
                    "negative_feature_count": int((gdf["smoothed_effect"] < 0).sum()),
                    "top_positive_features": "|".join(top_pos),
                    "top_negative_features": "|".join(top_neg),
                }
            )
    group_sum = pd.DataFrame(group_rows)
    if not group_sum.empty:
        group_sum.to_csv(reports_dir / "detail_premium_group_summary_v16.csv", index=False, encoding="utf-8-sig")

    return {"by_county": by_county, "basiskele": bas, "group_summary": group_sum}
