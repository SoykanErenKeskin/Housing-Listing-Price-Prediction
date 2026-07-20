"""V16 regime / residual layers (app-safe).

Deterministic:
- BasiskeleLargeHomeRegimeAdder
- KaramurselLocationAgeBaselineAdder (fold-safe, uses residual y)

OOF post-layers (fit on train fold only; never fit on validation actual):
- apply_basiskele_large_home_residual_layer
- apply_basiskele_spread_residual_layer
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASISKELE = "Başiskele"
KARAMURSEL = "Karamürsel"

BSK_LARGE_HOME_FEATURES = [
    "bsk_large_home_flag",
    "bsk_very_large_200p_flag",
    "bsk_room_4p1_flag",
    "bsk_large_home_m2_excess",
    "bsk_large_home_log_m2",
    "bsk_large_home_net_gross_quality",
    "bsk_large_home_quality_x_m2",
    "bsk_large_home_detail_x_m2",
    "bsk_large_home_view_x_m2",
    "bsk_large_home_outside_x_m2",
    "bsk_large_home_site_x_m2",
    "bsk_large_home_age_penalty",
    "bsk_large_home_new_bonus",
    "bsk_large_home_district_baseline_x_m2",
]

KARAMURSEL_BASELINE_FEATURES = [
    "karamursel_district_age_residual_median",
    "karamursel_district_m2_residual_median",
    "karamursel_district_room_residual_median",
    "karamursel_district_m2_age_residual_median",
    "karamursel_county_age_residual_median",
    "karamursel_county_m2_residual_median",
    "karamursel_baseline_confidence",
    "karamursel_baseline_level_code",
]


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _site_flag(df: pd.DataFrame) -> pd.Series:
    if "attr_is_site_inside" in df.columns:
        s = _num(df, "attr_is_site_inside", 0.0).clip(0, 1)
        if float(s.abs().sum()) > 0:
            return s
    if "site_inside" not in df.columns:
        return pd.Series(0.0, index=df.index)
    return df["site_inside"].astype(str).str.lower().isin(["evet", "var", "1", "true"]).astype(float)


def _rooms(df: pd.DataFrame) -> pd.Series:
    if "rooms" in df.columns:
        return pd.to_numeric(df["rooms"], errors="coerce").fillna(3.0)
    if "room_count" not in df.columns:
        return pd.Series(3.0, index=df.index)
    s = df["room_count"].astype(str).str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(s, errors="coerce").fillna(3.0)


def _is_large_mask(df: pd.DataFrame) -> pd.Series:
    gross = _num(df, "gross_m2", 0.0)
    rooms = _rooms(df)
    m2g = df["m2_group"].astype(str) if "m2_group" in df.columns else pd.Series("", index=df.index)
    flag = (gross >= 150) | (rooms >= 4) | m2g.isin(["151-200", "151–200", "200+"])
    if "is_large_flat" in df.columns:
        flag = flag | (_num(df, "is_large_flat", 0.0) > 0)
    return flag.fillna(False)


def _metric_r2(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 5:
        return float("nan")
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _metric_mape(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p) & (y > 0)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y[m] - p[m]) / y[m]))


def _var_ratio(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 5:
        return float("nan")
    av = float(np.var(y, ddof=1))
    pv = float(np.var(p, ddof=1))
    if av <= 1e-12:
        return float("nan")
    return pv / av


def _decile_bias(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Return (cheap_decile_mean_bias, expensive_decile_mean_bias)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 20:
        return float("nan"), float("nan")
    order = np.argsort(y)
    n = len(y)
    k = max(1, n // 10)
    cheap = order[:k]
    rich = order[-k:]
    return float(np.mean(p[cheap] - y[cheap])), float(np.mean(p[rich] - y[rich]))


class BasiskeleLargeHomeRegimeAdder(BaseEstimator, TransformerMixin):
    """Deterministic Başiskele large-home regime features (no target)."""

    def __init__(self, mode: str = "simple"):
        self.mode = mode

    def fit(self, X: pd.DataFrame, y: Any = None):
        self.enabled_ = str(self.mode or "none").lower() not in {"", "none"}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for c in BSK_LARGE_HOME_FEATURES:
            if c not in df.columns:
                df[c] = 0.0
        if not getattr(self, "enabled_", False):
            return df

        bas = df["county"].astype(str).eq(BASISKELE) if "county" in df.columns else pd.Series(False, index=df.index)
        large = _is_large_mask(df)
        active = (bas & large).astype(float)
        gross = _num(df, "gross_m2", 0.0)
        net = _num(df, "net_m2", 0.0)
        age = _num(df, "building_age", 0.0)
        rooms = _rooms(df)
        m2g = df["m2_group"].astype(str) if "m2_group" in df.columns else pd.Series("", index=df.index)
        attr_q = _num(df, "attr_total_quality_score", 0.0)
        detail = _num(df, "detail_effect_total_sum", 0.0)
        view = _num(df, "detail_effect_view_sum", 0.0)
        outside = _num(df, "detail_effect_outside_sum", 0.0)
        site = _site_flag(df)
        net_ratio = (net / np.maximum(gross, 1.0)).clip(0, 1.5).fillna(0.8)
        is_new = _num(df, "attr_is_new_building", 0.0)
        if float(is_new.abs().sum()) == 0:
            is_new = (age <= 3).astype(float)
        base = _num(df, "district_target_median", 0.0)
        if float(base.abs().sum()) == 0:
            base = _num(df, "location_baseline_m2", 0.0)

        df["bsk_large_home_flag"] = active
        df["bsk_very_large_200p_flag"] = (active * ((gross >= 200) | m2g.eq("200+")).astype(float)).astype(float)
        df["bsk_room_4p1_flag"] = (
            active
            * (
                (rooms >= 4)
                | df.get("room_count", pd.Series("", index=df.index)).astype(str).str.startswith("4+")
            ).astype(float)
        )
        df["bsk_large_home_m2_excess"] = np.maximum(gross - 150.0, 0.0) * active
        df["bsk_large_home_log_m2"] = np.log1p(np.maximum(gross, 0.0)) * active
        df["bsk_large_home_net_gross_quality"] = net_ratio * active
        df["bsk_large_home_quality_x_m2"] = attr_q * gross * active
        df["bsk_large_home_detail_x_m2"] = detail * gross * active
        df["bsk_large_home_view_x_m2"] = view * gross * active
        df["bsk_large_home_outside_x_m2"] = outside * gross * active
        df["bsk_large_home_site_x_m2"] = site * gross * active
        df["bsk_large_home_age_penalty"] = age * active
        df["bsk_large_home_new_bonus"] = is_new * active
        df["bsk_large_home_district_baseline_x_m2"] = base * gross * active
        return df


class KaramurselLocationAgeBaselineAdder(BaseEstimator, TransformerMixin):
    """Fold-safe residual stats for Karamürsel location×age keys (uses residual y)."""

    def __init__(self, mode: str = "location_age", alpha: float = 80.0, min_count: int = 20):
        self.mode = mode
        self.alpha = alpha
        self.min_count = min_count

    def fit(self, X: pd.DataFrame, y: Any):
        mode = str(self.mode or "none").lower()
        self.enabled_ = mode in {"location_age"}
        self.global_median_ = 0.0
        self.maps_: dict[str, dict[str, float]] = {}
        self.counts_: dict[str, dict[str, float]] = {}
        self.effect_rows_: list[dict[str, Any]] = []
        if not self.enabled_:
            return self

        df = X.copy()
        y_series = pd.Series(np.asarray(y, dtype=float), index=df.index)
        valid = y_series.notna() & np.isfinite(y_series)
        work = df.loc[valid].copy()
        work["__y__"] = y_series.loc[valid]
        mask = work["county"].astype(str).eq(KARAMURSEL) if "county" in work.columns else pd.Series(False, index=work.index)
        work = work.loc[mask].copy()
        if work.empty:
            self.global_median_ = float(np.nanmedian(y_series)) if len(y_series) else 0.0
            return self
        self.global_median_ = float(np.nanmedian(work["__y__"]))
        alpha = float(self.alpha)

        for col, default in [
            ("district", "missing"),
            ("building_age_group", "missing"),
            ("m2_group", "missing"),
            ("room_count", "missing"),
        ]:
            if col not in work.columns:
                work[col] = default
            work[col] = work[col].astype(str).fillna(default)

        def fit_key(cols: list[str], name: str):
            g = work.groupby(cols, dropna=False)["__y__"].agg(["median", "count"]).reset_index()
            mapping = {}
            counts = {}
            for _, row in g.iterrows():
                key = "||".join(str(row[c]) for c in cols)
                n = float(row["count"])
                local = float(row["median"])
                smoothed = (n / (n + alpha)) * local + (alpha / (n + alpha)) * self.global_median_
                mapping[key] = smoothed
                counts[key] = n
                self.effect_rows_.append(
                    {
                        "key_type": name,
                        "key_value": key,
                        "rows": n,
                        "local_effect": local,
                        "smoothed_effect": smoothed,
                        "level": "district" if "district" in cols else "county",
                        "reliability": "ok" if n >= float(self.min_count) else "low",
                    }
                )
            self.maps_[name] = mapping
            self.counts_[name] = counts

        fit_key(["district", "building_age_group"], "karamursel_district_age_residual_median")
        fit_key(["district", "m2_group"], "karamursel_district_m2_residual_median")
        fit_key(["district", "room_count"], "karamursel_district_room_residual_median")
        fit_key(["district", "m2_group", "building_age_group"], "karamursel_district_m2_age_residual_median")
        fit_key(["building_age_group"], "karamursel_county_age_residual_median")
        fit_key(["m2_group"], "karamursel_county_m2_residual_median")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for c in KARAMURSEL_BASELINE_FEATURES:
            if c not in df.columns:
                df[c] = 0.0
        if not getattr(self, "enabled_", False):
            return df

        mask = df["county"].astype(str).eq(KARAMURSEL) if "county" in df.columns else pd.Series(False, index=df.index)
        for col, default in [
            ("district", "missing"),
            ("building_age_group", "missing"),
            ("m2_group", "missing"),
            ("room_count", "missing"),
        ]:
            if col not in df.columns:
                df[col] = default
        district = df["district"].astype(str).fillna("missing")
        age = df["building_age_group"].astype(str).fillna("missing")
        m2g = df["m2_group"].astype(str).fillna("missing")
        room = df["room_count"].astype(str).fillna("missing")

        def lookup(name: str, keys: pd.Series, fallback: pd.Series | None = None) -> pd.Series:
            m = self.maps_.get(name, {})
            vals = keys.map(m)
            if fallback is not None:
                vals = vals.fillna(fallback)
            return vals.fillna(self.global_median_)

        d_age = district + "||" + age
        d_m2 = district + "||" + m2g
        d_room = district + "||" + room
        d_m2_age = district + "||" + m2g + "||" + age
        c_age = lookup("karamursel_county_age_residual_median", age)
        c_m2 = lookup("karamursel_county_m2_residual_median", m2g)

        df["karamursel_district_age_residual_median"] = np.where(mask, lookup("karamursel_district_age_residual_median", d_age, c_age), 0.0)
        df["karamursel_district_m2_residual_median"] = np.where(mask, lookup("karamursel_district_m2_residual_median", d_m2, c_m2), 0.0)
        df["karamursel_district_room_residual_median"] = np.where(
            mask, lookup("karamursel_district_room_residual_median", d_room, c_age), 0.0
        )
        df["karamursel_district_m2_age_residual_median"] = np.where(
            mask, lookup("karamursel_district_m2_age_residual_median", d_m2_age, c_age), 0.0
        )
        df["karamursel_county_age_residual_median"] = np.where(mask, c_age, 0.0)
        df["karamursel_county_m2_residual_median"] = np.where(mask, c_m2, 0.0)

        # confidence from most specific key count
        cnt = d_m2_age.map(self.counts_.get("karamursel_district_m2_age_residual_median", {})).fillna(0.0)
        cnt = cnt.where(cnt > 0, d_age.map(self.counts_.get("karamursel_district_age_residual_median", {})).fillna(0.0))
        conf = np.minimum(cnt, float(self.min_count)) / float(self.min_count)
        level = np.select(
            [
                d_m2_age.map(self.counts_.get("karamursel_district_m2_age_residual_median", {})).fillna(0) > 0,
                d_age.map(self.counts_.get("karamursel_district_age_residual_median", {})).fillna(0) > 0,
                age.map(self.counts_.get("karamursel_county_age_residual_median", {})).fillna(0) > 0,
            ],
            [3, 2, 1],
            default=0,
        )
        df["karamursel_baseline_confidence"] = np.where(mask, conf, 0.0)
        df["karamursel_baseline_level_code"] = np.where(mask, level, 0.0)
        return df

    def export_effect_table(self) -> pd.DataFrame:
        return pd.DataFrame(getattr(self, "effect_rows_", []) or [])


def get_v16_regime_feature_names(
    large_home_regime: str = "simple",
    karamursel_baseline_mode: str = "location_age",
) -> list[str]:
    names: list[str] = []
    if str(large_home_regime or "none").lower() not in {"", "none"}:
        names.extend(BSK_LARGE_HOME_FEATURES)
    if str(karamursel_baseline_mode or "none").lower() in {"location_age"}:
        names.extend(KARAMURSEL_BASELINE_FEATURES)
    # unique
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _build_delta_matrix(df: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    mats = []
    names = []
    for c in feature_cols:
        if c in df.columns:
            mats.append(_num(df, c, 0.0).to_numpy(dtype=float).reshape(-1, 1))
            names.append(c)
    # simple categorical one-hot (fit outside handled by caller with shared columns)
    for c in cat_cols:
        if c not in df.columns:
            continue
        # caller should already expand cats; skip here
    if not mats:
        return np.zeros((len(df), 1), dtype=float), ["bias"]
    return np.hstack(mats), names


def _onehot_cats(train: pd.DataFrame, val: pd.DataFrame, cat_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not cat_cols:
        return np.zeros((len(train), 0)), np.zeros((len(val), 0))
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    tr = train[cat_cols].astype(str).fillna("missing")
    va = val[cat_cols].astype(str).fillna("missing")
    Xtr = enc.fit_transform(tr)
    Xva = enc.transform(va)
    return np.asarray(Xtr, dtype=float), np.asarray(Xva, dtype=float)


def _delta_model_candidates(fast: bool = False) -> dict[str, Any]:
    return {
        "ridge_5": Ridge(alpha=5.0, solver="lsqr"),
        "ridge_10": Ridge(alpha=10.0, solver="lsqr"),
        "ridge_20": Ridge(alpha=20.0, solver="lsqr"),
        "ridge_50": Ridge(alpha=50.0, solver="lsqr"),
        "hgb": HistGradientBoostingRegressor(
            max_iter=80 if fast else 120,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42,
        ),
        "gb": GradientBoostingRegressor(
            n_estimators=60 if fast else 100,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=20,
            subsample=0.85,
            random_state=42,
        ),
    }


def apply_basiskele_large_home_residual_layer(
    X: pd.DataFrame,
    y: pd.Series,
    current_pred: pd.Series,
    *,
    mode: str = "residual",
    n_splits: int = 5,
    random_state: int = 42,
    fast_mode: bool = False,
) -> tuple[pd.Series, dict[str, Any]]:
    """OOF-safe Başiskele large_home residual blend. Disabled unless mode=residual."""
    report: dict[str, Any] = {
        "status": "disabled",
        "selected_model": "",
        "selected_lambda": 0.0,
        "note": "",
    }
    final = pd.Series(current_pred, index=X.index, dtype=float).copy()
    mode = str(mode or "none").lower()
    if mode != "residual":
        report["note"] = f"mode={mode} (residual layer only runs for residual)"
        return final, report

    bas = X["county"].astype(str).eq(BASISKELE) if "county" in X.columns else pd.Series(False, index=X.index)
    large = _is_large_mask(X)
    mask = bas & large
    idx = np.where(mask.to_numpy())[0]
    if len(idx) < 40:
        report["status"] = "skipped_too_few_rows"
        report["note"] = f"rows={len(idx)}"
        return final, report

    y_all = pd.Series(y, index=X.index).astype(float)
    p_all = pd.Series(current_pred, index=X.index).astype(float)
    before_lh_r2 = _metric_r2(y_all.iloc[idx].to_numpy(), p_all.iloc[idx].to_numpy())
    before_b_r2 = _metric_r2(y_all.loc[bas].to_numpy(), p_all.loc[bas].to_numpy())
    before_b_mape = _metric_mape(y_all.loc[bas].to_numpy(), p_all.loc[bas].to_numpy())
    before_g_mape = _metric_mape(y_all.to_numpy(), p_all.to_numpy())
    non_lh = bas & ~large
    before_non_lh = _metric_r2(y_all.loc[non_lh].to_numpy(), p_all.loc[non_lh].to_numpy())

    Xw = X.copy()
    Xw["rooms_numeric"] = _rooms(Xw)
    Xw["pred_current"] = p_all

    num_cols = [
        "gross_m2",
        "net_m2",
        "building_age",
        "rooms_numeric",
        "attr_total_quality_score",
        "detail_effect_total_sum",
        "detail_effect_view_sum",
        "detail_effect_outside_sum",
        "location_baseline_m2",
        "district_target_median",
        "pred_current",
        *list(BSK_LARGE_HOME_FEATURES),
    ]
    # unique preserve order
    seen: set[str] = set()
    num_cols = [c for c in num_cols if not (c in seen or seen.add(c))]
    cat_cols = [c for c in ["district", "m2_group"] if c in Xw.columns]

    n_splits = min(n_splits, max(2, len(idx) // 25))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    candidates = _delta_model_candidates(fast=fast_mode)
    oof_delta = {name: np.zeros(len(idx), dtype=float) for name in candidates}

    Xb = Xw.iloc[idx].reset_index(drop=True)
    yb = y_all.iloc[idx].reset_index(drop=True).to_numpy()
    pb = p_all.iloc[idx].reset_index(drop=True).to_numpy()

    for tr, va in cv.split(Xb):
        Xtr, Xva = Xb.iloc[tr], Xb.iloc[va]
        dy = yb[tr] - pb[tr]
        Xtr_num, _ = _build_delta_matrix(Xtr, num_cols, [])
        Xva_num, _ = _build_delta_matrix(Xva, num_cols, [])
        Xtr_cat, Xva_cat = _onehot_cats(Xtr, Xva, cat_cols)
        Xtr_m = np.hstack([Xtr_num, Xtr_cat]) if Xtr_cat.size else Xtr_num
        Xva_m = np.hstack([Xva_num, Xva_cat]) if Xva_cat.size else Xva_num
        for name, model in candidates.items():
            est = clone(model)
            with np.errstate(all="ignore"):
                est.fit(Xtr_m, dy)
                oof_delta[name][va] = np.asarray(est.predict(Xva_m), dtype=float)

    lambdas = [0.10, 0.15, 0.20, 0.25, 0.35]
    best = None
    for name, delta in oof_delta.items():
        for lam in lambdas:
            cand = pb + float(lam) * delta
            cand = np.maximum(cand, 0.0)
            gpred = p_all.to_numpy(dtype=float).copy()
            gpred[idx] = cand
            row = {
                "model": name,
                "lambda": float(lam),
                "large_home_r2": _metric_r2(yb, cand),
                "basiskele_r2": _metric_r2(y_all.loc[bas].to_numpy(), gpred[bas.to_numpy()]),
                "basiskele_mape": _metric_mape(y_all.loc[bas].to_numpy(), gpred[bas.to_numpy()]),
                "global_mape": _metric_mape(y_all.to_numpy(), gpred),
                "non_large_r2": _metric_r2(y_all.loc[non_lh].to_numpy(), gpred[non_lh.to_numpy()]),
                "pred": cand,
                "gpred": gpred,
            }
            if best is None or (
                row["large_home_r2"],
                row["basiskele_r2"],
                -row["basiskele_mape"],
            ) > (
                best["large_home_r2"],
                best["basiskele_r2"],
                -best["basiskele_mape"],
            ):
                best = row

    assert best is not None
    ok = (
        best["large_home_r2"] > before_lh_r2 + 1e-4
        and best["basiskele_r2"] >= before_b_r2 - 1e-4
        and best["basiskele_mape"] <= 0.115 + 1e-6
        and best["global_mape"] <= 0.131 + 1e-6
        and (not np.isfinite(before_non_lh) or best["non_large_r2"] >= before_non_lh - 1e-4)
    )
    report.update(
        {
            "large_home_r2_before": before_lh_r2,
            "large_home_r2_after": best["large_home_r2"],
            "basiskele_r2_before": before_b_r2,
            "basiskele_r2_after": best["basiskele_r2"],
            "basiskele_mape_before": before_b_mape,
            "basiskele_mape_after": best["basiskele_mape"],
            "global_mape_before": before_g_mape,
            "global_mape_after": best["global_mape"],
            "non_large_basiskele_r2_before": before_non_lh,
            "non_large_basiskele_r2_after": best["non_large_r2"],
            "selected_model": best["model"],
            "selected_lambda": best["lambda"],
        }
    )
    if ok:
        final.iloc[idx] = best["pred"]
        report["status"] = "applied"
        report["note"] = "OOF residual large_home layer applied"
    else:
        report["status"] = "rejected_guardrail"
        report["note"] = "disabled: large_home/global/non-large guardrails failed"
    return final, report


def apply_basiskele_spread_residual_layer(
    X: pd.DataFrame,
    y: pd.Series,
    current_pred: pd.Series,
    *,
    mode: str = "conservative",
    n_splits: int = 5,
    random_state: int = 42,
    fast_mode: bool = False,
) -> tuple[pd.Series, dict[str, Any]]:
    """OOF-safe Başiskele spread residual layer."""
    report: dict[str, Any] = {
        "status": "disabled",
        "selected_lambda": 0.0,
        "selected_model": "",
        "note": "",
    }
    final = pd.Series(current_pred, index=X.index, dtype=float).copy()
    mode = str(mode or "none").lower()
    if mode in {"", "none"}:
        report["note"] = "spread layer off"
        return final, report

    bas = X["county"].astype(str).eq(BASISKELE) if "county" in X.columns else pd.Series(False, index=X.index)
    idx = np.where(bas.to_numpy())[0]
    if len(idx) < 80:
        report["status"] = "skipped_too_few_rows"
        report["note"] = f"rows={len(idx)}"
        return final, report

    y_all = pd.Series(y, index=X.index).astype(float)
    p_all = pd.Series(current_pred, index=X.index).astype(float)
    yb = y_all.iloc[idx].to_numpy()
    pb = p_all.iloc[idx].to_numpy()
    before_r2 = _metric_r2(yb, pb)
    before_var = _var_ratio(yb, pb)
    before_mape = _metric_mape(yb, pb)
    before_g = _metric_mape(y_all.to_numpy(), p_all.to_numpy())
    cheap_b, rich_b = _decile_bias(yb, pb)

    Xw = X.copy()
    Xw["rooms_numeric"] = _rooms(Xw)
    Xw["pred_current"] = p_all
    Xw["is_large_flat"] = _is_large_mask(Xw).astype(float)
    num_cols = [
        c
        for c in [
            "pred_current",
            "gross_m2",
            "net_m2",
            "building_age",
            "rooms_numeric",
            "is_large_flat",
            "attr_total_quality_score",
            "detail_effect_total_sum",
            "detail_effect_view_sum",
            "detail_effect_outside_sum",
            "location_baseline_m2",
            "district_target_median",
            *BSK_LARGE_HOME_FEATURES,
            "pred_rank",
            "predicted_decile",
            "pred_z",
        ]
        if True
    ]
    cat_cols = [c for c in ["district", "m2_group"] if c in Xw.columns]

    n_splits = min(n_splits, max(2, len(idx) // 40))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # Prefer ridge for conservative; include HGB for full
    cands = _delta_model_candidates(fast=fast_mode)
    if mode == "conservative":
        cands = {k: v for k, v in cands.items() if k.startswith("ridge")}
    oof_delta = {name: np.zeros(len(idx), dtype=float) for name in cands}

    Xb = Xw.iloc[idx].reset_index(drop=True)
    for tr, va in cv.split(Xb):
        Xtr, Xva = Xb.iloc[tr].copy(), Xb.iloc[va].copy()
        ptr = Xtr["pred_current"].to_numpy(dtype=float)
        pva = Xva["pred_current"].to_numpy(dtype=float)
        # fold-safe predicted rank/decile/z from train pred distribution
        qs = np.quantile(ptr, np.linspace(0.1, 1.0, 10))
        Xtr["predicted_decile"] = np.clip(np.searchsorted(qs, ptr, side="right") + 1, 1, 10).astype(float)
        Xva["predicted_decile"] = np.clip(np.searchsorted(qs, pva, side="right") + 1, 1, 10).astype(float)
        # rank in [0,1]
        order = np.argsort(np.argsort(ptr))
        Xtr["pred_rank"] = order / max(len(ptr) - 1, 1)
        # validation rank vs train distribution
        Xva["pred_rank"] = np.searchsorted(np.sort(ptr), pva, side="left") / max(len(ptr), 1)
        mu, sd = float(np.mean(ptr)), float(np.std(ptr) + 1e-6)
        Xtr["pred_z"] = (ptr - mu) / sd
        Xva["pred_z"] = (pva - mu) / sd

        dy = yb[tr] - ptr
        use_num = [c for c in num_cols if c in Xtr.columns]
        Xtr_num, _ = _build_delta_matrix(Xtr, use_num, [])
        Xva_num, _ = _build_delta_matrix(Xva, use_num, [])
        Xtr_cat, Xva_cat = _onehot_cats(Xtr, Xva, cat_cols)
        Xtr_m = np.hstack([Xtr_num, Xtr_cat]) if Xtr_cat.size else Xtr_num
        Xva_m = np.hstack([Xva_num, Xva_cat]) if Xva_cat.size else Xva_num
        for name, model in cands.items():
            est = clone(model)
            with np.errstate(all="ignore"):
                est.fit(Xtr_m, dy)
                oof_delta[name][va] = np.asarray(est.predict(Xva_m), dtype=float)

    lambdas = [0.05, 0.10, 0.15, 0.20, 0.25] if mode == "conservative" else [0.05, 0.10, 0.15, 0.20, 0.25, 0.35]
    best = None
    for name, delta in oof_delta.items():
        for lam in lambdas:
            cand = np.maximum(pb + float(lam) * delta, 0.0)
            gpred = p_all.to_numpy(dtype=float).copy()
            gpred[idx] = cand
            cheap_a, rich_a = _decile_bias(yb, cand)
            row = {
                "model": name,
                "lambda": float(lam),
                "r2": _metric_r2(yb, cand),
                "var": _var_ratio(yb, cand),
                "mape": _metric_mape(yb, cand),
                "global_mape": _metric_mape(y_all.to_numpy(), gpred),
                "cheap": cheap_a,
                "rich": rich_a,
                "pred": cand,
            }
            if best is None or (row["r2"], row["var"], -row["mape"]) > (best["r2"], best["var"], -best["mape"]):
                best = row
    assert best is not None

    bias_improved = (
        (abs(best["cheap"]) < abs(cheap_b) - 1.0)
        or (abs(best["rich"]) < abs(rich_b) - 1.0)
    )
    ok = (
        best["r2"] > before_r2 + 1e-4
        and best["var"] > before_var + 1e-4
        and best["mape"] <= 0.115 + 1e-6
        and best["global_mape"] <= 0.131 + 1e-6
        and bias_improved
    )
    report.update(
        {
            "basiskele_r2_before": before_r2,
            "basiskele_r2_after": best["r2"],
            "variance_ratio_before": before_var,
            "variance_ratio_after": best["var"],
            "cheap_decile_bias_before": cheap_b,
            "cheap_decile_bias_after": best["cheap"],
            "expensive_decile_bias_before": rich_b,
            "expensive_decile_bias_after": best["rich"],
            "basiskele_mape_before": before_mape,
            "basiskele_mape_after": best["mape"],
            "global_mape_before": before_g,
            "global_mape_after": best["global_mape"],
            "selected_lambda": best["lambda"],
            "selected_model": best["model"],
        }
    )
    if ok:
        final.iloc[idx] = best["pred"]
        report["status"] = "applied"
        report["note"] = "OOF spread residual applied"
    else:
        report["status"] = "rejected_guardrail"
        report["note"] = "disabled: R2/var/MAPE/bias guardrails failed"
    return final, report
