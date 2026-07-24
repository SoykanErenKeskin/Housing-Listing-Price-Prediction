"""V22 satellite diagnostics, selection gates, and optional spatial leakage check."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from satellite_feature_builder import get_model_satellite_feature_names, get_satellite_feature_names

V21_REF = {
    "r2": 0.5059,
    "mape": 0.1055,
    "variance_ratio": 0.4590,
}

# Selection gates vs V21
R2_MIN = 0.5059
MAPE_MAX = 0.1105
COVERAGE_MIN = 0.65
# "not much worse" than V21 expensive bias (−10159): allow ~5% relative slack
EXPENSIVE_BIAS_FLOOR = -10159 * 1.15  # more negative = worse underprediction


def satellite_feature_coverage_table(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    cols = get_satellite_feature_names(mode if mode != "none" else "full")
    used, zero, _qa = get_model_satellite_feature_names(mode if mode != "none" else "full", df)
    used_set = set(used)
    zero_set = set(zero)
    rows = []
    has = (
        pd.to_numeric(df.get("sat_has_features"), errors="coerce").fillna(0)
        if "sat_has_features" in df.columns
        else pd.Series(0, index=df.index)
    )
    has_zv = "sat_has_features" in zero_set
    rows.append(
        {
            "feature": "sat_has_features",
            "non_null_rate": float((has > 0).mean()) if len(df) else 0.0,
            "mean": float(has.mean()) if len(df) else np.nan,
            "std": float(has.std()) if len(df) else np.nan,
            "zero_variance": bool(has_zv),
            "used_in_model": "sat_has_features" in used_set,
            "note": "zero_variance_excluded" if has_zv else "",
        }
    )
    for c in cols:
        if c == "sat_has_features":
            continue
        if c not in df.columns:
            rows.append(
                {
                    "feature": c,
                    "non_null_rate": 0.0,
                    "mean": np.nan,
                    "std": np.nan,
                    "zero_variance": True,
                    "used_in_model": False,
                    "note": "zero_variance_excluded" if c in zero_set else "missing_column",
                }
            )
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        note = "zero_variance_excluded" if c in zero_set else ""
        rows.append(
            {
                "feature": c,
                "non_null_rate": float(s.notna().mean()),
                "mean": float(s.mean()) if s.notna().any() else np.nan,
                "std": float(s.std()) if s.notna().sum() > 1 else np.nan,
                "zero_variance": c in zero_set,
                "used_in_model": c in used_set,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def satellite_feature_distribution_table(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """QA distribution for all mode satellite features (including zero-variance excluded)."""
    mode_eff = mode if mode != "none" else "full"
    cols = get_satellite_feature_names(mode_eff)
    used, zero, _qa = get_model_satellite_feature_names(mode_eff, df)
    used_set = set(used)
    zero_set = set(zero)
    n = max(len(df), 1)
    rows = []
    for c in cols:
        if c not in df.columns:
            rows.append(
                {
                    "feature": c,
                    "missing_rate": 1.0,
                    "min": np.nan,
                    "median": np.nan,
                    "mean": np.nan,
                    "max": np.nan,
                    "std": np.nan,
                    "nunique": 0,
                    "zero_variance": True,
                    "used_in_model": False,
                    "note": "zero_variance_excluded",
                }
            )
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        nn = s.dropna()
        nunique = int(nn.nunique(dropna=True)) if len(nn) else 0
        is_zero = c in zero_set
        rows.append(
            {
                "feature": c,
                "missing_rate": float(s.isna().mean()) if n else 1.0,
                "min": float(nn.min()) if len(nn) else np.nan,
                "median": float(nn.median()) if len(nn) else np.nan,
                "mean": float(nn.mean()) if len(nn) else np.nan,
                "max": float(nn.max()) if len(nn) else np.nan,
                "std": float(nn.std(ddof=0)) if len(nn) else np.nan,
                "nunique": nunique,
                "zero_variance": bool(is_zero),
                "used_in_model": bool(c in used_set),
                "note": "zero_variance_excluded" if is_zero else ("binary_like" if nunique == 2 else ""),
            }
        )
    return pd.DataFrame(rows)


def is_satellite_arm_eligible(row: dict[str, Any]) -> bool:
    return bool(evaluate_satellite_candidate(row).get("eligible"))


def evaluate_satellite_candidate(row: dict[str, Any]) -> dict[str, Any]:
    """Selection gates vs V21 reference for a satellite candidate row/metrics dict."""
    name = str(row.get("experiment") or row.get("candidate_experiment") or "")
    notes: list[str] = []
    if name == "control_v21":
        return {"eligible": False, "notes": ["control_not_candidate"], "reason": "control"}

    def _f(key: str, default: float = float("nan")) -> float:
        raw = row.get(key)
        if raw is None or raw == "":
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    r2 = _f("r2")
    mape = _f("mape")
    cov = _f("sat_feature_coverage", 0.0)
    exp_bias = _f("expensive_decile_bias")

    if not np.isfinite(r2) or r2 <= R2_MIN:
        if "basic" in name:
            notes.append("sat_basic_did_not_beat_v21_reference")
        else:
            notes.append("candidate_did_not_beat_v21_reference")
    if not np.isfinite(mape) or mape > MAPE_MAX:
        notes.append("mape_guardrail_fail")
    if cov < COVERAGE_MIN:
        notes.append(f"low_sat_coverage ({cov:.3f} < {COVERAGE_MIN})")
    if np.isfinite(exp_bias) and exp_bias < EXPENSIVE_BIAS_FLOOR:
        notes.append("expensive_decile_bias_much_worse")
    upstream = str(row.get("notes") or "")
    if "leakage" in upstream.lower():
        notes.append("leakage")

    eligible = len(notes) == 0
    if eligible:
        notes = ["passes_v21_selection_gates"]
    return {"eligible": eligible, "notes": notes, "reason": "pass" if eligible else "gates_failed"}


def select_satellite_experiment(rows: list[dict[str, Any]]) -> tuple[str, str]:
    """Return (selected_experiment, decision_overall).

    If no satellite arm clears V21 gates → control_v21 + DIAGNOSTIC_NO_LIFT.
    """
    sat_rows = [r for r in rows if str(r.get("experiment") or "") != "control_v21"]
    eligible = [r for r in sat_rows if is_satellite_arm_eligible(r)]
    if not eligible:
        # Prefer DIAGNOSTIC_NO_LIFT when sat arms ran but none beat V21.
        decision = "DIAGNOSTIC_NO_LIFT" if sat_rows else "DIAGNOSTIC_FAIL"
        return "control_v21", decision
    best = max(
        eligible,
        key=lambda r: (
            float(r.get("r2") or -1e9),
            -float(r.get("mape") or 1e9),
            float(r.get("variance_ratio") or -1e9),
            float(r.get("expensive_decile_bias") or -1e9),
        ),
    )
    return str(best["experiment"]), "PASS"


def write_metrics_summary_v22(
    path: Path,
    *,
    selected_experiment: str | None,
    sat_coverage: float,
    metrics: dict[str, Any],
    decision_overall: str,
    warnings: list[str] | None = None,
    qa_findings: list[str] | None = None,
    candidate_experiment: str | None = None,
    notes: list[str] | None = None,
    coverage_definition: str | None = None,
) -> None:
    ens = metrics.get("ensemble") or {}
    payload = {
        "generation": "v4_visual_satellite_experiments",
        "version": "v22_basiskele_satellite_environment_pilot",
        "base_checkpoint": "v21_basiskele_site_project_extraction",
        "reference": dict(V21_REF),
        "selected_experiment": selected_experiment,
        "candidate_experiment": candidate_experiment,
        "satellite_feature_coverage": sat_coverage,
        "coverage_definition": coverage_definition
        or "non_null_any_of_sat_ndvi_mean_250m|sat_ndbi_mean_250m|sat_ndwi_mean_250m",
        "r2": metrics.get("r2", ens.get("r2")),
        "mape": metrics.get("mape", ens.get("mape")),
        "variance_ratio": metrics.get("variance_ratio", ens.get("basiskele_variance_ratio")),
        "log_r2": ens.get("log_r2"),
        "large_home_r2": metrics.get("large_home_r2"),
        "expensive_decile_bias": metrics.get("expensive_decile_bias"),
        "cheap_decile_bias": metrics.get("cheap_decile_bias"),
        "notes": list(notes or []),
        "decision": {
            "overall": decision_overall,
            "warnings": list(warnings or []),
            "qa_findings": list(qa_findings or []),
            "notes": list(notes or []),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_satellite_feature_importance(
    models: dict[str, Any] | None,
    reports_dir: Path,
    *,
    used_sat_features: list[str] | None = None,
    get_inner_model=None,
    get_preprocess_feature_names=None,
) -> dict[str, Any]:
    """Write satellite_feature_importance_v22.csv from global FI or tree models.

    Prefer filtering feature_importance_v18_basiskele.csv (already written).
    Returns status dict; on failure writes empty CSV + notes for smoke diagnostics.
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "satellite_feature_importance_v22.csv"
    notes_path = reports_dir / "satellite_feature_importance_v22_notes.txt"
    notes: list[str] = []

    def _write_empty(extra: list[str]) -> dict[str, Any]:
        all_notes = list(notes) + list(extra)
        pd.DataFrame(columns=["feature", "importance_mean", "n_models"]).to_csv(
            out_path, index=False, encoding="utf-8-sig"
        )
        notes_path.write_text("\n".join(all_notes), encoding="utf-8")
        return {"ok": False, "notes": all_notes, "path": str(out_path)}

    def _is_sat_name(feat: str) -> bool:
        s = str(feat)
        base = s.split("__")[-1] if "__" in s else s
        if used_sat_features and (base in used_sat_features or s in used_sat_features):
            return True
        return base.startswith("sat_") or ("__sat_" in s) or s.startswith("sat_")

    # 1) Prefer already-written global importance table
    fi = reports_dir / "feature_importance_v18_basiskele.csv"
    if fi.exists():
        try:
            fi_df = pd.read_csv(fi)
            feat_col = "feature" if "feature" in fi_df.columns else fi_df.columns[0]
            imp_col = (
                "importance"
                if "importance" in fi_df.columns
                else ("importance_mean" if "importance_mean" in fi_df.columns else None)
            )
            if imp_col is not None:
                sat = fi_df[fi_df[feat_col].astype(str).map(_is_sat_name)].copy()
                if not sat.empty:
                    sat = sat.assign(feature_base=sat[feat_col].astype(str).str.split("__").str[-1])
                    if "model" in sat.columns:
                        agg = (
                            sat.groupby("feature_base", as_index=False)
                            .agg(importance_mean=(imp_col, "mean"), n_models=("model", "nunique"))
                            .rename(columns={"feature_base": "feature"})
                        )
                    else:
                        agg = (
                            sat.groupby("feature_base", as_index=False)[imp_col]
                            .mean()
                            .rename(columns={"feature_base": "feature", imp_col: "importance_mean"})
                        )
                        agg["n_models"] = 1
                    agg = agg.sort_values("importance_mean", ascending=False)
                    agg.to_csv(out_path, index=False, encoding="utf-8-sig")
                    notes.append("derived_from_feature_importance_v18")
                    notes_path.write_text("\n".join(notes), encoding="utf-8")
                    return {"ok": True, "notes": notes, "path": str(out_path), "n_features": int(len(agg))}
                notes.append("no_sat_rows_in_feature_importance_v18")
        except Exception as exc:
            notes.append(f"feature_importance_v18_parse_failed:{exc}")

    if not models or get_inner_model is None or get_preprocess_feature_names is None:
        return _write_empty(["satellite_feature_importance_unavailable_no_models"])

    rows: list[dict[str, Any]] = []
    for model_name, model in models.items():
        inner = get_inner_model(model)
        if inner is None or not hasattr(inner, "feature_importances_"):
            continue
        feat_names = get_preprocess_feature_names(model)
        if feat_names is None:
            continue
        imps = np.asarray(inner.feature_importances_, dtype=float)
        n = min(len(feat_names), len(imps))
        for fname, imp in zip(feat_names[:n], imps[:n]):
            if _is_sat_name(str(fname)):
                base = str(fname).split("__")[-1]
                rows.append({"feature": base, "model": model_name, "importance": float(imp)})

    if not rows:
        return _write_empty(["satellite_feature_importance_unavailable_no_tree_importances"])

    det = pd.DataFrame(rows)
    agg = det.groupby("feature", as_index=False).agg(
        importance_mean=("importance", "mean"), n_models=("model", "nunique")
    )
    agg = agg.sort_values("importance_mean", ascending=False)
    agg.to_csv(out_path, index=False, encoding="utf-8-sig")
    notes.append("derived_from_tree_feature_importances")
    notes_path.write_text("\n".join(notes), encoding="utf-8")
    return {"ok": True, "notes": notes, "path": str(out_path), "n_features": int(len(agg))}


def run_spatial_leakage_diagnostic(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    group_col: str = "district",
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Lightweight random vs grouped CV R² comparison on a linear probe of sat features.

    Does not retrain the full V21 stack — diagnostic only.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
    from sklearn.pipeline import Pipeline

    sat_cols = [c for c in X.columns if str(c).startswith("sat_") and c != "sat_missing_reason"]
    rows: list[dict[str, Any]] = []
    if not sat_cols or group_col not in X.columns:
        rows.append(
            {
                "diagnostic": "spatial_leakage",
                "status": "skipped",
                "reason": "missing_sat_cols_or_group",
                "random_cv_r2": np.nan,
                "grouped_cv_r2": np.nan,
                "warning": "",
            }
        )
        return pd.DataFrame(rows)

    Xs = X[sat_cols].apply(pd.to_numeric, errors="coerce")
    mask = Xs.notna().any(axis=1) & y.notna()
    Xs = Xs.loc[mask]
    ys = pd.to_numeric(y.loc[mask], errors="coerce")
    groups = X.loc[mask, group_col].astype(str).fillna("missing")
    if len(Xs) < 50:
        rows.append(
            {
                "diagnostic": "spatial_leakage",
                "status": "skipped",
                "reason": "too_few_rows",
                "random_cv_r2": np.nan,
                "grouped_cv_r2": np.nan,
                "warning": "",
            }
        )
        return pd.DataFrame(rows)

    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )
    kf = KFold(n_splits=min(n_splits, max(2, len(Xs) // 20)), shuffle=True, random_state=random_state)
    pred_rand = cross_val_predict(pipe, Xs, ys, cv=kf)
    r2_rand = float(r2_score(ys, pred_rand))

    n_groups = groups.nunique()
    gkf_splits = min(n_splits, max(2, n_groups))
    warning = ""
    try:
        gkf = GroupKFold(n_splits=gkf_splits)
        pred_grp = cross_val_predict(pipe, Xs, ys, cv=gkf, groups=groups)
        r2_grp = float(r2_score(ys, pred_grp))
    except Exception as exc:
        r2_grp = float("nan")
        warning = f"grouped_cv_failed:{exc}"

    if np.isfinite(r2_rand) and np.isfinite(r2_grp) and r2_rand > 0.05 and (r2_rand - r2_grp) > 0.05:
        warning = (warning + "; " if warning else "") + "satellite_lift_may_be_spatial_memorization"

    rows.append(
        {
            "diagnostic": "spatial_leakage",
            "status": "ok",
            "reason": "",
            "random_cv_r2": r2_rand,
            "grouped_cv_r2": r2_grp,
            "delta_random_minus_grouped": (r2_rand - r2_grp) if np.isfinite(r2_rand) and np.isfinite(r2_grp) else np.nan,
            "n_rows": int(len(Xs)),
            "n_groups": int(n_groups),
            "group_col": group_col,
            "warning": warning,
        }
    )
    return pd.DataFrame(rows)
