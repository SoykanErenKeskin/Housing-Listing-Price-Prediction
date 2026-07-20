"""V13 sensitivity / coverage / decision diagnostics."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from attribute_features import V12_SAFE_REF, add_attribute_quality_features


def _top_district(X: pd.DataFrame, county: str) -> str:
    sub = X[X["county"].astype(str) == str(county)]
    if sub.empty or "district" not in sub.columns:
        return "missing"
    return str(sub["district"].fillna("missing").astype(str).value_counts().idxmax())


def _base_record(county: str, district: str) -> dict[str, Any]:
    return {
        "city": "Kocaeli",
        "county": county,
        "district": district,
        "gross_m2": 120.0,
        "net_m2": 100.0,
        "room_count": "3+1",
        "building_age": 15.0,
        "floor_num": 2.0,
        "total_floors": 5.0,
        "elevator": "Yok",
        "parking": "Yok",
        "site_inside": "Hayır",
        "heating": "Kombi (Doğalgaz)",
        "balcony": "Var",
        "furnished": "Hayır",
        "credit_eligible": "Var",
        "bathroom_count": 1,
        "m2_group": "101-125",
    }


def _predict_rows(bundle: Any, rows: list[dict[str, Any]], template: pd.DataFrame | None = None) -> np.ndarray:
    df = pd.DataFrame(rows)
    if template is not None and not template.empty:
        # align columns to training frame
        for col in template.columns:
            if col not in df.columns:
                # use mode/median from template for missing context features
                if pd.api.types.is_numeric_dtype(template[col]):
                    df[col] = pd.to_numeric(template[col], errors="coerce").median()
                else:
                    mode = template[col].dropna().astype(str)
                    df[col] = mode.mode().iloc[0] if not mode.empty else "missing"
        df = df.reindex(columns=list(dict.fromkeys(list(template.columns) + list(df.columns))), fill_value=np.nan)
    return np.asarray(bundle.predict(df), dtype=float)


def run_prediction_pair_tests(bundle: Any, X: pd.DataFrame, reports_dir) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counties = ["İzmit", "Başiskele", "Gölcük", "Karamürsel"]
    tests = [
        ("new_vs_old", {"building_age": 2.0}, {"building_age": 35.0}, "higher_first"),
        ("elevator", {"elevator": "Var"}, {"elevator": "Yok"}, "higher_first"),
        ("site_inside", {"site_inside": "Evet"}, {"site_inside": "Hayır"}, "higher_or_equal_first"),
        ("parking", {"parking": "Var"}, {"parking": "Yok"}, "higher_first"),
        ("heating", {"heating": "Yerden Isıtma"}, {"heating": "Soba"}, "higher_first"),
        ("floor", {"floor_num": 2.0, "total_floors": 5.0}, {"floor_num": 0.0, "total_floors": 4.0}, "higher_first"),
        ("net_gross", {"net_m2": 102.0}, {"net_m2": 78.0}, "higher_first"),
        (
            "total_quality",
            {
                "building_age": 2.0,
                "elevator": "Var",
                "parking": "Var",
                "site_inside": "Evet",
                "heating": "Yerden Isıtma",
                "floor_num": 2.0,
                "total_floors": 5.0,
                "net_m2": 105.0,
                "balcony": "Var",
            },
            {
                "building_age": 35.0,
                "elevator": "Yok",
                "parking": "Yok",
                "site_inside": "Hayır",
                "heating": "Soba",
                "floor_num": 0.0,
                "total_floors": 4.0,
                "net_m2": 90.0,
                "balcony": "Yok",
            },
            "higher_first",
        ),
    ]
    rows = []
    for county in counties:
        district = _top_district(X, county)
        for name, a_over, b_over, direction in tests:
            base = _base_record(county, district)
            a = {**base, **a_over}
            b = {**base, **b_over}
            preds = _predict_rows(bundle, [a, b], template=X)
            pa, pb = float(preds[0]), float(preds[1])
            diff = pa - pb
            diff_pct = diff / pb if pb else np.nan
            if direction == "higher_first":
                passed = pa > pb
                expected = "a > b"
            else:
                passed = pa >= pb
                expected = "a >= b"
            rows.append(
                {
                    "county": county,
                    "district": district,
                    "test_name": name,
                    "base_prediction_tl_m2": pb,
                    "variant_prediction_tl_m2": pa,
                    "difference_tl_m2": diff,
                    "difference_pct": diff_pct,
                    "expected_direction": expected,
                    "passed_direction_check": bool(passed),
                }
            )
    pair = pd.DataFrame(rows)
    pair.to_csv(reports_dir / "prediction_pair_tests_v13.csv", index=False, encoding="utf-8-sig")
    pair.to_csv(reports_dir / "feature_sensitivity_v13.csv", index=False, encoding="utf-8-sig")
    county = (
        pair.groupby("county", dropna=False)
        .agg(
            n_tests=("test_name", "size"),
            mean_abs_diff_tl_m2=("difference_tl_m2", lambda s: float(np.mean(np.abs(s)))),
            mean_abs_diff_pct=("difference_pct", lambda s: float(np.mean(np.abs(s)))),
            pass_rate=("passed_direction_check", "mean"),
        )
        .reset_index()
    )
    county.to_csv(reports_dir / "county_feature_sensitivity_v13.csv", index=False, encoding="utf-8-sig")
    return pair, county, pair


def run_karamursel_sensitivity(bundle: Any, X: pd.DataFrame, reports_dir) -> dict[str, Any]:
    district = _top_district(X, "Karamürsel")
    old = {
        **_base_record("Karamürsel", district),
        "gross_m2": 120.0,
        "net_m2": 90.0,
        "building_age": 35.0,
        "floor_num": 0.0,
        "total_floors": 4.0,
        "elevator": "Yok",
        "parking": "Yok",
        "site_inside": "Hayır",
        "heating": "Soba",
        "balcony": "Yok",
        "furnished": "Hayır",
    }
    new = {
        **_base_record("Karamürsel", district),
        "gross_m2": 120.0,
        "net_m2": 105.0,
        "building_age": 2.0,
        "floor_num": 2.0,
        "total_floors": 5.0,
        "elevator": "Var",
        "parking": "Var",
        "site_inside": "Evet",
        "heating": "Yerden Isıtma",
        "balcony": "Var",
        "furnished": "Hayır",
    }
    preds = _predict_rows(bundle, [old, new], template=X)
    sale_old, sale_new = float(preds[0]), float(preds[1])
    diff = sale_new - sale_old
    diff_pct = diff / sale_old if sale_old else np.nan
    warning = ""
    if pd.notna(diff_pct) and abs(diff_pct) < 0.03:
        warning = "Model is still insensitive for Karamürsel quality differences"
    rep = pd.DataFrame(
        [
            {
                "district": district,
                "sale_pred_old": sale_old,
                "sale_pred_new": sale_new,
                "sale_diff_tl_m2": diff,
                "sale_diff_pct": diff_pct,
                "warning": warning,
            }
        ]
    )
    rep.to_csv(reports_dir / "karamursel_sensitivity_v13.csv", index=False, encoding="utf-8-sig")
    return rep.iloc[0].to_dict()


def run_basiskele_variance_diagnostics(
    oof: pd.DataFrame, pair_df: pd.DataFrame, reports_dir
) -> dict[str, Any]:
    sub = oof[oof.get("county", pd.Series(dtype=str)).astype(str) == "Başiskele"].copy()
    if sub.empty:
        return {}
    actual = pd.to_numeric(sub["actual_unit_price_gross"], errors="coerce")
    pred = pd.to_numeric(sub["pred_ensemble"], errors="coerce")
    var_a = float(np.nanvar(actual))
    var_p = float(np.nanvar(pred))
    ratio = var_p / var_a if var_a > 0 else np.nan
    from sklearn.metrics import mean_absolute_error, r2_score

    mape = float(np.nanmean(np.abs(pred - actual) / actual.replace(0, np.nan)))
    r2 = float(r2_score(actual, pred)) if len(actual) > 2 else np.nan
    try:
        sub = sub.copy()
        sub["decile"] = pd.qcut(actual, 10, labels=False, duplicates="drop")
        dec = (
            sub.groupby("decile")
            .apply(lambda g: float((pd.to_numeric(g["pred_ensemble"], errors="coerce") - pd.to_numeric(g["actual_unit_price_gross"], errors="coerce")).mean()), include_groups=False)
            .reset_index(name="mean_bias")
        )
    except Exception:
        dec = pd.DataFrame()
    pair_b = pair_df[pair_df["county"] == "Başiskele"] if not pair_df.empty else pd.DataFrame()
    avg_sens = float(np.mean(np.abs(pair_b["difference_tl_m2"]))) if not pair_b.empty else np.nan
    note = ""
    if pd.notna(ratio) and ratio < 0.45:
        note = "Başiskele predictions are compressed toward mean"
    row = {
        "rows": int(len(sub)),
        "actual_variance": var_a,
        "prediction_variance": var_p,
        "variance_explained_ratio": ratio,
        "mape": mape,
        "r2": r2,
        "attribute_sensitivity_avg_abs_diff": avg_sens,
        "note": note,
        "decile_bias_json": dec.to_dict(orient="records") if not dec.empty else [],
    }
    pd.DataFrame([{k: v for k, v in row.items() if k != "decile_bias_json"}]).to_csv(
        reports_dir / "basiskele_variance_diagnostics_v13.csv", index=False, encoding="utf-8-sig"
    )
    return row


def attribute_feature_coverage(X: pd.DataFrame, reports_dir) -> pd.DataFrame:
    df = add_attribute_quality_features(X.copy())
    cols = [
        "attr_has_elevator",
        "attr_has_parking",
        "attr_has_balcony",
        "attr_is_site_inside",
        "attr_credit_eligible",
        "attr_building_age_score",
        "attr_heating_quality_score",
        "attr_total_quality_score",
    ]
    rows = []
    for county, g in df.groupby(df.get("county", pd.Series(["all"] * len(df))).astype(str), dropna=False):
        for c in cols:
            s = pd.to_numeric(g.get(c, np.nan), errors="coerce")
            rows.append(
                {
                    "county": county,
                    "feature": c,
                    "rows": int(len(g)),
                    "non_null": int(s.notna().sum()),
                    "coverage": float(s.notna().mean()) if len(g) else 0.0,
                    "mean": float(s.mean()) if s.notna().any() else np.nan,
                }
            )
    rep = pd.DataFrame(rows)
    rep.to_csv(reports_dir / "attribute_feature_coverage_v13.csv", index=False, encoding="utf-8-sig")
    return rep


def save_attribute_feature_importance(models: dict[str, Any], reports_dir) -> None:
    rows = []
    for model_name, model in models.items():
        # unwrap residual / pipeline
        est = model
        if hasattr(est, "estimator_"):
            est = est.estimator_
        if hasattr(est, "named_steps") and "model" in est.named_steps:
            inner = est.named_steps["model"]
            pre = est.named_steps.get("preprocess")
        else:
            continue
        if not hasattr(inner, "feature_importances_"):
            continue
        try:
            names = list(pre.get_feature_names_out())
        except Exception:
            continue
        imps = np.asarray(inner.feature_importances_, dtype=float)
        n = min(len(names), len(imps))
        for feat, imp in zip(names[:n], imps[:n]):
            if "attr_" in str(feat):
                rows.append({"model": model_name, "feature": feat, "importance": float(imp)})
    if not rows:
        return
    rep = pd.DataFrame(rows).sort_values("importance", ascending=False)
    rep.to_csv(reports_dir / "attribute_feature_importance_v13.csv", index=False, encoding="utf-8-sig")


def evaluate_decision(
    ensemble_metrics: dict[str, Any],
    karamursel: dict[str, Any],
    pair_df: pd.DataFrame,
    basiskele: dict[str, Any],
) -> dict[str, Any]:
    r2 = float(ensemble_metrics.get("r2", np.nan))
    mape = float(ensemble_metrics.get("mape", np.nan))
    pass_guardrail = (mape <= V12_SAFE_REF["mape"] + 0.005) and (r2 >= V12_SAFE_REF["r2"] - 0.01)
    k_diff = float(karamursel.get("sale_diff_pct", 0) or 0)
    pass_k = k_diff >= 0.03
    pass_rate = float(pair_df["passed_direction_check"].mean()) if not pair_df.empty else 0.0
    pass_dir = pass_rate >= 0.70
    pass_sensitivity = pass_k and pass_dir
    warnings = []
    qa = []
    if not pass_k:
        warnings.append("karamursel_insensitive")
        qa.append({"finding": "Karamürsel quality diff < 3%", "severity": "High"})
    if not pass_dir:
        warnings.append("direction_pass_rate_low")
        qa.append({"finding": f"Direction pass rate {pass_rate:.2%}", "severity": "Medium"})
    if not pass_guardrail:
        warnings.append("guardrail_failed")
        qa.append({"finding": "Global MAPE/R2 outside V12 tolerance", "severity": "High"})
    if basiskele.get("note"):
        warnings.append("basiskele_compressed")
        qa.append({"finding": basiskele.get("note"), "severity": "Medium"})
    overall = "PASS" if (pass_guardrail and pass_sensitivity) else "FAIL"
    return {
        "pass_guardrail": pass_guardrail,
        "pass_sensitivity": pass_sensitivity,
        "pass_karamursel": pass_k,
        "direction_pass_rate": pass_rate,
        "karamursel_sale_diff_pct": k_diff,
        "overall": overall,
        "warnings": warnings,
        "qa_findings": qa,
        "v12_delta": {
            "r2": r2 - V12_SAFE_REF["r2"],
            "mape": mape - V12_SAFE_REF["mape"],
            "mae_tl_per_m2": float(ensemble_metrics.get("mae_tl_per_m2", np.nan)) - V12_SAFE_REF["mae_tl_per_m2"],
        },
    }


def select_attribute_mode(ablation_rows: list[dict[str, Any]]) -> str:
    """Prefer full, else basic if both guardrail+sensitivity pass; none is fallback only."""
    by_mode = {str(r.get("attribute_mode")): r for r in ablation_rows}

    def ok(row: dict[str, Any]) -> bool:
        return bool(row.get("pass_guardrail")) and bool(row.get("pass_sensitivity"))

    if "full" in by_mode and ok(by_mode["full"]):
        return "full"
    if "basic" in by_mode and ok(by_mode["basic"]):
        return "basic"
    # if neither passes, keep requested preference order for reporting
    for mode in ("full", "basic", "none"):
        if mode in by_mode:
            return mode
    return "full"
