"""V15 sensitivity / coverage / decision diagnostics."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from attribute_features import V12_SAFE_REF, add_attribute_quality_features

try:
    from detail_premium_features import V13_DEFAULT_REF, V14_DEFAULT_REF
except ImportError:
    from v15.detail_premium_features import V13_DEFAULT_REF, V14_DEFAULT_REF


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
    pair.to_csv(reports_dir / "prediction_pair_tests_v15.csv", index=False, encoding="utf-8-sig")
    pair.to_csv(reports_dir / "feature_sensitivity_v15.csv", index=False, encoding="utf-8-sig")
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
    county.to_csv(reports_dir / "county_feature_sensitivity_v15.csv", index=False, encoding="utf-8-sig")
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
    rep.to_csv(reports_dir / "karamursel_sensitivity_v15.csv", index=False, encoding="utf-8-sig")
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
        reports_dir / "basiskele_variance_diagnostics_v15.csv", index=False, encoding="utf-8-sig"
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
    rep.to_csv(reports_dir / "attribute_feature_coverage_v15.csv", index=False, encoding="utf-8-sig")
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
                rows.append({"model_scope": model_name, "feature": feat, "importance": float(imp)})
    if not rows:
        return
    rep = pd.DataFrame(rows).sort_values("importance", ascending=False)
    rep["rank"] = range(1, len(rep) + 1)
    rep.to_csv(reports_dir / "attribute_feature_importance_v15.csv", index=False, encoding="utf-8-sig")


def save_detail_premium_feature_importance(models: dict[str, Any], reports_dir, county: str | None = None) -> None:
    rows = []
    for model_name, model in models.items():
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
            if "detail_effect_" in str(feat):
                rows.append(
                    {
                        "model_scope": model_name,
                        "county": county,
                        "feature": feat,
                        "importance": float(imp),
                    }
                )
    if not rows:
        return
    rep = pd.DataFrame(rows).sort_values("importance", ascending=False)
    rep["rank"] = range(1, len(rep) + 1)
    rep.to_csv(reports_dir / "detail_premium_feature_importance_v15.csv", index=False, encoding="utf-8-sig")


def evaluate_decision(
    ensemble_metrics: dict[str, Any],
    karamursel: dict[str, Any],
    pair_df: pd.DataFrame,
    basiskele: dict[str, Any],
    county_metrics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """V15 go/no-go vs V14 group defaults (V13 retained for compatibility)."""
    r2 = float(ensemble_metrics.get("r2", np.nan))
    mape = float(ensemble_metrics.get("mape", np.nan))
    # V15 global guardrail: do not break V14/V13 band
    pass_global_guardrail = (mape <= 0.134) and (r2 >= 0.670)
    pass_guardrail = pass_global_guardrail

    k_diff = float(karamursel.get("sale_diff_pct", 0) or 0)
    pass_k = k_diff >= 0.03
    pass_rate = float(pair_df["passed_direction_check"].mean()) if not pair_df.empty else 0.0
    pass_dir = pass_rate >= 0.70
    pass_detail_sensitivity = pass_k and pass_dir
    pass_sensitivity = pass_detail_sensitivity

    county_metrics = county_metrics or []
    by_county = {str(r.get("county")): r for r in county_metrics}

    def _county_r2(name: str) -> float:
        row = by_county.get(name) or {}
        return float(row.get("r2", np.nan))

    basiskele_r2 = float(basiskele.get("r2", _county_r2("Başiskele")))
    basiskele_var = float(basiskele.get("variance_explained_ratio", np.nan))
    pass_basiskele_lift = bool(
        pd.notna(basiskele_r2) and basiskele_r2 > float(V14_DEFAULT_REF["basiskele_r2"])
    )
    pass_basiskele_variance_lift = bool(
        pd.notna(basiskele_var) and basiskele_var > float(V14_DEFAULT_REF["basiskele_variance_ratio"])
    )

    karamursel_r2 = _county_r2("Karamürsel")
    if not pd.notna(karamursel_r2):
        karamursel_r2 = float("nan")
    pass_karamursel_lift = (not pd.notna(karamursel_r2)) or (
        karamursel_r2 >= float(V14_DEFAULT_REF["karamursel_r2"]) - 1e-9
    )
    pass_karamursel_guardrail = pass_karamursel_lift

    golcuk_r2 = _county_r2("Gölcük")
    izmit_r2 = _county_r2("İzmit")
    pass_golcuk_guardrail = (not pd.notna(golcuk_r2)) or (golcuk_r2 >= 0.62)

    warnings: list[str] = []
    qa: list[dict[str, Any]] = []
    top_risks: list[str] = []
    top_opportunities: list[str] = []

    if not pass_k:
        warnings.append("karamursel_insensitive")
        qa.append({"finding": "Karamürsel quality diff < 3%", "severity": "High"})
        top_risks.append("Karamürsel attribute sensitivity weak")
    if not pass_dir:
        warnings.append("direction_pass_rate_low")
        qa.append({"finding": f"Direction pass rate {pass_rate:.2%}", "severity": "Medium"})
    if not pass_global_guardrail:
        warnings.append("global_guardrail_failed")
        qa.append({"finding": "Global MAPE/R2 outside V15 guardrail (R2>=0.670, MAPE<=0.134)", "severity": "High"})
        top_risks.append("Global metrics outside V15 guardrail")
    if not pass_basiskele_lift:
        warnings.append("basiskele_no_r2_lift")
        qa.append(
            {
                "finding": f"Başiskele R2 {basiskele_r2:.4f} not above V14 {V14_DEFAULT_REF['basiskele_r2']}",
                "severity": "Medium",
            }
        )
    else:
        top_opportunities.append(f"Başiskele R2 lift to {basiskele_r2:.4f}")
    if not pass_basiskele_variance_lift:
        warnings.append("basiskele_variance_no_lift")
        qa.append(
            {
                "finding": f"Başiskele variance ratio {basiskele_var} not above V14 {V14_DEFAULT_REF['basiskele_variance_ratio']}",
                "severity": "Medium",
            }
        )
    if pd.notna(basiskele_var) and basiskele_var < 0.45:
        warnings.append("basiskele_compressed")
        qa.append({"finding": basiskele.get("note") or "Başiskele variance compressed", "severity": "Medium"})
    if not pass_karamursel_lift:
        warnings.append("karamursel_r2_regression")
        qa.append(
            {
                "finding": f"Karamürsel R2 {karamursel_r2:.4f} below V14 {V14_DEFAULT_REF['karamursel_r2']}",
                "severity": "Medium",
            }
        )
        top_risks.append("Karamürsel R2 regression vs V14")

    if pd.notna(golcuk_r2):
        if golcuk_r2 < 0.55:
            warnings.append("golcuk_r2_critical")
            qa.append({"finding": f"Gölcük R2 {golcuk_r2:.4f} < 0.55", "severity": "High"})
            top_risks.append("Gölcük R2 critically low")
        elif golcuk_r2 < 0.62:
            warnings.append("golcuk_r2_soft_floor")
            qa.append({"finding": f"Gölcük R2 {golcuk_r2:.4f} < 0.62 soft floor", "severity": "Medium"})

    county_r2_table = {
        "İzmit": izmit_r2,
        "Başiskele": basiskele_r2,
        "Gölcük": golcuk_r2,
        "Karamürsel": karamursel_r2,
    }
    ship_ready = all(pd.notna(v) and float(v) >= 0.65 for v in county_r2_table.values())
    if not ship_ready:
        qa.append(
            {
                "finding": "not ship-ready until all counties R2 >= 0.65",
                "severity": "Info",
                "county_r2": county_r2_table,
            }
        )

    # Overall PASS: global guardrail + sensitivity; county lifts reported separately (not hard fail)
    overall = "PASS" if (pass_global_guardrail and pass_detail_sensitivity) else "FAIL"
    if overall == "PASS" and not ship_ready:
        qa.append({"finding": "PASS as experiment, NOT ship-ready.", "severity": "Info"})

    return {
        "pass_global_guardrail": pass_global_guardrail,
        "pass_guardrail": pass_guardrail,
        "pass_basiskele_lift": pass_basiskele_lift,
        "pass_basiskele_variance_lift": pass_basiskele_variance_lift,
        "pass_karamursel_lift": pass_karamursel_lift,
        "pass_karamursel_guardrail": pass_karamursel_guardrail,
        "pass_golcuk_guardrail": pass_golcuk_guardrail,
        "pass_detail_sensitivity": pass_detail_sensitivity,
        "pass_sensitivity": pass_sensitivity,
        "pass_karamursel": pass_k,
        "direction_pass_rate": pass_rate,
        "karamursel_sale_diff_pct": k_diff,
        "basiskele_r2": basiskele_r2,
        "basiskele_variance_ratio": basiskele_var,
        "basiskele_lift": {
            "r2": basiskele_r2,
            "r2_delta_vs_v14": basiskele_r2 - float(V14_DEFAULT_REF["basiskele_r2"]),
            "variance_ratio": basiskele_var,
            "variance_delta_vs_v14": (
                (basiskele_var - float(V14_DEFAULT_REF["basiskele_variance_ratio"]))
                if pd.notna(basiskele_var)
                else np.nan
            ),
        },
        "golcuk_r2": golcuk_r2,
        "izmit_r2": izmit_r2,
        "karamursel_r2": karamursel_r2,
        "county_r2_table": county_r2_table,
        "ship_ready_all_counties_r2_ge_0_65": bool(ship_ready),
        "selected_detail_effect_mode": ensemble_metrics.get("detail_effect_mode"),
        "selected_basiskele_specialist_mode": ensemble_metrics.get("basiskele_specialist_mode"),
        "selected_basiskele_variance_lift_mode": ensemble_metrics.get("basiskele_variance_lift"),
        "overall": overall,
        "warnings": warnings,
        "qa_findings": qa,
        "top_risks": top_risks[:3],
        "top_opportunities": top_opportunities[:3],
        "v13_reference": dict(V13_DEFAULT_REF),
        "v14_reference": dict(V14_DEFAULT_REF),
        "v13_delta": {
            "r2": r2 - float(V13_DEFAULT_REF["r2"]),
            "mape": mape - float(V13_DEFAULT_REF["mape"]),
            "basiskele_r2": basiskele_r2 - float(V13_DEFAULT_REF["basiskele_r2"]),
            "karamursel_r2": (karamursel_r2 - float(V13_DEFAULT_REF["karamursel_r2"]))
            if pd.notna(karamursel_r2)
            else np.nan,
        },
        "v14_delta": {
            "r2": r2 - float(V14_DEFAULT_REF["r2"]),
            "mape": mape - float(V14_DEFAULT_REF["mape"]),
            "basiskele_r2": basiskele_r2 - float(V14_DEFAULT_REF["basiskele_r2"]),
            "basiskele_variance_ratio": (
                (basiskele_var - float(V14_DEFAULT_REF["basiskele_variance_ratio"]))
                if pd.notna(basiskele_var)
                else np.nan
            ),
            "karamursel_r2": (karamursel_r2 - float(V14_DEFAULT_REF["karamursel_r2"]))
            if pd.notna(karamursel_r2)
            else np.nan,
            "golcuk_r2": (golcuk_r2 - float(V14_DEFAULT_REF["golcuk_r2"])) if pd.notna(golcuk_r2) else np.nan,
        },
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
    for mode in ("full", "basic", "none"):
        if mode in by_mode:
            return mode
    return "full"


def select_detail_effect_mode(ablation_rows: list[dict[str, Any]]) -> str:
    """Prefer group; full only if global+county guardrails hold; else none."""
    by_mode = {str(r.get("detail_effect_mode")): r for r in ablation_rows}

    def ok_full(row: dict[str, Any]) -> bool:
        return bool(row.get("pass_guardrail") or row.get("pass_global_guardrail")) and bool(
            row.get("pass_karamursel_guardrail", True)
        )

    def ok_group(row: dict[str, Any]) -> bool:
        # group is default candidate even if soft county warnings exist
        return bool(row.get("pass_guardrail") or row.get("pass_global_guardrail") or True)

    if "full" in by_mode and ok_full(by_mode["full"]):
        # only take full if it does not regress global vs group when group exists
        if "group" in by_mode:
            full_mape = float(by_mode["full"].get("mape", 1e9) or 1e9)
            group_mape = float(by_mode["group"].get("mape", 1e9) or 1e9)
            full_r2 = float(by_mode["full"].get("r2", -1e9) or -1e9)
            group_r2 = float(by_mode["group"].get("r2", -1e9) or -1e9)
            if full_mape <= group_mape + 0.002 and full_r2 >= group_r2 - 0.005:
                return "full"
            return "group"
        return "full"
    if "group" in by_mode and ok_group(by_mode["group"]):
        return "group"
    for mode in ("group", "none", "full"):
        if mode in by_mode:
            return mode
    return "group"
