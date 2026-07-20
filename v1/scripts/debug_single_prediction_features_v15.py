#!/usr/bin/env python
"""Compare two listing inputs at feature (+ optional prediction) level using V15 builders."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for cand in [here, *here.parents]:
        if (cand / "MANIFEST.json").is_file() and (cand / "data").is_dir():
            return cand
    return here.parents[1]


ROOT = _repo_root()
V15 = ROOT / "v1" / "source_versions" / "v15"
sys.path.insert(0, str(V15))
sys.path.insert(0, str(ROOT))

from attribute_features import add_attribute_quality_features, build_debug_feature_frame  # noqa: E402
from county_specialist_features import (  # noqa: E402
    BASISKELE_PREMIUM_NUMERIC_FEATURES,
    LARGE_HOME_NUMERIC_FEATURES,
    BasiskelePremiumSpecialistAdder,
    LargeHomeFeatureAdder,
)
from detail_premium_features import DETAIL_GROUPS, feature_group  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def frame_from_input(obj: dict, attribute_mode: str = "full") -> pd.DataFrame:
    df = pd.DataFrame([obj])
    return build_debug_feature_frame(df, attribute_mode=attribute_mode)


def feature_diff(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    cols = sorted(set(a.columns) | set(b.columns))
    rows = []
    for c in cols:
        va = a.iloc[0][c] if c in a.columns else np.nan
        vb = b.iloc[0][c] if c in b.columns else np.nan
        equal = False
        try:
            if pd.isna(va) and pd.isna(vb):
                equal = True
            else:
                equal = bool(va == vb)
        except Exception:
            equal = str(va) == str(vb)
        rows.append({"feature": c, "value_a": va, "value_b": vb, "is_equal": equal})
    return pd.DataFrame(rows).sort_values(["is_equal", "feature"])


def subset_diff(diff: pd.DataFrame, prefixes_or_names: list[str]) -> pd.DataFrame:
    names = set(prefixes_or_names)
    mask = diff["feature"].astype(str).isin(names) | diff["feature"].astype(str).str.startswith(
        tuple(p for p in prefixes_or_names if p.endswith("_") or p.endswith("__"))
    )
    # also match exact list
    mask = diff["feature"].astype(str).isin(names) | diff["feature"].astype(str).apply(
        lambda f: any(str(f) == n or str(f).startswith(n) for n in prefixes_or_names)
    )
    return diff.loc[mask].copy()


def detail_effect_diff(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    cols = sorted(
        c
        for c in (set(a.columns) | set(b.columns))
        if str(c).startswith("detail_effect_")
    )
    rows = []
    for c in cols:
        va = float(pd.to_numeric(a.iloc[0][c] if c in a.columns else 0, errors="coerce") or 0)
        vb = float(pd.to_numeric(b.iloc[0][c] if c in b.columns else 0, errors="coerce") or 0)
        raw = c.replace("detail_effect__", "") if c.startswith("detail_effect__") else None
        rows.append(
            {
                "feature": c,
                "value_a": (a.iloc[0][raw] if raw and raw in a.columns else np.nan),
                "value_b": (b.iloc[0][raw] if raw and raw in b.columns else np.nan),
                "effect_a": va,
                "effect_b": vb,
                "diff": vb - va,
                "group": feature_group(raw)
                if raw
                else (
                    next(
                        (
                            g
                            for g in DETAIL_GROUPS
                            if f"_{g}_" in c or c.endswith(f"_{g}_sum") or c.endswith(f"_{g}_mean")
                        ),
                        "agg",
                    )
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("diff", key=lambda s: s.abs(), ascending=False)


def top_changed(diff: pd.DataFrame, n: int = 30) -> list[dict]:
    changed = diff[~diff["is_equal"]].copy()
    return changed.head(n).to_dict(orient="records")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-a", required=True)
    ap.add_argument("--input-b", required=True)
    ap.add_argument("--bundle-path", default=None)
    ap.add_argument("--attribute-mode", default="full", choices=["none", "basic", "full"])
    ap.add_argument("--detail-effect-mode", default="group", choices=["none", "group", "full"])
    ap.add_argument(
        "--basiskele-specialist-mode",
        default="premium_target_stats",
        choices=["none", "premium", "premium_target_stats", "premium_target_stats_variance_lift"],
    )
    ap.add_argument("--out", default="outputs/v15_debug_pair")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    obj_a = load_json(Path(args.input_a))
    obj_b = load_json(Path(args.input_b))
    fa = frame_from_input(obj_a, attribute_mode=args.attribute_mode)
    fb = frame_from_input(obj_b, attribute_mode=args.attribute_mode)

    used_bundle = False
    pred_a = pred_b = None
    model_version = "feature_only"
    specialist_payload: dict = {}

    if args.bundle_path:
        bundle_path = Path(args.bundle_path)
        if bundle_path.exists():
            bundle = joblib.load(bundle_path)
            used_bundle = True
            model_version = str(
                getattr(bundle, "basiskele_specialist_mode", None)
                or getattr(bundle, "detail_effect_mode", None)
                or getattr(bundle, "attribute_mode", None)
                or bundle.metrics.get("model", "v15_bundle")
            )
            template_cols = list(getattr(bundle, "feature_columns", []) or [])
            for col in template_cols:
                if col not in fa.columns:
                    fa[col] = np.nan
                if col not in fb.columns:
                    fb[col] = np.nan
            if str(getattr(bundle, "attribute_mode", args.attribute_mode)).lower() != "none":
                fa = add_attribute_quality_features(fa)
                fb = add_attribute_quality_features(fb)
            try:
                any_model = next(iter(bundle.models.values()))
                est = any_model
                if hasattr(est, "estimator_"):
                    est = est.estimator_
                if hasattr(est, "named_steps"):
                    steps = []
                    stop_at = {
                        "detail_effects",
                        "basiskele_premium",
                        "large_home_features",
                        "basiskele_premium_target_stats",
                    }
                    for name, step in est.named_steps.items():
                        steps.append((name, step))
                        if name in stop_at and name == "basiskele_premium_target_stats":
                            break
                        if name == "basiskele_premium_target_stats":
                            break
                    # Prefer stopping after large_home / target stats if present
                    steps = []
                    for name, step in est.named_steps.items():
                        steps.append((name, step))
                        if name in {
                            "basiskele_premium_target_stats",
                            "feature_columns",
                        }:
                            if name == "basiskele_premium_target_stats":
                                break
                            if name == "feature_columns" and "basiskele_premium_target_stats" not in dict(
                                steps[:-1]
                            ):
                                break
                    from sklearn.pipeline import Pipeline

                    # Build up to last specialist step before feature_columns
                    cut = []
                    for name, step in est.named_steps.items():
                        if name == "feature_columns":
                            break
                        cut.append((name, step))
                    partial = Pipeline(cut)
                    fa_t = partial.transform(fa)
                    fb_t = partial.transform(fb)
                    if isinstance(fa_t, pd.DataFrame):
                        fa = fa_t
                        fb = fb_t
            except Exception as exc:
                print(f"WARNING: could not materialize specialist features via bundle: {exc}")
                # Fallback deterministic specialist
                try:
                    adder = BasiskelePremiumSpecialistAdder(mode="premium")
                    adder.fit(pd.concat([fa, fb], ignore_index=True))
                    fa = LargeHomeFeatureAdder().fit_transform(adder.transform(fa))
                    fb = LargeHomeFeatureAdder().fit_transform(adder.transform(fb))
                except Exception as exc2:
                    print(f"WARNING: fallback specialist failed: {exc2}")
            pred_a = float(np.asarray(bundle.predict(fa), dtype=float)[0])
            pred_b = float(np.asarray(bundle.predict(fb), dtype=float)[0])
            specialist_payload = {
                "basiskele_specialist_mode": getattr(bundle, "basiskele_specialist_mode", None),
                "basiskele_variance_lift": getattr(bundle, "basiskele_variance_lift", None),
                "detail_effect_mode": getattr(bundle, "detail_effect_mode", None),
            }
        else:
            print(f"WARNING: bundle not found at {bundle_path}; falling back to feature-only diff")
    else:
        try:
            adder = BasiskelePremiumSpecialistAdder(mode="premium")
            both = pd.concat([fa, fb], ignore_index=True)
            adder.fit(both)
            fa = LargeHomeFeatureAdder().fit_transform(adder.transform(fa))
            fb = LargeHomeFeatureAdder().fit_transform(adder.transform(fb))
        except Exception as exc:
            print(f"WARNING: specialist feature-only path failed: {exc}")

    diff = feature_diff(fa, fb)
    ddiff = detail_effect_diff(fa, fb)
    bdiff = subset_diff(diff, list(BASISKELE_PREMIUM_NUMERIC_FEATURES) + ["basiskele_"])
    ldiff = subset_diff(diff, list(LARGE_HOME_NUMERIC_FEATURES) + ["large_home_"])

    fa.to_csv(out / "old_house_features.csv", index=False, encoding="utf-8-sig")
    fb.to_csv(out / "new_house_features.csv", index=False, encoding="utf-8-sig")
    fa.to_csv(out / "input_a_features.csv", index=False, encoding="utf-8-sig")
    fb.to_csv(out / "input_b_features.csv", index=False, encoding="utf-8-sig")
    diff.to_csv(out / "feature_diff.csv", index=False, encoding="utf-8-sig")
    ddiff.to_csv(out / "detail_effect_diff.csv", index=False, encoding="utf-8-sig")
    bdiff.to_csv(out / "basiskele_premium_feature_diff.csv", index=False, encoding="utf-8-sig")
    ldiff.to_csv(out / "large_home_feature_diff.csv", index=False, encoding="utf-8-sig")

    pred_payload = {
        "pred_a_tl_m2": pred_a,
        "pred_b_tl_m2": pred_b,
        "diff_tl_m2": (None if pred_a is None or pred_b is None else pred_b - pred_a),
        "diff_pct": (
            None if pred_a is None or pred_b is None or not pred_a else (pred_b - pred_a) / pred_a
        ),
        "model_version": model_version,
        "used_bundle": used_bundle,
        "detail_effect_mode": getattr(args, "detail_effect_mode", "group"),
        "basiskele_specialist_mode": getattr(args, "basiskele_specialist_mode", "premium_target_stats"),
        "top_changed_features": top_changed(diff),
        "specialist": specialist_payload,
    }
    (out / "prediction_diff.json").write_text(
        json.dumps(pred_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if used_bundle and specialist_payload:
        (out / "specialist_prediction_diff.json").write_text(
            json.dumps({**pred_payload, **specialist_payload}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(json.dumps(pred_payload, indent=2, ensure_ascii=False))
    print(f"Wrote outputs to {out.resolve()}")


if __name__ == "__main__":
    main()
