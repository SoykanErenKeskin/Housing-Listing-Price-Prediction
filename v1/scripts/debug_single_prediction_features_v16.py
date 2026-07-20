#!/usr/bin/env python
"""Compare two listing inputs at feature (+ optional prediction) level using V16 builders."""
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
V16 = ROOT / "v1" / "source_versions" / "v16"
sys.path.insert(0, str(V16))
sys.path.insert(0, str(ROOT))

from attribute_features import add_attribute_quality_features, build_debug_feature_frame  # noqa: E402
from county_specialist_features import (  # noqa: E402
    BASISKELE_PREMIUM_NUMERIC_FEATURES,
    LARGE_HOME_NUMERIC_FEATURES,
    BasiskelePremiumSpecialistAdder,
    LargeHomeFeatureAdder,
)
from detail_premium_features import DETAIL_GROUPS, feature_group  # noqa: E402
from regime_residual_layers import (  # noqa: E402
    BSK_LARGE_HOME_FEATURES,
    KARAMURSEL_BASELINE_FEATURES,
    BasiskeleLargeHomeRegimeAdder,
)


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


def detail_effect_diff(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    cols = sorted(c for c in (set(a.columns) | set(b.columns)) if str(c).startswith("detail_effect_"))
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


def apply_manual_specialists(
    df: pd.DataFrame,
    *,
    specialist_mode: str,
    large_home_regime: str,
) -> pd.DataFrame:
    out = df.copy()
    out = BasiskelePremiumSpecialistAdder(mode=specialist_mode).fit(out).transform(out)
    out = LargeHomeFeatureAdder().fit(out).transform(out)
    out = BasiskeleLargeHomeRegimeAdder(mode=large_home_regime).fit(out).transform(out)
    return out


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
    ap.add_argument("--basiskele-large-home-regime", default="simple", choices=["none", "simple", "residual"])
    ap.add_argument("--out", default="outputs/v16_debug_pair")
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

    if args.bundle_path:
        bundle_path = Path(args.bundle_path)
        if bundle_path.exists():
            bundle = joblib.load(bundle_path)
            used_bundle = True
            model_version = str(
                getattr(bundle, "basiskele_large_home_regime", None)
                or getattr(bundle, "basiskele_specialist_mode", None)
                or getattr(bundle, "detail_effect_mode", None)
                or "v16_bundle"
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
                    from sklearn.pipeline import Pipeline

                    cut = []
                    for name, step in est.named_steps.items():
                        if name == "feature_columns":
                            break
                        cut.append((name, step))
                    partial = Pipeline(cut)
                    fa = partial.transform(fa)
                    fb = partial.transform(fb)
            except Exception as exc:
                print(f"partial pipeline transform failed: {exc}; falling back to manual specialists")
                fa = apply_manual_specialists(
                    fa,
                    specialist_mode=args.basiskele_specialist_mode,
                    large_home_regime=args.basiskele_large_home_regime,
                )
                fb = apply_manual_specialists(
                    fb,
                    specialist_mode=args.basiskele_specialist_mode,
                    large_home_regime=args.basiskele_large_home_regime,
                )
            try:
                pred_a = float(np.asarray(bundle.predict(fa)).ravel()[0])
                pred_b = float(np.asarray(bundle.predict(fb)).ravel()[0])
            except Exception as exc:
                print(f"bundle predict failed: {exc}")
        else:
            print(f"bundle not found: {bundle_path}")

    if not used_bundle:
        fa = apply_manual_specialists(
            fa,
            specialist_mode=args.basiskele_specialist_mode,
            large_home_regime=args.basiskele_large_home_regime,
        )
        fb = apply_manual_specialists(
            fb,
            specialist_mode=args.basiskele_specialist_mode,
            large_home_regime=args.basiskele_large_home_regime,
        )

    diff = feature_diff(fa, fb)
    diff.to_csv(out / "feature_diff_v16.csv", index=False, encoding="utf-8-sig")
    detail_effect_diff(fa, fb).to_csv(out / "detail_effect_diff_v16.csv", index=False, encoding="utf-8-sig")

    regime_names = list(BSK_LARGE_HOME_FEATURES) + list(KARAMURSEL_BASELINE_FEATURES)
    specialist_names = list(BASISKELE_PREMIUM_NUMERIC_FEATURES) + list(LARGE_HOME_NUMERIC_FEATURES)
    focus = diff[diff["feature"].astype(str).isin(regime_names + specialist_names)].copy()
    focus.to_csv(out / "regime_specialist_diff_v16.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model_version": model_version,
        "used_bundle": used_bundle,
        "pred_a": pred_a,
        "pred_b": pred_b,
        "pred_diff": (None if pred_a is None or pred_b is None else pred_b - pred_a),
        "n_features_changed": int((~diff["is_equal"]).sum()),
        "top_changed": diff.loc[~diff["is_equal"]].head(40).to_dict(orient="records"),
    }
    (out / "pair_summary_v16.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote: {out.resolve()}")


if __name__ == "__main__":
    main()
