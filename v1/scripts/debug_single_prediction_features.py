#!/usr/bin/env python
"""Compare two listing inputs at feature (+ optional prediction) level using V13 builders."""
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
V13 = ROOT / "v1" / "source_versions" / "v13"
sys.path.insert(0, str(V13))
sys.path.insert(0, str(ROOT))

from attribute_features import add_attribute_quality_features, build_debug_feature_frame  # noqa: E402


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


def top_changed(diff: pd.DataFrame, n: int = 30) -> list[dict]:
    changed = diff[~diff["is_equal"]].copy()
    return changed.head(n).to_dict(orient="records")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-a", required=True)
    ap.add_argument("--input-b", required=True)
    ap.add_argument("--bundle-path", default=None)
    ap.add_argument("--attribute-mode", default="full", choices=["none", "basic", "full"])
    ap.add_argument("--out", default="outputs/v13_debug_pair")
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
            model_version = str(getattr(bundle, "attribute_mode", None) or bundle.metrics.get("model", "v13_bundle"))
            # Prefer full training columns when available
            template_cols = list(getattr(bundle, "feature_columns", []) or [])
            for col in template_cols:
                if col not in fa.columns:
                    fa[col] = np.nan
                if col not in fb.columns:
                    fb[col] = np.nan
            # Re-run attribute quality so attr_* exist for predict path
            if str(getattr(bundle, "attribute_mode", args.attribute_mode)).lower() != "none":
                fa = add_attribute_quality_features(fa)
                fb = add_attribute_quality_features(fb)
            pred_a = float(np.asarray(bundle.predict(fa), dtype=float)[0])
            pred_b = float(np.asarray(bundle.predict(fb), dtype=float)[0])
        else:
            print(f"WARNING: bundle not found at {bundle_path}; falling back to feature-only diff")

    diff = feature_diff(fa, fb)
    fa.to_csv(out / "old_house_features.csv", index=False, encoding="utf-8-sig")
    fb.to_csv(out / "new_house_features.csv", index=False, encoding="utf-8-sig")
    # Keep names aligned with CLI inputs (a/b) as well
    fa.to_csv(out / "input_a_features.csv", index=False, encoding="utf-8-sig")
    fb.to_csv(out / "input_b_features.csv", index=False, encoding="utf-8-sig")
    diff.to_csv(out / "feature_diff.csv", index=False, encoding="utf-8-sig")

    pred_payload = {
        "pred_a_tl_m2": pred_a,
        "pred_b_tl_m2": pred_b,
        "diff_tl_m2": (None if pred_a is None or pred_b is None else pred_b - pred_a),
        "diff_pct": (
            None
            if pred_a is None or pred_b is None or not pred_a
            else (pred_b - pred_a) / pred_a
        ),
        "model_version": model_version,
        "used_bundle": used_bundle,
        "top_changed_features": top_changed(diff),
    }
    (out / "prediction_diff.json").write_text(json.dumps(pred_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(pred_payload, indent=2, ensure_ascii=False))
    print(f"Wrote outputs to {out.resolve()}")


if __name__ == "__main__":
    main()
