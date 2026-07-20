#!/usr/bin/env python
"""Compare two listing inputs at feature level for V17 (location + geo-context + comparable)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for cand in [here, *here.parents]:
        if (cand / "MANIFEST.json").is_file() and (cand / "data").is_dir():
            return cand
    return here.parents[1]


ROOT = _repo_root()
V17 = ROOT / "v2" / "source_versions" / "v17"
sys.path.insert(0, str(V17))
sys.path.insert(0, str(ROOT))

from attribute_features import build_debug_feature_frame  # noqa: E402
from comparable_market_features import COMPARABLE_NUMERIC_FEATURES, ComparableMarketFeatureAdder  # noqa: E402
from geo_context_features import GEO_CONTEXT_NUMERIC_FEATURES, GeoContextFeatureAdder  # noqa: E402
from location_features import LocationFeatureAdder  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def feature_diff(a: pd.DataFrame, b: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    use = cols or sorted(set(a.columns) | set(b.columns))
    rows = []
    for c in use:
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


def enrich(df: pd.DataFrame, *, location_mode: str, geo_context_mode: str, cache_dir: str) -> pd.DataFrame:
    out = build_debug_feature_frame(df, attribute_mode="full")
    out = LocationFeatureAdder(mode=location_mode if location_mode != "comparable" else "basic").fit(out).transform(out)
    out = (
        GeoContextFeatureAdder(mode=location_mode, context_mode=geo_context_mode, cache_dir=cache_dir)
        .fit(out)
        .transform(out)
    )
    # comparable needs y; use unit_price if present else zeros for debug shape only
    y = pd.to_numeric(out.get("unit_price_gross", 0.0), errors="coerce").fillna(0.0)
    if str(location_mode).lower() in {"comparable", "full"}:
        out = ComparableMarketFeatureAdder(mode=location_mode).fit(out, y).transform(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing-a", required=True, help="JSON file for listing A")
    ap.add_argument("--listing-b", required=True, help="JSON file for listing B")
    ap.add_argument("--out", default="reports/debug_pair_v17")
    ap.add_argument("--location-feature-mode", default="geo")
    ap.add_argument("--geo-context-mode", default="full")
    ap.add_argument("--geo-context-cache-dir", default="data/external/geo_context")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    a = enrich(
        pd.DataFrame([load_json(Path(args.listing_a))]),
        location_mode=args.location_feature_mode,
        geo_context_mode=args.geo_context_mode,
        cache_dir=args.geo_context_cache_dir,
    )
    b = enrich(
        pd.DataFrame([load_json(Path(args.listing_b))]),
        location_mode=args.location_feature_mode,
        geo_context_mode=args.geo_context_mode,
        cache_dir=args.geo_context_cache_dir,
    )

    loc_cols = [c for c in a.columns if c.startswith("location_") or c in {"lat", "lon", "has_lat_lon"} or "centroid" in c or c.startswith("distance_to_")]
    geo_cluster_cols = [c for c in a.columns if "geo_cluster" in c]
    dist_cols = [c for c in a.columns if c.startswith("distance_to_") or "coast" in c]
    geo_ctx_cols = [c for c in GEO_CONTEXT_NUMERIC_FEATURES if c in a.columns or c in b.columns]
    comp_cols = [c for c in COMPARABLE_NUMERIC_FEATURES if c in a.columns or c in b.columns]

    feature_diff(a, b, loc_cols).to_csv(out / "location_feature_diff.csv", index=False, encoding="utf-8-sig")
    feature_diff(a, b, comp_cols).to_csv(out / "comparable_feature_diff.csv", index=False, encoding="utf-8-sig")
    feature_diff(a, b, dist_cols + geo_ctx_cols).to_csv(out / "distance_feature_diff.csv", index=False, encoding="utf-8-sig")
    feature_diff(a, b, geo_cluster_cols).to_csv(out / "geo_cluster_diff.csv", index=False, encoding="utf-8-sig")

    summary = {
        "location_feature_mode": args.location_feature_mode,
        "geo_context_mode": args.geo_context_mode,
        "n_location_diffs": int((~feature_diff(a, b, loc_cols)["is_equal"]).sum()) if loc_cols else 0,
        "n_geo_context_diffs": int((~feature_diff(a, b, geo_ctx_cols)["is_equal"]).sum()) if geo_ctx_cols else 0,
        "n_comparable_diffs": int((~feature_diff(a, b, comp_cols)["is_equal"]).sum()) if comp_cols else 0,
        "listing_a_keys": list(load_json(Path(args.listing_a)).keys())[:40],
        "listing_b_keys": list(load_json(Path(args.listing_b)).keys())[:40],
    }
    (out / "prediction_diff.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote diffs under {out}")


if __name__ == "__main__":
    main()
