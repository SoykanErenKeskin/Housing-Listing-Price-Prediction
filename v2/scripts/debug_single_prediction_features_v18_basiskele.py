#!/usr/bin/env python
"""Debug a single Başiskele prediction feature vector (V18)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for cand in [here, *here.parents]:
        if (cand / "MANIFEST.json").is_file() and (cand / "data").is_dir():
            return cand
    return here.parents[1]


ROOT = _repo_root()
sys.path.insert(0, str(ROOT / "v2" / "source_versions" / "v18_basiskele"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Path to model_bundle_v18_basiskele.joblib")
    ap.add_argument("--row-csv", required=True, help="Single-row (or multi) CSV with listing features")
    ap.add_argument("--row-index", type=int, default=0)
    args = ap.parse_args()

    bundle = joblib.load(args.bundle)
    df = pd.read_csv(args.row_csv)
    row = df.iloc[[args.row_index]].copy()
    pred = float(bundle.predict(row)[0])
    out = {"pred_unit_price_gross": pred, "model_scope": "basiskele_only"}
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
