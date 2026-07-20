#!/usr/bin/env python
"""Audit Başiskele location / lat-lon coverage for V18."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for cand in [here, *here.parents]:
        if (cand / "MANIFEST.json").is_file() and (cand / "data").is_dir():
            return cand
    return here.parents[1]


ROOT = _repo_root()
sys.path.insert(0, str(ROOT / "v2" / "source_versions" / "v18_basiskele"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    load_dotenv()
except Exception:
    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", default=os.getenv("DATABASE_URL") or os.getenv("DB_URL") or "")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if not args.db_url:
        raise SystemExit("Set DATABASE_URL / --db-url")

    from train_v18_basiskele_comparable_pipeline import fetch_listing_table, create_db_engine

    eng = create_db_engine(args.db_url)
    sales = fetch_listing_table(eng, "sale_listings", "sale", "Kocaeli", limit=args.limit, county="Başiskele")
    counties = sorted(sales["county"].dropna().astype(str).unique().tolist()) if "county" in sales.columns else []
    lat = pd.to_numeric(sales.get("latitude", sales.get("lat")), errors="coerce")
    lon = pd.to_numeric(sales.get("longitude", sales.get("lon")), errors="coerce")
    has = lat.notna() & lon.notna()
    report = {
        "rows": int(len(sales)),
        "counties": counties,
        "basiskele_only_ok": counties == ["Başiskele"] or (len(counties) == 1 and counties[0] == "Başiskele"),
        "lat_lon_coverage": float(has.mean()) if len(sales) else 0.0,
        "with_coords": int(has.sum()),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["basiskele_only_ok"]:
        raise SystemExit("FAIL: non-Başiskele counties present")


if __name__ == "__main__":
    main()
