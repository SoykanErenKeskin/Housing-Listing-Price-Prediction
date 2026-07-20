#!/usr/bin/env python
"""Audit location / geo-context coverage for V17 (DB or CSV).

Examples:
  python scripts/audit_location_coverage_v17.py --out reports
  python scripts/audit_location_coverage_v17.py --sale-csv path/to/sales.csv --out reports
"""
from __future__ import annotations

import argparse
import os
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
sys.path.insert(0, str(ROOT / "v2" / "source_versions" / "v17"))
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    load_dotenv()
except Exception:
    pass


def _load_from_db(args: argparse.Namespace) -> pd.DataFrame:
    from sqlalchemy import create_engine, text

    url = args.db_url or os.getenv("DATABASE_URL") or ""
    if not url:
        raise SystemExit("No DATABASE_URL / --db-url; pass --sale-csv instead.")
    engine = create_engine(url)
    sql = text(
        f"""
        SELECT *
        FROM {args.sale_table}
        WHERE city = :city
        """
    )
    df = pd.read_sql(sql, engine, params={"city": args.city})
    if args.limit:
        df = df.head(int(args.limit))
    return df


def coverage_table(df: pd.DataFrame, purpose: str = "sale") -> pd.DataFrame:
    rows = []
    if "county" not in df.columns:
        df = df.copy()
        df["county"] = "unknown"
    for county, g in df.groupby(df["county"].astype(str)):
        lat = pd.to_numeric(g.get("latitude", g.get("lat", np.nan)), errors="coerce")
        lon = pd.to_numeric(g.get("longitude", g.get("lon", np.nan)), errors="coerce")
        has = lat.notna() & lon.notna()
        prec = (
            g["location_precision"].astype(str).str.strip().str.lower()
            if "location_precision" in g.columns
            else pd.Series([""] * len(g))
        )
        status = (
            g["location_backfill_status"].astype(str).str.strip().str.lower()
            if "location_backfill_status" in g.columns
            else pd.Series([""] * len(g))
        )
        src = (
            g["location_source"].astype(str)
            if "location_source" in g.columns
            else pd.Series(["missing"] * len(g))
        )
        top_src = str(src.value_counts().index[0]) if len(src) else "missing"
        rows.append(
            {
                "listing_purpose": purpose,
                "county": county,
                "rows": int(len(g)),
                "lat_lon_count": int(has.sum()),
                "lat_lon_rate": float(has.mean()) if len(g) else 0.0,
                "exact_map_count": int((prec == "exact_map").sum()),
                "exact_map_rate": float((prec == "exact_map").mean()) if len(g) else 0.0,
                "approx_map_count": int((prec == "approx_map").sum()),
                "district_only_count": int((prec == "district_only").sum()),
                "missing_count": int((~has).sum()),
                "listing_removed_count": int(status.isin(["listing_removed", "removed"]).sum()),
                "location_source_top": top_src,
                "lat_min": float(lat.min()) if has.any() else np.nan,
                "lat_max": float(lat.max()) if has.any() else np.nan,
                "lon_min": float(lon.min()) if has.any() else np.nan,
                "lon_max": float(lon.max()) if has.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def precision_by_county(df: pd.DataFrame) -> pd.DataFrame:
    if "location_precision" not in df.columns:
        return pd.DataFrame(columns=["county", "location_precision", "rows", "rate"])
    county = df["county"].astype(str) if "county" in df.columns else "unknown"
    tmp = df.copy()
    tmp["county"] = county
    tmp["location_precision"] = tmp["location_precision"].astype(str).str.strip().str.lower()
    g = tmp.groupby(["county", "location_precision"]).size().reset_index(name="rows")
    tot = g.groupby("county")["rows"].transform("sum")
    g["rate"] = g["rows"] / tot
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description="V17 location coverage audit")
    ap.add_argument("--db-url", default=None)
    ap.add_argument("--sale-table", default="sale_listings")
    ap.add_argument("--city", default="Kocaeli")
    ap.add_argument("--sale-csv", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    if args.sale_csv:
        df = pd.read_csv(args.sale_csv)
    else:
        df = _load_from_db(args)

    cov = coverage_table(df, purpose="sale")
    prec = precision_by_county(df)
    cov.to_csv(out / "location_coverage_v17.csv", index=False, encoding="utf-8-sig")
    prec.to_csv(out / "location_precision_by_county_v17.csv", index=False, encoding="utf-8-sig")
    print(cov.to_string(index=False))
    print(f"\nWrote {out / 'location_coverage_v17.csv'}")
    print(f"Wrote {out / 'location_precision_by_county_v17.csv'}")


if __name__ == "__main__":
    main()
