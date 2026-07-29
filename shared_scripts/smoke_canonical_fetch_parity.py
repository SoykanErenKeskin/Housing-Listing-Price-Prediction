"""Read-only dataset parity / fetch smoke against canonical Neon tables.

Usage (from repo root):
  python shared_scripts/smoke_canonical_fetch_parity.py

Does NOT train models or write to the database.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from canonical_db import (  # noqa: E402
    CANONICAL_NEIGHBORHOOD_DEMOGRAPHICS,
    CANONICAL_PRICE_OBSERVATIONS,
    CANONICAL_RENTAL_LISTINGS,
    CANONICAL_SALE_LISTINGS,
    fetch_demographics_table,
    fetch_latest_trend_table,
    fetch_listing_table,
)
from db_url import create_sqlalchemy_engine, sanitize_db_error, safe_database_label  # noqa: E402
from env_loader import load_root_env  # noqa: E402

CITY = "Kocaeli"
COUNTIES = ["Başiskele", "İzmit", "Gölcük", "Karamürsel", "Kartepe"]
LIMIT = 500  # bounded sample for smoke/parity shape checks


def _source_site() -> str:
    import os

    return (os.getenv("SOURCE_SITE") or "").strip() or "listing_portal"


def _summarize_listings(df, purpose: str) -> dict:
    out: dict = {
        "purpose": purpose,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "has_city": "city" in df.columns,
        "has_county": "county" in df.columns,
        "has_district": "district" in df.columns,
        "has_province_physical": "province" in df.columns,
        "null_city": int(df["city"].isna().sum()) if "city" in df.columns else None,
        "null_county": int(df["county"].isna().sum()) if "county" in df.columns else None,
        "null_district": int(df["district"].isna().sum()) if "district" in df.columns else None,
        "coord_coverage": None,
        "county_counts": {},
        "target_nulls": None,
        "target_min": None,
        "target_median": None,
        "target_max": None,
        "duplicate_classified_id": None,
    }
    if df.empty:
        return out
    if "latitude" in df.columns and "longitude" in df.columns:
        ok = df["latitude"].notna() & df["longitude"].notna()
        out["coord_coverage"] = float(ok.mean())
    if "county" in df.columns:
        out["county_counts"] = {
            str(k): int(v) for k, v in df["county"].value_counts(dropna=False).items()
        }
    if "classified_id" in df.columns:
        out["duplicate_classified_id"] = int(df["classified_id"].duplicated().sum())
    if purpose == "sale" and "unit_price_gross" in df.columns:
        t = df["unit_price_gross"]
        out["target_nulls"] = int(t.isna().sum())
        if t.notna().any():
            out["target_min"] = float(t.min())
            out["target_median"] = float(t.median())
            out["target_max"] = float(t.max())
    if purpose == "rental" and "rent_per_m2_gross" in df.columns:
        t = df["rent_per_m2_gross"]
        out["target_nulls"] = int(t.isna().sum())
        if t.notna().any():
            out["target_min"] = float(t.min())
            out["target_median"] = float(t.median())
            out["target_max"] = float(t.max())
    return out


def main() -> int:
    load_root_env(start=ROOT)
    print(f"target={safe_database_label()}")
    engine = create_sqlalchemy_engine()
    source_site = _source_site()

    report: dict = {"tables": {}, "fetches": {}, "notes": []}

    try:
        sales = fetch_listing_table(
            engine,
            CANONICAL_SALE_LISTINGS,
            "sale",
            CITY,
            limit=LIMIT,
            counties=COUNTIES,
            source_site=source_site,
            filter_source_site=True,
        )
        rentals = fetch_listing_table(
            engine,
            CANONICAL_RENTAL_LISTINGS,
            "rental",
            CITY,
            limit=LIMIT,
            counties=COUNTIES,
            source_site=source_site,
            filter_source_site=True,
        )
        trend = fetch_latest_trend_table(engine, CANONICAL_PRICE_OBSERVATIONS, CITY)
        demo = fetch_demographics_table(
            engine, CANONICAL_NEIGHBORHOOD_DEMOGRAPHICS, city=CITY
        )
    except Exception as exc:
        print(f"FAIL fetch: {sanitize_db_error(exc)}")
        return 2

    report["fetches"]["sale"] = _summarize_listings(sales, "sale")
    report["fetches"]["rental"] = _summarize_listings(rentals, "rental")
    report["fetches"]["trend"] = {
        "rows": int(len(trend)),
        "has_city_name": "city_name" in trend.columns,
        "has_county_name": "county_name" in trend.columns,
        "has_district_name": "district_name" in trend.columns,
        "has_district_id": "district_id" in trend.columns,
        "has_province_name_physical": "province_name" in trend.columns,
    }
    report["fetches"]["demographics"] = {
        "rows": int(len(demo)),
        "has_city_id": "city_id" in demo.columns,
        "has_county_id": "county_id" in demo.columns,
        "has_district_id": "district_id" in demo.columns,
        "has_province_id_physical": "province_id" in demo.columns,
    }

    # Classification notes for parity (no legacy DB to diff against).
    report["notes"].append(
        "Legacy Neon public.* tables are no longer available; "
        "parity is shape/contract based on a bounded LIMIT sample."
    )
    report["notes"].append(
        "Listing/trend/demo frames are aliased to legacy city/county/district "
        "names for artifact compatibility; physical province_* columns should "
        "not remain after alias."
    )

    ok = True
    for key in ("sale", "rental"):
        f = report["fetches"][key]
        if not f["has_city"] or not f["has_county"] or not f["has_district"]:
            ok = False
            print(f"FAIL {key}: missing legacy location columns after alias")
        if f["has_province_physical"]:
            ok = False
            print(f"FAIL {key}: physical province column leaked into aliased frame")
        if f["rows"] <= 0:
            ok = False
            print(f"FAIL {key}: zero rows (check SOURCE_SITE / filters)")

    t = report["fetches"]["trend"]
    if t["rows"] <= 0 or not t["has_city_name"] or not t["has_county_name"]:
        ok = False
        print("FAIL trend: missing rows or legacy name columns")

    d = report["fetches"]["demographics"]
    if d["rows"] <= 0 or not d["has_city_id"] or not d["has_district_id"]:
        ok = False
        print("FAIL demographics: missing rows or legacy id columns")

    out_path = HERE / "_canonical_fetch_parity_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote={out_path.name}")
    print(json.dumps(report["fetches"], indent=2, ensure_ascii=False)[:4000])
    if ok:
        print("parity_smoke=PASS")
        return 0
    print("parity_smoke=FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
