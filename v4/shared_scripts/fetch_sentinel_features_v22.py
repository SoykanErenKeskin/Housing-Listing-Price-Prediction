#!/usr/bin/env python
"""Fetch free Sentinel-2 environment features for Başiskele (V22).

Modes:
  --source gee         Google Earth Engine (optional; earthengine-api)
  --source cached_csv  Reuse existing CSV / write template if missing

Resume-safe GEE mode:
  --resume / --no-resume   skip classified_ids already in output CSV (default: resume)
  --save-every N           incremental CSV write every N new points (default: 25)

Does NOT train a model. Does NOT use paid map APIs or OSM tile bulk download.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PKG = ROOT / "v4" / "source_versions" / "v22_basiskele_satellite_environment_pilot"
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from satellite_feature_builder import (  # noqa: E402
    ALL_SATELLITE_NUMERIC_FEATURES,
    TEMPLATE_COLUMNS,
    empty_template_row,
)


DEFAULT_OUT = ROOT / "data" / "external" / "satellite_features" / "basiskele" / "sentinel_features_v22.csv"
DEFAULT_META = ROOT / "data" / "external" / "satellite_features" / "basiskele" / "metadata_v22.json"
DEFAULT_FAILED = ROOT / "data" / "external" / "satellite_features" / "basiskele" / "sentinel_features_v22_failed.csv"
README_FETCH = ROOT / "data" / "external" / "satellite_features" / "basiskele" / "README_FETCH_REQUIRED.md"


def _write_fetch_readme(out_csv: Path) -> None:
    README_FETCH.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Satellite features fetch required (V22)

Free Sentinel / GEE environment features were not generated yet.

## Expected output

`{out_csv.as_posix()}`

## How to fetch (Google Earth Engine)

```powershell
python v4/shared_scripts/fetch_sentinel_features_v22.py `
  --city Kocaeli --county Başiskele `
  --out {out_csv.as_posix()} `
  --source gee --resume --save-every 25
```

Smoke CSV rows are kept and skipped on resume. Ctrl+C saves progress.

## Rules

- No Google Maps Static API / paid tiles / OSM bulk download / CNN fine-tune
"""
    README_FETCH.write_text(text, encoding="utf-8")


def _write_template_csv(out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([empty_template_row()], columns=TEMPLATE_COLUMNS).to_csv(
        out_csv, index=False, encoding="utf-8-sig"
    )
    _write_fetch_readme(out_csv)


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "classified_id" in out.columns:
        out["classified_id"] = out["classified_id"].astype(str).str.strip()
    for c in TEMPLATE_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan if c != "sat_missing_reason" else ""
    return out[TEMPLATE_COLUMNS]


def _is_template_only(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return True
    if "classified_id" not in df.columns:
        return True
    if len(df) <= 2 and "sat_missing_reason" in df.columns:
        reasons = df["sat_missing_reason"].astype(str).str.lower()
        if reasons.str.contains("template").all():
            return True
    if "sat_has_features" in df.columns:
        has = pd.to_numeric(df["sat_has_features"], errors="coerce").fillna(0)
        if len(df) <= 2 and float(has.max()) <= 0:
            return True
    return False


def _load_existing_feature_csv(path: Path) -> pd.DataFrame:
    """Load prior smoke/full CSV for resume. Template-only files return empty."""
    if not path.exists() or path.stat().st_size < 50:
        return pd.DataFrame(columns=TEMPLATE_COLUMNS)
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception as exc:
        warnings.warn(f"Could not read existing CSV {path}: {exc}")
        return pd.DataFrame(columns=TEMPLATE_COLUMNS)
    if _is_template_only(df):
        print(f"Existing CSV looks like template only; starting fresh: {path}")
        return pd.DataFrame(columns=TEMPLATE_COLUMNS)
    df = _normalize_feature_frame(df)
    df = df[df["classified_id"].notna() & (df["classified_id"].astype(str).str.strip() != "")]
    df = df[df["classified_id"].astype(str).str.lower() != "none"]
    df = df.drop_duplicates("classified_id", keep="last")
    print(f"Resume base loaded: {len(df)} existing classified_id rows from {path}")
    return df


def _load_failed_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size < 20:
        return pd.DataFrame(columns=TEMPLATE_COLUMNS + ["error"])
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception:
        return pd.DataFrame(columns=TEMPLATE_COLUMNS + ["error"])
    if "classified_id" in df.columns:
        df["classified_id"] = df["classified_id"].astype(str).str.strip()
        df = df.drop_duplicates("classified_id", keep="last")
    return df


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    clean = _normalize_feature_frame(df).drop_duplicates("classified_id", keep="last")
    clean.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def _coverage(df: pd.DataFrame) -> float:
    if df is None or df.empty or "sat_has_features" not in df.columns:
        return 0.0
    return float(pd.to_numeric(df["sat_has_features"], errors="coerce").fillna(0).gt(0).mean())


def _load_listing_coords(
    city: str,
    county: str,
    limit: int | None,
    sale_table: str | None = None,
    source_site: str | None = None,
) -> pd.DataFrame:
    """Load sale listing coordinates from DB (root .env). Matches V21 table defaults."""
    try:
        from shared_scripts.env_loader import load_root_env
        load_root_env()
    except Exception:
        try:
            from dotenv import load_dotenv
            env = ROOT / ".env"
            if env.exists():
                load_dotenv(env)
        except Exception:
            pass

    db_url = os.getenv("DATABASE_URL") or os.getenv("DB_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set; cannot fetch listing coordinates.")

    from sqlalchemy import create_engine, text

    engine = create_engine(db_url)
    default_table = sale_table or os.getenv("SALE_TABLE", "sahibinden_sale_listings")
    source = source_site or os.getenv("SOURCE_SITE", "sahibinden")
    candidates = []
    for t in [default_table, "sahibinden_sale_listings", "sale_listings"]:
        if t and t not in candidates:
            candidates.append(t)

    limit_clause = f" LIMIT {int(limit)}" if limit else ""
    params: dict[str, Any] = {
        "city": city,
        "county": county,
        "purpose": "sale",
        "source_site": source,
    }
    errors: list[str] = []
    df = pd.DataFrame()

    for table in candidates:
        if not str(table).replace("_", "").isalnum():
            errors.append(f"{table}: invalid table name")
            continue
        sql = text(
            f"""
            SELECT
                classified_id,
                latitude,
                longitude,
                lat,
                lon,
                location_precision,
                location_source,
                county,
                district,
                city,
                listing_purpose
            FROM {table}
            WHERE lower(coalesce(city, '')) = lower(:city)
              AND lower(coalesce(listing_purpose, '')) = lower(:purpose)
              AND county = :county
              AND lower(coalesce(source_site, :source_site)) = lower(:source_site)
            ORDER BY saved_at DESC NULLS LAST, updated_at DESC NULLS LAST
            {limit_clause}
            """
        )
        try:
            with engine.connect() as conn:
                df = pd.read_sql(sql, conn, params=params)
            print(f"Loaded listing coords from table={table} rows={len(df)}")
            break
        except Exception:
            sql_min = text(
                f"""
                SELECT *
                FROM {table}
                WHERE lower(coalesce(city, '')) = lower(:city)
                  AND lower(coalesce(listing_purpose, '')) = lower(:purpose)
                  AND county = :county
                  AND lower(coalesce(source_site, :source_site)) = lower(:source_site)
                ORDER BY saved_at DESC NULLS LAST, updated_at DESC NULLS LAST
                {limit_clause}
                """
            )
            try:
                with engine.connect() as conn:
                    df = pd.read_sql(sql_min, conn, params=params)
                print(f"Loaded listing coords from table={table} (SELECT *) rows={len(df)}")
                break
            except Exception as exc2:
                errors.append(f"{table}: {exc2}")
                df = pd.DataFrame()

    if df.empty and errors:
        raise RuntimeError(
            "Could not load listing coordinates. Tried tables: "
            + ", ".join(candidates)
            + ". Errors: "
            + " | ".join(errors)
        )
    if df.empty:
        raise RuntimeError(f"No sale rows for city={city!r} county={county!r} in {candidates}")

    if "classified_id" not in df.columns:
        raise RuntimeError("Sale table missing classified_id")

    df = df.copy()
    df["classified_id"] = df["classified_id"].astype(str).str.strip()
    if "latitude" not in df.columns and "lat" in df.columns:
        df["latitude"] = df["lat"]
    if "longitude" not in df.columns and "lon" in df.columns:
        df["longitude"] = df["lon"]
    df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    keep = [
        c
        for c in [
            "classified_id",
            "latitude",
            "longitude",
            "location_precision",
            "location_source",
            "county",
            "district",
        ]
        if c in df.columns
    ]
    return df[keep].drop_duplicates("classified_id", keep="last")


def _usable_coords(df: pd.DataFrame) -> pd.Series:
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    return lat.between(40.0, 42.0) & lon.between(29.0, 31.0)


def _init_gee(project: str | None = None) -> Any:
    import ee  # type: ignore

    proj = (
        (project or "").strip()
        or os.getenv("EE_PROJECT", "").strip()
        or os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
        or os.getenv("GCLOUD_PROJECT", "").strip()
    )
    if not proj:
        raise RuntimeError(
            "GEE auth OK but no Cloud project set. Newer earthengine-api requires a project.\n"
            "1) Create/select a GCP project and register it for Earth Engine:\n"
            "   https://code.earthengine.google.com/register\n"
            "2) Then either:\n"
            "   set EE_PROJECT=your-project-id in root .env\n"
            "   or pass --gee-project your-project-id\n"
            "3) Test: python -c \"import ee; ee.Initialize(project='your-project-id'); print('gee ready')\""
        )

    try:
        ee.Initialize(project=proj)
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize(project=proj)
        except Exception as exc:
            raise RuntimeError(
                "Google Earth Engine init failed for project="
                f"{proj!r}. Register the project for EE and retry. Detail: {exc}"
            ) from exc
    print(f"GEE initialized with project={proj}")
    return ee


def _gee_point_features(ee: Any, lat: float, lon: float) -> dict[str, float | int | None]:
    """Median composite Sentinel-2 SR features at 100/250/500m buffers."""
    point = ee.Geometry.Point([float(lon), float(lat)])
    end = ee.Date(datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    start = end.advance(-12, "month")

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
    )

    def _mask(img: Any) -> Any:
        qa = img.select("QA60")
        mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return img.updateMask(mask).divide(10000.0)

    composite = col.map(_mask).median()
    ndvi = composite.normalizedDifference(["B8", "B4"]).rename("ndvi")
    ndwi = composite.normalizedDifference(["B3", "B8"]).rename("ndwi")
    ndbi = composite.normalizedDifference(["B11", "B8"]).rename("ndbi")
    brightness = composite.select(["B2", "B3", "B4"]).reduce(ee.Reducer.mean()).rename("brightness")
    stack = ndvi.addBands([ndwi, ndbi, brightness])

    out: dict[str, float | int | None] = {
        "sat_has_features": 0,
        "sat_cloud_coverage_proxy": None,
        "sat_feature_year": int(datetime.now(timezone.utc).year),
        "sat_feature_month": int(datetime.now(timezone.utc).month),
        "sat_missing_reason": "",
    }

    try:
        cloud_mean = col.aggregate_mean("CLOUDY_PIXEL_PERCENTAGE").getInfo()
        out["sat_cloud_coverage_proxy"] = float(cloud_mean) if cloud_mean is not None else None
    except Exception:
        out["sat_cloud_coverage_proxy"] = None

    for radius, tag in ((100, "100m"), (250, "250m"), (500, "500m")):
        buf = point.buffer(radius)
        try:
            stats = stack.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                geometry=buf,
                scale=10,
                maxPixels=1_000_000,
                bestEffort=True,
            ).getInfo() or {}
        except Exception:
            stats = {}

        def _g(key: str) -> float | None:
            v = stats.get(key)
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        out[f"sat_ndvi_mean_{tag}"] = _g("ndvi_mean")
        out[f"sat_ndwi_mean_{tag}"] = _g("ndwi_mean")
        out[f"sat_ndbi_mean_{tag}"] = _g("ndbi_mean")
        out[f"sat_brightness_mean_{tag}"] = _g("brightness_mean")
        if tag == "250m":
            out["sat_ndvi_std_250m"] = _g("ndvi_stdDev")
            out["sat_texture_proxy_250m"] = _g("ndvi_stdDev")
            ndvi_m = _g("ndvi_mean")
            ndwi_m = _g("ndwi_mean")
            ndbi_m = _g("ndbi_mean")
            out["sat_green_share_250m"] = float(ndvi_m > 0.35) if ndvi_m is not None else None
            out["sat_water_share_250m"] = float(ndwi_m > 0.20) if ndwi_m is not None else None
            out["sat_builtup_share_250m"] = float(ndbi_m > 0.05) if ndbi_m is not None else None

    if out.get("sat_ndvi_mean_250m") is not None or out.get("sat_ndbi_mean_250m") is not None:
        out["sat_has_features"] = 1
        out["sat_missing_reason"] = ""
    else:
        out["sat_has_features"] = 0
        out["sat_missing_reason"] = "gee_empty_or_cloudy"
    return out


def _progress_line(
    *,
    new_done: int,
    to_fetch: int,
    total: int,
    skipped_existing: int,
    failed: int,
    coverage_so_far: float,
    rows_saved: int,
) -> str:
    # done_all = already-on-disk skips + newly handled in this run
    done_all = int(skipped_existing + new_done)
    pending_new = max(0, int(to_fetch - new_done))
    return (
        f"GEE progress: new={new_done}/{to_fetch} "
        f"done_all={done_all}/{total} "
        f"skipped_existing={skipped_existing} failed={failed} "
        f"pending_new={pending_new} rows_saved={rows_saved} "
        f"coverage_so_far={coverage_so_far:.3f}"
    )


def fetch_via_gee(
    listings: pd.DataFrame,
    *,
    out_csv: Path,
    failed_csv: Path,
    max_points: int | None = None,
    project: str | None = None,
    resume: bool = True,
    save_every: int = 25,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resume-safe per-point GEE extraction with incremental CSV saves."""
    existing = _load_existing_feature_csv(out_csv) if resume else pd.DataFrame(columns=TEMPLATE_COLUMNS)
    existing_ids = set(existing["classified_id"].astype(str)) if not existing.empty else set()
    failed_prev = _load_failed_csv(failed_csv)
    failed_rows: list[dict[str, Any]] = failed_prev.to_dict(orient="records") if not failed_prev.empty else []

    usable = _usable_coords(listings)
    work = listings.copy()
    if max_points is not None and max_points > 0:
        work = pd.concat([work.loc[usable], work.loc[~usable]], axis=0).head(int(max_points))
    work = work.drop_duplicates("classified_id", keep="last")
    total = int(len(work))

    todo = work[~work["classified_id"].astype(str).isin(existing_ids)].copy() if resume else work.copy()
    skipped_existing = int(total - len(todo)) if resume else 0

    results = existing.copy() if not existing.empty else pd.DataFrame(columns=TEMPLATE_COLUMNS)
    processed = 0
    failed = 0
    since_save = 0
    interrupted = False
    status = "gee_ok"

    print(
        f"GEE plan: total={total} skipped_existing={skipped_existing} "
        f"to_fetch={len(todo)} resume={resume} save_every={save_every}",
        flush=True,
    )

    if todo.empty:
        _atomic_write_csv(results, out_csv)
        stats = {
            "status": "gee_resume_complete_nothing_todo",
            "total": total,
            "processed": 0,
            "skipped_existing": skipped_existing,
            "failed": failed,
            "rows": int(len(results)),
            "sat_feature_coverage": _coverage(results),
            "interrupted": False,
        }
        print(_progress_line(
            new_done=0,
            to_fetch=0,
            total=total,
            skipped_existing=skipped_existing,
            failed=failed,
            coverage_so_far=_coverage(results),
            rows_saved=int(len(results)),
        ), flush=True)
        return results, stats

    ee = _init_gee(project=project)
    to_fetch_n = int(len(todo))
    print(
        f"GEE extracting {to_fetch_n} new points "
        f"(already have {skipped_existing}; grand total listings={total}; "
        f"per-point API; first point can take ~30-90s)...",
        flush=True,
    )

    def _persist(reason: str) -> None:
        nonlocal results
        results = _normalize_feature_frame(results).drop_duplicates("classified_id", keep="last")
        _atomic_write_csv(results, out_csv)
        if failed_rows:
            fail_df = pd.DataFrame(failed_rows)
            if "classified_id" in fail_df.columns:
                fail_df["classified_id"] = fail_df["classified_id"].astype(str).str.strip()
                fail_df = fail_df.drop_duplicates("classified_id", keep="last")
            failed_csv.parent.mkdir(parents=True, exist_ok=True)
            fail_df.to_csv(failed_csv, index=False, encoding="utf-8-sig")
        print(
            f"Saved checkpoint ({reason}): rows={len(results)} failed={len(failed_rows)} -> {out_csv}",
            flush=True,
        )

    try:
        for i, (_, r) in enumerate(todo.iterrows(), start=1):
            base = {
                c: r.get(c)
                for c in [
                    "classified_id",
                    "latitude",
                    "longitude",
                    "location_precision",
                    "location_source",
                    "county",
                    "district",
                ]
            }
            cid = str(base.get("classified_id") or "").strip()
            lat = pd.to_numeric(r.get("latitude"), errors="coerce")
            lon = pd.to_numeric(r.get("longitude"), errors="coerce")

            row: dict[str, Any]
            is_fail = False
            if not (np.isfinite(lat) and np.isfinite(lon) and 40.0 <= lat <= 42.0 and 29.0 <= lon <= 31.0):
                feat = {k: None for k in ALL_SATELLITE_NUMERIC_FEATURES}
                feat["sat_has_features"] = 0
                feat["sat_missing_reason"] = "coords_unusable"
                row = {**base, **feat}
            else:
                try:
                    feat = _gee_point_features(ee, float(lat), float(lon))
                    row = {**base, **feat}
                    if int(feat.get("sat_has_features") or 0) == 0 and str(feat.get("sat_missing_reason") or "").startswith("gee_"):
                        is_fail = True
                except Exception as exc:
                    feat = {k: None for k in ALL_SATELLITE_NUMERIC_FEATURES}
                    feat["sat_has_features"] = 0
                    feat["sat_missing_reason"] = f"gee_error:{type(exc).__name__}"
                    row = {**base, **feat, "error": str(exc)[:500]}
                    is_fail = True

            processed += 1
            since_save += 1

            if is_fail:
                failed += 1
                failed_rows.append(row)
            else:
                # upsert into results
                if not results.empty and cid in set(results["classified_id"].astype(str)):
                    results = results[results["classified_id"].astype(str) != cid]
                results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)

            pending = int(len(todo) - i)
            cov = _coverage(results)
            if i == 1 or i % 5 == 0 or i == len(todo) or since_save >= save_every:
                print(
                    _progress_line(
                        new_done=processed,
                        to_fetch=to_fetch_n,
                        total=total,
                        skipped_existing=skipped_existing,
                        failed=failed,
                        coverage_so_far=cov,
                        rows_saved=int(len(results)),
                    ),
                    flush=True,
                )

            if since_save >= max(1, int(save_every)):
                _persist(f"every_{save_every}")
                since_save = 0

    except KeyboardInterrupt:
        interrupted = True
        status = "gee_interrupted_saved"
        print("\nKeyboardInterrupt — saving progress...", flush=True)
        _persist("keyboard_interrupt")
    except Exception as exc:
        interrupted = True
        status = "gee_error_saved"
        print(f"\nGEE loop error — saving progress: {exc}", flush=True)
        _persist("error")
        raise
    else:
        _persist("final")

    stats = {
        "status": status,
        "total": total,
        "processed": processed,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "rows": int(len(results)),
        "sat_feature_coverage": _coverage(results),
        "interrupted": interrupted,
        "to_fetch": int(len(todo)),
        "save_every": int(save_every),
        "resume": bool(resume),
    }
    return _normalize_feature_frame(results).drop_duplicates("classified_id", keep="last"), stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch free Sentinel environment features for V22 (GEE optional, resume-safe).")
    ap.add_argument("--city", default="Kocaeli")
    ap.add_argument("--county", default="Başiskele")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--metadata-out", default=str(DEFAULT_META))
    ap.add_argument("--failed-out", default=str(DEFAULT_FAILED), help="Failed point CSV path.")
    ap.add_argument("--source", choices=["gee", "cached_csv"], default=None, help="Default: cached_csv if exists else instructions")
    ap.add_argument("--limit", type=int, default=None, help="Optional listing limit (smoke).")
    ap.add_argument("--max-gee-points", type=int, default=None, help="Cap GEE point extractions.")
    ap.add_argument(
        "--sale-table",
        default=None,
        help="Sale listings table (default: SALE_TABLE env or sahibinden_sale_listings).",
    )
    ap.add_argument(
        "--source-site",
        default=None,
        help="source_site filter (default: SOURCE_SITE env or sahibinden).",
    )
    ap.add_argument(
        "--gee-project",
        default=None,
        help="Google Cloud / Earth Engine project id (or set EE_PROJECT in .env).",
    )
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip classified_ids already present in --out CSV (default: true).",
    )
    ap.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Incremental checkpoint write every N newly processed points (default: 25).",
    )
    args = ap.parse_args()

    out_csv = Path(args.out)
    meta_out = Path(args.metadata_out)
    failed_csv = Path(args.failed_out)
    source = args.source
    if source is None:
        source = "gee"

    meta: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "city": args.city,
        "county": args.county,
        "source_requested": args.source,
        "source_used": source,
        "out": str(out_csv),
        "failed_out": str(failed_csv),
        "resume": bool(args.resume),
        "save_every": int(args.save_every),
        "status": "unknown",
    }

    if source == "cached_csv":
        if out_csv.exists() and out_csv.stat().st_size > 50 and not _is_template_only(pd.read_csv(out_csv, encoding="utf-8-sig", nrows=5)):
            df = pd.read_csv(out_csv, encoding="utf-8-sig")
            print(f"Using cached CSV: {out_csv} ({len(df)} rows)")
            meta["status"] = "cached_ok"
            meta["rows"] = int(len(df))
            _write_metadata(meta_out, meta)
            return 0
        print("Cached CSV missing/template — writing template + README_FETCH_REQUIRED.md")
        _write_template_csv(out_csv)
        meta["status"] = "template_written_fetch_required"
        _write_metadata(meta_out, meta)
        print(f"Wrote template: {out_csv}")
        print(f"See: {README_FETCH}")
        return 0

    # GEE mode
    try:
        listings = _load_listing_coords(
            args.city,
            args.county,
            args.limit,
            sale_table=args.sale_table,
            source_site=args.source_site,
        )
        print(f"Listings loaded: {len(listings)}")
    except Exception as exc:
        warnings.warn(f"Could not load listing coords ({exc}). Writing template instead.")
        _write_template_csv(out_csv)
        meta["status"] = "listing_load_failed_template"
        meta["error"] = str(exc)
        _write_metadata(meta_out, meta)
        print("GEE fetch aborted gracefully. Training can still run control-only.")
        return 0

    try:
        feat, stats = fetch_via_gee(
            listings,
            out_csv=out_csv,
            failed_csv=failed_csv,
            max_points=args.max_gee_points,
            project=args.gee_project,
            resume=bool(args.resume),
            save_every=max(1, int(args.save_every)),
        )
        meta.update(stats)
        meta["status"] = stats.get("status", "gee_ok")
        _write_metadata(meta_out, meta)
        print(
            f"Done: rows={len(feat)} coverage={_coverage(feat):.3f} "
            f"processed={stats.get('processed')} skipped_existing={stats.get('skipped_existing')} "
            f"failed={stats.get('failed')} interrupted={stats.get('interrupted')}"
        )
        print(f"Wrote {out_csv}")
        if failed_csv.exists():
            print(f"Failed log: {failed_csv}")
        return 0 if not stats.get("interrupted") else 130
    except KeyboardInterrupt:
        meta["status"] = "gee_interrupted"
        _write_metadata(meta_out, meta)
        print("Interrupted. Partial CSV kept if checkpoints were written.")
        return 130
    except Exception as exc:
        # Do NOT overwrite a good resume CSV with a template
        existing = _load_existing_feature_csv(out_csv)
        if existing.empty:
            warnings.warn(f"GEE fetch failed ({exc}). Writing template; train will run control-only.")
            _write_template_csv(out_csv)
            meta["status"] = "gee_failed_template"
        else:
            warnings.warn(f"GEE fetch failed ({exc}). Keeping existing CSV with {len(existing)} rows.")
            meta["status"] = "gee_failed_kept_existing"
            _atomic_write_csv(existing, out_csv)
        meta["error"] = str(exc)
        _write_metadata(meta_out, meta)
        print(str(exc))
        print(f"See: {README_FETCH}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
