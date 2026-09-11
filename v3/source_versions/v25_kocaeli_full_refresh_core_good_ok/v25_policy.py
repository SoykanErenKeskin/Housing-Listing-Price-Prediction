"""V25 Kocaeli full-refresh core_good_ok policy + feature guards.

No training logic — helpers consumed by train_v25_kocaeli_full_pipeline.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODEL_VERSION = "v25-kocaeli-full-refresh-core-good-ok"
MODEL_SCOPE = "kocaeli_global"
COUNTY_POLICY = "core_good_ok"

INCLUDED_COUNTIES = [
    "Darıca",
    "İzmit",
    "Gebze",
    "Başiskele",
    "Çayırova",
    "Gölcük",
    "Kartepe",
    "Körfez",
    "Derince",
]

EXCLUDED_COUNTIES = [
    "Karamürsel",
    "Kandıra",
    "Dilovası",
]

EXCLUSION_REASON = (
    "WEAK inventory readiness on full-refresh audit: low sale/rental volume "
    "and/or sparse districts. Deferred from first V25 primary run."
)

# Schema-level columns / name prefixes that must survive into V25.
CRITICAL_SCHEMA_COLUMNS = [
    "heating",
    "kitchen",
    "balcony",
    "usage_status",
    "bathroom_count",
    "floor_segment",
    "county",
    "district",
]

CRITICAL_ENCODED_PREFIXES = [
    "cat__heating_",
    "num__attr_heating_quality_score",
    "num__attr_has_premium_heating",
    "cat__duplex_type_",
    "num__is_duplex",
    "num__is_large_home",
    "cat__site_project_id",
    "num__has_site_project_id",
    "num__site_project_oof_",
    "num__lat",
    "num__lon",
    "cat__kitchen_",
    "cat__balcony_",
    "cat__usage_status_",
    "num__bathroom_count",
    "cat__floor_segment_",
    "cat__county_",
    "cat__district_",
]

CRITICAL_ENCODED_CONTAINS = [
    "heating_Yerden Isıtma",
    "attr_heating_quality",
    "duplex_type",
    "is_large_home",
    "site_project_id",
    "site_project_oof",
    "geo_cluster",
    "coast_distance",
]

V24_1_REFERENCE = {
    "experiment": "full_v24",
    "path": "v3/outputs/v24_1_kocaeli_site_merge_repair/",
    "r2": 0.6523243261980677,
    "mape": 0.11762822096027113,
    "variance_ratio": 0.6315725457911608,
    "rows": 6667,
    "note": "Old-data reference only; do not select solely against this after distribution change.",
}


def normalize_source_site(raw: str | None) -> str:
    s = (raw or "").strip()
    if not s:
        return "sahibinden"
    if s.lower() in {"sahibinden.com", "sahibinden"}:
        return "sahibinden"
    if s.lower() in {"listing_portal", "portal"}:
        # Current Neon inventory uses sahibinden; remap legacy default.
        return "sahibinden"
    return s


def resolve_counties_for_policy(policy: str, counties_arg: str | None = None) -> dict[str, Any]:
    pol = (policy or COUNTY_POLICY).strip().lower()
    if pol in {"core_good_ok", "good_ok", "v25_core"}:
        included = list(INCLUDED_COUNTIES)
        excluded = list(EXCLUDED_COUNTIES)
    elif pol in {"all", "all_kocaeli"}:
        included = list(INCLUDED_COUNTIES) + list(EXCLUDED_COUNTIES)
        excluded = []
    elif pol == "explicit" and counties_arg:
        included = [c.strip() for c in counties_arg.split(",") if c.strip()]
        excluded = [c for c in EXCLUDED_COUNTIES if c not in included]
    else:
        raise ValueError(f"Unsupported county_policy={policy!r}")
    return {
        "county_policy": pol if pol != "v25_core" else COUNTY_POLICY,
        "included_counties": included,
        "excluded_counties": excluded,
        "exclusion_reason": EXCLUSION_REASON if excluded else "",
        "model_version": MODEL_VERSION,
        "model_scope": MODEL_SCOPE,
    }


def limit_per_county(df: pd.DataFrame, limit: int | None, *, seed: int = 42) -> pd.DataFrame:
    if limit is None or limit <= 0 or df.empty or "county" not in df.columns:
        return df
    parts = []
    for county, g in df.groupby(df["county"].astype(str), dropna=False):
        if len(g) <= limit:
            parts.append(g)
        else:
            parts.append(g.sample(n=limit, random_state=seed))
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


def county_counts(df: pd.DataFrame, purpose: str) -> list[dict[str, Any]]:
    if df.empty or "county" not in df.columns:
        return []
    vc = df["county"].astype(str).value_counts()
    return [{"purpose": purpose, "county": k, "rows": int(v)} for k, v in vc.items()]


def write_county_policy_report(
    path: Path,
    *,
    policy_info: dict[str, Any],
    sales_before: pd.DataFrame,
    rentals_before: pd.DataFrame,
    sales_after: pd.DataFrame,
    rentals_after: pd.DataFrame,
) -> dict[str, Any]:
    payload = {
        **policy_info,
        "rows_before": {
            "sale": int(len(sales_before)),
            "rental": int(len(rentals_before)),
            "total": int(len(sales_before) + len(rentals_before)),
        },
        "rows_after": {
            "sale": int(len(sales_after)),
            "rental": int(len(rentals_after)),
            "total": int(len(sales_after) + len(rentals_after)),
        },
        "sale_by_county_after": county_counts(sales_after, "sale"),
        "rental_by_county_after": county_counts(rentals_after, "rental"),
        "silent_drop_guard": {
            "unexpected_counties_in_after": sorted(
                set(sales_after["county"].astype(str)).union(set(rentals_after["county"].astype(str)))
                - set(policy_info["included_counties"])
            )
            if len(sales_after) or len(rentals_after)
            else [],
            "missing_included_counties_in_sale": [
                c
                for c in policy_info["included_counties"]
                if c not in set(sales_after["county"].astype(str) if len(sales_after) else [])
            ],
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def detailed_filter_reasons(df: pd.DataFrame, purpose: str, cfg: Any) -> dict[str, int]:
    """Aggregate removal reasons without emitting listing IDs."""
    if df.empty:
        return {}
    out = df.copy()
    reasons: dict[str, int] = {}
    gross = pd.to_numeric(out.get("gross_m2"), errors="coerce")
    reasons["gross_m2_le_20_or_missing"] = int((gross.isna() | (gross <= 20)).sum())
    reasons["gross_m2_ge_600"] = int((gross >= 600).sum())
    if purpose == "sale":
        price = pd.to_numeric(out.get("price"), errors="coerce")
        unit = pd.to_numeric(out.get("unit_price_gross"), errors="coerce")
        reasons["price_le_0"] = int((price.notna() & (price <= 0)).sum())
        reasons["unit_price_outside_8k_200k"] = int(
            (unit.isna() | ~unit.between(getattr(cfg, "min_sale_unit_price", 8000), getattr(cfg, "max_sale_unit_price", 200000))).sum()
        )
        reasons["missing_target"] = int(unit.isna().sum())
    else:
        rent = pd.to_numeric(out.get("monthly_rent"), errors="coerce")
        rpm = pd.to_numeric(out.get("rent_per_m2_gross"), errors="coerce")
        reasons["rent_le_0"] = int((rent.notna() & (rent <= 0)).sum())
        reasons["rent_m2_outside_50_2500"] = int(
            (rpm.isna() | ~rpm.between(getattr(cfg, "min_rent_m2", 50), getattr(cfg, "max_rent_m2", 2500))).sum()
        )
        reasons["missing_target"] = int(rpm.isna().sum())
    reasons["missing_gross_m2"] = int(gross.isna().sum())
    return reasons


def heating_yerden_audit(sales: pd.DataFrame, rentals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for purpose, df in (("sale", sales), ("rental", rentals)):
        if df.empty or "heating" not in df.columns:
            continue
        heat = df["heating"].astype("string")
        yerden = heat.fillna("").str.contains(r"yerden", case=False, regex=True)
        counties = sorted(set(df["county"].dropna().astype(str))) if "county" in df.columns else ["__all__"]
        for county in counties:
            sub = df[df["county"].astype(str) == county] if county != "__all__" else df
            ymask = sub["heating"].astype("string").fillna("").str.contains(r"yerden", case=False, regex=True)
            rows.append(
                {
                    "purpose": purpose,
                    "county": county,
                    "total_rows": int(len(sub)),
                    "yerden_count": int(ymask.sum()),
                    "yerden_share": float(ymask.mean()) if len(sub) else np.nan,
                    "median_unit_price_yerden": float(
                        pd.to_numeric(sub.loc[ymask, "unit_price_gross"], errors="coerce").median()
                    )
                    if purpose == "sale" and ymask.any() and "unit_price_gross" in sub.columns
                    else np.nan,
                    "median_unit_price_other": float(
                        pd.to_numeric(sub.loc[~ymask, "unit_price_gross"], errors="coerce").median()
                    )
                    if purpose == "sale" and (~ymask).any() and "unit_price_gross" in sub.columns
                    else np.nan,
                }
            )
        rows.append(
            {
                "purpose": purpose,
                "county": "__ALL__",
                "total_rows": int(len(df)),
                "yerden_count": int(yerden.sum()),
                "yerden_share": float(yerden.mean()) if len(df) else np.nan,
                "median_unit_price_yerden": float(
                    pd.to_numeric(df.loc[yerden, "unit_price_gross"], errors="coerce").median()
                )
                if purpose == "sale" and yerden.any() and "unit_price_gross" in df.columns
                else np.nan,
                "median_unit_price_other": float(
                    pd.to_numeric(df.loc[~yerden, "unit_price_gross"], errors="coerce").median()
                )
                if purpose == "sale" and (~yerden).any() and "unit_price_gross" in df.columns
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def collect_encoded_feature_names(models: dict[str, Any] | None) -> list[str]:
    names: list[str] = []
    if not models:
        return names
    for pipe in models.values():
        try:
            pre = pipe.named_steps.get("preprocess")
            if pre is not None and hasattr(pre, "get_feature_names_out"):
                names = list(map(str, pre.get_feature_names_out()))
                break
        except Exception:
            continue
    return names


def _has_any(names: list[str], *needles: str) -> bool:
    low = [n.lower() for n in names]
    for needle in needles:
        nl = needle.lower()
        if any(nl in n for n in low):
            return True
    return False


def build_feature_schema_diff(
    *,
    v25_schema_cols: list[str],
    v25_encoded_names: list[str],
    v24_importance_csv: Path | None,
    heating_raw_yerden_count: int,
    fast_mode: bool = False,
    duplex_feature_mode: str = "full",
    site_project_encoding: str = "foldsafe_target",
) -> dict[str, Any]:
    v24_names: set[str] = set()
    if v24_importance_csv and v24_importance_csv.exists():
        try:
            df = pd.read_csv(v24_importance_csv)
            col = "feature" if "feature" in df.columns else df.columns[0]
            v24_names = set(df[col].astype(str))
        except Exception:
            v24_names = set()

    v25_set = set(v25_encoded_names)
    missing = sorted(v24_names - v25_set) if v24_names else []
    added = sorted(v25_set - v24_names) if v24_names else sorted(v25_set)

    duplex_mode = str(duplex_feature_mode or "full").lower()
    site_enc = str(site_project_encoding or "foldsafe_target").lower()

    critical_missing: list[str] = []
    checks = {
        "heating_categorical": _has_any(v25_encoded_names, "cat__heating_"),
        "heating_yerden_leaf": _has_any(v25_encoded_names, "heating_Yerden Isıtma", "heating_yerden"),
        "premium_heating_score": _has_any(v25_encoded_names, "attr_heating_quality_score", "attr_has_premium_heating"),
        "duplex_flags": _has_any(v25_encoded_names, "is_duplex", "duplex_type")
        if duplex_mode not in {"", "none"}
        else True,
        "large_home": (
            _has_any(v25_encoded_names, "is_large_home", "large_home_bucket")
            if duplex_mode in {"largehome", "interactions", "full"}
            else True
        ),
        "site_project_id": _has_any(v25_encoded_names, "site_project_id", "has_site_project_id")
        if site_enc not in {"", "none"}
        else True,
        "foldsafe_site_oof": (
            _has_any(v25_encoded_names, "site_project_oof")
            if site_enc == "foldsafe_target"
            else True
        ),
        "lat_lon_or_geo": _has_any(v25_encoded_names, "num__lat", "num__lon", "geo_cluster", "coast_distance"),
        "kitchen": _has_any(v25_encoded_names, "cat__kitchen_"),
        "balcony": _has_any(v25_encoded_names, "cat__balcony_"),
        "usage_status": _has_any(v25_encoded_names, "cat__usage_status_"),
        "bathroom_count": _has_any(v25_encoded_names, "bathroom_count"),
        "floor_segment": _has_any(v25_encoded_names, "floor_segment"),
        "county_district": _has_any(v25_encoded_names, "cat__county_", "cat__district_"),
    }

    for col in CRITICAL_SCHEMA_COLUMNS:
        if col not in set(v25_schema_cols):
            # If encoded feature already proves the column was used, don't hard-fail schema absence
            # (cleaned exports sometimes drop/rename intermediate frames).
            mapped = {
                "heating": "heating_categorical",
                "kitchen": "kitchen",
                "balcony": "balcony",
                "usage_status": "usage_status",
                "bathroom_count": "bathroom_count",
                "floor_segment": "floor_segment",
                "county": "county_district",
                "district": "county_district",
            }.get(col)
            if mapped and checks.get(mapped):
                continue
            critical_missing.append(f"schema_missing:{col}")

    for key, ok in checks.items():
        if not ok:
            if key == "foldsafe_site_oof" and (fast_mode or site_enc != "foldsafe_target"):
                continue
            if key == "large_home" and duplex_mode not in {"largehome", "interactions", "full"}:
                continue
            if key == "heating_yerden_leaf" and heating_raw_yerden_count <= 0:
                continue
            if key == "heating_yerden_leaf" and heating_raw_yerden_count > 0 and not ok:
                critical_missing.append(key)
            elif key != "heating_yerden_leaf" and not ok:
                critical_missing.append(key)

    if heating_raw_yerden_count > 0 and not checks["heating_yerden_leaf"]:
        if "heating_yerden_leaf" not in critical_missing:
            critical_missing.append("heating_yerden_leaf")

    status = "FAIL" if critical_missing else "PASS"
    if fast_mode and critical_missing and all(
        x in {"foldsafe_site_oof"} or x.startswith("schema_missing:") for x in critical_missing
    ):
        status = "PASS_WITH_FAST_MODE_WARNINGS"

    return {
        "model_version": MODEL_VERSION,
        "model_scope": MODEL_SCOPE,
        "county_policy": COUNTY_POLICY,
        "duplex_feature_mode": duplex_mode,
        "site_project_encoding": site_enc,
        "v24_1_feature_count": len(v24_names),
        "v25_feature_count": len(v25_set),
        "v25_schema_column_count": len(v25_schema_cols),
        "missing_from_v25_count": len(missing),
        "added_in_v25_count": len(added),
        "missing_from_v25_sample": missing[:80],
        "added_in_v25_sample": added[:80],
        "critical_checks": checks,
        "critical_missing_features": critical_missing,
        "heating_raw_yerden_count": int(heating_raw_yerden_count),
        "status": status,
        "fast_mode": bool(fast_mode),
        "v24_1_reference": V24_1_REFERENCE,
    }


def assert_no_stale_basiskele_metadata(payload: dict[str, Any]) -> list[str]:
    """Return warnings if payload still looks Başiskele-only scoped."""
    warns = []
    scope = str(payload.get("model_scope") or payload.get("scope") or "")
    if scope.lower() in {"basiskele_only", "basiskele"}:
        warns.append("stale_basiskele_only_scope")
    if str(payload.get("county") or "").lower() == "başiskele" and payload.get("model_scope") != MODEL_SCOPE:
        warns.append("top_level_county_basiskele")
    mv = str(payload.get("model_version") or "")
    if mv and "v25" not in mv.lower() and "v24" not in mv.lower():
        if "v18" in mv.lower() or "basiskele" in mv.lower():
            warns.append(f"stale_model_version={mv}")
    return warns
