"""V24 Kocaeli county-aware site/project extraction.

Reuses V21/V23 extraction/dictionary logic, then namespaces every canonical id as:

    {county_slug}::{site_project_slug}

so the same display name in İzmit vs Başiskele never collapses.

Başiskele curated dictionary is preserved (local slug match) then scoped.
Unknown counties use conservative frequency-based stems only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from site_project_extraction import (  # noqa: F401 — re-export for train imports
    BASISKELE_SITE_DICTIONARY,
    FOLDSAFE_NUMERIC,
    INTERACTION_CATEGORICAL,
    INTERACTION_NUMERIC,
    SITE_CATEGORICAL_FEATURES,
    SITE_NAME_STOP,
    SITE_NUMERIC_FEATURES,
    SITE_UNIT_PRICE_COL,
    SiteDictEntry,
    SiteProjectExtractionAdder as _BaseSiteAdder,
    SiteProjectFoldSafeEncoder as _BaseFoldSafe,
    alias_map_table,
    build_audit_text,
    count_severe_bad_merges,
    coverage_table,
    dictionary_table,
    enrich_review_with_prices,
    extract_site_project_raw,
    fold_text,
    freq_bucket,
    match_dictionary,
    normalize_site_stem,
    stem_to_id,
    write_leakage_guard,
)

# Extra stop tokens for other Kocaeli counties (do not become site stems).
SITE_NAME_STOP_KOCAELI = set(SITE_NAME_STOP) | {
    "golcuk",
    "gölcük",
    "karamursel",
    "karamürsel",
    "kartepe",
    "korfez",
    "körfez",
    "derince",
    "cayirova",
    "çayırova",
    "dilovasi",
    "dilovası",
    "gebze",
    "kandira",
    "kandıra",
}

MANUAL_UNKNOWN_ID = "manual_unknown"
NO_SITE_ID = "site_yok"

KOCAELI_COUNTY_NUMERIC = [
    "site_project_county_frequency",
    "county_site_project_count",
    "known_premium_x_county_code",
    "site_project_freq_x_county_code",
]

KOCAELI_COUNTY_CATEGORICAL = [
    "site_project_id_county_scoped",  # alias of namespaced site_project_id for clarity
    "site_project_x_county",
    "site_tier_x_county",
]


def county_slug(county: Any) -> str:
    s = fold_text(county)
    if not s or s in {"nan", "none", "missing", ""}:
        return "unknown"
    # common Turkish display → ascii slug
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s or "unknown"


def scope_site_id(county: Any, local_id: str) -> str:
    """Namespace local site id under county. Keep missing/other/manual tokens special."""
    lid = str(local_id or "").strip()
    if lid in {"", "missing", "nan", "None"}:
        return "missing"
    if lid in {MANUAL_UNKNOWN_ID, NO_SITE_ID}:
        return lid
    if lid == "other":
        return f"{county_slug(county)}::other"
    return f"{county_slug(county)}::{lid}"


def unscope_site_id(scoped: str) -> tuple[str, str]:
    s = str(scoped or "")
    if "::" not in s:
        return "unknown", s
    a, b = s.split("::", 1)
    return a, b


def get_site_feature_names(
    site_extraction_mode: str = "full",
    site_project_encoding: str = "foldsafe_target",
) -> list[str]:
    mode = str(site_extraction_mode or "none").lower().strip()
    enc = str(site_project_encoding or "none").lower().strip()
    if mode in {"", "none"}:
        return []
    names: list[str] = []
    if mode in {"v20_parity", "alias", "dict", "tier", "interactions", "full"}:
        names += [
            "has_site_project_id",
            "site_project_listing_count",
            "site_project_freq_bucket",
            "site_project_known_premium_flag",
            "site_project_county_frequency",
            "county_site_project_count",
        ]
    if mode in {"tier", "interactions", "full"}:
        names += ["site_is_premium_tier", "site_is_mid_tier", "site_tier_code"]
    if mode in {"interactions", "full"}:
        names += list(INTERACTION_NUMERIC) + [
            "known_premium_x_county_code",
            "site_project_freq_x_county_code",
        ]
    if enc == "foldsafe_target" and mode not in {"", "none"}:
        names += list(FOLDSAFE_NUMERIC)
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def get_site_categorical_feature_names(
    site_extraction_mode: str = "full",
    site_project_encoding: str = "foldsafe_target",
) -> list[str]:
    mode = str(site_extraction_mode or "none").lower().strip()
    if mode in {"", "none"}:
        return []
    names: list[str] = []
    if mode in {"v20_parity", "alias", "dict", "tier", "interactions", "full"}:
        names += [
            "site_project_id",
            "site_project_id_county_scoped",
            "site_project_match_source",
            "site_project_x_county",
        ]
    if mode in {"tier", "interactions", "full"}:
        names += ["site_quality_tier", "site_tier_x_county"]
    if mode in {"interactions", "full"}:
        names += list(INTERACTION_CATEGORICAL)
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _county_code_map(series: pd.Series) -> dict[str, float]:
    folded = sorted({fold_text(x) for x in series.fillna("missing").astype(str)})
    return {v: float(i + 1) for i, v in enumerate(folded)}


class KocaeliSiteProjectExtractionAdder(_BaseSiteAdder):
    """County-scoped site extraction for Kocaeli global models."""

    def __init__(
        self,
        site_extraction_mode: str = "full",
        site_project_encoding: str = "frequency",
        min_site_freq: int = 1,
        merge_gap_warning_tl: float = 8000.0,
    ):
        super().__init__(
            site_extraction_mode=site_extraction_mode,
            site_project_encoding=site_project_encoding,
            min_site_freq=min_site_freq,
            merge_gap_warning_tl=merge_gap_warning_tl,
        )
        self.county_site_counts_: dict[str, int] = {}
        self.county_n_sites_: dict[str, int] = {}
        self.county_codes_: dict[str, float] = {}

    def fit(self, X, y=None):
        # Fit base on a copy with temporary local ids, then rebuild scoped counts.
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        super().fit(df, y)
        # Recompute scoped frequency from transform of training X
        mode = str(self.site_extraction_mode or "none").lower()
        if mode in {"", "none"}:
            return self
        tmp = super().transform(df)
        # Namespace IDs and recount
        counties = df["county"] if "county" in df.columns else pd.Series(["unknown"] * len(df), index=df.index)
        local_ids = tmp["site_project_id"].astype(str) if "site_project_id" in tmp.columns else pd.Series(["missing"] * len(df))
        scoped = [scope_site_id(c, lid) for c, lid in zip(counties, local_ids)]
        vc = pd.Series(scoped).value_counts()
        self.site_counts_ = {str(k): int(v) for k, v in vc.items() if str(k) not in {"missing"}}
        self.county_site_counts_ = dict(self.site_counts_)
        # per-county distinct site count
        by_c: dict[str, set[str]] = {}
        for c, sid in zip(counties.map(county_slug), scoped):
            if sid in {"missing", MANUAL_UNKNOWN_ID, NO_SITE_ID}:
                continue
            by_c.setdefault(str(c), set()).add(str(sid))
        self.county_n_sites_ = {k: len(v) for k, v in by_c.items()}
        self.county_codes_ = _county_code_map(counties)
        return self

    def _resolve_one(self, row: pd.Series) -> dict[str, Any]:
        base = super()._resolve_one(row)
        cslug = county_slug(row.get("county"))
        local = str(base["site_project_id"])
        scoped = scope_site_id(row.get("county"), local)
        # frequency lookups use scoped counts when available
        cnt = float(self.site_counts_.get(scoped, self.site_counts_.get(local, 0))) if scoped not in {"missing"} else 0.0
        # Başiskele dict premium flags already set for local match; keep them after scoping
        base["site_project_id"] = scoped
        base["site_project_id_county_scoped"] = scoped
        base["has_site_project_id"] = int(scoped not in {"missing", "other", f"{cslug}::other", ""})
        if scoped.endswith("::other"):
            base["has_site_project_id"] = 0
        base["site_project_listing_count"] = cnt
        base["site_project_freq_bucket"] = float(freq_bucket(int(cnt)))
        base["site_project_county_frequency"] = cnt
        base["county_site_project_count"] = float(self.county_n_sites_.get(cslug, 0))
        base["site_project_x_county"] = f"{scoped}__{cslug}" if scoped not in {"missing"} else f"missing__{cslug}"
        tier = str(base.get("site_quality_tier") or "unknown")
        base["site_tier_x_county"] = f"{tier}__{cslug}"
        return base

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        out = df.copy()
        mode = str(self.site_extraction_mode or "none").lower()
        if mode in {"", "none"}:
            return out

        n = len(out)
        resolved = [self._resolve_one(row) for _, row in out.iterrows()]
        counties = out["county"] if "county" in out.columns else pd.Series(["unknown"] * n, index=out.index)
        ccode = counties.map(lambda x: float(self.county_codes_.get(fold_text(x), 0.0))).to_numpy()

        block: dict[str, Any] = {
            "site_project_id": [r["site_project_id"] for r in resolved],
            "site_project_id_county_scoped": [r["site_project_id_county_scoped"] for r in resolved],
            "has_site_project_id": [r["has_site_project_id"] for r in resolved],
            "site_project_listing_count": [r["site_project_listing_count"] for r in resolved],
            "site_project_freq_bucket": [r["site_project_freq_bucket"] for r in resolved],
            "site_project_known_premium_flag": [r["site_project_known_premium_flag"] for r in resolved],
            "site_project_match_source": [r["site_project_match_source"] for r in resolved],
            "site_project_county_frequency": [r["site_project_county_frequency"] for r in resolved],
            "county_site_project_count": [r["county_site_project_count"] for r in resolved],
            "site_project_x_county": [r["site_project_x_county"] for r in resolved],
        }
        if mode in {"tier", "interactions", "full"}:
            block["site_quality_tier"] = [r["site_quality_tier"] for r in resolved]
            block["site_is_premium_tier"] = [r["site_is_premium_tier"] for r in resolved]
            block["site_is_mid_tier"] = [r["site_is_mid_tier"] for r in resolved]
            block["site_tier_code"] = [r["site_tier_code"] for r in resolved]
            block["site_tier_x_county"] = [r["site_tier_x_county"] for r in resolved]

        if mode in {"interactions", "full"}:
            from site_project_extraction import _coast_inv, _district_code, _large_home01, _site_inside01

            coast_inv = _coast_inv(out).to_numpy()
            large = _large_home01(out).to_numpy()
            site_in = _site_inside01(out).to_numpy()
            district = out["district"] if "district" in out.columns else pd.Series(["missing"] * n)
            dcode = _district_code(district).to_numpy()
            known = np.array([r["site_project_known_premium_flag"] for r in resolved], dtype=float)
            tier_c = np.array([r["site_tier_code"] for r in resolved], dtype=float)
            cnt = np.array([r["site_project_listing_count"] for r in resolved], dtype=float)
            has = np.array([r["has_site_project_id"] for r in resolved], dtype=float)
            buckets = [f"b{int(r['site_project_freq_bucket'])}" for r in resolved]
            tiers = [r.get("site_quality_tier", "unknown") for r in resolved]
            dist_f = district.astype(str).fillna("missing").map(fold_text)
            freq_b = np.array([r["site_project_freq_bucket"] for r in resolved], dtype=float)

            block["distance_to_coastline_inv"] = coast_inv
            block["known_premium_x_district_code"] = known * dcode
            block["site_tier_x_large_home"] = tier_c * large
            block["site_tier_x_coast_inv"] = tier_c * coast_inv
            block["canonical_count_x_large_home"] = cnt * large
            block["has_site_x_site_inside"] = has * site_in
            block["site_tier_x_district"] = [f"{a}__{b}" for a, b in zip(tiers, dist_f)]
            block["site_id_bucket_x_district"] = [f"{a}__{b}" for a, b in zip(buckets, dist_f)]
            block["known_premium_x_county_code"] = known * ccode
            block["site_project_freq_x_county_code"] = freq_b * ccode

        drop_cols = [c for c in block.keys() if c in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)
        return pd.concat([out, pd.DataFrame(block, index=out.index)], axis=1)


class SiteProjectFoldSafeEncoder(_BaseFoldSafe):
    """Fold-safe encoding on county-scoped site_project_id.

    Skips missing/other/manual_unknown (uses global fallbacks in transform).
    """

    def fit(self, X, y=None):
        super().fit(X, y)
        # Drop unsafe keys if present
        drop = [k for k in list(getattr(self, "stats_", {}) or {}) if k in {MANUAL_UNKNOWN_ID, NO_SITE_ID, "missing", "other"} or k.endswith("::other")]
        for k in drop:
            self.stats_.pop(k, None)
        return self


def candidates_table(df: pd.DataFrame) -> pd.DataFrame:
    if "site_project_id" not in df.columns:
        return pd.DataFrame()
    cols = {"n_listings": ("site_project_id", "size")}
    if "site_project_known_premium_flag" in df.columns:
        cols["known_premium"] = ("site_project_known_premium_flag", "max")
    if "county" in df.columns:
        g = (
            df.groupby(["county", "site_project_id"], dropna=False)
            .agg(**{k: v for k, v in cols.items()})
            .reset_index()
            .sort_values("n_listings", ascending=False)
        )
    else:
        g = (
            df.groupby("site_project_id", dropna=False)
            .agg(**{k: v for k, v in cols.items()})
            .reset_index()
            .sort_values("n_listings", ascending=False)
        )
    return g


def coverage_by_county(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "county" not in df.columns:
        return pd.DataFrame()
    rows = []
    for county, sub in df.groupby("county", dropna=False):
        n = len(sub)
        has = pd.to_numeric(sub.get("has_site_project_id"), errors="coerce").fillna(0) if "has_site_project_id" in sub.columns else pd.Series(0, index=sub.index)
        dict_hit = sub["site_project_match_source"].astype(str).eq("dict") if "site_project_match_source" in sub.columns else pd.Series(False, index=sub.index)
        rows.append(
            {
                "county": county,
                "rows": int(n),
                "site_coverage": float(has.mean()) if n else np.nan,
                "dict_hit_rate": float(dict_hit.mean()) if n else np.nan,
                "canonical_non_missing_rate": float((sub["site_project_id"].astype(str) != "missing").mean()) if "site_project_id" in sub.columns and n else np.nan,
                "n_unique_sites": int(sub["site_project_id"].nunique()) if "site_project_id" in sub.columns else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("rows", ascending=False)


def build_site_project_options(df: pd.DataFrame) -> pd.DataFrame:
    """App-safe site option catalog (county/district scoped)."""
    if df is None or df.empty or "site_project_id" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work = work[work["site_project_id"].astype(str).isin([MANUAL_UNKNOWN_ID, NO_SITE_ID, "missing"]) == False]  # noqa: E712
    work = work[~work["site_project_id"].astype(str).str.endswith("::other", na=False)]
    if work.empty:
        return pd.DataFrame()

    lat = pd.to_numeric(work.get("lat"), errors="coerce") if "lat" in work.columns else pd.Series(np.nan, index=work.index)
    lon = pd.to_numeric(work.get("lon"), errors="coerce") if "lon" in work.columns else pd.Series(np.nan, index=work.index)
    work = work.assign(_lat=lat, _lon=lon)
    src = work["site_project_match_source"].astype(str) if "site_project_match_source" in work.columns else pd.Series("", index=work.index)
    work = work.assign(_src_site=(src == "site_name").astype(int), _src_title=(src == "title").astype(int))

    group_cols = [c for c in ["city", "county", "district", "site_project_id"] if c in work.columns]
    if "county" not in group_cols:
        group_cols = ["site_project_id"]
    g = (
        work.groupby(group_cols, dropna=False)
        .agg(
            listing_count=("site_project_id", "size"),
            lat_median=("_lat", "median"),
            lon_median=("_lon", "median"),
            lat_mean=("_lat", "mean"),
            lon_mean=("_lon", "mean"),
            known_premium_flag=("site_project_known_premium_flag", "max")
            if "site_project_known_premium_flag" in work.columns
            else ("site_project_id", "size"),
            quality_tier=("site_quality_tier", lambda s: s.mode().iloc[0] if len(s.mode()) else "unknown")
            if "site_quality_tier" in work.columns
            else ("site_project_id", "size"),
            source_count_site_name=("_src_site", "sum"),
            source_count_title=("_src_title", "sum"),
        )
        .reset_index()
    )
    # display / canonical
    def _display(sid: str) -> str:
        _c, local = unscope_site_id(sid)
        return local.replace("_", " ").strip().title()

    g["display_name"] = g["site_project_id"].astype(str).map(_display)
    g["canonical_name"] = g["site_project_id"].astype(str).map(lambda s: unscope_site_id(s)[1])
    g["confidence"] = (pd.to_numeric(g["listing_count"], errors="coerce").fillna(0) / (pd.to_numeric(g["listing_count"], errors="coerce").fillna(0) + 5.0)).clip(0, 1)
    if "city" not in g.columns:
        g["city"] = "Kocaeli"
    if "last_seen_at" not in g.columns:
        g["last_seen_at"] = ""
    cols = [
        "city",
        "county",
        "district",
        "site_project_id",
        "display_name",
        "canonical_name",
        "listing_count",
        "lat_median",
        "lon_median",
        "lat_mean",
        "lon_mean",
        "confidence",
        "known_premium_flag",
        "quality_tier",
        "source_count_site_name",
        "source_count_title",
        "last_seen_at",
    ]
    for c in cols:
        if c not in g.columns:
            g[c] = np.nan if c.startswith("lat") or c.startswith("lon") else ""
    return g[cols].sort_values(["county", "listing_count"], ascending=[True, False])


def write_site_project_options(df: pd.DataFrame, artifacts_dir: Path) -> Path:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    opt = build_site_project_options(df)
    csv_path = artifacts_dir / "site_project_options_kocaeli_v24.csv"
    json_path = artifacts_dir / "site_project_options_kocaeli_v24.json"
    opt.to_csv(csv_path, index=False, encoding="utf-8-sig")
    payload = {
        "version": "v24",
        "city": "Kocaeli",
        "manual_options": [
            {"site_project_id": NO_SITE_ID, "display_name": "Site yok"},
            {"site_project_id": MANUAL_UNKNOWN_ID, "display_name": "Listede yok / manuel giriş"},
        ],
        "prediction_notes": {
            "manual_unknown": "Maps to site_project_id=manual_unknown; foldsafe oof features use county/global fallback; never treat as known encoded site.",
            "match_existing_alias": "If user text matches alias, use county-scoped canonical id.",
        },
        "sites": opt.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return json_path


# Alias used by train imports expecting SiteProjectExtractionAdder name
SiteProjectExtractionAdder = KocaeliSiteProjectExtractionAdder
