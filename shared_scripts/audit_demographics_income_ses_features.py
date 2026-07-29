#!/usr/bin/env python
"""Audit neighborhood demographics for income / SES columns.

Purpose:
  Before V20, check whether estimated income level and socio-economic status
  signals exist in neighborhood demographics and whether Başiskele coverage is
  usable for modeling.

Examples:
  python shared_scripts/audit_demographics_income_ses_features.py
  python shared_scripts/audit_demographics_income_ses_features.py --city Kocaeli --county Başiskele
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import inspect, text

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from db_utils import create_engine, get_database_url, validate_table_name  # noqa: E402
from env_loader import find_project_root, load_root_env  # noqa: E402

# Keyword search on column names (case-insensitive, accent-folded).
INCOME_SES_KEYWORDS: tuple[str, ...] = (
    "gelir",
    "income",
    "tahmini",
    "estimated",
    "seviye",
    "level",
    "sosyo",
    "socio",
    "ekonomik",
    "economic",
    "statü",
    "statu",
    "status",
    "ses",
    "segment",
    "class",
)

# V18/V19 safe demographics: raw district_demographics columns that map to demo_*.
# Source: train_v19_basiskele_calibration_pipeline.build_demographic_features (safe_base).
V18_V19_SAFE_BASE_COLUMNS: tuple[str, ...] = (
    "population_total",
    "population_male",
    "population_female",
    "population_density",
    "young_ratio",
    "middle_ratio",
    "old_ratio",
    "young_count",
    "middle_count",
    "old_count",
    "female_ratio",
    "male_ratio",
    "married_ratio",
    "single_ratio",
    "divorced_ratio",
    "widow_ratio",
    "never_married_count",
    "married_count",
    "divorced_count",
    "widow_count",
    "age_0_14",
    "age_15_24",
    "age_25_34",
    "age_35_44",
    "age_45_54",
    "age_55_64",
    "age_65_plus",
    "age_0_14_count",
    "age_15_24_count",
    "age_25_34_count",
    "age_35_44_count",
    "age_45_54_count",
    "age_55_64_count",
    "age_65_plus_count",
    "education_total",
    "education_university_ratio",
    "education_university_count",
    "education_high_school_ratio",
    "education_high_school_count",
    "education_middle_school_ratio",
    "education_middle_school_count",
    "education_primary_school_ratio",
    "education_primary_school_count",
    "education_primary_education_ratio",
    "education_graduate_ratio",
    "education_graduate_count",
    "education_doctorate_ratio",
    "education_doctorate_count",
    "education_non_literate_ratio",
    "education_unknown_ratio",
    "ses_a_plus_count",
    "ses_a_count",
    "ses_b_count",
    "ses_c_count",
    "ses_d_count",
    "ses_a_plus_ratio",
    "ses_a_ratio",
    "ses_b_ratio",
    "ses_c_ratio",
    "ses_d_ratio",
    "ses_ab_ratio",
    "ses_cd_ratio",
    "household_count",
    "household_size",
    "per_capita_income_try",
    "household_income_try",
    "residential_count",
    "workplace_count",
    "summer_house_count",
    "vehicle_count",
    "car_count",
    "atm_count",
    "pharmacy_count",
    "bank_count",
)

V18_V19_SAFE_CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "dominant_age_group",
    "dominant_marital_status",
    "dominant_education",
    "ses_group",
)

# full-mode-only extras that may still match income/SES keywords
V18_V19_FULL_EXTRA_COLUMNS: tuple[str, ...] = (
    "real_estate_agent_count",
    "agent_listing_count",
    "owner_listing_count",
    "sale_count",
    "mortgage_count",
    "turnover_ratio",
    "computed_turnover_ratio",
    "listing_count_2024",
    "bb_sale_count_2024",
    "bb_mortgaged_sale_count_2024",
    "saving_total",
    "expense_total",
    "expense_food",
    "expense_shelter",
    "expense_transportation",
    "expense_education",
    "ecommerce_count",
    "ecommerce_density",
    "online_retail",
)

SAFE_SET = {c.lower() for c in V18_V19_SAFE_BASE_COLUMNS} | {
    c.lower() for c in V18_V19_SAFE_CATEGORICAL_COLUMNS
}
FULL_EXTRA_SET = {c.lower() for c in V18_V19_FULL_EXTRA_COLUMNS}

# Strong income/SES signal keywords vs weak/ambiguous ones (level/status/class/segment).
STRONG_INCOME_SES_KEYWORDS = {
    "gelir",
    "income",
    "tahmini",
    "estimated",
    "seviye",
    "sosyo",
    "socio",
    "ekonomik",
    "economic",
    "ses",
}

# Coverage thresholds for summary judgment
COVERAGE_GOOD = 0.70
COVERAGE_OK = 0.40
MIN_UNIQUE_NUMERIC = 3
MIN_UNIQUE_CATEGORICAL = 2
TOP_VALUE_N = 15


def _keyword_list(matched: str) -> list[str]:
    return [k.strip() for k in str(matched or "").split(",") if k.strip()]


def _is_strong_income_ses_hit(matched: str) -> bool:
    kws = {_fold(k) for k in _keyword_list(matched)}
    return bool(kws & {_fold(k) for k in STRONG_INCOME_SES_KEYWORDS})


def _repo_root() -> Path:
    root = find_project_root(HERE)
    if root is None:
        raise RuntimeError("Could not locate project root (MANIFEST.json + data/).")
    return root


def _ts_folder() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _fold(s: str) -> str:
    """Lowercase + strip accents for keyword matching (statü -> statu)."""
    text_norm = unicodedata.normalize("NFKD", str(s or ""))
    text_norm = "".join(ch for ch in text_norm if not unicodedata.combining(ch))
    return text_norm.lower()


def _match_keywords(col_name: str) -> list[str]:
    folded = _fold(col_name)
    # Word-ish match: keyword as substring, but for short tokens like "ses"
    # require boundary-ish context to reduce false positives (e.g. "houses").
    hits: list[str] = []
    for kw in INCOME_SES_KEYWORDS:
        kw_f = _fold(kw)
        if len(kw_f) <= 3:
            # short keyword: match as token / underscore segment
            if re.search(rf"(^|_){re.escape(kw_f)}(_|$)", folded) or folded == kw_f:
                hits.append(kw)
        else:
            if kw_f in folded:
                hits.append(kw)
    return hits


def _classify_value_type(series: pd.Series, sql_type: str | None) -> str:
    """Return 'numeric' or 'categorical' based on values + SQL hint."""
    non_null = series.dropna()
    if non_null.empty:
        sql_l = (sql_type or "").lower()
        if any(t in sql_l for t in ("int", "float", "numeric", "double", "real", "decimal", "money")):
            return "numeric"
        return "categorical"

    numeric = pd.to_numeric(non_null, errors="coerce")
    numeric_ok_ratio = float(numeric.notna().mean())
    sql_l = (sql_type or "").lower()
    sql_numeric = any(
        t in sql_l for t in ("int", "float", "numeric", "double", "real", "decimal", "money")
    )
    sql_text = any(t in sql_l for t in ("char", "text", "enum", "uuid", "bool"))

    if sql_numeric and numeric_ok_ratio >= 0.8:
        return "numeric"
    if sql_text and numeric_ok_ratio < 0.95:
        return "categorical"
    if numeric_ok_ratio >= 0.9:
        return "numeric"
    return "categorical"


def introspect_columns(engine, table: str) -> pd.DataFrame:
    """Return column metadata from SQLAlchemy inspector."""
    relation = validate_table_name(table)
    schema, name = relation.split(".", 1)
    insp = inspect(engine)
    cols: list[dict[str, Any]] = []
    schema_used = None
    try:
        cols = insp.get_columns(name, schema=schema)
        schema_used = schema
    except Exception:
        for sch in insp.get_schema_names():
            try:
                got = insp.get_columns(name, schema=sch)
                if got:
                    cols = got
                    schema_used = sch
                    break
            except Exception:
                continue
    if not cols:
        raise RuntimeError(f"Could not introspect columns for table '{relation}'.")

    rows = []
    for c in cols:
        col_name = str(c.get("name", ""))
        dtype = c.get("type")
        rows.append(
            {
                "column_name": col_name,
                "sql_type": str(dtype) if dtype is not None else "",
                "nullable": bool(c.get("nullable", True)),
                "schema": schema_used or "",
            }
        )
    return pd.DataFrame(rows)


def fetch_demographics(
    engine,
    table: str,
    *,
    city: str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    import re

    from canonical_db import alias_demographics_frame, sql_relation

    _col_ident = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    relation_sql = sql_relation(table)
    if columns:
        # Map legacy requested columns to canonical physical names when needed.
        legacy_to_physical = {
            "city_id": "province_id",
            "county_id": "district_id",
            "district_id": "neighborhood_id",
            "city_name": "province_name",
            "county_name": "district_name",
            "district_name": "neighborhood_name",
            "district_slug": "neighborhood_slug",
        }
        physical_cols: list[str] = []
        for col in columns:
            if not _col_ident.match(col):
                raise ValueError(f"Unsafe column name: {col!r}")
            physical_cols.append(legacy_to_physical.get(col, col))
        # de-dupe preserving order
        seen: set[str] = set()
        ordered: list[str] = []
        for c in physical_cols:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        select_sql = ", ".join(ordered)
    else:
        select_sql = "*"

    where = ""
    params: dict[str, Any] = {}
    if city:
        where = "WHERE lower(coalesce(province_name, '')) = lower(:city)"
        params["city"] = city

    sql = text(f"SELECT {select_sql} FROM {relation_sql} {where}")
    df = pd.read_sql(sql, engine, params=params or None)
    return alias_demographics_frame(df)


def _county_mask(df: pd.DataFrame, county: str) -> pd.Series:
    if "county_name" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    target = _fold(county)
    return df["county_name"].map(lambda x: _fold(x) == target)


def coverage_for_scope(
    df: pd.DataFrame,
    candidate_cols: list[str],
    *,
    scope_name: str,
    mask: pd.Series | None = None,
) -> pd.DataFrame:
    scoped = df if mask is None else df.loc[mask]
    n = int(len(scoped))
    rows = []
    for col in candidate_cols:
        if col not in scoped.columns:
            rows.append(
                {
                    "scope": scope_name,
                    "column_name": col,
                    "row_count": n,
                    "non_null_count": 0,
                    "null_count": n,
                    "coverage_ratio": 0.0 if n else np.nan,
                    "distinct_non_null": 0,
                }
            )
            continue
        s = scoped[col]
        non_null = int(s.notna().sum())
        # Treat empty strings as null for coverage
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            empty = s.astype(str).str.strip().isin(["", "nan", "None", "none", "null", "<NA>"])
            non_null = int((s.notna() & ~empty).sum())
        rows.append(
            {
                "scope": scope_name,
                "column_name": col,
                "row_count": n,
                "non_null_count": non_null,
                "null_count": n - non_null,
                "coverage_ratio": (non_null / n) if n else np.nan,
                "distinct_non_null": int(s.dropna().nunique()),
            }
        )
    return pd.DataFrame(rows)


def value_distribution(
    df: pd.DataFrame,
    candidate_meta: pd.DataFrame,
    *,
    basiskele_mask: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, meta in candidate_meta.iterrows():
        col = str(meta["column_name"])
        vtype = str(meta["value_type"])
        if col not in df.columns:
            continue
        for scope_name, mask in (("basiskele", basiskele_mask), ("kocaeli", None)):
            scoped = df if mask is None else df.loc[mask]
            s = scoped[col]
            if vtype == "numeric":
                num = pd.to_numeric(s, errors="coerce")
                valid = num.dropna()
                if valid.empty:
                    rows.append(
                        {
                            "scope": scope_name,
                            "column_name": col,
                            "value_type": vtype,
                            "stat": "empty",
                            "value": np.nan,
                            "count": 0,
                            "share": np.nan,
                        }
                    )
                    continue
                stats = {
                    "min": float(valid.min()),
                    "max": float(valid.max()),
                    "mean": float(valid.mean()),
                    "median": float(valid.median()),
                    "p10": float(valid.quantile(0.10)),
                    "p90": float(valid.quantile(0.90)),
                    "non_null_count": float(len(valid)),
                }
                for stat_name, val in stats.items():
                    rows.append(
                        {
                            "scope": scope_name,
                            "column_name": col,
                            "value_type": vtype,
                            "stat": stat_name,
                            "value": val,
                            "count": int(len(valid)) if stat_name != "non_null_count" else int(val),
                            "share": np.nan,
                        }
                    )
            else:
                # categorical
                cleaned = s.astype("object").where(s.notna(), np.nan)
                cleaned = cleaned.map(
                    lambda x: str(x).strip() if pd.notna(x) else np.nan
                )
                cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan, "none": np.nan})
                valid = cleaned.dropna()
                n_valid = int(len(valid))
                rows.append(
                    {
                        "scope": scope_name,
                        "column_name": col,
                        "value_type": vtype,
                        "stat": "unique_count",
                        "value": float(valid.nunique()) if n_valid else 0.0,
                        "count": n_valid,
                        "share": np.nan,
                    }
                )
                if n_valid == 0:
                    continue
                vc = valid.value_counts(dropna=True).head(TOP_VALUE_N)
                for i, (val, cnt) in enumerate(vc.items(), start=1):
                    rows.append(
                        {
                            "scope": scope_name,
                            "column_name": col,
                            "value_type": vtype,
                            "stat": f"top_{i}",
                            "value": str(val),
                            "count": int(cnt),
                            "share": float(cnt / n_valid) if n_valid else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def safe_usage_check(candidate_meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    safe_cat = {c.lower() for c in V18_V19_SAFE_CATEGORICAL_COLUMNS}
    for _, meta in candidate_meta.iterrows():
        col = str(meta["column_name"])
        col_l = col.lower()
        matched_kw = str(meta.get("matched_keywords", ""))
        strong = _is_strong_income_ses_hit(matched_kw)
        nuniq_all = int(meta.get("distinct_non_null_all", 0) or 0)
        in_safe = col_l in SAFE_SET
        in_full_extra = col_l in FULL_EXTRA_SET
        mapped_demo = f"demo_{col_l}" if (in_safe or in_full_extra) else ""

        # Constant / weak-keyword-only hits are unlikely to be income/SES signals.
        likely_false_positive = (not strong) or (nuniq_all <= 1 and not strong)

        if in_safe:
            usage = "used_by_safe_demographics"
            note = f"Mapped to {mapped_demo} in V18/V19 demographics_mode=safe"
            if not strong:
                note += " (keyword hit via weak/ambiguous token; not a dedicated income/SES column)"
        elif in_full_extra:
            usage = "full_mode_only"
            if strong:
                note = "income/ses exists but not used by safe demographics (full mode only)"
            else:
                note = f"Mapped to {mapped_demo} only in demographics_mode=full"
        elif likely_false_positive:
            usage = "keyword_false_positive_likely"
            note = (
                "Keyword match only via weak/ambiguous token "
                f"({matched_kw}); not treated as unused income/SES signal"
            )
        else:
            usage = "not_used_by_safe_demographics"
            note = "income/ses exists but not used by safe demographics"

        rows.append(
            {
                "column_name": col,
                "matched_keywords": matched_kw,
                "strong_income_ses_keyword": strong,
                "value_type": meta.get("value_type"),
                "sql_type": meta.get("sql_type"),
                "in_v18_v19_safe_base": in_safe and col_l not in safe_cat,
                "in_v18_v19_safe_categorical": col_l in safe_cat,
                "in_v18_v19_full_extra": in_full_extra,
                "mapped_demo_feature": mapped_demo,
                "safe_usage_status": usage,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def _coverage_verdict(cov_df: pd.DataFrame) -> tuple[str, float]:
    if cov_df.empty:
        return "no_candidates", float("nan")
    ratios = pd.to_numeric(cov_df["coverage_ratio"], errors="coerce")
    med = float(ratios.median()) if ratios.notna().any() else float("nan")
    if np.isnan(med):
        return "unknown", med
    if med >= COVERAGE_GOOD:
        return "sufficient", med
    if med >= COVERAGE_OK:
        return "partial", med
    return "insufficient", med


def _model_suitability(
    candidate_meta: pd.DataFrame,
    basiskele_cov: pd.DataFrame,
    usage: pd.DataFrame,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if candidate_meta.empty:
        return "not_suitable", ["No income/SES keyword columns found in district_demographics."]

    cov_map = {
        str(r["column_name"]): float(r["coverage_ratio"])
        for _, r in basiskele_cov.iterrows()
        if pd.notna(r.get("coverage_ratio"))
    }
    usable = []
    for _, row in candidate_meta.iterrows():
        col = str(row["column_name"])
        cov = cov_map.get(col, 0.0)
        nuniq = int(row.get("distinct_non_null_basiskele", 0) or 0)
        vtype = str(row["value_type"])
        min_uniq = MIN_UNIQUE_NUMERIC if vtype == "numeric" else MIN_UNIQUE_CATEGORICAL
        if cov >= COVERAGE_OK and nuniq >= min_uniq:
            usable.append(col)

    if "strong_income_ses_keyword" in usage.columns:
        already_safe = usage.loc[
            (usage["safe_usage_status"] == "used_by_safe_demographics")
            & (usage["strong_income_ses_keyword"].astype(bool)),
            "column_name",
        ].tolist()
    else:
        already_safe = usage.loc[
            usage["safe_usage_status"] == "used_by_safe_demographics", "column_name"
        ].tolist()

    unused = usage.loc[
        usage["safe_usage_status"] == "not_used_by_safe_demographics", "column_name"
    ].tolist()
    false_pos = usage.loc[
        usage["safe_usage_status"] == "keyword_false_positive_likely", "column_name"
    ].tolist()

    if already_safe:
        reasons.append(
            f"Already in V18/V19 safe demographics (strong income/SES): {', '.join(already_safe)}"
        )
    if unused:
        reasons.append(
            f"Strong income/SES keyword match outside safe list: {', '.join(unused)}"
        )
    if false_pos:
        reasons.append(
            f"Weak/ambiguous keyword hits (likely false positives): {', '.join(false_pos)}"
        )
    if usable:
        reasons.append(f"Başiskele-usable columns (coverage≥{COVERAGE_OK:.0%}): {', '.join(usable)}")
    else:
        reasons.append("No candidate cleared Başiskele coverage + diversity thresholds.")

    if already_safe and any(c in usable for c in already_safe):
        verdict = "already_in_safe_and_usable"
    elif usable and unused and any(c in usable for c in unused):
        verdict = "usable_but_not_in_safe"
    elif already_safe:
        verdict = "in_safe_but_check_coverage"
    elif usable:
        verdict = "usable_candidate"
    else:
        verdict = "not_suitable"
    return verdict, reasons


def write_summary(
    path: Path,
    *,
    table: str,
    city: str,
    county: str,
    all_cols: pd.DataFrame,
    candidates: pd.DataFrame,
    basiskele_cov: pd.DataFrame,
    kocaeli_cov: pd.DataFrame,
    usage: pd.DataFrame,
    n_basiskele: int,
    n_kocaeli: int,
) -> None:
    found = not candidates.empty
    b_verdict, b_med = _coverage_verdict(basiskele_cov)
    k_verdict, k_med = _coverage_verdict(kocaeli_cov)
    suitability, suit_reasons = _model_suitability(candidates, basiskele_cov, usage)

    unused_income_ses = usage.loc[
        usage["safe_usage_status"].isin(
            ["not_used_by_safe_demographics", "full_mode_only"]
        )
    ]
    false_pos_rows = usage.loc[
        usage["safe_usage_status"] == "keyword_false_positive_likely"
    ]
    used_safe = usage.loc[usage["safe_usage_status"] == "used_by_safe_demographics"]
    if "strong_income_ses_keyword" in used_safe.columns:
        used_safe_strong = used_safe.loc[used_safe["strong_income_ses_keyword"].astype(bool)]
    else:
        used_safe_strong = used_safe

    lines: list[str] = []
    lines.append("# Neighborhood Demographics Income / SES Feature Audit")
    lines.append("")
    lines.append(f"- Table: `{table}`")
    lines.append(f"- City filter: `{city}`")
    lines.append(f"- County focus: `{county}`")
    lines.append(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Total columns introspected: **{len(all_cols)}**")
    lines.append(f"- Income/SES keyword candidates: **{len(candidates)}**")
    lines.append(f"- Kocaeli rows: **{n_kocaeli}** | Başiskele rows: **{n_basiskele}**")
    lines.append("")

    lines.append("## Gelir/SES kolonları bulundu mu?")
    if found:
        strong_cols = []
        weak_cols = []
        for _, r in candidates.iterrows():
            if _is_strong_income_ses_hit(str(r.get("matched_keywords", ""))):
                strong_cols.append(r)
            else:
                weak_cols.append(r)
        if strong_cols:
            lines.append(
                f"**Evet.** Güçlü gelir/SES keyword eşleşmesi: **{len(strong_cols)}** kolon."
            )
            lines.append("")
            for r in strong_cols:
                lines.append(
                    f"- `{r['column_name']}` ({r['value_type']}, sql=`{r['sql_type']}`) "
                    f"— keywords: {r['matched_keywords']}"
                )
        else:
            lines.append(
                "**Hayır (güçlü sinyal yok).** Sadece zayıf/ambiguous keyword eşleşmeleri var."
            )
        if weak_cols:
            lines.append("")
            lines.append("Zayıf keyword eşleşmeleri (muhtemel false positive):")
            for r in weak_cols:
                lines.append(
                    f"- `{r['column_name']}` ({r['value_type']}) — keywords: {r['matched_keywords']}"
                )
    else:
        lines.append("**Hayır.** `district_demographics` kolon adlarında gelir/SES keyword eşleşmesi yok.")
    lines.append("")

    lines.append("## Başiskele coverage yeterli mi?")
    if found:
        lines.append(
            f"- Median coverage (Başiskele): **{b_med:.1%}** → `{b_verdict}` "
            f"(good≥{COVERAGE_GOOD:.0%}, ok≥{COVERAGE_OK:.0%})"
        )
        lines.append(
            f"- Median coverage (Kocaeli): **{k_med:.1%}** → `{k_verdict}`"
        )
        lines.append("")
        lines.append("| column | basiskele_coverage | kocaeli_coverage |")
        lines.append("|---|---:|---:|")
        b_map = {str(r["column_name"]): r for _, r in basiskele_cov.iterrows()}
        k_map = {str(r["column_name"]): r for _, r in kocaeli_cov.iterrows()}
        for col in candidates["column_name"]:
            bc = b_map.get(str(col), {})
            kc = k_map.get(str(col), {})
            br = bc.get("coverage_ratio", np.nan)
            kr = kc.get("coverage_ratio", np.nan)
            lines.append(
                f"| `{col}` | {br:.1%} | {kr:.1%} |"
                if pd.notna(br) and pd.notna(kr)
                else f"| `{col}` | {br} | {kr} |"
            )
        if b_verdict == "sufficient":
            lines.append("")
            lines.append("Başiskele coverage **yeterli** görünüyor.")
        elif b_verdict == "partial":
            lines.append("")
            lines.append("Başiskele coverage **kısmi**; feature olarak denenebilir ama missing-handling şart.")
        else:
            lines.append("")
            lines.append("Başiskele coverage **yetersiz**; doğrudan model feature olarak riskli.")
    else:
        lines.append("Aday kolon olmadığı için coverage değerlendirilemedi.")
    lines.append("")

    lines.append("## Model feature olarak kullanmaya uygun mu?")
    lines.append(f"- Verdict: **`{suitability}`**")
    for reason in suit_reasons:
        lines.append(f"- {reason}")
    lines.append("")
    if not used_safe_strong.empty:
        lines.append("Safe demographics içinde **zaten kullanılan** gelir/SES kolonları:")
        for col in used_safe_strong["column_name"]:
            lines.append(f"- `{col}` → `demo_{str(col).lower()}`")
        lines.append("")
    if not unused_income_ses.empty:
        lines.append(
            "> **income/ses exists but not used by safe demographics**"
        )
        lines.append("")
        for _, r in unused_income_ses.iterrows():
            lines.append(f"- `{r['column_name']}`: {r['note']}")
        lines.append("")
    elif not false_pos_rows.empty:
        lines.append(
            "Safe dışı kalan keyword hit’ler gelir/SES sinyali olarak yorumlanmadı "
            "(weak keyword / constant column)."
        )
        for _, r in false_pos_rows.iterrows():
            lines.append(f"- `{r['column_name']}`: {r['note']}")
        lines.append("")
    else:
        lines.append(
            "Güçlü gelir/SES kolonlarının tamamı V18/V19 **safe demographics** içinde."
        )
        lines.append("")

    lines.append("## V20’de nasıl dahil edilmeli?")
    if not found:
        lines.append(
            "- Yeni gelir/SES kolon yoksa V20’de ekstra income/SES "
            "feature eklemeye gerek yok; mevcut safe set ile devam edilebilir."
        )
    else:
        lines.append(
            "1. Önce mevcut safe income/SES kolonlarının (`per_capita_income_try`, "
            "`household_income_try`, `ses_*`, `ses_group`) Başiskele coverage ve "
            "feature importance’ını V19 raporlarından doğrula."
        )
        lines.append(
            "2. Keyword eşleşip safe listede olmayan kolonlar varsa: raw adı → "
            "`demo_*` mapping’ini `build_demographic_features` safe_base/categorical "
            "listesine ekle; derived county ratios sonra otomatik üretilir."
        )
        lines.append(
            "3. Yeni kolon eklerken `demographics_mode=safe` ablation’ı (none/safe/full) "
            "ile Başiskele MAPE/bias etkisini ölç; leakage için market-activity "
            "kolonlarını safe’e alma."
        )
        lines.append(
            "4. Coverage kısmiyse `demo_income_available` / `demo_ses_available` "
            "flag’lerini koru; missing için median-impute yerine availability flag + "
            "tree model native missing tercih et."
        )
        lines.append(
            "5. Categorical SES (`ses_group` / tahmini seviye) için rare-level "
            "bucketing ve Başiskele-only target encoding denemeden önce frequency "
            "encoding / one-hot with `missing` level kullan."
        )
        if suitability in ("already_in_safe_and_usable", "in_safe_but_check_coverage"):
            lines.append(
                "6. Bu audit’e göre gelir/SES sinyali **zaten safe path’te**; V20’de "
                "öncelik yeni kolon eklemek değil, Başiskele’de interaction "
                "(SES × m2 / SES × coastal) ve specialist residual katmanını test etmek olmalı."
            )
        elif suitability == "usable_but_not_in_safe":
            lines.append(
                "6. Kullanılabilir ama safe dışı kolonlar var → V20’de önce safe_base’e "
                "ekleyip kontrollü ablation yap."
            )
        else:
            lines.append(
                "6. Coverage/çeşitlilik yetersizse V20’ye income/SES kolon ekleme; "
                "önce demografi veri kalitesini / join anahtarlarını düzelt."
            )
    lines.append("")
    lines.append("## Files")
    lines.append("- `income_ses_column_candidates.csv`")
    lines.append("- `basiskele_income_ses_coverage.csv`")
    lines.append("- `kocaeli_income_ses_coverage.csv`")
    lines.append("- `income_ses_value_distribution.csv`")
    lines.append("- `safe_demographics_usage_check.csv`")
    lines.append("- `summary.md`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Audit neighborhood demographics for income/SES features."
    )
    ap.add_argument("--table", default="geo.neighborhood_demographics")
    ap.add_argument("--city", default="Kocaeli")
    ap.add_argument("--county", default="Başiskele")
    ap.add_argument(
        "--out-root",
        default=None,
        help="Default: <repo>/analysis_outputs/demographics_income_ses_audit/<timestamp>/",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    try:
        env_path = load_root_env(start=HERE)
    except Exception as exc:
        print(f"ERROR: failed to load root .env — {exc}")
        return 1

    root = _repo_root()
    out_dir = (
        Path(args.out_root)
        if args.out_root
        else root / "analysis_outputs" / "demographics_income_ses_audit" / _ts_folder()
    )
    _ensure_dir(out_dir)

    print(f"loaded_env={env_path.name}")
    print(f"DATABASE_URL_set={bool((os.getenv('DATABASE_URL') or os.getenv('DB_URL') or '').strip())}")
    print(f"DB_ROLE_PASSWORD_set={bool((os.getenv('DB_ROLE_PASSWORD') or '').strip())}")
    print(f"out_dir={out_dir}")

    engine = create_engine()
    table = validate_table_name(args.table)

    print(f"Introspecting columns on {table}...")
    all_cols = introspect_columns(engine, table)
    all_cols["matched_keywords"] = all_cols["column_name"].map(
        lambda c: ",".join(_match_keywords(c))
    )
    candidates_schema = all_cols.loc[all_cols["matched_keywords"] != ""].copy()

    # Always pull identity + candidate columns for coverage
    name_map = {str(n).lower(): str(n) for n in all_cols["column_name"]}
    id_cols_actual = [
        name_map[c]
        for c in ("city_name", "county_name", "district_name", "city_id", "county_id", "district_id")
        if c in name_map
    ]
    cand_cols_actual = [str(c) for c in candidates_schema["column_name"].tolist()]
    fetch_cols = list(dict.fromkeys(id_cols_actual + cand_cols_actual))

    print(f"Fetching {args.city} rows from {table}...")
    if fetch_cols:
        demo = fetch_demographics(engine, table, city=args.city, columns=fetch_cols)
    else:
        # still fetch to report row counts even if no candidates
        demo = fetch_demographics(engine, table, city=args.city, columns=id_cols_actual or None)

    # normalize column access
    col_lookup = {c.lower(): c for c in demo.columns}
    if "county_name" not in demo.columns and "county_name" in col_lookup:
        demo = demo.rename(columns={col_lookup["county_name"]: "county_name"})

    n_kocaeli = int(len(demo))
    bmask = _county_mask(demo, args.county)
    n_basiskele = int(bmask.sum())
    print(f"rows: city={n_kocaeli}, {args.county}={n_basiskele}")
    print(f"income/SES candidate columns: {len(cand_cols_actual)}")

    # value type + distinct on Başiskele
    candidate_rows = []
    for _, sch in candidates_schema.iterrows():
        col = str(sch["column_name"])
        series = demo[col] if col in demo.columns else pd.Series(dtype=object)
        vtype = _classify_value_type(series, str(sch.get("sql_type") or ""))
        b_series = series.loc[bmask] if col in demo.columns else series
        candidate_rows.append(
            {
                "column_name": col,
                "sql_type": sch.get("sql_type"),
                "nullable": sch.get("nullable"),
                "matched_keywords": sch.get("matched_keywords"),
                "value_type": vtype,
                "distinct_non_null_all": int(series.dropna().nunique()) if col in demo.columns else 0,
                "distinct_non_null_basiskele": int(b_series.dropna().nunique())
                if col in demo.columns
                else 0,
            }
        )
    candidates = pd.DataFrame(candidate_rows)

    basiskele_cov = coverage_for_scope(
        demo, cand_cols_actual, scope_name="basiskele", mask=bmask
    )
    kocaeli_cov = coverage_for_scope(demo, cand_cols_actual, scope_name="kocaeli", mask=None)
    dist = value_distribution(demo, candidates, basiskele_mask=bmask) if not candidates.empty else pd.DataFrame()
    usage = safe_usage_check(candidates) if not candidates.empty else pd.DataFrame(
        columns=[
            "column_name",
            "matched_keywords",
            "strong_income_ses_keyword",
            "value_type",
            "sql_type",
            "in_v18_v19_safe_base",
            "in_v18_v19_safe_categorical",
            "in_v18_v19_full_extra",
            "mapped_demo_feature",
            "safe_usage_status",
            "note",
        ]
    )

    _write_csv(candidates, out_dir / "income_ses_column_candidates.csv")
    _write_csv(basiskele_cov, out_dir / "basiskele_income_ses_coverage.csv")
    _write_csv(kocaeli_cov, out_dir / "kocaeli_income_ses_coverage.csv")
    _write_csv(dist, out_dir / "income_ses_value_distribution.csv")
    _write_csv(usage, out_dir / "safe_demographics_usage_check.csv")
    write_summary(
        out_dir / "summary.md",
        table=table,
        city=args.city,
        county=args.county,
        all_cols=all_cols,
        candidates=candidates,
        basiskele_cov=basiskele_cov,
        kocaeli_cov=kocaeli_cov,
        usage=usage,
        n_basiskele=n_basiskele,
        n_kocaeli=n_kocaeli,
    )

    # also dump full schema for traceability
    _write_csv(all_cols, out_dir / "_all_columns_introspected.csv")

    print("Wrote:")
    for name in (
        "income_ses_column_candidates.csv",
        "basiskele_income_ses_coverage.csv",
        "kocaeli_income_ses_coverage.csv",
        "income_ses_value_distribution.csv",
        "safe_demographics_usage_check.csv",
        "summary.md",
    ):
        print(f"  - {out_dir / name}")

    if not usage.empty and (
        usage["safe_usage_status"].isin(
            ["not_used_by_safe_demographics", "full_mode_only"]
        )
    ).any():
        print("NOTE: income/ses exists but not used by safe demographics (see summary.md)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
