"""Small DB helpers for root shared analysis scripts (no training logic)."""

from __future__ import annotations

import os
import re
from typing import Any, Iterable, Sequence

import pandas as pd

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_IDENT_QUALIFIED = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$"
)

# Preferred columns for inventory analysis (requested if present).
# Includes both legacy pipeline aliases and canonical physical names.
PREFERRED_COLUMNS: tuple[str, ...] = (
    "classified_id",
    "source_url",
    "listing_purpose",
    "source_site",
    "city",
    "county",
    "district",
    "province",
    "neighborhood",
    "title",
    "site_name",
    # sale price candidates
    "price",
    "sale_price",
    "listing_price",
    "price_tl",
    "total_price",
    "raw_price",
    "unit_price_gross",
    # rental price candidates
    "rent_price",
    "rental_price",
    "monthly_rent",
    "rent",
    "rent_per_m2_gross",
    "rent_per_m2_net",
    "gross_m2",
    "net_m2",
    "room_count",
    "building_age",
    "floor_num",
    "total_floors",
    "heating",
    "kitchen",
    "balcony",
    "elevator",
    "parking",
    "furnished",
    "usage_status",
    "site_inside",
    "credit_eligible",
    "deed_status",
    "seller_type",
    "lat",
    "latitude",
    "lon",
    "longitude",
    "street_name",
    "address_text",
    "location_precision",
    "location_source",
    "location_backfill_status",
    "created_at",
    "updated_at",
    "saved_at",
    "scraped_at",
    "listing_date",
)

SALE_PRICE_CANDIDATES: tuple[str, ...] = (
    "price",
    "sale_price",
    "listing_price",
    "price_tl",
    "total_price",
    "raw_price",
)

RENTAL_PRICE_CANDIDATES: tuple[str, ...] = (
    "price",
    "rent_price",
    "rental_price",
    "monthly_rent",
    "rent",
    "listing_price",
    "price_tl",
    "total_price",
    "raw_price",
)

RENTAL_UNIT_PRICE_CANDIDATES: tuple[str, ...] = (
    "rent_per_m2_gross",
    "rent_per_m2_net",
    "unit_price_gross",
)

SALE_UNIT_PRICE_CANDIDATES: tuple[str, ...] = (
    "unit_price_gross",
)

# Logical fields satisfied by any of the aliases.
_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("lat", "latitude"),
    ("lon", "longitude"),
    SALE_PRICE_CANDIDATES,
    RENTAL_PRICE_CANDIDATES,
)


def resolve_first_present_column(df_or_cols, candidates: Sequence[str]) -> str | None:
    """Return first candidate present in a DataFrame or column-name sequence."""
    if hasattr(df_or_cols, "columns"):
        available = {str(c).lower(): str(c) for c in df_or_cols.columns}
    else:
        available = {str(c).lower(): str(c) for c in df_or_cols}
    for cand in candidates:
        key = cand.lower()
        if key in available:
            return available[key]
    return None


def validate_table_name(table: str) -> str:
    """Resolve legacy/bare names to allowlisted ``schema.table`` via canonical_db."""
    from canonical_db import resolve_relation

    return resolve_relation(table)


def get_database_url(database_url_override: str | None = None) -> str:
    """Return a driver-ready URL (``DATABASE_URL`` + ``DB_ROLE_PASSWORD``).

    Never log the return value — it contains the role password.
    """
    from db_url import resolve_database_url

    return resolve_database_url(database_url_override=database_url_override)


def create_engine(db_url: str | None = None):
    """Create SQLAlchemy engine via the central URL resolver."""
    from db_url import create_sqlalchemy_engine

    # ``db_url`` is treated as an optional password-less template override
    # (e.g. CLI ``--db-url``), not a fully credentialed DSN.
    return create_sqlalchemy_engine(database_url_override=db_url or None)


def list_table_columns(engine, table: str) -> list[str]:
    """Return lowercase column names present on ``table`` (schema-qualified)."""
    from sqlalchemy import inspect

    relation = validate_table_name(table)
    schema, name = relation.split(".", 1)
    insp = inspect(engine)
    cols = []
    try:
        cols = insp.get_columns(name, schema=schema)
    except Exception:
        # Fallback: search schemas (should be rare for allowlisted relations).
        for sch in insp.get_schema_names():
            try:
                cols = insp.get_columns(name, schema=sch)
                if cols:
                    break
            except Exception:
                continue
    return [str(c.get("name", "")).lower() for c in cols if c.get("name")]


def pick_available_columns(
    available: Sequence[str],
    preferred: Iterable[str] = PREFERRED_COLUMNS,
) -> tuple[list[str], list[str]]:
    """Return (selected, missing_preferred). Alias pairs count as present if any match."""
    avail_set = {a.lower() for a in available}
    preferred_list = list(preferred)
    selected: list[str] = []
    for col in preferred_list:
        cl = col.lower()
        if cl in avail_set:
            for a in available:
                if a.lower() == cl:
                    selected.append(a)
                    break

    # de-dupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for c in selected:
        k = c.lower()
        if k not in seen:
            seen.add(k)
            out.append(c)

    # missing: preferred names not covered, treating alias groups as one logical field
    covered = {c.lower() for c in out}
    for group in _ALIAS_GROUPS:
        if any(g in covered for g in group):
            covered.update(g.lower() for g in group)

    # Prefer reporting a single logical missing for price families
    logical_missing_skip = set()
    for group in (SALE_PRICE_CANDIDATES, RENTAL_PRICE_CANDIDATES):
        if any(g.lower() in covered for g in group):
            logical_missing_skip.update(g.lower() for g in group)

    missing: list[str] = []
    for col in preferred_list:
        cl = col.lower()
        skip = False
        for group in _ALIAS_GROUPS:
            if cl in {g.lower() for g in group} and any(g.lower() in covered for g in group):
                skip = True
                break
        if skip or cl in logical_missing_skip:
            continue
        if cl not in covered:
            missing.append(col)

    return out, missing


def introspect_price_columns(available: Sequence[str], purpose: str) -> dict[str, Any]:
    """Schema introspection helper for sale/rental price columns."""
    avail = [str(c) for c in available]
    if purpose == "sale":
        price_col = resolve_first_present_column(avail, SALE_PRICE_CANDIDATES)
        unit_col = resolve_first_present_column(avail, SALE_UNIT_PRICE_CANDIDATES)
    else:
        price_col = resolve_first_present_column(avail, RENTAL_PRICE_CANDIDATES)
        unit_col = resolve_first_present_column(avail, RENTAL_UNIT_PRICE_CANDIDATES)
    return {
        "purpose": purpose,
        "available_columns": sorted({a.lower() for a in avail}),
        "price_column": price_col,
        "unit_price_column": unit_col,
        "price_candidates_checked": list(
            SALE_PRICE_CANDIDATES if purpose == "sale" else RENTAL_PRICE_CANDIDATES
        ),
        "unit_candidates_checked": list(
            SALE_UNIT_PRICE_CANDIDATES if purpose == "sale" else RENTAL_UNIT_PRICE_CANDIDATES
        ),
    }


def fetch_listings(
    engine,
    *,
    table: str,
    purpose: str,
    city: str,
    county: str | None = None,
    district: str | None = None,
    source_site: str = "listing_portal",
    columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch listings with graceful column selection.

    SQL uses canonical ``province`` / ``district`` (ilçe) / ``neighborhood``.
    Returned frame is aliased to legacy ``city`` / ``county`` / ``district``
    for downstream inventory scripts that still expect the old names.

    Returns (dataframe, meta) where meta includes missing_columns / sql notes.
    """
    from sqlalchemy import text

    from canonical_db import alias_listing_frame, sql_relation

    relation = validate_table_name(table)
    rel_sql = sql_relation(relation)
    available = list_table_columns(engine, relation)
    if columns is None:
        selected, missing = pick_available_columns(available)
    else:
        selected, missing = pick_available_columns(available, preferred=columns)

    meta: dict[str, Any] = {
        "table": relation,
        "available_column_count": len(available),
        "available_columns": available,
        "selected_columns": selected,
        "missing_columns": missing,
        "price_introspection": introspect_price_columns(available, purpose),
    }

    if not selected:
        select_sql = "*"
        meta["note"] = "no preferred columns found; using SELECT *"
    else:
        select_sql = ", ".join(selected)

    # Prefer canonical location columns when present; fall back to legacy names.
    avail_l = {a.lower() for a in available}
    province_col = "province" if "province" in avail_l else "city"
    district_ilce_col = "district" if "province" in avail_l else "county"
    neighborhood_col = "neighborhood" if "neighborhood" in avail_l else "district"

    where = [
        f"lower(coalesce({province_col}, '')) = lower(:city)",
        "lower(coalesce(listing_purpose, '')) = lower(:purpose)",
    ]
    params: dict[str, Any] = {"city": city, "purpose": purpose}
    if source_site:
        where.append("lower(coalesce(source_site, 'listing_portal')) = lower(:source_site)")
        params["source_site"] = source_site
    if county:
        where.append(f"{district_ilce_col} = :county")
        params["county"] = county
    if district:
        # Caller "district" historically means mahalle.
        where.append(f"{neighborhood_col} = :district")
        params["district"] = district

    order = ""
    if "saved_at" in avail_l and "updated_at" in avail_l:
        order = "ORDER BY saved_at DESC NULLS LAST, updated_at DESC NULLS LAST"
    elif "updated_at" in avail_l:
        order = "ORDER BY updated_at DESC NULLS LAST"
    elif "created_at" in avail_l:
        order = "ORDER BY created_at DESC NULLS LAST"

    sql = text(
        f"""
        SELECT {select_sql}
        FROM {rel_sql}
        WHERE {' AND '.join(where)}
        {order}
        """
    )
    try:
        df = pd.read_sql(sql, engine, params=params)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to query table '{relation}' for purpose='{purpose}'. "
            f"Check DB connectivity and filters. Error: {exc}"
        ) from exc

    df = alias_listing_frame(df)
    meta["rows"] = int(len(df))
    return df, meta
