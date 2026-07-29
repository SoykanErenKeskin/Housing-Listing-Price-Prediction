"""Canonical Neon/PostgreSQL relation names + artifact-compatible DF aliases.

Production DB contract: RELEASE_3_DATABASE_CANONICAL_NAMING_READY.

Physical DB hierarchy: province → district (ilçe) → neighborhood (mahalle).
Legacy ML pipeline / artifacts still expect: city → county → district (mahalle).

This module:
- Resolves legacy / bare table names to schema-qualified allowlisted relations
- Issues SQL against canonical physical columns
- Renames result frames to legacy city/county/district names for training & artifacts

Never interpolate arbitrary env table names into SQL; only allowlisted relations.
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Any, Sequence

import pandas as pd

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# ---------------------------------------------------------------------------
# Allowlist (schema-qualified)
# ---------------------------------------------------------------------------

CANONICAL_SALE_LISTINGS = "market.sale_listings"
CANONICAL_RENTAL_LISTINGS = "market.rental_listings"
CANONICAL_PRICE_OBSERVATIONS = "market.price_observations"
CANONICAL_PRICE_FORECASTS = "market.price_forecasts"
CANONICAL_VW_LATEST_PRICE_OBSERVATIONS = "market.vw_latest_price_observations"
CANONICAL_VW_LATEST_PRICE_FORECASTS = "market.vw_latest_price_forecasts"
CANONICAL_FLOOR_SEGMENTS = "market.floor_segments"
CANONICAL_NEIGHBORHOOD_DEMOGRAPHICS = "geo.neighborhood_demographics"
CANONICAL_NEIGHBORHOOD_LOCATIONS = "geo.neighborhood_locations"
CANONICAL_VW_LOCATION_REFERENCE = "geo.vw_location_reference"
CANONICAL_NEIGHBORHOOD_MODEL_SCORES = "analytics.neighborhood_model_scores"
CANONICAL_LOCATION_NARRATIVE_ANALYSES = "analytics.location_narrative_analyses"
CANONICAL_VW_LATEST_LOCATION_NARRATIVE = "analytics.vw_latest_location_narrative_analyses"

ALLOWED_RELATIONS: frozenset[str] = frozenset(
    {
        CANONICAL_SALE_LISTINGS,
        CANONICAL_RENTAL_LISTINGS,
        CANONICAL_PRICE_OBSERVATIONS,
        CANONICAL_PRICE_FORECASTS,
        CANONICAL_VW_LATEST_PRICE_OBSERVATIONS,
        CANONICAL_VW_LATEST_PRICE_FORECASTS,
        CANONICAL_FLOOR_SEGMENTS,
        CANONICAL_NEIGHBORHOOD_DEMOGRAPHICS,
        CANONICAL_NEIGHBORHOOD_LOCATIONS,
        CANONICAL_VW_LOCATION_REFERENCE,
        CANONICAL_NEIGHBORHOOD_MODEL_SCORES,
        CANONICAL_LOCATION_NARRATIVE_ANALYSES,
        CANONICAL_VW_LATEST_LOCATION_NARRATIVE,
        "geo.countries",
        "geo.provinces",
        "geo.districts",
        "geo.neighborhoods",
        "geo.neighborhood_geometry_crosswalk",
        "analytics.db_health_snapshots",
        "analytics.province_market_metrics",
        "analytics.district_market_metrics",
        "operations.fetch_profiles",
        "operations.fetch_requests",
        "operations.fetch_runs",
        "operations.fetch_run_items",
        "operations.location_fetch_skips",
        "operations.collector_state",
        "operations.collector_logs",
    }
)

# Legacy / bare names → canonical (lowercase keys)
_LEGACY_TO_CANONICAL: dict[str, str] = {
    # listings
    "sale_listings": CANONICAL_SALE_LISTINGS,
    "public.sale_listings": CANONICAL_SALE_LISTINGS,
    "market.sale_listings": CANONICAL_SALE_LISTINGS,
    "rental_listings": CANONICAL_RENTAL_LISTINGS,
    "public.rental_listings": CANONICAL_RENTAL_LISTINGS,
    "market.rental_listings": CANONICAL_RENTAL_LISTINGS,
    # trends
    "trend_observed": CANONICAL_PRICE_OBSERVATIONS,
    "public.trend_observed": CANONICAL_PRICE_OBSERVATIONS,
    "price_observations": CANONICAL_PRICE_OBSERVATIONS,
    "market.price_observations": CANONICAL_PRICE_OBSERVATIONS,
    "vw_latest_observed_trend": CANONICAL_VW_LATEST_PRICE_OBSERVATIONS,
    "public.vw_latest_observed_trend": CANONICAL_VW_LATEST_PRICE_OBSERVATIONS,
    "market.vw_latest_price_observations": CANONICAL_VW_LATEST_PRICE_OBSERVATIONS,
    "trend_projection": CANONICAL_PRICE_FORECASTS,
    "public.trend_projection": CANONICAL_PRICE_FORECASTS,
    "price_forecasts": CANONICAL_PRICE_FORECASTS,
    "market.price_forecasts": CANONICAL_PRICE_FORECASTS,
    "vw_latest_projection_trend": CANONICAL_VW_LATEST_PRICE_FORECASTS,
    "public.vw_latest_projection_trend": CANONICAL_VW_LATEST_PRICE_FORECASTS,
    "market.vw_latest_price_forecasts": CANONICAL_VW_LATEST_PRICE_FORECASTS,
    # demographics / geo
    "district_demographics": CANONICAL_NEIGHBORHOOD_DEMOGRAPHICS,
    "public.district_demographics": CANONICAL_NEIGHBORHOOD_DEMOGRAPHICS,
    "neighborhood_demographics": CANONICAL_NEIGHBORHOOD_DEMOGRAPHICS,
    "geo.neighborhood_demographics": CANONICAL_NEIGHBORHOOD_DEMOGRAPHICS,
    "locations": CANONICAL_NEIGHBORHOOD_LOCATIONS,
    "public.locations": CANONICAL_NEIGHBORHOOD_LOCATIONS,
    "neighborhood_locations": CANONICAL_NEIGHBORHOOD_LOCATIONS,
    "geo.neighborhood_locations": CANONICAL_NEIGHBORHOOD_LOCATIONS,
    "floor_segments": CANONICAL_FLOOR_SEGMENTS,
    "public.floor_segments": CANONICAL_FLOOR_SEGMENTS,
    "market.floor_segments": CANONICAL_FLOOR_SEGMENTS,
    # analytics
    "neighborhood_ml_scores": CANONICAL_NEIGHBORHOOD_MODEL_SCORES,
    "public.neighborhood_ml_scores": CANONICAL_NEIGHBORHOOD_MODEL_SCORES,
    "neighborhood_model_scores": CANONICAL_NEIGHBORHOOD_MODEL_SCORES,
    "analytics.neighborhood_model_scores": CANONICAL_NEIGHBORHOOD_MODEL_SCORES,
    "location_llm_analyses": CANONICAL_LOCATION_NARRATIVE_ANALYSES,
    "public.location_llm_analyses": CANONICAL_LOCATION_NARRATIVE_ANALYSES,
    "analytics.location_narrative_analyses": CANONICAL_LOCATION_NARRATIVE_ANALYSES,
}

DEFAULT_SALE_TABLE = CANONICAL_SALE_LISTINGS
DEFAULT_RENTAL_TABLE = CANONICAL_RENTAL_LISTINGS
DEFAULT_TREND_TABLE = CANONICAL_PRICE_OBSERVATIONS
DEFAULT_DEMOGRAPHICS_TABLE = CANONICAL_NEIGHBORHOOD_DEMOGRAPHICS


def env_table(env_key: str, default: str) -> str:
    """Resolve optional env override through the allowlist."""
    raw = (os.getenv(env_key) or "").strip()
    return resolve_relation(raw or default)


# ---------------------------------------------------------------------------
# Relation validation
# ---------------------------------------------------------------------------


def resolve_relation(name: str) -> str:
    """Map legacy/bare/canonical names to an allowlisted ``schema.table``."""
    raw = str(name or "").strip()
    if not raw:
        raise ValueError("Empty table/relation name.")

    key = raw.lower()
    if key in _LEGACY_TO_CANONICAL:
        return _LEGACY_TO_CANONICAL[key]

    # Already schema-qualified?
    if "." in raw:
        parts = raw.split(".")
        if len(parts) != 2:
            raise ValueError(f"Unsafe relation name: {name!r}")
        schema, table = parts
        if not _IDENT.match(schema) or not _IDENT.match(table):
            raise ValueError(f"Unsafe relation name: {name!r}")
        qualified = f"{schema}.{table}"
        if qualified.lower() in {a.lower() for a in ALLOWED_RELATIONS}:
            for allowed in ALLOWED_RELATIONS:
                if allowed.lower() == qualified.lower():
                    return allowed
        raise ValueError(
            f"Relation not in canonical allowlist: {name!r}. "
            f"Use one of: {', '.join(sorted(ALLOWED_RELATIONS))}"
        )

    if not _IDENT.match(raw):
        raise ValueError(f"Unsafe table name: {name!r}")

    raise ValueError(
        f"Unknown bare table name {name!r}. "
        "Use a schema-qualified canonical name (e.g. market.sale_listings) "
        "or a known legacy alias."
    )


def validate_table_name(name: str) -> str:
    """Backward-compatible alias for ``resolve_relation``."""
    return resolve_relation(name)


def sql_relation(name: str) -> str:
    """Return a safe schema-qualified identifier for interpolation into SQL text."""
    rel = resolve_relation(name)
    schema, table = rel.split(".", 1)
    return f'"{schema}"."{table}"'


# ---------------------------------------------------------------------------
# DataFrame aliases (canonical physical → legacy pipeline / artifact names)
# ---------------------------------------------------------------------------

_LISTING_RENAME = {
    "province": "city",
    "district": "county",
    "neighborhood": "district",
    "province_id": "city_id",
    "district_id": "county_id",
    "neighborhood_id": "district_id",
    "province_name": "city_name",
    "district_name": "county_name",
    "neighborhood_name": "district_name",
}

_TREND_RENAME = {
    "province_name": "city_name",
    "district_name": "county_name",
    "neighborhood_name": "district_name",
    "neighborhood_id": "district_id",
    "province_id": "city_id",
    "district_id": "county_id",
}

_DEMO_RENAME = {
    "province_id": "city_id",
    "district_id": "county_id",
    "neighborhood_id": "district_id",
    "province_name": "city_name",
    "district_name": "county_name",
    "neighborhood_name": "district_name",
    "neighborhood_slug": "district_slug",
}


def _rename_present(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cols = {c: mapping[c] for c in df.columns if c in mapping}
    if not cols:
        return df
    return df.rename(columns=cols)


def alias_listing_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map listing location columns to legacy city/county/district names."""
    return _rename_present(df, _LISTING_RENAME)


def alias_trend_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map price-observation columns to legacy city_name/county_name/district_*."""
    return _rename_present(df, _TREND_RENAME)


def alias_demographics_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map neighborhood demographics IDs/names to legacy city/county/district keys."""
    return _rename_present(df, _DEMO_RENAME)


# ---------------------------------------------------------------------------
# Fetch helpers (canonical SQL + legacy DF contract)
# ---------------------------------------------------------------------------


def fetch_listing_table(
    engine,
    table: str,
    purpose: str,
    city: str,
    limit: int | None = None,
    *,
    counties: Sequence[str] | None = None,
    county: str | None = None,
    source_site: str | None = None,
    default_counties: Sequence[str] | None = None,
    filter_source_site: bool | None = None,
) -> pd.DataFrame:
    """Fetch listings from a canonical market.* table.

    Filters use physical columns ``province`` / ``district`` (ilçe).
    Returned frame uses legacy ``city`` / ``county`` / ``district`` (mahalle).
    """
    from sqlalchemy import text

    rel = sql_relation(table)
    limit_clause = f" LIMIT {int(limit)}" if limit else ""

    where = [
        "lower(coalesce(province, '')) = lower(:city)",
        "lower(coalesce(listing_purpose, '')) = lower(:purpose)",
    ]
    params: dict[str, Any] = {"city": city, "purpose": purpose}

    county_list: list[str] | None = None
    if county is not None and str(county).strip():
        county_list = [str(county).strip()]
    elif counties is not None:
        county_list = [str(c).strip() for c in counties if str(c).strip()]
    elif default_counties is not None:
        county_list = [str(c).strip() for c in default_counties if str(c).strip()]

    if county_list:
        if len(county_list) == 1:
            where.append("district = :county")
            params["county"] = county_list[0]
        else:
            county_params = {f"c{i}": c for i, c in enumerate(county_list)}
            in_clause = ", ".join(f":c{i}" for i in range(len(county_list)))
            where.append(f"district IN ({in_clause})")
            params.update(county_params)

    use_source = filter_source_site
    if use_source is None:
        use_source = source_site is not None
    if use_source and source_site:
        where.append("lower(coalesce(source_site, :source_site)) = lower(:source_site)")
        params["source_site"] = source_site

    sql = text(
        f"""
        SELECT *
        FROM {rel}
        WHERE {' AND '.join(where)}
        ORDER BY saved_at DESC NULLS LAST, updated_at DESC NULLS LAST
        {limit_clause}
        """
    )
    df = pd.read_sql(sql, engine, params=params)
    return alias_listing_frame(df)


def fetch_latest_trend_table(
    engine,
    table: str,
    city: str,
    max_date: str | None = None,
) -> pd.DataFrame:
    """Latest non-projection price observations per district+neighborhood.

    SQL uses canonical province/district/neighborhood names; the returned frame
    is aliased to legacy city_name/county_name/district_name/district_id.
    """
    from sqlalchemy import text

    rel = sql_relation(table)
    date_filter = "AND property_date <= :max_date" if max_date else ""
    params: dict[str, Any] = {"city": city}
    if max_date:
        params["max_date"] = max_date

    sql = text(
        f"""
        WITH filtered AS (
            SELECT
                id,
                property_date,
                property_year,
                property_month,
                province_name AS city_name,
                district_name AS county_name,
                neighborhood_name AS district_name,
                neighborhood_id AS district_id,
                unit_price_for_sale,
                unit_price_for_rent,
                count_for_sale,
                count_for_rent,
                listing_period_for_sale,
                yield,
                price_change_sale,
                unit_price_sale_annual_change,
                projection_like,
                ROW_NUMBER() OVER (
                    PARTITION BY district_name, neighborhood_name
                    ORDER BY property_date DESC
                ) AS rn
            FROM {rel}
            WHERE lower(coalesce(province_name, '')) = lower(:city)
              AND coalesce(projection_like, false) = false
              {date_filter}
        )
        SELECT *
        FROM filtered
        WHERE rn = 1
        ORDER BY county_name, district_name
        """
    )
    return pd.read_sql(sql, engine, params=params)


def fetch_demographics_table(
    engine,
    table: str,
    city: str | None = None,
) -> pd.DataFrame:
    """Fetch neighborhood demographics; return legacy city_*/county_*/district_* keys."""
    from sqlalchemy import text

    rel = sql_relation(table)
    city_filter = "WHERE lower(coalesce(province_name, '')) = lower(:city)" if city else ""
    params = {"city": city} if city else {}
    sql = text(
        f"""
        SELECT *
        FROM {rel}
        {city_filter}
        """
    )
    try:
        df = pd.read_sql(sql, engine, params=params)
    except Exception as exc:
        warnings.warn(
            f"Demographics table could not be fetched; continuing without demographics. Error: {exc}"
        )
        return pd.DataFrame()
    return alias_demographics_frame(df)


def location_level_to_canonical(level: str) -> str:
    """Map legacy location_level labels to canonical vocabulary."""
    raw = (level or "").strip().lower()
    mapping = {
        "city": "province",
        "province": "province",
        "county": "district",
        "district": "neighborhood",  # legacy district == mahalle
        "neighborhood": "neighborhood",
        "ilce": "district",
        "ilçe": "district",
        "mahalle": "neighborhood",
    }
    if raw not in mapping:
        raise ValueError(f"Unknown location_level: {level!r}")
    return mapping[raw]
