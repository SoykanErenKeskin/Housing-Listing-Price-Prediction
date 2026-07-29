"""Safe DB smoke checks for the ml_pipeline Neon role (no secrets printed).

Usage (from repo root):
  python shared_scripts/smoke_ml_pipeline_db.py

Performs only lightweight SELECT / permission probes. Does not train models.
Any attempted write is rolled back.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from canonical_db import (  # noqa: E402
    CANONICAL_NEIGHBORHOOD_DEMOGRAPHICS,
    CANONICAL_NEIGHBORHOOD_MODEL_SCORES,
    CANONICAL_PRICE_OBSERVATIONS,
    CANONICAL_RENTAL_LISTINGS,
    CANONICAL_SALE_LISTINGS,
)
from db_url import (  # noqa: E402
    EXPECTED_DB_USER,
    create_sqlalchemy_engine,
    safe_database_label,
    sanitize_db_error,
)
from env_loader import load_root_env  # noqa: E402


def _print(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    try:
        load_root_env(start=ROOT)
    except Exception as exc:
        _print(f"FAIL env: {sanitize_db_error(exc)}")
        return 2

    label = safe_database_label()
    _print(f"target={label}")

    try:
        engine = create_sqlalchemy_engine()
    except Exception as exc:
        _print(f"FAIL engine: {sanitize_db_error(exc)}")
        return 3

    results: dict[str, str] = {}

    try:
        with engine.connect() as conn:
            row = conn.exec_driver_sql(
                "SELECT current_user, current_database()"
            ).one()
            current_user, current_database = str(row[0]), str(row[1])
            results["current_user"] = current_user
            results["current_database"] = current_database
            _print(f"current_user={current_user}")
            _print(f"current_database={current_database}")
            if current_user != EXPECTED_DB_USER:
                _print(
                    f"FAIL: expected current_user={EXPECTED_DB_USER!r}, "
                    f"got {current_user!r}"
                )
                return 4

            one = conn.exec_driver_sql("SELECT 1").scalar()
            results["select_1"] = str(one)
            _print(f"select_1={one}")

            for table in (
                CANONICAL_SALE_LISTINGS,
                CANONICAL_RENTAL_LISTINGS,
                CANONICAL_PRICE_OBSERVATIONS,
                CANONICAL_NEIGHBORHOOD_DEMOGRAPHICS,
                CANONICAL_NEIGHBORHOOD_MODEL_SCORES,
            ):
                try:
                    with conn.begin_nested():
                        conn.exec_driver_sql(f"SELECT 1 FROM {table} LIMIT 1").first()
                    results[f"read:{table}"] = "ok"
                    _print(f"read:{table}=ok")
                except Exception as exc:
                    results[f"read:{table}"] = f"denied_or_missing:{sanitize_db_error(exc)}"
                    _print(f"read:{table}=DENIED_OR_MISSING ({sanitize_db_error(exc)})")

            # Legacy public / bare names should be gone after cutover.
            for table in (
                "trend_observed",
                "district_demographics",
                "public.sale_listings",
            ):
                try:
                    with conn.begin_nested():
                        conn.exec_driver_sql(f"SELECT 1 FROM {table} LIMIT 1").first()
                    results[f"legacy_read:{table}"] = "unexpectedly_present"
                    _print(f"legacy_read:{table}=UNEXPECTEDLY_PRESENT")
                except Exception as exc:
                    results[f"legacy_read:{table}"] = f"missing_ok:{sanitize_db_error(exc)}"
                    _print(f"legacy_read:{table}=missing_ok")

            try:
                with conn.begin_nested():
                    row = conn.exec_driver_sql(
                        "SELECT province, district, neighborhood "
                        f"FROM {CANONICAL_SALE_LISTINGS} LIMIT 1"
                    ).first()
                results["cols:sale_loc"] = "ok" if row is not None else "empty"
                _print(f"cols:sale_loc={results['cols:sale_loc']}")
            except Exception as exc:
                results["cols:sale_loc"] = f"fail:{sanitize_db_error(exc)}"
                _print(f"cols:sale_loc=FAIL ({sanitize_db_error(exc)})")

            try:
                with conn.begin_nested():
                    conn.exec_driver_sql(
                        "SELECT province_name, district_name, neighborhood_name, "
                        "neighborhood_id "
                        f"FROM {CANONICAL_PRICE_OBSERVATIONS} LIMIT 1"
                    ).first()
                results["cols:trend_loc"] = "ok"
                _print("cols:trend_loc=ok")
            except Exception as exc:
                results["cols:trend_loc"] = f"fail:{sanitize_db_error(exc)}"
                _print(f"cols:trend_loc=FAIL ({sanitize_db_error(exc)})")

            probe_table = None
            for table in (CANONICAL_SALE_LISTINGS,):
                if results.get(f"read:{table}") == "ok":
                    probe_table = table
                    break

            if probe_table:
                try:
                    with conn.begin_nested():
                        conn.exec_driver_sql(
                            f"UPDATE {probe_table} SET classified_id = classified_id "
                            "WHERE false"
                        )
                    results[f"write_probe:{probe_table}"] = "unexpectedly_allowed"
                    _print(
                        f"write_probe:{probe_table}=UNEXPECTEDLY_ALLOWED "
                        "(savepoint rolled back; prefer read-only on sources)"
                    )
                except Exception as exc:
                    results[f"write_probe:{probe_table}"] = (
                        f"denied:{sanitize_db_error(exc)}"
                    )
                    _print(
                        f"write_probe:{probe_table}=DENIED "
                        f"({sanitize_db_error(exc)})"
                    )
                _print("write_probe_txn=ROLLBACK_SAVEPOINT")
            else:
                _print("write_probe=skipped (no readable listing table)")

            conn.commit()

    except Exception as exc:
        _print(f"FAIL query: {sanitize_db_error(exc)}")
        return 5

    if results.get("current_user") != EXPECTED_DB_USER:
        return 4
    if results.get(f"read:{CANONICAL_SALE_LISTINGS}") != "ok":
        _print("FAIL: canonical market.sale_listings not readable")
        return 6
    if results.get("cols:sale_loc", "").startswith("fail"):
        _print("FAIL: sale location columns missing")
        return 7
    _print("smoke=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
