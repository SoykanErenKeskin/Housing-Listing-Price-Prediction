"""Unit tests for canonical_db relation resolution and DF aliases (no live DB)."""

from __future__ import annotations

import unittest

import pandas as pd

from canonical_db import (
    CANONICAL_NEIGHBORHOOD_DEMOGRAPHICS,
    CANONICAL_PRICE_OBSERVATIONS,
    CANONICAL_RENTAL_LISTINGS,
    CANONICAL_SALE_LISTINGS,
    alias_demographics_frame,
    alias_listing_frame,
    alias_trend_frame,
    location_level_to_canonical,
    resolve_relation,
    sql_relation,
)


class ResolveRelationTests(unittest.TestCase):
    def test_legacy_sale_aliases(self) -> None:
        for name in (
            "sale_listings",
            "public.sale_listings",
            "market.sale_listings",
        ):
            self.assertEqual(resolve_relation(name), CANONICAL_SALE_LISTINGS)

    def test_legacy_rental_and_trend_and_demo(self) -> None:
        self.assertEqual(resolve_relation("rental_listings"), CANONICAL_RENTAL_LISTINGS)
        self.assertEqual(resolve_relation("trend_observed"), CANONICAL_PRICE_OBSERVATIONS)
        self.assertEqual(
            resolve_relation("district_demographics"),
            CANONICAL_NEIGHBORHOOD_DEMOGRAPHICS,
        )

    def test_rejects_unknown_and_injection(self) -> None:
        with self.assertRaises(ValueError):
            resolve_relation("not_a_real_table")
        with self.assertRaises(ValueError):
            resolve_relation("market.sale_listings; DROP TABLE x")
        with self.assertRaises(ValueError):
            resolve_relation("market.evil_table")

    def test_sql_relation_quoted(self) -> None:
        self.assertEqual(sql_relation("sale_listings"), '"market"."sale_listings"')


class AliasFrameTests(unittest.TestCase):
    def test_listing_alias(self) -> None:
        df = pd.DataFrame(
            {
                "province": ["Kocaeli"],
                "district": ["İzmit"],
                "neighborhood": ["Erenler"],
                "price": [1],
            }
        )
        out = alias_listing_frame(df)
        self.assertListEqual(list(out.columns), ["city", "county", "district", "price"])
        self.assertEqual(out.loc[0, "city"], "Kocaeli")
        self.assertEqual(out.loc[0, "county"], "İzmit")
        self.assertEqual(out.loc[0, "district"], "Erenler")

    def test_trend_alias(self) -> None:
        df = pd.DataFrame(
            {
                "province_name": ["Kocaeli"],
                "district_name": ["Başiskele"],
                "neighborhood_name": ["Kullar"],
                "neighborhood_id": [1],
            }
        )
        out = alias_trend_frame(df)
        self.assertEqual(out.loc[0, "city_name"], "Kocaeli")
        self.assertEqual(out.loc[0, "county_name"], "Başiskele")
        self.assertEqual(out.loc[0, "district_name"], "Kullar")
        self.assertEqual(out.loc[0, "district_id"], 1)

    def test_demo_alias(self) -> None:
        df = pd.DataFrame(
            {
                "province_id": [41],
                "district_id": [100],
                "neighborhood_id": [200],
                "province_name": ["Kocaeli"],
                "district_name": ["İzmit"],
                "neighborhood_name": ["Erenler"],
            }
        )
        out = alias_demographics_frame(df)
        self.assertEqual(out.loc[0, "city_id"], 41)
        self.assertEqual(out.loc[0, "county_id"], 100)
        self.assertEqual(out.loc[0, "district_id"], 200)


class LocationLevelTests(unittest.TestCase):
    def test_legacy_to_canonical(self) -> None:
        self.assertEqual(location_level_to_canonical("city"), "province")
        self.assertEqual(location_level_to_canonical("county"), "district")
        self.assertEqual(location_level_to_canonical("district"), "neighborhood")
        self.assertEqual(location_level_to_canonical("neighborhood"), "neighborhood")


if __name__ == "__main__":
    unittest.main()
