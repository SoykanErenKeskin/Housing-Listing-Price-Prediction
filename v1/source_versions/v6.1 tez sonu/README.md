# Housing-Listing-Price-Prediction v6.1 — Sales + Rental Features (Neon export fix)

**Era:** V1 (`v1/source_versions/v6.1 tez sonu/`)  
**Status:** Historical thesis-end snapshot of v6 with a Neon export hotfix.

Same goal as v6: train a sales `unit_price_gross` model using district-level rent
market features from rental listings (no sale-price leakage).

## Rent feature set

```text
district_rent_m2_median
district_rent_m2_mean
district_rent_m2_count
district_rent_m2_iqr
district_rent_m2_cv
district_room_rent_m2_median
district_room_rent_m2_count
district_m2_group_rent_m2_median
district_m2_group_rent_m2_count
county_rent_m2_median
county_rent_m2_count
estimated_rent_m2_gross
estimated_monthly_rent_gross
rent_feature_confidence
rent_feature_level
```

## Files

```text
data/input/sale_listings_dataset.csv
data/input/rental_listings.csv
src/train_v6_sales_with_rental_features.py
src/train_v6_1_sales_with_rental_features.py
src/predict_v6.py
```

## How to run (v6.1)

```bash
python src/train_v6_1_sales_with_rental_features.py
```

Main outputs (same family as v6):

```text
reports/model_metrics_v6_rental.json
reports/model_comparison_v6_rental.csv
reports/rental_feature_report_v6.csv
data/output/sales_with_rental_features_preview.csv
artifacts/best_model_v6_rental_by_r2.joblib
```

## Leakage note

`estimated_monthly_rent_gross` is OK when built from rent medians × `gross_m2`.
Do not feed `sale_price / rent` or amortization months into training.

## v6.1 fix

New Neon exports put helper detail binaries (`front_*`, `view_*`, `transport_*`,
`near_*`, `out_*`, `in_*`, `subtype_*`) inside a top-level `raw` JSON column
instead of flat columns.

v6.1:

- lifts detail binaries from `raw` JSON to top level
- also completes `building_age_raw`, `building_age_group`, `detail_*` from `raw`
- if no detail columns exist, writes an empty coverage report instead of
  `KeyError: 'ones'`
