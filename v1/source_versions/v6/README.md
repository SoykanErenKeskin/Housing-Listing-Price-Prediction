# Housing-Listing-Price-Prediction v6 — Sales Model with Rental Market Features

**Era:** V1 (`v1/source_versions/v6/`)  
**Status:** Historical. Re-trains the sales ₺/m² model with district-level rent
signals derived from rental listings.

## Logic

Sales target remains `unit_price_gross`. Rental data supplies **external market**
features (not derived from sale price → no sale-target leakage), including:

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
src/predict_v6.py
```

## How to run

```bash
python src/train_v6_sales_with_rental_features.py
```

Main outputs:

```text
reports/model_metrics_v6_rental.json
reports/model_comparison_v6_rental.csv
reports/rental_feature_report_v6.csv
data/output/sales_with_rental_features_preview.csv
artifacts/best_model_v6_rental_by_r2.joblib
```

## Leakage note

`estimated_monthly_rent_gross` is safe as a model input when it comes only from
rent ₺/m² medians × the listing’s `gross_m2`.

Do **not** use `sale_price / rent` or `amortization_months` as model inputs —
those leak the sale target. Compute amortization after prediction for reporting.

## Expectation

Rent features help encode neighborhood attractiveness and market level. MAPE/MAE
gains depend on rent coverage by district / room slice.
