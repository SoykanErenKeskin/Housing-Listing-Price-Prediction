# Housing-Listing-Price-Prediction v0 — İzmit Thesis Pilot

**Era:** V1 / thesis pilot (`v1/source_versions/v1/`)  
**Status:** Historical baseline package.

This package trains a baseline housing **₺/m²** model using **listing-portal sources only
listing fields**. No external market sources:

- no external demographics vendor
- no `trend_observed`
- no `trend_projection`
- no `floor_segments`

## Goals

- Stand up a 500–1000 listing pilot for the İzmit county study
- Produce a clean starting point for a thesis “listing-feature valuation model”
- Re-train later on larger data with the same pipeline

## Target variable

The model predicts `unit_price_gross`:

`unit_price_gross = price / gross_m2`

Total price:

`predicted_total_price = predicted_unit_price_gross * gross_m2`

Both models train on `log1p(unit_price_gross)` and invert back to ₺/m² at predict time.

## Models

1. `RidgeCV` — closer to classical multiple regression; more interpretable; easier thesis narrative and mobile embed
2. `GradientBoostingRegressor` — more flexible; captures nonlinearities; useful as a performance baseline

## Main features

Numeric:
`gross_m2`, `net_m2`, `building_age`, `floor_num`, `total_floors`, `bathroom_count`,
`open_area_m2`, `net_gross_ratio`, `has_open_area`

Categorical:
`real_estate_type`, `room_count`, `floor_segment`, `heating`, `kitchen`, `balcony`,
`elevator`, `parking`, `furnished`, `usage_status`, `site_inside`, `credit_eligible`,
`energy_certificate`, `deed_status`, `seller_type`, `barter`, `city`, `county`, `district`

## Layout

```text
data/raw/listing_dataset.csv
src/train_pure_listing_model.py
src/predict_pure_listing_price.py
artifacts/pure_listing_ridge_log_unit_price_v0.joblib
artifacts/pure_listing_gradient_boosting_log_unit_price_v0.joblib
artifacts/model_metrics_v0.json
artifacts/cv_predictions_v0.csv
artifacts/ridge_coefficients_v0.csv
artifacts/gb_feature_importance_v0.csv
sample_input.json
```

Related: `listing_outlier_cleaning_pipeline/` cleans CSVs before training.

## Re-train

Place the latest CSV at:

```text
data/raw/listing_dataset.csv
```

Then:

```bash
python src/train_pure_listing_model.py
```

## Predict smoke

```bash
python src/predict_pure_listing_price.py
```

## Thesis note

This is a **pilot**, not a final valuation engine. Scores from ~100-row smoke
data are pipeline checks only. With 500–1000 listings, district / room / age /
m² / floor-segment relationships become more meaningful.

Suggested framing:

> Within this study, a ₺/m² prediction model was built from listing attributes
> only, without external market indices. The goal is to examine how listing-level
> variables explain housing prices and to develop a pilot valuation approach for
> İzmit county.
