# Housing-Listing-Price-Prediction v3 — Strict Standard Flat

**Era:** V1 (`v1/source_versions/v3/`)  
**Status:** Historical. Builds a more homogeneous “standard flat” subset to chase higher R².

## Main difference vs prior

Earlier packages only dropped duplex/luxury-style titles. This version applies a
**stricter** filter:

- Drop titles containing villa / detached / garden / duplex / triplex / luxury terms
- Keep `room_count` in `{1+1, 2+1, 3+1, 4+1}` only
- Drop `gross_m2 < 45` or `gross_m2 > 220`
- Drop villa / detached `real_estate_type`

## Models

- Ridge, ElasticNet
- GradientBoosting, HistGradientBoosting
- ExtraTrees, RandomForest
- CatBoost (native categorical) if installed
- R²-focused GradientBoosting tuning

## CatBoost

```bash
pip install catboost
```

## How to run

Place cleaned CSV at:

```text
data/input/listing_dataset_cleaned.csv
```

```bash
python src/train_v3_strict_standard_flat.py
python src/predict_v3.py
```

## Important outputs

```text
reports/model_metrics_v3.json
reports/model_comparison_v3.csv
artifacts/best_model_v3_by_r2.joblib
data/output/strict_standard_flat_dataset.csv
data/output/removed_by_strict_standard_flat_filter.csv
data/output/dataset_marked_with_strict_filter_reason.csv
data/output/<best_model>_cv_predictions.csv
data/output/<best_model>_top_50_errors.csv
data/output/<best_model>_error_by_district.csv
data/output/<best_model>_error_by_building_age_group.csv
data/output/<best_model>_error_by_m2_group.csv
```

## Notes

Higher R² is the goal, but row count can shrink. If too many listings are
removed, R² may worsen — loosen filters if that happens.
