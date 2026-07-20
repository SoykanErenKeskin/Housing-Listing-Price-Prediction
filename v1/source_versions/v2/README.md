# Housing-Listing-Price-Prediction v1 — Feature Engineering + Model Tuning

**Era:** V1 (`v1/source_versions/v2/` — historical folder name “v1” package)  
**Status:** Historical. listing-portal-only; no external demographics / trend / floor external refs.

## What this version adds

- Feature engineering
- KFold-safe target encoding
- Automatic drop of empty / single-value columns
- Multi-model comparison
- Light Gradient Boosting tuning
- Error-analysis reports

## How to run

Place the latest cleaned CSV at:

```text
data/input/listing_dataset_cleaned.csv
```

Train:

```bash
python src/train_v1_feature_engineering.py
```

Predict smoke:

```bash
python src/predict_v1.py
```

## Important outputs

```text
reports/model_metrics_v1.json
reports/model_comparison_v1.csv
artifacts/best_model.joblib
data/output/feature_engineered_dataset_preview.csv
data/output/<best_model>_cv_predictions.csv
data/output/<best_model>_top_50_errors.csv
data/output/<best_model>_error_by_district.csv
data/output/<best_model>_error_by_room_count.csv
data/output/<best_model>_error_by_building_age_group.csv
data/output/<best_model>_error_by_m2_group.csv
```

## Notes

Runtime is longer than v0; tuning and ExtraTrees add wall time. Use this package
when you want richer features and model bake-offs on cleaned listing CSVs only.
