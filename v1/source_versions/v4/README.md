# Housing-Listing-Price-Prediction v4 — Detail Features

**Era:** V1 (`v1/source_versions/v4/`)  
**Status:** Historical. Adds detail features collected by the helper on top of the
v1/v2-style pipeline.

Strict “standard flat” segment filtering is **off by default** so detail-feature
lift can be measured cleanly.

## New feature groups (auto-used)

```text
front_*
view_*
transport_*
near_*
out_*
in_*
subtype_*
```

Numeric aggregates:

```text
detail_selected_count
detail_quality_score
detail_front_count
detail_view_count
detail_transport_count
detail_near_count
detail_inside_count
detail_outside_count
detail_subtype_count
```

Raw categoricals:

```text
detail_cephe
detail_manzara
detail_konut_tipi
```

## How to run

Place the enriched cleaned CSV at:

```text
data/input/listing_dataset_cleaned.csv
```

```bash
python src/train_v4_detail_features.py
python src/predict_v4.py
```

## Important outputs

```text
reports/model_metrics_v4.json
reports/model_comparison_v4.csv
reports/detail_feature_coverage_v4.csv
artifacts/best_model_v4_by_r2.joblib
data/output/<best_model>_cv_predictions.csv
data/output/<best_model>_top_50_errors.csv
```

`detail_feature_coverage_v4.csv` is critical: it shows how often each detail
column is filled across listings.
