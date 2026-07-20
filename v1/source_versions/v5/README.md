# Housing-Listing-Price-Prediction v5.2 — Detail Selection + Scores + Ensemble + CatBoost

**Era:** V1 (`v1/source_versions/v5/`)  
**Status:** Historical. Bundles post-v4 ideas into one training package.

## Contents

1. **Rare detail feature selection**  
   `front_*`, `view_*`, `transport_*`, `near_*`, `out_*`, `in_*`, `subtype_*`  
   Binary detail features with `ones < 20` are excluded from the model.

2. **Detail score features**  
   `front_score`, `view_score`, `transport_score`, `nearby_score`,
   `inside_quality_score`, `outside_quality_score`, `premium_detail_score`,
   `site_security_score`, `accessibility_score`

3. **District interaction features**  
   `district_age_group`, `district_m2_group`, `district_room_count`,
   `district_view_group`, `district_transport_group`, `district_quality_group`,
   `district_site_inside`

4. **Models**  
   Ridge, ElasticNet, GradientBoosting, HistGradientBoosting, ExtraTrees,
   RandomForest, R²-tuned GradientBoosting, manual CatBoost CV, weighted
   ensemble, grid-best ensemble

5. **Reports**  
   Model comparison, detail coverage, ensemble weights, top-50 errors,
   error slices by district / room / m² / age / detail

## How to run

```text
data/input/listing_dataset_cleaned.csv
```

```bash
python src/train_v5_2_detail_selection_ensemble_catboost.py
python src/predict_v5_2.py
```

## CatBoost

If CatBoost is missing, the script continues and records `catboost_not_available`
in the report:

```bash
pip install catboost
```

## Main outputs

```text
reports/model_metrics_v5.json
reports/model_comparison_v5.csv
reports/detail_feature_coverage_v5.csv
reports/model_aux_v5.json
artifacts/best_model_v5_by_r2.joblib
data/output/<best_model>_cv_predictions.csv
data/output/<best_model>_top_50_errors.csv
```

## Runtime note

Expect a long run — CatBoost, ExtraTrees, and tuning dominate wall time.

## v5.1 bug fix

- Score columns (`front_score`, `view_score`, …) are no longer re-selected as raw
  detail binary features.
- Duplicate feature lists are cleaned automatically.
- Manual CatBoost CV is guarded against duplicate DataFrame column errors.
- Fixes `pd.to_numeric(...): arg must be a list, tuple, 1-d array, or Series`.

## v5.2 bug fix

In v5.1, non-CatBoost models could fail with:

```text
A given column is not a column of the dataframe
```

Cause: `RareBinaryDropper` physically deleted rare detail columns in some folds
while `ColumnTransformer` still expected them.

v5.2 fix: rare detail columns stay in the frame but are zeroed, so Ridge /
ElasticNet / GB / HistGB / ExtraTrees / RF / tuning / ensemble keep working.
