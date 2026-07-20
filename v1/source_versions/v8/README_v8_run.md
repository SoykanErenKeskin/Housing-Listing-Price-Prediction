# V8 DB Residual Pipeline

**Era:** V1 (`v1/source_versions/v8/`)  
**Status:** Historical. First DB-backed residual sales pipeline in this line.

Built on V7 with three deliberate changes:

1. CatBoost is **not** added.
2. Default target is **residual**: learn deviation from a location baseline, not
   raw ₺/m².
3. Stronger outlier cleaning: after basic IQR, filter by price / location-baseline
   ratio.

## Setup

```bash
pip install pandas numpy scikit-learn joblib matplotlib sqlalchemy psycopg2-binary python-dotenv
```

Put `DATABASE_URL` in the **repo-root** `.env` (do not recreate per-version secrets):

```env
DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/DB?sslmode=require
```

## Normal DB run

```bash
python train_v8_db_residual_pipeline.py --out outputs/v8_kocaeli
```

## Fast smoke

```bash
python train_v8_db_residual_pipeline.py --out outputs/v8_test --fast
```

## Target modes

Default:

```bash
--target-mode residual
```

Also available for comparison: `--target-mode log`, `--target-mode raw`.

## Outlier filter

On by default (`--location-outlier-filter`). Disable with
`--no-location-outlier-filter`.

Bounds:

```bash
--min-location-ratio 0.50 --max-location-ratio 1.90
```

Meaning: if listing ₺/m² is below ~0.50× or above ~1.90× its location baseline,
it is dropped from training and written to
`data/input/sales_removed_location_outliers_v8.csv`.

## Outputs

- `data/input/sales_training_table_v8.csv`
- `data/input/sales_removed_location_outliers_v8.csv`
- `data/output/oof_predictions_v8.csv`
- `reports/metrics_summary_v8.json`
- `reports/model_comparison_v8.csv`
- `reports/error_by_*_v8.csv`
- `reports/actual_vs_predicted_v8.png`
- `reports/residual_distribution_v8.png`
- `artifacts/ensemble_model_bundle_v8.joblib`

## Design note

V8 intentionally avoids title/text, photos, or premium signals users cannot
provide. The location baseline uses county, district, m² group, room count, and
app-available signals such as `trend_sale_m2` when present.
