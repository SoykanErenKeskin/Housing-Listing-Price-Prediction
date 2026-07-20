# V11 demographics pipeline

V11 builds on the V10 county-expert + segment-aware residual model and adds:

- PostgreSQL `district_demographics` loading through `DATABASE_URL`
- `--demographics-mode none|safe|full` with default `safe`
- automatic none/safe/full ablation with the same base pipeline
- county-level demographic aggregates
- neighborhood-vs-county demographic difference/ratio features
- anomaly exclusion before training with `--exclude-anomalies-threshold 25` by default

Recommended run:

```bash
python train_v11_demographics_pipeline.py \
  --out outputs/v11_kocaeli \
  --city Kocaeli \
  --counties "İzmit,Başiskele,Gölcük,Karamürsel" \
  --sale-table sale_listings \
  --rental-table rental_listings \
  --trend-table trend_observed \
  --demographics-table district_demographics \
  --demographics-mode safe \
  --run-demographics-ablation
```

Faster smoke run:

```bash
python train_v11_demographics_pipeline.py \
  --out outputs/v11_test \
  --fast \
  --limit-sale 800 \
  --limit-rental 800 \
  --demographics-mode safe \
  --no-run-demographics-ablation
```

Important outputs:

- `reports/metrics_summary_v11.json`
- `reports/metrics_demographics_ablation_v11.csv`
- `reports/demographic_feature_coverage_safe_v11.csv`
- `reports/demographic_feature_importance_v11.csv` if produced through normal feature importance flow
- `reports/feature_importance_by_county_v11.csv`
- `data/output/oof_predictions_v11.csv`

Notes:

- The pipeline does not read demographic CSV files. It pulls from PostgreSQL.
- `city_id`, `county_id`, and `district_id` are treated as the shared reference/external reference IDs.
- If demographics are unavailable, the pipeline can continue with missing demographic features, but the coverage report will show low match rates.
