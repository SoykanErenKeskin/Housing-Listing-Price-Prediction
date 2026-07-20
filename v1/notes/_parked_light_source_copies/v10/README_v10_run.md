# V10 County Expert + Anomaly Pipeline

This pipeline extends V9.1 with:

- Segment-aware ensemble retained from V9.1
- County-specific expert blend layer
- Listing anomaly scoring and anomaly reports
- County-level metrics
- County-level feature importance reports

## Recommended run

```bash
python train_v10_county_expert_anomaly_pipeline.py --out outputs/v10_kocaeli
```

## Fast smoke test

```bash
python train_v10_county_expert_anomaly_pipeline.py --out outputs/v10_test --fast
```

## Important outputs

- `reports/metrics_summary_v10.json`
- `reports/county_metrics_v10.csv`
- `reports/county_expert_report_v10.csv`
- `reports/segment_layer_report_v10.csv`
- `reports/feature_importance_by_county_v10.csv`
- `reports/feature_importance_by_county_top40_v10.csv`
- `reports/top_listing_anomalies_v10.csv`
- `reports/anomaly_by_county_v10.csv`
- `reports/anomaly_by_district_v10.csv`
- `reports/anomaly_metric_diagnostics_v10.csv`
- `data/output/oof_predictions_v10.csv`
- `data/output/listing_anomaly_scores_v10.csv`
- `artifacts/model_bundle_v10.joblib`
