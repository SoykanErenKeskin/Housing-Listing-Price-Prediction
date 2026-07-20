# V9.1 Segment-Aware Pipeline

**Era:** V1 (`v1/source_versions/v9.1/`)  
**Status:** Historical hotfix over V9 segment-layer decision logic.

## Main difference

V9 used a segment specialist only when it beat the base ensemble **alone**.
V9.1 tries blend weights per segment:

`0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.00`

If any blend improves segment MAPE, that blend is applied to the final
prediction. Outlier cleaning matches V8/V9. Still no CatBoost.

## How to run

```bash
python train_v9_1_segment_aware_pipeline.py --out outputs/v9_1_kocaeli
```

Fast smoke:

```bash
python train_v9_1_segment_aware_pipeline.py --out outputs/v9_1_test --fast
```

## Outputs to check

- `reports/metrics_summary_v9_1.json`
- `reports/county_metrics_v9_1.csv`
- `reports/segment_layer_report_v9_1.csv`
- `data/output/oof_predictions_v9_1.csv`

Use `segment_layer_report_v9_1.csv` to see which segments kept base vs used a
blend weight.
