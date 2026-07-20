# V9 Segment-Aware Residual Pipeline

**Era:** V1 (`v1/source_versions/v9/`)  
**Status:** Historical. Extends V8 with a segment-aware ensemble layer.

## What changed vs V8

- Still no CatBoost
- Same location-ratio outlier logic as V8
- Residual target: `log(unit_price_gross) - log(location_baseline)`
- Segment-aware ensemble added
- County-level R² / log R² / MAPE / MAE report added

## How to run

```powershell
python train_v9_segment_aware_pipeline.py --out outputs/v9_kocaeli
```

Fast smoke:

```powershell
python train_v9_segment_aware_pipeline.py --out outputs/v9_test --fast
```

## Important outputs

- `reports/metrics_summary_v9.json`
- `reports/county_metrics_v9.csv`
- `reports/segment_layer_report_v9.csv`
- `reports/model_comparison_v9.csv`
- `data/output/oof_predictions_v9.csv`
- `artifacts/segment_aware_model_bundle_v9.joblib`

## Segment-aware logic

The base ensemble predicts every row. Specialist models may blend in on
app-safe segments:

| Segment | Rule of thumb |
|---|---|
| `large_home` | ≥ 151 m² or 4+ rooms |
| `compact_home` | ≤ 85 m² or 1 room |
| `old_building` | age ≥ 26 |
| `mainstream_home` | 85–151 m², 2–3 rooms, age < 26 |

A segment model is used only if it beats the base ensemble **MAPE** inside that
segment; otherwise the base prediction is kept.
