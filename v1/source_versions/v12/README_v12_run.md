# V12 price-tier correction pipeline

**Era:** V1 (`v1/source_versions/v12/`)  
**Status:** Historical. Built on V11.3 with an optional OOF price-tier correction
layer.

## What is kept / added

Kept: demographics DB join, county aggregates from neighborhood demos, anomaly
filter, segment-aware layer, county expert layer.

**New:** V12 price-tier correction. After county+segment OOF predictions, learn
remaining expensive/cheap tail bias:

```text
correction_pct = (actual_unit_price_gross - pred_after_county) / pred_after_county
```

Correction is trained from app-safe features. Best blend is chosen on OOF. If
MAPE does not improve, the layer stays `kept_current`.

## Fast smoke

```powershell
python train_v12_price_tier_pipeline.py `
  --out outputs/v12_test `
  --fast `
  --limit-sale 800 `
  --limit-rental 800 `
  --demographics-mode safe `
  --no-run-demographics-ablation
```

One-liner:

```powershell
python train_v12_price_tier_pipeline.py --out outputs/v12_test --fast --limit-sale 800 --limit-rental 800 --demographics-mode safe --no-run-demographics-ablation
```

## Full train, safe final + ablation

```powershell
python train_v12_price_tier_pipeline.py --out outputs/v12_full --demographics-mode safe --run-demographics-ablation
```

## Safe full train only

```powershell
python train_v12_price_tier_pipeline.py --out outputs/v12_safe_full --demographics-mode safe --no-run-demographics-ablation
```

## New parameters

```text
--price-tier-correction / --no-price-tier-correction
--price-tier-low-quantile 0.15
--price-tier-high-quantile 0.85
--price-tier-min-rows 250
```

## Main outputs to inspect

```text
outputs/v12_full/reports/metrics_summary_v12.json
outputs/v12_full/reports/metrics_demographics_ablation_v12.csv
outputs/v12_full/reports/price_tier_correction_report_v12.csv
outputs/v12_full/reports/price_tier_decile_report_v12.csv
outputs/v12_full/reports/county_metrics_v12.csv
outputs/v12_full/data/output/oof_predictions_v12.csv
```

In `price_tier_decile_report_v12.csv`, compare `mean_bias_before` vs
`mean_bias_after` for the cheapest and most expensive deciles.

## Note

The V12 layer is auto-guarded: if price-tier correction does not improve MAPE,
final predictions remain the V11.3-style county+segment output. Later V13+
packages typically drop this layer when it stayed `kept_current`.
