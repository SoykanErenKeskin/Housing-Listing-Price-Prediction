# V13 Attribute Sensitivity Pipeline

**Era:** V1 (`v1/source_versions/v13/`)  
**Status:** Historical. Built on V12 **without** price-tier correction (V12 left
it as `kept_current`).

## Hypotheses

- **H1:** Holding location + m² fixed, quality differences move the prediction
  (Karamürsel `sale_diff_pct >= 3%`).
- **H2:** Başiskele prediction variance compression decreases.
- **H3:** Global MAPE/R² stay inside V12 safe guardrails.

## Go / no-go

| Rule | Threshold |
|---|---|
| Karamürsel sensitivity | `sale_diff_pct >= 0.03` |
| Direction pass rate | `>= 0.70` |
| Guardrail MAPE | `<= V12_MAPE + 0.005` (~0.1344) |
| Guardrail R² | `>= V12_R2 - 0.01` (~0.6639) |

Final attribute mode selection: **full** if guardrail+sensitivity pass → else
**basic** → `none` only as fallback.

## Amenity / heating normalize contract

Binary flags (`elevator`, `parking`, `balcony`, `furnished`, `site_inside`,
`credit_eligible`):

- `1`: yes / true / 1 / (values that include parking)
- `0`: no / false / 0
- `NaN`: missing / unspecified / empty

Heating score aliases include: Yerden Isıtma, Merkezi, Merkezi (Pay Ölçer),
Kombi (Doğalgaz), Doğalgaz Sobası, Klima, Soba, Yok, …

## Leakage checklist

- `attr_effect_*` fit only inside CV fold `fit`, on residual target
- No full-X target-encoding precompute
- No title / photo / description features

## Rent note (important)

V13 is a **sales** unit-price model. If the app computes rent as
`district_rent_m2_median * gross_m2`, two homes with the same m² get the same
rent. A separate rent attribute multiplier is **V14 backlog** and must not be
mixed into the sales model.

## Fast smoke

```powershell
cd v13
python train_v13_attribute_sensitivity_pipeline.py --out outputs/v13_test --fast --limit-sale 800 --limit-rental 800 --demographics-mode safe --attribute-mode full --no-run-demographics-ablation --no-run-attribute-ablation
```

## Interactive settings wizard

By default, main settings not provided on the CLI are asked in the terminal
(**before** sklearn loads — menu opens immediately):

- `↑` / `↓` to choose, `Enter` to confirm
- Short description under each option
- Only `--out` is free text
- Example: `python train_v13_attribute_sensitivity_pipeline.py --fast` skips
  fast prompt and asks the rest
- Fully silent: `--no-interactive`
- After settings you see “loading model libraries…” — that is the real wait

```powershell
cd v13
python train_v13_attribute_sensitivity_pipeline.py
```

## Full train (recommended)

```powershell
cd v13
python train_v13_attribute_sensitivity_pipeline.py --out outputs/v13_full --demographics-mode safe --attribute-mode full --run-attribute-ablation --no-interactive
```

Ablation matrix (at most 3+3, not a 9-grid):

- Attribute ablation: fixed demographics-mode → none/basic/full
- Demographics ablation (optional): fixed attribute-mode → none/safe/full

## Optional CLI

```text
--county-expert-min-rows 250   # default; use 180 for Karamürsel experiments (separate run)
--run-demographics-ablation
--attribute-mode none|basic|full
```

## Main outputs

```text
outputs/v13_full/reports/metrics_summary_v13.json
outputs/v13_full/reports/metrics_attribute_ablation_v13.csv
outputs/v13_full/reports/feature_sensitivity_v13.csv
outputs/v13_full/reports/karamursel_sensitivity_v13.csv
outputs/v13_full/reports/basiskele_variance_diagnostics_v13.csv
outputs/v13_full/reports/attribute_feature_coverage_v13.csv
outputs/v13_full/artifacts/model_bundle_v13.joblib
```

## Debug (same feature builder)

```powershell
python scripts/debug_single_prediction_features.py `
  --input-a samples/karamursel_old_house.json `
  --input-b samples/karamursel_new_house.json `
  --bundle-path v13/outputs/v13_full/artifacts/model_bundle_v13.joblib `
  --out outputs/v13_debug_pair
```

If the bundle is missing, FE + `attr_*` diff fallback still runs.
