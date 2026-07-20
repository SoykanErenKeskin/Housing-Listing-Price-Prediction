# V16 Regime Residual — run notes

**Era:** V1 (`v1/source_versions/v16/`)  
**Status:** Historical. Built on V15 county specialist. Focus is **controlled
regime residual / baseline experiments**, not more generic features.

V15 files are left untouched; all changes live under `v16/`.

## Phase-0 diagnostic findings (why V16)

- Başiskele mean-pulling confirmed: `var(pred)/var(actual) ≈ 0.452`
- Cheapest decile ≈ +5.7k ₺/m² too high; most expensive ≈ −10.5k ₺/m² too low
- Large_home R² ≈ 0.24 vs non-large ≈ 0.50 (share ≈ 22.7%)
- `m2_group=200+` R² ≈ −0.03; `room_count=4+1` R² ≈ 0.14
- Karamürsel n≈202 sparsity; errors concentrate in 4 Temmuz / Kayacık / Ereğli
- Karamürsel `building_age_group=31+` R² ≈ 0.02

V15 specialist did not deliver general R² lift → V16 goes regime-focused.

## V16 hypotheses

1. **Başiskele large_home regime features** (deterministic, app-safe) make
   large / 200+ / 4+1 variance visible to the model.
2. **Başiskele spread residual** (OOF-safe) shrinks cheap/expensive tail bias.
3. **Karamürsel location×age baseline** (fold-safe residual medians) gives
   structured signal in a sparse county without forcing an aggressive expert.
4. If residual layers fail guardrails, they stay **disabled** (skeleton + reports
   remain).

## Why not more generic features?

Phase 0 pointed to **regime and spread**, not missing generic signal.
Attr/detail expansion was already tried in V14/V15 without Başiskele R² lift.

## Why Başiskele large_home / spread?

- Large_home share is small but R² collapse is large → targeted features +
  optional residual.
- Mean-pulling looks good on MAPE and bad on R² → spread residual watches
  variance_ratio and tail bias.
- No V12-style global price-tier correction.

## Why Karamürsel location-age instead of aggressive expert?

- n≈202; forcing expert blend risks overfitting / instability.
- Neighborhood + age heterogeneity → smoothed `district × age/m2/room` residual
  medians.
- No separate post-hoc correction; enters as features.

## V15 reference (full)

| Metric | Value |
|--------|------:|
| Global R² | 0.6799 |
| Global MAPE | 0.1290 |
| Başiskele R² | 0.4534 |
| Başiskele MAPE | 0.1110 |
| Başiskele variance ratio | 0.4516 |
| Başiskele large_home R² | 0.2396 |
| Gölcük R² | 0.6481 |
| Karamürsel R² | 0.5681 |
| İzmit R² | 0.7109 |
| ship_ready_all_counties_r2_ge_0_65 | false |

## Defaults

```text
--demographics-mode safe
--attribute-mode full
--detail-effect-mode group
--county-expert-min-rows 180
--county-expert-min-rows-overrides "Karamürsel:180"
--basiskele-large-home-regime simple
--basiskele-spread-layer conservative
--karamursel-baseline-mode location_age
--basiskele-variance-lift none
--no-run-v16-regime-ablation
```

Not used in final: `detail-effect-mode full`, title/photo/description,
V12 price-tier, forced high county-expert blends.

## Leakage checklist

- `attr_effect_*`, `detail_effect_*`, `basiskele_*_target_stats`,
  `karamursel_*_residual_median` fit only on CV fold-train `y`
- Large_home / spread residual: delta model fits **train fold**
  `(actual − pred_current)`; validation actual is not in fit
- Spread decile/rank: train-fold predicted quantile thresholds applied to val
- Effect CSVs from final fitted encoder (in-sample); selection uses OOF metrics
- App-safe: no runtime title/photo/description

## Ship gate

Ideal targets:

- Başiskele R² ≥ 0.50; variance_ratio ≥ 0.55; large_home R² lift ≥ +0.08
- Karamürsel R² ≥ 0.60

Final **PASS** may only need global guardrail (R²≥0.670, MAPE≤0.131) +
sensitivity + no-regression vs V15.

`ship_ready_all_counties_r2_ge_0_65=true` only if every county R² ≥ 0.65.

If `overall=PASS` but ship_ready=false:

> **PASS as experiment, NOT ship-ready.**

## Reports

- `reports/metrics_summary_v16.json` (decision + `v15_delta` + `selected_v16_layers`)
- `reports/county_metrics_v16.csv`
- `reports/model_comparison_v16.csv`
- `reports/metrics_v16_regime_ablation.csv`
- `reports/basiskele_large_home_residual_layer_v16.csv`
- `reports/basiskele_spread_residual_layer_v16.csv`
- `reports/karamursel_location_age_baseline_v16.csv`
- `reports/basiskele_decile_bias_v16.csv`
- `reports/basiskele_large_home_error_v16.csv`
- `reports/karamursel_error_by_segment_v16.csv`
- `reports/large_home_diagnostics_v16.csv`
- `reports/county_error_heatmap_v16.csv`

## Commands

### Smoke

```bash
cd v16
python train_v16_regime_residual_pipeline.py --out outputs/v16_test --fast --limit-sale 800 --limit-rental 800 --demographics-mode safe --attribute-mode full --detail-effect-mode group --county-expert-min-rows 180 --county-expert-min-rows-overrides "Karamürsel:180" --basiskele-large-home-regime simple --basiskele-spread-layer conservative --karamursel-baseline-mode location_age --no-run-v16-regime-ablation --no-interactive
```

### Full (+ regime ablation)

```bash
cd v16
python train_v16_regime_residual_pipeline.py --out outputs/v16_full --demographics-mode safe --attribute-mode full --detail-effect-mode group --county-expert-min-rows 180 --county-expert-min-rows-overrides "Karamürsel:180" --basiskele-large-home-regime simple --basiskele-spread-layer conservative --karamursel-baseline-mode location_age --run-v16-regime-ablation --no-interactive
```

### Feature debug

```bash
python ../scripts/debug_single_prediction_features_v16.py --a path/a.json --b path/b.json
```

## Follow-on

Location / geo work continues in V17 under `v2/source_versions/v17/` (regime
residual layers that failed ablation stay off by default there).
