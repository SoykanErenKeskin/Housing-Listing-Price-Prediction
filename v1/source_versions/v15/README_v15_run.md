# V15 County Specialist — run notes

**Era:** V1 (`v1/source_versions/v15/`)  
**Status:** Historical. Built on V14 local detail premium. Optimizes for
**county-level lift** more than global score. No price-tier / post-hoc
correction. No title/photo/description features.

## Purpose

In V14, Başiskele had a detail-premium signal but predictions still collapsed
toward the mean (`var(pred)/var(actual) ≈ 0.42`). V15 adds:

1. Başiskele premium specialist (deterministic + fold-safe target stats)
2. Karamürsel `min_rows=180` override (global stays 250)
3. Large_home redesign features + segment report
4. Optional OOF-safe Başiskele variance-lift (conservative default)
5. Preserve global MAPE/R² guardrails

## V14 reference (group)

| Metric | Value |
|--------|------:|
| Global R² | 0.6787 |
| Global MAPE | 0.1290 |
| Başiskele R² | 0.4553 |
| Başiskele MAPE | 0.1103 |
| Başiskele variance ratio | 0.4224 |
| Gölcük R² | 0.6444 |
| Karamürsel R² | 0.5582 |
| İzmit R² | 0.7107 |
| ship_ready_all_counties_r2_ge_0_65 | false |

`detail-effect-mode full` worsened V14; V15 final keeps **group**.

## Başiskele mean-pulling

Detail premiums fire, but the model still pulls tails to the mean. V15 tries
premium scores, bucket target stats, and optional variance-lift to raise R² and
variance ratio.

## Karamürsel k180 override

```text
--county-expert-min-rows 250
--county-expert-min-rows-overrides "Karamürsel:180"
```

Başiskele / Gölcük / İzmit → 250; Karamürsel → 180. Unparseable overrides warn
and fall back to global 250.

## Large_home redesign

Deterministic features (`large_home_m2_excess`, quality×m2, detail×m2, …) enter
the base pipeline. The segment layer tries ridge / GB / ET / RF for large_home;
`kept_base` / `used_blend` is explicit in the report.

## Leakage checklist

- `attr_effect_*`, `detail_effect_*`, `basiskele_*_target_stats` fit only on CV
  fold-train `y`
- No full-X target-encoding precompute
- Variance-lift delta model fits train-fold `(actual-pred)`; validation actual
  is not used in fit
- Effect CSVs come from the final fitted encoder (in-sample); selection uses OOF
  metrics

## App-safe / deployment

If the app does not collect `front_*` / `view_*` / `near_*` / `out_*` / `in_*` /
`subtype_*`, do not deploy `--detail-effect-mode group|full`. No
title/photo/description features.

## Defaults

```text
--demographics-mode safe
--attribute-mode full
--detail-effect-mode group
--county-expert-min-rows 250
--county-expert-min-rows-overrides "Karamürsel:180"
--basiskele-specialist-mode premium_target_stats
--basiskele-variance-lift conservative
--large-home-specialist-mode redesigned
```

## Success targets (V15)

- Başiskele R² > 0.4553; variance ratio > 0.4224
- Karamürsel R² ≥ 0.5582 (preferably ≥ 0.5768)
- Gölcük R² ≥ 0.62 soft floor
- Global R² ≥ 0.670; MAPE ≤ 0.134
- Direction pass ≥ 0.70; Karamürsel sale_diff_pct ≥ 0.03
- Long-term ship: every county R² ≥ 0.65 → else `ship_ready=false`

## Ship gate

`overall` may PASS while `ship_ready_all_counties_r2_ge_0_65=false`:

> **PASS as experiment, NOT ship-ready.**

## Commands

### Smoke (run this first)

```bash
cd v15
python train_v15_county_specialist_pipeline.py --out outputs/v15_test --fast --limit-sale 800 --limit-rental 800 --demographics-mode safe --attribute-mode full --detail-effect-mode group --basiskele-specialist-mode premium_target_stats --basiskele-variance-lift conservative --county-expert-min-rows-overrides "Karamürsel:180" --no-run-demographics-ablation --no-run-attribute-ablation --no-run-detail-effect-ablation --no-run-basiskele-specialist-ablation --no-interactive
```

### Full + Başiskele specialist ablation

```bash
python train_v15_county_specialist_pipeline.py --out outputs/v15_full --demographics-mode safe --attribute-mode full --detail-effect-mode group --basiskele-specialist-mode premium_target_stats --basiskele-variance-lift conservative --county-expert-min-rows-overrides "Karamürsel:180" --run-basiskele-specialist-ablation --no-interactive
```

## Important reports

- `metrics_summary_v15.json`
- `county_expert_layer_report_v15.csv` (`min_rows_used`, `override_used`, …)
- `basiskele_premium_specialist_diagnostics_v15.csv`
- `basiskele_variance_diagnostics_v15.csv` / `basiskele_variance_lift_report_v15.csv`
- `large_home_diagnostics_v15.csv`
- `metrics_basiskele_specialist_ablation_v15.csv`
- `detail_premium_*` / `karamursel_sensitivity_v15.csv`

## Debug

```bash
python scripts/debug_single_prediction_features_v15.py --input-a a.json --input-b b.json --bundle-path v15/outputs/.../artifacts/model_bundle_v15.joblib --out outputs/v15_debug_pair
```
