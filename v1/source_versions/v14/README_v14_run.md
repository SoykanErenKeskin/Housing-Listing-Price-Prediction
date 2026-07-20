# V14 Local Detail Premium — run notes

**Era:** V1 (`v1/source_versions/v14/`)  
**Status:** Historical. Built on V13 attribute sensitivity. No price-tier
correction.

New layer: fold-safe `LocalDetailPremiumEncoder` — learns residual premium
effects of listing detail binaries (`front_*`, `view_*`, `near_*`, `out_*`,
`in_*`, `subtype_*`) in location context.

## Hypotheses

- **H1:** Local detail premiums lift Başiskele R² / variance_ratio.
- **H2:** Karamürsel does not regress vs V13 default.
- **H3:** Global MAPE/R² stay inside V13 guardrails
  (`MAPE ≤ V13+0.005`, `R² ≥ V13−0.01`).
- **H4:** `group` mode is stable; `full` is a challenger (overfit risk).

## Leakage checklist

- Encoder fits **only** inside the sklearn Pipeline on fold-train `y`.
- No full-dataframe target encoding / effect precompute.
- Residual target recommended: `log(price) - log(location_baseline)`.
- Effect CSVs export from the **final fitted** pipeline encoder (in-sample),
  not from OOF fold encoders. Read them as “in-sample final encoder effects”;
  ablation selection uses OOF metrics (no separate holdout).

## App-safe / deployment warning (mandatory)

If the app does **not** collect `front_*` / `view_*` / `near_*` / `out_*` /
`in_*` / `subtype_*` from users:

- do **not** deploy `--detail-effect-mode group|full`
- either add those fields to the app form
- or ship `--detail-effect-mode none`

Otherwise the model depends on features that are 1 in training and 0 at
inference (silent degrade).

## Defaults

```text
--demographics-mode safe
--attribute-mode full
--detail-effect-mode group
--county-expert-min-rows 250
```

`k180` (`--county-expert-min-rows 180`) is a separate experiment; it is not part
of the main V14 run.

## Commands

### Smoke (run this first)

```bash
cd v14
python train_v14_local_detail_premium_pipeline.py --out outputs/v14_test --fast --limit-sale 800 --limit-rental 800 --demographics-mode safe --attribute-mode full --detail-effect-mode group --no-run-demographics-ablation --no-run-attribute-ablation --no-run-detail-effect-ablation --no-interactive
```

### Full + detail ablation

```bash
python train_v14_local_detail_premium_pipeline.py --out outputs/v14_full --demographics-mode safe --attribute-mode full --detail-effect-mode group --run-detail-effect-ablation --no-interactive
```

### Karamürsel k180 experiment

```bash
python train_v14_local_detail_premium_pipeline.py --out outputs/v14_k180 --demographics-mode safe --attribute-mode full --detail-effect-mode group --county-expert-min-rows 180 --no-interactive
```

## Decision / ship gate

`metrics_summary_v14.json` → `decision`:

- `pass_global_guardrail`, `pass_basiskele_lift`, `pass_karamursel_guardrail`,
  `pass_detail_sensitivity`
- `ship_ready_all_counties_r2_ge_0_65` — **false** unless every county R² ≥ 0.65
  (V14 can PASS experimentally without being ship-ready)
- Gölcük R² < 0.62 → warning; < 0.55 → serious QA finding (does not alone fail
  overall PASS)

## Important reports

- `detail_feature_coverage_v14.csv`
- `detail_premium_effects_by_county_v14.csv`
- `basiskele_detail_premium_diagnostics_v14.csv`
- `detail_premium_group_summary_v14.csv`
- `detail_premium_feature_importance_v14.csv`
- `metrics_detail_effect_ablation_v14.csv`
- `basiskele_variance_diagnostics_v14.csv`
- `karamursel_sensitivity_v14.csv`

## Debug

```bash
python ../scripts/debug_single_prediction_features_v14.py --input-a ../samples/karamursel_old_house.json --input-b ../samples/karamursel_new_house.json --bundle-path outputs/v14_full/artifacts/model_bundle_v14.joblib --out outputs/v14_debug_pair
```

`detail_effect_diff.csv` is produced when a bundle path is provided.
