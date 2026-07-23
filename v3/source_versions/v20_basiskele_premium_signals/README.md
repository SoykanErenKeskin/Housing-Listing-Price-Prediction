# V20 Başiskele Premium Signals

**Package:** `v3/source_versions/v20_basiskele_premium_signals/`  
**Status:** **Superseded by V21** for best known Başiskele checkpoint  
**Base:** V18 Başiskele geo control / `comparable_mode=none`  
**V19:** Calibration / no_ridge rejected — not the V20 base.

V21 best checkpoint: `v3/outputs/v21_basiskele_site_extraction_full/`  
(`full_v21` / `interactions_foldsafe`). Keep V20 as evidence that site/project identity was the first premium signal with meaningful lift.

---

## Checkpoint (historical / evidence baseline)

| Field | Value |
|---|---|
| Output | `v3/outputs/v20_basiskele_premium_signals_full/` |
| selected_experiment | `site_foldsafe_target` |
| premium_feature_mode | `site` |
| site_project_encoding | `foldsafe_target` |
| comparable_mode | `none` |
| calibration | off |
| R² | 0.5017 |
| MAPE | 0.1060 |
| variance_ratio | 0.4616 |
| expensive_decile_bias | −10139 |

### vs V18 reference

| Metric | V18 | V20 selected |
|---|---:|---:|
| R² | 0.4731 | **0.5017** |
| MAPE | 0.1093 | **0.1060** |
| variance_ratio | 0.4264 | **0.4616** |
| expensive_decile_bias | −10811 | **−10139** |

**Decision (at ship time):** V20 replaced V18 as best known Başiskele checkpoint. Comparable and calibration remain rejected. Site/project identity is the first premium signal that produced meaningful lift.

**Later:** V21 superseded V20 on R²/MAPE/coverage with severe bad merge = 0. See [`../v21_basiskele_site_project_extraction/README.md`](../v21_basiskele_site_project_extraction/README.md).

Ablation table: `v3/outputs/v20_basiskele_premium_signals_full/reports/metrics_premium_ablation_v20_basiskele.csv`

---

## Goal

Explain / reduce expensive underprediction using controlled premium signals:

- site/project identity
- pool / sea-view / luxury / project text flags
- villa/duplex/garden subtype flags
- score + location/district interactions
- optional fold-safe site target encoding

Free text is **never** fed raw to the model.

---

## Defaults (CLI)

| Knob | Default |
|---|---|
| `comparable_mode` | `none` |
| `calibration_mode` | `none` (not present) |
| `premium_feature_mode` | `full` |
| `site_project_encoding` | `frequency` |
| location | `geo` + `geo_with_coast` |
| demographics | `safe` |
| attributes | `full` |
| detail effects | `group` |

Selected production-style checkpoint uses `premium_feature_mode=site` + `site_project_encoding=foldsafe_target` (not the CLI defaults).

---

## Smoke

```powershell
python v3/source_versions/v20_basiskele_premium_signals/train_v20_basiskele_premium_pipeline.py --out v3/outputs/v20_basiskele_premium_test --fast --limit-sale 300 --limit-rental 300 --model-scope basiskele_only --location-feature-mode geo --geo-context-mode geo_with_coast --premium-feature-mode flags --site-project-encoding frequency --comparable-mode none --no-run-premium-ablation --use-trend --no-interactive --geo-context-cache-dir data/external/geo_context
```

## Full ablation

```powershell
python v3/source_versions/v20_basiskele_premium_signals/train_v20_basiskele_premium_pipeline.py --out v3/outputs/v20_basiskele_premium_signals_full --model-scope basiskele_only --location-feature-mode geo --geo-context-mode geo_with_coast --premium-feature-mode full --site-project-encoding foldsafe_target --comparable-mode none --run-premium-ablation --use-trend --no-interactive --geo-context-cache-dir data/external/geo_context
```

Do **not** start full ablation until smoke passes.

---

## Selection vs V18 control

Reference: R²=0.4731, MAPE=0.1093, VR=0.4264

Pass if:

- R² > control
- MAPE ≤ control + 0.005
- foldsafe leakage guard passes when used

Priority: less-negative expensive decile bias, VR not worse, large_home R² not worse.

---

## Next — V21 (done)

V21 site extraction is complete and is now the **best known Başiskele checkpoint**  
(`v3/outputs/v21_basiskele_site_extraction_full/`). Do not reopen comparable or calibration as final unless they beat V21.
