# V21 Başiskele Site/Project Extraction

**Package:** `v3/source_versions/v21_basiskele_site_project_extraction/`  
**Status:** **Best known Başiskele checkpoint** (replaces V20)  
**Base:** V20 `site_foldsafe_target` (R² 0.5017 / MAPE 0.1060 / VR 0.4616)

---

## Best checkpoint

| Field | Value |
|---|---|
| Output | `v3/outputs/v21_basiskele_site_extraction_full/` |
| selected_experiment | `full_v21` (≡ `interactions_foldsafe`) |
| site_extraction_mode | `full` |
| site_project_encoding | `foldsafe_target` |
| comparable_mode | `none` |
| calibration | off |
| R² | 0.5059 |
| MAPE | 0.1055 |
| variance_ratio | 0.4590 |
| large_home_r2 | 0.309 |
| canonical_non_missing | 34.3% |
| dict_hit | 15.5% |
| severe_bad_merge | 0 |
| expensive_decile_bias | −10159 |

### vs V20

| Metric | V20 | V21 |
|---|---:|---:|
| R² | 0.5017 | **0.5059** |
| MAPE | 0.1060 | **0.1055** |
| variance_ratio | 0.4616 | 0.4590 |
| large_home_r2 | — | **0.309** |
| canonical coverage | ~20% | **34.3%** |
| dict_hit | ~6.7% | **15.5%** |
| expensive_decile_bias | −10139 | −10159 |

**Decision:** V21 replaces V20 as best known Başiskele checkpoint. V20 remains evidence that site/project identity is the useful premium signal. Comparable / calibration / no-ridge remain rejected.

**Caveat — expensive bias not solved:** expensive decile bias is slightly worse than V20 (−10139 → −10159). The gap is small enough that V21 is not rejected, but **expensive-segment underprediction remains an open problem.**

Ablation: `v3/outputs/v21_basiskele_site_extraction_full/reports/metrics_site_ablation_v21_basiskele.csv`

---

## Goal

Grow the V20 site/project signal with conservative extraction:

- alias normalization (generic suffixes only)
- canonical `site_project_id`
- known Başiskele dictionary
- coverage lift
- quality tier (no target leakage)
- site × location interactions
- fold-safe canonical target encoding
- coverage + performance ablation + merge audits

## Anti-overmerge

- Never strip brand tokens: life, park, kent, vadi, city, royal, koru, perla
- Do not auto-merge phase numbers (`orka_life` ≠ `orka_life_2`)
- Uncertain aliases → separate IDs + `site_project_alias_review_needed_v21.csv`
- `possible_bad_merge` in merge audit blocks selection when severe

## Smoke

```powershell
python v3/source_versions/v21_basiskele_site_project_extraction/train_v21_basiskele_site_pipeline.py `
  --out v3/outputs/v21_basiskele_site_fullmode_smoke `
  --fast --limit-sale 300 --limit-rental 300 `
  --model-scope basiskele_only `
  --location-feature-mode geo --geo-context-mode geo_with_coast `
  --site-extraction-mode full --site-project-encoding frequency `
  --comparable-mode none --no-run-site-ablation `
  --use-trend --no-interactive `
  --geo-context-cache-dir data/external/geo_context
```

## Full ablation

```powershell
python v3/source_versions/v21_basiskele_site_project_extraction/train_v21_basiskele_site_pipeline.py `
  --out v3/outputs/v21_basiskele_site_extraction_full `
  --model-scope basiskele_only `
  --location-feature-mode geo --geo-context-mode geo_with_coast `
  --site-extraction-mode full --site-project-encoding foldsafe_target `
  --comparable-mode none --run-site-ablation `
  --use-trend --no-interactive `
  --geo-context-cache-dir data/external/geo_context
```

## Selection vs V20

Pass if R² > 0.5017, MAPE ≤ 0.1110, leakage OK, and no severe `possible_bad_merge`.  
Coverage↑ with R²↓ → do not select.
