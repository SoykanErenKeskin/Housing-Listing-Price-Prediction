# V24.1 Kocaeli site merge repair

**Status:** Promoted — **best Kocaeli global checkpoint** (`full_v24`).

Repair package for V24 site over-merge / alias gates. After ablation gates passed
(`severe_bad_merge=0`, R²/MAPE/county lifts vs safe duplex), V24.1 was manually
promoted as the best Kocaeli global checkpoint.

## Hierarchy (do not conflate scopes)

| Scope | Checkpoint | Status |
|---|---|---|
| Kocaeli global | **V24.1 `full_v24`** | best Kocaeli global |
| Başiskele-only (refreshed data) | **V23 `duplex_interactions`** | best refreshed Başiskele-only |
| Visual / satellite | V22 | diagnostic no-lift |

Site/project extraction is now validated globally after merge repair.

## Selected metrics (`full_v24`)

| Metric | Value |
|---|---:|
| R² | 0.652324 |
| MAPE | 0.117628 |
| variance_ratio | 0.631573 |
| rows | 6667 |
| leakage_pass | true |
| severe_bad_merge | 0 |
| possible_bad_merge | 12 (non-blocking manual review) |

Reference: V24 safe `duplex_largehome_global` R²=0.635866 / MAPE=0.120037 / VR=0.618481  
Lift: R² +0.01646, MAPE −0.00241, VR improved.

## What changed vs V24

1. **Metrics metadata**: `model_scope=kocaeli_global`, top-level `county=null`; Başiskele-only notes moved under `historical_reference` / `basiskele_slice`.
2. **Arm-specific coverage**: writes `site_project_coverage_v24_by_county__{arm}.csv` for each site arm.
3. **Merge audit**: county-scoped IDs; severe rows persisted to `site_project_merge_audit_severe_v24.csv` (+ per-arm copies).
4. **Merge repair**:
   - reject furniture stems (`esyali`, …)
   - yakamoz dict no longer aliases bare `yakamoz`
   - split severe price-gap over-merges into per-variant IDs
   - rebuild merge audit after county scoping

## Site-repair-only run

Skips expensive `control` / `duplex_largehome` retrains; uses frozen V24 refs:

```powershell
python v3/source_versions/v24_1_kocaeli_site_merge_repair/train_v24_1_kocaeli_site_pipeline.py `
  --out v3/outputs/v24_1_kocaeli_site_merge_repair `
  --model-scope kocaeli_global `
  --location-feature-mode geo --geo-context-mode geo_with_coast `
  --location-scope global `
  --site-extraction-mode full --site-project-encoding foldsafe_target `
  --duplex-feature-mode full `
  --run-site-ablation --site-repair-only `
  --use-trend --no-interactive `
  --geo-context-cache-dir data/external/geo_context
```

Arms trained: `site_frequency_global`, `site_foldsafe_global`, `site_plus_duplex_global`, `full_v24`.

## Selection gates (vs duplex_largehome_global)

Select a site/full arm only if:

- `merge_audit_severe_warnings == 0` / `severe_bad_merge == 0`
- R² > duplex
- MAPE ≤ duplex + 0.005
- no material per-county R² regression (≤ 0.02 drop)
- leakage pass

`possible_bad_merge` rows remain non-blocking manual review candidates (12 in the promoted run).

See `v3/outputs/v24_1_kocaeli_site_merge_repair/reports/SELECTION_V24_1_KOCAELI_GLOBAL.md`.
