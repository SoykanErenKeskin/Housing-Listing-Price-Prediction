# V25 Kocaeli Full Refresh — core_good_ok

**Status:** Setup / in progress (first primary run on GOOD+OK counties)

Scaffolded from V24.1 (`v24_1_kocaeli_site_merge_repair`) without modifying V24.1 trees.

## Scope

| Field | Value |
|---|---|
| model_version | `v25-kocaeli-full-refresh-core-good-ok` |
| model_scope | `kocaeli_global` |
| county_policy | `core_good_ok` |
| included | Darıca, İzmit, Gebze, Başiskele, Çayırova, Gölcük, Kartepe, Körfez, Derince |
| excluded WEAK | Karamürsel, Kandıra, Dilovası |

## Preserve from V24.1

- location geo + geo_with_coast
- county-scoped site/project + foldsafe_target
- duplex/large-home
- heating OHE including **Yerden Isıtma** + premium heating score
- kitchen / balcony / usage_status / bathroom_count / floor_segment
- leakage guard; comparable off; calibration off; satellite off

## Smoke

```powershell
python v3/source_versions/v25_kocaeli_full_refresh_core_good_ok/train_v25_kocaeli_full_pipeline.py `
  --out v3/outputs/v25_kocaeli_full_refresh_core_good_ok_smoke `
  --fast `
  --limit-sale-per-county 350 `
  --limit-rental-per-county 200 `
  --model-scope kocaeli_global `
  --county-policy core_good_ok `
  --location-feature-mode geo `
  --geo-context-mode geo_with_coast `
  --site-extraction-mode full `
  --site-project-encoding frequency `
  --duplex-feature-mode flags `
  --use-trend `
  --no-interactive `
  --geo-context-cache-dir data/external/geo_context
```

## Full

```powershell
python v3/source_versions/v25_kocaeli_full_refresh_core_good_ok/train_v25_kocaeli_full_pipeline.py `
  --out v3/outputs/v25_kocaeli_full_refresh_core_good_ok_full `
  --model-scope kocaeli_global `
  --county-policy core_good_ok `
  --location-feature-mode geo `
  --geo-context-mode geo_with_coast `
  --site-extraction-mode full `
  --site-project-encoding foldsafe_target `
  --duplex-feature-mode full `
  --run-v25-ablation `
  --use-trend `
  --no-interactive `
  --geo-context-cache-dir data/external/geo_context
```

## Notes

- Do **not** update EDER/app artifacts from V25 in this task.
- Do **not** modify `v24_1` source/output trees.
- Old V24.1 R²≈0.652 is reference only (different county/inventory mix).
