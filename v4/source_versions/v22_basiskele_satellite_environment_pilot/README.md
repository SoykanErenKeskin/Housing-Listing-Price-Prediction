# V22 — Başiskele Satellite Environment Pilot

**Generation:** `v4/` (Visual / Satellite Experiments)  
**Package:** `v4/source_versions/v22_basiskele_satellite_environment_pilot/`  
**Outputs:** `v4/outputs/v22_basiskele_satellite_full/`  
**Status:** `DIAGNOSTIC_NO_LIFT` (closed — do not promote)

## What this is

A **free** satellite / environment feature pilot on top of the V21 tabular + site/project baseline.

- No paid Google Maps Static API  
- No Mapbox / MapTiler / paid tile APIs  
- No OSM tile bulk download  
- No CNN fine-tuning  
- Optional Google Earth Engine → Sentinel-2 SR proxies (NDVI / NDWI / NDBI / brightness / texture)

This is **not** an image-quality model. Sentinel-2 (~10 m) will not resolve building interiors; it may capture vegetation, water/coast, and built-up proxies.

## Base checkpoint (read-only)

V21 from V3 — Tabular Premium Signals:

| Metric | Value |
|---|---|
| R² | 0.5059 |
| MAPE | 0.1055 |
| variance_ratio | 0.4590 |
| selected | full_v21 / interactions_foldsafe |
| path | `v3/outputs/v21_basiskele_site_extraction_full/` |

V3 files are **not** modified.

## Paths

| Role | Path |
|---|---|
| Train entry | `train_v22_basiskele_satellite_pipeline.py` |
| Feature fetch | `v4/shared_scripts/fetch_sentinel_features_v22.py` |
| Feature CSV | `data/external/satellite_features/basiskele/sentinel_features_v22.csv` |
| Metadata | `data/external/satellite_features/basiskele/metadata_v22.json` |
| Image cache refs | `v4/image_cache_reference/` |

## Modes

`--satellite-feature-mode`:

| Mode | Features |
|---|---|
| `none` | control_v21 (no sat columns) |
| `basic` | 250m NDVI/NDWI/NDBI/brightness + `sat_has_features` |
| `radii` | 100/250/500m NDVI/NDWI/NDBI |
| `full` | radii + shares/texture/cloud/year/month |

If the satellite CSV is missing, training forces **control only** and does not crash.

## Run order

### 1) Smoke control (no CSV)

```powershell
python v4/source_versions/v22_basiskele_satellite_environment_pilot/train_v22_basiskele_satellite_pipeline.py `
  --out v4/outputs/v22_basiskele_satellite_smoke_control `
  --model-scope basiskele_only `
  --location-feature-mode geo `
  --geo-context-mode geo_with_coast `
  --site-extraction-mode full `
  --site-project-encoding foldsafe_target `
  --satellite-feature-mode none `
  --no-run-satellite-ablation `
  --use-trend --no-interactive `
  --geo-context-cache-dir data/external/geo_context
```

### 2) Fetch free Sentinel features (optional GEE, resume-safe)

Smoke (30 points):

```powershell
python v4/shared_scripts/fetch_sentinel_features_v22.py `
  --city Kocaeli --county Başiskele `
  --out data/external/satellite_features/basiskele/sentinel_features_v22.csv `
  --source gee --limit 30 --max-gee-points 30
```

Full fetch (skips existing smoke rows, checkpoints every 25):

```powershell
python v4/shared_scripts/fetch_sentinel_features_v22.py `
  --city Kocaeli --county Başiskele `
  --out data/external/satellite_features/basiskele/sentinel_features_v22.csv `
  --source gee --resume --save-every 25
```

Ctrl+C keeps the CSV. Failed points → `sentinel_features_v22_failed.csv`.

### 3) Smoke satellite (CSV present)

```powershell
python v4/source_versions/v22_basiskele_satellite_environment_pilot/train_v22_basiskele_satellite_pipeline.py `
  --out v4/outputs/v22_basiskele_satellite_smoke_sat `
  --satellite-feature-mode basic `
  --satellite-feature-csv data/external/satellite_features/basiskele/sentinel_features_v22.csv `
  --no-run-satellite-ablation `
  --model-scope basiskele_only --location-feature-mode geo --geo-context-mode geo_with_coast `
  --site-extraction-mode full --site-project-encoding foldsafe_target `
  --use-trend --no-interactive --geo-context-cache-dir data/external/geo_context
```

### 4) Full ablation (only if smoke sat is green)

```powershell
python v4/source_versions/v22_basiskele_satellite_environment_pilot/train_v22_basiskele_satellite_pipeline.py `
  --out v4/outputs/v22_basiskele_satellite_full `
  --satellite-feature-mode full `
  --run-satellite-ablation `
  --satellite-feature-csv data/external/satellite_features/basiskele/sentinel_features_v22.csv `
  --model-scope basiskele_only --location-feature-mode geo --geo-context-mode geo_with_coast `
  --site-extraction-mode full --site-project-encoding foldsafe_target `
  --use-trend --no-interactive --geo-context-cache-dir data/external/geo_context
```

## Selection gates

Promote a satellite arm only if:

- R² > 0.5059  
- MAPE ≤ 0.1105  
- sat_feature_coverage ≥ 0.65  
- expensive_decile_bias not much worse  
- no obvious leakage / spatial-memorization red flags  

Otherwise `selected = control_v21` and V22 stays **diagnostic**. Do **not** update the global best checkpoint unless full satellite clearly beats V21.

## Final decision — `DIAGNOSTIC_NO_LIFT`

Full ablation: `v4/outputs/v22_basiskele_satellite_full/`

| Experiment | R² | MAPE | VR | selected |
|---|---:|---:|---:|---|
| control_v21 | 0.4813 | 0.1084 | 0.4488 | yes |
| sat_basic_250m | 0.4827 | 0.1083 | 0.4491 | no |
| sat_radii | 0.4801 | 0.1086 | 0.4513 | no |
| sat_full | 0.4829 | 0.1086 | 0.4462 | no |

V21 reference: R² 0.5059 / MAPE 0.1055 / VR 0.4590 — **still the best tabular checkpoint** (`v3/outputs/v21_basiskele_site_extraction_full/`).

- Sentinel environment CSV features did not beat V21 gates.
- Tiny lift vs V22 internal control only; not meaningful.
- V22 `control_v21` did **not** reproduce exact V21 scores — diagnostic experiment only, not a replacement benchmark.
- Do **not** update `MODEL_WORKSPACE_INDEX` / `MANIFEST` best checkpoint to V22.
