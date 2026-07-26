# Model Workspace Index

Index of generation eras in this repository.

| Generation | Official name | Path | Role |
|---|---|---|---|
| v1 | Thesis / classic model archive | `./v1/` | Archived |
| v2 | Location + Başiskele (pre comparable/calibration close-out) | `./v2/` | Archived reference |
| v3 | **Tabular Premium Signals** | `./v3/` | Best Kocaeli global = **V24.1**; best refreshed Başiskele-only = **V23** |
| v4 | Visual / Satellite / Image-based experiments | `./v4/` | Active; V22 = diagnostic/no-lift |

### Current checkpoint hierarchy

| Scope | Best checkpoint | Path |
|---|---|---|
| **Kocaeli global** | **V24.1 `full_v24`** | `v3/outputs/v24_1_kocaeli_site_merge_repair/` |
| **Başiskele-only (refreshed data)** | **V23 `duplex_interactions`** | `v3/outputs/v23_basiskele_duplex_largehome_refresh_full/` |
| Visual / satellite | V22 | diagnostic no-lift (do not promote) |

Site/project extraction is now validated globally after merge repair (`severe_bad_merge=0`; `possible_bad_merge=12` non-blocking manual review).

---

## v1 — Thesis Legacy

**Path:** [`./v1/`](v1/README.md)

**Includes:**

- Original V1–V16 trees under `v1/source_versions/`
- Thesis-era reports / scripts / curated checkpoints under `v1/`

**Status:** Archived. Do not continue new model development here.

**Best checkpoint:** Manual review needed around V15/V16 (`v1/best_checkpoints/manual_review_v15_v16/`).

**Known issues:**

- Başiskele low R²
- Variance compression (predictions pulled toward the mean)

---

## v2 — Location + Başiskele Sandbox

**Path:** [`./v2/`](v2/README.md)

**Includes:**

- Original V17 and V18 Başiskele trees under `v2/source_versions/`
- Location / geo / comparable scripts under `v2/scripts/`
- Era outputs under `v2/outputs/` (local; gitignored)

**Status:** Archived / reference for the location era (comparable / calibration öncesi dönem).

**Best checkpoints (era-local):**

| Checkpoint | Location | Notes |
|---|---|---|
| Kocaeli location | `v2/best_checkpoints/best_kocaeli_location_checkpoint/` | V17 |
| Başiskele-only (superseded) | `v2/best_checkpoints/best_basiskele_only_checkpoint/` | V18 geo control — later superseded by V20 → V21 → **V23** (refreshed) |

**V18 Başiskele-only reference metrics (historical):**

| Metric | Approx. value |
|---|---|
| R² | 0.4731 |
| MAPE | 0.1093 |
| variance_ratio | 0.4264 |
| expensive_decile_bias | −10811 |

**Rejected (V18 comparable ablation):** nearest / similar / weighted / large_home / full comparable.

---

## v3 — Tabular Premium Signals

**Path:** [`./v3/`](v3/README.md)

**Short name:** `v3_tabular_premium_signals` (folder remains `./v3/`)

**Scope:**  
Tabular premium feature experiments — Başiskele-only through V23, then Kocaeli global site-aware (V24 / V24.1).

**Includes:**

- V19 calibration/no-ridge diagnostic
- V20 premium signal / site-project first lift
- V21 improved site/project extraction (older Başiskele distribution)
- V23 duplex / large-home refresh (refreshed Başiskele-only)
- V24 Kocaeli site-aware global refresh
- **V24.1** Kocaeli site merge repair → **best Kocaeli global checkpoint**

### Best Kocaeli global checkpoint — V24.1 `full_v24`

| Field | Value |
|---|---|
| Output | `v3/outputs/v24_1_kocaeli_site_merge_repair/` |
| Package | `v3/source_versions/v24_1_kocaeli_site_merge_repair/` |
| selected_experiment | `full_v24` |
| site_extraction_mode | `full` |
| site_project_encoding | `foldsafe_target` |
| duplex_feature_mode | `full` |
| R² | 0.652324 |
| MAPE | 0.117628 |
| variance_ratio | 0.631573 |
| rows | 6667 |
| leakage_pass | true |
| severe_bad_merge | **0** |
| possible_bad_merge | **12** (non-blocking manual review) |
| best_checkpoint | **true** |

**Lift vs V24 safe duplex (`duplex_largehome_global`):** R² +0.01646 (0.635866 → 0.652324), MAPE −0.00241 (0.120037 → 0.117628), VR improved (0.618481 → 0.631573).

**County metrics:**

| County | R² | MAPE |
|---|---:|---:|
| İzmit | 0.682681 | 0.129376 |
| Gölcük | 0.670027 | 0.118001 |
| Başiskele | 0.511732 | 0.102983 |
| Karamürsel | 0.578453 | 0.158589 |
| Kartepe | 0.550115 | 0.101380 |

**Decision:** Promote V24.1 as the best Kocaeli global checkpoint. Site/project extraction is now validated globally after merge repair. Keep `possible_bad_merge=12` as non-blocking manual review candidates. Document `severe_bad_merge=0`.

### Best refreshed Başiskele-only checkpoint — V23 `duplex_interactions`

| Field | Value |
|---|---|
| Output | `v3/outputs/v23_basiskele_duplex_largehome_refresh_full/` |
| Package | `v3/source_versions/v23_basiskele_duplex_largehome_refresh/` |
| selected_experiment | `duplex_interactions` |
| R² | ≈0.4918 |
| MAPE | ≈0.1039 |
| variance_ratio | ≈0.4774 |

**Decision:** V23 remains the best refreshed Başiskele-only checkpoint. Do not conflate with Kocaeli global V24.1.

### Historical Başiskele tabular — V21 (older distribution)

| Field | Value |
|---|---|
| Output | `v3/outputs/v21_basiskele_site_extraction_full/` |
| R² / MAPE / VR | 0.5059 / 0.1055 / 0.4590 |
| severe_bad_merge | 0 |

V21 remains the older-distribution Başiskele reference; refreshed-data Başiskele best is **V23**.

**Rejected / diagnostic:**

- V19 isotonic/linear calibration
- V19 no-ridge
- V20 comparable remains rejected from earlier generation
- V24 safe duplex kept only as promotion reference after V24.1 site merge repair

| Package / output | Role |
|---|---|
| `v3/source_versions/v24_1_kocaeli_site_merge_repair/` | **Best Kocaeli global package** |
| `v3/outputs/v24_1_kocaeli_site_merge_repair/` | **Best Kocaeli global run output** |
| `v3/source_versions/v23_basiskele_duplex_largehome_refresh/` | Best refreshed Başiskele-only package |
| `v3/outputs/v23_basiskele_duplex_largehome_refresh_full/` | Best refreshed Başiskele-only run |
| `v3/source_versions/v21_basiskele_site_project_extraction/` | Older Başiskele tabular package |
| `v3/outputs/v21_basiskele_site_extraction_full/` | Older Başiskele tabular run |
| `v3/source_versions/v20_basiskele_premium_signals/` | First site/project premium lift (superseded) |
| `v3/source_versions/v19_basiskele/` | Calibration / no-ridge diagnostic (closed) |

---

## v4 — Visual / Satellite Experiments

**Path:** [`./v4/`](v4/README.md)

**Status:** active (V22 closed as diagnostic)

**Base / best tabular Başiskele (older distribution):** V21 — `v3/outputs/v21_basiskele_site_extraction_full/`  
**Refreshed Başiskele-only best:** V23 — `v3/outputs/v23_basiskele_duplex_largehome_refresh_full/`  
**Best Kocaeli global:** V24.1 — `v3/outputs/v24_1_kocaeli_site_merge_repair/`

**Goal:** test satellite/static-map visual features and image embeddings as additional premium/micro-location signals

### V22 — Sentinel environment pilot = `DIAGNOSTIC_NO_LIFT`

| Role | Path |
|---|---|
| Package | `v4/source_versions/v22_basiskele_satellite_environment_pilot/` |
| Full ablation output | `v4/outputs/v22_basiskele_satellite_full/` |
| Feature CSV | `data/external/satellite_features/basiskele/sentinel_features_v22.csv` |

| Experiment | R² | MAPE | VR | selected |
|---|---:|---:|---:|---|
| control_v21 | 0.4813 | 0.1084 | 0.4488 | yes |
| sat_basic_250m | 0.4827 | 0.1083 | 0.4491 | no |
| sat_radii | 0.4801 | 0.1086 | 0.4513 | no |
| sat_full | 0.4829 | 0.1086 | 0.4462 | no |

**Decision:** V22 remains diagnostic no-lift. Free Sentinel-2 environment CSV features did not improve the real V21 Başiskele checkpoint. Do **not** promote V22.

**Caveat:** V22 `control_v21` did not reproduce the exact V21 reference score — treat V22 as a diagnostic satellite experiment, not a replacement benchmark.

**Notes:**

- V4 does **not** modify V3 (or v1/v2)
- V3 remains tabular + site/project/premium text only
- V4 may still explore static-map / image embeddings later
- Large image cache (optional later): `data/external/satellite_cache/basiskele/`
- Image cache references (lightweight): `v4/image_cache_reference/`

---

## Shared root paths

| Path | Role |
|---|---|
| [`./data/`](data/external/geo_context/) | Shared datasets / geo context / satellite cache |
| [`./outlier_cleaning/`](outlier_cleaning/README.md) | Standalone outlier cleaning package |
| [`./shared_scripts/`](shared_scripts/README.md) | Root analysis / maintenance scripts (not model training) |
| [`./analysis_outputs/`](analysis_outputs/) | Analysis script outputs (timestamped; gitignored) |
| [`./.env`](.env.example) | **Single** DB / secrets config (gitignored) |

Root-level `outputs/` and `scripts/` were removed; content lives under `v1/`, `v2/`, `v3/`, and `v4/`.

### Inventory analysis example

```powershell
python shared_scripts/analyze_listing_inventory.py --city Kocaeli
```

### Start from best Kocaeli global checkpoint (V24.1)

1. Configure root `.env` from `.env.example`
2. Use `v3/source_versions/v24_1_kocaeli_site_merge_repair/`
3. Reference metrics / ablation under `v3/outputs/v24_1_kocaeli_site_merge_repair/`
4. Keep `severe_bad_merge=0`; treat `possible_bad_merge=12` as manual review only
5. For refreshed Başiskele-only work, use V23 — do not replace V24.1 with V23 (different scope)
6. Put visual/satellite work under `v4/` — V22 remains diagnostic no-lift
