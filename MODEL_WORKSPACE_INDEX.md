# Model Workspace Index

Index of generation eras in this repository.

| Generation | Official name | Path | Role |
|---|---|---|---|
| v1 | Thesis / classic model archive | `./v1/` | Archived |
| v2 | Location + Başiskele (pre comparable/calibration close-out) | `./v2/` | Archived reference |
| v3 | **Tabular Premium Signals** | `./v3/` | Best tabular Başiskele checkpoint (V21) |
| v4 | Visual / Satellite / Image-based experiments | `./v4/` | Active; V22 = diagnostic/no-lift (V21 still best) |

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
| Başiskele-only (superseded) | `v2/best_checkpoints/best_basiskele_only_checkpoint/` | V18 geo control — later superseded by V20, then **V21** |

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
Başiskele-only tabular premium feature experiments.

**Includes:**

- V19 calibration/no-ridge diagnostic
- V20 premium signal / site-project first lift
- V21 improved site/project extraction

**Best checkpoint:**  
V21 Başiskele site/project extraction:

- R2 = 0.5059
- MAPE = 0.1055
- variance_ratio = 0.4590
- selected = full_v21 / interactions_foldsafe
- canonical_non_missing ≈ 34.3%
- dict_hit ≈ 15.5%
- severe_bad_merge = 0

**Status:**  
Current best tabular Başiskele checkpoint.

**Known issue:**  
Expensive decile underprediction still exists. Site/project extraction improved overall score but did not solve premium top-decile bias fully.

**Rejected / diagnostic:**

- V19 isotonic/linear calibration
- V19 no-ridge
- V20 comparable remains rejected from earlier generation
- V20 text flags alone gave small lift but site/project identity was stronger

**Next:**  
V4 visual/satellite work continues under `v4/` without mixing into V3. V22 Sentinel environment features did **not** beat V21.

| Package / output | Role |
|---|---|
| `v3/source_versions/v21_basiskele_site_project_extraction/` | Best tabular package |
| `v3/outputs/v21_basiskele_site_extraction_full/` | Best tabular run output |
| `v3/source_versions/v20_basiskele_premium_signals/` | First site/project premium lift (superseded) |
| `v3/source_versions/v19_basiskele/` | Calibration / no-ridge diagnostic (closed) |

---

## v4 — Visual / Satellite Experiments

**Path:** [`./v4/`](v4/README.md)

**Status:** active (V22 closed as diagnostic)

**Base / best tabular checkpoint (unchanged):** V21 — `v3/outputs/v21_basiskele_site_extraction_full/`  
(R² 0.5059 / MAPE 0.1055 / VR 0.4590)

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

**Decision:** free Sentinel-2 environment CSV features did not improve the real V21 Başiskele checkpoint. Best checkpoint remains V21. Do **not** promote V22.

**Caveat:** V22 `control_v21` did not reproduce the exact V21 reference score — treat V22 as a diagnostic satellite experiment, not a replacement benchmark. Satellite arms only showed tiny lift over the V22 internal control; none beat V21 gates.

**Interpretation:** free Sentinel-2 environment features at this resolution did not add meaningful predictive value over V21’s tabular + location + site/project features for Başiskele.

**Notes:**

- V4 does **not** modify V3 (or v1/v2)
- V3 remains tabular + site/project/premium text only
- V4 may still explore static-map / image embeddings later
- Large image cache (optional later): `data/external/satellite_cache/basiskele/`
- Image cache references (lightweight): `v4/image_cache_reference/`
- Best checkpoint remains **V21** until a V4 run beats it

---

## Shared root paths

| Path | Role |
|---|---|
| [`./data/`](data/external/geo_context/) | Shared datasets / geo context / satellite cache |
| [`./outlier_cleaning/`](outlier_cleaning/README.md) | Standalone outlier cleaning package |
| [`./shared_scripts/`](shared_scripts/README.md) | Root analysis / maintenance scripts (not model training) |
| [`./analysis_outputs/`](analysis_outputs/) | Analysis script outputs (timestamped; gitignored) |
| [`./.env`](.env.example) | **Single** DB / secrets config (gitignored) |
| [`./.cursor/`](.cursor/) | Editor metadata (gitignored) |

Root-level `outputs/` and `scripts/` were removed; content lives under `v1/`, `v2/`, `v3/`, and `v4/`.

### Inventory analysis example

```powershell
python shared_scripts/analyze_listing_inventory.py --city Kocaeli
```

### Start from best tabular Başiskele checkpoint (V21 in V3)

1. Configure root `.env` from `.env.example`
2. Use `v3/source_versions/v21_basiskele_site_project_extraction/`
3. Reference metrics / ablation under `v3/outputs/v21_basiskele_site_extraction_full/`
4. Keep comparable/calibration/no-ridge off unless a new experiment explicitly beats V21
5. Put visual/satellite work under `v4/` — do not mix into V3
