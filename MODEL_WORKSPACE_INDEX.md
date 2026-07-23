# Model Workspace Index

Index of generation eras in this repository. **Active development is `v3/`.**

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

**Status:** Archived / reference for the location era.

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

## v3 — Next Experiments

**Path:** [`./v3/`](v3/README.md)

**Includes:**

- V19+ experiment packages under `v3/source_versions/`
- Run outputs under `v3/outputs/` (gitignored)
- Shared helpers under `v3/shared_scripts/` (e.g. `env_loader.py`)

**Status:** Active. **Best known Başiskele checkpoint is V21.**

### Best known Başiskele checkpoint — V21

| Field | Value |
|---|---|
| Output | `v3/outputs/v21_basiskele_site_extraction_full/` |
| Package | [`v3/source_versions/v21_basiskele_site_project_extraction/`](v3/source_versions/v21_basiskele_site_project_extraction/README.md) |
| selected_experiment | `full_v21` (same metrics as `interactions_foldsafe`) |
| site_extraction_mode | `full` |
| site_project_encoding | `foldsafe_target` |
| R² | 0.5059 |
| MAPE | 0.1055 |
| variance_ratio | 0.4590 |
| large_home_r2 | 0.309 |
| canonical_non_missing | 34.3% |
| dict_hit | 15.5% |
| severe_bad_merge | 0 |
| expensive_decile_bias | −10159 |

**Decision:** V21 **replaces V20** as best known Başiskele checkpoint. V20 remains important evidence that site/project identity is the useful premium signal. Comparable / calibration / no-ridge remain **rejected**.

**Caveat:** expensive decile bias is slightly worse than V20 (−10139 → −10159). Gap is small; V21 is not rejected for it. **Expensive bias is not solved.**

### Prior Başiskele checkpoints (superseded)

| Version | Output | Role |
|---|---|---|
| V20 | `v3/outputs/v20_basiskele_premium_signals_full/` | First site/project premium lift; superseded by V21 |
| V18 | `v2/best_checkpoints/best_basiskele_only_checkpoint/` | Geo control baseline |

### Closed / rejected

**V19 minimal calibration ablation:** diagnostic / rejected_for_final_model.

**Still rejected:** comparable-market predictors (V18); OOF calibration / no_ridge as final (V19).

### Open issue / next direction

Expensive-segment underprediction (decile bias) remains. Further site segmentation / premium project quality may help; do not reopen comparable or calibration as final unless they beat V21.

---

## Shared root paths

| Path | Role |
|---|---|
| [`./data/`](data/external/geo_context/) | Shared datasets / geo context cache |
| [`./outlier_cleaning/`](outlier_cleaning/README.md) | Standalone outlier cleaning package |
| [`./shared_scripts/`](shared_scripts/README.md) | Root analysis / maintenance scripts (not model training) |
| [`./analysis_outputs/`](analysis_outputs/) | Analysis script outputs (timestamped; gitignored) |
| [`./.env`](.env.example) | **Single** DB / secrets config (gitignored) |
| [`./.cursor/`](.cursor/) | Editor metadata (gitignored) |

Root-level `outputs/` and `scripts/` were removed; content lives under `v1/`, `v2/`, and `v3/`.

### Inventory analysis example

```powershell
python shared_scripts/analyze_listing_inventory.py --city Kocaeli
```

### Start from best Başiskele checkpoint (V21)

1. Configure root `.env` from `.env.example`
2. Use `v3/source_versions/v21_basiskele_site_project_extraction/`
3. Reference metrics / ablation under `v3/outputs/v21_basiskele_site_extraction_full/`
4. Keep comparable/calibration/no-ridge off unless a new experiment explicitly beats V21
