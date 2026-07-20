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

**Best checkpoints:**

| Checkpoint | Location | Notes |
|---|---|---|
| Kocaeli location | `v2/best_checkpoints/best_kocaeli_location_checkpoint/` | V17 |
| Başiskele-only | `v2/best_checkpoints/best_basiskele_only_checkpoint/` | V18 geo control, `comparable_mode=none` |

**V18 Başiskele-only reference metrics:**

| Metric | Approx. value |
|---|---|
| R² | 0.4731 |
| MAPE | 0.1093 |
| variance_ratio | 0.4264 |

**Rejected (V18 comparable ablation):** nearest / similar / weighted / large_home / full comparable.

---

## v3 — Next Experiments

**Path:** [`./v3/`](v3/README.md)

**Includes:**

- V19+ experiment packages under `v3/source_versions/`
- Run outputs under `v3/outputs/` (gitignored)
- Shared helpers under `v3/shared_scripts/` (e.g. `env_loader.py`)

**Status:** **Active development.**

**Goal:** Reduce Başiskele mean-pulling / variance compression via OOF-safe calibration, ensemble profiles, and target profiles — without reintroducing rejected comparable predictors.

Primary package: [`v3/source_versions/v19_basiskele/`](v3/source_versions/v19_basiskele/README.md)

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

### Start V19 here

1. Configure root `.env` from `.env.example`
2. Use `v3/source_versions/v19_basiskele/`
3. Write runs to `v3/outputs/...`
4. Load env via `shared_scripts/env_loader.py` or `v3/shared_scripts/env_loader.py`
