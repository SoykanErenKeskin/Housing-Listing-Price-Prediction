# Housing-Listing-Price-Prediction

Research workspace for **housing listing price prediction** models.

Current focus areas include a multi-county thesis archive and an active
Başiskele-only calibration / anti-shrink track. The repository is organized by
**generation eras**; active development lives under **`v3/`**.

---

## Quick start (active: V19 Başiskele)

### 1. Environment

```powershell
# From repo root
copy .env.example .env
# Edit .env and set DATABASE_URL=...
```

Requires Python 3.10+ and typical ML stack (`pandas`, `scikit-learn`, `sqlalchemy`, `python-dotenv`, …).

### 2. Geo context cache

Offline OSM/coast/POI cache lives at:

`data/external/geo_context`

Pass it explicitly to training:

```text
--geo-context-cache-dir data/external/geo_context
```

### 3. Smoke train (fast)

```powershell
python v3/source_versions/v19_basiskele/train_v19_basiskele_calibration_pipeline.py `
  --out v3/outputs/v19_basiskele_test `
  --fast --limit-sale 300 --limit-rental 300 `
  --model-scope basiskele_only `
  --location-feature-mode geo `
  --geo-context-mode geo_with_coast `
  --calibration-mode linear `
  --ensemble-profile balanced `
  --target-profile residual_log `
  --no-run-calibration-ablation `
  --use-trend --no-interactive `
  --geo-context-cache-dir data/external/geo_context
```

### 4. Full calibration ablation

```powershell
python v3/source_versions/v19_basiskele/train_v19_basiskele_calibration_pipeline.py `
  --out v3/outputs/v19_basiskele_ablation `
  --model-scope basiskele_only `
  --location-feature-mode geo `
  --geo-context-mode geo_with_coast `
  --calibration-mode isotonic `
  --ensemble-profile balanced `
  --target-profile residual_log `
  --run-calibration-ablation `
  --use-trend --no-interactive `
  --geo-context-cache-dir data/external/geo_context
```

Outputs are written under `--out` and are **gitignored** (`**/outputs/`, `**/artifacts/`, `*.joblib`).

---

## Repository layout

| Path | Role | Status |
|---|---|---|
| [`v3/`](v3/README.md) | Next experiments (V19+ calibration / anti-shrink) | **Active** |
| [`v2/`](v2/README.md) | Location + Başiskele sandbox (V17 / V18) | Archived reference |
| [`v1/`](v1/README.md) | Thesis legacy (V1–V16 classic Kocaeli) | Archived |
| [`data/`](data/external/geo_context/) | Shared geo cache / external data | Shared |
| [`outlier_cleaning/`](outlier_cleaning/README.md) | Standalone listing outlier cleaner | Shared utility |
| [`shared_scripts/`](shared_scripts/README.md) | Inventory / DB health analysis (not model training) | Shared utility |
| [`analysis_outputs/`](analysis_outputs/) | Timestamped analysis runs | Local only (gitignored) |
| [`.env`](.env.example) | `DATABASE_URL` secrets | **Never commit** |

See also: [`MODEL_WORKSPACE_INDEX.md`](MODEL_WORKSPACE_INDEX.md) and [`MANIFEST.json`](MANIFEST.json).

---

## Security & secrets

- **Single secrets file:** repo-root `.env` with `DATABASE_URL`.
- Copy from [`.env.example`](.env.example). Do not commit real credentials.
- Training loaders walk parents to find root `.env` (`shared_scripts/env_loader.py`, `v3/shared_scripts/env_loader.py`).
- `.gitignore` blocks `.env`, credential JSON, private keys, model binaries, and multi-GB `outputs/` / `artifacts/` trees.

---

## What is in git vs local-only

**Tracked (for interpretation without your private DB):**

- Generation `reports/` and `best_checkpoints/` (metrics, ablation tables, key plots)
- Active/era run folders under `v1/outputs`, `v2/outputs`, `v3/outputs` **except** heavy dumps below

**Ignored (too large / not needed to read results):**

| Ignored | Why |
|---|---|
| `**/source_versions/**/outputs/` | Full historical run trees (hundreds of MB each) |
| `**/artifacts/`, `*.joblib`, `*.zip` | Serialized models / bundles |
| Large prediction CSVs (`oof_predictions*`, cleaned dumps, …) | Regenerable; not required to judge metrics |
| `analysis_outputs/` | Local inventory dumps |
| `outlier_cleaning/data/input\|output/` | Large CSV exports |
| `.env` | Database credentials |
| `.cursor/` | Editor metadata |

Use `best_checkpoints/` + `reports/` first when browsing results on GitHub.

---

## Research narrative (short)

1. **V1–V16 (thesis):** classic Kocaeli multi-county models. Başiskele often underperformed (low R² / variance compression).
2. **V17:** location / geo features; meaningful lift in places, still compression issues in Başiskele.
3. **V18 Başiskele-only:** geo control plateau (~R² 0.47, MAPE ~0.11). Comparable-market predictors ablated and **rejected**.
4. **V19 (active):** OOF-safe **prediction calibration** and anti-shrink research on the V18 geo control base (`comparable_mode=none`).

---

## Contributing / local workflow tips

- Prefer running training from **repo root** so relative paths and `.env` resolve correctly.
- After moving archived trees under `v1/source_versions/...`, old `../data` relatives may break — pass absolute/`data/external/...` flags.
- Do not edit archived V17/V18 trees when iterating V19; fork or copy into `v3/source_versions/`.

---

## License / data notes

Listing data comes from a private database (`DATABASE_URL`). Do not publish raw listing dumps or credentials.
