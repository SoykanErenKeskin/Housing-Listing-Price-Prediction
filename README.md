# Housing-Listing-Price-Prediction

Research workspace for **housing listing price prediction** models.

Current focus areas include a multi-county thesis archive and an active
Başiskele-only calibration / anti-shrink track. The repository is organized by
**generation eras**; active development lives under **`v3/`**.

---

## Dataset shape (what the model sees)

Training rows come from a **private listing database** (sale + rental). Raw dumps
are **not** in this repository. Below is only the schema shape and **synthetic**
example rows (no listing IDs, URLs, site names, or portal brands).

### Feature groups

| Group | Examples | Role |
|---|---|---|
| Identity / target | `price` (sale) or `monthly_rent` (rental), `currency`, `listing_purpose` | Supervision only; IDs/URLs never used as features |
| Size & layout | `gross_m2`, `net_m2`, `room_count`, `rooms`, `living_rooms`, `bathroom_count`, `m2_group` | Core structural signal |
| Building & floor | `building_age`, `building_age_group`, `floor_num`, `total_floors`, `floor_segment`, `is_ground_floor`, `is_top_floor`, … | Vertical / age structure |
| Amenities (categorical) | `heating`, `kitchen`, `balcony`, `elevator`, `parking`, `furnished`, `usage_status`, `site_inside`, `credit_eligible`, `deed_status`, `energy_certificate`, `seller_type` | Listing attributes |
| Location (categorical) | `city`, `county`, `district` | Geography keys |
| Detail flags / scores | `detail_*` counts, `front_*`, `view_*`, `transport_*`, `near_*`, `in_*`, `out_*`, quality scores | Parsed listing detail checkboxes → numeric scores |
| Market / rent context | `district_rent_m2_*`, `county_rent_m2_*`, `estimated_rent_m2_gross`, `trend_*`, `location_baseline_*` | Engineered from peer listings (no external brand) |
| Demographics (optional) | `demo_*`, `county_demo_*` | Neighborhood / county socio-economic context |
| Geo context (optional) | coast / road / POI distances from local `data/external/geo_context` | OSM-style distance features |

Active pipelines (e.g. V19) concatenate the numeric + categorical columns above into
the training matrix; exact lists live in each generation’s train script
(`NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`).

### Synthetic example rows (sale listings)

Illustrative only — values are made up for documentation.

Categorical values are shown in English; database rows store the original Turkish
labels (e.g. `Kombi`, `Var`, `Kapalı Otopark`), which the pipeline maps consistently.

| # | county | district | gross_m2 | net_m2 | room_count | building_age | floor_num | total_floors | heating | balcony | elevator | parking | site_inside | bathroom_count | price (TRY) |
|---|---|---|---:|---:|---|---:|---:|---:|---|---|---|---|---|---:|---:|
| A | Başiskele | Sample District A | 125 | 105 | 3+1 | 8 | 3 | 5 | Individual (combi) | Yes | Yes | Open parking | Yes | 1 | 4.250.000 |
| B | İzmit | Sample District B | 95 | 80 | 2+1 | 15 | 2 | 4 | Individual (combi) | Yes | No | None | No | 1 | 2.800.000 |
| C | Karamürsel | Sample District C | 160 | 135 | 4+1 | 3 | 7 | 8 | Central (heat-cost allocator) | Yes | Yes | Closed parking | Yes | 2 | 6.100.000 |
| D | Gölcük | Sample District D | 110 | 90 | 3+1 | 22 | 0 | 3 | Stove | No | No | None | No | 1 | 1.950.000 |

After feature engineering, row A might also carry derived fields such as
`net_gross_ratio ≈ 0.84`, `m2_group = 101-125`, `floor_segment = mid-floor`,
`is_middle_floor = 1`, plus district rent/trend and (if enabled) geo-distance
columns — still without any listing link or firm name.

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

### 4. Full single-profile run

```powershell
python v3/source_versions/v19_basiskele/train_v19_basiskele_calibration_pipeline.py `
  --out v3/outputs/v19_basiskele_ablation `
  --model-scope basiskele_only `
  --location-feature-mode geo `
  --geo-context-mode geo_with_coast `
  --calibration-mode isotonic `
  --ensemble-profile balanced `
  --target-profile residual_log `
  --no-run-calibration-ablation `
  --use-trend --no-interactive `
  --geo-context-cache-dir data/external/geo_context
```

### 5. Minimal calibration ablation (6 arms)

```powershell
python v3/source_versions/v19_basiskele/train_v19_basiskele_calibration_pipeline.py `
  --out v3/outputs/v19_basiskele_minimal_ablation `
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

Grid: `none|linear|isotonic` × `balanced|no_ridge` (`target_profile=residual_log` fixed).
Writes `reports/metrics_calibration_ablation_v19_basiskele.csv` and promotes the selected arm’s bundle.

Outputs are written under `--out` and are **gitignored** (`**/outputs/`, `**/artifacts/`, `*.joblib`).

---

## Repository layout

| Path | Role | Status |
|---|---|---|
| [`v3/`](v3/README.md) | Next experiments (V19 diagnostic; V18 remains best Başiskele) | Active scaffold |
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
4. **V19 (closed minimal ablation):** OOF-safe calibration × no-ridge grid selected `control_none_balanced`. Calibration/no_ridge **rejected** for final; V18 geo control remains best Başiskele checkpoint. Optional later: `direct_price` / `hybrid` only if they beat V18.

---

## Contributing / local workflow tips

- Prefer running training from **repo root** so relative paths and `.env` resolve correctly.
- After moving archived trees under `v1/source_versions/...`, old `../data` relatives may break — pass absolute/`data/external/...` flags.
- Do not edit archived V17/V18 trees when iterating V19; fork or copy into `v3/source_versions/`.

---

## License / data notes

Listing data comes from a private database (`DATABASE_URL`). Do not publish raw listing dumps or credentials.
