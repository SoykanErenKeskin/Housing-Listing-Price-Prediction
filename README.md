# Housing-Listing-Price-Prediction

Research workspace for **housing listing price prediction** models.

The repository is organized by **generation eras**:
**v1** thesis archive, **v2** location/Başiskele sandbox, **v3** Tabular Premium
Signals (best Kocaeli global = **V24.1**; best refreshed Başiskele-only = **V23**),
**v4** visual/satellite experiments (V22 = diagnostic no-lift).

---

## Current checkpoint hierarchy

| Scope | Best checkpoint | Path |
|---|---|---|
| **Kocaeli global** | **V24.1 `full_v24`** | `v3/outputs/v24_1_kocaeli_site_merge_repair/` |
| **Başiskele-only (refreshed data)** | **V23 `duplex_interactions`** | `v3/outputs/v23_basiskele_duplex_largehome_refresh_full/` |
| Visual / satellite | V22 | diagnostic no-lift (do not promote) |

Do not conflate scopes: V23 is Başiskele-only on refreshed inventory; V24.1 is multi-county Kocaeli global after site merge repair.

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
| Site / project (V20+) | county-scoped `site_project_id`, fold-safe target encoding, coverage flags | Premium identity signal |
| Duplex / large-home (V23+) | controlled duplex flags / interactions (no raw title text in model) | Segment-aware lift |

Active pipelines concatenate numeric + categorical columns into the training
matrix; exact lists live in each generation’s train script
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
`is_middle_floor = 1`, plus district rent/trend, geo-distance columns, and (when
enabled) site/project or duplex flags — still without any listing link or firm name.

---

## Quick start (active: V24.1 Kocaeli global)

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

### 3. Reproduce best Kocaeli global (V24.1 site-repair-only)

Uses frozen V24 duplex references; trains site/full arms after merge repair:

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

Selected arm: `full_v24` — R² ≈ 0.6523 / MAPE ≈ 0.1176 / VR ≈ 0.6316 (`severe_bad_merge=0`).

### 4. Reproduce best refreshed Başiskele-only (V23)

```powershell
python v3/source_versions/v23_basiskele_duplex_largehome_refresh/train_v23_basiskele_duplex_pipeline.py `
  --out v3/outputs/v23_basiskele_duplex_largehome_refresh_full `
  --model-scope basiskele_only `
  --location-feature-mode geo --geo-context-mode geo_with_coast `
  --site-extraction-mode full --site-project-encoding foldsafe_target `
  --duplex-feature-mode interactions --no-run-duplex-ablation `
  --use-trend --no-interactive `
  --geo-context-cache-dir data/external/geo_context
```

Outputs are written under `--out` and are **gitignored** (`**/outputs/`, `**/artifacts/`, `*.joblib`).

---

## Repository layout

| Path | Role | Status |
|---|---|---|
| [`v4/`](v4/README.md) | Visual / satellite / image-based experiments | Active; V22 = diagnostic no-lift |
| [`v3/`](v3/README.md) | Tabular Premium Signals (V19–V24.1) | Best Kocaeli global = V24.1; best refreshed Başiskele = V23 |
| [`v2/`](v2/README.md) | Location + Başiskele sandbox (V17 / V18) | Archived reference |
| [`v1/`](v1/README.md) | Thesis legacy (V1–V16 classic Kocaeli) | Archived |
| [`data/`](data/external/geo_context/) | Shared geo cache / satellite features / external data | Shared |
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

Use `best_checkpoints/` + `reports/` first when browsing results on GitHub.

---

## Research narrative (short)

1. **V1–V16 (thesis):** classic Kocaeli multi-county models. Başiskele often underperformed (low R² / variance compression).
2. **V17:** location / geo features; meaningful lift in places, still compression issues in Başiskele.
3. **V18 Başiskele-only:** geo control plateau (~R² 0.47, MAPE ~0.11). Comparable-market predictors ablated and **rejected**.
4. **V19:** calibration / no-ridge diagnostic — **rejected** for final (control won).
5. **V20 → V21:** site/project identity became the useful premium signal (older Başiskele distribution; V21 R² ≈ 0.5059).
6. **V22 (v4):** free Sentinel-2 environment features — **diagnostic no-lift**.
7. **V23:** duplex / large-home refresh on new Başiskele inventory — **best refreshed Başiskele-only** (`duplex_interactions`).
8. **V24 → V24.1:** Kocaeli global site-aware model; merge repair promoted **V24.1 `full_v24`** as best Kocaeli global checkpoint.

---

## Contributing / local workflow tips

- Prefer running training from **repo root** so relative paths and `.env` resolve correctly.
- After moving archived trees under `v1/source_versions/...`, old `../data` relatives may break — pass absolute/`data/external/...` flags.
- Do not edit archived V17/V18 trees when iterating V3+; fork or copy into `v3/source_versions/` (or `v4/` for visual work).
- Keep `comparable_mode=none` and calibration off unless a new, explicit experiment says otherwise.
- Put visual / satellite / image work under **`v4/`** — do not mix into V3.

---

## License / data notes

Listing data comes from a private database (`DATABASE_URL`). Do not publish raw listing dumps or credentials.
