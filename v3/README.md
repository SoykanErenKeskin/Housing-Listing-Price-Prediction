# V3 — Tabular Premium Signals

**Path:** `./v3/`  
**Short name:** `v3_tabular_premium_signals`  
**Status:** Best Kocaeli global = **V24.1**; best refreshed Başiskele-only = **V23**

This generation covers tabular premium signal experiments — Başiskele-only through
V23, then Kocaeli global site-aware (V24 / V24.1).

- **V19:** calibration / no-ridge diagnostic; not selected for the final model.
- **V20:** site/project identity delivered the first meaningful premium lift.
- **V21:** older-distribution Başiskele tabular reference.
- **V23:** best refreshed Başiskele-only checkpoint (`duplex_interactions`).
- **V24.1:** **best Kocaeli global checkpoint** (`full_v24`); site/project extraction validated globally after merge repair.
- **V22** (under v4) remains diagnostic no-lift.
- **V3** is not a catch-all for future work; **V4** is separate for visual/satellite/image experiments.

---

## Checkpoint hierarchy (do not conflate scopes)

| Scope | Checkpoint | Status |
|---|---|---|
| Kocaeli global | **V24.1 `full_v24`** | best Kocaeli global |
| Başiskele-only (refreshed data) | **V23 `duplex_interactions`** | best refreshed Başiskele-only |
| Başiskele (older distribution) | V21 `full_v21` | historical reference only |
| Visual / satellite | V22 | diagnostic no-lift |

---

## Version summary

| Version | Purpose | Result |
|---|---|---|
| V19 | calibration/no-ridge diagnostic | rejected; no candidate beat control |
| V20 | premium/site project first model | R² 0.5017, MAPE 0.1060 |
| V21 | improved site/project extraction | R² 0.5059, MAPE 0.1055 (older Başiskele ref) |
| V23 | duplex / large-home refresh | **best refreshed Başiskele-only** |
| V24 | Kocaeli site-aware global | safe duplex reference; site arms needed merge repair |
| V24.1 | Kocaeli site merge repair | **best Kocaeli global** (`full_v24`, R² 0.652324) |

---

## Best Kocaeli global checkpoint — V24.1

| Field | Value |
|---|---|
| Output | [`outputs/v24_1_kocaeli_site_merge_repair/`](outputs/v24_1_kocaeli_site_merge_repair/) |
| Package | [`source_versions/v24_1_kocaeli_site_merge_repair/`](source_versions/v24_1_kocaeli_site_merge_repair/README.md) |
| selected_experiment | `full_v24` |
| R² / MAPE / VR | 0.652324 / 0.117628 / 0.631573 |
| rows | 6667 |
| severe_bad_merge | 0 |
| possible_bad_merge | 12 (non-blocking manual review) |
| best_checkpoint | true |

**County slice (selected run):**

| County | R² | MAPE |
|---|---:|---:|
| İzmit | 0.682681 | 0.129376 |
| Gölcük | 0.670027 | 0.118001 |
| Başiskele | 0.511732 | 0.102983 |
| Karamürsel | 0.578453 | 0.158589 |
| Kartepe | 0.550115 | 0.101380 |

**Decision:** V24.1 is the best Kocaeli global checkpoint. V23 remains the best refreshed Başiskele-only checkpoint. V22 remains diagnostic no-lift.

See [`outputs/v24_1_kocaeli_site_merge_repair/reports/SELECTION_V24_1_KOCAELI_GLOBAL.md`](outputs/v24_1_kocaeli_site_merge_repair/reports/SELECTION_V24_1_KOCAELI_GLOBAL.md).

### Reproduce V24.1 (site-repair-only)

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

---

## Best refreshed Başiskele-only — V23

| Field | Value |
|---|---|
| Output | [`outputs/v23_basiskele_duplex_largehome_refresh_full/`](outputs/v23_basiskele_duplex_largehome_refresh_full/) |
| Package | [`source_versions/v23_basiskele_duplex_largehome_refresh/`](source_versions/v23_basiskele_duplex_largehome_refresh/README.md) |
| selected_experiment | `duplex_interactions` |
| R² / MAPE / VR | 0.4918 / 0.1039 / 0.4774 |
| large_home_r2 | 0.3542 |

Primary comparison is `control_v21_on_new_data` on the same refreshed inventory — do **not** compare blindly to old V21 R²=0.5059 (different distribution).

### Reproduce selected V23 arm

```powershell
python v3/source_versions/v23_basiskele_duplex_largehome_refresh/train_v23_basiskele_duplex_pipeline.py `
  --out v3/outputs/v23_basiskele_duplex_interactions_selected `
  --model-scope basiskele_only `
  --location-feature-mode geo --geo-context-mode geo_with_coast `
  --site-extraction-mode full --site-project-encoding foldsafe_target `
  --duplex-feature-mode interactions --no-run-duplex-ablation `
  --use-trend --no-interactive `
  --geo-context-cache-dir data/external/geo_context
```

---

## Historical Başiskele tabular — V21

| Field | Value |
|---|---|
| Output | [`outputs/v21_basiskele_site_extraction_full/`](outputs/v21_basiskele_site_extraction_full/) |
| Package | [`source_versions/v21_basiskele_site_project_extraction/`](source_versions/v21_basiskele_site_project_extraction/README.md) |
| selected_experiment | `full_v21` (≡ `interactions_foldsafe`) |
| R² / MAPE / VR | 0.5059 / 0.1055 / 0.4590 |
| severe_bad_merge | 0 |

V21 remains the older-distribution Başiskele reference and was the base for V22/V23 work. Refreshed-data Başiskele best is **V23**; Kocaeli global best is **V24.1**.

| | V20 | V21 selected |
|---|---:|---:|
| R² | 0.5017 | **0.5059** |
| MAPE | 0.1060 | **0.1055** |
| variance_ratio | 0.4616 | 0.4590 |
| large_home_r2 | — | **0.309** |
| canonical coverage | ~20% | **34.3%** |
| dict_hit | ~6.7% | **15.5%** |

**Known issue:** expensive decile underprediction persists across V21/V23.

---

## Rejected / diagnostic

1. V19 isotonic / linear calibration (diagnostic; not final)
2. V19 `no_ridge` ensemble (not final)
3. Comparable-market predictors (rejected in V2; stay off in V3)
4. V20 text flags alone — small lift; site/project identity was stronger
5. V22 Sentinel environment features (v4) — diagnostic no-lift

---

## Folders

| Path | Use |
|---|---|
| `source_versions/` | V19–V24.1 packages |
| `outputs/` | Training run outputs (**gitignored**) |
| `reports/` | Curated metrics / ablation tables |
| `artifacts/` | Model bundles (**gitignored**) |
| `diagnostics/` | Bias / variance / decile notes |
| `prompts/` | Experiment briefs |
| `shared_scripts/` | V3 helpers (`env_loader.py`, …) |
| `scripts/` | Ad-hoc era scripts |
| `next_experiments/` | Legacy label folder only (name kept; era = Tabular Premium Signals) |

### Packages

| Package | Role |
|---|---|
| [`v24_1_kocaeli_site_merge_repair/`](source_versions/v24_1_kocaeli_site_merge_repair/README.md) | **Best Kocaeli global** |
| [`v23_basiskele_duplex_largehome_refresh/`](source_versions/v23_basiskele_duplex_largehome_refresh/README.md) | **Best refreshed Başiskele-only** |
| [`v24_kocaeli_site_aware_global_refresh/`](source_versions/v24_kocaeli_site_aware_global_refresh/README.md) | Pre-repair global sprint (superseded by V24.1 for best checkpoint) |
| [`v21_basiskele_site_project_extraction/`](source_versions/v21_basiskele_site_project_extraction/README.md) | Older Başiskele tabular reference |
| [`v20_basiskele_premium_signals/`](source_versions/v20_basiskele_premium_signals/README.md) | First site/project premium lift |
| [`v19_basiskele/`](source_versions/v19_basiskele/README.md) | Calibration / no-ridge diagnostic (closed) |

Requires repo-root `.env` with `DATABASE_URL` and the geo cache under `data/external/geo_context`.

---

## Rules of engagement

- Do **not** modify archived V17/V18 trees under `../v2/source_versions/`.
- Do **not** modify closed V19–V24 packages when iterating a new V3 version; add a new package under `source_versions/`.
- Keep `comparable_mode=none` and calibration off unless a new, explicit experiment says otherwise.
- Prefer writing tabular runs under `v3/outputs/...` (ignored by git).
- Put visual / satellite / image work under **`../v4/`** — do not mix into V3.
- Never commit `.env`, joblibs, or full output trees.
