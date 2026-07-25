# V23 Başiskele Duplex / Large-Home Refresh

**Package:** `v3/source_versions/v23_basiskele_duplex_largehome_refresh/`  
**Status:** **Selected checkpoint for refreshed Başiskele inventory (new-data)**  
**Output:** `v3/outputs/v23_basiskele_duplex_largehome_refresh_full/`  
**Selected experiment:** `duplex_interactions` (`--duplex-feature-mode interactions`)  
**Base stack:** V21 site extraction (full + foldsafe_target), comparable/calibration/satellite off  

**Primary comparison:** `control_v21_on_new_data` (same V21 stack on refreshed inventory)  
**Secondary reference only:** old V21 on older distribution — R²=0.5059 / MAPE=0.1055 / VR=0.4590  

V22 satellite (`v4/`) is closed diagnostic — do not touch / do not use as base.

---

## Selected checkpoint (new-data)

| Field | Value |
|---|---|
| Output | `v3/outputs/v23_basiskele_duplex_largehome_refresh_full/` |
| selected_experiment | `duplex_interactions` |
| duplex_feature_mode | `interactions` |
| site_extraction_mode | `full` |
| site_project_encoding | `foldsafe_target` |
| R² | 0.4918 |
| MAPE | 0.1039 |
| variance_ratio | 0.4774 |
| large_home_r2 | 0.3542 |
| expensive_decile_bias | −9417 |
| leakage_guard | pass |

### vs `control_v21_on_new_data`

| Metric | Control | V23 duplex_interactions |
|---|---:|---:|
| R² | 0.4833 | **0.4918** |
| MAPE | 0.1048 | **0.1039** |
| variance_ratio | 0.4593 | **0.4774** |
| expensive_decile_bias | −9676 | **−9417** |
| large_home_r2 | 0.3469 | **0.3542** |

**Decision:** V23 `duplex_interactions` is the selected model for the **refreshed** Başiskele dataset. Lift vs new-data control is **modest but consistent** (duplex-aware): better R²/MAPE, higher variance ratio, slightly better expensive-decile bias and large-home R². Do **not** treat this as a large breakthrough.

**Do not compare blindly to old V21 (0.5059):** that score was on an older data distribution. Primary yardstick is `control_v21_on_new_data`.

### Known open issues

- Expensive top-decile underprediction persists.
- Garden duplex segment remains difficult.

Ablation: `v3/outputs/v23_basiskele_duplex_largehome_refresh_full/reports/metrics_duplex_ablation_v23_basiskele.csv`

---

## Goal

Measure contribution of newly added Başiskele duplex / large-home inventory on the V21 stack using **controlled** duplex features only (no raw title/detail text in the model).

## Duplex extraction

1. Parse `detail_konut_tipi` first (structured: Çatı / Bahçe / Ara Kat Dubleks).
2. Parse `title` second (ASCII-folded Turkish).
3. On conflict → detail wins (`duplex_match_source=conflict_detail_wins`).
4. Controlled outputs: `is_duplex`, roof/garden/middle_floor/standard flags, `duplex_type`, `duplex_match_source`, `duplex_source_conflict`.

## Frozen stack

- `model_scope=basiskele_only`
- `location_feature_mode=geo`
- `geo_context_mode=geo_with_coast`
- `site_extraction_mode=full`
- `site_project_encoding=foldsafe_target`
- comparable / calibration / satellite / no-ridge: **off**

## Selection gates (vs new-data control)

Select V23 only if an arm beats `control_v21_on_new_data` on:

- higher R²
- MAPE ≤ control + 0.005
- `large_home_r2` improves
- expensive-decile bias not materially worse
- no leakage

Do **not** promote solely by beating old V21 scores (distribution changed).

## Smoke

```powershell
python v3/source_versions/v23_basiskele_duplex_largehome_refresh/train_v23_basiskele_duplex_pipeline.py `
  --out v3/outputs/v23_basiskele_duplex_smoke `
  --fast --limit-sale 400 --limit-rental 300 `
  --model-scope basiskele_only `
  --location-feature-mode geo --geo-context-mode geo_with_coast `
  --site-extraction-mode full --site-project-encoding foldsafe_target `
  --duplex-feature-mode flags --no-run-duplex-ablation `
  --use-trend --no-interactive `
  --geo-context-cache-dir data/external/geo_context
```

## Full

```powershell
python v3/source_versions/v23_basiskele_duplex_largehome_refresh/train_v23_basiskele_duplex_pipeline.py `
  --out v3/outputs/v23_basiskele_duplex_largehome_refresh_full `
  --model-scope basiskele_only `
  --location-feature-mode geo --geo-context-mode geo_with_coast `
  --site-extraction-mode full --site-project-encoding foldsafe_target `
  --duplex-feature-mode full --run-duplex-ablation `
  --use-trend --no-interactive `
  --geo-context-cache-dir data/external/geo_context
```

Reproduce selected arm only:

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

## Key reports

- `duplex_extraction_audit_v23.csv`
- `duplex_source_coverage_v23.csv`
- `metrics_duplex_ablation_v23_basiskele.csv`
- `metrics_summary_v23_basiskele.json`
- `selection_decision_v23_new_data.json`
