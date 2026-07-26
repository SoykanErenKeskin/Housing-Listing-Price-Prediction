# V24 Kocaeli Site-Aware Global Refresh

**Package:** `v3/source_versions/v24_kocaeli_site_aware_global_refresh/`  
**Status:** Closed research sprint — superseded for best checkpoint by **V24.1**  
(`v3/source_versions/v24_1_kocaeli_site_merge_repair/`, selected `full_v24`)  
**Output:** `v3/outputs/v24_kocaeli_site_aware_global_refresh_full/`  
**Base:** V23 duplex + V21 site stack, generalized to Kocaeli multi-county  

V21/V23 sources are untouched. V4 satellite is out of scope.  
Keep this package as the pre-repair global ablation archive; do not promote over V24.1.

---

## Goal

Carry Başiskele-proven site/project identity + duplex/large-home features to a **Kocaeli global** model.

Counties (default): Başiskele, İzmit, Gölcük, Karamürsel, Kartepe (+ any extra Kocaeli counties present in pull are reported separately).

## County-scoped site IDs

Canonical format: `{county_slug}::{site_project_slug}`  
Examples: `basiskele::zeray_perla`, `izmit::x_site`

Başiskele curated dictionary is preserved then namespaced. Other counties use conservative extraction + frequency IDs.

## Frozen stack

- `model_scope=kocaeli_global`
- `location_feature_mode=geo`
- `geo_context_mode=geo_with_coast`
- `location_scope=global`
- site extraction enabled + `foldsafe_target` (full)
- duplex/largehome enabled
- comparable / calibration / satellite / no-ridge: **off**

## Selection (vs same-data control)

Primary: `control_v17_or_v21_like_global` on the same Kocaeli pull.  
Do not compare blindly to old Başiskele-only V23 or old V17.

Gates: global R²↑, MAPE ≤ control+0.005, Başiskele not materially worse, İzmit/Gölcük stable/improve, Karamürsel not severely worse, leakage pass, severe bad merge = 0.

## Smoke

```powershell
python v3/source_versions/v24_kocaeli_site_aware_global_refresh/train_v24_kocaeli_site_pipeline.py `
  --out v3/outputs/v24_kocaeli_site_smoke `
  --fast --limit-sale 1200 --limit-rental 800 `
  --model-scope kocaeli_global `
  --location-feature-mode geo --geo-context-mode geo_with_coast `
  --site-extraction-mode full --site-project-encoding frequency `
  --duplex-feature-mode flags --no-run-site-ablation `
  --use-trend --no-interactive `
  --geo-context-cache-dir data/external/geo_context
```

## Full

```powershell
python v3/source_versions/v24_kocaeli_site_aware_global_refresh/train_v24_kocaeli_site_pipeline.py `
  --out v3/outputs/v24_kocaeli_site_aware_global_refresh_full `
  --model-scope kocaeli_global `
  --location-feature-mode geo --geo-context-mode geo_with_coast `
  --site-extraction-mode full --site-project-encoding foldsafe_target `
  --duplex-feature-mode full --run-site-ablation `
  --use-trend --no-interactive `
  --geo-context-cache-dir data/external/geo_context
```

## App export

`artifacts/site_project_options_kocaeli_v24.json` (+ `.csv`) — county/district site picker options. Unknown manual entry → `manual_unknown` (no foldsafe known-site encoding).
