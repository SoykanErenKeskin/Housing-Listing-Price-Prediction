# V4 — Visual / Satellite Experiments

Scope:
This generation is for satellite imagery, map tiles, visual neighborhood features, image embeddings, and image-based real estate experiments.

Separation from V3:
- **V3 — Tabular Premium Signals** (`./v3/`, short name `v3_tabular_premium_signals`) holds V19–V21 tabular / site-project work.
- **V4** explores satellite / static-map / image embeddings and must **not** modify V3.

Base checkpoint (from V3):
V21 Başiskele site/project extraction model:
- R2 = 0.5059
- MAPE = 0.1055
- variance_ratio = 0.4590
- selected = full_v21 / interactions_foldsafe
- path = `v3/outputs/v21_basiskele_site_extraction_full`

Reason:
V3 improved tabular premium signals using site/project extraction. V4 explores whether visual neighborhood signals from satellite/static map imagery can further explain Başiskele premium pricing and expensive underprediction.

Rules:
- Do not modify v1/v2/v3 source trees.
- Do not put image experiments under v3.
- Do not fine-tune CNNs in the first pilot.
- Start with image cache + pretrained embeddings + simple image stats.
- Best checkpoint remains V21 until V4 beats it.

---

## Layout

| Path | Role |
|---|---|
| `source_versions/` | Version packages (first: V22) |
| `outputs/` | Run outputs / ablation packs |
| `reports/` | Cross-run summaries |
| `artifacts/` | Lightweight serialized artifacts (not full image dumps) |
| `diagnostics/` | Visual / leakage / coverage diagnostics |
| `prompts/` | Prompt notes for visual experiment design |
| `shared_scripts/` | V4-local helpers (do not change v3) |
| `image_cache_reference/` | Small reference manifests / sample pointers for image cache |

## Paths for V22

| Role | Path |
|---|---|
| Package | `v4/source_versions/v22_basiskele_satellite_visual_pilot/` |
| Outputs | `v4/outputs/v22_basiskele_satellite_visual_pilot/` |
| Image cache references | `v4/image_cache_reference/` |
| Large image cache (shared data) | `data/external/satellite_cache/basiskele/` |

V4 may **read** V21 as base checkpoint (`v3/outputs/v21_basiskele_site_extraction_full`) but must not modify V3 files.
