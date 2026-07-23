# V22 — Başiskele Satellite Visual Pilot

**Generation:** `v4/` (Visual / Satellite Experiments)  
**Status:** scaffold / planned  
**Base checkpoint:** V21 from **V3 — Tabular Premium Signals**  
(`v3/outputs/v21_basiskele_site_extraction_full`, selected `full_v21` / `interactions_foldsafe`)

## Intent

First V4 experiment: add satellite / static-map visual neighborhood signals on top of the V21 tabular + site/project baseline — without modifying V3 source trees.

## Paths

| Role | Path |
|---|---|
| Package | `v4/source_versions/v22_basiskele_satellite_visual_pilot/` |
| Outputs | `v4/outputs/v22_basiskele_satellite_visual_pilot/` |
| Image cache references | `v4/image_cache_reference/` |
| Large image cache | `data/external/satellite_cache/basiskele/` |

## Pilot constraints

- No CNN fine-tuning in the first pilot
- Start with image cache + pretrained embeddings + simple image stats
- Guard against spatial leakage
- Do not mark as best until metrics beat V21
