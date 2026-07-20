# Best Kocaeli location checkpoint (V17)

**Era:** V2 curated pack  
**Status:** Archived reference — multi-county Kocaeli location / geo run.

## Purpose

Preserve the V17 Kocaeli-wide run that measured the effect of Başiskele-scoped
location + geo-context features on the global listing model. This pack is for
metrics, config, and reports — not a “ship-ready” production drop.

## Canonical original run

`v2/source_versions/v17/outputs/v17_basiskele_geo_full/`  
(also historically `v17/outputs/v17_basiskele_geo_full/` before the tree move)

This archive copy includes config / README / reports. Large `*.joblib` artifacts
are path-referenced only under `artifacts_PATH_ONLY/`.

## Known reference metrics

From project history for the geo full run (approximate):

| Metric | Value |
|---|---:|
| Global R2 | ≈ 0.6888 |
| Başiskele R2 | ≈ 0.4834 |
| Başiskele variance_ratio | ≈ 0.4346 |

Ship gate (every county R2 ≥ 0.65) was **not** met. See the V17 run README for
ablation arms, location-scope rules, and PASS/ideal thresholds.

## How to re-read / re-run

1. Build offline geo cache: `data/external/geo_context` (OSM or seed).
2. Prefer scripts under `v2/source_versions/v17/` with repo-root `.env`.
3. Full comparable train used `--location-feature-mode geo`,
   `--geo-context-mode geo_with_coast`, `--location-scope basiskele_only`.

Details: `v2/source_versions/v17/README_v17_run.md` (and the nested copy under
`v17_basiskele_geo_full/` if present in this pack).
