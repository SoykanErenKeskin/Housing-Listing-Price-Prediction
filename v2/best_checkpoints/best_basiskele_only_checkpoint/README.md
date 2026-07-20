# Best Başiskele-only checkpoint (V18)

**Era:** V2 curated pack  
**Status:** Closed research checkpoint (2026-07-20) — not a Kocaeli-wide replacement.

## Purpose

Archive the selected V18 Başiskele-only geo control after comparable-feature ablation.
Use this pack for metrics/reports without digging through the full source tree.

## Selected config

| Field | Value |
|---|---|
| `comparable_mode` | `none` |
| `location_feature_mode` | `geo` |
| `geo_context_mode` | `geo_with_coast` |
| R2 | ≈ 0.4731 |
| MAPE | ≈ 0.1093 |
| variance_ratio | ≈ 0.4264 |
| rows (after anomaly filter) | 919 |

## Canonical dirs

Originals remain under the V18 source tree (local / often gitignored outputs):

- `v2/source_versions/v18_basiskele/outputs/v18_basiskele_full/` (final selected)
- `v2/source_versions/v18_basiskele/outputs/v18_basiskele_full/ablation_comparable_control_v17_geo/`

This pack keeps config / README / reports. Large `*.joblib` files are **not**
copied; see `artifacts_PATH_ONLY/` for path references.

## Decision summary

Comparable as predictor: **REJECTED**. Arms `nearest` / `similar` / `weighted` /
`large_home` / `full` did not beat geo control under the selection rules
(R2 > control and MAPE ≤ control + 0.005).

See [`README_v18_basiskele_run.md`](README_v18_basiskele_run.md) for the full
ablation table, reproduce commands, and report list.

## Follow-on

Active research continues in V19 (`v3/source_versions/v19_basiskele/`) with
prediction calibration / anti-shrink, keeping `comparable_mode=none`.
