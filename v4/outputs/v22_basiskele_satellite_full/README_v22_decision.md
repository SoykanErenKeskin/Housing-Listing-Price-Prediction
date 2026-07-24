# V22 decision — DIAGNOSTIC_NO_LIFT

**Date:** 2026-07-24

Sentinel satellite/environment CSV features did **not** improve the V21 Başiskele checkpoint.

## Best checkpoint (unchanged)

`v3/outputs/v21_basiskele_site_extraction_full/`

| Metric | V21 ref |
|---|---:|
| R² | 0.5059 |
| MAPE | 0.1055 |
| VR | 0.4590 |

## This run

`v4/outputs/v22_basiskele_satellite_full/`

| Experiment | R² | MAPE | VR | selected |
|---|---:|---:|---:|---|
| control_v21 | 0.4813 | 0.1084 | 0.4488 | yes |
| sat_basic_250m | 0.4827 | 0.1083 | 0.4491 | no |
| sat_radii | 0.4801 | 0.1086 | 0.4513 | no |
| sat_full | 0.4829 | 0.1086 | 0.4462 | no |

## Policy

- Do **not** promote V22.
- Do **not** update MODEL_WORKSPACE_INDEX / MANIFEST best checkpoint to V22.
- Keep V21 as best tabular checkpoint.
- Mark V22 as diagnostic / no-lift.

## Caveat

V22 `control_v21` did not reproduce the exact V21 reference score. Treat V22 as a diagnostic satellite experiment, not a replacement benchmark. Satellite arms only showed tiny lift over the V22 internal control; none beat the real V21 reference.

## Interpretation

Free Sentinel-2 environment features at this resolution did not add meaningful predictive value over V21’s tabular + location + site/project features for Başiskele.
