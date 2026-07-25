# V24.1 — Best Kocaeli global checkpoint

**Selected:** `full_v24`  
**Output:** `v3/outputs/v24_1_kocaeli_site_merge_repair/`  
**Package:** `v3/source_versions/v24_1_kocaeli_site_merge_repair/`  
**best_checkpoint:** `true`

## Metrics

| Metric | Value |
|---|---:|
| R² | 0.652324 |
| MAPE | 0.117628 |
| variance_ratio | 0.631573 |
| rows | 6667 |
| leakage_pass | true |
| merge_audit_severe_warnings / severe_bad_merge | 0 |
| possible_bad_merge | 12 (non-blocking manual review) |

## Lift vs V24 safe duplex (`duplex_largehome_global`)

| Metric | Safe duplex | V24.1 full_v24 | Δ |
|---|---:|---:|---:|
| R² | 0.635866 | 0.652324 | +0.01646 |
| MAPE | 0.120037 | 0.117628 | −0.00241 |
| variance_ratio | 0.618481 | 0.631573 | improved |

## County metrics

| County | R² | MAPE |
|---|---:|---:|
| İzmit | 0.682681 | 0.129376 |
| Gölcük | 0.670027 | 0.118001 |
| Başiskele | 0.511732 | 0.102983 |
| Karamürsel | 0.578453 | 0.158589 |
| Kartepe | 0.550115 | 0.101380 |

## Decision

Promote V24.1 as the **best Kocaeli global checkpoint**.

- Site/project extraction is now validated globally after merge repair.
- `severe_bad_merge = 0`.
- Keep `possible_bad_merge = 12` as non-blocking manual review candidates.
- **V23** remains the best refreshed Başiskele-only checkpoint.
- **V22** remains diagnostic no-lift.
