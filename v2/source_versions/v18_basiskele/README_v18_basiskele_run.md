# V18 Başiskele-only comparable pipeline

**Era:** V2 (Başiskele-only research sandbox)  
**Status: CLOSED** (2026-07-20)

**This model does not replace the Kocaeli-wide model.** It is a Başiskele-only
research checkpoint.

## Purpose

Test whether fold-safe **comparable market features** (nearest / similar /
weighted / large_home / full) improve Başiskele predictions beyond the V17-style
geo + coast control, on a county-scoped training set.

## Final V18 Başiskele-only checkpoint

| Field | Value |
|---|---|
| `comparable_mode` | `none` |
| `location_feature_mode` | `geo` |
| `geo_context_mode` | `geo_with_coast` |
| R2 | **0.4731** |
| MAPE | **0.1093** |
| variance_ratio | **0.4264** |
| rows (after anomaly filter) | 919 |

### Best artifact

Canonical pointer: [`V18_BEST_CHECKPOINT.json`](V18_BEST_CHECKPOINT.json)

| Role | Path |
|---|---|
| **Best / final model** | `outputs/v18_basiskele_full/` (`comparable_mode=none`) |
| Ablation arm (same config) | `outputs/v18_basiskele_full/ablation_comparable_control_v17_geo/` |
| Bundle | `artifacts/model_bundle_v18_basiskele.joblib` |
| Metrics | `reports/metrics_summary_v18_basiskele.json` |
| Ablation table | `reports/metrics_comparable_ablation_v18_basiskele.csv` |

Nearest / similar / weighted / large_home / full comparable modes are **not**
included in the final model.

Curated pack (metrics/reports; large joblibs path-only):  
`v2/best_checkpoints/best_basiskele_only_checkpoint/`

## Comparable as predictor rejected

Fold-safe comparable market features (`nearest`, `similar`, `weighted`,
`large_home`, `full`) were ablated against `control_v17_geo`
(`comparable=none`, geo + coast).

**Result: no comparable mode beat control.**

| experiment | comparable_mode | R2 | MAPE | variance_ratio | selected |
|---|---|---:|---:|---:|---|
| control_v17_geo | none | **0.4731** | **0.1093** | **0.4264** | **True** |
| large_home_only | large_home | 0.4713 | 0.1097 | 0.4164 | False |
| weighted_only | weighted | 0.4693 | 0.1099 | 0.4210 | False |
| nearest_only | nearest | 0.4690 | 0.1096 | 0.4187 | False |
| similar_only | similar | 0.4640 | 0.1104 | 0.4103 | False |
| full_comparable | full | 0.4359 | 0.1128 | 0.3851 | False |

Selection rule required R2 > control and MAPE ≤ control + 0.005. `nearest_only`
was already disqualified from a prior full nearest run (no lift). Every active
comparable arm also hurt variance_ratio vs the geo control reference.

**Decision**

- Comparable features are **rejected as model predictors** for V18.
- Default / shipped V18 Başiskele research config stays `comparable_mode=none`.
- Comparable code may remain for **diagnostic / confidence** side channels only
  (coverage, leakage guards, optional post-hoc comps) — not as training inputs
  to the final ensemble.

## Scope

- DB pull is **only** `county = 'Başiskele'`.
- İzmit / Gölcük / Karamürsel are **not** included.
- County expert layer is **off**.
- Location: `geo` + `geo_with_coast` (default).
- Comparable adder exists but is **off** in the closed checkpoint.

## Reproduce final checkpoint

```powershell
cd v18_basiskele
python train_v18_basiskele_comparable_pipeline.py --out outputs/v18_basiskele_full --model-scope basiskele_only --location-feature-mode geo --geo-context-mode geo_with_coast --comparable-mode none --no-run-comparable-ablation --use-trend --no-interactive --geo-context-cache-dir ../data/external/geo_context
```

After the repo layout move, prefer:

```powershell
python v2/source_versions/v18_basiskele/train_v18_basiskele_comparable_pipeline.py `
  --geo-context-cache-dir data/external/geo_context `
  --out v2/outputs/v18_rerun_local `
  --model-scope basiskele_only `
  --location-feature-mode geo `
  --geo-context-mode geo_with_coast `
  --comparable-mode none `
  --no-run-comparable-ablation `
  --use-trend `
  --no-interactive
```

## Comparable ablation (already completed; do not re-run unless intentional)

```powershell
python train_v18_basiskele_comparable_pipeline.py --out outputs/v18_basiskele_full --model-scope basiskele_only --location-feature-mode geo --geo-context-mode geo_with_coast --comparable-mode full --run-comparable-ablation --use-trend --no-interactive --geo-context-cache-dir ../data/external/geo_context
```

## Minimum reports (core)

- `reports/metrics_summary_v18_basiskele.json`
- `reports/metrics_comparable_ablation_v18_basiskele.csv`
- `reports/model_comparison_v18_basiskele.csv`
- `reports/comparable_leakage_guard_v18.json`
- `reports/comparable_feature_coverage_v18.csv`

## References

- V17 Başiskele (multi-county geo run, county slice): R2 ≈ 0.4834, variance_ratio ≈ 0.4346
- V18 geo control (Başiskele-only): R2 = 0.4731, MAPE = 0.1093, VR = 0.4264
- Ship gate (Başiskele-only research): R2 ≥ 0.65 — **not met**; V18 remains a
  research checkpoint, not a Kocaeli replacement.

Follow-on (historical → current): V19 calibration diagnostic and V20/V21
tabular premium / site-project modeling live under `v3/` (**Tabular Premium
Signals**; best = V21). Visual/satellite work continues under `v4/`.
