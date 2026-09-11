# V25 core_good_ok — Final Post-Run Summary

**Run root:** `v3/outputs/v25_kocaeli_full_refresh_core_good_ok_full/`  
**Package:** `v3/source_versions/v25_kocaeli_full_refresh_core_good_ok/`  
**Selected experiment:** `full_v25_core_good_ok`  
**Model version:** `v25-kocaeli-full-refresh-core-good-ok`  
**Scope:** `kocaeli_global` · **County policy:** `core_good_ok`

---

## Decision status (read first)

| Statement | Status |
|---|---|
| V25 is selected **within** the same-data core_good_ok ablation | **Yes** (`full_v25_core_good_ok`) |
| V25 is promoted to production over V24.1 | **No** |
| Current deployed / app reference remains | **V24.1** `full_v24` |
| App / EDER artifacts updated from this run | **No** |
| Old V24.1 metrics comparison | **Reference only** (inventory distribution changed) |

V25 is a successful **research checkpoint** on the refreshed core_good_ok inventory. It is **not** the production deployment target until an explicit promote step.

Same-data control (`control_v24_1_replay_on_core_good_ok`) matched full_v25 on global metrics (same feature modes). Selection marks the V25 arm as the named checkpoint without claiming lift over control.

---

## Global metrics (`full_v25_core_good_ok`)

| Metric | Value |
|---|---:|
| rows (OOF sale) | 14,363 |
| R² | 0.730769 |
| log R² | 0.739968 |
| MAPE | 0.108255 |
| median APE | 0.082262 |
| variance_ratio | 0.693997 |
| large_home R² | 0.615663 |
| large_home MAPE | 0.127005 |
| expensive_decile_bias (TL/m²) | −8210.54 |
| MAE (TL/m²) | 4416.95 |
| median AE (TL/m²) | 3262.87 |

**Feature modes**

| Mode | Value |
|---|---|
| site_extraction_mode | `full` |
| site_project_encoding | `foldsafe_target` |
| duplex_feature_mode | `full` |
| attribute_mode | `full` |
| location_scope | `global` |
| comparable_mode | `none` |

**Ablation (same data)**

| Experiment | R² | MAPE | VR | selected |
|---|---:|---:|---:|---|
| control_v24_1_replay_on_core_good_ok | 0.730769 | 0.108255 | 0.693997 | no |
| full_v25_core_good_ok | 0.730769 | 0.108255 | 0.693997 | **yes** |

**Old V24.1 reference** (`v3/outputs/v24_1_kocaeli_site_merge_repair/`, different distribution, n=6,667): R² 0.652324 · MAPE 0.117628 · VR 0.631573. Do **not** treat the numeric gap as a like-for-like promote signal.

---

## County metrics

Policy counties only (WEAK excluded from train).

| County | n | R² | MAPE | median APE |
|---|---:|---:|---:|---:|
| Darıca | 3318 | 0.717736 | 0.084595 | 0.059055 |
| Körfez | 786 | 0.724805 | 0.118276 | 0.094482 |
| Çayırova | 1234 | 0.698817 | 0.096868 | 0.073706 |
| Derince | 685 | 0.692537 | 0.109091 | 0.086115 |
| İzmit | 2597 | 0.684280 | 0.130359 | 0.103794 |
| Gölcük | 1179 | 0.662000 | 0.120140 | 0.092471 |
| Gebze | 1803 | 0.577563 | 0.124817 | 0.097930 |
| Kartepe | 1008 | 0.535867 | 0.101981 | 0.082672 |
| Başiskele | 1753 | 0.506848 | 0.102066 | 0.079920 |

**Included:** Darıca, İzmit, Gebze, Başiskele, Çayırova, Gölcük, Kartepe, Körfez, Derince  
**Excluded (WEAK):** Karamürsel, Kandıra, Dilovası

Weak spots for follow-up: **Gebze** (R²), **Başiskele / Kartepe** (R² / premium spread), **İzmit** (MAPE).

---

## Feature guard status

From `reports/feature_schema_diff_v24_1_vs_v25.json`:

| Check | Result |
|---|---|
| Overall status | **PASS** |
| critical_missing_features | `[]` |
| heating categorical + Yerden leaf | present |
| duplex flags / large_home | present |
| site_project_id + foldsafe OOF | present |
| geo / kitchen / balcony / usage / bathroom / floor / county-district | present |
| v24.1 feature count → v25 | 4,807 → 6,877 (schema cols 127) |
| Missing vs v24.1 (expected) | mostly WEAK-county / old-district one-hots |
| Added in v25 | new districts/sites under refreshed inventory |

---

## Heating / Yerden Isıtma

| Item | Value |
|---|---|
| Sale Yerden raw count (policy inventory) | 4,461 / 14,800 (30.1%) |
| OOF rows with heating=Yerden Isıtma | 4,417 |
| Yerden OOF MAPE / median APE | 0.0845 / 0.0630 |
| Schema leaf `cat__heating_Yerden Isıtma` | present (importance > 0 in trees) |
| Premium heating score | present (critical check PASS) |

Highest Yerden sale share among included counties: Başiskele ~55%, Kartepe ~52%, Darıca ~51%. Lowest: Gebze ~9%.

---

## Site coverage

Global (`site_project_coverage_v24.csv`, selected arm):

| Metric | Value |
|---|---:|
| n_rows | 14,363 |
| canonical non-missing rate | 0.2167 |
| extracted raw rate | 0.2170 |
| n_canonical_sites | 1,367 |
| foldsafe encoded sites | 293 |
| top expensive underpredicted × site-missing rate | **0.676** |

By county (canonical non-missing / site_coverage):

| County | coverage | unique sites |
|---|---:|---:|
| İzmit | 0.361 | 361 |
| Kartepe | 0.341 | 172 |
| Başiskele | 0.298 | 217 |
| Gölcük | 0.210 | 161 |
| Körfez | 0.207 | 84 |
| Derince | 0.204 | 83 |
| Gebze | 0.157 | 124 |
| Çayırova | 0.151 | 75 |
| Darıca | 0.087 | 153 |

Site signal is uneven; expensive underprediction still correlates with missing site (~68% among top expensive underpredicted).

---

## Leakage status

| Guard | Pass |
|---|---|
| Site foldsafe encoding leakage guard | **true** (train-fold unit prices only; n_sites_encoded=293) |
| Comparable leakage guard | **true** (mode=`none`) |
| Ablation `leakage_pass` | **true** |
| Metrics `leakage_guard_pass` | **true** |

---

## Merge audit status

| Item | Value |
|---|---|
| severe_bad_merge | **0** |
| possible_bad_merge | **17** (non-blocking; manual review) |
| Promote blocking? | No |

---

## Limitations

1. **Not production-promoted** — V24.1 remains the deployed/app reference; no EDER/app artifact copy from this run.
2. **Distribution shift** — Refreshed inventory (~14.3k OOF vs V24.1 ~6.7k). Cross-run R²/MAPE deltas are **not** A/B evidence for promote.
3. **Control == full** on same modes — V25 selection is a named research checkpoint on new data, not a mode-ablation win over control.
4. **WEAK counties excluded** — Karamürsel / Kandıra / Dilovası not in primary train; all-counties diagnostic is deferred.
5. **Site sparsity** — Overall ~22% canonical site coverage; expensive + site-missing underprediction remains a repair target.
6. **County R² floor** — Başiskele (~0.51) and Kartepe (~0.54); Gebze (~0.58) needs segment diagnostics.
7. **Variance compression residual** — Global VR ~0.69 (better on this corpus than old V24.1 ref) but expensive-decile bias still negative (~−8.2k TL/m²).
8. **App input gap** — Yerden is in the model schema; product UI coverage was out of scope for this model-only run.

---

## Key artifacts

| Artifact | Path |
|---|---|
| Selection decision | `reports/selection_decision_v25.json` |
| Ablation metrics | `reports/metrics_v25_core_good_ok_ablation.csv` |
| County policy | `reports/county_policy_v25.json` |
| Feature schema / guards | `reports/feature_schema_diff_v24_1_vs_v25.json` |
| Heating audit | `reports/heating_feature_audit_v25.csv` |
| Model card (this package) | `model_card_v25_core_good_ok.json` |
| Selected arm metrics | `ablation_v25_full_v25_core_good_ok/reports/metrics_summary_v18_basiskele.json` |
| Site coverage | `ablation_v25_full_v25_core_good_ok/reports/site_project_coverage_v24.csv` |
| Leakage guard | `ablation_v25_full_v25_core_good_ok/reports/site_project_encoding_leakage_guard_v24.json` |
| Merge audit | `ablation_v25_full_v25_core_good_ok/reports/site_project_merge_audit_v24.csv` |

---

## Next experiments

1. **V25 residual calibration for market bands** — band-wise residual / isotonic-style calibration to shrink expensive-decile underprediction without hurting MAPE.
2. **Gebze segment diagnostics** — slice by segment, age, site-missing, and district; isolate the R² gap vs Darıca/Körfez.
3. **Başiskele / Kartepe premium spread diagnostics** — re-check spread / large-home residual layers on refreshed inventory (layers were off in this run).
4. **site-missing expensive underprediction repair** — target the ~68% site-missing rate among top expensive underpredicted rows (extraction + encoding + fallback).
5. **all-counties diagnostic** — include Karamürsel / Kandıra / Dilovası as a diagnostic arm only (do not displace core_good_ok primary until readiness improves).

---

*Generated as a post-run research summary. Does not modify training code or app/EDER artifacts.*
