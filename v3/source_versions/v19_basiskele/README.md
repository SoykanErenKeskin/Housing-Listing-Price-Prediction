# V19 Başiskele-only — Calibration / Anti-Shrink Research

**Active package:** `v3/source_versions/v19_basiskele/`

V19 continues from the closed V18 Başiskele-only geo control checkpoint and tests
**OOF-safe prediction-level calibration** (and related anti-shrink levers) to reduce
mean-pulling / variance compression.

---

## Baseline (V18 geo control, `comparable_mode=none`)

| Metric | Value |
|---|---|
| R² | ≈ 0.4731 |
| MAPE | ≈ 0.1093 |
| variance_ratio | ≈ 0.4264 |

**Main failure mode:** cheap listings overpredicted, expensive listings underpredicted (compression).

V18 sources under `v2/source_versions/v18_basiskele/` are **not modified** by V19 work.

---

## Why no comparable features

Comparable market features were ablated in V18 and **rejected as predictors**.  
V19 keeps `comparable_mode=none` (forced).

---

## Why calibration

Instead of blind residual variance lift, V19 tests **prediction calibration** fit in an OOF-safe way:

| Mode | Idea |
|---|---|
| `none` | No calibration |
| `linear` | Linear map on prediction space |
| `isotonic` | Monotone calibration |
| `bin` | Bin-wise correction |
| `quantile_map` | Quantile mapping |

Also prepared for ablation:

- **Ensemble profiles:** `balanced`, `no_ridge`, `tree_heavy`, …
- **Target profiles:** `residual_log`, `direct_price`, `hybrid`
- **Cap:** `--calibration-cap` (default ~0.10 log units) to limit aggressive corrections

---

## Leakage-safe calibration protocol

For each outer CV fold:

1. Fit the calibrator on **other folds only** (OOF predictions + actuals)
2. Transform the held-out fold
3. Persist `reports/calibration_leakage_guard_v19.json`

Never fit calibrators on the same fold’s validation actuals.

---

## Environment

From **repo root**:

1. `copy .env.example .env` and set `DATABASE_URL`
2. Ensure geo cache exists: `data/external/geo_context`
3. Run commands below from repo root (recommended)

---

## Smoke command (fast)

```powershell
python v3/source_versions/v19_basiskele/train_v19_basiskele_calibration_pipeline.py `
  --out v3/outputs/v19_basiskele_test `
  --fast --limit-sale 300 --limit-rental 300 `
  --model-scope basiskele_only `
  --location-feature-mode geo `
  --geo-context-mode geo_with_coast `
  --calibration-mode linear `
  --ensemble-profile balanced `
  --target-profile residual_log `
  --no-run-calibration-ablation `
  --use-trend --no-interactive `
  --geo-context-cache-dir data/external/geo_context
```

---

## Full calibration ablation

```powershell
python v3/source_versions/v19_basiskele/train_v19_basiskele_calibration_pipeline.py `
  --out v3/outputs/v19_basiskele_ablation `
  --model-scope basiskele_only `
  --location-feature-mode geo `
  --geo-context-mode geo_with_coast `
  --calibration-mode isotonic `
  --ensemble-profile balanced `
  --target-profile residual_log `
  --run-calibration-ablation `
  --use-trend --no-interactive `
  --geo-context-cache-dir data/external/geo_context
```

`--run-calibration-ablation` expands the run (multiple calibrator / profile arms). Expect a long wall-clock.

---

## Typical outputs (under `--out`)

| Path | Contents |
|---|---|
| `reports/metrics_summary_v19_basiskele.json` | Core metrics |
| `reports/calibration_leakage_guard_v19.json` | Leakage guard result |
| `reports/calibration_curve_v19_basiskele.csv` | Calibration curve (when applicable) |
| `reports/metrics_calibration_ablation_*.csv` | Ablation table (when enabled) |
| `data/output/oof_predictions_v19_basiskele.csv` | OOF predictions |
| `artifacts/model_bundle_v19_basiskele.joblib` | Bundle (gitignored) |

All of `outputs/` / `artifacts/` / `*.joblib` are gitignored at repo root.

---

## Success criteria (vs V18 control)

| Tier | Criteria |
|---|---|
| Minimum PASS | R² > 0.4731 **and** MAPE ≤ 0.1143 **and** leakage_guard pass |
| Good | R² ≥ 0.50 **and** variance_ratio ≥ 0.46 |
| Very good | R² ≥ 0.53 **and** variance_ratio ≥ 0.50 |

---

## Related docs

- V18 closed run notes: `README_v18_basiskele_run.md` (historical reference copy)
- Workspace map: [`../../../MODEL_WORKSPACE_INDEX.md`](../../../MODEL_WORKSPACE_INDEX.md)
- Generation README: [`../../README.md`](../../README.md)
