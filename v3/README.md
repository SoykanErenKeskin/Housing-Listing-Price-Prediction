# V3 — Next Experiments

Active home for **V19+** research after the V18 Başiskele geo-control plateau.

**Path:** `./v3/`  
**Status:** **Active development.**

---

## Research goal

Reduce Başiskele **mean-pulling / variance compression** without bringing back
comparable-market predictors (rejected in V18).

Priority themes:

1. OOF-safe prediction calibration (`none` / `linear` / `isotonic` / `bin` / `quantile_map`)
2. Ensemble profiles (`balanced`, `no_ridge`, `tree_heavy`, …)
3. Target profiles (`residual_log`, `direct_price`, `hybrid`)
4. Leakage guards and decile / variance diagnostics

---

## Folders

| Path | Use |
|---|---|
| `next_experiments/` | Era label |
| `source_versions/` | V19+ packages (code) |
| `outputs/` | Training run outputs (**gitignored**) |
| `reports/` | Curated metrics / ablation tables |
| `artifacts/` | Model bundles (**gitignored**) |
| `diagnostics/` | Bias / variance / decile notes |
| `prompts/` | Experiment briefs |
| `shared_scripts/` | V3 helpers (`env_loader.py`, …) |
| `scripts/` | Ad-hoc era scripts |

---

## Active package: V19 Başiskele

See [`source_versions/v19_basiskele/README.md`](source_versions/v19_basiskele/README.md).

### Full calibration ablation (from repo root)

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

Requires repo-root `.env` with `DATABASE_URL` and the geo cache under `data/external/geo_context`.

---

## Rules of engagement

- Do **not** modify archived V17/V18 trees under `../v2/source_versions/` for V19 iteration.
- Keep `comparable_mode=none` unless a new, explicit experiment says otherwise.
- Prefer writing all runs under `v3/outputs/...` (ignored by git).
- Never commit `.env`, joblibs, or full output trees.
