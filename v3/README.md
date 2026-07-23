# V3 — Next Experiments

Home for **V19+** research after the V18 Başiskele geo-control plateau.

**Path:** `./v3/`  
**Status:** Active. **Best known Başiskele checkpoint is V21.**

---

## Best known Başiskele checkpoint — V21

| Field | Value |
|---|---|
| Output | [`outputs/v21_basiskele_site_extraction_full/`](outputs/v21_basiskele_site_extraction_full/) |
| Package | [`source_versions/v21_basiskele_site_project_extraction/`](source_versions/v21_basiskele_site_project_extraction/README.md) |
| selected_experiment | `full_v21` (≡ `interactions_foldsafe`) |
| site_extraction_mode | `full` |
| site_project_encoding | `foldsafe_target` |
| R² | 0.5059 |
| MAPE | 0.1055 |
| variance_ratio | 0.4590 |
| large_home_r2 | 0.309 |
| canonical_non_missing | 34.3% |
| dict_hit | 15.5% |
| severe_bad_merge | 0 |
| expensive_decile_bias | −10159 |

**Decision:** V21 **replaces V20** as best known Başiskele checkpoint.  
V20 remains the evidence baseline that site/project identity is the useful premium signal.

| | V20 | V21 selected |
|---|---:|---:|
| R² | 0.5017 | **0.5059** |
| MAPE | 0.1060 | **0.1055** |
| variance_ratio | 0.4616 | 0.4590 |
| large_home_r2 | — | **0.309** |
| canonical coverage | ~20% | **34.3%** |
| dict_hit | ~6.7% | **15.5%** |
| expensive_decile_bias | −10139 | −10159 |

**Caveat:** expensive decile bias is slightly worse than V20. Gap is small; not a reject. **Expensive bias is not solved.**

Comparable / calibration / no-ridge remain **rejected**.

---

## Research goal

Reduce Başiskele **mean-pulling / variance compression** without bringing back
comparable-market predictors (V18) or rejected calibration/no_ridge finals (V19).

### Open issue

Expensive-segment underprediction (decile bias) remains after V21.

### Closed / rejected

1. Comparable-market predictors as final (V18)
2. OOF-safe prediction calibration as final (V19)
3. `no_ridge` ensemble as final (V19)

Optional / research-only only if they beat V21:

- Target profiles (`direct_price`, `hybrid`)

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

## Active package: V21 Başiskele Site Extraction

See [`source_versions/v21_basiskele_site_project_extraction/README.md`](source_versions/v21_basiskele_site_project_extraction/README.md).

### Reproduce selected V21 checkpoint (from repo root)

```powershell
python v3/source_versions/v21_basiskele_site_project_extraction/train_v21_basiskele_site_pipeline.py `
  --out v3/outputs/v21_basiskele_site_extraction_full `
  --model-scope basiskele_only `
  --location-feature-mode geo `
  --geo-context-mode geo_with_coast `
  --site-extraction-mode full `
  --site-project-encoding foldsafe_target `
  --comparable-mode none `
  --run-site-ablation `
  --use-trend --no-interactive `
  --geo-context-cache-dir data/external/geo_context
```

Requires repo-root `.env` with `DATABASE_URL` and the geo cache under `data/external/geo_context`.

### Prior packages

- V20 (superseded for best checkpoint): [`source_versions/v20_basiskele_premium_signals/`](source_versions/v20_basiskele_premium_signals/README.md)
- V19 (closed diagnostic): [`source_versions/v19_basiskele/`](source_versions/v19_basiskele/README.md)

---

## Rules of engagement

- Do **not** modify archived V17/V18 trees under `../v2/source_versions/` for V19+ iteration.
- Do **not** modify V19/V20 packages when iterating beyond V21.
- Keep `comparable_mode=none` and calibration off unless a new, explicit experiment says otherwise.
- Prefer writing all runs under `v3/outputs/...` (ignored by git).
- Never commit `.env`, joblibs, or full output trees.
