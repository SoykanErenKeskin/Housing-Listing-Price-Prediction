# V3 — Tabular Premium Signals

**Path:** `./v3/`  
**Short name:** `v3_tabular_premium_signals`  
**Status:** Best tabular Başiskele checkpoint (V21)

This generation is for **Başiskele-only tabular premium signal** experiments —
site/project extraction and premium segment features — not a generic “next
experiments” bucket.

- **V19:** calibration / no-ridge diagnostic; final seçilmedi.
- **V20:** site/project identity ilk anlamlı premium lift verdi.
- **V21:** site/project extraction coverage iyileşti ve yeni best tabular checkpoint oldu.
- **V3** is not a catch-all for future work; **V4** is separate for visual/satellite/image experiments.

---

## Version summary

| Version | Purpose | Result |
|---|---|---|
| V19 | calibration/no-ridge diagnostic | rejected; no candidate beat control |
| V20 | premium/site project first model | R2 0.5017, MAPE 0.1060 |
| V21 | improved site/project extraction | R2 0.5059, MAPE 0.1055, **best tabular** |

---

## Best tabular checkpoint — V21

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

**Decision:** V21 replaces V20 as best tabular Başiskele checkpoint.  
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

**Known issue:** expensive decile underprediction persists. Site/project extraction improved overall score but did not fully solve premium top-decile bias.

Comparable / calibration / no-ridge remain **rejected**.

---

## Rejected / diagnostic

1. V19 isotonic / linear calibration (diagnostic; not final)
2. V19 `no_ridge` ensemble (not final)
3. Comparable-market predictors (rejected in V2; stay off in V3)
4. V20 text flags alone — small lift; site/project identity was stronger

---

## Folders

| Path | Use |
|---|---|
| `source_versions/` | V19–V21 packages (Başiskele tabular premium modeling) |
| `outputs/` | Training run outputs (**gitignored**) |
| `reports/` | Curated metrics / ablation tables |
| `artifacts/` | Model bundles (**gitignored**) |
| `diagnostics/` | Bias / variance / decile notes |
| `prompts/` | Experiment briefs |
| `shared_scripts/` | V3 helpers (`env_loader.py`, …) |
| `scripts/` | Ad-hoc era scripts |
| `next_experiments/` | Legacy label folder only (name kept; era = Tabular Premium Signals) |

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

- Do **not** modify archived V17/V18 trees under `../v2/source_versions/`.
- Do **not** modify V19/V20 packages when iterating beyond V21 inside V3.
- Keep `comparable_mode=none` and calibration off unless a new, explicit experiment says otherwise.
- Prefer writing tabular runs under `v3/outputs/...` (ignored by git).
- Put visual / satellite / image work under **`../v4/`** — do not mix into V3.
- Never commit `.env`, joblibs, or full output trees.
