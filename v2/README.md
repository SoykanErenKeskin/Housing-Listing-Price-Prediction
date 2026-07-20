# V2 — Location + Başiskele Sandbox

Generation where **location / geo features** became first-class, including the
**Başiskele-only** research sandbox (V18).

**Path:** `./v2/`  
**Status:** **Archived / reference.** New experiments: [`../v3/`](../v3/README.md).

---

## Layout

| Folder | Role |
|---|---|
| `location_basiskele/` | Era label |
| `source_versions/v17` | Original V17 tree |
| `source_versions/v18_basiskele` | Original V18 Başiskele-only tree |
| `best_checkpoints/` | Curated V17 Kocaeli + V18 Başiskele packs (metrics/reports; joblibs gitignored) |
| `rejected_experiments/` | Pointer / notes for rejected comparable arms |
| `reports/` / `diagnostics/` / `artifacts/` | Archive copies + path refs |
| `notes/` | Copy policy / path migration notes |
| `outputs/` | Local training dumps (gitignored) |

---

## Key findings

- **V17:** location features produced meaningful lift in parts of Kocaeli; Başiskele still showed compression.
- **V18:** Başiskele-only sandbox; fold-safe comparable market features were ablated.
- **Comparable predictors rejected** — best V18 research config is geo control with `comparable_mode=none`.
- Başiskele remains **compressed** (expensive homes underpredicted / cheap overpredicted tendency).

---

## Best checkpoints

| Name | Path | Notes |
|---|---|---|
| Kocaeli location | `best_checkpoints/best_kocaeli_location_checkpoint/` | V17 |
| Başiskele-only | `best_checkpoints/best_basiskele_only_checkpoint/` | V18 geo control |

**V18 Başiskele-only reference metrics** (`comparable_mode=none`):

| Metric | Value |
|---|---|
| R² | ≈ 0.4731 |
| MAPE | ≈ 0.1093 |
| variance_ratio | ≈ 0.4264 |

Full original run trees (local, gitignored) also lived under:

- `source_versions/v17/outputs/v17_basiskele_geo_full/`
- `source_versions/v18_basiskele/outputs/v18_basiskele_full/`

---

## Rejected comparable arms

`nearest` / `similar` / `weighted` / `large_home` / `full` comparable modes did **not** beat geo control under the selection rules. See `rejected_experiments/` and V18 run README.

---

## Re-running archived V18

```powershell
# From repo root — prefer root .env
python v2/source_versions/v18_basiskele/train_v18_basiskele_comparable_pipeline.py `
  --geo-context-cache-dir data/external/geo_context `
  --out v2/outputs/v18_rerun_local `
  ...
```

Avoid relying on old `../data` relatives after the tree move. See `notes/PATH_MIGRATION.md` if present.

---

## Secrets

Repo-root `.env` only. Do not recreate per-version secret files.
