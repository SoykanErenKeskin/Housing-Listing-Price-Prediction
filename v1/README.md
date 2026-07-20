# V1 — Thesis Legacy

Classic **Kocaeli multi-county** listing price models from the thesis era (through V16).

**Path:** `./v1/`  
**Status:** **Archived** — do not start new model development here. Active work: [`../v3/`](../v3/README.md).

---

## What this generation contains

| Folder | Role |
|---|---|
| `thesis_legacy/` | Era label |
| `source_versions/` | **Original** trees `v1` … `v16` (including `v6.1 tez sonu`, `v9.1`) |
| `best_checkpoints/` | Manual-review candidates (V15/V16 report packs) |
| `reports/` | Selected report copies |
| `artifacts/` | Path references for large joblibs (binaries gitignored) |
| `notes/` | Archive / reorg / path-migration notes |
| `unclassified/` | Unsorted historical material |
| `scripts/` | Era helper scripts (if present) |
| `outputs/` | Local training dumps (gitignored) |

Primary focus was **listing / attribute / segment / county-expert** modeling — **not** location/geo as the main research axis.

---

## Checkpoint status

- Late classic candidates: **V15 / V16**
- No single “official best” artifact was locked
- Review pack: `best_checkpoints/manual_review_v15_v16/` (**manual review needed**)

---

## Known issues

- Başiskele often showed **low R²** and **variance compression**
- Ship-style gate “all counties R² ≥ 0.65” was not met in late thesis runs

---

## How to re-run an archived version

Version trees were moved from the old repo root. Example for V16:

```powershell
cd v1\source_versions\v16
# Prefer root .env (do not copy secrets into version folders)
# From repo root, DATABASE_URL is loaded by walking parents when supported.
python train_v16_regime_residual_pipeline.py --help
```

**Path caveat:** scripts that still use relative `../data` assume the old cwd `repo/v16`.  
From `repo/v1/source_versions/v16`, that relative path breaks.

**Practical fix:** pass absolute paths or repo-root relative flags, e.g.:

```text
--geo-context-cache-dir "<repo-root>\data\external\geo_context"
```

Details: `notes/PATH_MIGRATION.md` (if present).

---

## Secrets

Use the **repo-root** `.env` (`DATABASE_URL`). Never commit `.env`. See [`.env.example`](../.env.example).
