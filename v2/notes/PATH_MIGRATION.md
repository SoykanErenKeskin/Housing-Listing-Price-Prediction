# Path migration notes (updated 2026-07-20 cleanup-2)

## Layout

| Item | Path |
|---|---|
| V17 / V18 trees | `v2/source_versions/v17`, `v2/source_versions/v18_basiskele` |
| Generation outputs | `v2/outputs/` (e.g. `v17_test`) |
| Generation scripts | `v2/scripts/` |
| Shared data | repo root `data/` |
| Shared DB config | repo root `.env` |

## Relative data / geo paths

From `v2/source_versions/v18_basiskele/`:

- Wrong: `../data/external/geo_context`
- Right: `../../../data/external/geo_context`
- Prefer absolute `--geo-context-cache-dir`

## `.env`

Root `.env` is canonical. V17/V18 train scripts walk parents for `.env`.

```powershell
cd v2\source_versions\v18_basiskele
python train_v18_basiskele_comparable_pipeline.py --geo-context-cache-dir "...\data\external\geo_context" ...
```

## Outputs / scripts

Former root `outputs/v17*` → `v2/outputs/`.  
Former root location/geo/comparable scripts → `v2/scripts/`.
