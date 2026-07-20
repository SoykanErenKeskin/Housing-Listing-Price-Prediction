# Path migration notes (updated 2026-07-20 cleanup-2)

## Layout

| Item | Path |
|---|---|
| Thesis versions | `v1/source_versions/v1` … `v16` |
| Generation outputs | `v1/outputs/` |
| Generation scripts | `v1/scripts/` |
| Shared data | repo root `data/` |
| Shared DB config | repo root `.env` |

## Relative data paths

From `v1/source_versions/v16/`:

- Wrong: `../data` → `v1/data`
- Right: `../../../data` → repo `data/`
- Prefer absolute `--…` flags when replaying archives

## `.env`

Single root `.env` only. Do not put per-version `.env` files back.

Train scripts walk parents for `.env` (v16 may still expect local; prefer running with env already loaded or patch similarly to v17/v18).

## Outputs / scripts

Former root `outputs/` and `scripts/` were moved under `v1/outputs` and `v1/scripts`.
