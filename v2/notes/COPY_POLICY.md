# Copy policy (V2 archive)

- Root `v17` and `v18_basiskele` left intact.
- `source_versions/` excludes `outputs/`, `__pycache__/`, `.env`, `*.joblib`.
- Reports copied under `reports/v17` and `reports/v18_basiskele`.
- Best checkpoint folders include config + reports + path refs for joblibs.
- Rejected comparable ablation arms: reports only under `best_checkpoints/rejected_experiments/`.

## Not copied due to size

- All `*.joblib` under `v17/outputs/**` and `v18_basiskele/outputs/**`
- Full train `data/` trees under those runs
- Ablation arm artifact bundles

## Comparable as predictor

Rejected. See `diagnostics/v18_comparable_ablation/` and generation README.
