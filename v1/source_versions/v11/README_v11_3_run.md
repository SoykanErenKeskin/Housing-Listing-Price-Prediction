# V11.3 demographics pipeline hotfix

**Era:** V1 (`v1/source_versions/v11/`)  
**Status:** Historical hotfix over V11.2 demographic feature construction.

## Fix

- `build_demographic_features()` no longer computes `demo_coverage_score` using
  `demo_has_county_demographics` before that column exists.
- `demo_has_county_demographics` is created after county aggregates are built
  from neighborhood-level `district_demographics` rows.
- `demo_coverage_score` is recalculated once both district-level and
  county-level availability flags exist.

## Fast test

```bash
python train_v11_3_demographics_pipeline.py ^
  --out outputs/v11_3_test ^
  --fast ^
  --limit-sale 800 ^
  --limit-rental 800 ^
  --demographics-mode safe ^
  --no-run-demographics-ablation
```

Then check:

```text
outputs/v11_3_test/reports/metrics_summary_v11.json
```

Look for:

- `demographic_features.match_rate`
- `demographic_features.county_match_rate`
- `demographic_features.county_join_method`

`county_match_rate` matters most here: county demo features are aggregated from
neighborhood rows, then joined by county.

## When to use this package

Prefer V11.3 over earlier V11.x trees whenever demographics join / coverage
scoring is part of the experiment. Later V12+ packages build on this hotfix line.
