# V11.3 demographics pipeline hotfix

This is a hotfix over V11.2.

Fix:
- `build_demographic_features()` no longer tries to calculate `demo_coverage_score` using `demo_has_county_demographics` before that column exists.
- `demo_has_county_demographics` is now created after county aggregate features are computed from neighborhood-level `district_demographics` rows.
- `demo_coverage_score` is recalculated after both district-level and county-level demographic availability flags exist.

Fast test:

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

The important part for your current setup is `county_match_rate`, because county demographic features are computed by aggregating neighborhood demographic rows and then joined by county.
