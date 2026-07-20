# V16 Model Run

## Executive Summary
- overall: PASS
- selected_attribute_mode: full
- demographics_mode: safe
- R2: 0.6802697028326123 | MAPE: 0.12846728050677086 | MAE: 4836.411487606826
- v12_delta: {"r2": 0.006336449791710663, "mape": -0.000907408391324166, "mae_tl_per_m2": -42.63284687314626}
- karamursel_sale_diff_pct: 0.49772499338312154
- direction_pass_rate: 0.84375
- warnings: ['basiskele_no_r2_lift', 'basiskele_variance_no_lift', 'basiskele_compressed']

### Rent note
V16 trains sale unit-price only. If the app rent path is `district_rent_m2_median * gross_m2`,
two homes with the same m2 in the same district get the same rent even when quality differs.
A separate rent attribute multiplier belongs in a later version — do not mix into this sales model.

### Leakage checklist
- attr_effect_* fit only inside CV folds on residual target (log price - log baseline)
- no full-X precompute of target encodings
- no title/photo/description features

## Config
{
  "city": "Kocaeli",
  "counties": [
    "İzmit",
    "Başiskele",
    "Gölcük",
    "Karamürsel"
  ],
  "target_mode": "residual",
  "n_splits": 5,
  "random_state": 42,
  "sale_table": "sale_listings",
  "rental_table": "rental_listings",
  "trend_table": "trend_observed",
  "use_trend": true,
  "selected_models": [
    "ridge",
    "gradient_boosting",
    "extra_trees",
    "random_forest"
  ],
  "fast_mode": false,
  "min_sale_unit_price": 8000.0,
  "max_sale_unit_price": 200000.0,
  "min_rent_m2": 50.0,
  "max_rent_m2": 2500.0,
  "use_location_outlier_filter": true,
  "min_location_ratio": 0.5,
  "max_location_ratio": 1.9,
  "location_mad_threshold": 3.5,
  "location_min_group_size": 12,
  "enable_county_experts": true,
  "county_expert_min_rows": 180,
  "enable_anomaly_reports": true,
  "demographics_mode": "safe",
  "demographics_table": "district_demographics",
  "exclude_anomalies_threshold": 25.0,
  "attribute_mode": "full",
  "detail_effect_mode": "group",
  "county_expert_min_rows_overrides": {
    "Karamürsel": 180
  },
  "basiskele_specialist_mode": "premium_target_stats",
  "basiskele_variance_lift": "none",
  "large_home_specialist_mode": "redesigned",
  "basiskele_large_home_regime": "none",
  "basiskele_spread_layer": "none",
  "karamursel_baseline_mode": "none"
}

## Cleaning report
{
  "sales_raw_rows": 3154,
  "sales_after_base_clean_rows": 3154,
  "sales_after_basic_filter_rows": 3153,
  "sales_final_rows": 3138,
  "sales_removed_basic_rows": 1,
  "sales_removed_iqr_rows": 15,
  "rentals_raw_rows": 2229,
  "rentals_after_base_clean_rows": 2229,
  "rentals_after_basic_filter_rows": 2227,
  "rentals_final_rows": 2210,
  "rentals_removed_basic_rows": 2,
  "rentals_removed_iqr_rows": 17,
  "location_outlier_filter_enabled": true,
  "rows_before_location_filter": 3138,
  "rows_after_location_filter": 3055,
  "rows_removed_location_filter": 83,
  "min_location_ratio": 0.5,
  "max_location_ratio": 1.9,
  "location_mad_threshold": 3.5,
  "location_min_group_size": 12,
  "removed_ratio_summary": {
    "min": 0.26130501711693627,
    "median": 1.9077267198024714,
    "max": 2.7292485560937516
  }
}

## Feature reports
{
  "rental_features": {
    "rental_rows_used": 2210,
    "global_rent_m2_median": 235.29,
    "global_rent_row_count": 2210,
    "rent_feature_level_counts": {
      "district_room": 2978,
      "district_m2_group": 80,
      "district": 64,
      "county": 16
    }
  },
  "trend_features": {
    "trend_rows_used": 213,
    "trend_district_matched_rows": 3138,
    "trend_date_min": "2022-11-01",
    "trend_date_max": "2027-05-01"
  },
  "demographic_features": {
    "mode": "safe",
    "demo_rows": 488,
    "listing_rows": 3055,
    "matched_listing_rows": 3035,
    "match_rate": 0.9934533551554828,
    "county_matched_listing_rows": 3035,
    "county_match_rate": 0.9934533551554828,
    "join_method": "name_fallback",
    "county_join_method": "county_id"
  },
  "demographics_ablation": {},
  "attribute_ablation": [],
  "detail_effect_ablation": [],
  "basiskele_specialist_ablation": []
}

## Decision
{
  "pass_global_guardrail": true,
  "pass_guardrail": true,
  "pass_basiskele_lift": false,
  "pass_basiskele_variance_lift": true,
  "pass_basiskele_large_home_lift": false,
  "pass_basiskele_spread_lift": true,
  "pass_karamursel_lift": true,
  "pass_karamursel_guardrail": true,
  "pass_golcuk_guardrail": true,
  "pass_detail_sensitivity": true,
  "pass_sensitivity": true,
  "pass_karamursel": true,
  "direction_pass_rate": 0.84375,
  "karamursel_sale_diff_pct": 0.49772499338312154,
  "basiskele_r2": 0.4401912183403204,
  "basiskele_variance_ratio": 0.4432031799204622,
  "basiskele_lift": {
    "r2": 0.4401912183403204,
    "r2_delta_vs_v15": -0.013208781659679647,
    "r2_delta_vs_v14": -0.015108781659679604,
    "variance_ratio": 0.4432031799204622,
    "variance_delta_vs_v15": -0.008396820079537792
  },
  "golcuk_r2": 0.6422171306421325,
  "izmit_r2": 0.7160787405813882,
  "karamursel_r2": 0.5929954711941858,
  "county_r2_table": {
    "İzmit": 0.7160787405813882,
    "Başiskele": 0.4401912183403204,
    "Gölcük": 0.6422171306421325,
    "Karamürsel": 0.5929954711941858
  },
  "ship_ready_all_counties_r2_ge_0_65": false,
  "selected_detail_effect_mode": "group",
  "selected_basiskele_specialist_mode": "premium_target_stats",
  "selected_basiskele_variance_lift_mode": "none",
  "selected_v16_layers": {
    "basiskele_large_home_regime": "none",
    "basiskele_spread_layer": "none",
    "karamursel_baseline_mode": "none"
  },
  "disabled_layers": [
    "large_home_residual:disabled",
    "spread_residual:disabled",
    "large_home_regime_features:off",
    "karamursel_baseline:off"
  ],
  "overall": "PASS",
  "warnings": [
    "basiskele_no_r2_lift",
    "basiskele_variance_no_lift",
    "basiskele_compressed"
  ],
  "qa_findings": [
    {
      "finding": "Başiskele R2 0.4402 below V15 0.4534",
      "severity": "Medium"
    },
    {
      "finding": "Başiskele variance ratio 0.4432031799204622 below V15 0.4516",
      "severity": "Medium"
    },
    {
      "finding": "Başiskele predictions are compressed toward mean",
      "severity": "Medium"
    },
    {
      "finding": "not ship-ready until all counties R2 >= 0.65",
      "severity": "Info",
      "county_r2": {
        "İzmit": 0.7160787405813882,
        "Başiskele": 0.4401912183403204,
        "Gölcük": 0.6422171306421325,
        "Karamürsel": 0.5929954711941858
      }
    },
    {
      "finding": "PASS as experiment, NOT ship-ready.",
      "severity": "Info"
    }
  ],
  "top_risks": [],
  "top_opportunities": [],
  "v13_reference": {
    "r2": 0.68,
    "mape": 0.1279,
    "basiskele_r2": 0.4449,
    "karamursel_r2": 0.5468,
    "k180_karamursel_r2": 0.5768
  },
  "v14_reference": {
    "r2": 0.6787,
    "mape": 0.129,
    "basiskele_r2": 0.4553,
    "basiskele_mape": 0.1103,
    "basiskele_variance_ratio": 0.4224,
    "golcuk_r2": 0.6444,
    "karamursel_r2": 0.5582,
    "izmit_r2": 0.7107,
    "k180_karamursel_r2": 0.5768
  },
  "v15_reference": {
    "r2": 0.6799,
    "mape": 0.129,
    "basiskele_r2": 0.4534,
    "basiskele_mape": 0.111,
    "basiskele_variance_ratio": 0.4516,
    "basiskele_large_home_r2": 0.2396,
    "golcuk_r2": 0.6481,
    "karamursel_r2": 0.5681,
    "izmit_r2": 0.7109,
    "ship_ready_all_counties_r2_ge_0_65": false
  },
  "v13_delta": {
    "r2": 0.0002697028326122952,
    "mape": 0.0005672805067708508,
    "basiskele_r2": -0.0047087816596796395,
    "karamursel_r2": 0.04619547119418588
  },
  "v14_delta": {
    "r2": 0.001569702832612374,
    "mape": -0.000532719493229139,
    "basiskele_r2": -0.015108781659679604,
    "basiskele_variance_ratio": 0.02080317992046221,
    "karamursel_r2": 0.0347954711941858,
    "golcuk_r2": -0.0021828693578674896
  },
  "v15_delta": {
    "r2": 0.0003697028326123952,
    "mape": -0.000532719493229139,
    "basiskele_r2": -0.013208781659679647,
    "basiskele_variance_ratio": -0.008396820079537792,
    "basiskele_large_home_r2": NaN,
    "karamursel_r2": 0.02489547119418578,
    "golcuk_r2": -0.005882869357867526,
    "izmit_r2": 0.00517874058138823
  },
  "v12_delta": {
    "r2": 0.006336449791710663,
    "mape": -0.000907408391324166,
    "mae_tl_per_m2": -42.63284687314626
  },
  "experiment_note": "PASS as experiment, NOT ship-ready.",
  "selected_attribute_mode": "full"
}

## Ensemble metrics
{
  "rows": 3023,
  "r2": 0.6802697028326123,
  "log_r2": 0.7032234451113915,
  "mape": 0.12846728050677086,
  "median_ape": 0.10403618302502839,
  "mae_tl_per_m2": 4836.411487606826,
  "median_ae_tl_per_m2": 3747.6221542709027,
  "model": "county_expert_segment_aware_ensemble_v16",
  "base_weights": {
    "extra_trees": 0.33673824374131556,
    "ridge": 0.33233605216172196,
    "gradient_boosting": 0.33092570409696254
  },
  "segment_weights": {
    "large_home": {
      "extra_trees": 0.5039143422504916,
      "gradient_boosting": 0.49608565774950847
    },
    "compact_home": {
      "ridge": 0.5083967570829729,
      "extra_trees": 0.49160324291702706
    },
    "old_building": {
      "gradient_boosting": 0.5006073419929497,
      "ridge": 0.49939265800705035
    },
    "mainstream_home": {
      "gradient_boosting": 0.5016286199367098,
      "ridge": 0.4983713800632903
    }
  },
  "segment_blend_weights": {
    "large_home": 0.15,
    "compact_home": 0.5,
    "old_building": 0.5,
    "mainstream_home": 0.5
  },
  "county_weights": {
    "Başiskele": {
      "extra_trees": 0.500348125549548,
      "gradient_boosting": 0.499651874450452
    },
    "Gölcük": {
      "ridge": 0.5159309091364428,
      "gradient_boosting": 0.48406909086355726
    },
    "Karamürsel": {
      "gradient_boosting": 0.5070221115917551,
      "random_forest": 0.4929778884082448
    },
    "İzmit": {
      "ridge": 0.5004879917483969,
      "extra_trees": 0.4995120082516031
    }
  },
  "county_blend_weights": {
    "Başiskele": 0.2,
    "Gölcük": 0.5,
    "Karamürsel": 0.5,
    "İzmit": 0.2
  },
  "attribute_mode": "full",
  "detail_effect_mode": "group",
  "basiskele_specialist_mode": "premium_target_stats",
  "basiskele_variance_lift": "none",
  "basiskele_variance_lift_report": {
    "mode": "none",
    "status": "disabled",
    "lambda": 0.0,
    "basiskele_r2_before": NaN,
    "basiskele_r2_after": NaN,
    "global_mape_before": NaN,
    "global_mape_after": NaN,
    "note": "variance lift off"
  },
  "basiskele_large_home_regime": "none",
  "basiskele_spread_layer": "none",
  "karamursel_baseline_mode": "none",
  "basiskele_large_home_residual_report": {
    "status": "disabled",
    "selected_model": "",
    "selected_lambda": 0.0,
    "note": "mode=none (residual layer only runs for residual)"
  },
  "basiskele_spread_residual_report": {
    "status": "disabled",
    "selected_lambda": 0.0,
    "selected_model": "",
    "note": "spread layer off"
  },
  "target_mode": "residual",
  "n_splits": 5,
  "pass_guardrail": true,
  "pass_global_guardrail": true,
  "pass_sensitivity": true,
  "pass_basiskele_lift": false,
  "pass_basiskele_variance_lift": true,
  "pass_karamursel_lift": true,
  "pass_karamursel_guardrail": true,
  "direction_pass_rate": 0.84375,
  "karamursel_sale_diff_pct": 0.49772499338312154,
  "basiskele_variance_ratio": 0.4432031799204622,
  "demographics_mode": "safe",
  "training_rows_before_anomaly_filter": 3055,
  "training_rows_after_anomaly_filter": 3023,
  "excluded_anomaly_rows": 32,
  "exclude_anomalies_threshold": 25.0
}

## Main outputs
- data/raw/sales_raw_from_source.csv
- data/raw/rentals_raw_from_source.csv
- data/input/sales_cleaned_v16.csv
- data/input/rentals_cleaned_v16.csv
- data/output/oof_predictions_v16.csv
- reports/model_comparison_v16.csv
- reports/metrics_summary_v16.json
- reports/feature_sensitivity_v16.csv
- reports/karamursel_sensitivity_v16.csv
- reports/basiskele_variance_diagnostics_v16.csv
- reports/metrics_attribute_ablation_v16.csv
- reports/error_by_*_v16.csv
- reports/*.png
- artifacts/model_*_v16.joblib
- artifacts/model_bundle_v16.joblib
