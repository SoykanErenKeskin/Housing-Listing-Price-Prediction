# V15 Model Run

## Executive Summary
- overall: PASS
- selected_attribute_mode: full
- demographics_mode: safe
- R2: 0.6799014234780145 | MAPE: 0.12897452096852122 | MAE: 4846.12696969571
- v12_delta: {"r2": 0.005968170437112841, "mape": -0.00040016792957381364, "mae_tl_per_m2": -32.917364784262645}
- karamursel_sale_diff_pct: 0.2833842832860724
- direction_pass_rate: 0.8125
- warnings: ['basiskele_no_r2_lift']

### Rent note
V15 trains sale unit-price only. If the app rent path is `district_rent_m2_median * gross_m2`,
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
  "county_expert_min_rows": 250,
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
  "basiskele_variance_lift": "conservative",
  "large_home_specialist_mode": "redesigned"
}

## Cleaning report
{
  "sales_raw_rows": 3085,
  "sales_after_base_clean_rows": 3085,
  "sales_after_basic_filter_rows": 3084,
  "sales_final_rows": 3069,
  "sales_removed_basic_rows": 1,
  "sales_removed_iqr_rows": 15,
  "rentals_raw_rows": 2229,
  "rentals_after_base_clean_rows": 2229,
  "rentals_after_basic_filter_rows": 2227,
  "rentals_final_rows": 2210,
  "rentals_removed_basic_rows": 2,
  "rentals_removed_iqr_rows": 17,
  "location_outlier_filter_enabled": true,
  "rows_before_location_filter": 3069,
  "rows_after_location_filter": 2989,
  "rows_removed_location_filter": 80,
  "min_location_ratio": 0.5,
  "max_location_ratio": 1.9,
  "location_mad_threshold": 3.5,
  "location_min_group_size": 12,
  "removed_ratio_summary": {
    "min": 0.26130501711693627,
    "median": 1.9107096534401116,
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
      "district_room": 2911,
      "district_m2_group": 77,
      "district": 63,
      "county": 18
    }
  },
  "trend_features": {
    "trend_rows_used": 213,
    "trend_district_matched_rows": 3069,
    "trend_date_min": "2022-11-01",
    "trend_date_max": "2027-05-01"
  },
  "demographic_features": {
    "mode": "safe",
    "demo_rows": 488,
    "listing_rows": 2989,
    "matched_listing_rows": 2969,
    "match_rate": 0.9933087989294078,
    "county_matched_listing_rows": 2969,
    "county_match_rate": 0.9933087989294078,
    "join_method": "name_fallback",
    "county_join_method": "county_id"
  },
  "demographics_ablation": {},
  "attribute_ablation": [],
  "detail_effect_ablation": [],
  "basiskele_specialist_ablation": [
    {
      "basiskele_specialist_mode": "none",
      "r2": 0.6817803546818522,
      "mape": 0.12843354761832437,
      "basiskele_r2": 0.45535258969555736,
      "basiskele_mape": 0.11009245825800698,
      "basiskele_variance_ratio": 0.4323514836753057,
      "karamursel_r2": 0.5719563120644527,
      "golcuk_r2": 0.6491505584791228,
      "izmit_r2": 0.7129499345573027,
      "global_guardrail": true,
      "ship_ready_all_counties_r2_ge_0_65": false
    },
    {
      "basiskele_specialist_mode": "premium",
      "r2": 0.6809181611178677,
      "mape": 0.12855820886073815,
      "basiskele_r2": 0.453080759058645,
      "basiskele_mape": 0.11030637608835688,
      "basiskele_variance_ratio": 0.43930827670434636,
      "karamursel_r2": 0.5685983176045186,
      "golcuk_r2": 0.6497691854765888,
      "izmit_r2": 0.7123622177785733,
      "global_guardrail": true,
      "ship_ready_all_counties_r2_ge_0_65": false
    },
    {
      "basiskele_specialist_mode": "premium_target_stats",
      "r2": 0.6799014234780145,
      "mape": 0.12897452096852122,
      "basiskele_r2": 0.4534310549577666,
      "basiskele_mape": 0.11097972611521927,
      "basiskele_variance_ratio": 0.45158203475222775,
      "karamursel_r2": 0.5680658123826178,
      "golcuk_r2": 0.6480910577285173,
      "izmit_r2": 0.7108802823655818,
      "global_guardrail": true,
      "ship_ready_all_counties_r2_ge_0_65": false
    },
    {
      "basiskele_specialist_mode": "premium_target_stats_variance_lift",
      "r2": 0.6799014234780145,
      "mape": 0.12897452096852122,
      "basiskele_r2": 0.4534310549577666,
      "basiskele_mape": 0.11097972611521927,
      "basiskele_variance_ratio": 0.4515820347522278,
      "karamursel_r2": 0.5680658123826177,
      "golcuk_r2": 0.6480910577285173,
      "izmit_r2": 0.7108802823655818,
      "global_guardrail": true,
      "ship_ready_all_counties_r2_ge_0_65": false
    }
  ]
}

## Decision
{
  "pass_global_guardrail": true,
  "pass_guardrail": true,
  "pass_basiskele_lift": false,
  "pass_basiskele_variance_lift": true,
  "pass_karamursel_lift": true,
  "pass_karamursel_guardrail": true,
  "pass_golcuk_guardrail": true,
  "pass_detail_sensitivity": true,
  "pass_sensitivity": true,
  "pass_karamursel": true,
  "direction_pass_rate": 0.8125,
  "karamursel_sale_diff_pct": 0.2833842832860724,
  "basiskele_r2": 0.4534310549577666,
  "basiskele_variance_ratio": 0.45158203475222775,
  "basiskele_lift": {
    "r2": 0.4534310549577666,
    "r2_delta_vs_v14": -0.0018689450422333942,
    "variance_ratio": 0.45158203475222775,
    "variance_delta_vs_v14": 0.029182034752227748
  },
  "golcuk_r2": 0.6480910577285173,
  "izmit_r2": 0.7108802823655818,
  "karamursel_r2": 0.5680658123826178,
  "county_r2_table": {
    "İzmit": 0.7108802823655818,
    "Başiskele": 0.4534310549577666,
    "Gölcük": 0.6480910577285173,
    "Karamürsel": 0.5680658123826178
  },
  "ship_ready_all_counties_r2_ge_0_65": false,
  "selected_detail_effect_mode": "group",
  "selected_basiskele_specialist_mode": "premium_target_stats",
  "selected_basiskele_variance_lift_mode": "conservative",
  "overall": "PASS",
  "warnings": [
    "basiskele_no_r2_lift"
  ],
  "qa_findings": [
    {
      "finding": "Başiskele R2 0.4534 not above V14 0.4553",
      "severity": "Medium"
    },
    {
      "finding": "not ship-ready until all counties R2 >= 0.65",
      "severity": "Info",
      "county_r2": {
        "İzmit": 0.7108802823655818,
        "Başiskele": 0.4534310549577666,
        "Gölcük": 0.6480910577285173,
        "Karamürsel": 0.5680658123826178
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
  "v13_delta": {
    "r2": -9.857652198552724e-05,
    "mape": 0.0010745209685212032,
    "basiskele_r2": 0.00853105495776657,
    "karamursel_r2": 0.02126581238261782
  },
  "v14_delta": {
    "r2": 0.0012014234780145516,
    "mape": -2.5479031478786718e-05,
    "basiskele_r2": -0.0018689450422333942,
    "basiskele_variance_ratio": 0.029182034752227748,
    "karamursel_r2": 0.009865812382617745,
    "golcuk_r2": 0.0036910577285172996
  },
  "v12_delta": {
    "r2": 0.005968170437112841,
    "mape": -0.00040016792957381364,
    "mae_tl_per_m2": -32.917364784262645
  },
  "experiment_note": "PASS as experiment, NOT ship-ready.",
  "selected_attribute_mode": "full"
}

## Ensemble metrics
{
  "rows": 2958,
  "r2": 0.6799014234780145,
  "log_r2": 0.7026284514141967,
  "mape": 0.12897452096852122,
  "median_ape": 0.10202390511932699,
  "mae_tl_per_m2": 4846.12696969571,
  "median_ae_tl_per_m2": 3754.544280111266,
  "model": "county_expert_segment_aware_ensemble_v15",
  "base_weights": {
    "extra_trees": 0.3351258202313365,
    "ridge": 0.3335036212377416,
    "gradient_boosting": 0.3313705585309219
  },
  "segment_weights": {
    "compact_home": {
      "ridge": 0.5023197391160468,
      "extra_trees": 0.49768026088395323
    },
    "old_building": {
      "gradient_boosting": 0.5036104672083439,
      "extra_trees": 0.4963895327916561
    },
    "mainstream_home": {
      "gradient_boosting": 0.5004868111224762,
      "ridge": 0.49951318887752383
    }
  },
  "segment_blend_weights": {
    "compact_home": 0.35,
    "old_building": 0.65,
    "mainstream_home": 0.5
  },
  "county_weights": {
    "Başiskele": {
      "gradient_boosting": 0.5014992917699944,
      "extra_trees": 0.49850070823000564
    },
    "Gölcük": {
      "ridge": 0.5159309091364428,
      "gradient_boosting": 0.48406909086355726
    },
    "Karamürsel": {
      "gradient_boosting": 0.5006647611623868,
      "random_forest": 0.49933523883761327
    },
    "İzmit": {
      "ridge": 0.5004879917483969,
      "extra_trees": 0.4995120082516032
    }
  },
  "county_blend_weights": {
    "Başiskele": 0.1,
    "Gölcük": 0.5,
    "Karamürsel": 0.2,
    "İzmit": 0.2
  },
  "attribute_mode": "full",
  "detail_effect_mode": "group",
  "basiskele_specialist_mode": "premium_target_stats",
  "basiskele_variance_lift": "conservative",
  "basiskele_variance_lift_report": {
    "mode": "conservative",
    "status": "rejected_no_lift",
    "lambda": 0.1,
    "basiskele_r2_before": 0.4534310549577666,
    "basiskele_r2_after": 0.45315404940106585,
    "global_mape_before": 0.12897452096852122,
    "global_mape_after": 0.1290458704179936,
    "note": "disabled: no Başiskele R2 lift or global MAPE worsened"
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
  "direction_pass_rate": 0.8125,
  "karamursel_sale_diff_pct": 0.2833842832860724,
  "basiskele_variance_ratio": 0.45158203475222775,
  "demographics_mode": "safe",
  "training_rows_before_anomaly_filter": 2989,
  "training_rows_after_anomaly_filter": 2958,
  "excluded_anomaly_rows": 31,
  "exclude_anomalies_threshold": 25.0
}

## Main outputs
- data/raw/sales_raw_from_source.csv
- data/raw/rentals_raw_from_source.csv
- data/input/sales_cleaned_v15.csv
- data/input/rentals_cleaned_v15.csv
- data/output/oof_predictions_v15.csv
- reports/model_comparison_v15.csv
- reports/metrics_summary_v15.json
- reports/feature_sensitivity_v15.csv
- reports/karamursel_sensitivity_v15.csv
- reports/basiskele_variance_diagnostics_v15.csv
- reports/metrics_attribute_ablation_v15.csv
- reports/error_by_*_v15.csv
- reports/*.png
- artifacts/model_*_v15.joblib
- artifacts/model_bundle_v15.joblib
