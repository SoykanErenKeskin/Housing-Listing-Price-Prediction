# V16 Model Run

## Executive Summary
- overall: FAIL
- selected_attribute_mode: full
- demographics_mode: safe
- R2: 0.4639455619857922 | MAPE: 0.11058232849855693 | MAE: 4555.447227955811
- v12_delta: {}
- karamursel_sale_diff_pct: 0.31074221507712513
- direction_pass_rate: 1.0
- warnings: ['global_guardrail_failed']

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
    "Başiskele"
  ],
  "target_mode": "residual",
  "n_splits": 5,
  "random_state": 42,
  "sale_table": "sahibinden_sale_listings",
  "rental_table": "sahibinden_rental_listings",
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
  "enable_county_experts": false,
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
  "karamursel_baseline_mode": "none",
  "location_feature_mode": "geo",
  "geo_context_mode": "geo_with_coast",
  "comparable_mode": "none",
  "model_scope": "basiskele_only",
  "location_scope": "basiskele_only",
  "location_min_precision": "any",
  "enable_coordinate_noise_check": true,
  "comparable_k_list": "5,10,20",
  "run_location_ablation": false,
  "geo_context_cache_dir": "C:\\Users\\soyka\\OneDrive\\Desktop\\Quarox Emlak\\pure_listing_model_v0_izmit_thesis\\data\\external\\geo_context",
  "location_coverage_min": 0.4,
  "calibration_mode": "isotonic",
  "calibration_cap": 0.1,
  "ensemble_profile": "balanced",
  "target_profile": "residual_log",
  "hybrid_lambda_grid": "0.0,0.1,0.2,0.3,0.5",
  "run_calibration_ablation": false
}

## Cleaning report
{
  "sales_raw_rows": 970,
  "sales_after_base_clean_rows": 970,
  "sales_after_basic_filter_rows": 943,
  "sales_final_rows": 936,
  "sales_removed_basic_rows": 27,
  "sales_removed_iqr_rows": 7,
  "rentals_raw_rows": 423,
  "rentals_after_base_clean_rows": 423,
  "rentals_after_basic_filter_rows": 397,
  "rentals_final_rows": 392,
  "rentals_removed_basic_rows": 26,
  "rentals_removed_iqr_rows": 5,
  "location_outlier_filter_enabled": true,
  "rows_before_location_filter": 936,
  "rows_after_location_filter": 927,
  "rows_removed_location_filter": 9,
  "min_location_ratio": 0.5,
  "max_location_ratio": 1.9,
  "location_mad_threshold": 3.5,
  "location_min_group_size": 12,
  "removed_ratio_summary": {
    "min": 0.4114731500270623,
    "median": 0.5421146984596804,
    "max": 1.9406772618028993
  }
}

## Feature reports
{
  "rental_features": {
    "rental_rows_used": 392,
    "global_rent_m2_median": 255.56,
    "global_rent_row_count": 392,
    "rent_feature_level_counts": {
      "district_room": 892,
      "district": 22,
      "district_m2_group": 21,
      "county": 1
    }
  },
  "trend_features": {
    "trend_rows_used": 38,
    "trend_district_matched_rows": 936,
    "trend_date_min": "2026-05-01",
    "trend_date_max": "2027-03-01"
  },
  "demographic_features": {
    "mode": "safe",
    "demo_rows": 488,
    "listing_rows": 927,
    "matched_listing_rows": 907,
    "match_rate": 0.9784250269687162,
    "county_matched_listing_rows": 907,
    "county_match_rate": 0.9784250269687162,
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
  "pass_global_guardrail": false,
  "pass_guardrail": false,
  "pass_basiskele_lift": true,
  "pass_basiskele_variance_lift": true,
  "pass_basiskele_large_home_lift": false,
  "pass_basiskele_spread_lift": true,
  "pass_karamursel_lift": true,
  "pass_karamursel_guardrail": true,
  "pass_golcuk_guardrail": true,
  "pass_izmit_guardrail": true,
  "pass_location_coverage": true,
  "pass_detail_sensitivity": true,
  "pass_sensitivity": true,
  "pass_karamursel": true,
  "direction_pass_rate": 1.0,
  "karamursel_sale_diff_pct": 0.31074221507712513,
  "basiskele_r2": 0.4639455619857922,
  "basiskele_variance_ratio": 0.5044821454676627,
  "basiskele_lift": {
    "r2": 0.4639455619857922,
    "r2_delta_vs_v16": 0.023745561985792196,
    "r2_delta_vs_v15": 0.010545561985792151,
    "variance_ratio": 0.5044821454676627,
    "variance_delta_vs_v16": 0.061282145467662696
  },
  "golcuk_r2": NaN,
  "izmit_r2": NaN,
  "karamursel_r2": NaN,
  "county_r2_table": {
    "İzmit": NaN,
    "Başiskele": 0.4639455619857922,
    "Gölcük": NaN,
    "Karamürsel": NaN
  },
  "ship_ready_all_counties_r2_ge_0_65": false,
  "selected_detail_effect_mode": "group",
  "selected_basiskele_specialist_mode": "premium_target_stats",
  "selected_basiskele_variance_lift_mode": "none",
  "selected_location_feature_mode": "geo",
  "selected_geo_context_mode": "geo_with_coast",
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
  "overall": "FAIL",
  "warnings": [
    "global_guardrail_failed"
  ],
  "qa_findings": [
    {
      "finding": "Global MAPE/R2 outside V17 guardrail",
      "severity": "High"
    },
    {
      "finding": "not ship-ready until all counties R2 >= 0.65",
      "severity": "Info"
    }
  ],
  "top_risks": [],
  "top_opportunities": [
    "Başiskele R2 0.4639 (vs V16 0.4402)"
  ],
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
  "v16_reference": {
    "r2": 0.6803,
    "mape": 0.1285,
    "basiskele_r2": 0.4402,
    "basiskele_variance_ratio": 0.4432,
    "karamursel_r2": 0.593,
    "golcuk_r2": 0.6422,
    "izmit_r2": 0.7161
  },
  "v16_delta": {
    "r2": -0.21635443801420784,
    "mape": -0.017917671501443078,
    "basiskele_r2": 0.023745561985792196,
    "basiskele_variance_ratio": 0.061282145467662696,
    "karamursel_r2": NaN,
    "golcuk_r2": NaN,
    "izmit_r2": NaN
  },
  "v15_delta": {
    "r2": -0.21595443801420777,
    "mape": -0.018417671501443078,
    "basiskele_r2": 0.010545561985792151,
    "karamursel_r2": NaN
  },
  "selected_attribute_mode": "full",
  "selected_location_scope": "basiskele_only"
}

## Ensemble metrics
{
  "rows": 919,
  "r2": 0.4639455619857922,
  "log_r2": 0.47015544736268,
  "mape": 0.11058232849855693,
  "median_ape": 0.09156183167990246,
  "mae_tl_per_m2": 4555.447227955811,
  "median_ae_tl_per_m2": 3594.0241156227494,
  "model": "county_expert_segment_aware_ensemble_v19",
  "calibration_mode": "isotonic",
  "calibration_cap": 0.1,
  "calibration_guard": {
    "pass": true,
    "outer_validation_targets_used_in_calibrator": false,
    "inner_oof_used": true,
    "calibration_mode": "isotonic",
    "correction_cap": 0.1,
    "n_folds": 5,
    "notes": []
  },
  "ensemble_profile": "balanced",
  "target_profile": "residual_log",
  "base_weights": {
    "gradient_boosting": 0.34207897634334833,
    "extra_trees": 0.3350759684302553,
    "ridge": 0.3228450552263964
  },
  "segment_weights": {
    "large_home": {
      "extra_trees": 0.5041334750689049,
      "random_forest": 0.49586652493109507
    },
    "mainstream_home": {
      "gradient_boosting": 0.5006793087184749,
      "extra_trees": 0.49932069128152506
    }
  },
  "segment_blend_weights": {
    "large_home": 0.25,
    "mainstream_home": 0.35
  },
  "county_weights": {},
  "county_blend_weights": {},
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
  "pass_guardrail": false,
  "pass_global_guardrail": false,
  "pass_sensitivity": true,
  "pass_basiskele_lift": true,
  "pass_basiskele_variance_lift": true,
  "pass_karamursel_lift": true,
  "pass_karamursel_guardrail": true,
  "direction_pass_rate": 1.0,
  "karamursel_sale_diff_pct": 0.31074221507712513,
  "basiskele_variance_ratio": 0.5044821454676627,
  "location_scope_report": {
    "location_scope": "basiskele_only",
    "enabled_counties": [
      "Başiskele"
    ],
    "coverage": {
      "Başiskele": 0.7606093579978237
    },
    "warnings": [],
    "n_numeric_masked": 91,
    "n_categorical_masked": 8
  },
  "demographics_mode": "safe",
  "location_feature_mode": "geo",
  "geo_context_mode": "geo_with_coast",
  "location_scope": "basiskele_only",
  "comparable_mode": "none",
  "model_scope": "basiskele_only",
  "location_feature_metadata": {
    "location_feature_mode": "geo",
    "numeric_features": [
      "has_lat_lon",
      "lat",
      "lon",
      "lat_centered_city",
      "lon_centered_city",
      "lat_centered_county",
      "lon_centered_county",
      "location_precision_exact",
      "location_precision_approx",
      "location_precision_district_only",
      "location_precision_missing",
      "location_source_data_attr_map",
      "location_backfill_ok",
      "location_backfill_listing_removed",
      "location_quality_score",
      "distance_to_county_centroid_m",
      "bearing_from_county_centroid_sin",
      "bearing_from_county_centroid_cos",
      "distance_to_district_centroid_m",
      "bearing_from_district_centroid_sin",
      "bearing_from_district_centroid_cos",
      "distance_to_izmit_center_m",
      "distance_to_basiskele_coast_m",
      "distance_to_yuvacik_m",
      "distance_to_bahcecik_m",
      "distance_to_kullar_m",
      "distance_to_sahil_m",
      "distance_to_golcuk_center_m",
      "distance_to_karamursel_center_m",
      "distance_to_coastline_m",
      "is_coastal_500m",
      "is_coastal_1000m",
      "is_coastal_2000m",
      "distance_to_geo_cluster_center_m",
      "location_quality_x_detail_effect_total",
      "distance_to_coast_x_view_sea",
      "distance_to_coast_x_near_sea_zero",
      "distance_to_coast_x_site_inside",
      "distance_to_coast_x_large_home",
      "basiskele_lat_lon_interaction",
      "basiskele_distance_to_coast_x_large_home",
      "basiskele_distance_to_coast_x_quality"
    ],
    "categorical_features": [
      "location_precision",
      "location_source",
      "geo_cluster_city",
      "geo_cluster_county",
      "basiskele_geo_cluster",
      "coast_distance_bucket",
      "basiskele_geo_cluster_x_m2_group",
      "geo_cluster_x_room_count"
    ],
    "exact_map_required_features": [
      "bearing_from_county_centroid_cos",
      "bearing_from_county_centroid_sin",
      "bearing_from_district_centroid_cos",
      "bearing_from_district_centroid_sin",
      "distance_to_bahcecik_m",
      "distance_to_basiskele_coast_m",
      "distance_to_coast_x_large_home",
      "distance_to_coast_x_near_sea_zero",
      "distance_to_coast_x_site_inside",
      "distance_to_coast_x_view_sea",
      "distance_to_coastline_m",
      "distance_to_county_centroid_m",
      "distance_to_district_centroid_m",
      "distance_to_geo_cluster_center_m",
      "distance_to_golcuk_center_m",
      "distance_to_izmit_center_m",
      "distance_to_karamursel_center_m",
      "distance_to_kullar_m",
      "distance_to_sahil_m",
      "distance_to_yuvacik_m",
      "is_coastal_1000m",
      "is_coastal_2000m",
      "is_coastal_500m",
      "lat",
      "lat_centered_city",
      "lat_centered_county",
      "lon",
      "lon_centered_city",
      "lon_centered_county"
    ],
    "app_safe": true,
    "uses_target": false,
    "note": "Distance/cluster features are unreliable when location_precision is district_only or missing."
  },
  "fast_mode": false,
  "training_rows_before_anomaly_filter": 927,
  "training_rows_after_anomaly_filter": 919,
  "excluded_anomaly_rows": 8,
  "exclude_anomalies_threshold": 25.0
}

## Main outputs
- data/raw/sales_raw_from_source.csv
- data/raw/rentals_raw_from_source.csv
- data/input/sales_cleaned_v19_basiskele.csv
- data/input/rentals_cleaned_v19_basiskele.csv
- data/output/oof_predictions_v19_basiskele.csv
- reports/model_comparison_v19_basiskele.csv
- reports/metrics_summary_v19_basiskele.json
- reports/feature_sensitivity_v19_basiskele.csv
- reports/karamursel_sensitivity_v19_basiskele.csv
- reports/basiskele_variance_diagnostics_v19_basiskele.csv
- reports/metrics_attribute_ablation_v19_basiskele.csv
- reports/error_by_*_v19_basiskele.csv
- reports/*.png
- artifacts/model_*_v19_basiskele.joblib
- artifacts/model_bundle_v19_basiskele.joblib
