# V16 Model Run

## Executive Summary
- overall: FAIL
- selected_attribute_mode: full
- demographics_mode: safe
- R2: 0.5197833849355997 | MAPE: 0.15015977154451243 | MAE: 5467.137268164475
- v12_delta: {}
- karamursel_sale_diff_pct: 0.23233582947067385
- direction_pass_rate: 0.6875
- warnings: ['global_guardrail_failed', 'basiskele_no_r2_lift', 'basiskele_variance_no_lift', 'basiskele_compressed', 'karamursel_r2_regression', 'direction_pass_rate_low']

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
  "fast_mode": true,
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
  "karamursel_baseline_mode": "none",
  "location_feature_mode": "geo",
  "geo_context_mode": "geo_with_coast",
  "location_min_precision": "any",
  "enable_coordinate_noise_check": true,
  "comparable_k_list": "5,10,20",
  "run_location_ablation": false,
  "geo_context_cache_dir": "data/external/geo_context"
}

## Cleaning report
{
  "sales_raw_rows": 800,
  "sales_after_base_clean_rows": 399,
  "sales_after_basic_filter_rows": 397,
  "sales_final_rows": 397,
  "sales_removed_basic_rows": 2,
  "sales_removed_iqr_rows": 0,
  "rentals_raw_rows": 800,
  "rentals_after_base_clean_rows": 800,
  "rentals_after_basic_filter_rows": 786,
  "rentals_final_rows": 785,
  "rentals_removed_basic_rows": 14,
  "rentals_removed_iqr_rows": 1,
  "location_outlier_filter_enabled": true,
  "rows_before_location_filter": 397,
  "rows_after_location_filter": 385,
  "rows_removed_location_filter": 12,
  "min_location_ratio": 0.5,
  "max_location_ratio": 1.9,
  "location_mad_threshold": 3.5,
  "location_min_group_size": 12,
  "removed_ratio_summary": {
    "min": 0.411693520973769,
    "median": 1.9676659573098223,
    "max": 2.405905033611858
  }
}

## Feature reports
{
  "rental_features": {
    "rental_rows_used": 785,
    "global_rent_m2_median": 222.22,
    "global_rent_row_count": 785,
    "rent_feature_level_counts": {
      "district_room": 362,
      "district": 14,
      "district_m2_group": 11,
      "county": 10
    }
  },
  "trend_features": {
    "trend_rows_used": 213,
    "trend_district_matched_rows": 397,
    "trend_date_min": "2026-03-01",
    "trend_date_max": "2027-05-01"
  },
  "demographic_features": {
    "mode": "safe",
    "demo_rows": 488,
    "listing_rows": 385,
    "matched_listing_rows": 385,
    "match_rate": 1.0,
    "county_matched_listing_rows": 385,
    "county_match_rate": 1.0,
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
  "pass_basiskele_lift": false,
  "pass_basiskele_variance_lift": true,
  "pass_basiskele_large_home_lift": false,
  "pass_basiskele_spread_lift": true,
  "pass_karamursel_lift": false,
  "pass_karamursel_guardrail": false,
  "pass_golcuk_guardrail": true,
  "pass_izmit_guardrail": true,
  "pass_location_coverage": true,
  "pass_detail_sensitivity": false,
  "pass_sensitivity": false,
  "pass_karamursel": true,
  "direction_pass_rate": 0.6875,
  "karamursel_sale_diff_pct": 0.23233582947067385,
  "basiskele_r2": -0.07058171130882096,
  "basiskele_variance_ratio": 0.33963082836761915,
  "basiskele_lift": {
    "r2": -0.07058171130882096,
    "r2_delta_vs_v16": -0.5107817113088209,
    "r2_delta_vs_v15": -0.523981711308821,
    "variance_ratio": 0.33963082836761915,
    "variance_delta_vs_v16": -0.10356917163238083
  },
  "golcuk_r2": 0.6242549636592766,
  "izmit_r2": NaN,
  "karamursel_r2": 0.2984183900034817,
  "county_r2_table": {
    "İzmit": NaN,
    "Başiskele": -0.07058171130882096,
    "Gölcük": 0.6242549636592766,
    "Karamürsel": 0.2984183900034817
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
    "global_guardrail_failed",
    "basiskele_no_r2_lift",
    "basiskele_variance_no_lift",
    "basiskele_compressed",
    "karamursel_r2_regression",
    "direction_pass_rate_low"
  ],
  "qa_findings": [
    {
      "finding": "Global MAPE/R2 outside V17 guardrail",
      "severity": "High"
    },
    {
      "finding": "Başiskele R2 -0.0706 not above V16 0.4402",
      "severity": "High"
    },
    {
      "finding": "not ship-ready until all counties R2 >= 0.65",
      "severity": "Info"
    }
  ],
  "top_risks": [
    "Karamürsel R2 regression vs V16"
  ],
  "top_opportunities": [],
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
    "r2": -0.1605166150644003,
    "mape": 0.02165977154451243,
    "basiskele_r2": -0.5107817113088209,
    "basiskele_variance_ratio": -0.10356917163238083,
    "karamursel_r2": -0.2945816099965183,
    "golcuk_r2": -0.01794503634072342,
    "izmit_r2": NaN
  },
  "v15_delta": {
    "r2": -0.16011661506440023,
    "mape": 0.02115977154451243,
    "basiskele_r2": -0.523981711308821,
    "karamursel_r2": -0.26968160999651836
  },
  "selected_attribute_mode": "full"
}

## Ensemble metrics
{
  "rows": 379,
  "r2": 0.5197833849355997,
  "log_r2": 0.5947532074618916,
  "mape": 0.15015977154451243,
  "median_ape": 0.12139517094999462,
  "mae_tl_per_m2": 5467.137268164475,
  "median_ae_tl_per_m2": 4041.2218939343365,
  "model": "county_expert_segment_aware_ensemble_v16",
  "base_weights": {
    "extra_trees": 0.3409578517694821,
    "ridge": 0.3316188660461477,
    "gradient_boosting": 0.3274232821843701
  },
  "segment_weights": {
    "mainstream_home": {
      "gradient_boosting": 0.5040194328670309,
      "extra_trees": 0.49598056713296906
    }
  },
  "segment_blend_weights": {
    "mainstream_home": 0.35
  },
  "county_weights": {
    "Gölcük": {
      "ridge": 0.5292751934077932,
      "extra_trees": 0.47072480659220683
    }
  },
  "county_blend_weights": {
    "Gölcük": 0.8
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
  "pass_guardrail": false,
  "pass_global_guardrail": false,
  "pass_sensitivity": false,
  "pass_basiskele_lift": false,
  "pass_basiskele_variance_lift": true,
  "pass_karamursel_lift": false,
  "pass_karamursel_guardrail": false,
  "direction_pass_rate": 0.6875,
  "karamursel_sale_diff_pct": 0.23233582947067385,
  "basiskele_variance_ratio": 0.33963082836761915,
  "demographics_mode": "safe",
  "location_feature_mode": "geo",
  "geo_context_mode": "geo_with_coast",
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
  "training_rows_before_anomaly_filter": 385,
  "training_rows_after_anomaly_filter": 379,
  "excluded_anomaly_rows": 6,
  "exclude_anomalies_threshold": 25.0
}

## Main outputs
- data/raw/sales_raw_from_source.csv
- data/raw/rentals_raw_from_source.csv
- data/input/sales_cleaned_v17.csv
- data/input/rentals_cleaned_v17.csv
- data/output/oof_predictions_v17.csv
- reports/model_comparison_v17.csv
- reports/metrics_summary_v17.json
- reports/feature_sensitivity_v17.csv
- reports/karamursel_sensitivity_v17.csv
- reports/basiskele_variance_diagnostics_v17.csv
- reports/metrics_attribute_ablation_v17.csv
- reports/error_by_*_v17.csv
- reports/*.png
- artifacts/model_*_v17.joblib
- artifacts/model_bundle_v17.joblib
