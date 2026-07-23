# V16 Model Run

## Executive Summary
- overall: None
- selected_attribute_mode: full
- demographics_mode: safe
- R2: 0.2740007241264225 | MAPE: 0.12768858790113902 | MAE: 5314.909183064171
- v12_delta: {}
- karamursel_sale_diff_pct: None
- direction_pass_rate: None
- warnings: ['global_guardrail_failed', 'basiskele_no_r2_lift', 'basiskele_variance_no_lift', 'basiskele_compressed']

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
  "site_extraction_mode": "full",
  "site_project_encoding": "frequency",
  "run_site_ablation": false,
  "merge_gap_warning_tl": 8000.0,
  "model_scope": "basiskele_only",
  "location_scope": "basiskele_only",
  "location_min_precision": "any",
  "enable_coordinate_noise_check": true,
  "comparable_k_list": "5,10,20",
  "run_location_ablation": false,
  "geo_context_cache_dir": "data/external/geo_context",
  "location_coverage_min": 0.4
}

## Cleaning report
{
  "sales_raw_rows": 300,
  "sales_after_base_clean_rows": 300,
  "sales_after_basic_filter_rows": 289,
  "sales_final_rows": 288,
  "sales_removed_basic_rows": 11,
  "sales_removed_iqr_rows": 1,
  "rentals_raw_rows": 300,
  "rentals_after_base_clean_rows": 300,
  "rentals_after_basic_filter_rows": 284,
  "rentals_final_rows": 280,
  "rentals_removed_basic_rows": 16,
  "rentals_removed_iqr_rows": 4,
  "location_outlier_filter_enabled": true,
  "rows_before_location_filter": 288,
  "rows_after_location_filter": 285,
  "rows_removed_location_filter": 3,
  "min_location_ratio": 0.5,
  "max_location_ratio": 1.9,
  "location_mad_threshold": 3.5,
  "location_min_group_size": 12,
  "removed_ratio_summary": {
    "min": 1.5663372667938906,
    "median": 1.5839944094351766,
    "max": 2.105748500241662
  }
}

## Feature reports
{
  "rental_features": {
    "rental_rows_used": 280,
    "global_rent_m2_median": 256.25,
    "global_rent_row_count": 280,
    "rent_feature_level_counts": {
      "district_room": 272,
      "district": 7,
      "district_m2_group": 6,
      "county": 3
    }
  },
  "trend_features": {
    "trend_rows_used": 38,
    "trend_district_matched_rows": 288,
    "trend_date_min": "2026-05-01",
    "trend_date_max": "2027-03-01"
  },
  "demographic_features": {
    "mode": "safe",
    "demo_rows": 488,
    "listing_rows": 285,
    "matched_listing_rows": 285,
    "match_rate": 1.0,
    "county_matched_listing_rows": 285,
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
  "selected_site_extraction_mode": "full",
  "selected_site_project_encoding": "frequency",
  "r2": 0.2740007241264225,
  "mape": 0.12768858790113902,
  "variance_ratio": 0.38169462386043185,
  "expensive_decile_bias": -11607.25768997954,
  "large_home_r2": 0.25706929853939886,
  "leakage_guard_pass": true,
  "coverage": {
    "extracted_raw_rate": 0.27208480565371024,
    "alias_hit_rate": 0.2332155477031802,
    "dict_hit_rate": 0.10954063604240283,
    "canonical_non_missing_rate": 0.27208480565371024,
    "n_rows": 283.0,
    "n_canonical_sites": 54.0,
    "foldsafe_encoded_site_count": 0.0,
    "top_expensive_underpredicted_site_missing_rate": 0.42857142857142855
  },
  "best_checkpoint": false,
  "note": "V21 is not auto-promoted; promote only after full ablation clears V20 + merge gates.",
  "selected_attribute_mode": "full",
  "selected_detail_effect_mode": "group",
  "selected_basiskele_specialist_mode": "premium_target_stats",
  "selected_basiskele_variance_lift_mode": "none",
  "selected_location_feature_mode": "geo",
  "selected_geo_context_mode": "geo_with_coast",
  "selected_location_scope": "basiskele_only",
  "selected_v16_layers": {
    "basiskele_large_home_regime": "none",
    "basiskele_spread_layer": "none",
    "karamursel_baseline_mode": "none"
  }
}

## Ensemble metrics
{
  "rows": 283,
  "r2": 0.2740007241264225,
  "log_r2": 0.29595069598269974,
  "mape": 0.12768858790113902,
  "median_ape": 0.10574270847107371,
  "mae_tl_per_m2": 5314.909183064171,
  "median_ae_tl_per_m2": 4224.928957334741,
  "model": "county_expert_segment_aware_ensemble_v16",
  "base_weights": {
    "extra_trees": 0.3405364485097639,
    "gradient_boosting": 0.3328959871258706,
    "random_forest": 0.32656756436436546
  },
  "segment_weights": {
    "mainstream_home": {
      "gradient_boosting": 0.5051957271617314,
      "random_forest": 0.4948042728382685
    }
  },
  "segment_blend_weights": {
    "mainstream_home": 0.5
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
  "pass_guardrail": null,
  "pass_global_guardrail": false,
  "pass_sensitivity": null,
  "pass_basiskele_lift": false,
  "pass_basiskele_variance_lift": true,
  "pass_karamursel_lift": true,
  "pass_karamursel_guardrail": true,
  "direction_pass_rate": 0.875,
  "karamursel_sale_diff_pct": 0.06099281340190736,
  "basiskele_variance_ratio": 0.38169462386043185,
  "site_project_encoding_leakage_guard": {
    "enabled": false,
    "uses_train_pool_only": true,
    "validation_targets_used": false,
    "outer_validation_targets_used_in_encoder": false,
    "min_count": 3,
    "alpha": 20.0,
    "pass": true,
    "notes": [
      "foldsafe encoder disabled"
    ]
  },
  "location_scope_report": {
    "location_scope": "basiskele_only",
    "enabled_counties": [
      "Başiskele"
    ],
    "coverage": {
      "Başiskele": 0.7703180212014135
    },
    "warnings": [],
    "n_numeric_masked": 92,
    "n_categorical_masked": 8
  },
  "demographics_mode": "safe",
  "location_feature_mode": "geo",
  "geo_context_mode": "geo_with_coast",
  "location_scope": "basiskele_only",
  "comparable_mode": "none",
  "site_extraction_mode": "full",
  "site_project_encoding": "frequency",
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
  "fast_mode": true,
  "comparability_warning": "fast_mode location result is smoke only, not comparable. V15/V16 comparisons only with full train.",
  "training_rows_before_anomaly_filter": 285,
  "training_rows_after_anomaly_filter": 283,
  "excluded_anomaly_rows": 2,
  "exclude_anomalies_threshold": 25.0
}

## Main outputs
- data/raw/sales_raw_from_source.csv
- data/raw/rentals_raw_from_source.csv
- data/input/sales_cleaned_v18_basiskele.csv
- data/input/rentals_cleaned_v18_basiskele.csv
- data/output/oof_predictions_v18_basiskele.csv
- reports/model_comparison_v18_basiskele.csv
- reports/metrics_summary_v18_basiskele.json
- reports/feature_sensitivity_v18_basiskele.csv
- reports/karamursel_sensitivity_v18_basiskele.csv
- reports/basiskele_variance_diagnostics_v18_basiskele.csv
- reports/metrics_attribute_ablation_v18_basiskele.csv
- reports/error_by_*_v18_basiskele.csv
- reports/*.png
- artifacts/model_*_v18_basiskele.joblib
- artifacts/model_bundle_v18_basiskele.joblib
