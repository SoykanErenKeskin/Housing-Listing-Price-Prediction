# V16 Model Run

## Executive Summary
- overall: None
- selected_attribute_mode: full
- demographics_mode: safe
- R2: 0.6358656162289547 | MAPE: 0.12003687930512742 | MAE: 4511.7582470511425
- v12_delta: {}
- karamursel_sale_diff_pct: None
- direction_pass_rate: None
- warnings: ['global_guardrail_failed', 'basiskele_compressed', 'izmit_r2_soft_floor']

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
    "Başiskele",
    "İzmit",
    "Gölcük",
    "Karamürsel",
    "Kartepe"
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
  "site_extraction_mode": "none",
  "site_project_encoding": "none",
  "run_site_ablation": true,
  "duplex_feature_mode": "full",
  "run_duplex_ablation": false,
  "merge_gap_warning_tl": 8000.0,
  "model_scope": "kocaeli_global",
  "location_scope": "global",
  "location_min_precision": "any",
  "enable_coordinate_noise_check": true,
  "comparable_k_list": "5,10,20",
  "run_location_ablation": false,
  "geo_context_cache_dir": "data/external/geo_context",
  "location_coverage_min": 0.4
}

## Cleaning report
{
  "sales_raw_rows": 6979,
  "sales_after_base_clean_rows": 6979,
  "sales_after_basic_filter_rows": 6920,
  "sales_final_rows": 6868,
  "sales_removed_basic_rows": 59,
  "sales_removed_iqr_rows": 52,
  "rentals_raw_rows": 3573,
  "rentals_after_base_clean_rows": 3573,
  "rentals_after_basic_filter_rows": 3545,
  "rentals_final_rows": 3525,
  "rentals_removed_basic_rows": 28,
  "rentals_removed_iqr_rows": 20,
  "location_outlier_filter_enabled": true,
  "rows_before_location_filter": 6868,
  "rows_after_location_filter": 6729,
  "rows_removed_location_filter": 139,
  "min_location_ratio": 0.5,
  "max_location_ratio": 1.9,
  "location_mad_threshold": 3.5,
  "location_min_group_size": 12,
  "removed_ratio_summary": {
    "min": 0.2674231601783091,
    "median": 1.8013588705629875,
    "max": 3.0943992773199924
  }
}

## Feature reports
{
  "rental_features": {
    "rental_rows_used": 3525,
    "global_rent_m2_median": 235.29,
    "global_rent_row_count": 3525,
    "rent_feature_level_counts": {
      "district_room": 6520,
      "district_m2_group": 202,
      "district": 132,
      "county": 14
    }
  },
  "trend_features": {
    "trend_rows_used": 246,
    "trend_district_matched_rows": 6868,
    "trend_date_min": "2026-03-01",
    "trend_date_max": "2027-05-01"
  },
  "demographic_features": {
    "mode": "safe",
    "demo_rows": 488,
    "listing_rows": 6729,
    "matched_listing_rows": 6696,
    "match_rate": 0.9950958537672759,
    "county_matched_listing_rows": 6696,
    "county_match_rate": 0.9950958537672759,
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
  "selected_site_extraction_mode": "none",
  "selected_site_project_encoding": "none",
  "r2": 0.6358656162289547,
  "mape": 0.12003687930512742,
  "variance_ratio": 0.6184810005227498,
  "expensive_decile_bias": -9102.864275998192,
  "large_home_r2": 0.5348295631065814,
  "leakage_guard_pass": true,
  "coverage": {
    "foldsafe_encoded_site_count": 0.0,
    "top_expensive_underpredicted_site_missing_rate": NaN
  },
  "best_checkpoint": false,
  "note": "V21 is not auto-promoted; promote only after full ablation clears V20 + merge gates.",
  "selected_attribute_mode": "full",
  "selected_detail_effect_mode": "group",
  "selected_basiskele_specialist_mode": "premium_target_stats",
  "selected_basiskele_variance_lift_mode": "none",
  "selected_location_feature_mode": "geo",
  "selected_geo_context_mode": "geo_with_coast",
  "selected_location_scope": "global",
  "selected_v16_layers": {
    "basiskele_large_home_regime": "none",
    "basiskele_spread_layer": "none",
    "karamursel_baseline_mode": "none"
  }
}

## Ensemble metrics
{
  "rows": 6667,
  "r2": 0.6358656162289547,
  "log_r2": 0.6545932421064347,
  "mape": 0.12003687930512742,
  "median_ape": 0.09242035603858913,
  "mae_tl_per_m2": 4511.7582470511425,
  "median_ae_tl_per_m2": 3392.0848314765135,
  "model": "county_expert_segment_aware_ensemble_v16",
  "base_weights": {
    "extra_trees": 0.33787149156289026,
    "ridge": 0.3339975089927965,
    "random_forest": 0.32813099944431323
  },
  "segment_weights": {
    "large_home": {
      "extra_trees": 0.5018292937411842,
      "ridge": 0.4981707062588157
    },
    "compact_home": {
      "gradient_boosting": 0.5065635085129234,
      "extra_trees": 0.4934364914870764
    },
    "old_building": {
      "extra_trees": 0.5002512318882383,
      "gradient_boosting": 0.4997487681117617
    },
    "mainstream_home": {
      "extra_trees": 0.503137044341652,
      "gradient_boosting": 0.496862955658348
    }
  },
  "segment_blend_weights": {
    "large_home": 0.35,
    "compact_home": 0.35,
    "old_building": 0.5,
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
  "pass_basiskele_lift": true,
  "pass_basiskele_variance_lift": true,
  "pass_karamursel_lift": true,
  "pass_karamursel_guardrail": true,
  "direction_pass_rate": 0.875,
  "karamursel_sale_diff_pct": 0.6536242806298381,
  "basiskele_variance_ratio": 0.44999256993219955,
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
    "location_scope": "global",
    "enabled_counties": [
      "Başiskele",
      "Gölcük",
      "Karamürsel",
      "Kartepe",
      "İzmit"
    ],
    "coverage": {
      "Başiskele": 0.8762122076440388,
      "Gölcük": 0.8807106598984772,
      "Karamürsel": 0.8871951219512195,
      "Kartepe": 1.0,
      "İzmit": 0.7284105131414268
    },
    "warnings": [],
    "n_numeric_masked": 91,
    "n_categorical_masked": 8
  },
  "demographics_mode": "safe",
  "location_feature_mode": "geo",
  "geo_context_mode": "geo_with_coast",
  "location_scope": "global",
  "comparable_mode": "none",
  "site_extraction_mode": "none",
  "site_project_encoding": "none",
  "duplex_feature_mode": "full",
  "model_scope": "kocaeli_global",
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
  "training_rows_before_anomaly_filter": 6729,
  "training_rows_after_anomaly_filter": 6667,
  "excluded_anomaly_rows": 62,
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
