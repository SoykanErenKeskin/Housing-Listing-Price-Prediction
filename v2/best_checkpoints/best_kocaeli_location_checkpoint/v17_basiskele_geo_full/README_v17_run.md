# V16 Model Run

## Executive Summary
- overall: PASS
- selected_attribute_mode: full
- demographics_mode: safe
- R2: 0.6888282987651515 | MAPE: 0.12706866510168197 | MAE: 4783.331255207735
- v12_delta: {}
- karamursel_sale_diff_pct: 0.4036630220960244
- direction_pass_rate: 0.9375
- warnings: ['basiskele_variance_no_lift', 'basiskele_compressed']

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
  "karamursel_baseline_mode": "none",
  "location_feature_mode": "geo",
  "geo_context_mode": "geo_with_coast",
  "location_scope": "basiskele_only",
  "location_min_precision": "any",
  "enable_coordinate_noise_check": true,
  "comparable_k_list": "5,10,20",
  "run_location_ablation": true,
  "geo_context_cache_dir": "data/external/geo_context",
  "location_coverage_min": 0.4
}

## Cleaning report
{
  "sales_raw_rows": 3668,
  "sales_after_base_clean_rows": 3267,
  "sales_after_basic_filter_rows": 3239,
  "sales_final_rows": 3222,
  "sales_removed_basic_rows": 28,
  "sales_removed_iqr_rows": 17,
  "rentals_raw_rows": 2386,
  "rentals_after_base_clean_rows": 2386,
  "rentals_after_basic_filter_rows": 2354,
  "rentals_final_rows": 2339,
  "rentals_removed_basic_rows": 32,
  "rentals_removed_iqr_rows": 15,
  "location_outlier_filter_enabled": true,
  "rows_before_location_filter": 3222,
  "rows_after_location_filter": 3143,
  "rows_removed_location_filter": 79,
  "min_location_ratio": 0.5,
  "max_location_ratio": 1.9,
  "location_mad_threshold": 3.5,
  "location_min_group_size": 12,
  "removed_ratio_summary": {
    "min": 0.2610727434670388,
    "median": 1.9116870358032039,
    "max": 2.587942278620662
  }
}

## Feature reports
{
  "rental_features": {
    "rental_rows_used": 2339,
    "global_rent_m2_median": 233.33,
    "global_rent_row_count": 2339,
    "rent_feature_level_counts": {
      "district_room": 3062,
      "district_m2_group": 82,
      "district": 71,
      "county": 7
    }
  },
  "trend_features": {
    "trend_rows_used": 213,
    "trend_district_matched_rows": 3222,
    "trend_date_min": "2026-03-01",
    "trend_date_max": "2027-05-01"
  },
  "demographic_features": {
    "mode": "safe",
    "demo_rows": 488,
    "listing_rows": 3143,
    "matched_listing_rows": 3123,
    "match_rate": 0.9936366528794146,
    "county_matched_listing_rows": 3123,
    "county_match_rate": 0.9936366528794146,
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
  "direction_pass_rate": 0.9375,
  "karamursel_sale_diff_pct": 0.4036630220960244,
  "basiskele_r2": 0.483426974660913,
  "basiskele_variance_ratio": 0.43462561422648777,
  "basiskele_lift": {
    "r2": 0.483426974660913,
    "r2_delta_vs_v16": 0.04322697466091302,
    "r2_delta_vs_v15": 0.030026974660912975,
    "variance_ratio": 0.43462561422648777,
    "variance_delta_vs_v16": -0.008574385773512216
  },
  "golcuk_r2": 0.667621639981864,
  "izmit_r2": 0.7109878551916231,
  "karamursel_r2": 0.6088426119567827,
  "county_r2_table": {
    "İzmit": 0.7109878551916231,
    "Başiskele": 0.483426974660913,
    "Gölcük": 0.667621639981864,
    "Karamürsel": 0.6088426119567827
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
  "overall": "PASS",
  "warnings": [
    "basiskele_variance_no_lift",
    "basiskele_compressed"
  ],
  "qa_findings": [
    {
      "finding": "not ship-ready until all counties R2 >= 0.65",
      "severity": "Info"
    },
    {
      "finding": "PASS as experiment, NOT ship-ready.",
      "severity": "Info"
    }
  ],
  "top_risks": [],
  "top_opportunities": [
    "Başiskele R2 0.4834 (vs V16 0.4402)"
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
    "r2": 0.008528298765151487,
    "mape": -0.0014313348983180374,
    "basiskele_r2": 0.04322697466091302,
    "basiskele_variance_ratio": -0.008574385773512216,
    "karamursel_r2": 0.015842611956782715,
    "golcuk_r2": 0.025421639981863975,
    "izmit_r2": -0.005112144808376851
  },
  "v15_delta": {
    "r2": 0.008928298765151554,
    "mape": -0.0019313348983180378,
    "basiskele_r2": 0.030026974660912975,
    "karamursel_r2": 0.04074261195678264
  },
  "experiment_note": "PASS as experiment, NOT ship-ready.",
  "selected_attribute_mode": "full",
  "selected_location_scope": "basiskele_only"
}

## Ensemble metrics
{
  "rows": 3104,
  "r2": 0.6888282987651515,
  "log_r2": 0.7123519679560801,
  "mape": 0.12706866510168197,
  "median_ape": 0.10243581812853203,
  "mae_tl_per_m2": 4783.331255207735,
  "median_ae_tl_per_m2": 3735.6188361319964,
  "model": "county_expert_segment_aware_ensemble_v16",
  "base_weights": {
    "extra_trees": 0.3390006654829951,
    "ridge": 0.3318276692378319,
    "gradient_boosting": 0.3291716652791731
  },
  "segment_weights": {
    "large_home": {
      "extra_trees": 0.5075558242990393,
      "ridge": 0.4924441757009606
    },
    "compact_home": {
      "ridge": 0.504296200719033,
      "extra_trees": 0.495703799280967
    },
    "old_building": {
      "gradient_boosting": 0.5059298814247285,
      "extra_trees": 0.49407011857527144
    },
    "mainstream_home": {
      "extra_trees": 0.5002234214986936,
      "gradient_boosting": 0.49977657850130636
    }
  },
  "segment_blend_weights": {
    "large_home": 0.25,
    "compact_home": 0.35,
    "old_building": 0.5,
    "mainstream_home": 0.35
  },
  "county_weights": {
    "Başiskele": {
      "gradient_boosting": 0.5040228306592491,
      "extra_trees": 0.49597716934075087
    },
    "Gölcük": {
      "ridge": 0.5123040466990023,
      "extra_trees": 0.48769595330099774
    },
    "Karamürsel": {
      "gradient_boosting": 0.5113601331183791,
      "random_forest": 0.488639866881621
    },
    "İzmit": {
      "ridge": 0.502561772729643,
      "extra_trees": 0.4974382272703571
    }
  },
  "county_blend_weights": {
    "Başiskele": 0.35,
    "Gölcük": 0.65,
    "Karamürsel": 0.35,
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
  "pass_basiskele_lift": true,
  "pass_basiskele_variance_lift": true,
  "pass_karamursel_lift": true,
  "pass_karamursel_guardrail": true,
  "direction_pass_rate": 0.9375,
  "karamursel_sale_diff_pct": 0.4036630220960244,
  "basiskele_variance_ratio": 0.43462561422648777,
  "location_scope_report": {
    "location_scope": "basiskele_only",
    "enabled_counties": [
      "Başiskele"
    ],
    "coverage": {
      "Başiskele": 0.7606093579978237,
      "Gölcük": 0.0,
      "Karamürsel": 0.7463414634146341,
      "İzmit": 0.0
    },
    "warnings": [],
    "n_numeric_masked": 126,
    "n_categorical_masked": 8
  },
  "demographics_mode": "safe",
  "location_feature_mode": "geo",
  "geo_context_mode": "geo_with_coast",
  "location_scope": "basiskele_only",
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
  "training_rows_before_anomaly_filter": 3143,
  "training_rows_after_anomaly_filter": 3104,
  "excluded_anomaly_rows": 39,
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
