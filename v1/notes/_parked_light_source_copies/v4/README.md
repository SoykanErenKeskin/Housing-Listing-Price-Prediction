# Housing-Listing-Price-Prediction v4 — Detail Features

Bu paket, v1/v2 pipeline üzerine yeni helper ile toplanan detay feature'larını ekler.

Strict segment filtresi bu pakette varsayılan olarak uygulanmaz. Önce detay feature'ların katkısını temiz şekilde görmek hedeflenir.

## Otomatik kullanılan yeni feature grupları

```text
front_*
view_*
transport_*
near_*
out_*
in_*
subtype_*
```

Ayrıca şu numeric alanlar kullanılır:

```text
detail_selected_count
detail_quality_score
detail_front_count
detail_view_count
detail_transport_count
detail_near_count
detail_inside_count
detail_outside_count
detail_subtype_count
```

Raw categorical olarak:

```text
detail_cephe
detail_manzara
detail_konut_tipi
```

## Çalıştırma

En güncel enriched CSV dosyanı şuraya koy:

```text
data/input/listing_dataset_cleaned.csv
```

Sonra:

```bash
python src/train_v4_detail_features.py
```

Tahmin:

```bash
python src/predict_v4.py
```

## Önemli çıktılar

```text
reports/model_metrics_v4.json
reports/model_comparison_v4.csv
reports/detail_feature_coverage_v4.csv
artifacts/best_model_v4_by_r2.joblib
data/output/<best_model>_cv_predictions.csv
data/output/<best_model>_top_50_errors.csv
```

`detail_feature_coverage_v4.csv` dosyası çok önemli. Detay kolonlarının kaç ilanda dolu olduğunu gösterir.
