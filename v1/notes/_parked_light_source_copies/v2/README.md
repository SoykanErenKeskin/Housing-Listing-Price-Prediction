# Housing-Listing-Price-Prediction v1 — Feature Engineering + Model Tuning

Bu paket yalnızca listing portal ilan verisiyle çalışır. external demographics provider/trend/floor external reference kullanılmaz.

## V1'de gelenler

- Feature engineering
- KFold-safe target encoding
- Boş/tek değerli kolonları otomatik çıkarma
- Birden fazla model karşılaştırması
- Gradient Boosting için hafif tuning
- Hata analizi raporları

## Çalıştırma

En güncel temiz CSV'yi buraya koy:

```text
data/input/listing_dataset_cleaned.csv
```

Sonra:

```bash
python src/train_v1_feature_engineering.py
```

Tahmin denemesi:

```bash
python src/predict_v1.py
```

## Önemli çıktılar

```text
reports/model_metrics_v1.json
reports/model_comparison_v1.csv
artifacts/best_model.joblib
data/output/feature_engineered_dataset_preview.csv
data/output/<best_model>_cv_predictions.csv
data/output/<best_model>_top_50_errors.csv
data/output/<best_model>_error_by_district.csv
data/output/<best_model>_error_by_room_count.csv
data/output/<best_model>_error_by_building_age_group.csv
data/output/<best_model>_error_by_m2_group.csv
```

## Not

Script, v0'a göre daha uzun çalışır. Tuning ve ExtraTrees kısmı süreyi artırabilir.
