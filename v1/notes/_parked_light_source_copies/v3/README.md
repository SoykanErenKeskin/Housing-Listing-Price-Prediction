# Housing-Listing-Price-Prediction v3 — Strict Standard Flat

Bu paket R² değerini yükseltmek için daha homojen bir "standart daire" veri seti oluşturur.

## Ana fark

v2 sadece dubleks/lüks tarzı başlıkları çıkarıyordu.  
v3 daha sıkı filtre uygular:

- title içinde villa / müstakil / bahçe / bahçeli / dubleks / tripleks / lüks geçenler çıkarılır
- room_count sadece 1+1, 2+1, 3+1, 4+1 kalır
- gross_m2 < 45 çıkarılır
- gross_m2 > 220 çıkarılır
- real_estate_type villa/müstakil ise çıkarılır

## Model tarafı

- Ridge
- ElasticNet
- GradientBoosting
- HistGradientBoosting
- ExtraTrees
- RandomForest
- CatBoost native categorical, kuruluysa
- R² odaklı GradientBoosting tuning

## CatBoost kurulumu

```bash
pip install catboost
```

## Kullanım

En güncel temiz CSV'yi şuraya koy:

```text
data/input/listing_dataset_cleaned.csv
```

Çalıştır:

```bash
python src/train_v3_strict_standard_flat.py
```

Tahmin:

```bash
python src/predict_v3.py
```

## Önemli çıktılar

```text
reports/model_metrics_v3.json
reports/model_comparison_v3.csv
artifacts/best_model_v3_by_r2.joblib
data/output/strict_standard_flat_dataset.csv
data/output/removed_by_strict_standard_flat_filter.csv
data/output/dataset_marked_with_strict_filter_reason.csv
data/output/<best_model>_cv_predictions.csv
data/output/<best_model>_top_50_errors.csv
data/output/<best_model>_error_by_district.csv
data/output/<best_model>_error_by_building_age_group.csv
data/output/<best_model>_error_by_m2_group.csv
```

## Not

Bu model daha yüksek R² hedefler ama veri sayısı azalabilir.  
Eğer çok fazla ilan çıkarsa R² beklenenden düşük kalabilir; bu durumda filtreleri gevşetmek gerekir.
