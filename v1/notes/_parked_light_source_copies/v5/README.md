# Housing-Listing-Price-Prediction v5.2 — Detail Selection + Scores + Ensemble + CatBoost

Bu paket v4 üzerine tüm geliştirme fikirlerini tek klasörde toplar.

## İçerik

1. Rare detail feature selection
   - `front_*`, `view_*`, `transport_*`, `near_*`, `out_*`, `in_*`, `subtype_*`
   - `ones < 20` olan detail binary feature'lar otomatik çıkarılır.

2. Detail score feature'ları
   - `front_score`
   - `view_score`
   - `transport_score`
   - `nearby_score`
   - `inside_quality_score`
   - `outside_quality_score`
   - `premium_detail_score`
   - `site_security_score`
   - `accessibility_score`

3. District interaction feature'ları
   - `district_age_group`
   - `district_m2_group`
   - `district_room_count`
   - `district_view_group`
   - `district_transport_group`
   - `district_quality_group`
   - `district_site_inside`

4. Modeller
   - Ridge
   - ElasticNet
   - GradientBoosting
   - HistGradientBoosting
   - ExtraTrees
   - RandomForest
   - R² odaklı tuned GradientBoosting
   - Manual CatBoost CV
   - Weighted ensemble
   - Grid-best ensemble

5. Raporlar
   - model karşılaştırması
   - detail coverage
   - ensemble ağırlıkları
   - top 50 hata
   - mahalle/oda/m²/bina yaşı/detay kırılımı hata raporları

## Kullanım

En güncel enriched CSV:

```text
data/input/listing_dataset_cleaned.csv
```

Çalıştır:

```bash
python src/train_v5_2_detail_selection_ensemble_catboost.py
```

Tahmin:

```bash
python src/predict_v5_2.py
```

## CatBoost

CatBoost kurulu değilse script hata vermeden rapora `catboost_not_available` yazar. Kurmak için:

```bash
pip install catboost
```

## Ana çıktılar

```text
reports/model_metrics_v5.json
reports/model_comparison_v5.csv
reports/detail_feature_coverage_v5.csv
reports/model_aux_v5.json
artifacts/best_model_v5_by_r2.joblib
data/output/<best_model>_cv_predictions.csv
data/output/<best_model>_top_50_errors.csv
```

## Not

Bu paket biraz uzun çalışabilir. Özellikle CatBoost, ExtraTrees ve tuning aşamaları süreyi artırır.


## v5.1 bug fix

- `front_score`, `view_score`, `transport_score`, `inside_quality_score` gibi skor kolonları artık raw detail binary feature olarak tekrar seçilmez.
- Duplicate feature listeleri otomatik temizlenir.
- Manual CatBoost CV içinde duplicate DataFrame column hatasına karşı koruma eklendi.
- Bu sürüm, `pd.to_numeric(...): arg must be a list, tuple, 1-d array, or Series` hatasını düzeltir.


## v5.2 bug fix

v5.1'de CatBoost dışındaki modellerin çoğu şu hataya düşebiliyordu:

```text
A given column is not a column of the dataframe
```

Sebep: `RareBinaryDropper` bazı fold'larda nadir detail kolonlarını fiziksel olarak siliyordu. Ancak `ColumnTransformer` bu kolonları hâlâ beklediği için modeller patlıyordu.

v5.2'de düzeltme:
- Nadir detail kolonları artık DataFrame'den silinmiyor.
- Kolon korunuyor ama değeri 0'a sabitleniyor.
- Böylece Ridge, ElasticNet, GradientBoosting, HistGB, ExtraTrees, RandomForest, tuning ve ensemble akışları düzgün çalışmalı.
