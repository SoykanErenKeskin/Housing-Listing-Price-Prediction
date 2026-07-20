# Housing-Listing-Price-Prediction v0 — İzmit Thesis Pilot

Bu paket, yalnızca listing portal ilan verisiyle eğitilen baseline konut m² fiyat modelidir.

Dış kaynaklar kullanılmadı:
- external demographics provider yok
- trend_observed yok
- trend_projection yok
- floor_segments yok

Amaç:
- İzmit ilçesi için 500–1000 ilanlık pilot çalışma altyapısı kurmak
- Tezde “ilan özelliklerine dayalı tahmin modeli” olarak kullanılabilecek temiz bir başlangıç üretmek
- Daha sonra aynı pipeline ile daha büyük veri üzerinde yeniden eğitim yapmak

## Hedef değişken

Model `unit_price_gross` tahmin eder.

Formül:

`unit_price_gross = price / gross_m2`

Toplam fiyat tahmini:

`predicted_total_price = predicted_unit_price_gross * gross_m2`

## Modeller

İki model eğitildi:

1. `RidgeCV`
   - Çoklu regresyon mantığına daha yakın
   - Daha açıklanabilir
   - Tez anlatımı için daha temiz
   - Mobil uygulamaya gömülmesi daha kolay

2. `GradientBoostingRegressor`
   - Daha esnek
   - Doğrusal olmayan ilişkileri yakalayabilir
   - Baseline performans kıyaslaması için iyi

İki model de hedef değişkeni log dönüşümüyle eğitir:

`log1p(unit_price_gross)`

Tahminde tekrar gerçek TL/m² ölçeğine döner.

## Kullanılan ana özellikler

Numerik:
['gross_m2', 'net_m2', 'building_age', 'floor_num', 'total_floors', 'bathroom_count', 'open_area_m2', 'net_gross_ratio', 'has_open_area']

Kategorik:
['real_estate_type', 'room_count', 'floor_segment', 'heating', 'kitchen', 'balcony', 'elevator', 'parking', 'furnished', 'usage_status', 'site_inside', 'credit_eligible', 'energy_certificate', 'deed_status', 'seller_type', 'barter', 'city', 'county', 'district']

## Dosya yapısı

```text
data/raw/listing_dataset.csv
src/train_pure_listing_model.py
src/predict_pure_listing_price.py
artifacts/pure_listing_ridge_log_unit_price_v0.joblib
artifacts/pure_listing_gradient_boosting_log_unit_price_v0.joblib
artifacts/model_metrics_v0.json
artifacts/cv_predictions_v0.csv
artifacts/ridge_coefficients_v0.csv
artifacts/gb_feature_importance_v0.csv
sample_input.json
```

## Yeniden eğitim

Yeni 500–1000 ilanlık CSV geldiğinde dosyayı şu konuma koy:

```text
data/raw/listing_dataset.csv
```

Sonra çalıştır:

```bash
python src/train_pure_listing_model.py
```

## Tahmin denemesi

```bash
python src/predict_pure_listing_price.py
```

## Tez için önemli not

Bu model bir final değerleme motoru değil, pilot çalışmadır.

100 satırlık veriyle çıkan skorlar yalnızca pipeline kontrolü olarak görülmeli. 500–1000 ilan toplandığında özellikle İzmit içi mahalle, oda sayısı, bina yaşı, m² ve kat segmenti ilişkileri daha anlamlı hale gelir.

Tezde bu modeli şöyle konumlandırmak mantıklı olur:

> Bu çalışma kapsamında dış piyasa endeksi kullanılmadan, yalnızca ilan özelliklerine dayalı bir m² fiyat tahmin modeli kurulmuştur. Modelin amacı, ilan bazlı değişkenlerin konut fiyatı üzerindeki açıklayıcı etkisini incelemek ve İzmit ilçesi için pilot bir değerleme yaklaşımı geliştirmektir.
