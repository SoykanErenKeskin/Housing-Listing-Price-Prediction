# Housing-Listing-Price-Prediction v6 — Sales Model with Rental Market Features

Bu paket, eski satılık veri setini kiralık ilanlardan üretilen mahalle bazlı kira feature'ları ile birleştirip satış m² fiyat modelini tekrar eğitir.

## Mantık

Satılık modelin target'ı yine `unit_price_gross`.
Kiralık veriden ise dış piyasa sinyali üretilir:

```text
district_rent_m2_median
district_rent_m2_mean
district_rent_m2_count
district_rent_m2_iqr
district_rent_m2_cv
district_room_rent_m2_median
district_room_rent_m2_count
district_m2_group_rent_m2_median
district_m2_group_rent_m2_count
county_rent_m2_median
county_rent_m2_count
estimated_rent_m2_gross
estimated_monthly_rent_gross
rent_feature_confidence
rent_feature_level
```

Bu feature'lar satış fiyatından üretilmez. Yani satış target leakage yoktur.

## Dosyalar

```text
data/input/sale_listings_dataset.csv
data/input/rental_listings.csv
src/train_v6_sales_with_rental_features.py
src/predict_v6.py
```

## Çalıştırma

```bash
python src/train_v6_sales_with_rental_features.py
```

Ana çıktılar:

```text
reports/model_metrics_v6_rental.json
reports/model_comparison_v6_rental.csv
reports/rental_feature_report_v6.csv
data/output/sales_with_rental_features_preview.csv
artifacts/best_model_v6_rental_by_r2.joblib
```

## Önemli not

`estimated_monthly_rent_gross` model input olarak kullanılabilir çünkü yalnızca kira m² medyanı ve satış ilanının brüt m² bilgisinden gelir.
Ama `sale_price / rent` veya `amortization_months` gibi direkt satış fiyatını kullanan değişkenler model input'u yapılmamalıdır; bunlar target leakage olur. Amortismanı tahminden sonra raporlama için hesaplamak doğru yoldur.

## Beklenti

Kira feature'ları özellikle mahalle cazibesini ve piyasa seviyesini yakalamaya yarar. MAPE/MAE tarafında iyileşme beklemek mantıklı; ancak sonuç, kiralık verinin mahalle/oda kırılımındaki coverage'ına bağlıdır.


## v6.1 fix

Bu sürüm Neon export kaynaklı hatayı düzeltir.

Yeni `sale_listings.csv` dosyasında helper'ın ürettiği `front_*`, `view_*`, `transport_*`, `near_*`, `out_*`, `in_*`, `subtype_*` kolonları üst seviyede değil, `raw` JSON kolonu içinde geliyor.

v6.1 artık:
- `raw` JSON içindeki detail binary kolonları üst seviyeye çıkarır.
- `building_age_raw`, `building_age_group`, `detail_*` alanlarını da `raw` içinden tamamlar.
- Detail kolon hiç bulunamazsa `KeyError: 'ones'` hatası yerine boş coverage raporu üretir.

Çalıştır:

```bash
python src/train_v6_1_sales_with_rental_features.py
```
