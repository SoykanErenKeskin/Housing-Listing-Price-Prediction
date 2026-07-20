# V8 DB Residual Pipeline

Bu sürüm V7'nin üzerine üç değişiklik getirir:

1. CatBoost eklenmedi.
2. Hedef varsayılan olarak residual hale getirildi: model doğrudan m² fiyatı değil, lokasyon baseline'ına göre sapmayı öğrenir.
3. Outlier temizliği güçlendirildi: temel IQR filtresinden sonra fiyat / lokasyon baseline oranına göre ek filtre uygulanır.

## Kurulum

```bash
pip install pandas numpy scikit-learn joblib matplotlib sqlalchemy psycopg2-binary python-dotenv
```

## DB ile normal çalıştırma

`.env` dosyasına şunu koy:

```env
DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/DB?sslmode=require
```

Sonra:

```bash
python train_v8_db_residual_pipeline.py --out outputs/v8_kocaeli
```

## Hızlı test

```bash
python train_v8_db_residual_pipeline.py --out outputs/v8_test --fast
```

## Varsayılan hedef modu

Varsayılan:

```bash
--target-mode residual
```

Alternatif kıyas için şunlar da çalışır:

```bash
--target-mode log
--target-mode raw
```

## Outlier filtresi

Varsayılan olarak açıktır:

```bash
--location-outlier-filter
```

Kapatmak için:

```bash
--no-location-outlier-filter
```

Filtre sınırları:

```bash
--min-location-ratio 0.50 --max-location-ratio 1.90
```

Bu şu anlama gelir: ilan m² fiyatı, kendi lokasyon baseline'ının yaklaşık 0.50x altında veya 1.90x üstündeyse eğitimden çıkarılır. Çıkarılanlar `data/input/sales_removed_location_outliers_v8.csv` dosyasına yazılır.

## Çıktılar

Ana dosyalar:

- `data/input/sales_training_table_v8.csv`
- `data/input/sales_removed_location_outliers_v8.csv`
- `data/output/oof_predictions_v8.csv`
- `reports/metrics_summary_v8.json`
- `reports/model_comparison_v8.csv`
- `reports/error_by_*_v8.csv`
- `reports/actual_vs_predicted_v8.png`
- `reports/residual_distribution_v8.png`
- `artifacts/ensemble_model_bundle_v8.joblib`

## Not

V8 bilinçli olarak title/text, fotoğraf ya da kullanıcıdan alınmayacak premium sinyalleri eklemez. Lokasyon baseline; county, district, m² grubu, oda sayısı ve varsa trend_sale_m2 gibi uygulamada karşılığı olabilecek sinyallerden oluşturulur.
