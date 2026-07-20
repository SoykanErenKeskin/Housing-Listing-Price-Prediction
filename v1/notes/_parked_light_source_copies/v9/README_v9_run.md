# V9 Segment-Aware Residual Pipeline

Bu sürüm V8'in üstüne kuruldu.

Ana farklar:
- CatBoost yok.
- Outlier temizliği V8 ile aynı mantıkta kaldı.
- Hedef residual: `log(unit_price_gross) - log(location_baseline)`.
- Segment-aware ensemble eklendi.
- İlçe bazlı R² / log R² / MAPE / MAE raporu eklendi.

## Çalıştırma

```powershell
python train_v9_segment_aware_pipeline.py --out outputs/v9_kocaeli
```

Hızlı test:

```powershell
python train_v9_segment_aware_pipeline.py --out outputs/v9_test --fast
```

## Önemli çıktılar

- `reports/metrics_summary_v9.json`
- `reports/county_metrics_v9.csv`
- `reports/segment_layer_report_v9.csv`
- `reports/model_comparison_v9.csv`
- `data/output/oof_predictions_v9.csv`
- `artifacts/segment_aware_model_bundle_v9.joblib`

## Segment-aware mantık

Base ensemble her satır için tahmin verir. Sonra bazı app-safe segmentlerde uzman model eğitilir:

- `large_home`: 151 m² ve üzeri veya 4+ oda
- `compact_home`: 85 m² ve altı veya 1 oda
- `old_building`: 26 yaş ve üzeri
- `mainstream_home`: 85–151 m², 2–3 oda, 26 yaş altı

Segment modeli sadece kendi segmentinde base ensemble'dan daha iyi MAPE verirse kullanılır. Aksi halde base tahmin korunur.
