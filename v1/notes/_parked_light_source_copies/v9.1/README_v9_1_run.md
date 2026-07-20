# V9.1 Segment-Aware Pipeline

Bu sürüm V9'daki segment layer karar mantığını düzeltir.

## Ana fark
V9 yalnızca segment modeli tek başına base ensemble'dan iyi olursa segment katmanını kullanıyordu. V9.1 ise her segment için şu blend ağırlıklarını dener:

0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.00

Segmentte MAPE'yi en çok iyileştiren blend varsa final tahmine uygular. Outlier temizliği V8/V9 ile aynıdır. CatBoost yoktur.

## Çalıştırma

```bash
python train_v9_1_segment_aware_pipeline.py --out outputs/v9_1_kocaeli
```

Hızlı test:

```bash
python train_v9_1_segment_aware_pipeline.py --out outputs/v9_1_test --fast
```

## Bakılacak çıktılar

- reports/metrics_summary_v9_1.json
- reports/county_metrics_v9_1.csv
- reports/segment_layer_report_v9_1.csv
- data/output/oof_predictions_v9_1.csv
