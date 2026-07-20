# V12 price-tier correction pipeline

V12, V11.3 üzerine kuruldu.

Eklenen ana katman:

- Demografi DB join korunur.
- Mahalle demografilerinden county-level aggregate feature üretimi korunur.
- Anomaly filtreleme korunur.
- Segment-aware layer korunur.
- County expert layer korunur.
- Yeni V12 price-tier correction layer eklenir.

Yeni katman, county+segment sonrası OOF tahminlerde kalan pahalı/ucuz uç sapmasını öğrenir. Correction target şu mantıktadır:

```text
correction_pct = (actual_unit_price_gross - pred_after_county) / pred_after_county
```

Bu correction app-safe feature'lardan öğrenilir. En iyi blend OOF üzerinde denenir. Eğer MAPE iyileşmiyorsa layer otomatik `kept_current` kalır.

## Hızlı test

```powershell
python train_v12_price_tier_pipeline.py `
  --out outputs/v12_test `
  --fast `
  --limit-sale 800 `
  --limit-rental 800 `
  --demographics-mode safe `
  --no-run-demographics-ablation
```

Tek satır:

```powershell
python train_v12_price_tier_pipeline.py --out outputs/v12_test --fast --limit-sale 800 --limit-rental 800 --demographics-mode safe --no-run-demographics-ablation
```

## Full train, safe final + ablation

```powershell
python train_v12_price_tier_pipeline.py `
  --out outputs/v12_full `
  --demographics-mode safe `
  --run-demographics-ablation
```

Tek satır:

```powershell
python train_v12_price_tier_pipeline.py --out outputs/v12_full --demographics-mode safe --run-demographics-ablation
```

## Sadece safe full train

```powershell
python train_v12_price_tier_pipeline.py --out outputs/v12_safe_full --demographics-mode safe --no-run-demographics-ablation
```

## Yeni parametreler

```text
--price-tier-correction / --no-price-tier-correction
--price-tier-low-quantile 0.15
--price-tier-high-quantile 0.85
--price-tier-min-rows 250
```

## Bakılacak ana çıktılar

```text
outputs/v12_full/reports/metrics_summary_v12.json
outputs/v12_full/reports/metrics_demographics_ablation_v12.csv
outputs/v12_full/reports/price_tier_correction_report_v12.csv
outputs/v12_full/reports/price_tier_decile_report_v12.csv
outputs/v12_full/reports/county_metrics_v12.csv
outputs/v12_full/data/output/oof_predictions_v12.csv
```

Özellikle `price_tier_decile_report_v12.csv` dosyasında en ucuz ve en pahalı decile için `mean_bias_before` ve `mean_bias_after` farkına bak.

## Not

V12 layer otomatik korumalıdır. Price-tier correction MAPE'yi iyileştirmezse kullanılmaz ve final prediction V11.3 tarzı county+segment sonrası tahminde kalır.
