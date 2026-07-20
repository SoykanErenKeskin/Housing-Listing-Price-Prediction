# V13 Attribute Sensitivity Pipeline

V13, V12 üzerine kuruludur. Price-tier correction **yoktur** (V12'de `kept_current` olduğu için tekrar edilmez).

## Hipotezler

- **H1:** Aynı lokasyon + aynı m²'de kalite farkı tahmini değiştirir (Karamürsel `sale_diff_pct >= 3%`).
- **H2:** Başiskele prediction variance compression azalır.
- **H3:** Global MAPE/R² V12 safe guardrail içinde kalır.

## Go / no-go

| Kural | Eşik |
|---|---|
| Karamürsel sensitivity | `sale_diff_pct >= 0.03` |
| Direction pass rate | `>= 0.70` |
| Guardrail MAPE | `<= V12_MAPE + 0.005` (~0.1344) |
| Guardrail R² | `>= V12_R2 - 0.01` (~0.6639) |

Final attribute mode seçimi: **full** (guardrail+sensitivity) → değilse **basic** → `none` sadece fallback.

## Amenity / heating normalize contract

Binary flags (`elevator`, `parking`, `balcony`, `furnished`, `site_inside`, `credit_eligible`):

- `1`: var / evet / true / 1 / (otopark içeren değerler)
- `0`: yok / hayır / false / 0
- `NaN`: missing / belirtilmemiş / boş

Heating score aliases: Yerden Isıtma, Merkezi, Merkezi (Pay Ölçer), Kombi (Doğalgaz), Doğalgaz Sobası, Klima, Soba, Yok, …

## Leakage checklist

- `attr_effect_*` sadece CV fold `fit` içinde, residual target üzerinde
- Full-X üzerinde target encoding precompute yok
- Title / photo / description yok

## Kira notu (önemli)

V13 satış unit-price modelidir. Uygulamada kira `district_rent_m2_median * gross_m2` ise aynı m²'de iki evin kirası aynı çıkar. Ayrı rent attribute multiplier **V14 backlog**; sales modeline karıştırılmaz.

## Hızlı test

```powershell
cd v13
python train_v13_attribute_sensitivity_pipeline.py --out outputs/v13_test --fast --limit-sale 800 --limit-rental 800 --demographics-mode safe --attribute-mode full --no-run-demographics-ablation --no-run-attribute-ablation
```

## Etkileşimli ayar sihirbazı

Varsayılan olarak, komut satırında **vermediğin** ana ayarlar terminalde sorulur
(**sklearn yüklenmeden önce** — menü hemen açılır):

- `↑` / `↓` ile şık seç, `Enter` ile onayla
- Her şıkkı altında kısa açıklama görünür
- Sadece `--out` serbest metin
- Örn. `python train_v13_attribute_sensitivity_pipeline.py --fast` → fast atlanır, diğerleri sorulur
- Tamamen sessiz/script: `--no-interactive`
- Ayarlardan sonra "Model kütüphaneleri yükleniyor..." görürsün; asıl bekleme buradadır

```powershell
cd v13
python train_v13_attribute_sensitivity_pipeline.py
```

## Full train (önerilen)

```powershell
cd v13
python train_v13_attribute_sensitivity_pipeline.py --out outputs/v13_full --demographics-mode safe --attribute-mode full --run-attribute-ablation --no-interactive
```

Ablation matrisi (en fazla 3+3, 9-grid yok):

- Attribute ablation: seçili demographics-mode sabit → none/basic/full
- Demographics ablation (opsiyonel): seçili attribute-mode sabit → none/safe/full

## Opsiyonel CLI

```text
--county-expert-min-rows 250   # default; Karamürsel deneyi için 180 verilebilir (ayrı run)
--run-demographics-ablation
--attribute-mode none|basic|full
```

## Ana çıktılar

```text
outputs/v13_full/reports/metrics_summary_v13.json
outputs/v13_full/reports/metrics_attribute_ablation_v13.csv
outputs/v13_full/reports/feature_sensitivity_v13.csv
outputs/v13_full/reports/karamursel_sensitivity_v13.csv
outputs/v13_full/reports/basiskele_variance_diagnostics_v13.csv
outputs/v13_full/reports/attribute_feature_coverage_v13.csv
outputs/v13_full/artifacts/model_bundle_v13.joblib
```

## Debug (aynı feature builder)

```powershell
python scripts/debug_single_prediction_features.py `
  --input-a samples/karamursel_old_house.json `
  --input-b samples/karamursel_new_house.json `
  --bundle-path v13/outputs/v13_full/artifacts/model_bundle_v13.joblib `
  --out outputs/v13_debug_pair
```

Bundle yoksa FE + `attr_*` diff fallback çalışır.
