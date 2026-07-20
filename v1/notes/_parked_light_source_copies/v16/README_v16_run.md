# V16 Regime Residual — run notes

V16, V15 county specialist üzerine kuruludur. **Daha fazla genel feature değil**;
kontrollü rejim bazlı residual / baseline deneyleri hedeflenir.

V15 dosyalarına dokunulmaz. Tüm değişiklikler `v16/` içindedir.

## Faz 0 teşhis bulguları (neden V16)

- Başiskele mean-pulling confirmed: `var(pred)/var(actual) ≈ 0.452`
- En ucuz decile ≈ +5.7k TL/m² fazla; en pahalı ≈ −10.5k TL/m² düşük
- Large_home R² ≈ 0.24 vs non-large ≈ 0.50 (share ≈ %22.7)
- `m2_group=200+` R² ≈ −0.03; `room_count=4+1` R² ≈ 0.14
- Karamürsel n≈202 sparsity; hata 4 Temmuz / Kayacık / Ereğli’de yoğun
- Karamürsel `building_age_group=31+` R² ≈ 0.02

V15 specialist genel R² lift vermedi → V16 rejim odaklı.

## V16 hipotezleri

1. **Başiskele large_home regime features** (deterministic, app-safe) large/200+/4+1
   varyansını modele görünür kılar.
2. **Başiskele spread residual** (OOF-safe) ucuz/pahalı uç bias’ını küçültür.
3. **Karamürsel location×age baseline** (fold-safe residual medians) sparse ilçede
   aggressive expert zorlamadan düzenli sinyal verir.
4. Residual layer’lar guardrail fail ederse **disabled** kalır (skeleton + rapor korunur).

## Neden genel feature eklenmedi?

Faz 0, sinyal eksikliğinden çok **rejim ve spread** problemi gösterdi.
Generic attr/detail genişletmesi V14/V15’te zaten denendi; Başiskele R² lift gelmedi.

## Neden Başiskele large_home / spread?

- Large_home share küçük ama R² çöküşü büyük → hedefe odaklı feature + opsiyonel residual.
- Mean-pulling MAPE’yi iyi, R²’yi kötü gösterir → spread residual variance_ratio + uç bias’a bakar.
- V12 tarzı global price-tier correction **yok**.

## Neden Karamürsel’de aggressive expert değil location-age?

- n≈202; expert blend zorlamak overfitting / nestabilite riski.
- Mahalle + yaş heterojenliği → `district × age/m2/room` smoothed residual medians.
- Ayrı post-hoc correction yok; feature olarak modele girer.

## V15 referans (full)

| Metrik | Değer |
|--------|------:|
| Global R² | 0.6799 |
| Global MAPE | 0.1290 |
| Başiskele R² | 0.4534 |
| Başiskele MAPE | 0.1110 |
| Başiskele variance ratio | 0.4516 |
| Başiskele large_home R² | 0.2396 |
| Gölcük R² | 0.6481 |
| Karamürsel R² | 0.5681 |
| İzmit R² | 0.7109 |
| ship_ready_all_counties_r2_ge_0_65 | false |

## Defaults

```text
--demographics-mode safe
--attribute-mode full
--detail-effect-mode group
--county-expert-min-rows 180
--county-expert-min-rows-overrides "Karamürsel:180"
--basiskele-large-home-regime simple
--basiskele-spread-layer conservative
--karamursel-baseline-mode location_age
--basiskele-variance-lift none
--no-run-v16-regime-ablation
```

Kullanılmayanlar: `detail-effect-mode full` final, title/photo/description,
V12 price-tier, zorla yükseltilmiş county expert blend.

## Leakage checklist

- `attr_effect_*`, `detail_effect_*`, `basiskele_*_target_stats`,
  `karamursel_*_residual_median` yalnız CV fold-train `y` ile fit
- Large_home / spread residual: delta modeli **train fold** `(actual − pred_current)` ile fit;
  validation actual fit’te yok
- Spread decile/rank: train fold predicted quantile threshold → val’a uygulanır
- Effect CSV’leri final fitted encoder’dan (in-sample); seçim OOF metriklerine dayanır
- App-safe: runtime’da olmayan title/photo/description yok

## Ship gate

Ideal hedefler:

- Başiskele R² ≥ 0.50; variance_ratio ≥ 0.55; large_home R² lift ≥ +0.08
- Karamürsel R² ≥ 0.60

Final **PASS** için: global guardrail (R²≥0.670, MAPE≤0.131) + sensitivity +
V15’e göre no-regression yeterli olabilir.

`ship_ready_all_counties_r2_ge_0_65=true` ancak her county R² ≥ 0.65.

`overall=PASS` ama ship_ready=false ise:

> **PASS as experiment, NOT ship-ready.**

## Raporlar

- `reports/metrics_summary_v16.json` (decision + `v15_delta` + `selected_v16_layers`)
- `reports/county_metrics_v16.csv`
- `reports/model_comparison_v16.csv`
- `reports/metrics_v16_regime_ablation.csv`
- `reports/basiskele_large_home_residual_layer_v16.csv`
- `reports/basiskele_spread_residual_layer_v16.csv`
- `reports/karamursel_location_age_baseline_v16.csv`
- `reports/basiskele_decile_bias_v16.csv`
- `reports/basiskele_large_home_error_v16.csv`
- `reports/karamursel_error_by_segment_v16.csv`
- `reports/large_home_diagnostics_v16.csv`
- `reports/county_error_heatmap_v16.csv`

## Komutlar

### Smoke

```bash
cd v16
python train_v16_regime_residual_pipeline.py --out outputs/v16_test --fast --limit-sale 800 --limit-rental 800 --demographics-mode safe --attribute-mode full --detail-effect-mode group --county-expert-min-rows 180 --county-expert-min-rows-overrides "Karamürsel:180" --basiskele-large-home-regime simple --basiskele-spread-layer conservative --karamursel-baseline-mode location_age --no-run-v16-regime-ablation --no-interactive
```

### Full (+ regime ablation)

```bash
cd v16
python train_v16_regime_residual_pipeline.py --out outputs/v16_full --demographics-mode safe --attribute-mode full --detail-effect-mode group --county-expert-min-rows 180 --county-expert-min-rows-overrides "Karamürsel:180" --basiskele-large-home-regime simple --basiskele-spread-layer conservative --karamursel-baseline-mode location_age --run-v16-regime-ablation --no-interactive
```

### Feature debug

```bash
python ../scripts/debug_single_prediction_features_v16.py --a path/a.json --b path/b.json
```
