# V15 County Specialist — run notes

V15, V14 local detail premium üzerine kuruludur. Global skordan çok **county-level lift** hedefler.
Price-tier / post-hoc correction yoktur. Title/photo/description feature yoktur.

## Amaç

V14’te Başiskele’de detail premium sinyali vardı ama tahminler ortalamaya sıkışıyordu
(`var(pred)/var(actual) ≈ 0.42`). V15:

1. Başiskele premium specialist (deterministic + fold-safe target stats)
2. Karamürsel `min_rows=180` override (global 250 kalır)
3. Large_home redesign feature’ları + segment raporu
4. Opsiyonel OOF-safe Başiskele variance-lift (conservative default)
5. Global MAPE/R² guardrail’i bozmamak

## V14 referans (group)

| Metrik | Değer |
|--------|------:|
| Global R² | 0.6787 |
| Global MAPE | 0.1290 |
| Başiskele R² | 0.4553 |
| Başiskele MAPE | 0.1103 |
| Başiskele variance ratio | 0.4224 |
| Gölcük R² | 0.6444 |
| Karamürsel R² | 0.5582 |
| İzmit R² | 0.7107 |
| ship_ready_all_counties_r2_ge_0_65 | false |

`detail-effect-mode full` V14’te kötüleşti; V15 final’de **group** seçilir.

## Başiskele mean-pulling

Detail premiums sinyal verdi ama model hâlâ uçları ortalamaya çekiyor. V15 premium skor,
bucket target stats ve (opsiyonel) variance-lift ile R² + variance ratio artırmayı dener.

## Karamürsel k180 override

```text
--county-expert-min-rows 250
--county-expert-min-rows-overrides "Karamürsel:180"
```

Başiskele / Gölcük / İzmit → 250; Karamürsel → 180. Parse edilemezse uyarı + global 250.

## Large_home redesign

Deterministic feature’lar (`large_home_m2_excess`, quality×m2, detail×m2, …) base pipeline’a girer.
Segment layer large_home için ridge / GB / ET / RF dener; `kept_base` / `used_blend` raporda açık yazılır.

## Leakage checklist

- `attr_effect_*`, `detail_effect_*`, `basiskele_*_target_stats` yalnız CV fold-train `y` ile fit
- Full-X target encoding precompute yok
- Variance-lift: delta modeli train fold `(actual-pred)` ile fit; validation actual görmez
- Effect CSV’leri final fitted encoder’dan (in-sample); seçim OOF metriklerine dayanır

## App-safe / deployment

Uygulama `front_*` / `view_*` / `near_*` / `out_*` / `in_*` / `subtype_*` almıyorsa
`--detail-effect-mode group|full` deploy edilmemeli. Title/photo/description yok.

## Defaults

```text
--demographics-mode safe
--attribute-mode full
--detail-effect-mode group
--county-expert-min-rows 250
--county-expert-min-rows-overrides "Karamürsel:180"
--basiskele-specialist-mode premium_target_stats
--basiskele-variance-lift conservative
--large-home-specialist-mode redesigned
```

## Success hedefleri (V15)

- Başiskele R² > 0.4553; variance ratio > 0.4224
- Karamürsel R² ≥ 0.5582 (tercihen ≥ 0.5768)
- Gölcük R² ≥ 0.62 soft floor
- Global R² ≥ 0.670; MAPE ≤ 0.134
- Direction pass ≥ 0.70; Karamürsel sale_diff_pct ≥ 0.03
- Long-term ship: her county R² ≥ 0.65 → aksi halde `ship_ready=false`

## Ship gate

`overall` PASS olabilir ama `ship_ready_all_counties_r2_ge_0_65=false` ise:

> **PASS as experiment, NOT ship-ready.**

## Komutlar

### Smoke (önce bunu çalıştır)

```bash
cd v15
python train_v15_county_specialist_pipeline.py --out outputs/v15_test --fast --limit-sale 800 --limit-rental 800 --demographics-mode safe --attribute-mode full --detail-effect-mode group --basiskele-specialist-mode premium_target_stats --basiskele-variance-lift conservative --county-expert-min-rows-overrides "Karamürsel:180" --no-run-demographics-ablation --no-run-attribute-ablation --no-run-detail-effect-ablation --no-run-basiskele-specialist-ablation --no-interactive
```

### Full + Başiskele specialist ablation

```bash
python train_v15_county_specialist_pipeline.py --out outputs/v15_full --demographics-mode safe --attribute-mode full --detail-effect-mode group --basiskele-specialist-mode premium_target_stats --basiskele-variance-lift conservative --county-expert-min-rows-overrides "Karamürsel:180" --run-basiskele-specialist-ablation --no-interactive
```

## Önemli raporlar

- `metrics_summary_v15.json`
- `county_expert_layer_report_v15.csv` (`min_rows_used`, `override_used`, …)
- `basiskele_premium_specialist_diagnostics_v15.csv`
- `basiskele_variance_diagnostics_v15.csv` / `basiskele_variance_lift_report_v15.csv`
- `large_home_diagnostics_v15.csv`
- `metrics_basiskele_specialist_ablation_v15.csv`
- `detail_premium_*` / `karamursel_sensitivity_v15.csv`

## Debug

```bash
python scripts/debug_single_prediction_features_v15.py --input-a a.json --input-b b.json --bundle-path v15/outputs/.../artifacts/model_bundle_v15.joblib --out outputs/v15_debug_pair
```
