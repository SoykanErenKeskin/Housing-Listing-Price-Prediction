# V14 Local Detail Premium — run notes

V14, V13 attribute-sensitivity üzerine kuruludur. Price-tier correction yoktur.
Yeni katman: fold-safe `LocalDetailPremiumEncoder` — listing detail binary’lerinin
(`front_*`, `view_*`, `near_*`, `out_*`, `in_*`, `subtype_*`) lokasyon bağlamındaki
residual premium etkisini öğrenir.

## Hipotezler

- **H1:** Local detail premiums Başiskele R² / variance_ratio lift sağlar.
- **H2:** Karamürsel V13 default’tan kötüleşmez.
- **H3:** Global MAPE/R² V13 guardrail içinde kalır (`MAPE ≤ V13+0.005`, `R² ≥ V13−0.01`).
- **H4:** `group` mode stabil; `full` challenger (overfit riski).

## Leakage checklist

- Encoder **yalnızca** sklearn Pipeline içinde fold-train `y` ile fit edilir.
- Full dataframe üzerinde önceden target encoding / effect precompute **yok**.
- Residual target önerilir: `log(price) - log(location_baseline)`.
- Effect CSV’leri **final fitted** pipeline encoder state’inden export edilir (in-sample).
  OOF fold encoder’larından export edilmez. Bu yüzden effect tabloları “in-sample final
  encoder effects” olarak okunmalıdır; ablation seçimi OOF metriklerine dayanır (ayrı holdout yok).

## App-safe / deployment uyarısı (zorunlu)

Uygulama kullanıcıdan `front_*` / `view_*` / `near_*` / `out_*` / `in_*` / `subtype_*`
detaylarını **almıyorsa**:

- `--detail-effect-mode group|full` **deploy edilmemelidir**
- Ya app formu bu alanları ekler
- Ya da deployment modeli `--detail-effect-mode none` olur

Aksi halde model eğitimde görüp inference’ta 0 kalan feature’lara bağımlı kalır (sessiz degrade).

## Defaults

```text
--demographics-mode safe
--attribute-mode full
--detail-effect-mode group
--county-expert-min-rows 250
```

k180 (`--county-expert-min-rows 180`) ayrı deneydir; ana V14 run’a girmez.

## Komutlar

### Smoke (önce bunu çalıştır)

```bash
cd v14
python train_v14_local_detail_premium_pipeline.py --out outputs/v14_test --fast --limit-sale 800 --limit-rental 800 --demographics-mode safe --attribute-mode full --detail-effect-mode group --no-run-demographics-ablation --no-run-attribute-ablation --no-run-detail-effect-ablation --no-interactive
```

### Full + detail ablation

```bash
python train_v14_local_detail_premium_pipeline.py --out outputs/v14_full --demographics-mode safe --attribute-mode full --detail-effect-mode group --run-detail-effect-ablation --no-interactive
```

### Karamürsel k180 deney

```bash
python train_v14_local_detail_premium_pipeline.py --out outputs/v14_k180 --demographics-mode safe --attribute-mode full --detail-effect-mode group --county-expert-min-rows 180 --no-interactive
```

## Decision / ship gate

`metrics_summary_v14.json` → `decision`:

- `pass_global_guardrail`, `pass_basiskele_lift`, `pass_karamursel_guardrail`, `pass_detail_sensitivity`
- `ship_ready_all_counties_r2_ge_0_65` — **tüm** ilçe R² ≥ 0.65 değilse `false`
  (V14 PASS olsa bile ship-ready sayılmaz)
- Gölcük R² < 0.62 → warning; < 0.55 → ciddi QA finding (overall PASS’i tek başına düşürmez)

## Önemli raporlar

- `detail_feature_coverage_v14.csv`
- `detail_premium_effects_by_county_v14.csv`
- `basiskele_detail_premium_diagnostics_v14.csv`
- `detail_premium_group_summary_v14.csv`
- `detail_premium_feature_importance_v14.csv`
- `metrics_detail_effect_ablation_v14.csv`
- `basiskele_variance_diagnostics_v14.csv`
- `karamursel_sensitivity_v14.csv`

## Debug

```bash
python ../scripts/debug_single_prediction_features_v14.py --input-a ../samples/karamursel_old_house.json --input-b ../samples/karamursel_new_house.json --bundle-path outputs/v14_full/artifacts/model_bundle_v14.joblib --out outputs/v14_debug_pair
```

`detail_effect_diff.csv` bundle path ile üretilir.
