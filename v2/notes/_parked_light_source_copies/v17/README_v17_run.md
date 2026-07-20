# V17 — Location + Geo-Context Pipeline

## Neden V17?

V16’da ham `latitude`/`longitude` eklemek Başiskele R2’yi yaklaşık **+%1** artırdı.
Ama asıl hedef “koordinat var/yok” değil; **koordinatın gayrimenkul değerindeki anlamını** modele vermektir:

- denize / körfez sahiline mesafe
- ana yola / TEM / D-100 proxy mesafe
- okul / sağlık / market / park erişimi
- sahil bandı (coastal flags)
- mikro-lokasyon cluster
- yakın / benzer emsal (comparable)

V16 neden tıkandı?
- Başiskele hâlâ compressed / mean-pulling (`variance_ratio ≈ 0.44`)
- Regime residual katmanları (large_home / spread / karamursel location-age) ablation’da finali geçmedi → **V17’de default kapalı**
- Ship gate (`all counties R2 ≥ 0.65`) false kaldı

## V16 referans

| Metric | V16 |
|--------|-----|
| Global R2 | 0.6803 |
| Global MAPE | 0.1285 |
| İzmit R2 | 0.7161 |
| Gölcük R2 | 0.6422 |
| Karamürsel R2 | 0.5930 |
| Başiskele R2 | 0.4402 |
| Başiskele variance | 0.4432 |

## Offline geo-context cache (önce bunu çalıştır)

Training sırasında **internet yok**. Önce cache üret:

```bash
python scripts/build_geo_context_cache_v17.py --city Kocaeli --out data/external/geo_context --source osm
```

OSM erişilemezse seed fallback:

```bash
python scripts/build_geo_context_cache_v17.py --out data/external/geo_context --source seed
```

Üretilen dosyalar (`data/external/geo_context/`):

- `kocaeli_pois.parquet` veya `.csv`
- `kocaeli_roads.parquet` veya `.csv`
- `kocaeli_coastline.parquet` / `.csv` / `.geojson`
- `kocaeli_anchors.json` — **approx static anchors** (survey-grade değil)
- `geo_context_metadata.json`

Mesafe hesapları **EPSG:32635 (UTM 35N)** metric CRS ile yapılır.

## Feature grupları

### `--location-feature-mode`

| Mode | İçerik |
|------|--------|
| `none` | V16-like, location yok |
| `basic` | lat/lon + precision/coverage flags |
| `geo` | basic + centroid/anchor/cluster + **GeoContext** (coast/POI/road) |
| `comparable` | basic + fold-safe emsal stats |
| `full` | geo + comparable |

### `--geo-context-mode` (geo/full içinde)

| Mode | İçerik |
|------|--------|
| `geo_no_poi` | sadece coverage flags |
| `geo_with_coast` | deniz mesafesi + coastal |
| `geo_with_poi` | coast + POI/road |
| `full` | tüm context + Başiskele interactions |

### Reference-ID karşılık (comparable)

- **Dar**: similarity-based similar_k_*
- **Derin**: nearest_k_* distance-based
- **Yaygın**: county/district broad similar
- **Weighted**: mesafe × benzerlik

## Leakage checklist

- Location + GeoContext **target kullanmaz** → leakage yok
- Comparable **target kullanır** → mutlaka fold-safe fit; self-match / aynı `classified_id` exclude
- `district_only` / missing konumlarda mesafe feature’ları güvenilmez → `location_quality_score` + missing flags
- Inference’ta comparable stats DB’deki geçmiş ilanlardan üretilecek
- Metadata: `exact_map` gerektiren distance feature’lar `location_feature_metadata` içinde işaretlenir

## App-safe / inference

Location feature’ları app-safe kabul edilir (kullanıcı haritadan / mahalle-sokaktan konum verebilir).
Model metadata içinde hangi feature’ın `exact_map` gerektirdiği yazılır.

## Korunan / kapatılan

Korunan: V16 base, demographics safe, attribute full, detail group, Karamürsel min_rows 180,
residual target, anomaly filter, segment + county expert, heartbeat, interactive CLI, reports.

Default kapalı (V16 ablation geçmedi):
- `basiskele_large_home_regime=none`
- `basiskele_spread_layer=none`
- `karamursel_baseline_mode=none`

## Location scope (önemli)

Location şu an çoğunlukla **Başiskele**’de dolu. Diğer ilçelerde missing-location pattern global modeli bozabilir.

`--location-scope basiskele_only` (default):
- Location/geo/context feature’ları **sadece Başiskele** satırlarında değer taşır
- Diğer county’lerde numeric → `0`, categorical → `location_not_used`

`--location-scope global`:
- Lat/lon coverage **< 0.40** olan county’lerde location kapatılır
- Warning: `location_disabled_for_county_due_to_low_coverage:...`

## Fast mode uyarısı

`fast_mode=true` / `--limit-*` smoke sonuçları **kıyaslanamaz**.
V15/V16 karşılaştırması **yalnızca full train** ile yapılır.

## Komutlar

Coverage audit (train öncesi):

```bash
python scripts/audit_location_coverage_v17.py --out reports
```

Smoke (sadece kod doğrulama — R2 yorumlanmaz):

```bash
python v17/train_v17_location_features_pipeline.py --out outputs/v17_test --fast --limit-sale 800 --limit-rental 800 --demographics-mode safe --attribute-mode full --detail-effect-mode group --location-feature-mode geo --geo-context-mode geo_with_coast --location-scope basiskele_only --county-expert-min-rows-overrides "Karamürsel:180" --no-run-location-ablation --no-interactive
```

Full Başiskele geo + ablation (kıyaslanabilir):

```bash
python v17/train_v17_location_features_pipeline.py --out outputs/v17_basiskele_geo_full --demographics-mode safe --attribute-mode full --detail-effect-mode group --location-feature-mode geo --geo-context-mode geo_with_coast --location-scope basiskele_only --county-expert-min-rows 180 --county-expert-min-rows-overrides "Karamürsel:180" --run-location-ablation --no-interactive
```

## Location ablation

- `control_v16_like` — location none
- `basiskele_basic` — basic + basiskele_only
- `basiskele_geo` — geo + basiskele_only
- `basiskele_geo_context` — geo + geo_with_coast + basiskele_only
- `global_geo` — geo + global (coverage gate)

Selection: Başiskele R2 > control, MAPE/guardrail bozulmasın.

## PASS / ideal hedefler

PASS:
- Global MAPE ≤ 0.131
- İzmit R2 ≥ 0.70
- Başiskele R2 > V16
- Karamürsel R2 ≥ V16 − 0.02
- Gölcük R2 ≥ 0.62

Ideal:
- Global R2 ≥ 0.69
- Başiskele R2 ≥ 0.50, variance ≥ 0.50
- Gölcük ≥ 0.65, İzmit ≥ 0.71, Karamürsel ≥ 0.59

Ship gate: her county R2 ≥ 0.65.
