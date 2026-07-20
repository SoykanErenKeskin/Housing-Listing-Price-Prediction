# V17 — Location + Geo-Context Pipeline

**Era:** V2 (Kocaeli multi-county + location features)  
**Status:** Archived reference run notes. Prefer curated packs under
`v2/best_checkpoints/best_kocaeli_location_checkpoint/` for metrics/reports.

## Why V17?

In V16, raw `latitude` / `longitude` improved Başiskele R² by roughly **+1%**.
The real goal is not “coordinates exist/missing,” but giving the model the
**real-estate meaning of location**:

- distance to sea / gulf shoreline
- distance to main road / TEM / D-100 proxies
- school / health / market / park access
- coastal band flags
- micro-location clusters
- nearby / similar comparables

Why V16 stalled:

- Başiskele still compressed / mean-pulling (`variance_ratio ≈ 0.44`)
- Regime residual layers (large_home / spread / Karamürsel location-age) did not
  beat the final config in ablation → **kept off by default in V17**
- Ship gate (`all counties R2 ≥ 0.65`) remained false

## V16 reference

| Metric | V16 |
|--------|-----|
| Global R2 | 0.6803 |
| Global MAPE | 0.1285 |
| İzmit R2 | 0.7161 |
| Gölcük R2 | 0.6422 |
| Karamürsel R2 | 0.5930 |
| Başiskele R2 | 0.4402 |
| Başiskele variance | 0.4432 |

## Offline geo-context cache (run this first)

Training runs **without internet**. Build the cache beforehand:

```bash
python scripts/build_geo_context_cache_v17.py --city Kocaeli --out data/external/geo_context --source osm
```

If OSM is unreachable, use the seed fallback:

```bash
python scripts/build_geo_context_cache_v17.py --out data/external/geo_context --source seed
```

Files written under `data/external/geo_context/`:

- `kocaeli_pois.parquet` or `.csv`
- `kocaeli_roads.parquet` or `.csv`
- `kocaeli_coastline.parquet` / `.csv` / `.geojson`
- `kocaeli_anchors.json` — **approx static anchors** (not survey-grade)
- `geo_context_metadata.json`

Distance math uses **EPSG:32635 (UTM 35N)** as the metric CRS.

## Feature groups

### `--location-feature-mode`

| Mode | Contents |
|------|----------|
| `none` | V16-like, no location |
| `basic` | lat/lon + precision/coverage flags |
| `geo` | basic + centroid/anchor/cluster + **GeoContext** (coast/POI/road) |
| `comparable` | basic + fold-safe comparable stats |
| `full` | geo + comparable |

### `--geo-context-mode` (inside geo/full)

| Mode | Contents |
|------|----------|
| `geo_no_poi` | coverage flags only |
| `geo_with_coast` | sea distance + coastal flags |
| `geo_with_poi` | coast + POI/road |
| `full` | full context + Başiskele interactions |

### Reference-ID comparable mapping

- **Narrow:** similarity-based `similar_k_*`
- **Deep:** nearest-neighbor `nearest_k_*` (distance-based)
- **Broad:** county/district broad similar
- **Weighted:** distance × similarity

## Leakage checklist

- Location + GeoContext **do not use the target** → no leakage
- Comparable **does use the target** → must be fold-safe; exclude self-match / same `classified_id`
- On `district_only` / missing locations, distance features are unreliable → use
  `location_quality_score` + missing flags
- At inference, comparable stats should come from historical listings in the DB
- Metadata: distance features that require `exact_map` are flagged in `location_feature_metadata`

## App-safe / inference

Location features are treated as app-safe (user can pick a map pin or
neighborhood/street). Model metadata records which features require `exact_map`.

## Kept / disabled

Kept from prior stack: V16 base, demographics safe, attribute full, detail group,
Karamürsel `min_rows` 180, residual target, anomaly filter, segment + county expert,
heartbeat, interactive CLI, reports.

Default-off (failed V16 ablation):

- `basiskele_large_home_regime=none`
- `basiskele_spread_layer=none`
- `karamursel_baseline_mode=none`

## Location scope (important)

Location coverage is still mostly filled for **Başiskele**. Missing-location patterns
in other counties can hurt the global model.

`--location-scope basiskele_only` (default):

- Location/geo/context features carry values **only on Başiskele rows**
- Other counties: numeric → `0`, categorical → `location_not_used`

`--location-scope global`:

- Location is disabled for counties with lat/lon coverage **< 0.40**
- Warning: `location_disabled_for_county_due_to_low_coverage:...`

## Fast-mode warning

`fast_mode=true` / `--limit-*` smoke results are **not comparable**.
V15/V16 comparisons must use **full train** only.

## Commands

Coverage audit (before train):

```bash
python scripts/audit_location_coverage_v17.py --out reports
```

Smoke (code validation only — do not interpret R2):

```bash
python v17/train_v17_location_features_pipeline.py --out outputs/v17_test --fast --limit-sale 800 --limit-rental 800 --demographics-mode safe --attribute-mode full --detail-effect-mode group --location-feature-mode geo --geo-context-mode geo_with_coast --location-scope basiskele_only --county-expert-min-rows-overrides "Karamürsel:180" --no-run-location-ablation --no-interactive
```

Full Başiskele geo + ablation (comparable metrics):

```bash
python v17/train_v17_location_features_pipeline.py --out outputs/v17_basiskele_geo_full --demographics-mode safe --attribute-mode full --detail-effect-mode group --location-feature-mode geo --geo-context-mode geo_with_coast --location-scope basiskele_only --county-expert-min-rows 180 --county-expert-min-rows-overrides "Karamürsel:180" --run-location-ablation --no-interactive
```

After the tree move, prefer paths under `v2/source_versions/v17/` and repo-root
`data/external/geo_context` / `.env`.

## Location ablation

- `control_v16_like` — location none
- `basiskele_basic` — basic + basiskele_only
- `basiskele_geo` — geo + basiskele_only
- `basiskele_geo_context` — geo + geo_with_coast + basiskele_only
- `global_geo` — geo + global (coverage gate)

Selection: Başiskele R2 > control, without MAPE/guardrail regression.

## PASS / ideal targets

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

Ship gate: every county R2 ≥ 0.65.

## Known reference metrics (V17 geo full, historical)

From curated checkpoint notes (approximate):

| Metric | Value |
|--------|------:|
| Global R2 | ≈ 0.6888 |
| Başiskele R2 | ≈ 0.4834 |
| Başiskele variance_ratio | ≈ 0.4346 |
