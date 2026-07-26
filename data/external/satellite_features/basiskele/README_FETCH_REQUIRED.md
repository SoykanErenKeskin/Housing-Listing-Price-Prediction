# Satellite features (V22) — Başiskele

Free Sentinel / GEE environment features for the V22 pilot.

## Status

CSV **already generated** (GEE fetch complete):

| Field | Value |
|---|---|
| Feature CSV | `data/external/satellite_features/basiskele/sentinel_features_v22.csv` |
| Metadata | `data/external/satellite_features/basiskele/metadata_v22.json` |
| Rows | 997 |
| Coverage | ≈ 0.77 |
| Source | Google Earth Engine (Sentinel-2 SR proxies) |

V22 full ablation selected `control_v21` — **diagnostic no-lift**. Do not treat these
features as a promoted model input for production checkpoints.

## Re-fetch (optional)

1. Install: `pip install earthengine-api`
2. Authenticate: `earthengine authenticate`
3. Run (resume-safe):

```powershell
python v4/shared_scripts/fetch_sentinel_features_v22.py `
  --city Kocaeli --county Başiskele `
  --out data/external/satellite_features/basiskele/sentinel_features_v22.csv `
  --source gee --resume --save-every 25
```

## Rules

- No Google Maps Static API
- No Mapbox / MapTiler / paid tiles
- No OSM tile bulk download
- No CNN fine-tune
- Features are environmental proxies only (NDVI / NDWI / NDBI / brightness / texture)

If the CSV is missing, V22 training runs **control_v21 only**.
