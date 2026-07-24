# Satellite features fetch required (V22)

Free Sentinel / GEE environment features were not generated yet.

## Expected output

`data/external/satellite_features/basiskele/sentinel_features_v22.csv`

## How to fetch (Google Earth Engine)

1. Install: `pip install earthengine-api`
2. Authenticate: `earthengine authenticate`
3. Run:

```powershell
python v4/shared_scripts/fetch_sentinel_features_v22.py --city Kocaeli --county Başiskele --out data/external/satellite_features/basiskele/sentinel_features_v22.csv --source gee
```

## Rules

- No Google Maps Static API
- No Mapbox / MapTiler / paid tiles
- No OSM tile bulk download
- No CNN fine-tune
- Features are environmental proxies only (NDVI / NDWI / NDBI / brightness / texture)

Until the CSV exists, V22 training runs **control_v21 only**.
