#!/usr/bin/env python
"""Build offline geo-context cache for V17 (OSM POI / roads / coastline).

Training must NOT call the internet. Run this once (or when refreshing OSM):

  python scripts/build_geo_context_cache_v17.py --city Kocaeli --out data/external/geo_context --source osm

Outputs under --out:
  kocaeli_pois.parquet|.csv
  kocaeli_roads.parquet|.csv
  kocaeli_coastline.parquet|.csv|.geojson
  kocaeli_anchors.json
  geo_context_metadata.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for cand in [here, *here.parents]:
        if (cand / "MANIFEST.json").is_file() and (cand / "data").is_dir():
            return cand
    return here.parents[1]


ROOT = _repo_root()
sys.path.insert(0, str(ROOT / "v2" / "source_versions" / "v17"))

# Kocaeli approximate bounding box (WGS84)
KOCAELI_BBOX = {
    "south": 40.55,
    "west": 29.40,
    "north": 41.00,
    "east": 30.25,
}

# Fallback gulf coastline polyline (approx static anchors) if OSM coastline fails
FALLBACK_COASTLINE = [
    {"lat": 40.6914, "lon": 29.6153, "name": "karamursel", "seq": 0},
    {"lat": 40.7000, "lon": 29.7000, "name": "karamursel_east", "seq": 1},
    {"lat": 40.7100, "lon": 29.7800, "name": "golcuk_west", "seq": 2},
    {"lat": 40.7167, "lon": 29.8167, "name": "golcuk", "seq": 3},
    {"lat": 40.7400, "lon": 29.8800, "name": "izmit_west", "seq": 4},
    {"lat": 40.7656, "lon": 29.9406, "name": "izmit", "seq": 5},
    {"lat": 40.7450, "lon": 29.9300, "name": "basiskele_north", "seq": 6},
    {"lat": 40.7200, "lon": 29.9050, "name": "basiskele_sahil", "seq": 7},
    {"lat": 40.7120, "lon": 29.9200, "name": "basiskele_coast", "seq": 8},
    {"lat": 40.7000, "lon": 29.9400, "name": "basiskele_east", "seq": 9},
]

STATIC_ANCHORS = {
    "izmit_center": {"lat": 40.7656, "lon": 29.9406},
    "basiskele_coast_center": {"lat": 40.7120, "lon": 29.9200},
    "basiskele_yuvacik": {"lat": 40.6780, "lon": 29.9750},
    "basiskele_bahcecik": {"lat": 40.6950, "lon": 29.9550},
    "basiskele_kullar": {"lat": 40.7350, "lon": 29.9800},
    "basiskele_sahil": {"lat": 40.7200, "lon": 29.9050},
    "golcuk_center": {"lat": 40.7167, "lon": 29.8167},
    "karamursel_center": {"lat": 40.6914, "lon": 29.6153},
}

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def _bbox_str(bbox: dict[str, float]) -> str:
    return f"{bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']}"


def overpass_query(query: str, timeout: int = 180) -> dict[str, Any]:
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    last_err: Exception | None = None
    for url in OVERPASS_URLS:
        req = urllib.request.Request(url, data=data, method="POST", headers={"User-Agent": "listing-geo-cache/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2.0)
    raise RuntimeError(f"Overpass request failed: {last_err}")


def _node_latlon(el: dict[str, Any], nodes: dict[int, tuple[float, float]]) -> tuple[float, float] | None:
    if "lat" in el and "lon" in el:
        return float(el["lat"]), float(el["lon"])
    if el.get("type") == "way" and "nodes" in el:
        pts = [nodes[n] for n in el["nodes"] if n in nodes]
        if not pts:
            return None
        return float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))
    center = el.get("center") or {}
    if "lat" in center and "lon" in center:
        return float(center["lat"]), float(center["lon"])
    return None


def _index_nodes(elements: list[dict[str, Any]]) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for el in elements:
        if el.get("type") == "node" and "lat" in el and "lon" in el:
            out[int(el["id"])] = (float(el["lat"]), float(el["lon"]))
    return out


def fetch_pois(bbox: dict[str, float]) -> pd.DataFrame:
    b = _bbox_str(bbox)
    query = f"""
    [out:json][timeout:180];
    (
      node["amenity"~"school|kindergarten|college|university|hospital|clinic|doctors|pharmacy|marketplace|bank|atm|ferry_terminal|restaurant|cafe"]({b});
      way["amenity"~"school|kindergarten|college|university|hospital|clinic|doctors|pharmacy|marketplace|bank|atm|ferry_terminal"]({b});
      node["shop"~"supermarket|convenience"]({b});
      way["shop"~"supermarket|convenience"]({b});
      node["leisure"="park"]({b});
      way["leisure"="park"]({b});
      node["highway"="bus_stop"]({b});
      node["public_transport"~"platform|station|stop_position"]({b});
      node["railway"="station"]({b});
      way["railway"="station"]({b});
    );
    out center tags;
    """
    payload = overpass_query(query)
    elements = payload.get("elements", [])
    nodes = _index_nodes(elements)
    rows = []
    for el in elements:
        tags = el.get("tags") or {}
        ll = _node_latlon(el, nodes)
        if ll is None:
            continue
        lat, lon = ll
        amenity = tags.get("amenity", "")
        shop = tags.get("shop", "")
        leisure = tags.get("leisure", "")
        highway = tags.get("highway", "")
        public_transport = tags.get("public_transport", "")
        railway = tags.get("railway", "")

        category = "other"
        subcategory = amenity or shop or leisure or highway or public_transport or railway or "other"
        if amenity in {"school", "kindergarten", "college", "university"} or subcategory in {"school", "kindergarten", "college", "university"}:
            category = "education"
        elif amenity in {"hospital", "clinic", "doctors", "pharmacy"}:
            category = "health"
        elif shop in {"supermarket", "convenience"} or amenity in {"marketplace", "restaurant", "cafe"} or leisure == "park" or amenity in {"bank", "atm"}:
            category = "daily"
        elif highway == "bus_stop" or public_transport or railway == "station" or amenity == "ferry_terminal":
            category = "transport"

        rows.append(
            {
                "osm_id": el.get("id"),
                "osm_type": el.get("type"),
                "lat": lat,
                "lon": lon,
                "category": category,
                "subcategory": subcategory,
                "name": tags.get("name", ""),
                "amenity": amenity,
                "shop": shop,
                "leisure": leisure,
                "highway": highway,
                "public_transport": public_transport,
                "railway": railway,
            }
        )
    return pd.DataFrame(rows)


def fetch_roads(bbox: dict[str, float]) -> pd.DataFrame:
    b = _bbox_str(bbox)
    query = f"""
    [out:json][timeout:180];
    (
      way["highway"~"motorway|trunk|primary|secondary|tertiary"]({b});
    );
    out geom tags;
    """
    payload = overpass_query(query)
    rows = []
    for el in payload.get("elements", []):
        tags = el.get("tags") or {}
        geom = el.get("geometry") or []
        if not geom:
            continue
        hwy = tags.get("highway", "")
        # sample every Nth vertex + endpoints to keep file smaller
        step = max(1, len(geom) // 20)
        sampled = geom[::step]
        if geom[-1] not in sampled:
            sampled = list(sampled) + [geom[-1]]
        for i, pt in enumerate(sampled):
            rows.append(
                {
                    "osm_id": el.get("id"),
                    "highway": hwy,
                    "name": tags.get("name", ""),
                    "ref": tags.get("ref", ""),
                    "lat": float(pt["lat"]),
                    "lon": float(pt["lon"]),
                    "vertex_i": i,
                    "road_class": _road_class(hwy, tags.get("ref", ""), tags.get("name", "")),
                }
            )
    return pd.DataFrame(rows)


def _road_class(highway: str, ref: str, name: str) -> str:
    ref_u = (ref or "").upper()
    name_u = (name or "").upper()
    if highway in {"motorway", "motorway_link"} or "TEM" in name_u or "O-4" in ref_u or "O4" in ref_u:
        return "highway"
    if highway in {"trunk", "trunk_link"} or "D100" in ref_u or "E-5" in name_u or "E5" in name_u or "D-100" in ref_u:
        return "primary"
    if highway in {"primary", "primary_link"}:
        return "primary"
    if highway in {"secondary", "secondary_link", "tertiary", "tertiary_link"}:
        return "major"
    return "other"


def fetch_coastline(bbox: dict[str, float]) -> pd.DataFrame:
    b = _bbox_str(bbox)
    query = f"""
    [out:json][timeout:180];
    (
      way["natural"="coastline"]({b});
      relation["natural"="coastline"]({b});
      way["natural"="water"]["water"~"sea|bay|lagoon"]({b});
    );
    out geom tags;
    """
    try:
        payload = overpass_query(query)
    except Exception as exc:  # noqa: BLE001
        print(f"Coastline OSM fetch failed ({exc}); using fallback polyline.")
        return pd.DataFrame(FALLBACK_COASTLINE)

    rows = []
    seq = 0
    for el in payload.get("elements", []):
        geom = el.get("geometry") or []
        tags = el.get("tags") or {}
        if not geom:
            continue
        step = max(1, len(geom) // 40)
        sampled = geom[::step]
        if geom[-1] not in sampled:
            sampled = list(sampled) + [geom[-1]]
        for pt in sampled:
            rows.append(
                {
                    "lat": float(pt["lat"]),
                    "lon": float(pt["lon"]),
                    "name": tags.get("name", "") or "coastline",
                    "seq": seq,
                    "osm_id": el.get("id"),
                    "source": "osm",
                }
            )
            seq += 1
    if len(rows) < 5:
        print("Coastline OSM returned too few points; using fallback polyline.")
        return pd.DataFrame([{**r, "source": "fallback"} for r in FALLBACK_COASTLINE])
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, path_stem: Path) -> str:
    """Save parquet if possible, else csv. Returns used extension."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    pq = path_stem.with_suffix(".parquet")
    csv = path_stem.with_suffix(".csv")
    try:
        df.to_parquet(pq, index=False)
        return str(pq)
    except Exception as exc:  # noqa: BLE001
        print(f"parquet unavailable ({exc}); writing CSV fallback: {csv}")
        df.to_csv(csv, index=False, encoding="utf-8-sig")
        return str(csv)


def save_coastline(df: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    stem = out_dir / "kocaeli_coastline"
    paths["table"] = save_table(df, stem)
    # geojson-like fallback (simple FeatureCollection of points / line)
    gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[float(r.lon), float(r.lat)] for r in df.itertuples()],
                },
                "properties": {"name": "kocaeli_coastline", "n_points": int(len(df))},
            }
        ],
    }
    gj_path = out_dir / "kocaeli_coastline.geojson"
    gj_path.write_text(json.dumps(gj, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["geojson"] = str(gj_path)
    return paths


def build_seed_cache(out_dir: Path) -> dict[str, Any]:
    """Offline seed without OSM (fallback coastline + empty POI/road frames)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pois = pd.DataFrame(
        columns=[
            "osm_id", "osm_type", "lat", "lon", "category", "subcategory", "name",
            "amenity", "shop", "leisure", "highway", "public_transport", "railway",
        ]
    )
    roads = pd.DataFrame(columns=["osm_id", "highway", "name", "ref", "lat", "lon", "vertex_i", "road_class"])
    coast = pd.DataFrame([{**r, "source": "fallback", "osm_id": None} for r in FALLBACK_COASTLINE])
    return {
        "pois_path": save_table(pois, out_dir / "kocaeli_pois"),
        "roads_path": save_table(roads, out_dir / "kocaeli_roads"),
        "coast_paths": save_coastline(coast, out_dir),
        "n_pois": 0,
        "n_roads": 0,
        "n_coast": len(coast),
        "source": "seed_fallback",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build V17 geo-context offline cache from OSM / fallback.")
    ap.add_argument("--city", default="Kocaeli")
    ap.add_argument("--out", default="data/external/geo_context")
    ap.add_argument("--source", choices=["osm", "seed"], default="osm")
    ap.add_argument("--bbox-south", type=float, default=KOCAELI_BBOX["south"])
    ap.add_argument("--bbox-west", type=float, default=KOCAELI_BBOX["west"])
    ap.add_argument("--bbox-north", type=float, default=KOCAELI_BBOX["north"])
    ap.add_argument("--bbox-east", type=float, default=KOCAELI_BBOX["east"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bbox = {
        "south": float(args.bbox_south),
        "west": float(args.bbox_west),
        "north": float(args.bbox_north),
        "east": float(args.bbox_east),
    }

    print(f"Building geo-context cache → {out_dir}")
    print(f"city={args.city} source={args.source} bbox={bbox}")

    if args.source == "seed":
        built = build_seed_cache(out_dir)
    else:
        try:
            print("Fetching POIs from Overpass...")
            pois = fetch_pois(bbox)
            print(f"  POIs: {len(pois)}")
            print("Fetching roads from Overpass...")
            roads = fetch_roads(bbox)
            print(f"  Road vertices: {len(roads)}")
            print("Fetching coastline from Overpass...")
            coast = fetch_coastline(bbox)
            print(f"  Coast points: {len(coast)}")
            built = {
                "pois_path": save_table(pois, out_dir / "kocaeli_pois"),
                "roads_path": save_table(roads, out_dir / "kocaeli_roads"),
                "coast_paths": save_coastline(coast, out_dir),
                "n_pois": int(len(pois)),
                "n_roads": int(len(roads)),
                "n_coast": int(len(coast)),
                "source": "osm",
            }
        except Exception as exc:  # noqa: BLE001
            print(f"OSM fetch failed ({exc}); writing seed fallback cache.")
            built = build_seed_cache(out_dir)

    anchors_path = out_dir / "kocaeli_anchors.json"
    anchors_payload = {
        "crs": "EPSG:4326",
        "metric_crs": "EPSG:32635",
        "note": "Approx static anchors for Kocaeli / Başiskele — not survey-grade.",
        "anchors": STATIC_ANCHORS,
        "fallback_coastline": FALLBACK_COASTLINE,
    }
    anchors_path.write_text(json.dumps(anchors_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "city": args.city,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": built.get("source"),
        "bbox": bbox,
        "metric_crs": "EPSG:32635",
        "files": {
            "pois": built["pois_path"],
            "roads": built["roads_path"],
            "coastline": built["coast_paths"],
            "anchors": str(anchors_path),
        },
        "counts": {
            "pois": built["n_pois"],
            "road_vertices": built["n_roads"],
            "coast_points": built["n_coast"],
        },
        "feature_groups": [
            "sea_coast",
            "roads",
            "education",
            "health",
            "daily_life",
            "public_transport",
            "composite_scores",
            "basiskele_interactions",
        ],
        "training_note": "Training pipeline must load these files offline; no internet during train.",
    }
    meta_path = out_dir / "geo_context_metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
