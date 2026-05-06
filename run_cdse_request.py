#!/usr/bin/env python3
import os, json, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv(Path(__file__).parent / '.env.local')

import numpy as np
import rasterio
from shapely.geometry import shape, mapping, Polygon
from shapely.ops import transform as shp_transform
import pyproj
from pyproj import Transformer

from sentinelhub import SHConfig, DataCollection, CRS, BBox, MimeType, SentinelHubRequest, MosaickingOrder

# Retrieve CDSE credentials from environment (set by supabase start or user)
CDSE_ID = os.getenv('CDSE_CLIENT_ID')
CDSE_SECRET = os.getenv('CDSE_CLIENT_SECRET')
if not (CDSE_ID and CDSE_SECRET):
    raise RuntimeError('CDSE credentials not set in environment')

# Configure Sentinel Hub
config = SHConfig()
config.sh_client_id = CDSE_ID
config.sh_client_secret = CDSE_SECRET
config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
config.sh_base_url = "https://sh.dataspace.copernicus.eu"
# Do not write to disk

print('Endpoint:', config.sh_base_url)
print('Token URL:', config.sh_token_url)

# Example AOI (Bangkok industrial area)
EXAMPLE_AOI = {
    "type": "Feature",
    "properties": {"name": "Bangkok-LatKrabang-test"},
    "geometry": {
        "type": "Polygon",
        "coordinates": [[
            [100.7748, 13.7220],
            [100.7848, 13.7220],
            [100.7848, 13.7320],
            [100.7748, 13.7320],
            [100.7748, 13.7220]
        ]]
    }
}

aoi_geom = shape(EXAMPLE_AOI["geometry"])

def utm_epsg_for_geom(geom):
    cx, cy = geom.centroid.x, geom.centroid.y
    zone = int((cx + 180) / 6) + 1
    return 32600 + zone if cy >= 0 else 32700 + zone

epsg_utm = utm_epsg_for_geom(aoi_geom)
print(f"AOI centroid: ({aoi_geom.centroid.x:.4f}, {aoi_geom.centroid.y:.4f})")
print(f"UTM zone EPSG: {epsg_utm}")

# Transform to UTM
to_utm = Transformer.from_crs(4326, epsg_utm, always_xy=True).transform

aoi_utm = shp_transform(to_utm, aoi_geom)
minx, miny, maxx, maxy = aoi_utm.bounds
ground_w_m = maxx - minx
ground_h_m = maxy - miny
print(f"Ground extent: {ground_w_m:.0f} × {ground_h_m:.0f} m")
print(f"S2 native: {ground_w_m/10:.0f} × {ground_h_m/10:.0f} pixels @ 10m")

# Evalscript with SCL in DN units (updated)
EVALSCRIPT_S2_FULL = """
//VERSION=3
function setup() {
  return {
    input: [
      { bands: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"], units: "REFLECTANCE" },
      { bands: ["SCL"], units: "DN" }
    ],
    output: { bands: 11, sampleType: "FLOAT32" },
    mosaicking: "ORBIT"
  };
}

function evaluatePixel(sample) {
  return [sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, sample.B07, sample.B08, sample.B8A, sample.B11, sample.B12, sample.SCL];
}
"""

OUT_W, OUT_H = 384, 384

# Search for recent scene (use fixed date for reproducibility as in notebook)
# Here we pick the most recent scene from catalog within 90 days and <20% cloud.
from sentinelhub import SentinelHubCatalog
catalog = SentinelHubCatalog(config=config)

def search_s2_scenes(aoi_geom, days_back=90, max_cloud=20.0):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    bbox = BBox(bbox=aoi_geom.bounds, crs=CRS.WGS84)
    results = list(catalog.search(
        collection=DataCollection.SENTINEL2_L2A,
        bbox=bbox,
        time=(start, end),
        filter=f"eo:cloud_cover<{max_cloud}",
        fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover", "properties.s2:mgrs_tile", "properties.platform"]}
    ))
    return sorted(results, key=lambda x: x["properties"]["datetime"], reverse=True)

scenes = search_s2_scenes(aoi_geom)
if not scenes:
    raise RuntimeError('No scenes found')
best_scene = scenes[0]
acquisition_date = best_scene["properties"]["datetime"][:10]
time_window = (acquisition_date, acquisition_date)

bbox_utm = BBox(bbox=(minx, miny, maxx, maxy), crs=CRS(epsg_utm))

request = SentinelHubRequest(
    evalscript=EVALSCRIPT_S2_FULL,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=time_window,
            mosaicking_order=MosaickingOrder.LEAST_CC,
            maxcc=0.6,
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
    bbox=bbox_utm,
    size=(OUT_W, OUT_H),
    config=config,
)

print(f"Requesting Sentinel-2 L2A tile…")
print(f"  bbox UTM: {bbox_utm}")
print(f"  size: {OUT_W}×{OUT_H}")
print(f"  date: {acquisition_date}")

t0 = time.time()
data = request.get_data()[0]
print(f"✓ Received in {time.time()-t0:.1f}s")
print(f"  shape: {data.shape} dtype: {data.dtype}")
print(f"  reflectance range B04: {data[...,2].min():.3f} – {data[...,2].max():.3f}")
