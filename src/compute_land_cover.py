"""
Compute NLCD land cover class at each station location.

Queries the USGS NLCD 2021 Land Cover via the MRLC ArcGIS MapServer identify
endpoint, which returns the pixel value (land cover class) at a given lat/lon.
No rasterio or GDAL dependency required.

Each NLCD class is mapped to a simplified category for the ML feature vector:
    urban      - Developed (classes 21-24)
    vegetation - Forest, shrub, grassland, crops, wetlands (41-43, 51-52, 71-74, 81-82, 90, 95)
    water      - Open water (11)
    barren     - Barren land, ice/snow (12, 31)

Usage:
    python src/compute_land_cover.py

S3 layout:
    Input:
        raw/synoptic/metadata/stations.parquet
    Output:
        features/static/land_cover.parquet
"""

import os
import sys
import time
import logging
from io import BytesIO

import pandas as pd
import boto3
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import upload_df_to_s3, S3_BUCKET

# ── Config ────────────────────────────────────────────────────────────────────

# MRLC NLCD 2021 Land Cover MapServer (layer 0 = NLCD_2021_Land_Cover_L48)
NLCD_IDENTIFY_URL = (
    "https://www.mrlc.gov/geoserver/mrlc_display/NLCD_2021_Land_Cover_L48/ows"
)

# NLCD class code to human-readable name
NLCD_CLASSES = {
    11: "Open Water",
    12: "Perennial Ice/Snow",
    21: "Developed, Open Space",
    22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity",
    24: "Developed, High Intensity",
    31: "Barren Land",
    41: "Deciduous Forest",
    42: "Evergreen Forest",
    43: "Mixed Forest",
    51: "Dwarf Scrub",
    52: "Shrub/Scrub",
    71: "Grassland/Herbaceous",
    72: "Sedge/Herbaceous",
    73: "Lichens",
    74: "Moss",
    81: "Pasture/Hay",
    82: "Cultivated Crops",
    90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands",
}

# Simplified category mapping
NLCD_TO_CATEGORY = {}
for c in [11]:
    NLCD_TO_CATEGORY[c] = "water"
for c in [12, 31]:
    NLCD_TO_CATEGORY[c] = "barren"
for c in [21, 22, 23, 24]:
    NLCD_TO_CATEGORY[c] = "urban"
for c in [41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95]:
    NLCD_TO_CATEGORY[c] = "vegetation"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/compute_land_cover.log"),
    ],
)
log = logging.getLogger(__name__)

# ── S3 helpers ────────────────────────────────────────────────────────────────

s3 = boto3.client("s3")


def load_parquet_from_s3(key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))

# ── NLCD query via WMS GetFeatureInfo ─────────────────────────────────────────

def query_nlcd_point(lat: float, lon: float) -> int | None:
    """
    Query NLCD 2021 land cover class at a single lat/lon point via WMS
    GetFeatureInfo. Returns the NLCD class code (int) or None on failure.
    """
    # WMS GetFeatureInfo requires a bounding box around the point and pixel coords.
    # We create a tiny bbox (0.0001 deg) centered on the point and query the center pixel.
    delta = 0.0001
    bbox = f"{lon - delta},{lat - delta},{lon + delta},{lat + delta}"

    params = {
        "service": "WMS",
        "version": "1.1.1",
        "request": "GetFeatureInfo",
        "layers": "NLCD_2021_Land_Cover_L48",
        "query_layers": "NLCD_2021_Land_Cover_L48",
        "info_format": "application/json",
        "srs": "EPSG:4326",
        "width": 3,
        "height": 3,
        "x": 1,
        "y": 1,
        "bbox": bbox,
    }

    max_retries = 5
    base_delay = 2.0

    for attempt in range(max_retries):
        try:
            r = requests.get(NLCD_IDENTIFY_URL, params=params, timeout=30)

            if r.status_code == 200:
                data = r.json()
                features = data.get("features", [])
                if features:
                    props = features[0].get("properties", {})
                    # The pixel value field name varies; try common names
                    for key in ["GRAY_INDEX", "Gray_Index", "PALETTE_INDEX", "value"]:
                        if key in props:
                            return int(props[key])
                    # If none of the expected keys, return first numeric value
                    for v in props.values():
                        try:
                            return int(v)
                        except (ValueError, TypeError):
                            continue
                return None

            elif r.status_code == 429 or r.status_code >= 500:
                delay = base_delay * (2 ** attempt)
                log.warning(f"  NLCD API {r.status_code}, retry {attempt+1}/{max_retries} "
                            f"in {delay:.0f}s...")
                time.sleep(delay)
            else:
                log.warning(f"  NLCD API error {r.status_code}: {r.text[:200]}")
                return None

        except requests.exceptions.RequestException as e:
            delay = base_delay * (2 ** attempt)
            log.warning(f"  Request error: {e}, retry {attempt+1}/{max_retries} in {delay:.0f}s...")
            time.sleep(delay)

    log.warning(f"  NLCD query failed after {max_retries} retries for ({lat}, {lon})")
    return None

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("Loading station metadata from S3...")
    stations = load_parquet_from_s3("raw/synoptic/metadata/stations.parquet")
    log.info(f"  {len(stations)} stations")

    lats = stations["lat"].tolist()
    lons = stations["lon"].tolist()
    stids = stations["stid"].tolist()

    nlcd_codes = []
    nlcd_names = []
    categories = []

    log.info("Querying NLCD 2021 land cover at each station...")
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        code = query_nlcd_point(lat, lon)

        if code is not None:
            nlcd_codes.append(code)
            nlcd_names.append(NLCD_CLASSES.get(code, f"Unknown ({code})"))
            categories.append(NLCD_TO_CATEGORY.get(code, "unknown"))
        else:
            nlcd_codes.append(None)
            nlcd_names.append(None)
            categories.append(None)

        if (i + 1) % 50 == 0:
            valid = sum(1 for c in nlcd_codes if c is not None)
            log.info(f"  Queried {i+1}/{len(lats)} stations ({valid} valid)")

        time.sleep(0.5)  # polite delay

    valid_count = sum(1 for c in nlcd_codes if c is not None)
    log.info(f"  Completed: {valid_count}/{len(lats)} stations with valid NLCD data")

    result = pd.DataFrame({
        "stid": stids,
        "nlcd_code": nlcd_codes,
        "nlcd_name": nlcd_names,
        "land_cover_category": categories,
    })

    # Summary
    log.info("\nLand cover distribution:")
    for cat, count in result["land_cover_category"].value_counts().items():
        log.info(f"  {cat}: {count}")

    out_key = "features/static/land_cover.parquet"
    upload_df_to_s3(result, out_key, log)
    log.info("\nLand cover classification complete.")


if __name__ == "__main__":
    main()
