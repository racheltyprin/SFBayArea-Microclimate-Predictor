"""
Compute DEM-derived terrain features (elevation, slope, aspect) for each station.

Uses the Open-Meteo Elevation API to query SRTM-derived elevation at each station
location plus 4 cardinal neighbors, then computes slope and aspect via finite
differences. No rasterio/GDAL dependency required.

Also computes a station-density proxy for land cover (urban vs. rural) since
NLCD requires rasterio. This proxy will be replaced with real NLCD data when
rasterio is available.

Usage:
    python src/compute_terrain_features.py

S3 layout:
    Input:
        raw/synoptic/metadata/stations.parquet
    Output:
        features/static/terrain_features.parquet
"""

import os
import sys
import math
import time
import logging
from io import BytesIO

import pandas as pd
import boto3
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import upload_df_to_s3, S3_BUCKET

# ── Config ────────────────────────────────────────────────────────────────────

# Open-Meteo Elevation API accepts up to 100 coordinate pairs per request
ELEVATION_API = "https://api.open-meteo.com/v1/elevation"
BATCH_SIZE = 100

# Finite difference step for slope/aspect: ~30m at Bay Area latitudes
# 0.0003 degrees * 111320 m/degree * cos(37.5) = ~26.5m
NEIGHBOR_OFFSET_DEG = 0.0003

# Station density radius for urban proxy (km)
DENSITY_RADIUS_KM = 2.0

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/compute_terrain_features.log"),
    ],
)
log = logging.getLogger(__name__)

# ── S3 helpers ────────────────────────────────────────────────────────────────

s3 = boto3.client("s3")


def load_parquet_from_s3(key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))

# ── Elevation API ─────────────────────────────────────────────────────────────

def fetch_elevations(lats: list[float], lons: list[float]) -> list[float]:
    """
    Query Open-Meteo Elevation API for SRTM-derived elevation at given points.
    Batches into groups of BATCH_SIZE to respect API limits.
    Retries with exponential backoff on rate limiting (429) or server errors.
    Returns list of elevations in meters.
    """
    all_elevations = []

    for i in range(0, len(lats), BATCH_SIZE):
        batch_lats = lats[i:i + BATCH_SIZE]
        batch_lons = lons[i:i + BATCH_SIZE]

        max_retries = 5
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                r = requests.get(ELEVATION_API, params={
                    "latitude": ",".join(f"{lat:.6f}" for lat in batch_lats),
                    "longitude": ",".join(f"{lon:.6f}" for lon in batch_lons),
                }, timeout=30)

                if r.status_code == 200:
                    data = r.json()
                    elevations = data.get("elevation", [])
                    all_elevations.extend(elevations)
                    break
                elif r.status_code == 429 or r.status_code >= 500:
                    delay = base_delay * (2 ** attempt)
                    log.warning(f"  Elevation API {r.status_code}, retry {attempt+1}/{max_retries} "
                                f"in {delay:.0f}s...")
                    time.sleep(delay)
                else:
                    log.warning(f"  Elevation API error {r.status_code}: {r.text[:200]}")
                    all_elevations.extend([float("nan")] * len(batch_lats))
                    break
            except requests.exceptions.RequestException as e:
                delay = base_delay * (2 ** attempt)
                log.warning(f"  Request error: {e}, retry {attempt+1}/{max_retries} in {delay:.0f}s...")
                time.sleep(delay)
        else:
            log.warning(f"  Batch {i//BATCH_SIZE} failed after {max_retries} retries")
            all_elevations.extend([float("nan")] * len(batch_lats))

        time.sleep(1.0)  # polite delay between successful requests

    return all_elevations

# ── Slope and aspect computation ──────────────────────────────────────────────

def compute_slope_aspect(center_lats: list[float], center_lons: list[float]) -> tuple[list[float], list[float]]:
    """
    Compute slope (degrees) and aspect (degrees, 0=N clockwise) for each station
    by querying elevation at 4 cardinal neighbors and using finite differences.
    """
    n = len(center_lats)
    d = NEIGHBOR_OFFSET_DEG

    # Build arrays for all 4 neighbor directions: N, S, E, W
    all_lats = []
    all_lons = []
    for i in range(n):
        lat, lon = center_lats[i], center_lons[i]
        all_lats.extend([lat + d, lat - d, lat, lat])       # N, S, E, W
        all_lons.extend([lon, lon, lon + d, lon - d])        # N, S, E, W

    log.info(f"  Querying {len(all_lats)} neighbor elevations for slope/aspect...")
    neighbor_elevs = fetch_elevations(all_lats, all_lons)

    slopes = []
    aspects = []

    for i in range(n):
        elev_n = neighbor_elevs[i * 4]
        elev_s = neighbor_elevs[i * 4 + 1]
        elev_e = neighbor_elevs[i * 4 + 2]
        elev_w = neighbor_elevs[i * 4 + 3]

        lat = center_lats[i]

        # Convert degree offset to meters
        dy = d * 110540.0  # meters per degree latitude
        dx = d * 111320.0 * math.cos(math.radians(lat))  # meters per degree longitude

        dz_dy = (elev_n - elev_s) / (2 * dy)  # north-south gradient
        dz_dx = (elev_e - elev_w) / (2 * dx)  # east-west gradient

        slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
        slopes.append(round(math.degrees(slope_rad), 2))

        # Aspect: compass bearing (0=N, 90=E, 180=S, 270=W)
        aspect_rad = math.atan2(-dz_dx, dz_dy)
        aspect_deg = math.degrees(aspect_rad) % 360
        aspects.append(round(aspect_deg, 1))

    return slopes, aspects

# ── Station density proxy ─────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def compute_station_density(lats: list[float], lons: list[float],
                            radius_km: float) -> list[int]:
    """Count how many other stations are within radius_km of each station."""
    n = len(lats)
    densities = []
    for i in range(n):
        count = 0
        for j in range(n):
            if i != j and haversine_km(lats[i], lons[i], lats[j], lons[j]) <= radius_km:
                count += 1
        densities.append(count)
    return densities

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("Loading station metadata from S3...")
    stations = load_parquet_from_s3("raw/synoptic/metadata/stations.parquet")
    log.info(f"  {len(stations)} stations")

    lats = stations["lat"].tolist()
    lons = stations["lon"].tolist()
    stids = stations["stid"].tolist()

    # Step 1: DEM elevation at station locations
    log.info("Fetching DEM elevations from Open-Meteo...")
    dem_elevs = fetch_elevations(lats, lons)
    log.info(f"  Got {sum(1 for e in dem_elevs if not math.isnan(e))} valid elevations")

    # Step 2: Slope and aspect from cardinal neighbors
    log.info("Computing slope and aspect...")
    slopes, aspects = compute_slope_aspect(lats, lons)

    # Step 3: Station density as urban proxy
    log.info(f"Computing station density within {DENSITY_RADIUS_KM}km...")
    densities = compute_station_density(lats, lons, DENSITY_RADIUS_KM)
    log.info(f"  Max density: {max(densities)}, mean: {sum(densities)/len(densities):.1f}")

    # Build output DataFrame
    result = pd.DataFrame({
        "stid": stids,
        "dem_elev_m": [round(e, 1) for e in dem_elevs],
        "slope_deg": slopes,
        "aspect_deg": aspects,
        "station_density_2km": densities,
        "is_urban_proxy": [1 if d >= 5 else 0 for d in densities],
    })

    log.info(f"\nTerrain features summary:")
    log.info(f"  Elevation range: {result['dem_elev_m'].min():.0f}m to {result['dem_elev_m'].max():.0f}m")
    log.info(f"  Slope range: {result['slope_deg'].min():.1f} to {result['slope_deg'].max():.1f} degrees")
    log.info(f"  Urban proxy stations: {result['is_urban_proxy'].sum()}/{len(result)}")

    out_key = "features/static/terrain_features.parquet"
    upload_df_to_s3(result, out_key, log)
    log.info("\nTerrain features complete.")


if __name__ == "__main__":
    main()
