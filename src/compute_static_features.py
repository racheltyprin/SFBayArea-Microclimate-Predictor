"""
Compute static geographic features for each weather station.
Reads station metadata from S3, enriches with static features, and writes
the result back to S3 for use in ML training.

Static features capture the physical geography that explains why two stations
at the same elevation and ERA5 state can have different temperatures — proximity
to the ocean, bay exposure, terrain shielding, etc.

Usage:
    python src/compute_static_features.py

Dependencies (all standard):
    pip install pandas boto3 pyarrow

CURRENT DATA SOURCE COVERAGE — SYNOPTIC ONLY:
    This script currently processes station metadata from Synoptic only
    (s3://bay-area-microclimate/raw/synoptic/metadata/stations.parquet).

    TODO (Open-Meteo): When Open-Meteo dense grid data is available, also load:
        s3://bay-area-microclimate/raw/open_meteo/metadata/grid_points.parquet
    Then concatenate both station/grid-point sets, compute features for all,
    and save to the same output path. The feature computation functions below
    are source-agnostic and require no changes.

S3 layout:
    Input:
        raw/synoptic/metadata/stations.parquet
        [TODO: raw/open_meteo/metadata/grid_points.parquet]
    Output:
        features/static/stations_with_features.parquet
"""

import os
import sys
import math
import logging
from io import BytesIO

import pandas as pd
import boto3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import upload_df_to_s3, S3_BUCKET

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ── Geography reference points ────────────────────────────────────────────────

# Reference waypoints along the Pacific coast of the Bay Area.
# Used to compute each station's distance to the open ocean, which is the
# dominant driver of the marine layer / coastal cooling effect.
PACIFIC_COAST_WAYPOINTS = [
    (38.27, -122.97),  # Bodega Bay
    (38.00, -122.98),  # Pt. Reyes
    (37.90, -122.75),  # Muir Beach area
    (37.75, -122.51),  # Ocean Beach / Golden Gate
    (37.60, -122.50),  # Pacifica
    (37.44, -122.49),  # Half Moon Bay
    (37.20, -122.41),  # Pescadero
    (37.00, -122.17),  # Santa Cruz coast
]

# Reference waypoints along the San Francisco Bay shoreline.
# Bay proximity matters for afternoon sea breeze and morning fog burn-off timing.
BAY_WAYPOINTS = [
    (37.92, -122.42),  # San Pablo Bay north
    (37.87, -122.35),  # Richmond / San Rafael area
    (37.80, -122.38),  # Berkeley / Oakland waterfront
    (37.73, -122.22),  # Alameda / San Leandro Bay
    (37.66, -122.13),  # Hayward / San Mateo Bridge
    (37.56, -122.06),  # Fremont / Dumbarton
    (37.47, -122.10),  # Palo Alto / Menlo Park
    (37.80, -122.45),  # SF Embarcadero
    (37.70, -122.40),  # SFO area
    (38.05, -122.27),  # Napa / Vallejo area (top of bay)
]

# ── Distance computation ──────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def min_dist_to_waypoints(lat: float, lon: float,
                           waypoints: list[tuple[float, float]]) -> float:
    """Minimum haversine distance (km) from a point to a list of waypoints."""
    return min(haversine_km(lat, lon, wlat, wlon) for wlat, wlon in waypoints)


# ── Static feature computation ────────────────────────────────────────────────

def compute_features(stations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add static geographic features to a stations DataFrame.

    Input columns required: stid, lat, lon, elev_m
    Added columns:
        dist_coast_km   - distance to Pacific coast (key marine layer driver)
        dist_bay_km     - distance to SF Bay shoreline (sea breeze timing)
        coastal_exposure - composite score: low = sheltered inland, high = coastal
                           (derived from dist_coast_km + elev_m)

    TODO (elevation accuracy): Synoptic elevation values come from station
    metadata and can be inaccurate for non-official stations. Once WU data is
    integrated, cross-check both sources against SRTM 30m DEM for ground truth:
        pip install elevation rasterio
        import elevation; elevation.clip(bounds=(...), output='dem.tif')
    This is especially important for WU backyard stations where reported
    elevation may be missing or wrong.

    TODO (terrain aspect): Add slope/aspect from SRTM to capture cold-air
    pooling in valleys and shadow effects. Requires rasterio + SRTM tiles.
    """
    df = stations_df.copy()

    log.info("Computing distance to Pacific coast...")
    df["dist_coast_km"] = df.apply(
        lambda r: min_dist_to_waypoints(r["lat"], r["lon"], PACIFIC_COAST_WAYPOINTS),
        axis=1,
    ).round(2)

    log.info("Computing distance to SF Bay...")
    df["dist_bay_km"] = df.apply(
        lambda r: min_dist_to_waypoints(r["lat"], r["lon"], BAY_WAYPOINTS),
        axis=1,
    ).round(2)

    # Coastal exposure: stations close to coast + low elevation get high scores.
    # Simple heuristic — will be replaced by a learned feature once the model
    # has enough data to infer it empirically.
    # Scale: dist_coast normalized over ~60km range, elev over ~1000m range.
    df["coastal_exposure"] = (
        1.0 - (df["dist_coast_km"].clip(0, 60) / 60)
        - (df["elev_m"].clip(0, 1000) / 1000) * 0.3
    ).clip(0, 1).round(3)

    return df


# ── S3 I/O ────────────────────────────────────────────────────────────────────

def load_parquet_from_s3(key: str) -> pd.DataFrame:
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load station metadata ─────────────────────────────────────────────────

    # SYNOPTIC ONLY — currently the sole source of station metadata.
    log.info("Loading Synoptic station metadata from S3...")
    synoptic_key = "raw/synoptic/metadata/stations.parquet"
    synoptic_stations = load_parquet_from_s3(synoptic_key)
    log.info(f"  Synoptic: {len(synoptic_stations)} stations")

    # TODO (Open-Meteo): Uncomment when Open-Meteo dense grid data is available.
    # om_key = "raw/open_meteo/metadata/grid_points.parquet"
    # om_stations = load_parquet_from_s3(om_key)
    # log.info(f"  Open-Meteo: {len(om_stations)} grid points")
    #
    # Combine both sources. Open-Meteo grid points provide dense spatial coverage in
    # residential areas, dramatically improving zone boundary resolution.
    # stations = pd.concat([synoptic_stations, wu_stations], ignore_index=True)
    # stations = stations.drop_duplicates(subset=["stid"])
    #
    # For now, only Synoptic:
    stations = synoptic_stations

    log.info(f"Total stations to process: {len(stations)}")

    # ── Compute features ──────────────────────────────────────────────────────
    log.info("Computing static geographic features...")
    stations_enriched = compute_features(stations)

    cols_added = ["dist_coast_km", "dist_bay_km", "coastal_exposure"]
    log.info(f"Features added: {cols_added}")
    log.info(f"\n{stations_enriched[['stid', 'lat', 'lon', 'elev_m'] + cols_added].head(10)}")

    # ── Upload to S3 ──────────────────────────────────────────────────────────
    out_key = "features/static/stations_with_features.parquet"
    upload_df_to_s3(stations_enriched, out_key, log)
    log.info("\nStatic features complete.")
    log.info("   NOTE: Currently Synoptic stations only. Re-run after WU data is available.")


if __name__ == "__main__":
    main()
