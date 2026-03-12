"""
Interpolate ERA5 reanalysis grid data to Synoptic station locations.

Reads ERA5 parquet files from S3 (42 grid points at 0.25 degree resolution)
and bilinearly interpolates all 9 ERA5 variables to each Synoptic station
lat/lon for every hourly timestep. Output is one parquet per month in S3.

This script can be re-run for Stage 3 with Open-Meteo grid points as targets
by changing the station metadata source.

Usage:
    python src/interpolate_era5.py

S3 layout:
    Input:
        raw/era5/metadata/grid_points.parquet
        raw/era5/monthly/YYYY-MM/grid_{lat}_{lon}.parquet
        raw/synoptic/metadata/stations.parquet
    Output:
        features/era5_at_stations/YYYY-MM.parquet
"""

import os
import sys
import logging
from io import BytesIO

import numpy as np
import pandas as pd
import boto3
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import upload_df_to_s3, s3_key_exists, S3_BUCKET

# ── Config ────────────────────────────────────────────────────────────────────

S3_PREFIX = "features/era5_at_stations"

# Note: temp_850hpa_c is requested in download_era5.py but Open-Meteo's archive
# API returns null for pressure-level variables. It is excluded here.
ERA5_VARS = [
    "temp_2m_c",
    "humidity_2m",
    "pressure_hpa",
    "wind_speed_10m_kph",
    "wind_dir_10m_deg",
    "precip_mm",
    "cloud_cover_pct",
    "boundary_layer_height_m",
]

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/interpolate_era5.log"),
    ],
)
log = logging.getLogger(__name__)

# ── S3 helpers ────────────────────────────────────────────────────────────────

s3 = boto3.client("s3")


def load_parquet_from_s3(key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))


def list_s3_keys(prefix: str) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys

# ── Interpolation ─────────────────────────────────────────────────────────────

def interpolate_era5_month(era5_df: pd.DataFrame,
                           station_lats: np.ndarray,
                           station_lons: np.ndarray,
                           station_stids: list[str]) -> pd.DataFrame:
    """
    Bilinearly interpolate all ERA5 variables from the grid to station locations
    for every timestep in the month.

    era5_df: concatenated ERA5 data for one month (all 42 grid points).
    Returns a DataFrame with one row per (station, timestep).
    """
    # Build the regular grid axes
    grid_lats = np.sort(era5_df["grid_lat"].unique())
    grid_lons = np.sort(era5_df["grid_lon"].unique())
    timestamps = era5_df["datetime"].unique()
    timestamps = np.sort(timestamps)

    n_stations = len(station_stids)

    # Clamp station coordinates to ERA5 grid bounds so we get nearest-edge
    # values instead of wild extrapolation for stations outside the grid.
    clamped_lats = np.clip(station_lats, grid_lats.min(), grid_lats.max())
    clamped_lons = np.clip(station_lons, grid_lons.min(), grid_lons.max())
    station_points = np.column_stack([clamped_lats, clamped_lons])

    all_rows = []

    for i, ts in enumerate(timestamps):
        ts_data = era5_df[era5_df["datetime"] == ts]

        # Pivot to 2D grid for each variable
        row_dict = {"datetime": np.full(n_stations, ts), "stid": station_stids}

        for var in ERA5_VARS:
            # Build 2D array (n_lat, n_lon)
            pivot = ts_data.pivot_table(
                index="grid_lat", columns="grid_lon", values=var, aggfunc="first"
            )
            # Reindex to ensure sorted order
            pivot = pivot.reindex(index=grid_lats, columns=grid_lons)
            values_2d = pivot.values

            # Handle NaN in grid -- fill with grid mean
            if np.any(np.isnan(values_2d)):
                grid_mean = np.nanmean(values_2d)
                if np.isnan(grid_mean):
                    grid_mean = 0.0
                values_2d = np.where(np.isnan(values_2d), grid_mean, values_2d)

            interp = RegularGridInterpolator(
                (grid_lats, grid_lons), values_2d,
                method="linear", bounds_error=False, fill_value=None
            )
            row_dict[var] = interp(station_points)

        all_rows.append(pd.DataFrame(row_dict))

        if (i + 1) % 168 == 0:  # log weekly
            log.info(f"    Interpolated {i+1}/{len(timestamps)} timesteps")

    result = pd.concat(all_rows, ignore_index=True)
    result["datetime"] = pd.to_datetime(result["datetime"], utc=True)
    return result

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("Loading station metadata...")
    stations = load_parquet_from_s3("raw/synoptic/metadata/stations.parquet")
    station_lats = stations["lat"].values.astype(float)
    station_lons = stations["lon"].values.astype(float)
    station_stids = stations["stid"].tolist()
    log.info(f"  {len(stations)} stations")

    log.info("Loading ERA5 grid metadata...")
    grid_meta = load_parquet_from_s3("raw/era5/metadata/grid_points.parquet")
    log.info(f"  {len(grid_meta)} ERA5 grid points")

    # Discover available months from ERA5 data
    all_era5_keys = list_s3_keys("raw/era5/monthly/")
    months = sorted(set(k.split("/")[3] for k in all_era5_keys if len(k.split("/")) > 3))
    log.info(f"  {len(months)} months available: {months[0]} to {months[-1]}")

    for month_str in months:
        out_key = f"{S3_PREFIX}/{month_str}.parquet"

        if s3_key_exists(out_key):
            log.info(f"  {month_str}: already exists, skipping")
            continue

        log.info(f"\n  Processing {month_str}...")

        # Load all ERA5 grid files for this month
        month_keys = [k for k in all_era5_keys if f"/monthly/{month_str}/" in k]
        frames = []
        for k in month_keys:
            frames.append(load_parquet_from_s3(k))

        if not frames:
            log.warning(f"  No ERA5 data for {month_str}")
            continue

        era5_month = pd.concat(frames, ignore_index=True)
        log.info(f"    Loaded {len(era5_month):,} ERA5 rows from {len(month_keys)} grid files")

        # Interpolate to station locations
        result = interpolate_era5_month(era5_month, station_lats, station_lons, station_stids)
        log.info(f"    Interpolated: {len(result):,} rows ({len(station_stids)} stations x "
                 f"{len(result) // max(len(station_stids), 1)} timesteps)")

        upload_df_to_s3(result, out_key, log)

    log.info("\nERA5 interpolation complete.")


if __name__ == "__main__":
    main()
