"""
Assemble the unified Stage 1 training dataset from all feature sources.

Joins hourly Synoptic observations (targets) with ERA5 interpolated features,
static terrain features, land cover, and coastal distance. Computes lag features
and neighbor-station aggregates.

The output is one large parquet per month at features/training/YYYY-MM.parquet,
ready for GBT model training.

Usage:
    python src/build_training_set.py

S3 layout:
    Input:
        raw/synoptic/monthly/YYYY-MM/chunk_*.parquet
        features/era5_at_stations/YYYY-MM.parquet
        features/static/terrain_features.parquet
        features/static/land_cover.parquet
        features/static/stations_with_features.parquet
    Output:
        features/training/YYYY-MM.parquet
"""

import os
import sys
import logging
from io import BytesIO

import numpy as np
import pandas as pd
import boto3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import upload_df_to_s3, s3_key_exists, S3_BUCKET

# ── Config ────────────────────────────────────────────────────────────────────

S3_PREFIX = "features/training_v2"

# Target variables from Synoptic observations
TARGET_VARS = ["temp_c", "humidity", "wind_speed_kph", "wind_dir_deg", "precip_mm"]

# ERA5 feature columns (exclude datetime and stid which are join keys)
ERA5_FEATURES = [
    "temp_2m_c", "humidity_2m", "pressure_hpa", "wind_speed_10m_kph",
    "wind_dir_10m_deg", "precip_mm", "cloud_cover_pct", "boundary_layer_height_m",
]

# Lag intervals for observation features (hours)
LAG_HOURS = [1, 3, 6]

# Number of nearest neighbors for spatial aggregation
N_NEIGHBORS = 5

# Stations excluded due to confirmed sensor faults (physically impossible values
# detected in training data: temp outside -20..50°C, humidity outside 0..100%,
# or within-station std > 15°C).
BAD_STATIONS = {
    '340PG', '355PG', '496PG', '583PG', 'AN844', 'BINC1', 'CQ127', 'D0946',
    'E4971', 'F2503', 'GGBC1', 'JEPC1', 'KRHV', 'KSQL', 'LBNL1', 'MCKCA',
    'OAMC1', 'PG230', 'PG510', 'PRWC1', 'PTRCA', 'SFOC1', 'SFXC1', 'SJS01',
    'SJS61', 'UCYL', 'UP594', 'UP598', 'UP621', 'UP641', 'UP657', 'UP667',
    'UP680', 'UR476', 'UR481', 'UR604',
    # KRHV: mixed Celsius/Fahrenheit in same record (temp max 97°C, min 0°C)
    # KSQL: temp max 78°C + humidity max 174% -- two independent failure modes
}
# Salvaged ASOS stations (removed from blocklist 2026-03-12):
# KC83, KDVO, KHAF, KPAO, KSNS -- valid temperature ranges, humidity clipped to 100%

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/build_training_set.log"),
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

# ── Neighbor station features ────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def build_neighbor_map(stations: pd.DataFrame, n: int) -> dict[str, list[str]]:
    """
    For each station, find the N nearest neighbors by haversine distance.
    Returns {stid: [neighbor_stid_1, ..., neighbor_stid_n]}.
    """
    lats = stations["lat"].values
    lons = stations["lon"].values
    stids = stations["stid"].values

    neighbor_map = {}
    for i in range(len(stids)):
        dists = haversine_km(lats[i], lons[i], lats, lons)
        # Exclude self (distance 0), sort by distance
        idx = np.argsort(dists)
        neighbors = [stids[j] for j in idx[1:n + 1]]
        neighbor_map[stids[i]] = neighbors

    return neighbor_map

# ── Feature engineering ──────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features (hour of day, day of year)."""
    dt = df["datetime"]
    hour = dt.dt.hour + dt.dt.minute / 60.0
    doy = dt.dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged observation values for each target variable.
    Operates per-station after sorting by time.
    """
    df = df.sort_values(["stid", "datetime"])

    for var in TARGET_VARS:
        for lag_h in LAG_HOURS:
            col_name = f"{var}_lag{lag_h}h"
            # Shift by lag_h rows (each row is 1 hour after rounding)
            df[col_name] = df.groupby("stid")[var].shift(lag_h)

    return df


def add_neighbor_features(df: pd.DataFrame, neighbor_map: dict) -> pd.DataFrame:
    """
    Add mean of nearest-neighbor station values for temp and humidity.
    These are the two variables with strongest spatial correlation.
    """
    neighbor_vars = ["temp_c", "humidity"]

    # Build a lookup: (stid, datetime) -> {var: value}
    lookup = df.set_index(["stid", "datetime"])[neighbor_vars].to_dict("index")

    for var in neighbor_vars:
        col_name = f"neighbor_mean_{var}"
        values = []
        for _, row in df.iterrows():
            neighbors = neighbor_map.get(row["stid"], [])
            neighbor_vals = []
            for n_stid in neighbors:
                key = (n_stid, row["datetime"])
                if key in lookup:
                    v = lookup[key].get(var)
                    if v is not None and not np.isnan(v):
                        neighbor_vals.append(v)
            if neighbor_vals:
                values.append(np.mean(neighbor_vals))
            else:
                values.append(np.nan)
        df[col_name] = values

    return df

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load static features
    log.info("Loading static features...")
    terrain = load_parquet_from_s3("features/static/terrain_features.parquet")
    land_cover = load_parquet_from_s3("features/static/land_cover.parquet")
    station_features = load_parquet_from_s3("features/static/stations_with_features.parquet")
    log.info(f"  {len(terrain)} terrain, {len(land_cover)} land cover, "
             f"{len(station_features)} station features")

    # Merge static features into one table keyed by stid
    static = station_features[["stid", "lat", "lon", "elev_m",
                                "dist_coast_km", "dist_bay_km", "coastal_exposure"]]
    static = static.merge(terrain[["stid", "dem_elev_m", "slope_deg", "aspect_deg",
                                    "station_density_2km", "is_urban_proxy"]],
                          on="stid", how="left")
    static = static.merge(land_cover[["stid", "nlcd_code", "land_cover_category"]],
                          on="stid", how="left")

    # One-hot encode land cover category
    lc_dummies = pd.get_dummies(static["land_cover_category"], prefix="lc")
    static = pd.concat([static, lc_dummies], axis=1)
    static = static.drop(columns=["land_cover_category"])

    log.info(f"  Static feature columns: {static.columns.tolist()}")

    # Build neighbor map
    log.info("Building neighbor map...")
    neighbor_map = build_neighbor_map(station_features, N_NEIGHBORS)
    log.info(f"  {len(neighbor_map)} stations with {N_NEIGHBORS} neighbors each")

    # Discover months
    all_synoptic_keys = list_s3_keys("raw/synoptic/monthly/")
    months = sorted(set(k.split("/")[3] for k in all_synoptic_keys
                        if len(k.split("/")) > 3))
    log.info(f"  {len(months)} months: {months[0]} to {months[-1]}")

    for month_str in months:
        out_key = f"{S3_PREFIX}/{month_str}.parquet"

        if s3_key_exists(out_key):
            log.info(f"  {month_str}: already exists, skipping")
            continue

        log.info(f"\n  Processing {month_str}...")

        # Load Synoptic observations for this month
        syn_keys = [k for k in all_synoptic_keys if f"/monthly/{month_str}/" in k]
        syn_frames = [load_parquet_from_s3(k) for k in syn_keys]
        if not syn_frames:
            log.warning(f"  No Synoptic data for {month_str}")
            continue
        synoptic = pd.concat(syn_frames, ignore_index=True)
        # Remove known bad-sensor stations
        synoptic = synoptic[~synoptic["stid"].isin(BAD_STATIONS)]
        # Clip humidity to [0, 100]: ASOS/airport stations occasionally report
        # values of 101-115% due to hygrometer over-reads near saturation.
        # These are valid sensors with rare artifacts, not unit errors.
        if "humidity" in synoptic.columns:
            synoptic["humidity"] = synoptic["humidity"].clip(lower=0, upper=100)
        log.info(f"    Synoptic: {len(synoptic):,} rows from {len(syn_keys)} chunks "
                 f"({len(BAD_STATIONS)} bad stations excluded)")

        # Round Synoptic timestamps to nearest hour for ERA5 join
        synoptic["datetime"] = pd.to_datetime(synoptic["datetime"], utc=True)
        synoptic["datetime_hour"] = synoptic["datetime"].dt.round("h")

        # Aggregate sub-hourly obs to hourly means per station
        hourly_obs = (synoptic
                      .groupby(["stid", "datetime_hour"])[TARGET_VARS]
                      .mean()
                      .reset_index()
                      .rename(columns={"datetime_hour": "datetime"}))
        log.info(f"    Hourly obs: {len(hourly_obs):,} rows")

        # Load ERA5 interpolated features
        era5_key = f"features/era5_at_stations/{month_str}.parquet"
        if not s3_key_exists(era5_key):
            log.warning(f"  No ERA5 data for {month_str}, skipping")
            continue
        era5 = load_parquet_from_s3(era5_key)
        era5["datetime"] = pd.to_datetime(era5["datetime"], utc=True)

        # Rename ERA5 precip_mm to avoid collision with Synoptic precip_mm
        era5 = era5.rename(columns={"precip_mm": "era5_precip_mm"})
        log.info(f"    ERA5: {len(era5):,} rows")

        # Join observations with ERA5 on (stid, datetime)
        merged = hourly_obs.merge(era5, on=["stid", "datetime"], how="inner")
        log.info(f"    After ERA5 join: {len(merged):,} rows")

        if len(merged) == 0:
            log.warning(f"  No matching rows after join for {month_str}")
            continue

        # Add time features
        merged = add_time_features(merged)

        # Add lag features
        merged = add_lag_features(merged)
        log.info(f"    Added lag features")

        # Add neighbor features (expensive -- use vectorized approach for speed)
        log.info(f"    Computing neighbor features...")
        # Vectorized neighbor approach: pivot then lookup
        for var in ["temp_c", "humidity"]:
            pivot = merged.pivot_table(index="datetime", columns="stid",
                                       values=var, aggfunc="first")
            neighbor_means = {}
            for stid, neighbors in neighbor_map.items():
                valid_neighbors = [n for n in neighbors if n in pivot.columns]
                if valid_neighbors:
                    neighbor_means[stid] = pivot[valid_neighbors].mean(axis=1)
                else:
                    neighbor_means[stid] = pd.Series(np.nan, index=pivot.index)

            neighbor_df = pd.DataFrame(neighbor_means)
            neighbor_melted = (neighbor_df.stack()
                               .reset_index()
                               .rename(columns={"level_0": "datetime",
                                                "level_1": "stid",
                                                0: f"neighbor_mean_{var}"}))
            merged = merged.merge(neighbor_melted, on=["stid", "datetime"], how="left")

        log.info(f"    Added neighbor features")

        # Join static features
        merged = merged.merge(static, on="stid", how="left", suffixes=("", "_static"))
        log.info(f"    After static join: {len(merged):,} rows, {len(merged.columns)} columns")

        # Drop redundant columns
        drop_cols = [c for c in merged.columns if c.endswith("_static")]
        if drop_cols:
            merged = merged.drop(columns=drop_cols)

        # Derived inversion features (require both ERA5 BLH and static elevation)
        # elev_above_blh_m: positive = station above marine layer inversion,
        #                   negative = station within marine layer
        if "boundary_layer_height_m" in merged.columns and "elev_m" in merged.columns:
            merged["elev_above_blh_m"] = merged["elev_m"] - merged["boundary_layer_height_m"]

            # Northness: cos(aspect) maps 0°(N)→1, 90°(E)→0, 180°(S)→-1, 270°(W)→0
            if "aspect_deg" in merged.columns:
                merged["aspect_northness"] = np.cos(np.radians(merged["aspect_deg"]))
                # Interaction: north-facing exposure conditional on being near the inversion.
                # Positive when north-facing and below inversion (fog retention),
                # negative when north-facing and above inversion.
                merged["northness_x_blh_deficit"] = (
                    merged["aspect_northness"] * (merged["boundary_layer_height_m"] - merged["elev_m"])
                )

        # Upload
        upload_df_to_s3(merged, out_key, log)

    log.info("\nTraining set assembly complete.")


if __name__ == "__main__":
    main()
