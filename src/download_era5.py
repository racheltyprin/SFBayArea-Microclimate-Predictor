"""
ERA5 reanalysis downloader via Open-Meteo Archive API (free, no key required).
Fetches large-scale atmospheric variables at a grid of points covering the Bay Area
and saves hourly timeseries to S3 as parquet.

These grid-point timeseries serve as the "synoptic state" input features for the
microclimate ML model — the large-scale atmospheric context that drives local
temperature differences between microclimate zones.

Usage:
    python src/download_era5.py

Dependencies:
    pip install requests pandas boto3 pyarrow

DATA SOURCE COVERAGE NOTE:
    ERA5 features are purely reanalysis (model output) and are source-agnostic —
    they apply equally to Synoptic stations and Open-Meteo dense grid points
    without modification.

S3 layout:
    s3://bay-area-microclimate/
        raw/era5/
            metadata/grid_points.parquet   ← ERA5 grid points and their lat/lon
            monthly/YYYY-MM/
                grid_{lat}_{lon}.parquet   ← hourly ERA5 for one grid point, one month
"""

import os
import sys
import time
import logging
import datetime as dt
from datetime import datetime, timedelta

from dotenv import load_dotenv
import requests
import pandas as pd

load_dotenv()
from dateutil.relativedelta import relativedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import upload_df_to_s3, s3_key_exists

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL   = "https://archive-api.open-meteo.com/v1/archive"
S3_PREFIX  = "raw/era5"

# How many years of ERA5 to fetch — should match your Synoptic/WU download range.
# ERA5 goes back to 1940 and is free regardless of how far back you go.
YEARS_BACK = float(os.environ.get("YEARS_BACK", 1))

# Bay Area bounding box: lon_min, lat_min, lon_max, lat_max
BBOX = (-123.0, 36.9, -121.5, 38.3)

# ERA5 native resolution is ~0.25°. Sampling at 0.25° gives full coverage
# with no gaps and matches the reanalysis grid exactly.
GRID_STEP_DEG = 0.25

# Hourly ERA5 variables to fetch from Open-Meteo.
# These are selected to capture the large-scale atmospheric drivers of Bay Area
# microclimates: marine layer depth, temperature gradient, wind regime.
ERA5_VARIABLES = [
    "temperature_2m",           # large-scale surface temperature (°C)
    "relative_humidity_2m",     # large-scale surface humidity (%)
    "surface_pressure",         # sea-level-equivalent pressure (hPa)
    "wind_speed_10m",           # large-scale surface wind speed (km/h)
    "wind_direction_10m",       # large-scale wind direction (°)
    "precipitation",            # hourly precip (mm)
    "cloud_cover",              # total cloud cover (%)
    "boundary_layer_height",    # PBL height — key driver of marine layer intrusion (m)
    "temperature_850hPa",       # free-atmosphere temp, proxy for subsidence inversion (°C)
]

SLEEP_BETWEEN_CALLS = 1  # seconds — Open-Meteo is free but be polite

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/download_era5.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Grid generation ───────────────────────────────────────────────────────────

def build_grid() -> list[tuple[float, float]]:
    """
    Generate lat/lon grid points covering the Bay Area bbox at ERA5 resolution.
    Returns list of (lat, lon) tuples rounded to 2 decimal places.
    """
    lon_min, lat_min, lon_max, lat_max = BBOX
    points = []
    lat = lat_min
    while lat <= lat_max + 1e-9:
        lon = lon_min
        while lon <= lon_max + 1e-9:
            points.append((round(lat, 2), round(lon, 2)))
            lon += GRID_STEP_DEG
        lat += GRID_STEP_DEG
    return points


def lat_lon_key(lat: float, lon: float) -> str:
    """Consistent string key for a grid point, used in S3 paths."""
    lat_s = f"{lat:.2f}".replace("-", "neg").replace(".", "p")
    lon_s = f"{lon:.2f}".replace("-", "neg").replace(".", "p")
    return f"{lat_s}_{lon_s}"

# ── Open-Meteo API ────────────────────────────────────────────────────────────

def fetch_era5_month(lat: float, lon: float,
                     month_start: datetime, month_end: datetime) -> pd.DataFrame | None:
    """
    Fetch hourly ERA5 data for one grid point over one month via Open-Meteo.
    Returns a DataFrame or None if the request fails.
    """
    r = requests.get(BASE_URL, params={
        "latitude":           lat,
        "longitude":          lon,
        "start_date":         month_start.strftime("%Y-%m-%d"),
        "end_date":           (month_end - timedelta(days=1)).strftime("%Y-%m-%d"),
        "hourly":             ",".join(ERA5_VARIABLES),
        "timezone":           "UTC",
    }, timeout=60)

    if r.status_code != 200:
        log.warning(f"  Open-Meteo error {r.status_code} for ({lat}, {lon}): {r.text[:200]}")
        return None

    data = r.json()
    hourly = data.get("hourly", {})
    times  = hourly.get("time")
    if not times:
        return None

    df = pd.DataFrame({
        "datetime":              pd.to_datetime(times, utc=True),
        "temp_2m_c":             hourly.get("temperature_2m"),
        "humidity_2m":           hourly.get("relative_humidity_2m"),
        "pressure_hpa":          hourly.get("surface_pressure"),
        "wind_speed_10m_kph":    hourly.get("wind_speed_10m"),
        "wind_dir_10m_deg":      hourly.get("wind_direction_10m"),
        "precip_mm":             hourly.get("precipitation"),
        "cloud_cover_pct":       hourly.get("cloud_cover"),
        "boundary_layer_height_m": hourly.get("boundary_layer_height"),
        "temp_850hpa_c":         hourly.get("temperature_850hPa"),
        "grid_lat":              lat,
        "grid_lon":              lon,
    })
    return df

# ── Month range helper ────────────────────────────────────────────────────────

def month_range(start: datetime, end: datetime):
    current = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while current < end:
        next_month = current + relativedelta(months=1)
        yield current, min(next_month, end)
        current = next_month

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    end_date   = datetime.now(dt.timezone.utc).replace(tzinfo=None)
    start_date = end_date - timedelta(days=int(YEARS_BACK * 365))
    log.info(f"ERA5 download range: {start_date.date()} → {end_date.date()}")

    grid = build_grid()
    log.info(f"ERA5 grid: {len(grid)} points at {GRID_STEP_DEG}° spacing "
             f"({BBOX[0]}–{BBOX[2]} lon, {BBOX[1]}–{BBOX[3]} lat)")

    # Save grid point metadata to S3
    meta_df = pd.DataFrame([{"grid_lat": lat, "grid_lon": lon} for lat, lon in grid])
    upload_df_to_s3(meta_df, f"{S3_PREFIX}/metadata/grid_points.parquet", log)

    months = list(month_range(start_date, end_date))
    log.info(f"Downloading {len(months)} months × {len(grid)} grid points "
             f"= {len(months) * len(grid)} total requests")

    for month_start, month_end in months:
        month_str = month_start.strftime("%Y-%m")
        log.info(f"\n{'─'*60}\nMonth: {month_str}")
        success = skipped = failed = 0

        for lat, lon in grid:
            key = lat_lon_key(lat, lon)
            s3_key = f"{S3_PREFIX}/monthly/{month_str}/grid_{key}.parquet"

            if s3_key_exists(s3_key):
                skipped += 1
                continue

            df = fetch_era5_month(lat, lon, month_start, month_end)
            if df is not None and len(df) > 0:
                upload_df_to_s3(df, s3_key, log)
                success += 1
            else:
                log.warning(f"  No data for grid ({lat}, {lon}) {month_str}")
                failed += 1

            time.sleep(SLEEP_BETWEEN_CALLS)

        log.info(f"  Month {month_str}: {success} uploaded, "
                 f"{skipped} skipped, {failed} failed")

    log.info("\nERA5 download complete. s3://bay-area-microclimate/raw/era5/")


if __name__ == "__main__":
    main()
