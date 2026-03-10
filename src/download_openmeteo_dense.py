"""
Open-Meteo dense grid downloader for Bay Area Microclimate project.
Fetches high-resolution historical weather data at a dense grid of points
across the Bay Area to provide fine-grained spatial coverage.

This replaces the Weather Underground PWS pipeline. Advantages:
    - No API key required (free, open access)
    - Consistent data quality (model output, not noisy PWS sensors)
    - 20+ years of history available (not limited like Synoptic free tier)
    - No dependency on registering a physical weather station device

Usage:
    python src/download_openmeteo_dense.py

Dependencies:
    pip install requests pandas boto3 pyarrow python-dotenv

S3 layout:
    s3://bay-area-microclimate/
        raw/openmeteo_dense/
            metadata/grid_points.parquet   <- dense grid point coordinates
            monthly/YYYY-MM/
                grid_{lat}_{lon}.parquet   <- hourly obs for one point, one month
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
from dateutil.relativedelta import relativedelta

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import upload_df_to_s3, s3_key_exists

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL   = "https://archive-api.open-meteo.com/v1/archive"
S3_PREFIX  = "raw/openmeteo_dense"

# How many years of history to fetch. Open-Meteo archive goes back to 1940.
YEARS_BACK = float(os.environ.get("YEARS_BACK", 1))

# Bay Area bounding box: lon_min, lat_min, lon_max, lat_max
BBOX = (-123.0, 36.9, -121.5, 38.3)

# Dense grid at 0.05° (~5km) spacing — provides fine-grained spatial coverage.
# For the Bay Area bbox this gives roughly 29 lat × 31 lon = ~900 grid points.
# Much denser than ERA5's 0.25° grid (36 points) while still manageable API-wise.
GRID_STEP_DEG = 0.05

# Surface-level variables matching the observation schema.
# These complement ERA5's synoptic-scale variables with local surface detail.
DENSE_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
    "surface_pressure",
]

SLEEP_BETWEEN_CALLS = 1  # seconds — Open-Meteo is free but be polite

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("download_openmeteo_dense.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Grid generation ───────────────────────────────────────────────────────────

def build_dense_grid() -> list[tuple[float, float]]:
    """
    Generate lat/lon grid points covering the Bay Area bbox at dense resolution.
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

def fetch_dense_month(lat: float, lon: float,
                      month_start: datetime, month_end: datetime) -> pd.DataFrame | None:
    """
    Fetch hourly surface weather data for one grid point over one month.
    Returns a DataFrame or None if the request fails.
    """
    r = requests.get(BASE_URL, params={
        "latitude":    lat,
        "longitude":   lon,
        "start_date":  month_start.strftime("%Y-%m-%d"),
        "end_date":    (month_end - timedelta(days=1)).strftime("%Y-%m-%d"),
        "hourly":      ",".join(DENSE_VARIABLES),
        "timezone":    "UTC",
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
        "datetime":         pd.to_datetime(times, utc=True),
        "temp_c":           hourly.get("temperature_2m"),
        "humidity":         hourly.get("relative_humidity_2m"),
        "wind_speed_kph":   hourly.get("wind_speed_10m"),
        "wind_dir_deg":     hourly.get("wind_direction_10m"),
        "precip_mm":        hourly.get("precipitation"),
        "cloud_cover_pct":  hourly.get("cloud_cover"),
        "pressure_hpa":     hourly.get("surface_pressure"),
        "grid_lat":         lat,
        "grid_lon":         lon,
        "source":           "openmeteo_dense",
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
    log.info(f"Open-Meteo dense download range: {start_date.date()} → {end_date.date()}")

    grid = build_dense_grid()
    log.info(f"Dense grid: {len(grid)} points at {GRID_STEP_DEG}° spacing "
             f"({BBOX[0]}–{BBOX[2]} lon, {BBOX[1]}–{BBOX[3]} lat)")

    # Save grid point metadata to S3
    meta_df = pd.DataFrame([{"grid_lat": lat, "grid_lon": lon} for lat, lon in grid])
    upload_df_to_s3(meta_df, f"{S3_PREFIX}/metadata/grid_points.parquet", log)

    months = list(month_range(start_date, end_date))
    total_requests = len(months) * len(grid)
    log.info(f"Downloading {len(months)} months × {len(grid)} grid points "
             f"= {total_requests:,} total requests")

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

            df = fetch_dense_month(lat, lon, month_start, month_end)
            if df is not None and len(df) > 0:
                upload_df_to_s3(df, s3_key, log)
                success += 1
            else:
                log.warning(f"  No data for grid ({lat}, {lon}) {month_str}")
                failed += 1

            time.sleep(SLEEP_BETWEEN_CALLS)

        log.info(f"  Month {month_str}: {success} uploaded, "
                 f"{skipped} skipped, {failed} failed")

    log.info(f"\nOpen-Meteo dense download complete. "
             f"s3://bay-area-microclimate/{S3_PREFIX}/")


if __name__ == "__main__":
    main()
