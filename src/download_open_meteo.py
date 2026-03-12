"""
Open-Meteo surface weather downloader for Bay Area Microclimate project.
Fetches historical hourly surface observations from the Open-Meteo Archive API
at a dense grid of points covering the Bay Area and saves to S3.

This replaces the Weather Underground PWS downloader. Open-Meteo provides free
historical weather data at any lat/lon point without an API key, combining
weather model output and station observations. The grid is denser than ERA5
(0.05° ≈ 5km vs ERA5's 0.25° ≈ 27km) to better resolve microclimate gradients.

Usage:
    python src/download_open_meteo.py

Environment variables:
    AWS credentials - handled automatically by boto3 via ~/.aws/credentials
    YEARS_BACK      - years of history to fetch (default: 1)

No API key required.

S3 layout:
    s3://bay-area-microclimate/
        raw/open_meteo/
            metadata/grid_points.parquet
            monthly/YYYY-MM/grid_{lat}_{lon}.parquet

Output schema matches download_synoptic.py for unified ML pipeline use.
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
S3_PREFIX  = "raw/open_meteo"

YEARS_BACK = float(os.environ.get("YEARS_BACK", 1))

# Bay Area bounding box: lon_min, lat_min, lon_max, lat_max
BBOX = (-123.0, 36.9, -121.5, 38.3)

# Dense grid at 0.05° (~5km) to resolve microclimate gradients.
# This is 5x finer than the ERA5 grid (0.25°) and produces roughly
# 31 lat × 31 lon = ~961 grid points.
GRID_STEP_DEG = 0.05

# Hourly surface variables to fetch. Selected to match the shared observation
# schema used by download_synoptic.py.
SURFACE_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]

SLEEP_BETWEEN_CALLS = 0.5  # seconds — Open-Meteo is free but be polite

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/download_open_meteo.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Grid generation ──────────────────────────────────────────────────────────

def build_grid() -> list[tuple[float, float]]:
    """
    Generate lat/lon grid points covering the Bay Area bbox at 0.05° spacing.
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

# ── Open-Meteo API ───────────────────────────────────────────────────────────

def fetch_surface_month(lat: float, lon: float,
                        month_start: datetime, month_end: datetime) -> pd.DataFrame | None:
    """
    Fetch hourly surface weather for one grid point over one month
    via Open-Meteo Archive API. Returns a DataFrame in the shared
    observation schema, or None on failure.
    """
    r = requests.get(BASE_URL, params={
        "latitude":   lat,
        "longitude":  lon,
        "start_date": month_start.strftime("%Y-%m-%d"),
        "end_date":   (month_end - timedelta(days=1)).strftime("%Y-%m-%d"),
        "hourly":     ",".join(SURFACE_VARIABLES),
        "timezone":   "UTC",
    }, timeout=60)

    if r.status_code != 200:
        log.warning(f"  Open-Meteo error {r.status_code} for ({lat}, {lon}): {r.text[:200]}")
        return None

    data = r.json()
    hourly = data.get("hourly", {})
    times = hourly.get("time")
    if not times:
        return None

    stid = f"OM_{lat_lon_key(lat, lon)}"

    df = pd.DataFrame({
        "datetime":       pd.to_datetime(times, utc=True),
        "temp_c":         hourly.get("temperature_2m"),
        "humidity":       hourly.get("relative_humidity_2m"),
        "wind_speed_kph": hourly.get("wind_speed_10m"),
        "wind_dir_deg":   hourly.get("wind_direction_10m"),
        "precip_mm":      hourly.get("precipitation"),
        "stid":           stid,
        "name":           f"Open-Meteo grid ({lat}, {lon})",
        "lat":            lat,
        "lon":            lon,
        "elev_m":         data.get("elevation"),
        "network":        "open_meteo",
        "source":         "open_meteo",
    })
    return df

# ── Month range helper ───────────────────────────────────────────────────────

def month_range(start: datetime, end: datetime):
    current = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while current < end:
        next_month = current + relativedelta(months=1)
        yield current, min(next_month, end)
        current = next_month

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    end_date   = datetime.now(dt.timezone.utc).replace(tzinfo=None)
    start_date = end_date - timedelta(days=int(YEARS_BACK * 365))
    log.info(f"Open-Meteo surface download range: {start_date.date()} → {end_date.date()}")

    grid = build_grid()
    log.info(f"Surface grid: {len(grid)} points at {GRID_STEP_DEG}° spacing "
             f"({BBOX[0]}–{BBOX[2]} lon, {BBOX[1]}–{BBOX[3]} lat)")

    # Save grid point metadata to S3
    meta_df = pd.DataFrame([{
        "stid":    f"OM_{lat_lon_key(lat, lon)}",
        "name":    f"Open-Meteo grid ({lat}, {lon})",
        "lat":     lat,
        "lon":     lon,
        "network": "open_meteo",
        "source":  "open_meteo",
    } for lat, lon in grid])
    upload_df_to_s3(meta_df, f"{S3_PREFIX}/metadata/grid_points.parquet", log)

    months = list(month_range(start_date, end_date))
    log.info(f"Downloading {len(months)} months × {len(grid)} grid points "
             f"= {len(months) * len(grid):,} total requests")

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

            df = fetch_surface_month(lat, lon, month_start, month_end)
            if df is not None and len(df) > 0:
                upload_df_to_s3(df, s3_key, log)
                success += 1
            else:
                log.warning(f"  No data for grid ({lat}, {lon}) {month_str}")
                failed += 1

            time.sleep(SLEEP_BETWEEN_CALLS)

        log.info(f"  Month {month_str}: {success} uploaded, "
                 f"{skipped} skipped, {failed} failed")

    log.info(f"\nOpen-Meteo surface download complete. "
             f"s3://bay-area-microclimate/{S3_PREFIX}/")


if __name__ == "__main__":
    main()
