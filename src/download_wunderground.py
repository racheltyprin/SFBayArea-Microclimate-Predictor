"""
Weather Underground PWS API downloader for Bay Area Microclimate project.
Fetches historical hourly observations from WU's PWS network and saves to S3.

Usage:
    python src/download_wunderground.py

Environment variables required:
    WU_API_KEY      - Weather Underground API key (from wunderground.com/member/api-keys)
    AWS credentials - handled automatically by boto3 via ~/.aws/credentials

Notes:
    - Requires a PWS contributor account (buy a personal weather station and
      register it at wunderground.com to unlock API access).
    - WU history API is per-station per-day, so this script parallelizes across
      stations to stay within rate limits while maximizing throughput.
    - Resumable: already-uploaded station-months are skipped on re-run.
    - S3 layout: s3://bay-area-microclimate/raw/wunderground/
        metadata/stations.parquet
        monthly/YYYY-MM/{station_id}.parquet
    - Output schema matches download_synoptic.py for unified ML pipeline use.
"""

import os
import sys
import time
import logging
import datetime as dt
from datetime import datetime, timedelta, date
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

# Allow running from project root or src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import upload_df_to_s3, s3_key_exists

# ── Config ────────────────────────────────────────────────────────────────────

WU_API_KEY = os.environ["WU_API_KEY"]
S3_PREFIX  = "raw/wunderground"
BASE_URL   = "https://api.weather.com/v2/pws"

# How many years of history to attempt (WU PWS stations vary — many only have
# 1-3 years of data regardless of what you request).
YEARS_BACK = float(os.environ.get("YEARS_BACK", 1))

# Bay Area bounding box for station discovery
# lon_min, lat_min, lon_max, lat_max
BBOX = (-123.0, 36.9, -121.5, 38.3)

# Grid spacing for nearby-station discovery queries (degrees ~11km at this lat)
GRID_STEP_DEG = 0.15

# Radius (km) for each nearby query — large enough to overlap grid cells
NEARBY_RADIUS_KM = 12

# Max stations returned per nearby query (WU API max is 150)
NEARBY_LIMIT = 150

# Parallel workers for station history downloads.
# Tune down if you hit rate-limit errors (HTTP 429).
MAX_WORKERS = 5

# Pause between API calls per worker (seconds)
SLEEP_PER_CALL = 1.0

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("download_wunderground.log"),
    ],
)
log = logging.getLogger(__name__)

# ── WU API helpers ────────────────────────────────────────────────────────────

def _wu_params(**kwargs) -> dict:
    """Build standard WU API query params (metric units, JSON output)."""
    return {"apiKey": WU_API_KEY, "format": "json", "units": "m", **kwargs}


def discover_stations() -> list[dict]:
    """
    Sample a grid of lat/lon points across the Bay Area bbox and collect all
    nearby PWS stations. Returns deduplicated list of station dicts.
    """
    lon_min, lat_min, lon_max, lat_max = BBOX
    seen: set[str] = set()
    stations: list[dict] = []

    lats = []
    lat = lat_min
    while lat <= lat_max:
        lats.append(lat)
        lat += GRID_STEP_DEG

    lons = []
    lon = lon_min
    while lon <= lon_max:
        lons.append(lon)
        lon += GRID_STEP_DEG

    total_points = len(lats) * len(lons)
    log.info(f"Sampling {total_points} grid points ({len(lats)} lat × {len(lons)} lon) "
             f"with {NEARBY_RADIUS_KM}km radius each...")

    for i, lat in enumerate(lats):
        for lon in lons:
            try:
                r = requests.get(
                    f"{BASE_URL}/nearby",
                    params=_wu_params(lat=lat, lon=lon,
                                      radius=NEARBY_RADIUS_KM,
                                      limit=NEARBY_LIMIT),
                    timeout=15,
                )
                r.raise_for_status()
                for s in r.json().get("stations", []):
                    stid = s.get("stationIdentifier")
                    if stid and stid not in seen:
                        seen.add(stid)
                        stations.append(s)
            except requests.RequestException as e:
                log.warning(f"  nearby query failed at ({lat:.2f}, {lon:.2f}): {e}")
            time.sleep(SLEEP_PER_CALL)

        if (i + 1) % 5 == 0:
            log.info(f"  Grid progress: {(i+1)*len(lons)}/{total_points} points "
                     f"— {len(stations)} unique stations so far")

    log.info(f"Station discovery complete: {len(stations)} unique WU stations found")
    return stations


def fetch_station_day(stid: str, day: date) -> list[dict]:
    """
    Fetch hourly observations for one station on one day.
    Returns a list of observation dicts (empty on error or no data).
    """
    r = requests.get(
        f"{BASE_URL}/history/hourly",
        params=_wu_params(
            stationId=stid,
            date=day.strftime("%Y%m%d"),
            numericPrecision="decimal",
        ),
        timeout=15,
    )
    if r.status_code == 204:  # no content — station has no data for this day
        return []
    r.raise_for_status()
    return r.json().get("observations", [])


def parse_observations(obs_list: list[dict], station_meta: dict) -> pd.DataFrame | None:
    """
    Parse WU hourly observation list into a DataFrame matching the Synoptic schema.
    Uses metric sub-dict (temp in °C, speed in kph, precip in mm).
    """
    if not obs_list:
        return None

    rows = []
    for obs in obs_list:
        m = obs.get("metric", {})
        rows.append({
            "datetime":        pd.Timestamp(obs["obsTimeUtc"]),
            "temp_c":          m.get("tempAvg"),
            "humidity":        obs.get("humidityAvg"),
            "wind_speed_kph":  m.get("windspeedAvg"),
            "wind_dir_deg":    obs.get("winddirAvg"),
            "precip_mm":       m.get("precipTotal"),
            "stid":            obs.get("stationID", station_meta.get("stationIdentifier", "")),
            "name":            station_meta.get("name", ""),
            "lat":             obs.get("lat", station_meta.get("lat")),
            "lon":             obs.get("lon", station_meta.get("lon")),
            "elev_m":          station_meta.get("elevation"),  # already meters (units=m)
            "network":         "wunderground",
            "source":          "wunderground",
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


def download_station_month(station: dict, year: int, month: int) -> pd.DataFrame | None:
    """
    Download all hourly obs for one station over one calendar month.
    Returns combined DataFrame or None if no data found.
    """
    stid = station["stationIdentifier"]
    start = date(year, month, 1)
    # Last day of month
    if month == 12:
        end = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)

    frames = []
    current = start
    while current <= end:
        try:
            obs_list = fetch_station_day(stid, current)
            df = parse_observations(obs_list, station)
            if df is not None:
                frames.append(df)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                log.warning(f"  Rate limited on {stid} {current} — sleeping 30s")
                time.sleep(30)
            else:
                log.warning(f"  HTTP error for {stid} {current}: {e}")
        except requests.RequestException as e:
            log.warning(f"  Request error for {stid} {current}: {e}")
        current += timedelta(days=1)
        time.sleep(SLEEP_PER_CALL)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)

# ── Main ──────────────────────────────────────────────────────────────────────

def iter_year_months(years_back: float):
    """Yield (year, month) tuples from years_back ago to now, oldest first."""
    end   = datetime.now(dt.timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=int(years_back * 365))
    current = start.replace(day=1)
    while current <= end:
        yield current.year, current.month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)


def process_station_month(station: dict, year: int, month: int):
    """Worker function: download one station-month and upload to S3 if not present."""
    stid    = station["stationIdentifier"]
    s3_key  = f"{S3_PREFIX}/monthly/{year:04d}-{month:02d}/{stid}.parquet"

    if s3_key_exists(s3_key):
        return stid, "skipped"

    df = download_station_month(station, year, month)
    if df is not None and len(df) > 0:
        upload_df_to_s3(df, s3_key, log)
        return stid, f"{len(df)} rows"
    else:
        return stid, "no data"


def main():
    log.info(f"Weather Underground downloader starting — {YEARS_BACK} year(s) back")

    # Phase 1: Discover stations
    stations = discover_stations()
    if not stations:
        log.error("No stations found — check your WU_API_KEY and network connection")
        return

    # Save station metadata to S3
    meta_df = pd.DataFrame([{
        "stid":    s["stationIdentifier"],
        "name":    s.get("name", ""),
        "lat":     s.get("lat"),
        "lon":     s.get("lon"),
        "elev_m":  s.get("elevation"),
        "network": "wunderground",
        "source":  "wunderground",
    } for s in stations])
    upload_df_to_s3(meta_df, f"{S3_PREFIX}/metadata/stations.parquet", log)
    log.info(f"Saved metadata for {len(stations)} WU stations")

    # Phase 2: Download history month by month
    year_months = list(iter_year_months(YEARS_BACK))
    total_tasks = len(stations) * len(year_months)
    log.info(f"Downloading {len(year_months)} months × {len(stations)} stations "
             f"= {total_tasks:,} station-months  (workers={MAX_WORKERS})")

    for year, month in year_months:
        log.info(f"\n{'─'*60}\nMonth: {year:04d}-{month:02d}")
        done = skipped = no_data = errors = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(process_station_month, station, year, month): station
                for station in stations
            }
            for future in as_completed(futures):
                try:
                    _, result = future.result()
                    if result == "skipped":
                        skipped += 1
                    elif result == "no data":
                        no_data += 1
                    else:
                        done += 1
                except Exception as e:
                    errors += 1
                    station = futures[future]
                    log.warning(f"  Error on {station['stationIdentifier']}: {e}")

        log.info(f"  Month {year:04d}-{month:02d} done — "
                 f"uploaded: {done}, skipped: {skipped}, "
                 f"no data: {no_data}, errors: {errors}")

    log.info("\nWU download complete. s3://bay-area-microclimate/raw/wunderground/")


if __name__ == "__main__":
    main()
