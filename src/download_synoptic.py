"""
Synoptic Data API downloader for Bay Area Microclimate project.
Fetches historical weather observations and saves to S3 as parquet.

Usage:
    python src/download_synoptic.py

Environment variables required:
    SYNOPTIC_TOKEN   - Synoptic API token (NOT the API key — generate a token
                       from your key at customer.synopticdata.com)
    AWS credentials  - handled automatically by boto3 via ~/.aws/credentials

Notes:
    - Free Open Access tier: 1 year of history max. Set YEARS_BACK env var to
      override (requires paid plan for > 1 year).
    - Resumable: already-uploaded S3 keys are skipped on re-run.
    - S3 layout: s3://bay-area-microclimate/raw/synoptic/
        metadata/stations.parquet
        monthly/YYYY-MM/chunk_XXXX.parquet
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta

import datetime as dt
from dotenv import load_dotenv
import requests
import pandas as pd

load_dotenv()
from dateutil.relativedelta import relativedelta

# Allow running from project root or src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import upload_df_to_s3, s3_key_exists

# ── Config ────────────────────────────────────────────────────────────────────

SYNOPTIC_TOKEN = os.environ["SYNOPTIC_TOKEN"]
S3_PREFIX = "raw/synoptic"
BASE_URL = "https://api.synopticdata.com/v2"

# Free Open Access tier limit: 1 year. Upgrade at synopticdata.com/pricing for more.
YEARS_BACK = float(os.environ.get("YEARS_BACK", 1))

# Variables to pull (Synoptic variable names)
VARS = "air_temp,relative_humidity,wind_speed,wind_direction,precip_accum"

# Bay Area bounding box: lon_min,lat_min,lon_max,lat_max
BAY_AREA_BBOX = "-123.0,36.9,-121.5,38.3"

# Known Bay Area ASOS/official stations — prepended so chunk 0 always contains
# known-good stations useful for verifying end-to-end data flow.
KNOWN_ASOS_STIDS = [
    "KSFO", "KOAK", "KSJC", "KSNS", "KHAF", "KCCR", "KSQL", "KNUQ",
    "KPAO", "KWVI", "KVCB", "KSMF", "KAPC", "KSTS", "KLVK",
    "KHWD", "KSUU", "KMRY", "KMOD",
]

STATIONS_PER_CHUNK = 50
MONTHS_PER_CHUNK = 1
SLEEP_BETWEEN_CALLS = 2  # seconds — stay within rate limits

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("download_synoptic.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Synoptic API ──────────────────────────────────────────────────────────────

def get_bay_area_stations() -> list[dict]:
    """Discover all active stations in the Bay Area that report temperature."""
    log.info("Fetching Bay Area station list from Synoptic...")
    r = requests.get(f"{BASE_URL}/stations/metadata", params={
        "token": SYNOPTIC_TOKEN,
        "bbox": BAY_AREA_BBOX,
        "status": "active",
        "vars": "air_temp",
        "output": "json",
    })
    r.raise_for_status()
    stations = r.json().get("STATION", [])
    log.info(f"Found {len(stations)} stations in Bay Area bbox")
    return stations


def fetch_timeseries(stids: list[str], start: datetime, end: datetime,
                     chunk_index: int = -1) -> list[dict]:
    """Fetch hourly timeseries for up to 50 station IDs over a time range."""
    r = requests.get(f"{BASE_URL}/stations/timeseries", params={
        "token": SYNOPTIC_TOKEN,
        "stid": ",".join(stids),
        "start": start.strftime("%Y%m%d%H%M"),
        "end": end.strftime("%Y%m%d%H%M"),
        "vars": VARS,
        "qc_checks": "basic",
        "output": "json",
        "units": "temp|C,speed|kph,precip|mm",
    }, timeout=60)
    r.raise_for_status()

    data = r.json()
    summary = data.get("SUMMARY", {})
    station_list = data.get("STATION", [])

    log.info(
        f"  [chunk {chunk_index}] API response — "
        f"HTTP {r.status_code} | SUMMARY: {summary} | "
        f"stations returned: {len(station_list)} | "
        f"stids requested: {stids[:5]}{'...' if len(stids) > 5 else ''}"
    )
    if station_list:
        with_obs = [s["STID"] for s in station_list
                    if s.get("OBSERVATIONS", {}).get("date_time")]
        log.info(f"  [chunk {chunk_index}] Stations with observations: {with_obs}")

    return station_list


def parse_station_observations(station: dict) -> pd.DataFrame | None:
    """Parse one station's Synoptic response into a DataFrame row per observation."""
    obs = station.get("OBSERVATIONS", {})
    times = obs.get("date_time")
    if not times:
        return None

    df = pd.DataFrame({
        "datetime":        pd.to_datetime(times, utc=True),
        "temp_c":          obs.get("air_temp_set_1"),
        "humidity":        obs.get("relative_humidity_set_1"),
        "wind_speed_kph":  obs.get("wind_speed_set_1"),
        "wind_dir_deg":    obs.get("wind_direction_set_1"),
        "precip_mm":       obs.get("precip_accum_set_1"),
    })
    df["stid"]    = station["STID"]
    df["name"]    = station.get("NAME", "")
    df["lat"]     = float(station["LATITUDE"])
    df["lon"]     = float(station["LONGITUDE"])
    df["elev_m"]  = float(station.get("ELEVATION") or 0) * 0.3048  # feet → meters
    df["network"] = station.get("MNET_ID", "")
    df["source"]  = "synoptic"
    return df

# ── Helpers ───────────────────────────────────────────────────────────────────

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def month_range(start: datetime, end: datetime):
    """Yield (month_start, month_end) tuples between start and end."""
    current = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while current < end:
        next_month = current + relativedelta(months=MONTHS_PER_CHUNK)
        yield current, min(next_month, end)
        current = next_month

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    end_date   = datetime.now(dt.timezone.utc).replace(tzinfo=None)
    start_date = end_date - timedelta(days=int(YEARS_BACK * 365))
    log.info(f"Download range: {start_date.date()} → {end_date.date()}")

    # Discover stations — ASOS first so chunk 0 contains known-good stations
    bbox_stations  = get_bay_area_stations()
    existing_stids = {s["STID"] for s in bbox_stations}

    log.info(f"Fetching metadata for {len(KNOWN_ASOS_STIDS)} known ASOS stations...")
    r = requests.get(f"{BASE_URL}/stations/metadata", params={
        "token": SYNOPTIC_TOKEN,
        "stid": ",".join(KNOWN_ASOS_STIDS),
        "output": "json",
    })
    r.raise_for_status()
    asos_stations = [s for s in r.json().get("STATION", [])
                     if s["STID"] not in existing_stids]
    log.info(f"Adding {len(asos_stations)} ASOS stations not already in bbox results")

    stations  = asos_stations + bbox_stations
    stid_meta = {s["STID"]: s for s in stations}
    all_stids = list(stid_meta.keys())
    log.info(f"Total stations: {len(all_stids)}  "
             f"(ASOS: {len(asos_stations)}, bbox: {len(bbox_stations)})")

    # Station metadata → S3
    meta_df = pd.DataFrame([{
        "stid":    s["STID"],
        "name":    s.get("NAME", ""),
        "lat":     float(s["LATITUDE"]),
        "lon":     float(s["LONGITUDE"]),
        "elev_m":  float(s.get("ELEVATION") or 0) * 0.3048,
        "network": s.get("MNET_ID", ""),
        "state":   s.get("STATE", ""),
        "source":  "synoptic",
    } for s in stations])
    upload_df_to_s3(meta_df, f"{S3_PREFIX}/metadata/stations.parquet", log)

    # Download loop
    total_months = list(month_range(start_date, end_date))
    num_chunks   = -(-len(all_stids) // STATIONS_PER_CHUNK)
    log.info(f"Downloading {len(total_months)} months × {len(all_stids)} stations "
             f"in chunks of {STATIONS_PER_CHUNK} ({num_chunks} chunks/month)")

    for month_start, month_end in total_months:
        month_str = month_start.strftime("%Y-%m")
        log.info(f"\n{'─'*60}\nMonth: {month_str}")

        for i, stid_chunk in enumerate(chunks(all_stids, STATIONS_PER_CHUNK)):
            s3_key = f"{S3_PREFIX}/monthly/{month_str}/chunk_{i:04d}.parquet"

            if s3_key_exists(s3_key):
                log.info(f"  Skipping chunk {i} (already in S3)")
                continue

            log.info(f"  Chunk {i+1}/{num_chunks}: {len(stid_chunk)} stations")

            try:
                raw_stations = fetch_timeseries(stid_chunk, month_start, month_end,
                                                chunk_index=i)
            except requests.HTTPError as e:
                log.warning(f"  HTTP error on chunk {i}: {e} — skipping")
                time.sleep(SLEEP_BETWEEN_CALLS * 3)
                continue
            except requests.Timeout:
                log.warning(f"  Timeout on chunk {i} — skipping")
                continue

            frames = [parse_station_observations(s) for s in raw_stations]
            frames = [f for f in frames if f is not None and len(f) > 0]

            if frames:
                chunk_df = pd.concat(frames, ignore_index=True)
                upload_df_to_s3(chunk_df, s3_key, log)
            else:
                log.info(f"  No data for chunk {i} "
                         f"({len(raw_stations)} stations in response, 0 with observations)")

            time.sleep(SLEEP_BETWEEN_CALLS)

        log.info(f"Month {month_str} complete.")

    log.info("\nSynoptic download complete. s3://bay-area-microclimate/raw/synoptic/")


if __name__ == "__main__":
    main()
