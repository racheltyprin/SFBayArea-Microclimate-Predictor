"""
Microclimate zone clustering for the Bay Area.

Groups weather stations into distinct microclimate zones based on their
temperature and humidity signatures. These zones become the prediction
targets for the ML model — instead of predicting weather at arbitrary
lat/lon (intractable), the model predicts conditions per zone given the
current large-scale atmospheric state.

Expected Bay Area microclimate zones (approximate):
    - Coastal fog belt (Ocean Beach, Pacifica, Half Moon Bay)
    - North Bay valleys (Petaluma, Santa Rosa)
    - Inner East Bay (Oakland, Berkeley hills rain shadow)
    - South Bay / Silicon Valley (San Jose, Sunnyvale)
    - Hot inland (Livermore, Antioch, Concord)
    - Marine-influenced Bay shores (Alameda, Redwood City)

Usage:
    python src/cluster_zones.py

Dependencies:
    pip install pandas boto3 pyarrow scikit-learn matplotlib

CURRENT DATA SOURCE COVERAGE — SYNOPTIC ONLY:
    This script currently clusters using only Synoptic station observations
    (~1,124 stations, 1 year). This provides reasonable geographic coverage
    of the Bay Area but is sparse in residential neighborhoods.

    TODO (Open-Meteo): When Open-Meteo dense grid data is available, load from:
        s3://bay-area-microclimate/raw/open_meteo/monthly/
    Merge with Synoptic observations before building the feature matrix.
    The dense grid (~961 points at 0.05° spacing) will sharpen zone
    boundaries significantly, particularly along the coastal fog gradient.

S3 layout:
    Input:
        raw/synoptic/monthly/YYYY-MM/chunk_XXXX.parquet
        [TODO: raw/open_meteo/monthly/YYYY-MM/grid_{lat}_{lon}.parquet]
        features/static/stations_with_features.parquet
    Output:
        features/zones/zone_assignments.parquet   ← stid → zone_id + zone metadata
        features/zones/zone_profiles.parquet      ← per-zone mean diurnal cycle
"""

import os
import sys
import logging
from io import BytesIO

import boto3
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import upload_df_to_s3, S3_BUCKET

# ── Config ────────────────────────────────────────────────────────────────────

# Number of microclimate zones. The Bay Area is typically described as having
# 5–10 distinct microclimates. Start at 7 and tune with elbow plot output.
N_ZONES = int(os.environ.get("N_ZONES", 7))

# Minimum observations a station must have (across all months) to be included
# in clustering. Sparse stations bias the cluster centroids.
MIN_OBS_PER_STATION = 500

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("cluster_zones.log"),
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

# ── Data loading ──────────────────────────────────────────────────────────────

def load_synoptic_observations() -> pd.DataFrame:
    """
    Load all Synoptic monthly parquet files from S3 and return a single
    DataFrame with columns: datetime, stid, temp_c, humidity.

    Only temperature and humidity are used for clustering — they most strongly
    reflect microclimate character. Wind is more transient/topographic.
    """
    log.info("Scanning Synoptic monthly parquet files...")
    keys = list_s3_keys("raw/synoptic/monthly/")
    log.info(f"Found {len(keys)} Synoptic chunk files")

    frames = []
    for i, key in enumerate(keys):
        try:
            df = load_parquet_from_s3(key)
            frames.append(df[["datetime", "stid", "temp_c", "humidity"]].copy())
        except Exception as e:
            log.warning(f"  Failed to load {key}: {e}")
        if (i + 1) % 50 == 0:
            log.info(f"  Loaded {i+1}/{len(keys)} files...")

    if not frames:
        raise RuntimeError("No Synoptic observation data found in S3.")

    obs = pd.concat(frames, ignore_index=True)
    obs["datetime"] = pd.to_datetime(obs["datetime"], utc=True)
    obs["source"] = "synoptic"
    log.info(f"Synoptic observations loaded: {len(obs):,} rows, "
             f"{obs['stid'].nunique()} stations")
    return obs


def load_all_observations() -> pd.DataFrame:
    """
    Load observations from all available data sources.
    Currently: Synoptic only.

    TODO (Open-Meteo): When Open-Meteo data is available, also load:
        keys = list_s3_keys("raw/open_meteo/monthly/")
        om_frames = [load_parquet_from_s3(k)[["datetime","stid","temp_c","humidity"]]
                     for k in keys]
        om_obs = pd.concat(om_frames, ignore_index=True)
        om_obs["source"] = "open_meteo"

        obs = pd.concat([synoptic_obs, om_obs], ignore_index=True)

    WU data will dramatically improve zone boundary sharpness, especially in:
    - The coastal fog gradient (Half Moon Bay → San Jose)
    - Hillside neighborhoods with cold-air drainage
    - Urban heat island within cities
    """
    synoptic_obs = load_synoptic_observations()

    # TODO (WU): replace this return with merged obs (see docstring above)
    return synoptic_obs

# ── Feature matrix ────────────────────────────────────────────────────────────

def build_station_feature_matrix(obs: pd.DataFrame,
                                  static: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature matrix where each row is a station and each column is a
    feature used for clustering.

    Temporal features (from observations):
        - mean_temp_c        : annual mean temperature
        - temp_std_c         : temperature variability (high = more continental)
        - diurnal_range_c    : mean daily max − min (low = coastal, high = inland)
        - hourly_0..23_temp  : mean temperature at each hour of day (diurnal cycle)
          — the diurnal cycle shape is the strongest microclimate signal

    Static geographic features (from compute_static_features.py):
        - dist_coast_km      : proximity to Pacific coast
        - dist_bay_km        : proximity to SF Bay
        - elev_m             : elevation
        - coastal_exposure   : composite coastal score

    TODO (WU): When WU stations are included, the feature matrix grows from
    ~1,124 rows to potentially ~10,000 rows. This improves cluster quality but
    may require scaling N_ZONES up to 10–15 to capture finer zone boundaries.
    """
    log.info("Building station feature matrix...")

    obs = obs.copy()
    obs["hour"] = obs["datetime"].dt.hour

    # Aggregate per station
    agg = obs.groupby("stid")["temp_c"].agg(
        mean_temp_c="mean",
        temp_std_c="std",
    ).reset_index()

    # Diurnal range: mean of (daily max - daily min)
    obs["date"] = obs["datetime"].dt.date
    daily = obs.groupby(["stid", "date"])["temp_c"].agg(
        daily_max="max", daily_min="min"
    ).reset_index()
    daily["diurnal_range"] = daily["daily_max"] - daily["daily_min"]
    diurnal_range = daily.groupby("stid")["diurnal_range"].mean().reset_index()
    diurnal_range.columns = ["stid", "diurnal_range_c"]

    # Hourly mean temperature (24 features — the diurnal cycle shape)
    hourly_means = (
        obs.groupby(["stid", "hour"])["temp_c"]
        .mean()
        .unstack(level="hour")
        .reset_index()
    )
    hourly_means.columns = (
        ["stid"] + [f"hour_{h:02d}_temp" for h in range(24)]
    )

    # Merge temporal features
    features = (
        agg
        .merge(diurnal_range, on="stid", how="left")
        .merge(hourly_means, on="stid", how="left")
    )

    # Merge static geographic features
    static_cols = ["stid", "lat", "lon", "elev_m",
                   "dist_coast_km", "dist_bay_km", "coastal_exposure"]
    available_static_cols = [c for c in static_cols if c in static.columns]
    features = features.merge(static[available_static_cols], on="stid", how="left")

    # Drop stations with too few observations
    obs_counts = obs.groupby("stid").size().rename("n_obs")
    features = features.merge(obs_counts, on="stid", how="left")
    before = len(features)
    features = features[features["n_obs"] >= MIN_OBS_PER_STATION].copy()
    log.info(f"Dropped {before - len(features)} stations with < "
             f"{MIN_OBS_PER_STATION} observations, {len(features)} remain")

    return features


# ── Clustering ────────────────────────────────────────────────────────────────

def run_clustering(features: pd.DataFrame, n_zones: int) -> pd.DataFrame:
    """
    Cluster stations into microclimate zones using K-means.

    The feature columns used for clustering are standardized before fitting so
    that temperature (°C scale) and distance (km scale) contribute equally.

    Returns the features DataFrame with a 'zone_id' column added.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Columns to cluster on — all numeric except identifiers
    exclude = {"stid", "source", "network", "state", "name", "n_obs"}
    cluster_cols = [c for c in features.columns
                    if c not in exclude and features[c].dtype in (float, "float64", int, "int64")]

    feature_matrix = features[cluster_cols].copy()
    # Fill NaN with column median — some stations may be missing static features
    feature_matrix = feature_matrix.fillna(feature_matrix.median())

    log.info(f"Clustering {len(feature_matrix)} stations into {n_zones} zones "
             f"using {len(cluster_cols)} features...")

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)

    kmeans = KMeans(n_clusters=n_zones, random_state=42, n_init=20)
    features = features.copy()
    features["zone_id"] = kmeans.fit_predict(X)

    # Log zone sizes
    zone_sizes = features.groupby("zone_id").size().sort_values(ascending=False)
    log.info(f"Zone sizes:\n{zone_sizes.to_string()}")

    # Log zone centroids (mean temp and coast distance — most interpretable)
    summary = features.groupby("zone_id")[
        ["mean_temp_c", "diurnal_range_c", "dist_coast_km", "elev_m"]
    ].mean().round(1)
    log.info(f"\nZone profiles (mean values):\n{summary.to_string()}")

    return features, kmeans.inertia_


def elbow_plot(features: pd.DataFrame, k_range=range(3, 16)):
    """
    Compute inertia for a range of K values and log the elbow curve.
    Use this to choose N_ZONES before the full run.

    TODO (WU): Re-run the elbow analysis after adding WU stations — the optimal
    K will likely increase (more stations → more resolvable fine-scale zones).
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    exclude = {"stid", "source", "network", "state", "name", "n_obs"}
    cluster_cols = [c for c in features.columns
                    if c not in exclude and features[c].dtype in (float, "float64", int, "int64")]
    X = StandardScaler().fit_transform(
        features[cluster_cols].fillna(features[cluster_cols].median())
    )

    log.info("Running elbow analysis...")
    for k in k_range:
        inertia = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_
        log.info(f"  K={k:2d}  inertia={inertia:.1f}")


# ── Zone profiles ─────────────────────────────────────────────────────────────

def build_zone_profiles(obs: pd.DataFrame,
                         assignments: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean diurnal temperature cycle for each zone.
    This is the zone's "fingerprint" used at inference time.

    TODO (WU): When WU data is included in obs, zone profiles will be computed
    from many more stations per zone, making them more representative and
    robust to individual station noise/calibration errors.
    """
    obs_with_zone = obs.merge(assignments[["stid", "zone_id"]], on="stid", how="inner")
    obs_with_zone["hour"] = obs_with_zone["datetime"].dt.hour
    obs_with_zone["month"] = obs_with_zone["datetime"].dt.month

    profiles = (
        obs_with_zone
        .groupby(["zone_id", "month", "hour"])["temp_c"]
        .agg(mean_temp_c="mean", std_temp_c="std", n_obs="count")
        .reset_index()
    )
    return profiles


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info(f"Zone clustering starting — target K={N_ZONES}")
    log.info("NOTE: Currently using Synoptic data only. "
             "Re-run after WU integration for finer zone boundaries.")

    # Load data
    obs     = load_all_observations()
    log.info("Loading static station features...")
    static  = load_parquet_from_s3("features/static/stations_with_features.parquet")

    # Build feature matrix and cluster
    features  = build_station_feature_matrix(obs, static)

    # Optional: run elbow analysis first to pick K
    # elbow_plot(features)

    features_with_zones, inertia = run_clustering(features, N_ZONES)
    log.info(f"Clustering complete — inertia: {inertia:.1f}")

    # Save zone assignments
    assignments = features_with_zones[
        ["stid", "zone_id", "lat", "lon", "elev_m",
         "mean_temp_c", "diurnal_range_c", "dist_coast_km",
         "dist_bay_km", "coastal_exposure"]
    ].copy()

    upload_df_to_s3(assignments, "features/zones/zone_assignments.parquet", log)

    # Save zone diurnal profiles
    profiles = build_zone_profiles(obs, assignments)
    upload_df_to_s3(profiles, "features/zones/zone_profiles.parquet", log)

    log.info(f"\nZone clustering complete. {N_ZONES} zones saved to S3.")
    log.info("   NOTE: Synoptic-only clustering. Zones will sharpen once WU "
             "data is integrated — re-run this script after WU download completes.")


if __name__ == "__main__":
    main()
