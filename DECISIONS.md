# Implementation Decisions

Running log of architectural and methodological decisions made during development.
Intended to support uncertainty quantification (UQ) and validation testing.

Pipeline status key used throughout this doc:
- [verified]         Implemented and verified end-to-end
- [not yet run]      Implemented, not yet run / partially verified
- [not implemented] Specced, not yet implemented (no validated code)

---

## Data Pipeline

### Synoptic API [verified]

**Status (2026-03-02)**
The Synoptic download pipeline is verified end-to-end. In the first confirmed run:
- 1,124 stations discovered (1,118 from bbox + 6 ASOS additions)
- Chunk 0 (KSFO, KOAK, KSJC, etc.) returned 44 stations with observations
- `chunk_0000.parquet` uploaded with 155,218 rows
- `chunk_0001.parquet` uploaded with 182,085 rows
- `chunk_0002.parquet` uploaded with 186,401 rows
- Run was manually stopped after 3 chunks; full 1-year download not yet completed

Full 1-year download (13 months × 23 chunks = 299 API calls) still needs to be run to completion. The pipeline is correct but the S3 dataset is partial.

**20-year baseline: blocked by free tier (2026-03-02)**
The original project goal was 20 years of data to span multiple ENSO cycles and capture interannual variability. This is not achievable on the Synoptic Open Access free tier, which caps historical access at 1 year regardless of station.

This is a hard constraint on the project scope, not a TODO:
- Seasonal models trained on 1 year of data cannot generalize across ENSO phases
- Winter 2025–2026 La Niña conditions will be overrepresented in the training set
- Models will likely underperform during anomalous years (strong El Niño, drought)

Options to resolve:
1. Synoptic Enterprise plan (cost unknown, contact sales)
2. Supplement with NOAA ISD free archive for ASOS stations only (KSFO etc. back to 1973)
3. Treat 1-year scope as MVP and revisit after project is otherwise complete

**Token vs. API Key (2026-03-02)**
Synoptic issues two separate credentials: an API Key and a Token generated from that key. The Token is what gets passed as `?token=` in all API requests. Using the API Key directly returns HTTP 401. All scripts use `SYNOPTIC_TOKEN` env var which must be a generated token, not the key itself.

**Variable name: `precip_accum` not `precip_accumulated` (2026-03-02)**
Synoptic's timeseries API uses `precip_accum` as the variable identifier. Using `precip_accumulated` causes the entire API response to return 0 stations with `RESPONSE_CODE: -1`. Discovered via debug logging in `fetch_timeseries()`. Verified correct name from API error message.

**Station filtering: no client-side RESTRICTED filter (2026-03-02)**
An earlier version of the script (`download_bay_area_weather.py`, now deleted) attempted to filter stations client-side using `s.get("RESTRICTED", True)`. This was broken: the default of `True` excluded stations with no RESTRICTED field, and the string `"0"` used by Synoptic for unrestricted stations is truthy in Python, so `not "0"` excluded them too.

The rewritten script (`src/download_synoptic.py`) has no client-side RESTRICTED filter. The API naturally returns only data the token has access to. Adding `restricted=0` as a server-side query param caused HTTP 401, so that approach was also dropped.

**ASOS stations prepended to station list (2026-03-02)**
Known major ASOS stations (KSFO, KOAK, KSJC, etc.) are fetched via separate metadata query and prepended to the station list so they always land in chunk 0. This ensures the first API call contains known-good stations with long records, making it easy to verify end-to-end data flow before committing to a full run.

**Chunk size: 50 stations per timeseries request (2026-03-02)**
Synoptic timeseries API supports batching multiple stations in one call. 50 stations/call was chosen to balance response size (~180k rows/month/chunk) against timeout risk (60s timeout). Larger chunks risk timeouts for months with high data density.

**S3 layout: monthly chunks (2026-03-02)**
`raw/synoptic/monthly/YYYY-MM/chunk_XXXX.parquet` — one file per 50-station group per month. Enables resumable downloads (skip existing S3 keys). Tradeoff: many small files vs. fewer large files. Can be consolidated with a compaction job if read performance suffers during training.

---

### Weather Underground (WU) [not implemented]

**Status (2026-03-02)**
The WU pipeline (`src/download_wunderground.py`) is fully written and reviewed but has not been run. No WU API key exists yet. Access is contingent on registering a personal weather station (PWS) device with the WU network — the device has been purchased and is in transit. Until the device is registered and an API key is issued, none of the WU code has been validated against real API responses.

All WU architectural decisions below should be treated as design intent, not verified behavior.

**Access model: PWS contributor (2026-03-02)**
WU API access requires registering a PWS with the WU network. The device uploads live observations; in return the API key grants access to the full historical PWS network. API key goes in `WU_API_KEY` env var.

**Per-station-per-day API structure (2026-03-02)**
WU history API (`/v2/pws/history/hourly`) returns one day per station per call — no bulk batching. With ~2,000 Bay Area stations × 365 days = ~730,000 calls/year. Mitigated with `ThreadPoolExecutor` (default 5 workers). Rate limits not yet known — tune `MAX_WORKERS` based on actual API responses (watch for HTTP 429).

**Station discovery via grid sampling (2026-03-02)**
WU has no bbox query for historical data. `/v2/pws/nearby` is queried at a 0.15° grid (~15km spacing) across the Bay Area with 12km radius, then deduplicated. Approximately 100 grid queries to discover all stations. Overlap between grid cells intentional to avoid gaps at cell boundaries.

**S3 layout: per-station-month (2026-03-02)**
`raw/wunderground/monthly/YYYY-MM/{stid}.parquet` — one file per station per month. Matches the download granularity and enables efficient resumability. Different from Synoptic's chunk layout because WU cannot be batched.

**WU data quality: not yet assessed (2026-03-02)**
WU backyard stations vary significantly in quality. Known issues: poor siting (near AC exhaust, asphalt, under eaves), cheap sensors, calibration drift, missing elevation metadata. Quality validation plan: compare each WU station's mean temperature against the nearest Synoptic ASOS station adjusted for elevation lapse rate; flag stations with persistent bias > 2°C. Not yet implemented.

---

### ERA5 (Open-Meteo) Implemented, not yet run

**Status (2026-03-02)**
`src/download_era5.py` is written and ready to run. Has not been executed yet — ERA5 download should be kicked off in parallel with or immediately after the Synoptic full run. No API key required.

**Source: Open-Meteo Archive API (2026-03-02)**
Free, no API key required. Provides ERA5 reanalysis back to 1940 at 0.25° resolution. Unlike Synoptic, ERA5 has no historical depth limit on the free tier — can download 20 years freely. This partially offsets the Synoptic 1-year constraint: ERA5 large-scale features will cover the full historical range even if station observations are limited to 1 year.

**Grid resolution: 0.25° (2026-03-02)**
Matches ERA5 native resolution (~27km). For the Bay Area domain this gives a 6×6=36 point grid. ERA5 is intentionally coarse — it provides synoptic context, not microclimate. Microclimate signal comes from the station observations.

**Variables selected (2026-03-02)**
`boundary_layer_height` is included because marine layer intrusion depth is the primary driver of coastal vs. inland temperature differences. `temperature_850hPa` captures the free-atmosphere temperature that determines subsidence inversion strength. Both are non-standard and not included in basic weather APIs.

---

## Static Features Implemented, not yet run

**Status (2026-03-02)**
`src/compute_static_features.py` is written. Depends on Synoptic metadata parquet being in S3 (already uploaded). Can be run as soon as the Synoptic download produces `raw/synoptic/metadata/stations.parquet`.

**Coast distance: waypoint approximation (2026-03-02)**
Pacific coast and SF Bay distances are computed as minimum great-circle distance to a manually defined set of waypoints, not a full coastline polygon. Accuracy is sufficient for ~10km zone boundaries but will introduce errors for stations in complex coastal geometry (Sausalito, Tiburon). A proper coastline shapefile would improve accuracy at the cost of a `shapely`/`geopandas` dependency.

**Elevation source: Synoptic metadata (2026-03-02)**
Station elevation comes from Synoptic's station metadata (reported by operators, in feet, converted to meters). Known to be inaccurate for some stations. SRTM 30m DEM would provide ground truth. Not yet implemented — especially important for WU stations where elevation is often missing entirely.

**Coastal exposure: heuristic composite (2026-03-02)**
`coastal_exposure = 1 - dist_coast_normalized - elev_normalized * 0.3` is a hand-crafted formula, not learned from data. Will be replaced by an empirical feature once there is enough observation data to regress station temperature anomaly against candidate features.

---

## Zone Clustering Implemented, not yet run

**Status (2026-03-02)**
`src/cluster_zones.py` is written. Depends on both Synoptic observations and static features being in S3. Has not been run — no output to evaluate yet.

**Algorithm: K-means (2026-03-02)**
K-means chosen for interpretability and speed. Alternatives considered:
- Hierarchical clustering: better for discovering K, doesn't scale to 10k+ WU stations
- DBSCAN: no predefined K, but every station must belong to a zone — noise points problematic
- GMM: soft assignments useful for fog-belt edge stations, adds complexity — revisit post-WU

**Feature matrix: diurnal cycle + static features (2026-03-02)**
The 24-dimensional diurnal temperature cycle is the dominant clustering signal — coastal stations peak late with small amplitude, inland stations peak earlier with large amplitude. Static features add geographic regularization so clusters are spatially coherent.

**K=7 default (2026-03-02)**
Based on qualitative description of Bay Area microclimates. Not yet validated against data. Run `elbow_plot()` in `cluster_zones.py` before committing to K=7.

**Minimum observations filter: 500 (2026-03-02)**
Stations with < 500 observations excluded from clustering. 500 obs ≈ 3 weeks of hourly data. Somewhat arbitrary — increase to 1,000 if noisy sparse stations are visibly distorting clusters.

---

## Schema

**`source` column (2026-03-02)**
All observation parquets include `source` (`"synoptic"` or `"wunderground"`). Allows filtering, differential weighting, or source-specific validation downstream. WU observations may receive lower training weight until quality validation passes.

**Shared schema across sources (2026-03-02)**
Both sources use identical columns: `datetime`, `temp_c`, `humidity`, `wind_speed_kph`, `wind_dir_deg`, `precip_mm`, `stid`, `name`, `lat`, `lon`, `elev_m`, `network`, `source`. All training code treats sources uniformly.

**Elevation: meters throughout (2026-03-02)**
Synoptic reports in feet, converted on ingest (`* 0.3048`). WU reports in meters with `units=m`. ERA5 grid points have no elevation field. All downstream code assumes `elev_m` is in meters.

---

## Secrets Management

**Pattern: environment variables only (2026-03-02)**
All secrets (API tokens, keys) are passed via environment variables. No secrets are hardcoded in any script. Current variables:
- `SYNOPTIC_TOKEN` — Synoptic API token (not the API key; generate from customer.synopticdata.com)
- `WU_API_KEY` — Weather Underground API key (not yet obtained)
- AWS credentials — managed via `~/.aws/credentials`, not env vars; handled automatically by boto3

**`.gitignore` coverage (2026-03-02)**
`.gitignore` explicitly excludes:
- `*.env` / `.env` — environment files
- `*accessKeys.csv` — AWS IAM key exports (the file `bay-area-microclimate-s3_accessKeys.csv` is present in the repo directory and must never be committed)
- `keys` — empty placeholder file in project root, also excluded

**Exposure incident (2026-03-02)**
Both the Synoptic API key and a subsequently generated token were exposed in HTTP error URLs printed to the terminal during debugging. Both credentials were rotated immediately. The current token in use was generated after rotation. AWS credentials were not exposed.

---

## Open Questions and To Do

Items are marked with acceptance criteria where the answer gates further ML work.

- [ ] **Complete Synoptic 1-year download**
  Done when: all 13 months × 23 chunks exist in S3 with no "No data" chunks for KSFO/KOAK rows.

- [ ] **Run ERA5 download**
  Done when: all 36 grid points × 13 months exist in `raw/era5/monthly/`.

- [ ] **Run static features and zone clustering**
  Done when: `features/zones/zone_assignments.parquet` exists and zone map (lat/lon colored by zone_id) is visually coherent with known Bay Area microclimate geography.

- [ ] **Decide on 20-year data strategy**
  Options: Synoptic Enterprise, NOAA ISD supplement for ASOS only, or accept 1-year scope. Decision needed before designing the seasonal component of the ML model.

- [ ] **Validate WU station quality against Synoptic**
  Done when: a per-station bias score (WU vs. nearest ASOS adjusted for elevation) is computed and a threshold for exclusion is chosen and documented here.

- [ ] **Obtain WU API key and run WU discovery + download**
  Blocked on: device delivery and PWS registration.

- [ ] **Re-run zone clustering after WU integration**
  Revisit K and cluster boundaries. Expected: K increases, coastal fog gradient zones sharpen.

- [ ] **Define train/val/test split strategy**
  Options: temporal split (last 2 months = test), spatial split (hold out zones), zone-stratified random. Choice affects how well the model generalizes to unseen times vs. unseen locations. Decision needed before any model training.

- [ ] **Replace heuristic coastal_exposure with learned feature**
  Done when: linear regression of (temp anomaly vs. dist_coast, elev, dist_bay) has R² > 0.5 on held-out stations.
