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

**Multi-year ENSO coverage strategy (2026-03-10)**
Synoptic free tier is capped at 1 year, but ERA5 has no limit. The decided approach:
- **ERA5:** download 10 years (2016–2026) to capture multiple ENSO phases:
  - El Niño: 2023–2024 (strong), 2018–2019 (weak)
  - La Niña: 2020–2022 (triple-dip), 2025–2026 (current)
  - Neutral: 2019–2020, 2024–2025
- **Synoptic:** 1 year only (free tier cap). Primary training data from real observations.

This resolves the original "blocked by free tier" concern without requiring Synoptic Enterprise or NOAA ISD supplements.

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

### Removed data sources

**Weather Underground (removed 2026-03-10)**
Abandoned because WU's API access model is unreliable — requires registering a physical PWS device, the API key issuance process is opaque, and the APIs have known issues.

**Open-Meteo Dense Grid (removed 2026-03-10)**
Originally planned as a replacement for Weather Underground, providing ~899 grid points at 0.05° spacing across the Bay Area. Removed because Open-Meteo surface data is model-interpolated reanalysis output, not real observations. Training on it would teach the model to replicate another model's interpolation assumptions rather than learning from reality. Synoptic's ~600+ professional stations already provide sufficient spatial coverage with real observations, and terrain features (DEM, NLCD) encode the neighborhood-level physics that a dense grid was meant to capture. The `src/download_open_meteo.py` and `src/download_openmeteo_dense.py` scripts are retained but unused.

---

### ERA5 (Open-Meteo) [verified]

**Status (2026-03-10)**
Complete: 546 files (42 grid points × 13 months) in S3. Download verified end-to-end.

**Source: Open-Meteo Archive API (2026-03-02)**
Free, no API key required. Provides ERA5 reanalysis back to 1940 at 0.25° resolution. No historical depth limit — downloading 10 years (2016–2026) for ENSO coverage.

**Grid resolution: 0.25° (2026-03-02)**
Matches ERA5 native resolution (~27km). For the Bay Area domain this gives 42 grid points. ERA5 is intentionally coarse — it provides synoptic-scale atmospheric context, not microclimate detail.

**ERA5 as a primary model input, not just background context (2026-03-10)**
`boundary_layer_height` is the depth of the marine layer — directly controlling fog penetration and the Sunset/Mission dynamic. Very few microclimate studies use this as an input feature. Combined with `temp_850hPa` (the temperature aloft that drives the subsidence inversion), ERA5 provides physical variables that encode *why* fog behaves differently across neighborhoods, not just surface conditions. This upgrades ERA5 from "coarse background feature" to a genuinely important model input despite its resolution limitation.

**ERA5 interpolation to station/grid locations (2026-03-10)**
ERA5 values must be spatially interpolated to each Synoptic station location (and later to arbitrary prediction grid points for Stage 3). The 42-point ERA5 grid is too coarse to use directly — bilinear interpolation at each target lat/lon will produce per-location-timestep ERA5 features. This is a preprocessing step, not a new download.

---

## Static / Terrain Features

**Current status (2026-03-10)**
`src/compute_static_features.py` computes coastal distance and bay distance from waypoints. Already run — output in S3. However, the Stage 1 GBT model needs additional terrain features not yet collected.

**What exists:**
- `dist_coast_km` — great-circle distance to Pacific coast waypoints (sufficient for ~10km zones, imprecise at Sausalito/Tiburon)
- `dist_bay_km` — distance to SF Bay shoreline waypoints
- `elev_m` — from Synoptic metadata (feet→meters). Known inaccuracies for some stations.
- `coastal_exposure` — heuristic composite, will be replaced by model-learned feature

**What's needed for Stage 1 GBT (not yet collected):**
- **DEM-derived terrain features** — elevation, slope, aspect at each station and grid point. Source: SRTM 30m or USGS 3DEP (both free). Critical for cold-air pooling and shadow effects. Current Synoptic elevation metadata is operator-reported and unreliable.
- **Land cover type** — urban/vegetation/water classification at each station. Source: NLCD (National Land Cover Database, free). Needed to distinguish built environment thermal effects from natural terrain.
- Both must be computed at Synoptic station locations (Stage 1) and at prediction grid points (Stage 3).

---

## Modeling Strategy

### Architecture decision: GBT-first, not Geo-LSTM-Kriging (2026-03-10)

**Decision:** Lead with gradient-boosted trees (XGBoost/LightGBM), not the Geo-LSTM-Kriging architecture from Han et al.

**Rationale — literature failures at mesoscale:**
- Han et al.'s Warsaw results: Kriging achieved RMSE 3.0°C and R² 0.58 at mesoscale. These numbers are a concrete failure case for Bay Area microclimate prediction where we need sub-degree accuracy to differentiate neighborhoods.
- The vertical dimension is absent from Geo-LSTM-Kriging. SF fog is fundamentally a vertical phenomenon — marine layer height, subsidence inversion — that none of these architectures handle natively.
- At neighborhood scale with uneven station density, the Kriging interpolation layer is more likely to hurt than help.

**Predict variables separately (2026-03-10)**
One model per target variable (temperature, humidity, wind speed, wind direction, precipitation). Each has different dominant drivers — fog is primarily coastal distance + season, temperature is terrain + land cover, humidity is both. Separate models reveal what's driving each variable. Multi-output architectures are a later optimization, not a starting point.

### Stage 1: GBT per variable [verified, ongoing]

One HistGradientBoostingRegressor model per target variable (LightGBM/XGBoost dropped — `libomp` unavailable on macOS; sklearn HGBR has no OpenMP dependency and natively handles NaN). Each model takes a flat feature vector per station-timestep.

**Observation features:**
- Synoptic lag values for that station (t-1hr, t-3hr, t-6hr)
- Neighboring station values (mean of 5 nearest stations)

**ERA5 features interpolated to station location:**
- `boundary_layer_height` — most important feature for fog dynamics
- `cloud_cover`, `pressure`, wind components
- `elev_above_blh_m` — derived: station elevation minus BLH. Positive = above inversion. Top-15 feature for both temp and humidity.
- `aspect_northness` — cos(aspect_deg), encodes north-facing exposure
- `northness_x_blh_deficit` — interaction: north-facing exposure × distance below inversion

**Static terrain features per station:**
- Elevation, coastal distance, slope, aspect (from SRTM via Open-Meteo Elevation API)
- Land cover type — urban/vegetation/water (from NLCD 2021 via MRLC WMS)

**Results (v2 models, Feb–Mar 2026 test set):**

| Variable | RMSE | R² | vs ERA5 direct | vs persistence |
|---|---|---|---|---|
| temp_c | 0.709°C | 0.978 | −71% RMSE | −38% RMSE |
| humidity | 4.303% | 0.949 | −72% RMSE | −19% RMSE |
| wind_speed_kph | 2.999 kph | 0.814 | — | — |
| wind_dir_deg | 54.6° | 0.630 | — | — |
| precip_mm | 21.7 mm | 0.995 | — | — |

Models saved to `models/stage1_v2/`. v1 models (without inversion features) retained at `models/stage1/` for comparison. Training data at `features/training_v2/` (S3); v1 data retained at `features/training/`.

**Known limitation:** Ridge stations near the marine layer inversion boundary (e.g. F2543, Twin Peaks, elev 152m) show elevated humidity error (RMSE 21.8%) even after adding `elev_above_blh_m`. The GBT cannot fully learn the sharp inversion boundary from a smooth training set. This is a Stage 3 spatial interpolation problem — once ridge stations anchor their own microclimate zone rather than being averaged with lower-elevation neighbors, this should resolve.

**Status note (2026-03-16):** Stage 1 is functionally complete and producing valid results, but accuracy improvements are ongoing. Feature engineering iterations (e.g. inversion height features added in v2) will continue as new physical signals are identified. The v1/v2 model split is retained in `models/stage1/` and `models/stage1_v2/` to support before/after comparison as further changes are made.

**Identified improvement opportunities (2026-03-16):**

_High impact:_
- **Increase training data cap** — `MAX_TRAIN_SAMPLES=500K` discards ~85% of available data. HGBR handles large datasets; raise to 2M+ or remove cap.
- **Hyperparameter tuning** — current params are untouched defaults. Lower LR (0.01) + more iterations (2000+), and sweep `max_leaf_nodes`/`max_depth`/`min_samples_leaf`.
- **Distance-weighted neighbor features** — current unweighted mean treats a 0.5km neighbor the same as 15km. Inverse-distance weighting is more physical.
- **Clip humidity predictions** — 2.4% out of [0,100] bounds. Post-prediction clip or HGBR `monotonic_cst`.

_Medium impact:_
- **Wind direction decomposition** — worst-performing variable (RMSE 54.6°, R² 0.630). Decompose into u/v components, train two models, reconstruct via atan2. Applies to target and lag/ERA5 wind features.
- **Neighbor spread features** — add `neighbor_std_{var}` alongside mean. High spread signals microclimate boundaries (e.g., fog edge).
- **Coastal distance × BLH interaction** — encodes "how far inland does the marine layer reach right now."

_Lower priority:_
- **Lag feature gap handling** — `shift(lag_h)` assumes consecutive hourly rows; stations with gaps get incorrect lags.
- **Per-variable model params** — precip (skewed, mostly zero) and wind_dir (circular) need different configs than temp/humidity.

### Stage 2: LSTM for temperature and humidity [not implemented]

Once Stage 1 is working, wrap an LSTM around temperature and humidity specifically. These two have strong diurnal cycles and sequential memory — yesterday's afternoon temperature predicts tonight's low more reliably than spatial features alone. Wind and precipitation are more event-driven and probably don't benefit as much from temporal modeling.

- Input sequence: 24-hour rolling window of observations + ERA5 features per station
- Terrain features concatenated as static context at each timestep (not part of the sequence)
- Minimum ~6 months of clean per-station history needed. We have 13 months.
- Requires restructuring chunked Synoptic data into per-station continuous time series with gap handling.

### Stage 3: Spatial interpolation to prediction grid [not implemented]

Use trained Stage 1/2 model to predict at arbitrary lat/lon points across the Bay Area using only ERA5 + terrain features as inputs (no Synoptic observations needed — the model has learned to predict from spatial/atmospheric features alone). The prediction grid is defined by us (e.g., a regular lat/lon mesh at whatever resolution we choose), not tied to any external data source. This produces a continuous weather map across the Bay Area.

This is the microclimate parcellation step — the predicted patterns across the grid are what we cluster to define zones. Zones emerge from learned weather behavior, not raw geography. Sunset and Mission naturally separate because their predicted fog frequency and diurnal patterns differ.

Supersedes the original pre-clustering approach (`src/cluster_zones.py`, retained for baseline comparison).

---

## Schema

**Synoptic observation schema (2026-03-02)**
All Synoptic observation parquets use columns: `datetime`, `temp_c`, `humidity`, `wind_speed_kph`, `wind_dir_deg`, `precip_mm`, `stid`, `name`, `lat`, `lon`, `elev_m`, `network`, `source`. ERA5 reanalysis uses its own schema with `grid_lat`/`grid_lon`.

**Elevation: meters throughout (2026-03-02)**
Synoptic reports in feet, converted on ingest (`* 0.3048`). ERA5 grid points have no explicit elevation field. All downstream code assumes `elev_m` is in meters where present.

---

## Secrets Management

**Pattern: environment variables only (2026-03-02)**
All secrets (API tokens, keys) are passed via environment variables. No secrets are hardcoded in any script. Current variables:
- `SYNOPTIC_TOKEN` — Synoptic API token (not the API key; generate from customer.synopticdata.com)
- ~~`WU_API_KEY`~~ — removed (WU pipeline abandoned; see "Removed data sources")
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

### Decided

- [x] **Multi-year data strategy** — 10 years of ERA5 for ENSO coverage; Synoptic 1-year as primary training data (real observations).
- [x] **ML architecture** — 3-stage pipeline: GBT per variable → LSTM for temp/humidity → spatial interpolation to prediction grid. Not Geo-LSTM-Kriging.
- [x] **ERA5 role** — Upgraded from background context to primary model input. `boundary_layer_height` and `temp_850hPa` are key fog dynamics features.
- [x] **Open-Meteo dense grid removed** — Model-interpolated data would teach the model to replicate another model's assumptions. Synoptic stations + terrain features provide sufficient coverage with real observations.

### Stage 1 data gaps (all resolved)

| Data | Stage | Status |
|------|-------|--------|
| Complete Synoptic chunks | 1 | Done — 13 months, ~23 chunks/month |
| ERA5 → Synoptic station interpolation | 1 | Done — bilinear interp, `features/era5_at_stations/` |
| DEM terrain features at stations (elevation, slope, aspect) | 1 | Done — Open-Meteo Elevation API, `features/static/terrain_features.parquet` |
| NLCD land cover at stations | 1 | Done — MRLC WMS GetFeatureInfo, `features/static/land_cover.parquet` |
| Synoptic lag features (t-1hr, t-3hr, t-6hr) | 1 | Done — computed in `build_training_set.py` |
| Neighboring station features (mean of 5 nearest) | 1 | Done — vectorized pivot approach in `build_training_set.py` |

### Station QC: three mechanistically distinct failure modes (2026-03-12)

Automated QC on the 1,124 Synoptic stations identified 41 bad stations (3.6%) via
physical range thresholds: temp outside [−20, 50]°C, humidity outside [0, 100]%, or
within-station std > 15°C. Investigation revealed three distinct root causes — not
undifferentiated noise — each requiring a different response.

**Mode 1 — Hardware failures (excluded):** JEPC1, OAMC1, PRWC1, SFXC1, BINC1,
GGBC1, PTRCA, and others. Mid-record sensor failures producing physically impossible
values (temp 1243°C, humidity 4108%). Valid portions of the record are usable in
principle but indistinguishable from corrupt portions without manual inspection.
Decision: exclude entirely.

**Mode 2 — ASOS hygrometer over-reads (clipped, kept):** KDVO, KHAF, KRHV, KPAO,
KSNS, KSQL. Airport ASOS stations with overwhelmingly valid data but rare humidity
readings of 101–115%. Occurs during coastal fog saturation events when capacitive
hygrometers saturate slightly above 100%. Affected readings: <2% of station records.
Decision: clip humidity to [0, 100] in ingestion pipeline. These are valuable stations
with long, otherwise reliable records.

**Mode 3 — Consistent Fahrenheit submission (investigated, excluded):** UP641, UP657,
UP667, UP680, UR476, UR481, UR604. Maximum temperatures of 63–110°C consistently
convert to plausible Bay Area values under (F−32)×5/9. However, cross-month audit
revealed these stations also produce implausible minimums (−45 to −17°C) that do
not resolve with Fahrenheit conversion. The Fahrenheit issue co-occurs with
intermittent sensor glitches, making the record unsalvageable without manual
per-timestamp unit detection. Decision: exclude entirely.

The QC blocklist lives in `src/build_training_set.py: BAD_STATIONS`. The ASOS clip
is applied at ingestion in the same file. Net result: 582/623 active stations
(93.4%) retained after QC.

### Later stages

| Data | Stage | Status |
|------|-------|--------|
| Per-station continuous time series with gap handling | 2 | Derive from chunked Synoptic data |
| Define prediction grid (regular lat/lon mesh) | 3 | Choose resolution, generate grid points |
| ERA5 interpolated to prediction grid points | 3 | Derive from existing ERA5 grid |
| DEM + NLCD at prediction grid points | 3 | Not collected — same sources as Stage 1 |

### Modeling decisions (after data gaps filled)

- [x] **Define train/val/test split strategy** — temporal split: last 2 months (Feb–Mar 2026) as test, remainder as train
- [x] **Train Stage 1 GBT** — separate models per variable, evaluated with baseline comparison, spatial/temporal error, cold-start, fog stratification, and climatological plausibility checks
- [ ] **Stage 2: LSTM for temp/humidity** — 24hr rolling window, static terrain context
- [ ] **Stage 3: predict at prediction grid, cluster into zones** — microclimate parcellation

---

## References

- Han, S. et al. — "Geo-LSTM-Kriging: A spatiotemporal deep learning approach for temperature interpolation." Warsaw mesoscale evaluation: Kriging RMSE 3.0C, R2 0.58. Informed the decision to reject Kriging at neighborhood scale and the absence of vertical atmospheric structure (marine layer height, inversion) as a gap for Bay Area fog modeling.
- Open-Meteo Archive API — https://open-meteo.com/en/docs/historical-weather-api. Source for ERA5 reanalysis data. Free, no API key required.
- Synoptic Data API — https://docs.synopticdata.com/services/time-series. Source for ground-truth station observations. Free Open Access tier capped at 1 year of history.
- ERA5 reanalysis (Hersbach et al., 2020) — ECMWF's fifth-generation global atmospheric reanalysis, 0.25 degree resolution. Accessed via Open-Meteo. Key variables: `boundary_layer_height`, `temperature_850hPa`.
- SRTM 30m DEM — https://earthexplorer.usgs.gov/. Planned source for elevation, slope, aspect terrain features.
- NLCD (National Land Cover Database) — https://www.mrlc.gov/. Planned source for land cover classification (urban/vegetation/water).
