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

### Open-Meteo Dense Grid [not yet run] (replaces Weather Underground)

**Pivot from WU to Open-Meteo (2026-03-10)**
The Weather Underground pipeline (`src/download_wunderground.py`) was abandoned because WU's API access model is unreliable — it requires registering a physical PWS device, the API key issuance process is opaque, and the APIs have known issues. Open-Meteo's free archive API provides a cleaner replacement with several advantages:
- No API key or device registration required
- Consistent data quality (model-interpolated, not noisy backyard sensors)
- 20+ years of history available (not limited like Synoptic free tier)
- Deterministic grid coverage (no station discovery step, no gaps)

**Status (2026-03-10)**
`src/download_open_meteo.py` is written and ready to run. Follows the same pattern as `download_era5.py` but at much higher spatial resolution. Has not been executed yet.

**Grid resolution: 0.05° (~5km) (2026-03-10)**
Dense grid at 0.05° spacing gives ~900 grid points across the Bay Area bbox. This is 25x denser than the ERA5 grid (36 points at 0.25°) and provides fine-grained spatial variation needed for microclimate modeling. The underlying ERA5 reanalysis is 0.25° native, so Open-Meteo interpolates — but the interpolated values still capture local terrain effects through the model's orography.

**Variables: surface-level observations (2026-03-10)**
`temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `wind_direction_10m`, `precipitation`, `cloud_cover`, `surface_pressure`. These complement ERA5's synoptic-scale variables (boundary layer height, 850hPa temperature) with local surface detail.

**S3 layout: per-grid-point per month (2026-03-10)**
`raw/open_meteo/monthly/YYYY-MM/grid_{lat}_{lon}.parquet` — matches the ERA5 layout for consistency. Resumable via S3 key existence checks.

**Tradeoff vs. real station observations (2026-03-10)**
Open-Meteo dense grid data is model output, not direct observations. It will not capture hyper-local effects (street-level heat islands, building shadows, irrigation cooling) that real PWS data would. However, the consistent quality and coverage make it a better foundation for the model-first workflow — the model can learn spatial patterns from the dense grid, and Synoptic ASOS stations provide ground-truth calibration.

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

## Modeling & Zone Definition Strategy

**Revised approach: model-first, cluster-second (2026-03-10)**
The original plan was to cluster stations into microclimate zones upfront and then train per-zone or zone-aware models. This has been replaced with a model-first workflow that lets the data define zones rather than imposing them a priori:

1. **Train model with spatial features as inputs.** The model receives elevation, coastal distance, bay distance, slope aspect, terrain exposure, and other geographic features alongside weather observations. It learns how spatial features relate to weather outcomes directly, without needing predefined zones.

2. **Extract learned representations.** After training, extract the model's internal embeddings or evaluate predicted weather behavior across a dense spatial grid. This produces a "weather profile" at every point — predicted fog frequency, temperature variance, diurnal patterns, etc.

3. **Cluster on predicted profiles.** Apply clustering (K-means or other) to the model-derived weather profiles, not raw geography. Zones emerge from learned weather behavior, so areas with similar predicted microclimates group together naturally. For example, Sunset and Mission would separate because their predicted fog frequency and diurnal patterns differ, even though they're geographically close.

**Why this supersedes pre-clustering (2026-03-10)**
Pre-clustering on raw observations + static features has several weaknesses:
- Requires choosing K before seeing model performance
- Clusters are constrained by input feature engineering (e.g., the heuristic `coastal_exposure`)
- Geographically close but climatologically different areas (Sunset vs. Mission) may not separate without carefully engineered features
- Model-derived zones adapt automatically as more data sources (WU) are added — no need to manually re-run clustering

**Status of existing `cluster_zones.py` (2026-03-10)**
`src/cluster_zones.py` remains in the codebase and may still be useful for exploratory analysis or as a baseline comparison against model-derived zones. Its original design decisions are preserved below for reference.

### Legacy: Pre-clustering Design (reference only)

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

**`source` column (2026-03-02, updated 2026-03-10)**
All observation parquets include `source` (`"synoptic"` or `"open_meteo"`). Allows filtering, differential weighting, or source-specific validation downstream. ERA5 reanalysis uses its own schema with `grid_lat`/`grid_lon`.

**Shared schema across sources (2026-03-02, updated 2026-03-10)**
Both Synoptic and Open-Meteo surface grid use identical columns: `datetime`, `temp_c`, `humidity`, `wind_speed_kph`, `wind_dir_deg`, `precip_mm`, `stid`, `name`, `lat`, `lon`, `elev_m`, `network`, `source`. Open-Meteo grid points use synthetic station IDs (`OM_{lat}_{lon}`) and include DEM-derived elevation. All training code treats sources uniformly.

**Elevation: meters throughout (2026-03-02)**
Synoptic reports in feet, converted on ingest (`* 0.3048`). Open-Meteo provides DEM-derived elevation in meters via the API response. ERA5 grid points have no explicit elevation field. All downstream code assumes `elev_m` is in meters where present.

---

## Secrets Management

**Pattern: environment variables only (2026-03-02)**
All secrets (API tokens, keys) are passed via environment variables. No secrets are hardcoded in any script. Current variables:
- `SYNOPTIC_TOKEN` — Synoptic API token (not the API key; generate from customer.synopticdata.com)
- ~~`WU_API_KEY`~~ — removed (WU pipeline replaced by Open-Meteo, which requires no key)
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

- [ ] **Run static features computation**
  Done when: `features/static/stations_with_features.parquet` exists with dist_coast, dist_bay, elev_m, coastal_exposure for all stations.

- [ ] **Decide on 20-year data strategy**
  Options: Synoptic Enterprise, NOAA ISD supplement for ASOS only, or accept 1-year scope. Decision needed before designing the seasonal component of the ML model.

- [ ] **Run Open-Meteo dense grid download**
  Done when: all ~900 grid points × 13 months exist in `raw/open_meteo/monthly/`.

- [ ] **Validate Open-Meteo dense grid against Synoptic ASOS**
  Done when: mean bias and RMSE of Open-Meteo grid points vs. co-located ASOS stations (KSFO, KOAK, KSJC) are computed and documented. Expect small bias since both use model/reanalysis data, but quantify it.

- [ ] **Define train/val/test split strategy**
  Options: temporal split (last 2 months = test), spatial split (hold out stations), stratified random. Choice affects how well the model generalizes to unseen times vs. unseen locations. Decision needed before any model training.

- [ ] **Train spatial-feature model (model-first workflow step 1)**
  Build model with spatial features (elevation, coastal distance, bay distance, terrain exposure, etc.) as inputs alongside weather observations. Done when: model achieves reasonable prediction skill on held-out stations.

- [ ] **Extract learned representations and define zones (model-first workflow steps 2-3)**
  Extract model embeddings or predicted weather profiles across a spatial grid. Cluster on predicted profiles to define microclimate zones. Done when: zone map is visually coherent and zones separate known microclimate boundaries (e.g., Sunset vs. Mission, coastal fog belt vs. inland heat).

- [ ] **Compare model-derived zones against legacy K-means baseline**
  Run `cluster_zones.py` as baseline. Compare zone maps qualitatively and quantitatively (e.g., silhouette score, within-zone temperature variance). Document which approach produces more coherent zones.

- [ ] **Re-evaluate zone resolution after Open-Meteo dense integration**
  With ~900 dense grid points providing continuous spatial coverage, zones should be sharper than with sparse station data alone. May increase K or adopt continuous spatial interpolation instead of discrete zones.
