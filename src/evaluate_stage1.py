"""
Stage 1 evaluation: tests the GBT models beyond aggregate RMSE/R2.

Runs five evaluations:
  1. Baseline comparison: GBT vs persistence vs ERA5-direct vs climatological mean
  2. Spatial error: per-station RMSE ranked worst to best
  3. Temporal error: RMSE by hour of day and month
  4. Cold-start: GBT without lag features (spatial-only, simulates stale observations)
  5. Fog-event performance: errors stratified by boundary_layer_height_m quartile

Prints a concise report and saves CSV results to models/stage1_v3/eval/.

Usage:
    python src/evaluate_stage1.py
"""

import os
import sys
import logging
from io import BytesIO

import numpy as np
import pandas as pd
import joblib
import boto3
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import S3_BUCKET

# ── Config ────────────────────────────────────────────────────────────────────

# Only evaluate the two most microclimate-relevant variables
EVAL_VARS = ["temp_c", "humidity"]

TEST_MONTHS = ["2026-02", "2026-03"]

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/evaluate_stage1.log"),
    ],
)
log = logging.getLogger(__name__)

# ── S3 helpers ────────────────────────────────────────────────────────────────

s3 = boto3.client("s3")


def load_parquet_from_s3(key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))

# ── Metrics ──────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return np.nan
    return np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))


def r2(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return np.nan
    return r2_score(y_true[mask], y_pred[mask])


def mae(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("models/stage1_v3/eval", exist_ok=True)

    # Load test data
    log.info("Loading test data...")
    frames = []
    for month in TEST_MONTHS:
        key = f"features/training_v3/{month}.parquet"
        try:
            df = load_parquet_from_s3(key)
            df["month"] = month
            frames.append(df)
            log.info(f"  {key}: {len(df):,} rows")
        except Exception as e:
            log.warning(f"  Could not load {key}: {e}")

    if not frames:
        log.error("No test data found.")
        return

    data = pd.concat(frames, ignore_index=True)
    data["datetime"] = pd.to_datetime(data["datetime"], utc=True)
    data["hour"] = data["datetime"].dt.hour
    log.info(f"Total test rows: {len(data):,}")

    for var in EVAL_VARS:
        log.info(f"\n{'='*65}")
        log.info(f"Evaluating: {var}")
        log.info(f"{'='*65}")

        # Load model
        model_path = f"models/stage1_v3/{var}_model.joblib"
        if not os.path.exists(model_path):
            log.error(f"  Model not found: {model_path}")
            continue

        bundle = joblib.load(model_path)
        model = bundle["model"]
        features = bundle["features"]

        # Filter valid rows
        valid = data.dropna(subset=[var]).copy()
        y_true = valid[var].values

        # ── 1. Baseline comparison ───────────────────────────────────────────
        log.info("\n1. Baseline comparison:")

        # GBT prediction
        valid_features = [f for f in features if f in valid.columns]
        X = valid[valid_features].values.astype(np.float32)
        y_gbt = model.predict(X)

        # Clip humidity predictions to physical bounds [0, 100]
        if var == "humidity":
            y_gbt = np.clip(y_gbt, 0.0, 100.0)

        # Persistence baseline (1h lag)
        lag_col = f"{var}_lag1h"
        y_persist = valid[lag_col].values if lag_col in valid.columns else np.full(len(valid), np.nan)

        # ERA5-direct baseline (ERA5 equivalent column)
        era5_col = "temp_2m_c" if var == "temp_c" else "humidity_2m"
        y_era5 = valid[era5_col].values if era5_col in valid.columns else np.full(len(valid), np.nan)

        # Climatological mean (per-station hourly mean from training data -- use overall mean as proxy)
        y_clim = np.full(len(valid), np.nanmean(y_true))

        baselines = {
            "GBT (our model)": y_gbt,
            "Persistence (lag 1h)": y_persist,
            "ERA5 direct": y_era5,
            "Climatological mean": y_clim,
        }

        log.info(f"  {'Baseline':<25s}  {'RMSE':>8s}  {'MAE':>8s}  {'R2':>8s}")
        log.info(f"  {'-'*53}")
        baseline_rows = []
        for name, y_pred in baselines.items():
            r = rmse(y_true, y_pred)
            m = mae(y_true, y_pred)
            r2_val = r2(y_true, y_pred)
            log.info(f"  {name:<25s}  {r:>8.4f}  {m:>8.4f}  {r2_val:>8.4f}")
            baseline_rows.append({"variable": var, "baseline": name,
                                   "rmse": r, "mae": m, "r2": r2_val})

        pd.DataFrame(baseline_rows).to_csv(
            f"models/stage1_v3/eval/{var}_baselines.csv", index=False)

        skill_score = 1 - rmse(y_true, y_gbt) / rmse(y_true, y_persist)
        log.info(f"\n  Skill score vs persistence: {skill_score:.3f}  "
                 f"({'improvement' if skill_score > 0 else 'worse than persistence'})")

        # ── 2. Spatial error (per-station RMSE) ──────────────────────────────
        log.info("\n2. Spatial error (per-station RMSE, worst 10 / best 10):")

        valid["y_true"] = y_true
        valid["y_pred"] = y_gbt
        station_rmse = (valid
                        .groupby("stid")
                        .apply(lambda g: pd.Series({
                            "rmse": rmse(g["y_true"].values, g["y_pred"].values),
                            "r2": r2(g["y_true"].values, g["y_pred"].values),
                            "n": len(g),
                            "lat": g["lat"].iloc[0] if "lat" in g.columns else np.nan,
                            "lon": g["lon"].iloc[0] if "lon" in g.columns else np.nan,
                            "dist_coast_km": g["dist_coast_km"].iloc[0] if "dist_coast_km" in g.columns else np.nan,
                        }))
                        .reset_index()
                        .dropna(subset=["rmse"])
                        .sort_values("rmse", ascending=False))

        log.info(f"  {'Station':<12s}  {'RMSE':>8s}  {'R2':>8s}  {'N':>6s}  {'Dist Coast':>10s}")
        log.info(f"  {'-'*54}")
        log.info(f"  -- Worst 10 --")
        for _, row in station_rmse.head(10).iterrows():
            log.info(f"  {row['stid']:<12s}  {row['rmse']:>8.3f}  {row['r2']:>8.3f}  {int(row['n']):>6d}  {row['dist_coast_km']:>10.1f}")
        log.info(f"  -- Best 10 --")
        for _, row in station_rmse.tail(10).iterrows():
            log.info(f"  {row['stid']:<12s}  {row['rmse']:>8.3f}  {row['r2']:>8.3f}  {int(row['n']):>6d}  {row['dist_coast_km']:>10.1f}")

        station_rmse.to_csv(f"models/stage1_v3/eval/{var}_station_rmse.csv", index=False)

        # Correlation: does RMSE correlate with coastal distance?
        corr = station_rmse[["rmse", "dist_coast_km"]].dropna().corr().iloc[0, 1]
        log.info(f"\n  RMSE vs dist_coast_km correlation: {corr:.3f}")

        # ── 3. Temporal error ────────────────────────────────────────────────
        log.info("\n3. Temporal error (RMSE by hour of day):")

        hourly = (valid
                  .groupby("hour")
                  .apply(lambda g: rmse(g["y_true"].values, g["y_pred"].values))
                  .reset_index()
                  .rename(columns={0: "rmse"}))

        worst_hour = hourly.loc[hourly["rmse"].idxmax()]
        best_hour = hourly.loc[hourly["rmse"].idxmin()]
        log.info(f"  Best hour:  {int(best_hour['hour']):02d}:00  RMSE={best_hour['rmse']:.4f}")
        log.info(f"  Worst hour: {int(worst_hour['hour']):02d}:00  RMSE={worst_hour['rmse']:.4f}")
        hourly.to_csv(f"models/stage1_v3/eval/{var}_hourly_rmse.csv", index=False)

        # ── 4. Cold-start: no lag features ───────────────────────────────────
        log.info("\n4. Cold-start performance (no lag features):")

        lag_feature_cols = [f for f in valid_features if "_lag" in f]
        no_lag_features = [f for f in valid_features if "_lag" not in f]
        log.info(f"  Removing {len(lag_feature_cols)} lag features, keeping {len(no_lag_features)}")

        X_no_lag = valid[no_lag_features].values.astype(np.float32)
        # Rebuild a model without lag features to get a fair comparison
        # Instead, just zero out the lag features in the input
        X_zero_lag = X.copy()
        for col in lag_feature_cols:
            if col in valid_features:
                idx = valid_features.index(col)
                X_zero_lag[:, idx] = np.nan  # NaN = HGBR treats as missing

        y_cold = model.predict(X_zero_lag)
        cold_rmse = rmse(y_true, y_cold)
        cold_r2 = r2(y_true, y_cold)
        full_rmse = rmse(y_true, y_gbt)

        log.info(f"  Full model RMSE:      {full_rmse:.4f}")
        log.info(f"  Cold-start RMSE:      {cold_rmse:.4f}")
        log.info(f"  Degradation factor:   {cold_rmse / full_rmse:.2f}x  "
                 f"({'spatial features still useful' if cold_rmse < rmse(y_true, y_clim) else 'worse than climatology'})")

        pd.DataFrame([{
            "variable": var,
            "full_rmse": full_rmse,
            "cold_start_rmse": cold_rmse,
            "degradation_factor": cold_rmse / full_rmse,
            "clim_rmse": rmse(y_true, y_clim),
        }]).to_csv(f"models/stage1_v3/eval/{var}_cold_start.csv", index=False)

        # ── 5. Fog-event performance ─────────────────────────────────────────
        log.info("\n5. Fog-event performance (stratified by BLH quartile):")

        if "boundary_layer_height_m" in valid.columns:
            blh = valid["boundary_layer_height_m"]
            q25, q50, q75 = blh.quantile([0.25, 0.5, 0.75])
            log.info(f"  BLH quartiles: {q25:.0f}m / {q50:.0f}m / {q75:.0f}m")

            strata = {
                f"Low BLH <{q25:.0f}m (fog/marine layer)": blh <= q25,
                f"Mid-low BLH {q25:.0f}-{q50:.0f}m": (blh > q25) & (blh <= q50),
                f"Mid-high BLH {q50:.0f}-{q75:.0f}m": (blh > q50) & (blh <= q75),
                f"High BLH >{q75:.0f}m (clear)": blh > q75,
            }

            fog_rows = []
            log.info(f"  {'Stratum':<40s}  {'N':>7s}  {'RMSE':>8s}  {'R2':>8s}")
            log.info(f"  {'-'*67}")
            for stratum_name, mask in strata.items():
                n = mask.sum()
                if n < 100:
                    continue
                r_val = rmse(y_true[mask], y_gbt[mask])
                r2_val = r2(y_true[mask], y_gbt[mask])
                log.info(f"  {stratum_name:<40s}  {n:>7,d}  {r_val:>8.4f}  {r2_val:>8.4f}")
                fog_rows.append({"variable": var, "stratum": stratum_name,
                                  "n": n, "rmse": r_val, "r2": r2_val})

            pd.DataFrame(fog_rows).to_csv(
                f"models/stage1_v3/eval/{var}_fog_strata.csv", index=False)
        else:
            log.warning("  boundary_layer_height_m not found in test data")

        # ── 6. Climatological plausibility check ─────────────────────────────
        log.info("\n6. Climatological plausibility check:")

        # Bay Area physical bounds (hard limits -- values outside are physically impossible)
        PLAUSIBILITY_BOUNDS = {
            "temp_c":          (-5.0,  45.0),
            "humidity":        (0.0,  100.0),
            "wind_speed_kph":  (0.0,  120.0),
            "wind_dir_deg":    (0.0,  360.0),
            "precip_mm":       (0.0,   50.0),
        }

        lo, hi = PLAUSIBILITY_BOUNDS.get(var, (-np.inf, np.inf))
        n_total = len(y_gbt)
        n_out_lo = int(np.sum(y_gbt < lo))
        n_out_hi = int(np.sum(y_gbt > hi))
        n_out = n_out_lo + n_out_hi
        pct_out = 100.0 * n_out / n_total if n_total > 0 else 0.0

        log.info(f"  Physical bounds: [{lo}, {hi}]")
        log.info(f"  Predictions out of bounds: {n_out:,} / {n_total:,}  ({pct_out:.3f}%)")
        if n_out_lo:
            log.info(f"    Below {lo}: {n_out_lo:,}  (min predicted = {y_gbt.min():.3f})")
        if n_out_hi:
            log.info(f"    Above {hi}: {n_out_hi:,}  (max predicted = {y_gbt.max():.3f})")

        # Distribution comparison: predicted vs observed
        log.info(f"\n  Distribution comparison (predicted vs observed):")
        stats_rows = []
        for label, vals in [("Observed", y_true), ("Predicted", y_gbt)]:
            mask_v = ~np.isnan(vals)
            v = vals[mask_v]
            p5, p25, p50, p75, p95 = np.percentile(v, [5, 25, 50, 75, 95])
            log.info(f"  {label:<12s}  mean={np.mean(v):>7.3f}  std={np.std(v):>6.3f}  "
                     f"p5={p5:>7.3f}  p25={p25:>7.3f}  p50={p50:>7.3f}  "
                     f"p75={p75:>7.3f}  p95={p95:>7.3f}")
            stats_rows.append({
                "variable": var, "split": label,
                "mean": np.mean(v), "std": np.std(v),
                "p5": p5, "p25": p25, "p50": p50, "p75": p75, "p95": p95,
                "n_out_of_bounds": n_out if label == "Predicted" else 0,
                "pct_out_of_bounds": pct_out if label == "Predicted" else 0.0,
            })

        # Bias check
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_gbt))
        bias = float(np.mean(y_gbt[valid_mask] - y_true[valid_mask]))
        log.info(f"\n  Mean bias (predicted - observed): {bias:+.4f}  "
                 f"({'over-predict' if bias > 0 else 'under-predict'})")

        pd.DataFrame(stats_rows).to_csv(
            f"models/stage1_v3/eval/{var}_plausibility.csv", index=False)

    # ── Wind direction: reconstruct from u/v models and evaluate ─────────────
    log.info(f"\n{'='*65}")
    log.info("Wind direction (reconstructed from wind_dir_u / wind_dir_v models)")
    log.info(f"{'='*65}")

    u_path = "models/stage1_v3/wind_dir_u_model.joblib"
    v_path = "models/stage1_v3/wind_dir_v_model.joblib"
    if os.path.exists(u_path) and os.path.exists(v_path) and "wind_dir_deg" in data.columns:
        u_bundle = joblib.load(u_path)
        v_bundle = joblib.load(v_path)

        valid_wd = data.dropna(subset=["wind_dir_deg"]).copy()
        y_true_deg = valid_wd["wind_dir_deg"].values

        u_feats = [f for f in u_bundle["features"] if f in valid_wd.columns]
        v_feats = [f for f in v_bundle["features"] if f in valid_wd.columns]
        u_pred = u_bundle["model"].predict(valid_wd[u_feats].values.astype(np.float32))
        v_pred = v_bundle["model"].predict(valid_wd[v_feats].values.astype(np.float32))

        # Reconstruct angle from unit vector predictions
        y_pred_deg = (np.degrees(np.arctan2(-u_pred, -v_pred))) % 360

        # Circular RMSE: shortest angular distance between predicted and observed
        diff = np.abs(y_pred_deg - y_true_deg)
        diff = np.minimum(diff, 360 - diff)
        circ_rmse = float(np.sqrt(np.mean(diff ** 2)))
        circ_mae = float(np.mean(diff))
        log.info(f"  Circular RMSE: {circ_rmse:.2f}°   Circular MAE: {circ_mae:.2f}°")
        log.info(f"  (v2 raw angle RMSE was 54.6° for reference)")

        pd.DataFrame([{"circ_rmse": circ_rmse, "circ_mae": circ_mae}]).to_csv(
            "models/stage1_v3/eval/wind_dir_circular.csv", index=False)
    else:
        log.warning("  wind_dir_u/v models not found or wind_dir_deg missing from test data")

    log.info(f"\n{'='*65}")
    log.info("Evaluation complete. Results in models/stage1_v3/eval/")
    log.info(f"{'='*65}")


if __name__ == "__main__":
    main()
