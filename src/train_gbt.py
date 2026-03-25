"""
Stage 1: Train gradient-boosted tree models (one per target variable).

Loads the unified training set from S3, splits by time (last 2 months = test),
trains HistGradientBoostingRegressor for each of: temp_c, humidity,
wind_speed_kph, wind_dir_deg, precip_mm. Reports per-variable RMSE and R-squared.

Memory-efficient: loads data per-variable, subsamples training set if needed.

Usage:
    python src/train_gbt.py

S3 layout:
    Input:
        features/training/YYYY-MM.parquet
    Output:
        models/stage1_v3/{variable}_model.joblib  (local)
"""

import os
import sys
import gc
import logging
from io import BytesIO

import numpy as np
import pandas as pd
import boto3
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import S3_BUCKET

# ── Config ────────────────────────────────────────────────────────────────────

TARGET_VARS = ["temp_c", "humidity", "wind_speed_kph", "wind_dir_u", "wind_dir_v", "precip_mm"]

# Feature columns to exclude (targets, join keys, metadata).
# wind_dir_deg is excluded because it is replaced by wind_dir_u/v targets.
EXCLUDE_COLS = [
    "datetime", "stid", "name", "network", "source",
    "nlcd_code", "nlcd_name", "month",
    "wind_dir_deg",
] + TARGET_VARS

# Time-based train/test split: last 2 months = test
TEST_MONTHS = 2

# Training sample cap. None = use all data.
# HGBR handles large datasets well; removing the previous 500K cap.
MAX_TRAIN_SAMPLES = None

# Default HistGradientBoostingRegressor parameters.
# Lower LR + more iterations generalises better with larger training sets.
MODEL_PARAMS = {
    "learning_rate": 0.01,
    "max_iter": 2000,
    "max_leaf_nodes": 63,
    "max_depth": 8,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "n_iter_no_change": 50,
    "validation_fraction": 0.1,
    "random_state": 42,
    "verbose": 0,
}

# Per-variable overrides applied on top of MODEL_PARAMS.
# precip_mm: heavily zero-inflated, needs stronger regularisation and shallower trees.
# wind_dir_u/v: unit vectors in [-1, 1], benefit from tighter leaf size.
MODEL_PARAM_OVERRIDES = {
    "precip_mm": {"max_depth": 6, "max_leaf_nodes": 31, "l2_regularization": 5.0},
    "wind_dir_u": {"min_samples_leaf": 30},
    "wind_dir_v": {"min_samples_leaf": 30},
}

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/train_gbt.log"),
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

# ── Feature selection ────────────────────────────────────────────────────────

def get_feature_columns(columns: list[str], target: str) -> list[str]:
    """Select numeric feature columns for a given target variable."""
    other_targets = [t for t in TARGET_VARS if t != target]
    exclude = set(EXCLUDE_COLS) | set(other_targets)

    features = []
    for col in columns:
        if col in exclude:
            continue
        if col.startswith("neighbor_mean_") and col != f"neighbor_mean_{target}":
            if col not in ["neighbor_mean_temp_c", "neighbor_mean_humidity"]:
                continue
        features.append(col)

    return features

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("models/stage1_v3", exist_ok=True)

    # Discover training files and determine train/test split
    train_keys = sorted(list_s3_keys("features/training_v3/"))
    if not train_keys:
        log.error("No training data found. Run build_training_set.py first.")
        return

    months = [k.split("/")[-1].replace(".parquet", "") for k in train_keys]
    test_month_names = months[-TEST_MONTHS:]
    train_month_names = months[:-TEST_MONTHS]

    train_file_keys = [k for k, m in zip(train_keys, months) if m in train_month_names]
    test_file_keys = [k for k, m in zip(train_keys, months) if m in test_month_names]

    log.info(f"Train months: {train_month_names[0]} to {train_month_names[-1]} "
             f"({len(train_month_names)} months)")
    log.info(f"Test months:  {test_month_names[0]} to {test_month_names[-1]} "
             f"({len(test_month_names)} months)")

    results = []

    for target in TARGET_VARS:
        log.info(f"\n{'='*60}")
        log.info(f"Training model for: {target}")
        log.info(f"{'='*60}")

        # Load training data incrementally, keeping only needed columns
        # First pass: determine feature columns from first file
        sample_df = load_parquet_from_s3(train_file_keys[0])
        all_cols = sample_df.columns.tolist()

        # Filter to numeric columns only
        numeric_cols = [c for c in all_cols
                        if sample_df[c].dtype in (np.float64, np.float32,
                                                   np.int64, np.int32,
                                                   np.uint8, np.bool_)]
        feature_cols = get_feature_columns(numeric_cols, target)
        keep_cols = feature_cols + [target]
        del sample_df
        gc.collect()

        log.info(f"  {len(feature_cols)} features")

        # Load train data (only needed columns)
        train_frames = []
        for key in train_file_keys:
            df = load_parquet_from_s3(key)
            valid_keep = [c for c in keep_cols if c in df.columns]
            train_frames.append(df[valid_keep])
            del df

        train_df = pd.concat(train_frames, ignore_index=True)
        del train_frames
        gc.collect()

        # Drop NaN targets
        train_df = train_df.dropna(subset=[target])

        # Subsample if cap is set
        if MAX_TRAIN_SAMPLES is not None and len(train_df) > MAX_TRAIN_SAMPLES:
            log.info(f"  Subsampling train: {len(train_df):,} -> {MAX_TRAIN_SAMPLES:,}")
            train_df = train_df.sample(n=MAX_TRAIN_SAMPLES, random_state=42)

        valid_features = [c for c in feature_cols if c in train_df.columns]
        X_train = train_df[valid_features].values.astype(np.float32)
        y_train = train_df[target].values.astype(np.float32)
        del train_df
        gc.collect()

        log.info(f"  Train: {len(X_train):,} rows")

        # Load test data
        test_frames = []
        for key in test_file_keys:
            df = load_parquet_from_s3(key)
            valid_keep = [c for c in keep_cols if c in df.columns]
            test_frames.append(df[valid_keep])
            del df

        test_df = pd.concat(test_frames, ignore_index=True)
        del test_frames
        gc.collect()

        test_df = test_df.dropna(subset=[target])
        X_test = test_df[valid_features].values.astype(np.float32)
        y_test = test_df[target].values.astype(np.float32)
        del test_df
        gc.collect()

        log.info(f"  Test: {len(X_test):,} rows")

        # Train (apply any per-variable parameter overrides)
        params = {**MODEL_PARAMS, **MODEL_PARAM_OVERRIDES.get(target, {})}
        model = HistGradientBoostingRegressor(**params)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        log.info(f"\n  Results for {target}:")
        log.info(f"    RMSE: {rmse:.4f}")
        log.info(f"    R2:   {r2:.4f}")
        log.info(f"    Iterations: {model.n_iter_}")

        results.append({"variable": target, "rmse": rmse, "r2": r2,
                        "n_iter": model.n_iter_})

        # Permutation importance on a subsample of test data
        log.info("  Computing feature importance...")
        n_imp = min(10_000, len(X_test))
        imp_idx = np.random.RandomState(42).choice(len(X_test), n_imp, replace=False)
        perm_result = permutation_importance(
            model, X_test[imp_idx], y_test[imp_idx],
            n_repeats=5, random_state=42, n_jobs=1,
        )
        importance = pd.DataFrame({
            "feature": valid_features,
            "importance": perm_result.importances_mean,
        }).sort_values("importance", ascending=False)

        log.info(f"\n  Top 15 features (permutation importance):")
        for _, row in importance.head(15).iterrows():
            log.info(f"    {row['feature']:40s}  {row['importance']:.4f}")

        # Save model
        model_path = f"models/stage1_v3/{target}_model.joblib"
        joblib.dump({"model": model, "features": valid_features}, model_path)
        log.info(f"  Saved model to {model_path}")

        # Save feature importance
        importance.to_csv(f"models/stage1_v3/{target}_importance.csv", index=False)

        # Free memory before next variable
        del X_train, y_train, X_test, y_test, y_pred, model
        gc.collect()

    # Summary table
    log.info(f"\n{'='*60}")
    log.info("Stage 1 GBT Results Summary")
    log.info(f"{'='*60}")
    log.info(f"{'Variable':<20s} {'RMSE':>10s} {'R2':>10s} {'Iterations':>10s}")
    log.info("-" * 52)
    for r in results:
        log.info(f"{r['variable']:<20s} {r['rmse']:>10.4f} {r['r2']:>10.4f} {r['n_iter']:>10d}")

    log.info("\nStage 1 training complete.")


if __name__ == "__main__":
    main()
