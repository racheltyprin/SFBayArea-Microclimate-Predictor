"""
Shared utilities for Bay Area Microclimate data pipeline.
Handles S3 upload/check operations used by all download scripts.
"""

import boto3
import botocore.exceptions
from io import BytesIO
import pandas as pd

S3_BUCKET = "bay-area-microclimate"

s3 = boto3.client("s3")


def upload_df_to_s3(df: pd.DataFrame, key: str, log):
    """Upload a DataFrame as parquet to S3."""
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
    log.info(f"  Uploaded s3://{S3_BUCKET}/{key}  ({len(df):,} rows)")


def s3_key_exists(key: str) -> bool:
    """Return True if the S3 key exists (used to skip already-downloaded chunks)."""
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise
