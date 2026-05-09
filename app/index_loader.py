"""
Download/upload the FAISS index from/to S3.
Only runs when S3_INDEX_BUCKET environment variable is set.

Required env vars:
    S3_INDEX_BUCKET  — S3 bucket name
    S3_INDEX_PREFIX  — key prefix (default: "index")
                       Expected files: <prefix>/unified.index
                                       <prefix>/metadata.json
                                       <prefix>/index_config.json

AWS credentials are read from env vars or IAM role automatically.
"""

import os
from pathlib import Path


def download_index_from_s3(local_index_dir: str) -> None:
    """Download index files from S3. No-op if S3_INDEX_BUCKET is not set."""
    bucket = os.getenv("S3_INDEX_BUCKET")
    if not bucket:
        return

    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for S3 index loading. "
            "Add 'boto3' to requirements.txt."
        ) from exc

    prefix = os.getenv("S3_INDEX_PREFIX", "index").rstrip("/")
    local_dir = Path(local_index_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")

    files_to_download = [
        "unified.index",
        "metadata.json",
        "index_config.json",
    ]

    print(f"[index_loader] Downloading index from s3://{bucket}/{prefix}/ ...")

    for filename in files_to_download:
        s3_key = f"{prefix}/{filename}"
        local_path = local_dir / filename

        try:
            s3.download_file(bucket, s3_key, str(local_path))
            print(f"[index_loader]   ✓ {s3_key} → {local_path}")
        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            if error_code == "404" and filename == "index_config.json":
                # index_config.json is optional for older indexes
                print(f"[index_loader]   - {s3_key} not found (optional, skipping)")
            else:
                raise RuntimeError(
                    f"Failed to download s3://{bucket}/{s3_key}: {exc}"
                ) from exc

    print("[index_loader] Index download complete.")


def upload_index_to_s3(local_index_dir: str) -> None:
    """Upload index files to S3. No-op if S3_INDEX_BUCKET is not set."""
    bucket = os.getenv("S3_INDEX_BUCKET")
    if not bucket:
        return

    import boto3

    prefix = os.getenv("S3_INDEX_PREFIX", "index").rstrip("/")
    local_dir = Path(local_index_dir)

    s3 = boto3.client("s3")

    files_to_upload = [
        "unified.index",
        "metadata.json",
        "index_config.json",
    ]

    print(f"[index_loader] Uploading index to s3://{bucket}/{prefix}/ ...")

    for filename in files_to_upload:
        local_path = local_dir / filename
        if not local_path.exists():
            print(f"[index_loader]   - {filename} not found locally, skipping")
            continue

        s3_key = f"{prefix}/{filename}"
        s3.upload_file(str(local_path), bucket, s3_key)
        print(f"[index_loader]   ✓ {local_path} → s3://{bucket}/{s3_key}")

    print("[index_loader] Index upload complete.")
