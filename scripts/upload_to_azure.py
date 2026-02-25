"""Upload fundamentals parquet files from OneDrive to Azure Blob Storage.

Usage:
    python scripts/upload_to_azure.py           # upload all missing files
    python scripts/upload_to_azure.py --force   # re-upload even if file already exists
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from azure.storage.blob import BlobServiceClient
from src.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def get_blob_service() -> BlobServiceClient:
    sas = config.AZURE_STORAGE_SAS_TOKEN
    account_url = f"https://{config.AZURE_STORAGE_ACCOUNT}.blob.core.windows.net"
    return BlobServiceClient(account_url=account_url, credential=sas)


def upload_files(force: bool = False):
    fund_dir = Path(config.DATA_DIR) / "foundamentals"
    parquet_files = sorted(fund_dir.glob("*.parquet"))

    if not parquet_files:
        logger.error(f"No parquet files found in {fund_dir}")
        sys.exit(1)

    container = config.AZURE_BLOB_CONTAINER
    prefix = config.AZURE_FUNDAMENTALS_PATH.strip("/")

    client = get_blob_service()
    container_client = client.get_container_client(container)

    # Get existing blobs to skip already-uploaded files
    existing = set()
    if not force:
        existing = {b.name for b in container_client.list_blobs()}
        logger.info(f"{len(existing)} files already in Azure container '{container}'")

    logger.info(f"Uploading {len(parquet_files)} parquet files → {container}/{prefix or '(root)'}")
    logger.info("─" * 50)

    total_bytes = 0
    for i, filepath in enumerate(parquet_files, 1):
        blob_name = f"{prefix}/{filepath.name}" if prefix else filepath.name

        if blob_name in existing:
            logger.info(f"[{i}/{len(parquet_files)}] SKIP  {filepath.name} (already exists)")
            continue

        size_mb = filepath.stat().st_size / 1_048_576
        logger.info(f"[{i}/{len(parquet_files)}] Uploading {filepath.name} ({size_mb:.1f} MB)...")
        start = time.time()

        with open(filepath, "rb") as f:
            container_client.upload_blob(name=blob_name, data=f, overwrite=force)

        elapsed = time.time() - start
        total_bytes += filepath.stat().st_size
        logger.info(f"  Done in {elapsed:.1f}s")

    logger.info("─" * 50)
    logger.info(f"Upload complete. Total: {total_bytes / 1_048_576:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Upload fundamentals parquet files to Azure Blob")
    parser.add_argument("--force", action="store_true", help="Re-upload even if file already exists")
    args = parser.parse_args()

    if not config.use_azure:
        logger.error("Azure credentials not set. Check AZURE_STORAGE_ACCOUNT and AZURE_STORAGE_SAS_TOKEN in .env")
        sys.exit(1)

    upload_files(force=args.force)


if __name__ == "__main__":
    main()
