"""Load risk indicators from CSV."""

import logging
from pathlib import Path
from math import floor

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text

from src.db.models import RiskIndicator

logger = logging.getLogger(__name__)


def clean_value(value):
    """Convert NaN, 'NA', 'N/A', and empty strings to None."""
    if pd.isna(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == '' or stripped.upper() in ('NA', 'N/A'):
            return None
    return value


def load_risk_indicators(session: Session, csv_path: Path) -> int:
    """Load risk indicators from CSV with chunked reading."""
    logger.info("Loading risk indicators from CSV...")

    # Clear existing data (idempotent design)
    session.query(RiskIndicator).delete()
    session.commit()
    logger.info("Cleared existing risk_indicators data")

    # Get set of valid u3_company_numbers from companies table
    logger.info("Loading valid company IDs from database...")
    valid_companies = set(
        row[0] for row in session.execute(
            text("SELECT u3_company_number FROM companies")
        ).fetchall()
    )
    logger.info(f"Found {len(valid_companies)} valid company IDs")

    # Column mapping from CSV to database
    # Note: pandas will rename duplicate 'DTDmedian' columns to 'DTDmedian' and 'DTDmedian.1'
    column_mapping = {
        'year': 'year',
        'month': 'month',
        'StkIndx': 'stk_index',
        'STInt': 'st_int',
        'm2b': 'm2b',
        'sigma': 'sigma',
        'DTDmedian': 'dtd_median',      # First DTDmedian (column H)
        'DTDmedian.1': 'dtd_median_i',  # Second DTDmedian (column I, renamed by pandas)
        'dtd': 'dtd',
        'liquidity_r': 'liquidity_r',
        'ni2ta': 'ni2ta',
        'size': 'size',
        'liquidity_fin': 'liquidity_fin'
    }

    total_rows = 0
    inserted_rows = 0
    skipped_rows = 0
    batch_size = 5000
    chunk_size = 50000

    # Read CSV in chunks to handle large file
    logger.info(f"Reading CSV in chunks of {chunk_size:,} rows...")

    chunk_num = 0
    for chunk_df in pd.read_csv(csv_path, chunksize=chunk_size):
        chunk_num += 1
        chunk_start_row = total_rows
        total_rows += len(chunk_df)

        logger.info(f"Processing chunk {chunk_num} (rows {chunk_start_row:,} to {total_rows:,})...")

        # Calculate u3_company_number from Company_Number
        # Formula: u3_company_number = floor(Company_Number / 1000)
        chunk_df['u3_company_number'] = (chunk_df['Company_Number'] // 1000).astype(int)

        # Filter to only include companies that exist in the database
        chunk_df = chunk_df[chunk_df['u3_company_number'].isin(valid_companies)]
        valid_in_chunk = len(chunk_df)
        skipped_in_chunk = (chunk_start_row + chunk_size) - total_rows - valid_in_chunk
        skipped_rows += skipped_in_chunk

        if valid_in_chunk == 0:
            logger.info(f"  No valid company matches in this chunk, skipping")
            continue

        logger.info(f"  Found {valid_in_chunk:,} rows with valid company IDs")

        # Replace empty strings and 'NA' with NaN
        chunk_df = chunk_df.replace(['', 'NA', 'na', 'N/A'], pd.NA)

        # Prepare records for insertion
        records = []
        for idx, row in chunk_df.iterrows():
            try:
                record_data = {
                    'u3_company_number': int(row['u3_company_number']),
                    'year': int(row['year']),
                    'month': int(row['month'])
                }

                # Map and clean all float columns
                for csv_col, db_col in column_mapping.items():
                    if csv_col not in ['year', 'month']:  # Skip time columns
                        if csv_col in row.index:
                            record_data[db_col] = clean_value(row.get(csv_col))

                record = RiskIndicator(**record_data)
                records.append(record)

                # Batch insert
                if len(records) >= batch_size:
                    session.bulk_save_objects(records)
                    session.commit()
                    inserted_rows += len(records)
                    logger.info(f"  Inserted {inserted_rows:,} risk indicator records...")
                    records = []

            except Exception as e:
                logger.warning(f"  Error processing row {idx}: {e}")
                continue

        # Insert remaining records from chunk
        if records:
            session.bulk_save_objects(records)
            session.commit()
            inserted_rows += len(records)
            logger.info(f"  Inserted {inserted_rows:,} risk indicator records...")

    logger.info(f"[OK] Processed {total_rows:,} total rows from CSV")
    logger.info(f"[OK] Loaded {inserted_rows:,} risk indicator records")
    if skipped_rows > 0:
        logger.info(f"[INFO] Skipped {skipped_rows:,} rows (company not found in database)")

    return inserted_rows


def load_risk_indicator_data(session: Session, data_dir: Path) -> dict:
    """Load risk indicators from data directory."""
    csv_path = data_dir / "risk_indicators.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Risk indicators file not found: {csv_path}")

    # Check file size
    file_size_mb = csv_path.stat().st_size / (1024 * 1024)
    logger.info(f"Risk indicators file size: {file_size_mb:.1f} MB")

    count = load_risk_indicators(session, csv_path)
    return {'risk_indicators': count}
