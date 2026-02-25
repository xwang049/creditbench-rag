"""Load market data (mktcap.parquet) into PostgreSQL incrementally."""

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.db.models import MarketData

logger = logging.getLogger(__name__)

CHUNK_SIZE = 100_000


def get_max_trading_date(session: Session):
    """Return the latest trading_date already in DB, or None if empty."""
    result = session.execute(text("SELECT MAX(trading_date) FROM market_data")).scalar()
    return result  # date or None


def load_market_data(session: Session, data_dir: Path) -> dict:
    """Incrementally load mktcap.parquet — only rows newer than what's in DB."""
    parquet_path = data_dir / "mktcap.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"mktcap.parquet not found: {parquet_path}")

    max_date = get_max_trading_date(session)
    if max_date:
        logger.info(f"DB has data up to {max_date}, loading newer rows only...")
        filters = [("trading_date", ">", str(max_date))]
    else:
        logger.info("DB is empty, loading all market data...")
        filters = None

    # Read parquet with date filter (avoids loading 76M rows into memory)
    df = pd.read_parquet(parquet_path, filters=filters)
    logger.info(f"Read {len(df):,} rows from parquet")

    if df.empty:
        logger.info("[OK] No new market data to load")
        return {"market_data_new": 0}

    # Normalize column names
    df.columns = df.columns.str.lower()
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date

    inserted = 0
    skipped = 0
    total = len(df)

    for start in range(0, total, CHUNK_SIZE):
        chunk = df.iloc[start: start + CHUNK_SIZE].copy()

        # Convert to list of dicts for Core insert (faster than ORM)
        records = []
        for _, row in chunk.iterrows():
            records.append({
                "u3_company_number": int(row["u3_company_number"]),
                "trading_date": row["trading_date"],
                "last_price": None if pd.isna(row["last_price"]) else float(row["last_price"]),
                "shares_outstanding": None if pd.isna(row["shares_outstanding"]) else float(row["shares_outstanding"]),
                "volume": None if pd.isna(row["volume"]) else float(row["volume"]),
                "marketcap": None if pd.isna(row["marketcap"]) else float(row["marketcap"]),
            })

        stmt = insert(MarketData).values(records).on_conflict_do_nothing(
            index_elements=["u3_company_number", "trading_date"]
        )
        result = session.execute(stmt)
        session.commit()

        inserted += result.rowcount
        skipped += len(records) - result.rowcount
        logger.info(f"  Inserted {inserted:,} / {total:,} rows (skipped {skipped:,} duplicates)...")

    logger.info(f"[OK] {inserted:,} rows inserted, {skipped:,} duplicates skipped")
    return {"market_data_new": inserted, "market_data_skipped": skipped}
