"""Add risk_indicators table to existing database and load data.

This script is designed to be run on an existing database to:
1. Create the risk_indicators table (if it doesn't exist)
2. Load data from risk_indicators.csv
3. Verify the data was loaded correctly

Safe to run multiple times (idempotent).
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from src.db.session import get_session, engine
from src.db.models import Base, RiskIndicator
from src.ingestion.load_risk_indicators import load_risk_indicator_data


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_risk_indicators_table():
    """Create risk_indicators table if it doesn't exist."""
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Creating risk_indicators table...")
    logger.info("=" * 60)

    try:
        # Only create tables that don't exist yet (checkfirst=True)
        Base.metadata.create_all(bind=engine, checkfirst=True)
        logger.info("✓ risk_indicators table created (or already exists)")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to create table: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_data():
    """Verify the loaded data with sample queries."""
    logger = logging.getLogger(__name__)

    logger.info("\n" + "=" * 60)
    logger.info("Verifying data...")
    logger.info("=" * 60)

    with get_session() as session:
        # Query 1: Total count
        total_count = session.query(RiskIndicator).count()
        logger.info(f"\n1. Total records in risk_indicators: {total_count:,}")

        # Query 2: Distinct companies
        distinct_companies = session.execute(
            text("SELECT COUNT(DISTINCT u3_company_number) FROM risk_indicators")
        ).scalar()
        logger.info(f"2. Distinct companies in risk_indicators: {distinct_companies:,}")

        # Query 3: Sample data with company names
        logger.info("\n3. Sample records with DTD values:")
        logger.info("   " + "-" * 100)
        logger.info(f"   {'U3':>8} | {'Company Name':<40} | {'Year':>4} | {'Month':>2} | {'DTD':>10}")
        logger.info("   " + "-" * 100)

        sample_query = text("""
            SELECT
                ri.u3_company_number,
                c.company_name,
                ri.year,
                ri.month,
                ri.dtd
            FROM risk_indicators ri
            JOIN companies c ON ri.u3_company_number = c.u3_company_number
            WHERE ri.dtd IS NOT NULL
            ORDER BY ri.year DESC, ri.month DESC, ri.u3_company_number
            LIMIT 10
        """)

        results = session.execute(sample_query).fetchall()
        for row in results:
            u3, name, year, month, dtd = row
            # Truncate long company names
            name_display = name[:40] if name and len(name) > 40 else (name or 'N/A')
            logger.info(f"   {u3:>8} | {name_display:<40} | {year:>4} | {month:>2} | {dtd:>10.4f}")

        logger.info("   " + "-" * 100)

        # Query 4: Data completeness
        logger.info("\n4. Data completeness (non-null counts):")
        completeness_query = text("""
            SELECT
                COUNT(*) as total_rows,
                COUNT(dtd) as dtd_count,
                COUNT(stk_index) as stk_index_count,
                COUNT(sigma) as sigma_count,
                COUNT(m2b) as m2b_count
            FROM risk_indicators
        """)

        result = session.execute(completeness_query).fetchone()
        total, dtd_cnt, stk_cnt, sigma_cnt, m2b_cnt = result

        logger.info(f"   Total rows:           {total:>10,}")
        logger.info(f"   DTD non-null:         {dtd_cnt:>10,} ({100*dtd_cnt/total if total > 0 else 0:.1f}%)")
        logger.info(f"   Stock Index non-null: {stk_cnt:>10,} ({100*stk_cnt/total if total > 0 else 0:.1f}%)")
        logger.info(f"   Sigma non-null:       {sigma_cnt:>10,} ({100*sigma_cnt/total if total > 0 else 0:.1f}%)")
        logger.info(f"   M2B non-null:         {m2b_cnt:>10,} ({100*m2b_cnt/total if total > 0 else 0:.1f}%)")

        # Query 5: Year/month range
        logger.info("\n5. Data time range:")
        range_query = text("""
            SELECT
                MIN(year) as min_year,
                MAX(year) as max_year,
                MIN(month) as min_month,
                MAX(month) as max_month
            FROM risk_indicators
        """)

        result = session.execute(range_query).fetchone()
        if result and result[0]:
            min_year, max_year, min_month, max_month = result
            logger.info(f"   Year range: {min_year} to {max_year}")
            logger.info(f"   Month range: {min_month} to {max_month}")

        logger.info("\n" + "=" * 60)
        logger.info("✓ Verification complete!")
        logger.info("=" * 60)


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    print()
    print("=" * 70)
    print("  Add risk_indicators Table to CreditBench Database")
    print("=" * 70)
    print()

    data_dir = Path("./data")
    csv_path = data_dir / "risk_indicators.csv"

    # Check if data file exists
    if not csv_path.exists():
        logger.error(f"Data file not found: {csv_path}")
        logger.error("Please ensure risk_indicators.csv exists in the ./data directory")
        return 1

    file_size_mb = csv_path.stat().st_size / (1024 * 1024)
    logger.info(f"Found data file: {csv_path} ({file_size_mb:.1f} MB)")
    print()

    # Step 1: Create table
    print("Step 1: Creating risk_indicators table...")
    print("-" * 70)
    if not create_risk_indicators_table():
        return 1
    print()

    # Step 2: Load data
    print("Step 2: Loading data from CSV...")
    print("-" * 70)
    try:
        with get_session() as session:
            stats = load_risk_indicator_data(session, data_dir)
            print()
            logger.info(f"✓ Loaded {stats['risk_indicators']:,} risk indicator records")
    except Exception as e:
        logger.error(f"✗ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    print()

    # Step 3: Verify data
    print("Step 3: Verifying loaded data...")
    print("-" * 70)
    try:
        verify_data()
    except Exception as e:
        logger.error(f"✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print()
    print("=" * 70)
    print("  ✓ risk_indicators table successfully added and populated!")
    print("=" * 70)
    print()
    print("You can now query the table using:")
    print("  SELECT * FROM risk_indicators LIMIT 10;")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
