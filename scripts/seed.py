"""One-click data loading script for creditbench database."""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.init_db import init_database
from src.db.session import get_session
from src.ingestion.load_all import load_all_data
from src.ingestion.load_companies import load_company_data
from src.ingestion.load_credit_events import load_credit_event_data
from src.ingestion.load_macros import load_macro_data
from src.ingestion.load_risk_indicators import load_risk_indicator_data
from src.config import settings


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Load CreditBench data from Excel files into PostgreSQL'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Path to data directory containing Excel files (default: ./data)'
    )
    parser.add_argument(
        '--skip-init',
        action='store_true',
        help='Skip database initialization (assume tables already exist)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--only',
        type=str,
        choices=['companies', 'credit_events', 'macros', 'risk_indicators'],
        help='Load only a specific dataset (default: load all)'
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    print("=" * 60)
    print("CreditBench RAG - Database Seeding Script")
    print("=" * 60)
    print()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1

    # Step 1: Initialize database
    if not args.skip_init:
        print("Step 1: Initializing database and creating tables...")
        try:
            init_database()
            print("[OK] Database initialized successfully")
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize database: {e}")
            import traceback
            traceback.print_exc()
            return 1
        print()
    else:
        print("Step 1: Skipped database initialization (--skip-init)")
        print()

    # Step 2: Load data (all or specific dataset)
    if args.only:
        print(f"Step 2: Loading {args.only} data only...")
    else:
        print("Step 2: Loading all data from source files...")
    print("-" * 60)

    try:
        with get_session() as session:
            # Load specific dataset or all data
            if args.only == 'companies':
                stats = load_company_data(session, data_dir)
            elif args.only == 'credit_events':
                stats = load_credit_event_data(session, data_dir)
            elif args.only == 'macros':
                stats = load_macro_data(session, data_dir)
            elif args.only == 'risk_indicators':
                stats = load_risk_indicator_data(session, data_dir)
            else:
                stats = load_all_data(session, data_dir)

            print()
            print("=" * 60)
            print("Data Loading Summary:")
            print("=" * 60)

            if args.only:
                # Print only the loaded dataset stats
                for key, value in stats.items():
                    print(f"  {key:.<30} {value:>10,}")
            else:
                # Print all stats with categories
                print("\nCompany Data:")
                print(f"  Industry mappings:     {stats.get('industry_mapping', 0):>10,}")
                print(f"  Companies:             {stats.get('companies', 0):>10,}")

                print("\nCredit Events:")
                print(f"  Credit events:         {stats.get('credit_events', 0):>10,}")

                print("\nMacroeconomic Data:")
                print(f"  Commodity prices:      {stats.get('commodities', 0):>10,}")
                print(f"  Bond yields:           {stats.get('bond_yields', 0):>10,}")
                print(f"  US macro indicators:   {stats.get('us_macros', 0):>10,}")
                print(f"  FX rates:              {stats.get('fx_rates', 0):>10,}")

                print("\nRisk Indicators:")
                print(f"  Risk indicators:       {stats.get('risk_indicators', 0):>10,}")

                print("\nEmbeddings:")
                print(f"  Embeddings generated:  {stats.get('embeddings', 0):>10,}")

            print("=" * 60)
            print()
            print("[OK] Data loaded successfully!")

    except Exception as e:
        logger.error(f"\n[ERROR] Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print()
    print("Database URL:", settings.DATABASE_URL)
    print()
    print("Next steps:")
    print("  1. Query the database using src/rag/chain.py")
    print("  2. Run the API: python -m src.api.main")
    print("  3. Run tests: pytest tests/")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
