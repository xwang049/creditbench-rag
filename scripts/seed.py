"""One-click data loading script for creditbench database."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.init_db import init_database, create_tables
from src.db.session import get_session
from src.ingestion.load_all import load_all_data
from src.config import settings


def main():
    """Main entry point for seeding the database."""
    print("=" * 60)
    print("CreditBench RAG - Database Seeding Script")
    print("=" * 60)
    print()

    # Step 1: Initialize database and create tables
    print("Step 1: Initializing database and creating tables...")
    try:
        init_database()
        print("✓ Database initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize database: {e}")
        return 1

    print()

    # Step 2: Load all data
    print("Step 2: Loading data from source files...")
    print("-" * 60)

    try:
        with get_session() as session:
            stats = load_all_data(session)

            print()
            print("=" * 60)
            print("Data Loading Summary:")
            print("=" * 60)
            print(f"Companies loaded:      {stats.get('companies', 0):>6}")
            print(f"Credit events loaded:  {stats.get('credit_events', 0):>6}")
            print(f"Macro indicators:      {stats.get('macro_indicators', 0):>6}")
            print(f"Embeddings generated:  {stats.get('embeddings', 0):>6}")
            print("=" * 60)
            print()
            print("✓ All data loaded successfully!")

    except Exception as e:
        print(f"\n✗ Failed to load data: {e}")
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
