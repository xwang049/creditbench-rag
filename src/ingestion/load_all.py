"""Orchestrate loading all data sources."""

import logging
from pathlib import Path
from typing import Optional

from src.db.init_db import init_db
from src.ingestion.load_companies import load_companies
from src.ingestion.load_credit_events import load_credit_events
from src.ingestion.load_macros import load_macros

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_all_data(
    data_dir: str | Path,
    companies_file: str = "companies.xlsx",
    events_file: str = "credit_events.xlsx",
    macros_file: str = "macro_indicators.xlsx",
    reset_db: bool = False,
    generate_embeddings: bool = True,
) -> dict[str, int]:
    """
    Load all data from a directory.

    Args:
        data_dir: Directory containing data files
        companies_file: Company data filename
        events_file: Credit events filename
        macros_file: Macro indicators filename
        reset_db: Whether to reset the database before loading
        generate_embeddings: Whether to generate embeddings

    Returns:
        Dictionary with counts of loaded records
    """
    data_dir = Path(data_dir)
    logger.info(f"Loading all data from {data_dir}")

    # Initialize or reset database
    if reset_db:
        from src.db.init_db import drop_all
        logger.warning("Resetting database...")
        drop_all()

    init_db()

    results = {}

    # Load companies first (required for foreign keys)
    companies_path = data_dir / companies_file
    if companies_path.exists():
        results["companies"] = load_companies(
            companies_path,
            generate_embeddings=generate_embeddings
        )
    else:
        logger.warning(f"Companies file not found: {companies_path}")
        results["companies"] = 0

    # Load credit events
    events_path = data_dir / events_file
    if events_path.exists():
        results["credit_events"] = load_credit_events(
            events_path,
            generate_embeddings=generate_embeddings
        )
    else:
        logger.warning(f"Events file not found: {events_path}")
        results["credit_events"] = 0

    # Load macro indicators
    macros_path = data_dir / macros_file
    if macros_path.exists():
        results["macro_indicators"] = load_macros(macros_path)
    else:
        logger.warning(f"Macros file not found: {macros_path}")
        results["macro_indicators"] = 0

    logger.info("Data loading complete!")
    logger.info(f"Results: {results}")
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.load_all <data_directory>")
        sys.exit(1)

    load_all_data(sys.argv[1], reset_db=True)
