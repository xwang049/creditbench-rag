"""Data update script for CreditBench.

Usage:
    # Update only credit events (incremental)
    python scripts/update_data.py --credit-events

    # Full reload of static tables (companies, macros, risk indicators)
    python scripts/update_data.py --static

    # Embed transcripts (incremental — skips already-embedded paragraphs)
    python scripts/update_data.py --transcripts

    # Run everything
    python scripts/update_data.py --all
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.db.session import get_session
from src.ingestion.load_companies import load_company_data
from src.ingestion.load_credit_events import load_credit_event_data
from src.ingestion.load_macros import load_macro_data
from src.ingestion.load_market_data import load_market_data
from src.ingestion.load_risk_indicators import load_risk_indicator_data
from src.ingestion.load_transcripts import load_transcripts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_credit_events(data_dir: Path) -> dict:
    """Incremental update — only inserts new credit events."""
    logger.info("─" * 50)
    logger.info("Credit Events (incremental)")
    start = time.time()
    with get_session() as session:
        result = load_credit_event_data(session, data_dir)
    elapsed = time.time() - start
    logger.info(f"Done in {elapsed:.1f}s → {result}")
    return result


def run_market_data(data_dir: Path) -> dict:
    """Incremental update — only inserts market data newer than what's in DB."""
    logger.info("─" * 50)
    logger.info("Market Data / mktcap (incremental)")
    start = time.time()
    with get_session() as session:
        result = load_market_data(session, data_dir)
    elapsed = time.time() - start
    logger.info(f"Done in {elapsed:.1f}s → {result}")
    return result


def run_static(data_dir: Path) -> dict:
    """Full reload of companies, macros, risk indicators."""
    results = {}

    for label, fn in [
        ("Companies + Industry Mapping", load_company_data),
        ("Macroeconomic data", load_macro_data),
        ("Risk Indicators", load_risk_indicator_data),
    ]:
        logger.info("─" * 50)
        logger.info(f"{label} (full reload)")
        start = time.time()
        try:
            with get_session() as session:
                result = fn(session, data_dir)
            elapsed = time.time() - start
            logger.info(f"Done in {elapsed:.1f}s → {result}")
            results.update(result)
        except Exception as e:
            logger.error(f"FAILED: {e}")
            raise

    return results


def run_transcripts(data_dir: Path, year_from: int | None = None, year_to: int | None = None) -> dict:
    """Incremental embedding of transcript chunks."""
    logger.info("─" * 50)
    label = "Transcript Embeddings (incremental)"
    if year_from or year_to:
        label += f" [{year_from or ''}–{year_to or ''}]"
    logger.info(label)
    start = time.time()
    with get_session() as session:
        result = load_transcripts(session, data_dir, year_from=year_from, year_to=year_to)
    elapsed = time.time() - start
    logger.info(f"Done in {elapsed:.1f}s → {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Update CreditBench database from OneDrive data")
    parser.add_argument("--credit-events", action="store_true", help="Incremental update of credit events")
    parser.add_argument("--market-data", action="store_true", help="Incremental update of market data (mktcap)")
    parser.add_argument("--static", action="store_true", help="Full reload of companies, macros, risk indicators")
    parser.add_argument("--transcripts", action="store_true", help="Incremental embedding of transcript chunks")
    parser.add_argument("--year-from", type=int, default=None, help="Only embed transcripts from this year (e.g. 2020)")
    parser.add_argument("--year-to", type=int, default=None, help="Only embed transcripts up to this year (e.g. 2024)")
    parser.add_argument("--all", action="store_true", help="Run everything (static reload + all incremental)")
    parser.add_argument("--data-dir", type=str, default=None, help="Override DATA_DIR from .env")
    args = parser.parse_args()

    if not any([args.credit_events, args.market_data, args.static, args.transcripts, args.all]):
        parser.print_help()
        sys.exit(1)

    data_dir = Path(args.data_dir) if args.data_dir else Path(config.DATA_DIR)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    logger.info("=" * 50)
    logger.info(f"Data source: {data_dir}")
    logger.info("=" * 50)

    total_start = time.time()
    all_results = {}

    if args.all or args.static:
        all_results.update(run_static(data_dir))

    if args.all or args.credit_events:
        all_results.update(run_credit_events(data_dir))

    if args.all or args.market_data:
        all_results.update(run_market_data(data_dir))

    if args.all or args.transcripts:
        all_results.update(run_transcripts(data_dir, year_from=args.year_from, year_to=args.year_to))

    logger.info("=" * 50)
    logger.info(f"All done in {time.time() - total_start:.1f}s")
    for k, v in all_results.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
