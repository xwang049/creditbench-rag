"""Orchestrate loading of all data."""

import logging
import time
from pathlib import Path
from typing import Dict
from sqlalchemy.orm import Session
from .load_companies import load_company_data
from .load_credit_events import load_credit_event_data
from .load_macros import load_macro_data
from .load_risk_indicators import load_risk_indicator_data

logger = logging.getLogger(__name__)


def load_all_data(session: Session, data_dir: Path = None) -> Dict[str, int]:
    """Load all data from Excel files into database."""
    if data_dir is None:
        data_dir = Path("./data")

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    logger.info("=" * 60)
    logger.info("Starting full data load...")
    logger.info("=" * 60)

    all_stats = {}
    total_start = time.time()

    # Step 1: Load industry mapping and companies
    try:
        logger.info("\n[1/5] Loading company data...")
        step_start = time.time()
        company_stats = load_company_data(session, data_dir)
        step_duration = time.time() - step_start
        logger.info(f"[OK] Company data loaded in {step_duration:.2f}s")
        all_stats.update(company_stats)
    except Exception as e:
        logger.error(f"[ERROR] Failed to load company data: {e}")
        session.rollback()
        raise

    # Step 2: Load credit events
    try:
        logger.info("\n[2/5] Loading credit events...")
        step_start = time.time()
        event_stats = load_credit_event_data(session, data_dir)
        step_duration = time.time() - step_start
        logger.info(f"[OK] Credit events loaded in {step_duration:.2f}s")
        all_stats.update(event_stats)
    except Exception as e:
        logger.error(f"[ERROR] Failed to load credit events: {e}")
        session.rollback()
        raise

    # Step 3: Load macroeconomic data
    try:
        logger.info("\n[3/5] Loading macroeconomic data...")
        step_start = time.time()
        macro_stats = load_macro_data(session, data_dir)
        step_duration = time.time() - step_start
        logger.info(f"[OK] Macro data loaded in {step_duration:.2f}s")
        all_stats.update(macro_stats)
    except Exception as e:
        logger.error(f"[ERROR] Failed to load macro data: {e}")
        session.rollback()
        raise

    # Step 4: Load risk indicators
    try:
        logger.info("\n[4/5] Loading risk indicators...")
        step_start = time.time()
        risk_stats = load_risk_indicator_data(session, data_dir)
        step_duration = time.time() - step_start
        logger.info(f"[OK] Risk indicators loaded in {step_duration:.2f}s")
        all_stats.update(risk_stats)
    except Exception as e:
        logger.error(f"[ERROR] Failed to load risk indicators: {e}")
        session.rollback()
        raise

    # Step 5: TODO - Generate embeddings (placeholder)
    logger.info("\n[5/5] Embedding generation...")
    logger.info("Note: Embedding generation not yet implemented")
    all_stats['embeddings'] = 0

    total_duration = time.time() - total_start

    logger.info("\n" + "=" * 60)
    logger.info("Data loading complete!")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_duration:.2f}s")
    logger.info("\nRecords loaded:")
    for key, value in all_stats.items():
        logger.info(f"  {key:.<30} {value:>10,}")
    logger.info("=" * 60)

    return all_stats
