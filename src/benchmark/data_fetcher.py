"""Time-windowed data fetcher for benchmark experiments.

Wraps src.pipeline.data_fetcher.fetch_all() with:
- as_of = case.cutoff_date (prevents look-ahead)
- n_months / n_days tuned to cfg.lookback_months
- Optional suppression of data types not needed for this run
"""

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from src.benchmark.case_builder import CaseRecord
from src.benchmark.config import BenchmarkConfig
from src.config import config as app_config
from src.pipeline.data_fetcher import (
    _fetch_fundamentals,
    _fetch_risk_indicators,
    _fetch_market_data,
    _fetch_credit_events,
    _fetch_macro_context,
    _fetch_transcripts,
)

logger = logging.getLogger(__name__)

# Transcripts are excluded from benchmark (embedding not available by default)
_INCLUDE_TRANSCRIPTS = False


def _fundamentals_available() -> bool:
    """Return True if at least one fundamentals parquet file is accessible locally."""
    local_dir = Path(app_config.DATA_DIR) / "foundamentals"
    return local_dir.exists() and any(local_dir.glob("*.parquet"))


def fetch_for_case(
    session: Session,
    case: CaseRecord,
    cfg: BenchmarkConfig,
) -> dict:
    """Fetch all data for a benchmark case, strictly bounded by cutoff_date.

    Args:
        session: Active DB session
        case:    CaseRecord specifying company and cutoff_date
        cfg:     BenchmarkConfig controlling which data types to include

    Returns:
        Dict with keys: company, fundamentals, risk_indicators,
                        market_data, credit_events, macro, transcripts
    """
    as_of = case.cutoff_date
    u3 = case.u3_company_number
    bb = case.id_bb_company

    # Build a company dict compatible with context_builder
    company = {
        "u3_company_number": u3,
        "id_bb_company": bb,
        "ticker": case.ticker,
        "company_name": case.company_name,
        "country_name": "United States",
        "market_status": "—",
        "prime_exchange": "—",
        "id_isin": None,
        "industry_sector": case.industry_sector,
        "industry_group": None,
        "industry_subgroup": None,
    }

    # Risk indicators (monthly, lookback window)
    risk = []
    if cfg.include_risk_indicators:
        risk = _fetch_risk_indicators(
            session, u3,
            n_months=cfg.lookback_months,
            as_of=as_of,
        )

    # Market data (daily, lookback window converted to trading days)
    market = []
    if cfg.include_market_data:
        n_days = cfg.lookback_months * 21  # ~21 trading days per month
        market = _fetch_market_data(session, u3, n_days=n_days, as_of=as_of)

    # Historical credit events (background context only — exclude the target event)
    events = _fetch_credit_events(session, u3, as_of=as_of)

    # Macro context (latest available on or before cutoff)
    macro = {}
    if cfg.include_macro:
        macro = _fetch_macro_context(session, as_of=as_of)

    # Fundamentals via DuckDB (skip if files not available or disabled)
    fundamentals = {"income_statement": [], "balance_sheet": [], "cash_flow": []}
    if cfg.include_fundamentals and bb is not None and _fundamentals_available():
        try:
            fundamentals = _fetch_fundamentals(bb, years=6, as_of=as_of)
        except Exception as e:
            logger.warning(f"Fundamentals fetch failed for bb={bb}: {e}")
    elif cfg.include_fundamentals and not _fundamentals_available():
        logger.debug("Fundamentals parquet not found locally — skipping")

    # Transcripts excluded from benchmark (embeddings not guaranteed)
    transcripts = []
    if _INCLUDE_TRANSCRIPTS:
        transcripts = _fetch_transcripts(session, u3, as_of=as_of)

    logger.debug(
        f"[{case.group}] {case.company_name} as_of={as_of}: "
        f"risk={len(risk)}, market={len(market)}, events={len(events)}, "
        f"IS={len(fundamentals['income_statement'])}"
    )

    return {
        "company": company,
        "fundamentals": fundamentals,
        "risk_indicators": risk,
        "market_data": market,
        "credit_events": events,
        "macro": macro,
        "transcripts": transcripts,
    }
