"""Fetch all relevant data for a company from PostgreSQL and DuckDB fundamentals."""

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.config import config
from src.db.duckdb_query import FUNDAMENTALS_FILES
from src.pipeline.feature_engineer import compute_financial_signals
from src.pipeline.adaptive_retriever import retrieve_transcripts

logger = logging.getLogger(__name__)


def _local_duckdb_query(sql: str) -> pd.DataFrame:
    """Query fundamentals parquets via DuckDB using local paths (skips Azure)."""
    local_dir = Path(config.DATA_DIR) / "foundamentals"
    con = duckdb.connect()
    for name, filename in FUNDAMENTALS_FILES.items():
        path = local_dir / filename
        if path.exists():
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
    result = con.execute(sql).df()
    con.close()
    return result

# Fundamentals columns to pull — balanced set covering credit-relevant signals
_IS_COLS = [
    "FISCAL_YEAR_PERIOD", "ANNOUNCEMENT_DT",
    "SALES_REV_TURN",   # Revenue
    "IS_OPER_INC",      # Operating income (IS field)
    "NET_OPER_INCOME",  # Operating income (alt)
    "EBIT", "EBITDA",
    "NET_INCOME",
    "PRETAX_INC",
    "IS_INT_EXPENSE",   # Interest expense
    "TOT_INT_EXP",
    "IS_INC_TAX_EXP",
    "IS_EPS", "IS_DILUTED_EPS",
    "IS_RD_EXPEND",     # R&D
    "IS_GNRL_AND_ADMIN_EXP",
]

_BS_COLS = [
    "FISCAL_YEAR_PERIOD", "ANNOUNCEMENT_DT",
    "BS_TOT_ASSET",
    "BS_CASH_NEAR_CASH_ITEM",   # Cash & equivalents
    "BS_CUR_ASSET_REPORT",      # Current assets
    "BS_CUR_LIAB",              # Current liabilities
    "BS_ST_BORROW",             # Short-term debt
    "BS_LT_BORROW",             # Long-term debt
    "BS_ST_PORTION_OF_LT_DEBT", # ST portion of LT debt
    "BS_NET_FIX_ASSET",         # PP&E net
]

_CF_COLS = [
    "FISCAL_YEAR_PERIOD", "ANNOUNCEMENT_DT",
    "CF_CASH_FROM_OPER",            # Operating cash flow
    "CF_CAP_EXPEND_INC_FIX_ASSET",  # Capex
    "CF_CASH_FROM_INV_ACT",         # Investing CF
    "CF_CASH_FROM_FNC_ACT",         # Financing CF
    "CF_INCR_LT_BORROW",            # Debt raised
    "CF_REIMB_LT_BORROW",           # Debt repaid
    "CF_ACT_CASH_PAID_FOR_INT_DEBT",# Cash interest paid
]


def _na_to_none(val):
    """Convert Bloomberg 'N.A.' strings to None."""
    if val == "N.A." or val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return val


def _fetch_fundamentals(
    id_bb_company: int,
    years: int = 6,
    as_of: Optional[date] = None,
) -> dict:
    """Fetch income statement, balance sheet, and cash flow from DuckDB.

    Args:
        id_bb_company: Bloomberg company ID
        years: Number of annual periods to return
        as_of: If provided, only include periods announced on or before this date
    """
    if id_bb_company is None:
        return {"income_statement": [], "balance_sheet": [], "cash_flow": []}

    as_of_filter = ""
    if as_of:
        as_of_filter = f"AND (ANNOUNCEMENT_DT IS NULL OR ANNOUNCEMENT_DT <= '{as_of.isoformat()}')"

    results = {}

    for table, cols, key in [
        ("is_",  _IS_COLS,  "income_statement"),
        ("bs",   _BS_COLS,  "balance_sheet"),
        ("cf",   _CF_COLS,  "cash_flow"),
    ]:
        col_list = ", ".join(cols)
        sql = f"""
            WITH deduped AS (
                SELECT {col_list},
                       ROW_NUMBER() OVER (
                           PARTITION BY FISCAL_YEAR_PERIOD
                           ORDER BY ANNOUNCEMENT_DT DESC NULLS LAST
                       ) AS rn
                FROM {table}
                WHERE ID_BB_COMPANY = {id_bb_company}
                  AND FISCAL_YEAR_PERIOD NOT LIKE '%Q%'
                  {as_of_filter}
            )
            SELECT {col_list}
            FROM deduped
            WHERE rn = 1
            ORDER BY FISCAL_YEAR_PERIOD DESC
            LIMIT {years}
        """
        try:
            df = _local_duckdb_query(sql)
            records = []
            for _, row in df.iterrows():
                rec = {}
                for col in cols:
                    if col in ("FISCAL_YEAR_PERIOD", "ANNOUNCEMENT_DT"):
                        rec[col] = str(row[col]) if pd.notna(row.get(col)) else None
                    else:
                        rec[col] = _na_to_none(row.get(col))
                records.append(rec)
            results[key] = records
        except Exception as e:
            logger.warning(f"Could not fetch {table} for id_bb_company={id_bb_company}: {e}")
            results[key] = []

    return results


def _fetch_risk_indicators(
    session: Session,
    u3: int,
    n_months: int = 36,
    as_of: Optional[date] = None,
) -> list[dict]:
    """Fetch risk indicator observations (DTD, sigma, etc.).

    Args:
        as_of: If provided, only return rows on or before this date
    """
    date_filter = ""
    params: dict = {"u3": u3, "n": n_months}
    if as_of:
        date_filter = "AND (year < :as_of_year OR (year = :as_of_year AND month <= :as_of_month))"
        params["as_of_year"] = as_of.year
        params["as_of_month"] = as_of.month

    rows = session.execute(text(f"""
        SELECT year, month, dtd, sigma, m2b, ni2ta, size, liquidity_r, liquidity_fin
        FROM risk_indicators
        WHERE u3_company_number = :u3
          {date_filter}
        ORDER BY year DESC, month DESC
        LIMIT :n
    """), params).fetchall()

    return [
        {
            "year": r.year, "month": r.month,
            "dtd": r.dtd, "sigma": r.sigma, "m2b": r.m2b,
            "ni2ta": r.ni2ta, "size": r.size,
            "liquidity_r": r.liquidity_r, "liquidity_fin": r.liquidity_fin,
        }
        for r in rows
    ]


def _fetch_market_data(
    session: Session,
    u3: int,
    n_days: int = 504,
    as_of: Optional[date] = None,
) -> list[dict]:
    """Fetch daily market data (price, market cap).

    Args:
        as_of: If provided, only return rows on or before this date
    """
    date_filter = ""
    params: dict = {"u3": u3, "n": n_days}
    if as_of:
        date_filter = "AND trading_date <= :as_of"
        params["as_of"] = as_of

    rows = session.execute(text(f"""
        SELECT trading_date, last_price, marketcap, volume, shares_outstanding
        FROM market_data
        WHERE u3_company_number = :u3
          {date_filter}
        ORDER BY trading_date DESC
        LIMIT :n
    """), params).fetchall()

    return [
        {
            "date": str(r.trading_date),
            "price": r.last_price,
            "marketcap": r.marketcap,
            "volume": r.volume,
            "shares_outstanding": r.shares_outstanding,
        }
        for r in rows
    ]


def _fetch_credit_events(
    session: Session,
    u3: int,
    as_of: Optional[date] = None,
) -> list[dict]:
    """Fetch credit events for a company (historical context only).

    Args:
        as_of: If provided, only return events announced strictly before this date
    """
    date_filter = ""
    params: dict = {"u3": u3}
    if as_of:
        date_filter = "AND (announcement_date IS NULL OR announcement_date < :as_of)"
        params["as_of"] = as_of

    rows = session.execute(text(f"""
        SELECT announcement_date, effective_date, event_type, action_name, subcategory
        FROM credit_events
        WHERE u3_company_number = :u3
          {date_filter}
        ORDER BY announcement_date DESC
    """), params).fetchall()

    return [
        {
            "announcement_date": str(r.announcement_date) if r.announcement_date else None,
            "effective_date": str(r.effective_date) if r.effective_date else None,
            "event_type": r.event_type,
            "action_name": r.action_name,
            "subcategory": r.subcategory,
        }
        for r in rows
    ]


def _fetch_macro_context(
    session: Session,
    as_of: Optional[date] = None,
) -> dict:
    """Fetch macro snapshot (most recent available on or before as_of).

    Args:
        as_of: If provided, only use macro data up to this date
    """
    us_filter = "WHERE sp500 IS NOT NULL"
    yields_filter = ""
    params_us: dict = {}
    params_yields: dict = {}

    if as_of:
        us_filter += " AND date <= :as_of"
        params_us["as_of"] = as_of
        yields_filter = "WHERE data_date <= :as_of"
        params_yields["as_of"] = as_of

    us = session.execute(text(f"""
        SELECT date, sp500, vix, unemployment, cpi
        FROM macro_us
        {us_filter}
        ORDER BY date DESC LIMIT 1
    """), params_us).fetchone()

    yields = session.execute(text(f"""
        SELECT data_date, us_2y, us_5y, us_10y, us_30y
        FROM macro_bond_yields
        {yields_filter}
        ORDER BY data_date DESC LIMIT 1
    """), params_yields).fetchone()

    return {
        "us": {
            "date": str(us.date) if us else None,
            "sp500": us.sp500 if us else None,
            "vix": us.vix if us else None,
            "unemployment": us.unemployment if us else None,
            "cpi": us.cpi if us else None,
        } if us else None,
        "bond_yields": {
            "date": str(yields.data_date) if yields else None,
            "us_2y": yields.us_2y if yields else None,
            "us_5y": yields.us_5y if yields else None,
            "us_10y": yields.us_10y if yields else None,
            "us_30y": yields.us_30y if yields else None,
        } if yields else None,
    }


def _fetch_transcripts(
    session: Session,
    u3: int,
    n: int = 5,
    as_of: Optional[date] = None,
) -> list[dict]:
    """Fetch transcript chunks for the company (embedded only).

    Args:
        as_of: If provided, only return transcripts on or before this date
    """
    date_filter = ""
    params: dict = {"u3": u3, "n": n}
    if as_of:
        date_filter = "AND (call_date IS NULL OR call_date <= :as_of)"
        params["as_of"] = as_of

    rows = session.execute(text(f"""
        SELECT call_date, year, quarter, speaker_type, speaker_name, text_content
        FROM transcript_chunks
        WHERE u3_company_number = :u3
          AND embedding IS NOT NULL
          {date_filter}
        ORDER BY call_date DESC NULLS LAST
        LIMIT :n
    """), params).fetchall()

    return [
        {
            "call_date": str(r.call_date) if r.call_date else None,
            "year": r.year,
            "quarter": r.quarter,
            "speaker_type": r.speaker_type,
            "speaker_name": r.speaker_name,
            "text_content": r.text_content[:500],  # truncate for context
        }
        for r in rows
    ]


def fundamentals_available() -> bool:
    """Return True if at least one fundamentals parquet file is accessible locally."""
    local_dir = Path(config.DATA_DIR) / "foundamentals"
    return local_dir.exists() and any(local_dir.glob("*.parquet"))


def fetch_all(
    session: Session,
    company: dict,
    fundamentals_years: int = 6,
    as_of: Optional[date] = None,
    include_fundamentals: bool = True,
    include_risk_indicators: bool = True,
    include_market_data: bool = True,
    include_macro: bool = True,
    include_transcripts: bool = True,
    transcript_mode: str = "rag",
    transcript_k: int = 8,
    lookback_months: Optional[int] = None,
) -> dict:
    """Fetch all available data for a company.

    Args:
        session:               Active DB session
        company:               Dict with at least u3_company_number and id_bb_company
        fundamentals_years:    How many annual periods to pull from fundamentals
        as_of:                 If provided, only include data up to this date.
        include_fundamentals:  Pull income statement / balance sheet / cash flow
        include_risk_indicators: Pull DTD / sigma / etc.
        include_market_data:   Pull daily price / market cap
        include_macro:         Pull macro snapshot
        include_transcripts:   Pull earnings call transcript chunks
        transcript_mode:       "rag" (signal-driven semantic search) or
                               "date" (most-recent N chunks, no embedding)
        transcript_k:          Number of transcript chunks to retrieve
        lookback_months:       If set, limit risk_indicators and market_data windows

    Returns:
        Dict with keys: company, fundamentals, signals, risk_indicators,
                        market_data, credit_events, macro, transcripts
    """
    u3 = company["u3_company_number"]
    bb = company["id_bb_company"]

    logger.info(f"Fetching data for {company['company_name']} (u3={u3}, bb={bb})"
                + (f" as_of={as_of}" if as_of else ""))

    # ── Fundamentals ──────────────────────────────────────────────────────────
    empty_fundamentals = {"income_statement": [], "balance_sheet": [], "cash_flow": []}
    if include_fundamentals and bb is not None and fundamentals_available():
        try:
            fundamentals = _fetch_fundamentals(bb, years=fundamentals_years, as_of=as_of)
        except Exception as e:
            logger.warning(f"Fundamentals fetch failed for bb={bb}: {e}")
            fundamentals = empty_fundamentals
    else:
        fundamentals = empty_fundamentals

    # ── Layer 1: Feature engineering ─────────────────────────────────────────
    signals = compute_financial_signals(fundamentals) if include_fundamentals else None

    # ── Risk indicators ───────────────────────────────────────────────────────
    risk_n = lookback_months if lookback_months else 36
    risk = _fetch_risk_indicators(session, u3, n_months=risk_n, as_of=as_of) if include_risk_indicators else []

    # ── Market data ───────────────────────────────────────────────────────────
    market_n = (lookback_months * 21) if lookback_months else 504
    market = _fetch_market_data(session, u3, n_days=market_n, as_of=as_of) if include_market_data else []

    # ── Credit events (always fetched) ────────────────────────────────────────
    events = _fetch_credit_events(session, u3, as_of=as_of)

    # ── Macro ─────────────────────────────────────────────────────────────────
    macro = _fetch_macro_context(session, as_of=as_of) if include_macro else {}

    # ── Layer 2: Transcript retrieval ─────────────────────────────────────────
    transcripts: list[dict] = []
    if include_transcripts:
        if transcript_mode == "rag" and signals is not None:
            transcripts = retrieve_transcripts(
                session=session,
                signals=signals,
                u3_company_number=u3,
                industry=company.get("industry_sector"),
                as_of=as_of,
                k=transcript_k,
            )
        else:
            # Fallback: date-filtered, no semantic search
            transcripts = _fetch_transcripts(session, u3, n=transcript_k, as_of=as_of)

    logger.info(
        f"  IS={len(fundamentals['income_statement'])} periods, "
        f"BS={len(fundamentals['balance_sheet'])} periods, "
        f"CF={len(fundamentals['cash_flow'])} periods, "
        f"risk={len(risk)}, market={len(market)}, "
        f"events={len(events)}, transcripts={len(transcripts)} "
        f"({'rag' if transcript_mode == 'rag' and signals else 'date'})"
    )

    return {
        "company":          company,
        "fundamentals":     fundamentals,
        "signals":          signals,          # Layer 1 output (None if no fundamentals)
        "risk_indicators":  risk,
        "market_data":      market,
        "credit_events":    events,
        "macro":            macro,
        "transcripts":      transcripts,
    }
