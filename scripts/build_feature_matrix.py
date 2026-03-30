#!/usr/bin/env python3
"""Build the feature matrix for ML credit default prediction.

Produces a parquet file with 105 features per (company, month) observation,
plus default labels at 3m/6m/12m horizons.

Usage:
    python scripts/build_feature_matrix.py --output features_matrix.parquet
    python scripts/build_feature_matrix.py --output features_matrix.parquet --year-from 2005 --year-to 2023
    python scripts/build_feature_matrix.py --output features_matrix.parquet --dry-run

Data sources:
    CR2 (56 features)         : foundamentals/fundamentals_namr_cr2_history.parquet
    GPT-proposed (32 fin + 4 mkt) : IS/BS/CF parquets + mktcap.parquet
    CRI (13 features)         : risk_indicators.csv
    Labels                    : credit_events table (PostgreSQL)
    Company mapping           : companies table (PostgreSQL)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.session import get_session
from src.features.config import (
    CR2_FEATURES, CRI_DIRECT, CRI_TREND,
    DATA_DIR, FUND_DIR, MKTCAP_PATH, CRI_PATH,
)

logger = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _timer(label: str):
    class _T:
        def __enter__(self):
            self.t0 = time.time()
            logger.info(f"  {label}...")
            return self
        def __exit__(self, *_):
            logger.info(f"  {label} done ({time.time()-self.t0:.1f}s)")
    return _T()


def _register_fundamentals(con: duckdb.DuckDBPyConnection) -> None:
    """Register parquet files as DuckDB views."""
    tables = {
        "cr2": "fundamentals_namr_cr2_history.parquet",
        "is_": "fundamentals_namr_is_history.parquet",
        "bs":  "fundamentals_namr_bs_history.parquet",
        "cf":  "fundamentals_namr_cf_history.parquet",
    }
    for name, filename in tables.items():
        path = FUND_DIR / filename
        if path.exists():
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
            logger.info(f"    Registered view: {name} → {filename}")
        else:
            logger.warning(f"    Missing: {path}")

    if MKTCAP_PATH.exists():
        con.execute(f"CREATE VIEW mktcap AS SELECT * FROM read_parquet('{MKTCAP_PATH}')")
        logger.info(f"    Registered view: mktcap")

    if CRI_PATH.exists():
        con.execute(f"CREATE VIEW cri AS SELECT * FROM read_csv('{CRI_PATH}', auto_detect=true)")
        logger.info(f"    Registered view: cri")


# ── Step 1: Company universe & observation grid ──────────────────────────────

def build_observation_grid(
    con: duckdb.DuckDBPyConnection,
    year_from: int,
    year_to: int,
) -> int:
    """Build monthly observation grid from mktcap data.

    Uses month-end market data as the spine. Each row = (u3, month_end_date, marketcap).
    Only includes months where the company had at least one trading day.

    Creates table: obs_grid(u3_company_number, observation_date, marketcap_eom)
    Returns: number of observation rows
    """
    con.execute(f"""
        CREATE TABLE obs_grid AS
        WITH monthly AS (
            SELECT
                U3_company_number AS u3_company_number,
                LAST_DAY(CAST(trading_date AS DATE)) AS observation_date,
                -- Use last available trading day's data per month
                LAST(marketcap ORDER BY trading_date) AS marketcap_eom,
                LAST(last_price ORDER BY trading_date) AS price_eom,
            FROM mktcap
            WHERE CAST(trading_date AS DATE) >= '{year_from}-01-01'
              AND CAST(trading_date AS DATE) <= '{year_to}-12-31'
              AND marketcap IS NOT NULL
              AND marketcap > 0
            GROUP BY U3_company_number, LAST_DAY(CAST(trading_date AS DATE))
        )
        SELECT * FROM monthly
        ORDER BY u3_company_number, observation_date
    """)
    n = con.execute("SELECT COUNT(*) FROM obs_grid").fetchone()[0]
    n_companies = con.execute("SELECT COUNT(DISTINCT u3_company_number) FROM obs_grid").fetchone()[0]
    logger.info(f"  Observation grid: {n:,} rows, {n_companies:,} companies, {year_from}-{year_to}")
    return n


# ── Step 2: Company metadata ─────────────────────────────────────────────────

def load_company_metadata(con: duckdb.DuckDBPyConnection, session) -> None:
    """Load company mapping (u3 ↔ id_bb_company) and credit events from PostgreSQL."""
    # Companies
    rows = session.execute(text("""
        SELECT c.u3_company_number, c.id_bb_company, c.company_name,
               c.country_name, im.industry_sector
        FROM companies c
        LEFT JOIN industry_mapping im ON im.industry_subgroup_num = c.industry_subgroup_num
    """)).fetchall()
    companies = pd.DataFrame(rows, columns=[
        "u3_company_number", "id_bb_company", "company_name",
        "country", "industry_sector",
    ])
    con.register("companies_df", companies)
    con.execute("""
        CREATE TABLE companies AS
        SELECT * FROM companies_df
        WHERE id_bb_company IS NOT NULL
    """)
    n = con.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    logger.info(f"  Companies: {n:,} with id_bb_company")

    # Credit events (for labels)
    events = session.execute(text("""
        SELECT u3_company_number, announcement_date AS event_date, action_name
        FROM credit_events
        WHERE action_name IN ('Default Corp Action', 'Bankruptcy Filing')
          AND announcement_date IS NOT NULL
    """)).fetchall()
    events_df = pd.DataFrame(events, columns=["u3_company_number", "event_date", "action_name"])
    con.register("events_df", events_df)
    con.execute("CREATE TABLE credit_events AS SELECT * FROM events_df")
    n_events = con.execute("SELECT COUNT(*) FROM credit_events").fetchone()[0]
    logger.info(f"  Credit events: {n_events:,} default/bankruptcy events")


# ── Step 3: CR2 features ─────────────────────────────────────────────────────

def build_cr2_features(con: duckdb.DuckDBPyConnection) -> None:
    """Extract 56 CR2 features with point-in-time alignment.

    Uses ASOF JOIN to efficiently find the most recently announced
    CR2 row (ANNOUNCEMENT_DT <= observation_date) without materializing
    all candidate pairs.
    """
    cr2_cols_sql = ", ".join(f'cr2."{c}"' for c in CR2_FEATURES)

    con.execute(f"""
        CREATE TABLE cr2_features AS
        WITH
        grid_with_bb AS (
            SELECT g.u3_company_number, g.observation_date,
                   STRFTIME(g.observation_date, '%Y%m%d') AS obs_yyyymmdd,
                   c.id_bb_company
            FROM obs_grid g
            JOIN companies c ON g.u3_company_number = c.u3_company_number
        ),
        cr2_deduped AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY ID_BB_COMPANY, FISCAL_YEAR_PERIOD
                    ORDER BY ANNOUNCEMENT_DT DESC NULLS LAST
                ) AS rn
            FROM cr2
            WHERE ANNOUNCEMENT_DT IS NOT NULL
              AND ANNOUNCEMENT_DT != 'N.A.'
              AND FISCAL_YEAR_PERIOD NOT LIKE '%Q%'
        ),
        cr2_clean AS (
            SELECT * FROM cr2_deduped WHERE rn = 1
        )
        SELECT g.u3_company_number, g.observation_date,
               {cr2_cols_sql}
        FROM grid_with_bb g
        ASOF JOIN cr2_clean cr2
          ON g.id_bb_company = cr2.ID_BB_COMPANY
         AND g.obs_yyyymmdd >= cr2.ANNOUNCEMENT_DT
    """)
    n = con.execute("SELECT COUNT(*) FROM cr2_features").fetchone()[0]
    logger.info(f"  CR2 features: {n:,} rows")


# ── Step 4: GPT-proposed financial features ──────────────────────────────────

def build_gpt_financial_features(con: duckdb.DuckDBPyConnection) -> None:
    """Compute 32 GPT-proposed financial features from IS/BS/CF.

    Point-in-time aligned using ANNOUNCEMENT_DT.
    """
    # Build grid with bb mapping once
    con.execute("""
        CREATE TEMP TABLE grid_bb AS
        SELECT g.u3_company_number, g.observation_date,
               STRFTIME(g.observation_date, '%Y%m%d') AS obs_yyyymmdd,
               c.id_bb_company
        FROM obs_grid g
        JOIN companies c ON g.u3_company_number = c.u3_company_number
    """)

    # Deduplicate and point-in-time join for IS, BS, CF using ASOF JOIN
    for table, alias in [("is_", "is_pit"), ("bs", "bs_pit"), ("cf", "cf_pit")]:
        con.execute(f"""
            CREATE TABLE {alias} AS
            WITH deduped AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY ID_BB_COMPANY, FISCAL_YEAR_PERIOD
                        ORDER BY ANNOUNCEMENT_DT DESC NULLS LAST
                    ) AS rn
                FROM {table}
                WHERE ANNOUNCEMENT_DT IS NOT NULL
                  AND ANNOUNCEMENT_DT != 'N.A.'
                  AND FISCAL_YEAR_PERIOD NOT LIKE '%Q%'
            ),
            clean AS (SELECT * FROM deduped WHERE rn = 1)
            SELECT g.u3_company_number, g.observation_date, t.*
            FROM grid_bb g
            ASOF JOIN clean t
              ON g.id_bb_company = t.ID_BB_COMPANY
             AND g.obs_yyyymmdd >= t.ANNOUNCEMENT_DT
        """)
        n = con.execute(f"SELECT COUNT(*) FROM {alias}").fetchone()[0]
        logger.info(f"    {alias}: {n:,} rows")

    # Prior-year IS/BS for ROE (avg equity) and dividend/share growth.
    # Strategy: ASOF JOIN gives the most recent (rank 1). For rank 2,
    # join the current PIT result back to get the prior period.
    for table, pit_alias, prior_alias in [("bs", "bs_pit", "bs_prior"), ("is_", "is_pit", "is_prior")]:
        con.execute(f"""
            CREATE TABLE {prior_alias} AS
            WITH deduped AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY ID_BB_COMPANY, FISCAL_YEAR_PERIOD
                        ORDER BY ANNOUNCEMENT_DT DESC NULLS LAST
                    ) AS rn
                FROM {table}
                WHERE ANNOUNCEMENT_DT IS NOT NULL
                  AND ANNOUNCEMENT_DT != 'N.A.'
                  AND FISCAL_YEAR_PERIOD NOT LIKE '%Q%'
            ),
            clean AS (SELECT * FROM deduped WHERE rn = 1)
            SELECT p.u3_company_number, p.observation_date, t.*
            FROM {pit_alias} p
            ASOF JOIN clean t
              ON p.ID_BB_COMPANY = t.ID_BB_COMPANY
             AND t.ANNOUNCEMENT_DT < p.ANNOUNCEMENT_DT
        """)
        n = con.execute(f"SELECT COUNT(*) FROM {prior_alias}").fetchone()[0]
        logger.info(f"    {prior_alias}: {n:,} rows")

    # Now compute all 32 features
    con.execute("""
        CREATE TABLE gpt_financial AS
        SELECT
            i.u3_company_number,
            i.observation_date,

            -- Profitability
            CASE WHEN TRY_CAST(b.BS_TOT_ASSET AS DOUBLE) != 0
                 THEN TRY_CAST(i.EBITDA AS DOUBLE) / TRY_CAST(b.BS_TOT_ASSET AS DOUBLE)
            END AS EBITDA_TO_TOTAL_ASSETS,

            CASE WHEN TRY_CAST(b.BS_TOT_ASSET AS DOUBLE) != 0
                 THEN TRY_CAST(i.EBIT AS DOUBLE) / TRY_CAST(b.BS_TOT_ASSET AS DOUBLE)
            END AS EBIT_TO_TOTAL_ASSETS,

            CASE WHEN (TRY_CAST(b.BS_TOT_EQY AS DOUBLE) + TRY_CAST(bp.BS_TOT_EQY AS DOUBLE)) != 0
                 THEN TRY_CAST(i.NET_INCOME AS DOUBLE)
                      / ((TRY_CAST(b.BS_TOT_EQY AS DOUBLE) + TRY_CAST(bp.BS_TOT_EQY AS DOUBLE)) / 2.0)
            END AS ROE,

            -- Coverage
            CASE WHEN COALESCE(TRY_CAST(i.IS_INT_EXPENSE AS DOUBLE), TRY_CAST(i.TOT_INT_EXP AS DOUBLE)) != 0
                 THEN TRY_CAST(i.EBIT AS DOUBLE)
                      / COALESCE(TRY_CAST(i.IS_INT_EXPENSE AS DOUBLE), TRY_CAST(i.TOT_INT_EXP AS DOUBLE))
            END AS EBIT_TO_INTEREST_EXPENSE,

            CASE WHEN COALESCE(TRY_CAST(i.IS_INT_EXPENSE AS DOUBLE), TRY_CAST(i.TOT_INT_EXP AS DOUBLE)) != 0
                 THEN TRY_CAST(c.CF_CASH_FROM_OPER AS DOUBLE)
                      / COALESCE(TRY_CAST(i.IS_INT_EXPENSE AS DOUBLE), TRY_CAST(i.TOT_INT_EXP AS DOUBLE))
            END AS CFO_TO_INTEREST_EXPENSE,

            CASE WHEN TRY_CAST(b.BS_TOT_ASSET AS DOUBLE) != 0
                 THEN TRY_CAST(c.CF_CASH_FROM_OPER AS DOUBLE) / TRY_CAST(b.BS_TOT_ASSET AS DOUBLE)
            END AS CFO_TO_TOTAL_ASSETS,

            CASE WHEN (COALESCE(TRY_CAST(b.BS_ST_BORROW AS DOUBLE), 0)
                      + COALESCE(TRY_CAST(b.BS_LT_BORROW AS DOUBLE), 0)) != 0
                 THEN TRY_CAST(c.CF_CASH_FROM_OPER AS DOUBLE)
                      / (COALESCE(TRY_CAST(b.BS_ST_BORROW AS DOUBLE), 0)
                         + COALESCE(TRY_CAST(b.BS_LT_BORROW AS DOUBLE), 0))
            END AS CFO_TO_TOTAL_DEBT,

            CASE WHEN TRY_CAST(i.EBITDA AS DOUBLE) != 0
                 THEN (COALESCE(TRY_CAST(b.BS_ST_BORROW AS DOUBLE), 0)
                       + COALESCE(TRY_CAST(b.BS_LT_BORROW AS DOUBLE), 0)
                       - COALESCE(TRY_CAST(b.BS_CASH_NEAR_CASH_ITEM AS DOUBLE), 0))
                      / TRY_CAST(i.EBITDA AS DOUBLE)
            END AS NET_DEBT_TO_EBITDA,

            CASE WHEN TRY_CAST(i.EBITDA AS DOUBLE) != 0
                 THEN (COALESCE(TRY_CAST(b.BS_ST_BORROW AS DOUBLE), 0)
                       + COALESCE(TRY_CAST(b.BS_LT_BORROW AS DOUBLE), 0))
                      / TRY_CAST(i.EBITDA AS DOUBLE)
            END AS TOTAL_DEBT_TO_EBITDA,

            CASE WHEN TRY_CAST(b.BS_TOT_ASSET AS DOUBLE) != 0
                 THEN (TRY_CAST(c.CF_CASH_FROM_OPER AS DOUBLE)
                       - ABS(TRY_CAST(c.CF_CAP_EXPEND_INC_FIX_ASSET AS DOUBLE)))
                      / TRY_CAST(b.BS_TOT_ASSET AS DOUBLE)
            END AS FCF_TO_TOTAL_ASSETS,

            CASE WHEN (COALESCE(TRY_CAST(b.BS_ST_BORROW AS DOUBLE), 0)
                      + COALESCE(TRY_CAST(b.BS_LT_BORROW AS DOUBLE), 0)) != 0
                 THEN (TRY_CAST(c.CF_CASH_FROM_OPER AS DOUBLE)
                       - ABS(TRY_CAST(c.CF_CAP_EXPEND_INC_FIX_ASSET AS DOUBLE)))
                      / (COALESCE(TRY_CAST(b.BS_ST_BORROW AS DOUBLE), 0)
                         + COALESCE(TRY_CAST(b.BS_LT_BORROW AS DOUBLE), 0))
            END AS FCF_TO_TOTAL_DEBT,

            -- Liquidity
            CASE WHEN TRY_CAST(b.BS_CUR_LIAB AS DOUBLE) != 0
                 THEN TRY_CAST(b.BS_CASH_NEAR_CASH_ITEM AS DOUBLE)
                      / TRY_CAST(b.BS_CUR_LIAB AS DOUBLE)
            END AS CASH_TO_CURRENT_LIABILITIES,

            CASE WHEN TRY_CAST(b.BS_CUR_LIAB AS DOUBLE) != 0
                 THEN TRY_CAST(c.CF_CASH_FROM_OPER AS DOUBLE)
                      / TRY_CAST(b.BS_CUR_LIAB AS DOUBLE)
            END AS OPERATING_CASH_FLOW_TO_CURRENT_LIABILITIES,

            CASE WHEN TRY_CAST(b.BS_CUR_LIAB AS DOUBLE) != 0
                 THEN (TRY_CAST(b.BS_CUR_ASSET_REPORT AS DOUBLE)
                       - COALESCE(TRY_CAST(b.BS_INVENTORIES AS DOUBLE), 0))
                      / TRY_CAST(b.BS_CUR_LIAB AS DOUBLE)
            END AS QUICK_RATIO,

            CASE WHEN TRY_CAST(b.BS_TOT_ASSET AS DOUBLE) != 0
                 THEN (TRY_CAST(b.BS_CUR_ASSET_REPORT AS DOUBLE)
                       - TRY_CAST(b.BS_CUR_LIAB AS DOUBLE))
                      / TRY_CAST(b.BS_TOT_ASSET AS DOUBLE)
            END AS WORKING_CAPITAL_TO_TOTAL_ASSETS,

            -- Leverage (market-dependent ones filled later in assembly)
            CASE WHEN TRY_CAST(b.BS_TOT_ASSET AS DOUBLE) != 0
                 THEN (COALESCE(TRY_CAST(b.BS_ST_BORROW AS DOUBLE), 0)
                       + COALESCE(TRY_CAST(b.BS_LT_BORROW AS DOUBLE), 0)
                       - COALESCE(TRY_CAST(b.BS_CASH_NEAR_CASH_ITEM AS DOUBLE), 0))
                      / TRY_CAST(b.BS_TOT_ASSET AS DOUBLE)
            END AS NET_DEBT_TO_TOTAL_ASSETS,

            CASE WHEN TRY_CAST(b.BS_TOT_ASSET AS DOUBLE) != 0
                 THEN TRY_CAST(b.BS_RETAIN_EARN AS DOUBLE) / TRY_CAST(b.BS_TOT_ASSET AS DOUBLE)
            END AS RETAINED_EARNINGS_TO_TOTAL_ASSETS,

            CASE WHEN (COALESCE(TRY_CAST(b.BS_ST_BORROW AS DOUBLE), 0)
                       + COALESCE(TRY_CAST(b.BS_LT_BORROW AS DOUBLE), 0)
                       + COALESCE(TRY_CAST(b.BS_TOT_EQY AS DOUBLE), 0)) != 0
                 THEN (COALESCE(TRY_CAST(b.BS_ST_BORROW AS DOUBLE), 0)
                       + COALESCE(TRY_CAST(b.BS_LT_BORROW AS DOUBLE), 0))
                      / (COALESCE(TRY_CAST(b.BS_ST_BORROW AS DOUBLE), 0)
                         + COALESCE(TRY_CAST(b.BS_LT_BORROW AS DOUBLE), 0)
                         + COALESCE(TRY_CAST(b.BS_TOT_EQY AS DOUBLE), 0))
            END AS TOTAL_DEBT_TO_CAPITAL,

            CASE WHEN (COALESCE(TRY_CAST(b.BS_ST_BORROW AS DOUBLE), 0)
                       + COALESCE(TRY_CAST(b.BS_LT_BORROW AS DOUBLE), 0)
                       - COALESCE(TRY_CAST(b.BS_CASH_NEAR_CASH_ITEM AS DOUBLE), 0)
                       + COALESCE(TRY_CAST(b.BS_TOT_EQY AS DOUBLE), 0)) != 0
                 THEN (COALESCE(TRY_CAST(b.BS_ST_BORROW AS DOUBLE), 0)
                       + COALESCE(TRY_CAST(b.BS_LT_BORROW AS DOUBLE), 0)
                       - COALESCE(TRY_CAST(b.BS_CASH_NEAR_CASH_ITEM AS DOUBLE), 0))
                      / (COALESCE(TRY_CAST(b.BS_ST_BORROW AS DOUBLE), 0)
                         + COALESCE(TRY_CAST(b.BS_LT_BORROW AS DOUBLE), 0)
                         - COALESCE(TRY_CAST(b.BS_CASH_NEAR_CASH_ITEM AS DOUBLE), 0)
                         + COALESCE(TRY_CAST(b.BS_TOT_EQY AS DOUBLE), 0))
            END AS NET_DEBT_TO_CAPITAL,

            -- Asset structure
            CASE WHEN COALESCE(TRY_CAST(i.IS_DEPR_EXP AS DOUBLE), TRY_CAST(c.CF_DEPR_AMORT AS DOUBLE)) != 0
                 THEN ABS(TRY_CAST(c.CF_CAP_EXPEND_INC_FIX_ASSET AS DOUBLE))
                      / COALESCE(TRY_CAST(i.IS_DEPR_EXP AS DOUBLE), TRY_CAST(c.CF_DEPR_AMORT AS DOUBLE))
            END AS CAPEX_TO_DEPRECIATION,

            CASE WHEN TRY_CAST(b.BS_TOT_ASSET AS DOUBLE) != 0
                 THEN (COALESCE(TRY_CAST(b.BS_DISCLOSED_INTANGIBLES AS DOUBLE), 0)
                       + COALESCE(TRY_CAST(b.BS_GOODWILL AS DOUBLE), 0))
                      / TRY_CAST(b.BS_TOT_ASSET AS DOUBLE)
            END AS INTANGIBLE_ASSETS_TO_TOTAL_ASSETS,

            CASE WHEN TRY_CAST(b.BS_TOT_ASSET AS DOUBLE) != 0
                 THEN TRY_CAST(b.BS_INVENTORIES AS DOUBLE)
                      / TRY_CAST(b.BS_TOT_ASSET AS DOUBLE)
            END AS INVENTORY_TO_TOTAL_ASSETS,

            CASE WHEN (COALESCE(TRY_CAST(b.BS_ST_BORROW AS DOUBLE), 0)
                      + COALESCE(TRY_CAST(b.BS_LT_BORROW AS DOUBLE), 0)) != 0
                 THEN TRY_CAST(b.BS_NET_FIX_ASSET AS DOUBLE)
                      / (COALESCE(TRY_CAST(b.BS_ST_BORROW AS DOUBLE), 0)
                         + COALESCE(TRY_CAST(b.BS_LT_BORROW AS DOUBLE), 0))
            END AS PPE_TO_TOTAL_DEBT,

            CASE WHEN TRY_CAST(b.BS_TOT_ASSET AS DOUBLE) != 0
                 THEN TRY_CAST(b.BS_ACCT_NOTE_RCV AS DOUBLE)
                      / TRY_CAST(b.BS_TOT_ASSET AS DOUBLE)
            END AS RECEIVABLES_TO_TOTAL_ASSETS,

            CASE WHEN TRY_CAST(b.BS_TOT_ASSET AS DOUBLE) != 0
                 THEN (TRY_CAST(b.BS_TOT_ASSET AS DOUBLE)
                       - COALESCE(TRY_CAST(b.BS_DISCLOSED_INTANGIBLES AS DOUBLE), 0)
                       - COALESCE(TRY_CAST(b.BS_GOODWILL AS DOUBLE), 0))
                      / TRY_CAST(b.BS_TOT_ASSET AS DOUBLE)
            END AS TANGIBLE_ASSETS_TO_TOTAL_ASSETS,

            -- Payout
            CASE WHEN TRY_CAST(i.NET_INCOME AS DOUBLE) != 0
                 THEN ABS(TRY_CAST(i.IS_TOT_CASH_COM_DVD AS DOUBLE))
                      / TRY_CAST(i.NET_INCOME AS DOUBLE)
            END AS DIVIDEND_PAYOUT_RATIO,

            CASE WHEN ABS(COALESCE(TRY_CAST(ip.IS_TOT_CASH_COM_DVD AS DOUBLE), 0)) > 0
                  AND ABS(TRY_CAST(i.IS_TOT_CASH_COM_DVD AS DOUBLE))
                      < ABS(TRY_CAST(ip.IS_TOT_CASH_COM_DVD AS DOUBLE))
                 THEN 1 ELSE 0
            END AS DIVIDEND_CUT_FLAG,

            CASE WHEN TRY_CAST(ip.IS_SH_FOR_DILUTED_EPS AS DOUBLE) IS NOT NULL
                  AND TRY_CAST(ip.IS_SH_FOR_DILUTED_EPS AS DOUBLE) != 0
                 THEN TRY_CAST(i.IS_SH_FOR_DILUTED_EPS AS DOUBLE)
                      / TRY_CAST(ip.IS_SH_FOR_DILUTED_EPS AS DOUBLE) - 1.0
            END AS SHARE_COUNT_GROWTH_1Y,

            -- Store total_debt and net_debt for market-dependent features in assembly
            (COALESCE(TRY_CAST(b.BS_ST_BORROW AS DOUBLE), 0)
             + COALESCE(TRY_CAST(b.BS_LT_BORROW AS DOUBLE), 0)) AS _total_debt,
            COALESCE(TRY_CAST(b.BS_CASH_NEAR_CASH_ITEM AS DOUBLE), 0) AS _cash,
            TRY_CAST(i.SALES_REV_TURN AS DOUBLE) AS _revenue,
            TRY_CAST(b.BS_TOT_ASSET AS DOUBLE) AS _total_assets

        FROM is_pit i
        LEFT JOIN bs_pit b
          ON i.u3_company_number = b.u3_company_number
         AND i.observation_date = b.observation_date
        LEFT JOIN cf_pit c
          ON i.u3_company_number = c.u3_company_number
         AND i.observation_date = c.observation_date
        LEFT JOIN bs_prior bp
          ON i.u3_company_number = bp.u3_company_number
         AND i.observation_date = bp.observation_date
        LEFT JOIN is_prior ip
          ON i.u3_company_number = ip.u3_company_number
         AND i.observation_date = ip.observation_date
    """)
    n = con.execute("SELECT COUNT(*) FROM gpt_financial").fetchone()[0]
    logger.info(f"  GPT financial features: {n:,} rows")


# ── Step 5: Market features ──────────────────────────────────────────────────

def build_market_features(con: duckdb.DuckDBPyConnection) -> None:
    """Compute market cap returns at 1M/3M/6M/12M lookbacks."""
    # Use year/month comparison instead of INTERVAL subtraction to avoid
    # month-end date mismatches (e.g. Nov 30 - 3 months = Aug 30, not Aug 31)
    con.execute("""
        CREATE TABLE market_features AS
        WITH monthly AS (
            SELECT u3_company_number, observation_date, marketcap_eom,
                   YEAR(observation_date) * 12 + MONTH(observation_date) AS ym
            FROM obs_grid
        )
        SELECT
            cur.u3_company_number,
            cur.observation_date,
            CASE WHEN m1.marketcap_eom > 0
                 THEN cur.marketcap_eom / m1.marketcap_eom - 1.0 END AS MARKET_CAP_RETURN_1M,
            CASE WHEN m3.marketcap_eom > 0
                 THEN cur.marketcap_eom / m3.marketcap_eom - 1.0 END AS MARKET_CAP_RETURN_3M,
            CASE WHEN m6.marketcap_eom > 0
                 THEN cur.marketcap_eom / m6.marketcap_eom - 1.0 END AS MARKET_CAP_RETURN_6M,
            CASE WHEN m12.marketcap_eom > 0
                 THEN cur.marketcap_eom / m12.marketcap_eom - 1.0 END AS MARKET_CAP_RETURN_12M,
            cur.marketcap_eom AS _marketcap
        FROM monthly cur
        LEFT JOIN monthly m1
          ON cur.u3_company_number = m1.u3_company_number
         AND m1.ym = cur.ym - 1
        LEFT JOIN monthly m3
          ON cur.u3_company_number = m3.u3_company_number
         AND m3.ym = cur.ym - 3
        LEFT JOIN monthly m6
          ON cur.u3_company_number = m6.u3_company_number
         AND m6.ym = cur.ym - 6
        LEFT JOIN monthly m12
          ON cur.u3_company_number = m12.u3_company_number
         AND m12.ym = cur.ym - 12
    """)
    n = con.execute("SELECT COUNT(*) FROM market_features").fetchone()[0]
    logger.info(f"  Market features: {n:,} rows")


# ── Step 6: CRI features ─────────────────────────────────────────────────────

def build_cri_features(con: duckdb.DuckDBPyConnection) -> None:
    """Extract 13 CRI features: 9 direct + 4 trend (12-month diff)."""
    direct_cols = ", ".join(
        f'cur."{csv_col}" AS "{feat_name}"'
        for feat_name, csv_col in CRI_DIRECT.items()
    )
    trend_cols = ", ".join(
        f'(TRY_CAST(cur."{csv_col}" AS DOUBLE) - TRY_CAST(lag."{csv_col}" AS DOUBLE)) AS "{feat_name}"'
        for feat_name, csv_col in CRI_TREND.items()
    )

    con.execute(f"""
        CREATE TABLE cri_features AS
        SELECT
            g.u3_company_number,
            g.observation_date,
            {direct_cols},
            {trend_cols}
        FROM obs_grid g
        LEFT JOIN cri cur
          ON (cur.Company_Number // 1000) = g.u3_company_number
         AND cur.year = YEAR(g.observation_date)
         AND cur.month = MONTH(g.observation_date)
        LEFT JOIN cri lag
          ON (lag.Company_Number // 1000) = g.u3_company_number
         AND lag.year = YEAR(g.observation_date) - 1
         AND lag.month = MONTH(g.observation_date)
    """)
    n = con.execute("SELECT COUNT(*) FROM cri_features").fetchone()[0]
    n_nonnull = con.execute("SELECT COUNT(*) FROM cri_features WHERE dtdlevel IS NOT NULL").fetchone()[0]
    logger.info(f"  CRI features: {n:,} rows ({n_nonnull:,} with data)")


# ── Step 7: Labels ───────────────────────────────────────────────────────────

def build_labels(con: duckdb.DuckDBPyConnection) -> None:
    """Attach default labels and censor post-default observations."""
    con.execute("""
        CREATE TABLE labels AS
        WITH
        first_default AS (
            SELECT u3_company_number,
                   MIN(event_date) AS first_default_date
            FROM credit_events
            GROUP BY u3_company_number
        )
        SELECT
            g.u3_company_number,
            g.observation_date,
            fd.first_default_date,
            CASE WHEN fd.first_default_date IS NOT NULL
                  AND fd.first_default_date > g.observation_date
                  AND fd.first_default_date <= g.observation_date + INTERVAL '3 months'
                 THEN TRUE ELSE FALSE END AS default_within_3m,
            CASE WHEN fd.first_default_date IS NOT NULL
                  AND fd.first_default_date > g.observation_date
                  AND fd.first_default_date <= g.observation_date + INTERVAL '6 months'
                 THEN TRUE ELSE FALSE END AS default_within_6m,
            CASE WHEN fd.first_default_date IS NOT NULL
                  AND fd.first_default_date > g.observation_date
                  AND fd.first_default_date <= g.observation_date + INTERVAL '12 months'
                 THEN TRUE ELSE FALSE END AS default_within_12m
        FROM obs_grid g
        LEFT JOIN first_default fd ON g.u3_company_number = fd.u3_company_number
        -- Censor: exclude observations AFTER the first default
        WHERE fd.first_default_date IS NULL
           OR g.observation_date < fd.first_default_date
    """)
    n = con.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    n_def = con.execute("SELECT COUNT(*) FROM labels WHERE default_within_6m").fetchone()[0]
    logger.info(f"  Labels: {n:,} rows ({n_def:,} positive at 6m horizon)")


# ── Step 8: Assemble ─────────────────────────────────────────────────────────

def assemble_and_write(con: duckdb.DuckDBPyConnection, output_path: str) -> None:
    """JOIN all feature tables and write the final parquet."""

    # Cast CR2 VARCHAR columns to DOUBLE, converting 'N.A.' to NULL
    cr2_cols = ", ".join(
        f'TRY_CAST(NULLIF(cr2."{c}", \'N.A.\') AS DOUBLE) AS "{c}"'
        for c in CR2_FEATURES
    )

    gpt_fin_cols = [
        "EBITDA_TO_TOTAL_ASSETS", "EBIT_TO_TOTAL_ASSETS", "ROE",
        "EBIT_TO_INTEREST_EXPENSE", "CFO_TO_INTEREST_EXPENSE",
        "CFO_TO_TOTAL_ASSETS", "CFO_TO_TOTAL_DEBT",
        "NET_DEBT_TO_EBITDA", "TOTAL_DEBT_TO_EBITDA",
        "FCF_TO_TOTAL_ASSETS", "FCF_TO_TOTAL_DEBT",
        "CASH_TO_CURRENT_LIABILITIES",
        "OPERATING_CASH_FLOW_TO_CURRENT_LIABILITIES",
        "QUICK_RATIO", "WORKING_CAPITAL_TO_TOTAL_ASSETS",
        "NET_DEBT_TO_TOTAL_ASSETS", "RETAINED_EARNINGS_TO_TOTAL_ASSETS",
        "TOTAL_DEBT_TO_CAPITAL", "NET_DEBT_TO_CAPITAL",
        "CAPEX_TO_DEPRECIATION", "INTANGIBLE_ASSETS_TO_TOTAL_ASSETS",
        "INVENTORY_TO_TOTAL_ASSETS", "PPE_TO_TOTAL_DEBT",
        "RECEIVABLES_TO_TOTAL_ASSETS", "TANGIBLE_ASSETS_TO_TOTAL_ASSETS",
        "DIVIDEND_PAYOUT_RATIO", "DIVIDEND_CUT_FLAG", "SHARE_COUNT_GROWTH_1Y",
    ]
    gpt_fin_sql = ", ".join(f'gf."{c}"' for c in gpt_fin_cols)

    cri_all = list(CRI_DIRECT.keys()) + list(CRI_TREND.keys())
    cri_sql = ", ".join(f'TRY_CAST(cri."{c}" AS DOUBLE) AS "{c}"' for c in cri_all)

    con.execute(f"""
        COPY (
            SELECT
                -- Identity
                lab.u3_company_number,
                lab.observation_date,
                comp.company_name,
                comp.country,
                comp.industry_sector,

                -- CR2 (56)
                {cr2_cols},

                -- Market (4)
                mkt.MARKET_CAP_RETURN_1M,
                mkt.MARKET_CAP_RETURN_3M,
                mkt.MARKET_CAP_RETURN_6M,
                mkt.MARKET_CAP_RETURN_12M,

                -- GPT financial (28 pure financial)
                {gpt_fin_sql},

                -- GPT market-dependent leverage (4) — need marketcap + debt from both tables
                CASE WHEN (gf._total_debt + mkt._marketcap) > 0
                     THEN gf._total_debt / (gf._total_debt + mkt._marketcap)
                END AS MARKET_LEVERAGE_RATIO,

                CASE WHEN mkt._marketcap > 0
                     THEN (gf._total_debt - gf._cash) / mkt._marketcap
                END AS NET_DEBT_TO_MARKET_CAP,

                CASE WHEN TRY_CAST(gf._revenue AS DOUBLE) > 0
                     THEN (mkt._marketcap + gf._total_debt - gf._cash)
                          / gf._revenue
                END AS ENTERPRISE_VALUE_TO_SALES,

                CASE WHEN gf._total_assets > 0
                     THEN (mkt._marketcap + gf._total_debt - gf._cash)
                          / gf._total_assets
                END AS ENTERPRISE_VALUE_TO_ASSETS,

                -- CRI (13)
                {cri_sql},

                -- Labels
                lab.default_within_3m,
                lab.default_within_6m,
                lab.default_within_12m,
                lab.first_default_date

            FROM labels lab
            LEFT JOIN companies comp ON lab.u3_company_number = comp.u3_company_number
            LEFT JOIN cr2_features cr2
              ON lab.u3_company_number = cr2.u3_company_number
             AND lab.observation_date = cr2.observation_date
            LEFT JOIN market_features mkt
              ON lab.u3_company_number = mkt.u3_company_number
             AND lab.observation_date = mkt.observation_date
            LEFT JOIN gpt_financial gf
              ON lab.u3_company_number = gf.u3_company_number
             AND lab.observation_date = gf.observation_date
            LEFT JOIN cri_features cri
              ON lab.u3_company_number = cri.u3_company_number
             AND lab.observation_date = cri.observation_date

            ORDER BY lab.u3_company_number, lab.observation_date
        ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    # Report stats
    stats = con.execute(f"""
        SELECT COUNT(*) as n,
               COUNT(DISTINCT u3_company_number) as companies
        FROM read_parquet('{output_path}')
    """).fetchone()
    logger.info(f"  Output: {stats[0]:,} rows, {stats[1]:,} companies → {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build credit feature matrix")
    parser.add_argument("--output", default="features_matrix.parquet",
                        help="Output parquet path")
    parser.add_argument("--year-from", type=int, default=2000,
                        help="Start year (default: 2000)")
    parser.add_argument("--year-to", type=int, default=2025,
                        help="End year (default: 2025)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build grid and report stats, skip feature computation")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    con = duckdb.connect()
    con.execute("SET memory_limit = '12GB'")
    con.execute("SET threads TO 4")
    con.execute("SET preserve_insertion_order = false")

    logger.info("Registering data sources...")
    _register_fundamentals(con)

    logger.info("Loading company metadata from PostgreSQL...")
    with get_session() as session:
        load_company_metadata(con, session)

    logger.info("Building observation grid...")
    with _timer("Observation grid"):
        n = build_observation_grid(con, args.year_from, args.year_to)

    if args.dry_run:
        logger.info("Dry-run mode — skipping feature computation")
        return

    logger.info("Building CR2 features (56)...")
    with _timer("CR2"):
        build_cr2_features(con)

    logger.info("Building GPT-proposed financial features (32)...")
    with _timer("GPT financial"):
        build_gpt_financial_features(con)

    logger.info("Building market features (4)...")
    with _timer("Market"):
        build_market_features(con)

    logger.info("Building CRI features (13)...")
    with _timer("CRI"):
        build_cri_features(con)

    logger.info("Building labels...")
    with _timer("Labels"):
        build_labels(con)

    logger.info("Assembling final matrix...")
    with _timer("Assembly"):
        assemble_and_write(con, args.output)

    con.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
