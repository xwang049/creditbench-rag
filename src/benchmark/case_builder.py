"""Build experiment cases for Group 1 (default) and Group 2 (control).

A "case" is the unit of prediction:
    (company, cutoff_date, label)

where cutoff_date is the last date of data the LLM is allowed to see,
and label is True (company DID default) or False (control, no event).
"""

import logging
import random
from dataclasses import dataclass
from datetime import date
from typing import Optional

from dateutil.relativedelta import relativedelta
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.benchmark.config import BenchmarkConfig

logger = logging.getLogger(__name__)


@dataclass
class CaseRecord:
    """A single prediction case."""
    u3_company_number: int
    id_bb_company: Optional[int]
    ticker: Optional[str]
    company_name: str
    industry_sector: Optional[str]
    cutoff_date: date          # LLM data cutoff (= event_date - horizon for Group 1)
    event_date: Optional[date] # Actual event date (Group 1 only)
    label: bool                # True = defaulted within horizon, False = control
    group: str                 # "default" | "control"


def build_default_cases(
    session: Session,
    cfg: BenchmarkConfig,
) -> list[CaseRecord]:
    """Group 1: companies that had a Default Corp Action or Bankruptcy Filing.

    Selects one event per company (earliest default), computes cutoff_date as
    event_date - prediction_horizon_months.  Filters out companies where
    cutoff_date is before any risk_indicator data (no data to show the LLM).
    """
    rows = session.execute(text("""
        SELECT DISTINCT ON (ce.u3_company_number)
            ce.u3_company_number,
            ce.announcement_date AS event_date,
            c.company_name,
            c.ticker,
            c.id_bb_company,
            im.industry_sector
        FROM credit_events ce
        JOIN companies c ON ce.u3_company_number = c.u3_company_number
        LEFT JOIN industry_mapping im
               ON im.industry_subgroup_num = c.industry_subgroup_num
        WHERE ce.action_name IN ('Default Corp Action', 'Bankruptcy Filing')
          AND ce.announcement_date IS NOT NULL
        ORDER BY ce.u3_company_number, ce.announcement_date
    """)).fetchall()

    cases: list[CaseRecord] = []
    skipped_no_data = 0

    for row in rows:
        event_date: date = row.event_date

        # Year-range filter (for targeted experiments)
        if cfg.event_year_from and event_date.year < cfg.event_year_from:
            continue
        if cfg.event_year_to and event_date.year > cfg.event_year_to:
            continue

        cutoff_date = event_date - relativedelta(months=cfg.prediction_horizon_months)

        # Require id_bb_company for DuckDB fundamentals lookup
        if row.id_bb_company is None:
            skipped_no_data += 1
            continue

        cases.append(CaseRecord(
            u3_company_number=row.u3_company_number,
            id_bb_company=row.id_bb_company,
            ticker=row.ticker,
            company_name=row.company_name,
            industry_sector=row.industry_sector,
            cutoff_date=cutoff_date,
            event_date=event_date,
            label=True,
            group="default",
        ))

    logger.info(
        f"Default cases: {len(cases)} valid, {skipped_no_data} skipped (no id_bb_company)"
    )

    # Sample down to n_default_cases if needed
    rng = random.Random(cfg.random_seed)
    if len(cases) > cfg.n_default_cases:
        cases = rng.sample(cases, cfg.n_default_cases)
        logger.info(f"Sampled to {cfg.n_default_cases} default cases")

    return cases


def build_control_cases(
    session: Session,
    cfg: BenchmarkConfig,
    reference_cutoff_dates: list[date],
) -> list[CaseRecord]:
    """Group 2: Active US companies with no credit events at all.

    Each control company is assigned a cutoff_date sampled from the
    distribution of Group 1 cutoff dates (reference_cutoff_dates).
    """
    rows = session.execute(text("""
        SELECT
            c.u3_company_number,
            c.company_name,
            c.ticker,
            c.id_bb_company,
            im.industry_sector
        FROM companies c
        LEFT JOIN industry_mapping im
               ON im.industry_subgroup_num = c.industry_subgroup_num
        WHERE c.market_status = 'ACTV'
          AND c.country_name = 'United States'
          AND c.u3_company_number NOT IN (
              SELECT DISTINCT u3_company_number FROM credit_events
          )
        ORDER BY RANDOM()
    """)).fetchall()

    rng = random.Random(cfg.random_seed + 1)
    cases: list[CaseRecord] = []
    skipped_no_data = 0

    for row in rows:
        if len(cases) >= cfg.n_control_cases:
            break

        # Assign a cutoff_date from Group 1's distribution
        cutoff_date = rng.choice(reference_cutoff_dates)

        # Require id_bb_company for DuckDB fundamentals lookup
        if row.id_bb_company is None:
            skipped_no_data += 1
            continue

        cases.append(CaseRecord(
            u3_company_number=row.u3_company_number,
            id_bb_company=row.id_bb_company,
            ticker=row.ticker,
            company_name=row.company_name,
            industry_sector=row.industry_sector,
            cutoff_date=cutoff_date,
            event_date=None,
            label=False,
            group="control",
        ))

    logger.info(
        f"Control cases: {len(cases)} valid, {skipped_no_data} skipped (no id_bb_company)"
    )
    return cases


def build_cases(session: Session, cfg: BenchmarkConfig) -> list[CaseRecord]:
    """Build all experiment cases (Group 1 + Group 2).

    Returns:
        Combined list of default and control cases.
    """
    default_cases = build_default_cases(session, cfg)

    if not default_cases:
        logger.warning("No default cases found — check credit_events table")
        return []

    reference_dates = [c.cutoff_date for c in default_cases]
    control_cases = build_control_cases(session, cfg, reference_dates)

    all_cases = default_cases + control_cases
    logger.info(
        f"Total cases: {len(all_cases)} "
        f"({len(default_cases)} default, {len(control_cases)} control)"
    )
    return all_cases
