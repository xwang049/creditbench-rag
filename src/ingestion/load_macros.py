"""Load macroeconomic indicator data from files."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from src.db.models import MacroIndicator
from src.db.session import session_scope

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_macros(
    file_path: str | Path,
    session: Optional[Session] = None,
) -> int:
    """
    Load macroeconomic indicator data from Excel/CSV file.

    Args:
        file_path: Path to the data file
        session: Database session (optional, will create if not provided)

    Returns:
        Number of macro indicators loaded
    """
    file_path = Path(file_path)
    logger.info(f"Loading macro indicators from {file_path}")

    # Read data file
    if file_path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    logger.info(f"Found {len(df)} macro indicators in file")

    def _load(session: Session) -> int:
        count = 0
        for _, row in df.iterrows():
            # Parse indicator date
            indicator_date = row.get("indicator_date")
            if isinstance(indicator_date, str):
                indicator_date = pd.to_datetime(indicator_date)

            indicator = MacroIndicator(
                indicator_date=indicator_date,
                country=row.get("country"),
                gdp_growth=row.get("gdp_growth"),
                inflation_rate=row.get("inflation_rate"),
                unemployment_rate=row.get("unemployment_rate"),
                interest_rate=row.get("interest_rate"),
                exchange_rate=row.get("exchange_rate"),
                stock_index=row.get("stock_index"),
                credit_spread=row.get("credit_spread"),
            )
            session.add(indicator)
            count += 1

            if count % 100 == 0:
                session.commit()
                logger.info(f"Loaded {count} indicators...")

        session.commit()
        return count

    if session:
        return _load(session)
    else:
        with session_scope() as session:
            return _load(session)
