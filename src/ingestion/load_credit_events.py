"""Load credit event data from files."""

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from src.db.models import CreditEvent, Company
from src.db.session import session_scope
from src.rag.embeddings import generate_embedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_credit_events(
    file_path: str | Path,
    session: Optional[Session] = None,
    generate_embeddings: bool = True
) -> int:
    """
    Load credit event data from Excel/CSV file.

    Args:
        file_path: Path to the data file
        session: Database session (optional, will create if not provided)
        generate_embeddings: Whether to generate embeddings for events

    Returns:
        Number of credit events loaded
    """
    file_path = Path(file_path)
    logger.info(f"Loading credit events from {file_path}")

    # Read data file
    if file_path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    logger.info(f"Found {len(df)} credit events in file")

    def _load(session: Session) -> int:
        # Build company_code to id mapping
        companies = {c.company_code: c.id for c in session.query(Company).all()}
        logger.info(f"Found {len(companies)} companies in database")

        count = 0
        skipped = 0

        for _, row in df.iterrows():
            company_code = row.get("company_code")
            if company_code not in companies:
                logger.warning(f"Company {company_code} not found, skipping event")
                skipped += 1
                continue

            # Parse event date
            event_date = row.get("event_date")
            if isinstance(event_date, str):
                event_date = pd.to_datetime(event_date)

            # Create text for embedding
            text_for_embedding = (
                f"Event Type: {row.get('event_type', '')}\n"
                f"Date: {event_date}\n"
                f"Rating: {row.get('rating_before', '')} -> {row.get('rating_after', '')}\n"
                f"Description: {row.get('description', '')}"
            )

            # Generate embedding if requested
            embedding = None
            if generate_embeddings:
                try:
                    embedding = generate_embedding(text_for_embedding)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding: {e}")

            event = CreditEvent(
                company_id=companies[company_code],
                event_date=event_date,
                event_type=row.get("event_type"),
                rating_before=row.get("rating_before"),
                rating_after=row.get("rating_after"),
                description=row.get("description"),
                total_assets=row.get("total_assets"),
                total_liabilities=row.get("total_liabilities"),
                revenue=row.get("revenue"),
                ebitda=row.get("ebitda"),
                embedding=embedding,
            )
            session.add(event)
            count += 1

            if count % 100 == 0:
                session.commit()
                logger.info(f"Loaded {count} events...")

        session.commit()
        logger.info(f"Successfully loaded {count} events, skipped {skipped}")
        return count

    if session:
        return _load(session)
    else:
        with session_scope() as session:
            return _load(session)
