"""Load company data from files."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from src.db.models import Company
from src.db.session import session_scope
from src.rag.embeddings import generate_embedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_companies(
    file_path: str | Path,
    session: Optional[Session] = None,
    generate_embeddings: bool = True
) -> int:
    """
    Load company data from Excel/CSV file.

    Args:
        file_path: Path to the data file
        session: Database session (optional, will create if not provided)
        generate_embeddings: Whether to generate embeddings for companies

    Returns:
        Number of companies loaded
    """
    file_path = Path(file_path)
    logger.info(f"Loading companies from {file_path}")

    # Read data file
    if file_path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # Expected columns (adjust based on your actual data)
    # company_code, company_name, industry, sector, country, listing_status
    logger.info(f"Found {len(df)} companies in file")

    def _load(session: Session) -> int:
        count = 0
        for _, row in df.iterrows():
            # Create text for embedding
            text_for_embedding = (
                f"Company: {row.get('company_name', '')}\n"
                f"Industry: {row.get('industry', '')}\n"
                f"Sector: {row.get('sector', '')}\n"
                f"Country: {row.get('country', '')}"
            )

            # Generate embedding if requested
            embedding = None
            if generate_embeddings:
                try:
                    embedding = generate_embedding(text_for_embedding)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for {row.get('company_code')}: {e}")

            company = Company(
                company_code=row.get("company_code"),
                company_name=row.get("company_name"),
                industry=row.get("industry"),
                sector=row.get("sector"),
                country=row.get("country"),
                listing_status=row.get("listing_status"),
                embedding=embedding,
            )
            session.add(company)
            count += 1

            if count % 100 == 0:
                session.commit()
                logger.info(f"Loaded {count} companies...")

        session.commit()
        return count

    if session:
        return _load(session)
    else:
        with session_scope() as session:
            return _load(session)
