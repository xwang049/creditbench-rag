"""Load credit events from Excel."""

import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from src.db.models import CreditEvent

logger = logging.getLogger(__name__)


def clean_value(value):
    """Convert NaN and empty strings to None."""
    if pd.isna(value):
        return None
    if isinstance(value, str) and value.strip() == '':
        return None
    return value


def convert_to_date(value):
    """Convert datetime to date, handle None."""
    if pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return pd.to_datetime(value).date()
        except:
            return None
    return value


def load_credit_events(session: Session, excel_path: Path) -> int:
    """Load credit events from Excel."""
    logger.info("Loading credit events...")
    
    df = pd.read_excel(excel_path, sheet_name="Sheet1")
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    logger.info(f"Read {len(df)} credit event rows from Excel")
    
    session.query(CreditEvent).delete()
    session.commit()
    logger.info("Cleared existing credit_events data")
    
    records = []
    batch_size = 1000
    
    for idx, row in df.iterrows():
        # Strip subcategory if it's a string
        subcategory = clean_value(row.get('subcategory'))
        if isinstance(subcategory, str):
            subcategory = subcategory.strip()
        
        record = CreditEvent(
            u3_company_number=int(row['u3_company_number']),
            id_bb_company=clean_value(row.get('id_bb_company')),
            announcement_date=convert_to_date(row.get('announcement_date')),
            effective_date=convert_to_date(row.get('effective_date')),
            event_type=clean_value(row.get('event_type')),
            action_name=clean_value(row.get('action_name')),
            subcategory=subcategory,
        )
        records.append(record)
        
        if (idx + 1) % 5000 == 0:
            logger.info(f"  Prepared {idx + 1} credit events...")
        
        if len(records) >= batch_size:
            session.bulk_save_objects(records)
            session.commit()
            logger.info(f"  Inserted {idx + 1} credit events...")
            records = []
    
    if records:
        session.bulk_save_objects(records)
        session.commit()
    
    total = len(df)
    logger.info(f"[OK] Loaded {total} credit event records")
    return total


def load_credit_event_data(session: Session, data_dir: Path) -> dict:
    """Load credit events from data directory."""
    excel_path = data_dir / "Credit Events.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(f"Credit events file not found: {excel_path}")
    
    count = load_credit_events(session, excel_path)
    return {'credit_events': count}
