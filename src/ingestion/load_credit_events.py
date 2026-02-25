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


def load_credit_events(session: Session, excel_path: Path) -> dict:
    """Load credit events incrementally - only insert records not already in DB."""
    logger.info("Loading credit events (incremental)...")

    df = pd.read_excel(excel_path, sheet_name="Sheet1")
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    logger.info(f"Read {len(df)} rows from Excel")

    # Build set of existing event keys from DB
    existing_rows = session.query(
        CreditEvent.u3_company_number,
        CreditEvent.announcement_date,
        CreditEvent.effective_date,
        CreditEvent.event_type,
        CreditEvent.action_name,
    ).all()
    existing_keys = {
        (r.u3_company_number, r.announcement_date, r.effective_date, r.event_type, r.action_name)
        for r in existing_rows
    }
    logger.info(f"{len(existing_keys)} existing events in DB, checking for new ones...")

    records = []
    batch_size = 1000
    inserted = 0
    skipped = 0

    for _, row in df.iterrows():
        u3 = int(row['u3_company_number'])
        ann_date = convert_to_date(row.get('announcement_date'))
        event_type = clean_value(row.get('event_type'))
        action_name = clean_value(row.get('action_name'))

        eff_date = convert_to_date(row.get('effective_date'))
        if (u3, ann_date, eff_date, event_type, action_name) in existing_keys:
            skipped += 1
            continue

        subcategory = clean_value(row.get('subcategory'))
        if isinstance(subcategory, str):
            subcategory = subcategory.strip()

        records.append(CreditEvent(
            u3_company_number=u3,
            id_bb_company=clean_value(row.get('id_bb_company')),
            announcement_date=ann_date,
            effective_date=convert_to_date(row.get('effective_date')),
            event_type=event_type,
            action_name=action_name,
            subcategory=subcategory,
        ))

        if len(records) >= batch_size:
            session.bulk_save_objects(records)
            session.commit()
            inserted += len(records)
            logger.info(f"  Inserted {inserted} new events so far...")
            records = []

    if records:
        session.bulk_save_objects(records)
        session.commit()
        inserted += len(records)

    logger.info(f"[OK] {inserted} new events inserted, {skipped} already existed")
    return inserted, skipped


def load_credit_event_data(session: Session, data_dir: Path) -> dict:
    """Load credit events from data directory."""
    excel_path = data_dir / "Credit Events.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(f"Credit events file not found: {excel_path}")

    inserted, skipped = load_credit_events(session, excel_path)
    return {'credit_events_new': inserted, 'credit_events_skipped': skipped}
