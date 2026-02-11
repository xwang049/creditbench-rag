"""Load company information and industry mapping from Excel."""

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from src.db.models import Company, IndustryMapping

logger = logging.getLogger(__name__)


def clean_value(value):
    """Convert NaN and empty strings to None."""
    if pd.isna(value):
        return None
    if isinstance(value, str) and value.strip() == '':
        return None
    return value


def load_industry_mapping(session: Session, excel_path: Path) -> int:
    """Load industry code mapping from Excel."""
    logger.info("Loading industry mapping...")
    df = pd.read_excel(excel_path, sheet_name="Industry Code Mapping")
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    logger.info(f"Read {len(df)} industry mapping rows from Excel")
    
    session.query(IndustryMapping).delete()
    session.commit()
    logger.info("Cleared existing industry_mapping data")
    
    records = []
    for idx, row in df.iterrows():
        record = IndustryMapping(
            industry_sector=clean_value(row.get('industry_sector')),
            industry_sector_num=clean_value(row.get('industry_sector_num')),
            industry_group=clean_value(row.get('industry_group')),
            industry_group_num=clean_value(row.get('industry_group_num')),
            industry_subgroup=clean_value(row.get('industry_subgroup')),
            industry_subgroup_num=clean_value(row.get('industry_subgroup_num')),
        )
        records.append(record)
        if (idx + 1) % 1000 == 0:
            logger.info(f"  Prepared {idx + 1} industry records...")
    
    logger.info(f"Inserting {len(records)} industry mapping records...")
    session.bulk_save_objects(records)
    session.commit()
    logger.info(f"[OK] Loaded {len(records)} industry mapping records")
    return len(records)


def load_companies(session: Session, excel_path: Path) -> int:
    """Load company information from Excel."""
    logger.info("Loading company information...")
    df = pd.read_excel(excel_path, sheet_name="Company Information")
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    logger.info(f"Read {len(df)} company rows from Excel")
    
    session.query(Company).delete()
    session.commit()
    logger.info("Cleared existing companies data")
    
    records = []
    batch_size = 1000
    
    for idx, row in df.iterrows():
        record = Company(
            u3_company_number=int(row['u3_company_number']),
            id_bb_unique=clean_value(row.get('id_bb_unique')),
            id_bb_company=clean_value(row.get('id_bb_company')),
            ticker=clean_value(row.get('ticker')),
            company_name=clean_value(row.get('company_name')),
            country_name=clean_value(row.get('country_name')),
            security_type=clean_value(row.get('security_type')),
            market_status=clean_value(row.get('market_status')),
            prime_exchange=clean_value(row.get('prime_exchange')),
            domicile=clean_value(row.get('domicile')),
            industry_sector_num=clean_value(row.get('industry_sector_num')),
            industry_group_num=clean_value(row.get('industry_group_num')),
            industry_subgroup_num=clean_value(row.get('industry_subgroup_num')),
            id_isin=clean_value(row.get('id_isin')),
            id_cusip=clean_value(row.get('id_cusip')),
        )
        records.append(record)
        
        if len(records) >= batch_size:
            session.bulk_save_objects(records)
            session.commit()
            logger.info(f"  Inserted {idx + 1} companies...")
            records = []
    
    if records:
        session.bulk_save_objects(records)
        session.commit()
    
    total = len(df)
    logger.info(f"[OK] Loaded {total} company records")
    return total


def load_company_data(session: Session, data_dir: Path) -> dict:
    """Load both industry mapping and company information."""
    excel_path = data_dir / "Company Information.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(f"Company file not found: {excel_path}")
    
    industry_count = load_industry_mapping(session, excel_path)
    company_count = load_companies(session, excel_path)
    
    return {
        'industry_mapping': industry_count,
        'companies': company_count,
    }
