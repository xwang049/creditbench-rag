"""Load macroeconomic data from Excel."""

import logging
from pathlib import Path
from datetime import datetime, time, date
import re
import pandas as pd
from sqlalchemy.orm import Session
from src.db.models import MacroCommodities, MacroBondYields, MacroUS, MacroFX

logger = logging.getLogger(__name__)

def clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, str) and value.strip() in ('', 'NA', 'N/A'):
        return None
    if isinstance(value, (datetime, time)):
        logger.warning(f"Skipping datetime value: {value}")
        return None
    return value

def convert_to_date(value):
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

def parse_quarter_date(quarter_str):
    if pd.isna(quarter_str):
        return None
    try:
        match = re.match(r'Q(\d)\s+(\d{4})', str(quarter_str).strip())
        if not match:
            return None
        quarter, year = int(match.group(1)), int(match.group(2))
        qe = {1:(year,3,31), 2:(year,6,30), 3:(year,9,30), 4:(year,12,31)}
        if quarter not in qe:
            return None
        return datetime(*qe[quarter]).date()
    except:
        return None

def load_commodities(session: Session, excel_path: Path) -> int:
    logger.info("Loading commodity prices...")
    df = pd.read_excel(excel_path, sheet_name="Commodities", header=0)
    logger.info(f"Read commodities sheet: {df.shape}")
    session.query(MacroCommodities).delete()
    session.commit()

    cmap = {'WTI Crude':'wti_crude','Brent Crude':'brent_crude','Gasoline':'gasoline',
            'Heating Oil':'heating_oil','Gasoil':'gasoil','Natural Gas':'natural_gas',
            'Aluminum':'aluminum','Copper':'copper','Lead':'lead','Nickel':'nickel',
            'Zinc':'zinc','Gold':'gold','Silver':'silver','Wheat':'wheat','Corn':'corn',
            'Soybeans':'soybeans','Cotton':'cotton','Sugar':'sugar','Coffee':'coffee',
            'Cocoa':'cocoa','Kansas Financial Stress Index':'kansas_financial_stress',
            'Iron Ore':'iron_ore','Coal':'coal','Palm Oil':'palm_oil','Rubber':'rubber'}

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    date_records = {}

    for col in df.columns[1:]:
        col_name = str(col).strip()
        db_field = None
        for k, v in cmap.items():
            if k.lower() in col_name.lower():
                db_field = v
                break
        if not db_field:
            continue
        for idx, row in df.iterrows():
            dv = row[date_col]
            pv = clean_value(row[col])
            if pd.isna(dv):
                continue
            dk = dv.date()
            if dk not in date_records:
                date_records[dk] = {'date':dk}
            if db_field not in date_records[dk] and pv is not None:
                date_records[dk][db_field] = float(pv)

    records = [MacroCommodities(**date_records[dk]) for dk in sorted(date_records.keys())]
    logger.info(f"Inserting {len(records)} commodity records...")
    for i in range(0, len(records), 1000):
        session.bulk_save_objects(records[i:i+1000])
        session.commit()
        logger.info(f"  Inserted {min(i+1000, len(records))} records...")
    logger.info(f"[OK] Loaded {len(records)} commodity records")
    return len(records)

def load_bond_yields(session: Session, excel_path: Path) -> int:
    logger.info("Loading bond yields...")
    df = pd.read_excel(excel_path, sheet_name="Gov Bond Yield", header=0)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    logger.info(f"Read {len(df)} bond yield rows")
    session.query(MacroBondYields).delete()
    session.commit()

    records = []
    for idx, row in df.iterrows():
        dv = convert_to_date(row.get('data_date'))
        if not dv:
            continue
        rec = MacroBondYields(
            data_date=dv,
            us_1m=clean_value(row.get('us_generic_govt_1_month_yield')),
            us_3m=clean_value(row.get('us_generic_govt_3_month_yield')),
            us_6m=clean_value(row.get('us_generic_govt_6_month_yield')),
            us_1y=clean_value(row.get('us_generic_govt_12_month_yield')),
            us_2y=clean_value(row.get('us_generic_govt_2_year_yield')),
            us_3y=clean_value(row.get('us_generic_govt_3_year_yield')),
            us_5y=clean_value(row.get('us_generic_govt_5_year_yield')),
            us_7y=clean_value(row.get('us_generic_govt_7_year_yield')),
            us_10y=clean_value(row.get('us_generic_govt_10_year_yield')),
            us_30y=clean_value(row.get('us_generic_govt_30_year_yield')))
        records.append(rec)
        if len(records) >= 1000:
            session.bulk_save_objects(records)
            session.commit()
            records = []
    if records:
        session.bulk_save_objects(records)
        session.commit()
    logger.info(f"[OK] Loaded {len(df)} bond yield records")
    return len(df)

def load_us_macros(session: Session, excel_path: Path) -> int:
    logger.info("Loading US macro indicators...")
    df = pd.read_excel(excel_path, sheet_name="other US macros", header=0)
    session.query(MacroUS).delete()
    session.commit()

    date_records = {}
    cutoff = datetime(1990, 1, 1).date()
    i = 0
    while i < len(df.columns):
        col = df.columns[i]
        cn = str(col).strip().lower()
        fn = None
        if 'gsci' in cn: fn='sp_gsci'
        elif 's&p 500' in cn or 'sp500' in cn: fn='sp500'
        elif 'nasdaq' in cn: fn='nasdaq'
        elif 'vix' in cn: fn='vix'
        elif 'gdp' in cn: fn='gdp'
        elif 'unemployment' in cn: fn='unemployment'
        elif 'cpi' in cn: fn='cpi'
        elif 'ppi' in cn: fn='ppi'
        elif 'exchange rate' in cn: fn='effective_exchange_rate'
        elif 'interbank' in cn: fn='interbank_3m'
        elif 'house price' in cn: fn='house_price_index'
        elif 'current account' in cn: fn='current_account'
        else:
            i += 1
            continue

        # The matched column is the value column, date column is one column before
        dcol = df.columns[i-1] if i > 0 else df.columns[0]
        vcol = col
        if dcol and vcol:
            for idx, row in df.iterrows():
                dv = row[dcol]
                vv = clean_value(row[vcol])
                if pd.isna(dv):
                    continue
                if isinstance(dv, str) and dv.strip().startswith('Q'):
                    dk = parse_quarter_date(dv)
                else:
                    dk = convert_to_date(dv)
                if not dk or not isinstance(dk, date) or dk < cutoff:
                    continue
                if dk not in date_records:
                    date_records[dk] = {'date':dk}
                if vv is not None:
                    date_records[dk][fn] = float(vv)
        i += 2

    records = [MacroUS(**date_records[dk]) for dk in sorted(date_records.keys())]
    for i in range(0, len(records), 1000):
        session.bulk_save_objects(records[i:i+1000])
        session.commit()
    logger.info(f"[OK] Loaded {len(records)} US macro records")
    return len(records)

def load_fx_rates(session: Session, excel_path: Path) -> int:
    logger.info("Loading FX rates...")
    df = pd.read_excel(excel_path, sheet_name="Fx Rate", header=None)
    session.query(MacroFX).delete()
    session.commit()

    currency_row = df.iloc[1]
    cmap = {}
    pairs = ['audusd','usdcny','usdhkd','usdinr','usdidr','usdjpy','usdmyr','usdphp',
             'usdsgd','usdkrw','usdtwd','usdthb','eurusd','gbpusd','usdchf','usdzar',
             'usdnok','usdsek','usdbrl','usdmxn','usdcad']
    for idx, desc in enumerate(currency_row):
        if pd.isna(desc):
            continue
        ds = str(desc).lower()
        for p in pairs:
            if p in ds:
                cmap[idx] = p
                break

    logger.info(f"Found {len(cmap)} FX pairs")
    records = []
    for ridx in range(2, len(df)):
        row = df.iloc[ridx]
        dv = row[0]
        if pd.isna(dv):
            continue
        try:
            ds = str(int(dv))
            if len(ds) == 8:
                do = datetime(int(ds[:4]), int(ds[4:6]), int(ds[6:8])).date()
            else:
                continue
        except:
            continue
        rd = {'date':do}
        for cidx, fn in cmap.items():
            v = clean_value(row[cidx])
            if v is not None:
                rd[fn] = float(v)
        records.append(MacroFX(**rd))
        if len(records) >= 1000:
            session.bulk_save_objects(records)
            session.commit()
            records = []
    if records:
        session.bulk_save_objects(records)
        session.commit()
    logger.info(f"[OK] Loaded {len(records)} FX records")
    return len(records)

def load_macro_data(session: Session, data_dir: Path) -> dict:
    excel_path = data_dir / "Macros.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(f"Macros file not found: {excel_path}")
    stats = {}
    stats['commodities'] = load_commodities(session, excel_path)
    stats['bond_yields'] = load_bond_yields(session, excel_path)
    stats['us_macros'] = load_us_macros(session, excel_path)
    stats['fx_rates'] = load_fx_rates(session, excel_path)
    return stats
