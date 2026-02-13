# Risk Indicators Table Setup Guide

## 📋 Overview

This document describes the `risk_indicators` table that was added to the CreditBench database and how to use it.

## 🗃️ Table Structure

### Schema

```sql
CREATE TABLE risk_indicators (
    id SERIAL PRIMARY KEY,
    u3_company_number INTEGER NOT NULL REFERENCES companies(u3_company_number),
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,  -- 1-12

    -- Risk indicators (all nullable)
    stk_index FLOAT,           -- Stock Index (StkIndx)
    st_int FLOAT,              -- Short-term Interest Rate (STInt)
    m2b FLOAT,                 -- Market-to-Book ratio
    sigma FLOAT,               -- Stock volatility
    dtd_median FLOAT,          -- Distance-to-Default median (column H)
    dtd_median_i FLOAT,        -- Distance-to-Default median (column I)
    dtd FLOAT,                 -- Distance-to-Default (MOST IMPORTANT)
    liquidity_r FLOAT,         -- Liquidity ratio
    ni2ta FLOAT,               -- Net income to total assets
    size FLOAT,                -- Company size
    liquidity_fin FLOAT,       -- Financial liquidity

    CONSTRAINT uq_risk_indicators_company_year_month
        UNIQUE (u3_company_number, year, month)
);

CREATE INDEX ix_risk_year_month ON risk_indicators (year, month);
CREATE INDEX ix_risk_indicators_u3_company_number ON risk_indicators (u3_company_number);
```

### Key Features

- **Foreign Key**: Links to `companies` table via `u3_company_number`
- **Unique Constraint**: Each company can have only one record per year-month
- **Composite Index**: Optimized for time-based queries
- **11 Risk Indicators**: Various financial and market metrics

## 🔑 Important: ID Mapping

The CSV file uses `Company_Number` (e.g., 26978004) which must be converted:

```python
u3_company_number = Company_Number // 1000
# Example: 26978004 // 1000 = 26978
```

## 📂 Files Modified/Created

### Modified Files
1. ✅ `src/db/models.py` - Added RiskIndicator model
2. ✅ `src/ingestion/load_all.py` - Integrated risk_indicators loading
3. ✅ `scripts/seed.py` - Added `--only risk_indicators` support

### New Files
1. ✅ `src/ingestion/load_risk_indicators.py` - Data loader
2. ✅ `scripts/add_risk_indicators_table.py` - Migration script
3. ✅ `tests/test_risk_indicators.py` - Test suite
4. ✅ `scripts/verify_risk_indicators_setup.py` - Setup verification

## 🚀 Installation Steps

### Step 1: Verify Setup

```bash
# Run verification script (no database connection needed)
python -m scripts.verify_risk_indicators_setup
```

All checks should pass ✓

### Step 2: Start Database

```bash
# Start PostgreSQL with pgvector
docker-compose up -d

# Verify it's running
docker ps
```

### Step 3: Create Table and Load Data

**Option A: One-Command Migration (Recommended)**
```bash
python -m scripts.add_risk_indicators_table
```

This will:
- Create the `risk_indicators` table (if it doesn't exist)
- Load data from `./data/risk_indicators.csv`
- Verify the data with 5 validation queries
- Show sample records

**Option B: Using seed.py**
```bash
# Create table first (if needed)
python -c "from src.db.init_db import create_tables; create_tables()"

# Then load data
python -m scripts.seed --data-dir ./data --skip-init --only risk_indicators
```

**Option C: Full Database Reset (Caution!)**
```bash
# This will reload ALL data (companies, events, macros, risk_indicators)
python -m scripts.seed --data-dir ./data
```

## 📊 Data Specifications

- **Source File**: `./data/risk_indicators.csv` (~118 MB)
- **Expected Records**: Hundreds of thousands of rows
- **Processing**: Chunked reading (50,000 rows per chunk)
- **Batch Insert**: 5,000 records per batch
- **Orphan Filtering**: Only loads records for companies in the database

## 🔍 Verification Queries

After loading, verify with these SQL queries:

### 1. Total Records
```sql
SELECT COUNT(*) FROM risk_indicators;
```

### 2. Distinct Companies
```sql
SELECT COUNT(DISTINCT u3_company_number) FROM risk_indicators;
```

### 3. Sample Data with Company Names
```sql
SELECT
    ri.u3_company_number,
    c.company_name,
    ri.year,
    ri.month,
    ri.dtd
FROM risk_indicators ri
JOIN companies c ON ri.u3_company_number = c.u3_company_number
WHERE ri.dtd IS NOT NULL
ORDER BY ri.year DESC, ri.month DESC
LIMIT 10;
```

### 4. Data Completeness
```sql
SELECT
    COUNT(*) as total_rows,
    COUNT(dtd) as dtd_count,
    ROUND(100.0 * COUNT(dtd) / COUNT(*), 2) as dtd_completeness_pct,
    COUNT(sigma) as sigma_count,
    COUNT(m2b) as m2b_count
FROM risk_indicators;
```

### 5. Time Range
```sql
SELECT
    MIN(year) as min_year,
    MAX(year) as max_year,
    COUNT(DISTINCT year) as year_count,
    COUNT(DISTINCT month) as month_count
FROM risk_indicators;
```

### 6. Check for Orphan Records (Should be 0)
```sql
SELECT COUNT(*)
FROM risk_indicators
WHERE u3_company_number NOT IN (
    SELECT u3_company_number FROM companies
);
```

## 🔧 Usage Examples

### Python Query Examples

```python
from src.db.session import get_session
from src.db.models import RiskIndicator, Company
from sqlalchemy import text

# Get risk indicators for a specific company
with get_session() as session:
    company_u3 = 26978
    indicators = session.query(RiskIndicator).filter(
        RiskIndicator.u3_company_number == company_u3
    ).order_by(
        RiskIndicator.year.desc(),
        RiskIndicator.month.desc()
    ).limit(12).all()

    for ind in indicators:
        print(f"{ind.year}-{ind.month:02d}: DTD={ind.dtd}")

# Get companies with highest DTD in recent period
with get_session() as session:
    results = session.query(
        Company.company_name,
        RiskIndicator.year,
        RiskIndicator.month,
        RiskIndicator.dtd
    ).join(
        RiskIndicator,
        Company.u3_company_number == RiskIndicator.u3_company_number
    ).filter(
        RiskIndicator.year == 2023,
        RiskIndicator.dtd.isnot(None)
    ).order_by(
        RiskIndicator.dtd.desc()
    ).limit(10).all()

    for name, year, month, dtd in results:
        print(f"{name}: {dtd:.4f} ({year}-{month:02d})")
```

## 🐳 Docker Configuration

**No Docker changes needed!** The existing `docker-compose.yml` already has:
- ✅ `pgvector/pgvector:pg15` image
- ✅ PostgreSQL on port 5432
- ✅ Persistent volume for data

## 🧪 Running Tests

```bash
# Run all risk_indicators tests
pytest tests/test_risk_indicators.py -v

# Run specific test
pytest tests/test_risk_indicators.py::TestRiskIndicatorModel::test_model_columns -v
```

## 📝 Column Mapping Reference

| CSV Column Name | Database Column | Description |
|----------------|-----------------|-------------|
| Company_Number | u3_company_number | Converted: Company_Number // 1000 |
| year | year | Year (YYYY) |
| month | month | Month (1-12) |
| StkIndx | stk_index | Stock Index |
| STInt | st_int | Short-term Interest Rate |
| m2b | m2b | Market-to-Book ratio |
| sigma | sigma | Stock volatility |
| DTDmedian (col H) | dtd_median | Distance-to-Default median (first) |
| DTDmedian (col I) | dtd_median_i | Distance-to-Default median (second) |
| dtd | dtd | Distance-to-Default ⭐ **Most Important** |
| liquidity_r | liquidity_r | Liquidity ratio |
| ni2ta | ni2ta | Net income to total assets |
| size | size | Company size |
| liquidity_fin | liquidity_fin | Financial liquidity |

## ⚠️ Troubleshooting

### Issue: CSV file not found
```
FileNotFoundError: Risk indicators file not found: ./data/risk_indicators.csv
```
**Solution**: Ensure the CSV file exists at `./data/risk_indicators.csv`

### Issue: Foreign key constraint violation
```
ForeignKeyViolation: insert or update on table "risk_indicators" violates foreign key constraint
```
**Solution**: The company must exist in the `companies` table first. The loader automatically filters orphan records.

### Issue: Unique constraint violation
```
UniqueViolation: duplicate key value violates unique constraint
```
**Solution**: You're trying to insert duplicate (u3_company_number, year, month). The loader clears existing data first (idempotent).

### Issue: Large file memory issues
**Solution**: The loader uses chunked reading (50K rows at a time) to handle the 118 MB file efficiently.

## 🎯 Next Steps

After successful setup:
1. ✅ Query risk indicators in your RAG system
2. ✅ Join with `companies` table for company info
3. ✅ Join with `credit_events` for event correlation
4. ✅ Analyze trends over time using year/month indexes
5. ✅ Focus on `dtd` (Distance-to-Default) as the primary risk metric

## 📚 Related Documentation

- Main README: `README.md`
- Database Schema: `src/db/models.py`
- Data Loading: `src/ingestion/`
- API Documentation: `src/api/`

## 🆘 Support

If you encounter issues:
1. Run `python -m scripts.verify_risk_indicators_setup` to check setup
2. Check logs for specific error messages
3. Verify database connection: `docker-compose ps`
4. Ensure data file exists and is readable

---

**Last Updated**: 2026-02-12
**Database Version**: PostgreSQL 15 with pgvector
**Python Version**: 3.10+
