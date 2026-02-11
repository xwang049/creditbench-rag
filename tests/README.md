# CreditBench RAG Tests

Comprehensive test suite for validating data import and query functionality.

## Test Structure

### conftest.py
- **db_session (session scope)**: Shared database session for all tests
- **db_session_function (function scope)**: Fresh session for each test with rollback

### test_queries.py

#### 1. TestBasicStatistics
Validates basic counts and data import completeness:
- ✓ Companies count (~29,118 expected)
- ✓ Credit events count (~40,936 expected)
- ✓ Industry mappings count (exactly 64)
- ✓ Distinct action names (defaults, bankruptcies, delistings)
- ✓ Macro data tables (bond yields, commodities, US macros, FX)

#### 2. TestRelationalQueries
Tests complex SQL queries across tables:
- ✓ Top 10 companies with most defaults
- ✓ Monthly default counts during 2008-2009 crisis
- ✓ Energy sector companies with credit events
- ✓ VIX around Lehman Brothers collapse (Sep 15, 2008)
- ✓ 10Y Treasury yields during 2008 Q3

#### 3. TestDataIntegrity
Validates foreign key relationships and data quality:
- ✓ Foreign key: companies → industry_mapping
- ✓ Foreign key: credit_events → companies
- ✓ Date ranges validation

#### 4. TestSampleQueries
Additional analytical queries:
- ✓ Bankruptcy filings by year
- ✓ Market status distribution
- ✓ Commodity prices on specific dates

## Running Tests

### Run all tests
```bash
pytest tests/test_queries.py -v
```

### Run with output (see print statements)
```bash
pytest tests/test_queries.py -v -s
```

### Run specific test class
```bash
pytest tests/test_queries.py::TestBasicStatistics -v -s
```

### Run specific test
```bash
pytest tests/test_queries.py::TestBasicStatistics::test_companies_count -v -s
```

### Run and stop on first failure
```bash
pytest tests/test_queries.py -v -s -x
```

## Prerequisites

1. **Database must be seeded first**:
   ```bash
   python scripts/seed.py --data-dir ./data
   ```

2. **PostgreSQL must be running**:
   ```bash
   docker-compose up -d
   ```

3. **Install test dependencies**:
   ```bash
   pip install pytest pytest-asyncio
   ```

## Expected Output

```
tests/test_queries.py::TestBasicStatistics::test_companies_count
✓ Companies count: 29,118
PASSED

tests/test_queries.py::TestBasicStatistics::test_credit_events_count
✓ Credit events count: 40,936
PASSED

tests/test_queries.py::TestBasicStatistics::test_industry_mapping_count
✓ Industry mappings count: 64
PASSED

tests/test_queries.py::TestBasicStatistics::test_distinct_action_names
✓ Distinct action names: 4
  - Default Corp Action: 15,234
  - Bankruptcy Filing: 8,456
  - Delisting: 12,345
  - Change in Listing: 4,901
PASSED

tests/test_queries.py::TestRelationalQueries::test_top_companies_with_defaults
✓ Top 10 companies with 'Default Corp Action':
  - Company A (CMPA): 12 defaults
  - Company B (CMPB): 10 defaults
  - Company C (CMPC): 8 defaults
  ...
PASSED

tests/test_queries.py::TestRelationalQueries::test_vix_around_lehman_collapse
✓ VIX around Lehman collapse (2008-08-15 to 2008-10-15):
  - 2008-08-15: VIX = 19.45
  - 2008-09-12: VIX = 24.32
  - 2008-09-15: VIX = 31.70 <-- Lehman collapse
  - 2008-09-16: VIX = 36.22
  - 2008-09-29: VIX = 46.72
  ...
PASSED
```

## Troubleshooting

### Test failures

**"Expected >25k companies, got 0"**
- Database not seeded. Run `python scripts/seed.py`

**"No VIX data available for this period"**
- Macro data not loaded or date range different than expected
- Check `SELECT MIN(date), MAX(date) FROM macro_us;`

**Connection errors**
- PostgreSQL not running: `docker-compose up -d`
- Wrong connection string: check `.env` file

### Slow tests

Tests query the actual database, so performance depends on:
- Database size
- Available indexes
- Hardware

Typical run time: 5-15 seconds for full suite

## CI/CD Integration

Add to GitHub Actions:
```yaml
- name: Run data validation tests
  run: |
    docker-compose up -d
    sleep 10
    python scripts/seed.py --data-dir ./test_data
    pytest tests/test_queries.py -v
```
