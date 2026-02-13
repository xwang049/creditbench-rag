# Migration to Text-to-SQL RAG

## Summary of Changes

This document summarizes the migration from vector embedding-based RAG to Text-to-SQL RAG for the CreditBench system.

## What Changed

### ✅ New: Text-to-SQL RAG System

**New file:** `src/rag/sql_retriever.py`

Complete Text-to-SQL RAG implementation with:
- `get_schema_description()` - Returns detailed database schema for LLM
- `text_to_sql()` - Converts natural language to SQL using Claude API
- `execute_safe_sql()` - Safely executes SELECT queries with validation
- `sql_rag_answer()` - Complete pipeline: question → SQL → results → answer
- Interactive CLI: `python -m src.rag.sql_retriever`

**Features:**
- Uses Claude Sonnet 4.5 (claude-sonnet-4-20250514)
- SQL injection prevention (SELECT only)
- Automatic LIMIT 100 if not specified
- 30-second query timeout
- Natural language answer generation
- Detailed error handling

### 🔧 Modified: Database Models

**File:** `src/db/models.py`

**Changes:**
- ✂️ Commented out `pgvector` import
- ✂️ Removed `Company.embedding` field
- ✂️ Removed `CreditEvent.embedding` field
- ✂️ Disabled `CreditEventEmbedding` model (commented out)
- ✂️ Disabled vector indexes (ivfflat, hnsw)

**Reason:** Using Text-to-SQL RAG instead of vector embeddings for structured data.

### 🔧 Modified: Embeddings Module

**File:** `src/rag/embeddings.py`

**Changes:**
- ⚠️ Added warning comments that this module is DISABLED
- 🛑 Modified `generate_credit_event_embeddings()` to return early with warning
- 📝 Added instructions for re-enabling if needed in the future

**Reason:** Not generating embeddings since we're using Text-to-SQL RAG.

### 🔧 Modified: Retriever Module

**File:** `src/rag/retriever.py`

**Changes:**
- ✂️ Removed `vector_retrieve()` function
- ✂️ Removed `hybrid_retrieve()` function
- ✂️ Removed `CreditEventEmbedding` from imports
- ✅ Kept `sql_retrieve()` for backward compatibility
- 📝 Updated `VectorRetriever` class docstrings
- ⚠️ Added warnings pointing users to `sql_retriever.py`

### 🔧 Modified: RAG Module Init

**File:** `src/rag/__init__.py`

**Changes:**
- ✅ Added exports for Text-to-SQL RAG functions:
  - `sql_rag_answer`
  - `text_to_sql`
  - `execute_safe_sql`
  - `get_schema_description`
- 📝 Added comments about primary vs legacy implementations

### 📚 New: Documentation

**New files:**
- `docs/text_to_sql_rag_usage.md` - Complete usage guide
- `docs/MIGRATION_TO_TEXT_TO_SQL.md` - This file

### ✅ New: Tests

**New file:** `tests/test_sql_retriever_basic.py`

26 tests covering:
- Schema description generation
- SQL safety validation
- LIMIT clause handling
- Result formatting
- Dangerous keyword blocking

**All tests passing!** ✅

## Why Text-to-SQL?

For structured databases like CreditBench, Text-to-SQL RAG is superior because:

1. **Precision**: SQL provides exact, deterministic results
2. **Aggregations**: Easy to compute COUNT, AVG, SUM, etc.
3. **Joins**: Properly combines data from multiple tables
4. **Performance**: No need to pre-compute/store embeddings
5. **Cost**: No embedding API calls for millions of rows
6. **Interpretability**: Users can see and verify the SQL

Vector embeddings are great for unstructured text, but CreditBench is a structured database.

## Quick Start

### Interactive Mode

```bash
python -m src.rag.sql_retriever
```

### Programmatic Usage

```python
from src.rag import sql_rag_answer

result = sql_rag_answer("How many bankruptcy filings in 2023?")
print(result["answer"])
```

See `docs/text_to_sql_rag_usage.md` for detailed documentation.

## Database Schema

The system knows about all tables:
- `companies` (29K rows)
- `credit_events` (41K rows)
- `risk_indicators` (millions of rows, monthly panel)
- `industry_mapping` (64 rows)
- `macro_commodities`, `macro_bond_yields`, `macro_us`, `macro_fx`

**Key metric:** `risk_indicators.dtd` (Distance-to-Default) - primary credit risk measure.

## Example Questions

- "How many companies are in each sector?"
- "Show me bankruptcy filings from 2023"
- "Which energy companies had the lowest DTD in 2022?"
- "What was the 10-year Treasury yield on 2023-06-30?"
- "List companies with DTD < 2 in the most recent month"

## Safety Features

✅ Only SELECT queries allowed
✅ Blocks dangerous keywords (INSERT, UPDATE, DELETE, DROP, etc.)
✅ Automatic LIMIT 100 if not specified
✅ 30-second query timeout
✅ SQL parsing and validation
✅ Graceful error handling

## Re-enabling Vector Embeddings (if needed)

If you want to use vector embeddings in the future:

1. Uncomment `pgvector` import in `src/db/models.py`
2. Uncomment `CreditEventEmbedding` model
3. Uncomment embedding fields on `Company` and `CreditEvent`
4. Uncomment vector indexes
5. Run database migrations
6. Remove early return in `generate_credit_event_embeddings()`
7. Run embedding generation

But for now, **Text-to-SQL is the recommended approach** for this structured database.

## Dependencies

Already installed:
- `anthropic>=0.79.0` - Claude API
- `sqlparse>=0.4.2` - SQL parsing and validation
- `sqlalchemy>=2.0` - Database ORM
- `psycopg2-binary` - PostgreSQL driver

## Environment Variables

Required in `.env`:
```bash
ANTHROPIC_API_KEY=your-api-key-here
DATABASE_URL=postgresql://user:pass@localhost:5432/creditbench
```

## Testing

Run tests:
```bash
# Basic unit tests (no DB/API required)
python -m pytest tests/test_sql_retriever_basic.py -v

# All tests
python -m pytest tests/ -v
```

## Files Modified

- ✅ `src/rag/sql_retriever.py` - NEW
- 🔧 `src/db/models.py` - Disabled embeddings
- 🔧 `src/rag/embeddings.py` - Disabled
- 🔧 `src/rag/retriever.py` - Removed vector functions
- 🔧 `src/rag/__init__.py` - Added Text-to-SQL exports
- ✅ `docs/text_to_sql_rag_usage.md` - NEW
- ✅ `docs/MIGRATION_TO_TEXT_TO_SQL.md` - NEW (this file)
- ✅ `tests/test_sql_retriever_basic.py` - NEW

## Next Steps

1. Try the interactive CLI: `python -m src.rag.sql_retriever`
2. Read the usage guide: `docs/text_to_sql_rag_usage.md`
3. Explore example questions in the documentation
4. Customize system prompts for your specific use cases
5. Add more tests in `tests/test_sql_retriever_basic.py`

## Support

For questions or issues:
- See `docs/text_to_sql_rag_usage.md` for detailed usage
- Check `tests/test_sql_retriever_basic.py` for examples
- Review error messages (they're detailed and helpful)

---

**Status:** ✅ Complete - All tests passing, ready to use!
