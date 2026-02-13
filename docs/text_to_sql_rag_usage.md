# Text-to-SQL RAG System for CreditBench

## Overview

The CreditBench Text-to-SQL RAG system converts natural language questions into SQL queries, executes them safely, and generates natural language answers. This is the **primary RAG implementation** for querying structured data.

## Why Text-to-SQL instead of Vector Embeddings?

For structured databases like CreditBench, Text-to-SQL RAG is superior to vector embeddings because:

1. **Precision**: SQL queries provide exact, deterministic results
2. **Aggregations**: Easy to compute statistics (COUNT, AVG, SUM, etc.)
3. **Joins**: Can combine data from multiple tables correctly
4. **Performance**: No need to pre-compute and store embeddings
5. **Interpretability**: Users can see and verify the generated SQL
6. **Cost**: No embedding API calls for millions of rows

## Architecture

```
User Question
    ↓
[Claude API] Generate SQL from natural language
    ↓
[Safety Check] Validate SQL (SELECT only, no dangerous ops)
    ↓
[PostgreSQL] Execute query with timeout
    ↓
[Claude API] Generate natural language answer from results
    ↓
Final Answer
```

## Quick Start

### Interactive CLI

```bash
python -m src.rag.sql_retriever
```

Then ask questions:
```
Question: How many bankruptcy filings were there in 2023?
Question: Which technology companies had the lowest DTD in 2022?
Question: Show me recent credit events for Apple Inc.
```

### Programmatic Usage

```python
from src.rag import sql_rag_answer

# Ask a question
result = sql_rag_answer("How many companies are in the Energy sector?")

print(f"SQL: {result['sql']}")
print(f"Answer: {result['answer']}")
print(f"Results: {len(result['results'])} rows")
```

## API Reference

### `sql_rag_answer(question, model, session)`

Complete Text-to-SQL RAG pipeline.

**Parameters:**
- `question` (str): Natural language question
- `model` (str, optional): Claude model to use (default: "claude-sonnet-4-20250514")
- `session` (Session, optional): Database session (creates new one if not provided)

**Returns:**
```python
{
    "question": str,      # Original question
    "sql": str,          # Generated SQL query
    "results": list,     # Query results as list of dicts
    "answer": str,       # Natural language answer
    "success": bool,     # Whether pipeline succeeded
    "error": str | None  # Error message if failed
}
```

**Example:**
```python
result = sql_rag_answer(
    "What was the average DTD for financial companies in Q1 2023?"
)

if result["success"]:
    print(result["answer"])
    # Output: "The average Distance-to-Default for financial companies
    # in Q1 2023 was 5.23, based on 1,247 company-month observations..."
else:
    print(f"Error: {result['error']}")
```

### `text_to_sql(question, model)`

Convert natural language to SQL query.

**Parameters:**
- `question` (str): Natural language question
- `model` (str, optional): Claude model (default: "claude-sonnet-4-20250514")

**Returns:**
- SQL query string

**Example:**
```python
sql = text_to_sql("How many bankruptcies in 2023?")
print(sql)
# SELECT COUNT(*) FROM credit_events
# WHERE action_name = 'Bankruptcy Filing'
# AND announcement_date >= '2023-01-01'
# AND announcement_date < '2024-01-01'
```

### `execute_safe_sql(sql, session, timeout_seconds)`

Execute SQL query safely with validation.

**Parameters:**
- `sql` (str): SQL query
- `session` (Session): Database session
- `timeout_seconds` (int, optional): Query timeout (default: 30)

**Returns:**
```python
{
    "success": bool,
    "data": list,        # Query results
    "error": str | None,
    "row_count": int
}
```

**Safety features:**
- Only allows SELECT queries
- Blocks dangerous keywords (INSERT, UPDATE, DELETE, DROP, etc.)
- Adds LIMIT 100 if not specified
- 30-second query timeout
- Catches and reports exceptions

### `get_schema_description()`

Get detailed database schema description for LLM context.

**Returns:**
- Formatted schema description string with table structures, relationships, and usage notes

## Example Questions

### Company and Industry Queries
- "How many companies are in each sector?"
- "List all technology companies with market_status='ACTV'"
- "Which industries have the most companies?"

### Credit Events
- "Show me bankruptcy filings from 2023"
- "How many credit events occurred per year?"
- "List companies that had delisting events in 2022"
- "What are the most common types of credit events?"

### Risk Indicators (DTD = Distance-to-Default)
- "Which companies had the lowest DTD in 2023?" (highest risk)
- "What was the average DTD for energy companies in Q4 2022?"
- "Show me companies with DTD < 2 in the most recent month" (high default risk)
- "How did average DTD change over time for the financial sector?"

### Macroeconomic Data
- "What was the 10-year Treasury yield on 2023-01-15?"
- "Show me gold and oil prices during 2020"
- "What was the VIX level during March 2020?" (COVID crisis)
- "Show me EUR/USD exchange rate trends in 2023"

### Complex Joins
- "List energy companies with bankruptcy filings in 2023"
- "Show me the average DTD for companies by sector in 2023"
- "Which sectors had the most credit events in 2022?"

## Database Schema Summary

**Key Tables:**
- `companies` (29K rows) - Company master data
- `credit_events` (41K rows) - Credit events (bankruptcies, defaults, delistings)
- `risk_indicators` (millions of rows) - Monthly panel of credit risk metrics
- `industry_mapping` (64 rows) - Industry classification hierarchy
- `macro_*` tables - Commodities, bonds, FX, US economic indicators

**Most Important Field:**
- `risk_indicators.dtd` - **Distance-to-Default** (Merton model credit risk)
  - Higher DTD = Lower default risk (safer)
  - Lower DTD = Higher default risk (riskier)
  - Typical range: 0 to 10+
  - DTD < 2 indicates high default probability

## Configuration

The system uses these environment variables from `.env`:

```bash
ANTHROPIC_API_KEY=your-api-key-here
DATABASE_URL=postgresql://user:pass@localhost:5432/creditbench
```

## Safety Features

1. **SQL Injection Prevention**: Only SELECT queries allowed
2. **Resource Limits**: Automatic LIMIT 100, 30-second timeout
3. **Keyword Blocking**: Blocks INSERT, UPDATE, DELETE, DROP, etc.
4. **Error Handling**: Graceful failure with detailed error messages
5. **SQL Parsing**: Uses sqlparse to validate query structure

## Performance Tips

1. **Be specific**: "Show me 10 companies" is faster than "Show me companies"
2. **Use date ranges**: "in 2023" helps the LLM add proper WHERE clauses
3. **Aggregations**: COUNT/AVG queries are much faster than fetching raw data
4. **Indexes**: The database has indexes on key columns (dates, tickers, etc.)

## Troubleshooting

### "ANTHROPIC_API_KEY not configured"
- Add your API key to `.env` file
- Or set environment variable: `export ANTHROPIC_API_KEY=your-key`

### "Database error: connection refused"
- Check that PostgreSQL is running
- Verify DATABASE_URL in `.env` is correct
- Test connection: `psql $DATABASE_URL`

### "Unsafe SQL: Dangerous keyword detected"
- The system only allows SELECT queries
- This is a safety feature, not a bug
- Try rephrasing to ask for data retrieval instead of modification

### Query timeout
- Simplify the question
- Add date ranges to filter data
- Use aggregations instead of fetching all rows

## Migration from Vector Embeddings

If you previously used vector embeddings (`embeddings.py`, `vector_retrieve`):

1. **They are now disabled** - see comments in those files
2. Use `sql_rag_answer()` instead of `vector_retrieve()`
3. Text-to-SQL provides better results for structured data
4. No need to pre-compute embeddings (saves time and cost)

## Next Steps

- See `tests/test_rag_integration.py` for more examples
- Try the interactive CLI: `python -m src.rag.sql_retriever`
- Customize the schema description in `get_schema_description()` for your domain
- Adjust system prompts in `text_to_sql()` for your query patterns

## Credits

Built with:
- Claude Sonnet 4.5 (claude-sonnet-4-20250514) for SQL generation and answer synthesis
- PostgreSQL for data storage
- sqlparse for SQL validation
- SQLAlchemy for database ORM
