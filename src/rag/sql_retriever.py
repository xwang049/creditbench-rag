"""Text-to-SQL RAG system for CreditBench database.

This module provides a complete Text-to-SQL RAG pipeline:
1. Convert natural language questions to SQL queries
2. Execute queries safely against the database
3. Generate natural language answers from the results
"""

import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import sqlparse
from sqlparse.sql import Statement
from sqlparse.tokens import Keyword, DML

from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from src.config import settings
from src.db.session import SessionLocal

logger = logging.getLogger(__name__)


def get_schema_description() -> str:
    """Return detailed database schema description for LLM context.

    Returns:
        Formatted schema description string
    """
    return """Database: CreditBench - Credit Research Database (data through June 2024)

Table: companies (29,118 rows)
- u3_company_number (INT, PK) - Unique company identifier
- ticker (VARCHAR) - Bloomberg ticker, e.g. 'AAPL US'
- company_name (VARCHAR) - Full company name
- country_name (VARCHAR) - Country of listing
- security_type (VARCHAR) - Common Stock, Depositary Receipt, etc.
- market_status (VARCHAR) - ACTV=Active, DLST=Delisted, ACQU=Acquired, MERG=Merged, LIQU=Liquidated, etc.
- prime_exchange (VARCHAR) - Primary exchange
- domicile (VARCHAR) - Country of domicile
- industry_sector_num (INT) - Links to industry_mapping.industry_sector_num
- industry_group_num (INT) - Links to industry_mapping.industry_group_num
- industry_subgroup_num (INT) - Links to industry_mapping.industry_subgroup_num
- id_isin (VARCHAR) - ISIN identifier
- id_cusip (VARCHAR) - CUSIP identifier

Table: industry_mapping (64 rows)
- industry_sector (VARCHAR) - e.g. 'Energy', 'Financials', 'Technology'
- industry_sector_num (INT) - Sector code (can link to companies.industry_sector_num)
- industry_group (VARCHAR) - e.g. 'Oil & Gas', 'Banking'
- industry_group_num (INT) - Group code
- industry_subgroup (VARCHAR) - e.g. 'Integrated Oil', 'Regional Banks'
- industry_subgroup_num (INT) - Subgroup code (UNIQUE, can link to companies.industry_subgroup_num)

Table: credit_events (~40,936 rows)
- id (INT, PK)
- u3_company_number (INT, FK → companies.u3_company_number)
- announcement_date (DATE) - When the event was announced
- effective_date (DATE) - When the event took effect
- event_type (INT) - Event type code: 110=Bankruptcy Filing, 208=Delisting, 301=Default Corp Action
- action_name (VARCHAR) - Human-readable event name: 'Delisting', 'Default Corp Action', 'Bankruptcy Filing', 'Change in Listing (Exchange to OTC)', etc.
- subcategory (TEXT) - Additional details, e.g. 'Reason: Bankruptcy', 'Filing Type: Receivership'

Table: risk_indicators (large, monthly panel data - millions of rows)
- id (INT, PK)
- u3_company_number (INT, FK → companies.u3_company_number)
- year (INT) - Year of the observation
- month (INT) - Month (1-12)
- dtd (FLOAT) - **Distance-to-Default** (MOST IMPORTANT - Merton model credit risk measure, higher = safer)
- sigma (FLOAT) - Stock return volatility
- m2b (FLOAT) - Market-to-book ratio
- ni2ta (FLOAT) - Net income to total assets ratio
- size (FLOAT) - Firm size (log market cap)
- liquidity_r (FLOAT) - Market liquidity measure
- liquidity_fin (FLOAT) - Financial liquidity measure
- stk_index (FLOAT) - Stock market index level
- st_int (FLOAT) - Short-term interest rate

Table: macro_commodities (~9,241 rows, daily since 1990)
- date (DATE, PK)
- wti_crude (FLOAT) - WTI crude oil price
- brent_crude (FLOAT) - Brent crude oil price
- gold (FLOAT) - Gold price
- silver (FLOAT) - Silver price
- copper (FLOAT) - Copper price
- natural_gas (FLOAT) - Natural gas price
- aluminum, lead, nickel, zinc (FLOAT) - Industrial metal prices
- wheat, corn, soybeans, cotton, sugar, coffee, cocoa (FLOAT) - Agricultural commodity prices
- kansas_financial_stress (FLOAT) - Kansas City Fed Financial Stress Index

Table: macro_bond_yields (~9,252 rows, daily since 1990)
- data_date (DATE, PK)
- us_1m, us_3m, us_6m (FLOAT) - US Treasury yields (short-term)
- us_1y, us_2y, us_3y (FLOAT) - US Treasury yields (mid-term)
- us_5y, us_7y, us_10y (FLOAT) - US Treasury yields (long-term)
- us_30y (FLOAT) - 30-year Treasury yield

Table: macro_us (~24,495 rows, mixed daily/quarterly frequency)
- date (DATE, PK)
- sp500 (FLOAT, daily) - S&P 500 index level
- nasdaq (FLOAT, daily) - NASDAQ index level
- vix (FLOAT, daily) - VIX volatility index
- gdp (FLOAT, quarterly) - Real GDP
- unemployment (FLOAT, monthly) - Unemployment rate
- cpi (FLOAT, monthly) - Consumer Price Index
- ppi (FLOAT, monthly) - Producer Price Index
- house_price_index (FLOAT, quarterly) - Case-Shiller House Price Index
- current_account (FLOAT, quarterly) - US current account balance

Table: macro_fx (~12,869 rows, daily since 1990)
- date (DATE, PK)
- eurusd, gbpusd (FLOAT) - Major currency pairs
- usdjpy, usdchf (FLOAT) - USD vs major currencies
- usdcny, usdhkd, usdsgd, usdkrw (FLOAT) - USD vs Asian currencies
- audusd, usdcad, usdmxn, usdbrl (FLOAT) - USD vs commodity/Americas currencies
- usdinr, usdidr, usdmyr, usdphp, usdtwd, usdthb (FLOAT) - Other Asian currencies
- usdzar, usdnok, usdsek (FLOAT) - Other currencies

Key relationships:
- companies.u3_company_number ← credit_events.u3_company_number
- companies.u3_company_number ← risk_indicators.u3_company_number
- companies.industry_sector_num → industry_mapping.industry_sector_num (NOT UNIQUE, JOIN carefully)
- companies.industry_subgroup_num → industry_mapping.industry_subgroup_num (UNIQUE, can use for JOIN)

Important notes:
- dtd (Distance-to-Default) in risk_indicators is the PRIMARY credit risk metric
- Lower dtd = higher default risk, higher dtd = lower default risk
- risk_indicators is a monthly panel (one row per company-year-month)
- Macro tables have different frequencies (daily/monthly/quarterly)
"""


def text_to_sql(question: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Convert natural language question to SQL query using Claude API.

    Args:
        question: Natural language question
        model: Claude model to use (default: claude-sonnet-4-20250514)

    Returns:
        SQL query string

    Raises:
        RuntimeError: If Anthropic API is not available
        Exception: If API call fails
    """
    if not HAS_ANTHROPIC:
        raise RuntimeError("anthropic package not installed")

    if not settings.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not configured")

    client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    schema = get_schema_description()

    system_prompt = f"""You are a SQL expert for the CreditBench credit research database (PostgreSQL).
Given a natural language question, generate a valid SQL query.

Rules:
- Only generate SELECT queries (never INSERT, UPDATE, DELETE, DROP, etc.)
- Always use table aliases for readability (e.g., c for companies, ce for credit_events)
- When joining companies with credit_events, use u3_company_number
- When joining companies with industry_mapping, prefer using industry_subgroup_num (UNIQUE) over industry_sector_num (NOT UNIQUE)
- For date filtering, use standard SQL date functions (e.g., date >= '2020-01-01')
- Limit results to 50 rows unless the user asks for more
- For "how many" questions, use COUNT(*)
- For sector/industry queries, JOIN with industry_mapping
- The most important risk metric is dtd (Distance-to-Default) in risk_indicators - lower dtd means higher risk
- Return ONLY the SQL query, no explanation or markdown code blocks
- Use proper date formatting: YYYY-MM-DD for date literals
- For aggregations, use appropriate GROUP BY clauses
- For time series queries on risk_indicators, remember it's monthly panel data

{schema}

Examples:

Q: "Show me recent bankruptcy filings"
A: SELECT c.company_name, c.ticker, ce.announcement_date, ce.subcategory FROM credit_events ce JOIN companies c ON ce.u3_company_number = c.u3_company_number WHERE ce.action_name = 'Bankruptcy Filing' ORDER BY ce.announcement_date DESC LIMIT 50

Q: "Which energy companies had the highest default risk in 2023?"
A: SELECT c.company_name, c.ticker, AVG(ri.dtd) as avg_dtd, MIN(ri.dtd) as min_dtd FROM risk_indicators ri JOIN companies c ON ri.u3_company_number = c.u3_company_number JOIN industry_mapping im ON c.industry_sector_num = im.industry_sector_num WHERE im.industry_sector = 'Energy' AND ri.year = 2023 GROUP BY c.company_name, c.ticker ORDER BY avg_dtd ASC LIMIT 50

Q: "How many credit events occurred in 2022?"
A: SELECT COUNT(*) as event_count FROM credit_events WHERE announcement_date >= '2022-01-01' AND announcement_date < '2023-01-01'
"""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
            temperature=0
        )

        sql = response.content[0].text.strip()

        # Remove markdown code blocks if present
        sql = re.sub(r'^```sql\s*', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'^```\s*', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'\s*```$', '', sql, flags=re.MULTILINE)
        sql = sql.strip()

        logger.info(f"Generated SQL for question: {question}")
        logger.debug(f"SQL: {sql}")

        return sql

    except Exception as e:
        logger.error(f"Error generating SQL: {e}")
        raise


def is_safe_sql(sql: str) -> tuple[bool, Optional[str]]:
    """Check if SQL query is safe (SELECT only, no dangerous operations).

    Args:
        sql: SQL query string

    Returns:
        Tuple of (is_safe, error_message)
    """
    # Parse SQL
    try:
        parsed = sqlparse.parse(sql)
    except Exception as e:
        return False, f"Failed to parse SQL: {e}"

    if not parsed:
        return False, "Empty SQL query"

    # Check each statement
    for statement in parsed:
        # Get the first token (should be SELECT)
        first_token = None
        for token in statement.tokens:
            if not token.is_whitespace:
                first_token = token
                break

        if not first_token:
            return False, "Empty statement"

        # Check if it's a SELECT statement
        if first_token.ttype != DML or first_token.value.upper() != 'SELECT':
            return False, f"Only SELECT queries allowed, got: {first_token.value}"

        # Check for dangerous keywords in the entire statement
        sql_upper = statement.value.upper()
        dangerous_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
            'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE',
            'INTO OUTFILE', 'INTO DUMPFILE', 'LOAD_FILE'
        ]

        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return False, f"Dangerous keyword detected: {keyword}"

    return True, None


def add_limit_if_missing(sql: str, default_limit: int = 100) -> str:
    """Add LIMIT clause to SQL if not present.

    Args:
        sql: SQL query string
        default_limit: Default limit to add

    Returns:
        SQL with LIMIT clause
    """
    sql_upper = sql.upper()

    # Check if LIMIT already exists
    if 'LIMIT' in sql_upper:
        return sql

    # Add LIMIT before semicolon if present, otherwise at the end
    sql = sql.rstrip(';').strip()
    return f"{sql} LIMIT {default_limit}"


def execute_safe_sql(
    sql: str,
    session: Session,
    timeout_seconds: int = 30
) -> Dict[str, Any]:
    """Execute SQL query safely with validation and timeout.

    Args:
        sql: SQL query string
        session: Database session
        timeout_seconds: Query timeout in seconds

    Returns:
        Dictionary with keys:
        - success (bool): Whether query succeeded
        - data (List[Dict]): Query results if successful
        - error (str): Error message if failed
        - row_count (int): Number of rows returned
    """
    # Validate SQL is safe
    is_safe, error_msg = is_safe_sql(sql)
    if not is_safe:
        return {
            "success": False,
            "error": f"Unsafe SQL: {error_msg}",
            "data": [],
            "row_count": 0
        }

    # Add LIMIT if missing
    sql = add_limit_if_missing(sql, default_limit=100)

    try:
        # Set statement timeout (PostgreSQL specific)
        session.execute(text(f"SET statement_timeout = {timeout_seconds * 1000}"))

        # Execute query
        result = session.execute(text(sql))

        # Fetch results
        rows = result.fetchall()
        columns = result.keys()

        # Convert to list of dicts
        data = []
        for row in rows:
            row_dict = {}
            for col, value in zip(columns, row):
                # Convert special types to JSON-serializable formats
                if isinstance(value, datetime):
                    row_dict[col] = value.isoformat()
                elif hasattr(value, 'isoformat'):  # date, time
                    row_dict[col] = value.isoformat()
                else:
                    row_dict[col] = value
            data.append(row_dict)

        return {
            "success": True,
            "data": data,
            "error": None,
            "row_count": len(data)
        }

    except SQLAlchemyError as e:
        logger.error(f"Database error executing SQL: {e}")
        return {
            "success": False,
            "error": f"Database error: {str(e)}",
            "data": [],
            "row_count": 0
        }
    except Exception as e:
        logger.error(f"Unexpected error executing SQL: {e}")
        return {
            "success": False,
            "error": f"Error: {str(e)}",
            "data": [],
            "row_count": 0
        }


def format_results_for_llm(results: List[Dict[str, Any]], max_rows: int = 50) -> str:
    """Format query results as a readable table for LLM context.

    Args:
        results: List of result dictionaries
        max_rows: Maximum number of rows to include

    Returns:
        Formatted table string
    """
    if not results:
        return "No results found."

    # Limit rows
    results = results[:max_rows]

    # Get column names
    columns = list(results[0].keys())

    # Calculate column widths
    col_widths = {col: len(str(col)) for col in columns}
    for row in results:
        for col in columns:
            val_len = len(str(row.get(col, '')))
            col_widths[col] = max(col_widths[col], val_len)

    # Build table
    lines = []

    # Header
    header = " | ".join(str(col).ljust(col_widths[col]) for col in columns)
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for row in results:
        line = " | ".join(str(row.get(col, '')).ljust(col_widths[col]) for col in columns)
        lines.append(line)

    return "\n".join(lines)


def sql_rag_answer(
    question: str,
    model: str = "claude-sonnet-4-20250514",
    session: Optional[Session] = None
) -> Dict[str, Any]:
    """Complete Text-to-SQL RAG pipeline.

    Args:
        question: Natural language question
        model: Claude model to use
        session: Database session (will create new one if not provided)

    Returns:
        Dictionary with keys:
        - question (str): Original question
        - sql (str): Generated SQL query
        - results (List[Dict]): Query results
        - answer (str): Natural language answer
        - success (bool): Whether the pipeline succeeded
        - error (str): Error message if failed
    """
    own_session = False
    if session is None:
        session = SessionLocal()
        own_session = True

    try:
        # Step 1: Generate SQL
        logger.info(f"Question: {question}")
        try:
            sql = text_to_sql(question, model=model)
        except Exception as e:
            return {
                "question": question,
                "sql": None,
                "results": [],
                "answer": f"Error generating SQL: {str(e)}",
                "success": False,
                "error": f"SQL generation failed: {str(e)}"
            }

        # Step 2: Execute SQL
        logger.info(f"Executing SQL: {sql}")
        exec_result = execute_safe_sql(sql, session)

        if not exec_result["success"]:
            return {
                "question": question,
                "sql": sql,
                "results": [],
                "answer": f"Error executing SQL: {exec_result['error']}",
                "success": False,
                "error": exec_result['error']
            }

        results = exec_result["data"]
        logger.info(f"Query returned {len(results)} rows")

        # Step 3: Generate answer from results
        if not HAS_ANTHROPIC or not settings.ANTHROPIC_API_KEY:
            # Fallback: return formatted results
            formatted = format_results_for_llm(results)
            return {
                "question": question,
                "sql": sql,
                "results": results,
                "answer": f"Query returned {len(results)} rows:\n\n{formatted}",
                "success": True,
                "error": None
            }

        # Use Claude to generate natural language answer
        client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

        formatted_results = format_results_for_llm(results, max_rows=50)

        answer_prompt = f"""Question: {question}

SQL Query: {sql}

Results ({len(results)} rows):
{formatted_results}

Please answer the original question based on these query results. Be specific with numbers, dates, and company names. If the results are empty or insufficient to answer the question, say so clearly."""

        try:
            response = client.messages.create(
                model=model,
                max_tokens=2000,
                system="You are a credit research analyst. Answer the user's question based on the SQL query results from the CreditBench database. Be specific with numbers and dates. If the data is insufficient, say so.",
                messages=[{"role": "user", "content": answer_prompt}],
                temperature=0
            )

            answer = response.content[0].text.strip()

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            answer = f"Query succeeded but failed to generate answer: {str(e)}\n\nResults:\n{formatted_results}"

        return {
            "question": question,
            "sql": sql,
            "results": results,
            "answer": answer,
            "success": True,
            "error": None
        }

    finally:
        if own_session:
            session.close()


def main():
    """Interactive CLI for Text-to-SQL RAG system."""
    print("=" * 80)
    print("CreditBench Text-to-SQL RAG System")
    print("=" * 80)
    print("\nAsk questions about credit events, companies, risk indicators, and macro data.")
    print("Type 'quit' or 'exit' to stop.\n")

    session = SessionLocal()

    try:
        while True:
            try:
                question = input("\n🔍 Question: ").strip()

                if not question:
                    continue

                if question.lower() in ('quit', 'exit', 'q'):
                    print("\nGoodbye!")
                    break

                print("\n⏳ Generating SQL and fetching results...\n")

                result = sql_rag_answer(question, session=session)

                if result["success"]:
                    print(f"📊 SQL Query:\n{result['sql']}\n")
                    print(f"✅ Results: {len(result['results'])} rows\n")
                    print(f"💡 Answer:\n{result['answer']}\n")
                else:
                    print(f"❌ Error: {result['error']}\n")
                    if result["sql"]:
                        print(f"Generated SQL:\n{result['sql']}\n")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.")
                continue
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}\n")
                logger.exception("Unexpected error in main loop")
                continue

    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
