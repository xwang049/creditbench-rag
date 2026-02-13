"""SQL-based retrieval for creditbench data.

Note: Vector embedding functionality has been disabled in favor of Text-to-SQL RAG.
For the complete Text-to-SQL RAG system, see sql_retriever.py instead.
This module is kept for backward compatibility with existing SQL retrieval functions.
"""

from typing import List, Dict, Any, Optional
import logging
import re
from sqlalchemy import select, text
from sqlalchemy.orm import Session

try:
    from anthropic import Anthropic
    USE_ANTHROPIC = True
except ImportError:
    USE_ANTHROPIC = False

try:
    from sentence_transformers import SentenceTransformer
    USE_SENTENCE_TRANSFORMERS = True
except ImportError:
    USE_SENTENCE_TRANSFORMERS = False

from src.db.models import (
    Company, CreditEvent, IndustryMapping,
    MacroCommodities, MacroBondYields, MacroUS, MacroFX, RiskIndicator
)
# Note: CreditEventEmbedding removed - using Text-to-SQL RAG instead
from src.config import settings

logger = logging.getLogger(__name__)


# Database schema for SQL generation
DATABASE_SCHEMA = """
Database Schema:

1. companies (29,118 records)
   - u3_company_number (INTEGER, PRIMARY KEY)
   - id_bb_unique (STRING)
   - ticker (STRING, indexed)
   - company_name (STRING, indexed)
   - country_name (STRING)
   - security_type (STRING)
   - market_status (STRING): ACTV, PRNA, ACQU, DLST, MERG, LIQU, RCNA, HAAI
   - prime_exchange (STRING)
   - domicile (STRING)
   - industry_sector_num (INTEGER)
   - industry_group_num (INTEGER)
   - industry_subgroup_num (INTEGER)
   - id_isin (STRING)
   - id_cusip (STRING)

2. credit_events (40,936 records)
   - id (INTEGER, PRIMARY KEY)
   - u3_company_number (INTEGER, FOREIGN KEY to companies)
   - id_bb_company (INTEGER)
   - announcement_date (DATE, indexed)
   - effective_date (DATE)
   - event_type (INTEGER): 208=Delisting, 301=Default, 110=Bankruptcy
   - action_name (STRING, indexed): Delisting, Default Corp Action, Bankruptcy Filing, etc.
   - subcategory (TEXT)

3. industry_mapping
   - industry_sector (STRING)
   - industry_sector_num (INTEGER)
   - industry_group (STRING)
   - industry_group_num (INTEGER)
   - industry_subgroup (STRING)
   - industry_subgroup_num (INTEGER)

4. risk_indicators
   - u3_company_number (INTEGER, FOREIGN KEY to companies)
   - year (INTEGER)
   - month (INTEGER)
   - dtd (FLOAT): Distance-to-Default (most important)
   - stk_index, st_int, m2b, sigma, liquidity_r, ni2ta, size, liquidity_fin (FLOAT)

5. macro_commodities
   - date (DATE, PRIMARY KEY)
   - wti_crude, brent_crude, gold, silver, copper, etc. (FLOAT)

6. macro_bond_yields
   - data_date (DATE, PRIMARY KEY)
   - us_1m, us_3m, us_6m, us_1y, us_2y, us_3y, us_5y, us_7y, us_10y, us_30y (FLOAT)

7. macro_us
   - date (DATE, PRIMARY KEY)
   - sp500, nasdaq, vix, gdp, unemployment, cpi, ppi (FLOAT)

8. macro_fx
   - date (DATE, PRIMARY KEY)
   - eurusd, usdjpy, gbpusd, usdcny, etc. (FLOAT)
"""


def sql_retrieve(query: str, session: Session, max_results: int = 100) -> List[Dict[str, Any]]:
    """Convert natural language query to SQL and execute.

    Args:
        query: Natural language query
        session: Database session
        max_results: Maximum number of results to return

    Returns:
        List of result dictionaries
    """
    if not USE_ANTHROPIC or not hasattr(settings, 'ANTHROPIC_API_KEY') or not settings.ANTHROPIC_API_KEY:
        logger.warning("Anthropic API not available for SQL generation")
        return []

    try:
        client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

        # Generate SQL using Claude
        system_prompt = f"""You are a PostgreSQL SQL expert. Convert natural language queries to SQL.

{DATABASE_SCHEMA}

Rules:
1. ONLY generate SELECT statements
2. NO INSERT, UPDATE, DELETE, DROP, CREATE, ALTER statements
3. Use proper JOINs when querying multiple tables
4. Always add LIMIT clause (max {max_results})
5. Use table aliases for clarity
6. Return ONLY the SQL query, no explanation
7. Use proper date formatting for date comparisons
8. For company searches, join with companies table to get company names

Example:
Query: "Show me recent bankruptcy filings"
SQL: SELECT c.company_name, ce.action_name, ce.announcement_date
     FROM credit_events ce
     JOIN companies c ON ce.u3_company_number = c.u3_company_number
     WHERE ce.action_name = 'Bankruptcy Filing'
     ORDER BY ce.announcement_date DESC
     LIMIT 10;"""

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": query}]
        )

        sql_query = response.content[0].text.strip()

        # Security check: ensure only SELECT
        sql_query_upper = sql_query.upper()
        dangerous_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'GRANT', 'REVOKE']
        if any(keyword in sql_query_upper for keyword in dangerous_keywords):
            logger.error(f"Dangerous SQL detected: {sql_query}")
            return []

        # Remove markdown code blocks if present
        sql_query = re.sub(r'^```sql\s*|\s*```$', '', sql_query, flags=re.MULTILINE).strip()

        logger.info(f"Generated SQL: {sql_query}")

        # Execute query
        result = session.execute(text(sql_query))
        rows = result.fetchall()

        # Convert to list of dicts
        if rows:
            columns = result.keys()
            return [dict(zip(columns, row)) for row in rows]

        return []

    except Exception as e:
        logger.error(f"Error in sql_retrieve: {e}")
        return []


# Note: Vector and hybrid retrieval functions have been removed.
# For Text-to-SQL RAG with natural language answers, use sql_retriever.py instead.
#
# The following functions are no longer available:
# - vector_retrieve() - removed (vector embeddings disabled)
# - hybrid_retrieve() - removed (vector embeddings disabled)
#
# Use sql_retrieve() below for basic SQL-based retrieval,
# or use sql_retriever.sql_rag_answer() for complete Text-to-SQL RAG with NL answers.


class VectorRetriever:
    """SQL-based retriever for CreditBench data.

    Note: Vector search has been removed. This class now only supports SQL retrieval.
    For complete Text-to-SQL RAG with natural language answers, use sql_retriever.py instead.
    """

    def __init__(self, session: Session):
        """Initialize retriever.

        Args:
            session: SQLAlchemy session
        """
        self.session = session

    def search(
        self,
        query: str,
        method: str = "sql",
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search using SQL retrieval.

        Args:
            query: Search query (natural language, will be converted to SQL)
            method: Only 'sql' is supported (vector/hybrid removed)
            top_k: Number of results

        Returns:
            List of results
        """
        if method == "sql":
            return sql_retrieve(query, self.session, max_results=top_k)
        else:
            raise ValueError(f"Method '{method}' not supported. Only 'sql' is available. For Text-to-SQL RAG, use sql_retriever.py instead.")

    def get_company_credit_events(
        self,
        u3_company_number: int,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get credit events for a specific company.

        Args:
            u3_company_number: Company U3 number
            limit: Maximum results

        Returns:
            List of credit events
        """
        query = text("""
            SELECT
                ce.id,
                ce.action_name,
                ce.subcategory,
                ce.announcement_date,
                ce.effective_date,
                c.company_name,
                c.ticker
            FROM credit_events ce
            JOIN companies c ON ce.u3_company_number = c.u3_company_number
            WHERE ce.u3_company_number = :u3
            ORDER BY ce.announcement_date DESC
            LIMIT :limit
        """)

        result = self.session.execute(query, {"u3": u3_company_number, "limit": limit})
        rows = result.fetchall()
        columns = result.keys()

        return [dict(zip(columns, row)) for row in rows]
