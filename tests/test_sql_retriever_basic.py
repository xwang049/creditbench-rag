"""Basic tests for Text-to-SQL RAG system.

Run with: python -m pytest tests/test_sql_retriever_basic.py -v
"""

import pytest
from src.rag.sql_retriever import (
    get_schema_description,
    is_safe_sql,
    add_limit_if_missing,
    format_results_for_llm,
)


def test_get_schema_description():
    """Test schema description generation."""
    schema = get_schema_description()

    assert isinstance(schema, str)
    assert len(schema) > 1000  # Should be detailed
    assert "companies" in schema
    assert "credit_events" in schema
    assert "risk_indicators" in schema
    assert "dtd" in schema.lower()  # Distance-to-Default mentioned
    assert "Distance-to-Default" in schema  # Key metric


def test_is_safe_sql_select():
    """Test that SELECT queries are allowed."""
    sql = "SELECT * FROM companies LIMIT 10"
    is_safe, error = is_safe_sql(sql)

    assert is_safe is True
    assert error is None


def test_is_safe_sql_with_joins():
    """Test that complex SELECT with JOINs is allowed."""
    sql = """
        SELECT c.company_name, ce.action_name
        FROM companies c
        JOIN credit_events ce ON c.u3_company_number = ce.u3_company_number
        WHERE ce.announcement_date >= '2023-01-01'
        LIMIT 50
    """
    is_safe, error = is_safe_sql(sql)

    assert is_safe is True
    assert error is None


def test_is_safe_sql_blocks_insert():
    """Test that INSERT is blocked."""
    sql = "INSERT INTO companies (ticker) VALUES ('HACK')"
    is_safe, error = is_safe_sql(sql)

    assert is_safe is False
    assert error is not None
    assert "SELECT" in error or "allowed" in error.lower()


def test_is_safe_sql_blocks_update():
    """Test that UPDATE is blocked."""
    sql = "UPDATE companies SET ticker = 'HACK' WHERE id = 1"
    is_safe, error = is_safe_sql(sql)

    assert is_safe is False
    assert "UPDATE" in error or "dangerous" in error.lower()


def test_is_safe_sql_blocks_delete():
    """Test that DELETE is blocked."""
    sql = "DELETE FROM companies WHERE id = 1"
    is_safe, error = is_safe_sql(sql)

    assert is_safe is False


def test_is_safe_sql_blocks_drop():
    """Test that DROP is blocked."""
    sql = "DROP TABLE companies"
    is_safe, error = is_safe_sql(sql)

    assert is_safe is False
    assert "DROP" in error


def test_is_safe_sql_blocks_alter():
    """Test that ALTER is blocked."""
    sql = "ALTER TABLE companies ADD COLUMN hack TEXT"
    is_safe, error = is_safe_sql(sql)

    assert is_safe is False


def test_is_safe_sql_empty():
    """Test that empty SQL is rejected."""
    sql = ""
    is_safe, error = is_safe_sql(sql)

    assert is_safe is False
    assert error is not None


def test_add_limit_if_missing_adds_limit():
    """Test that LIMIT is added when missing."""
    sql = "SELECT * FROM companies"
    result = add_limit_if_missing(sql, default_limit=50)

    assert "LIMIT 50" in result
    assert result.strip().endswith("LIMIT 50")


def test_add_limit_if_missing_preserves_existing():
    """Test that existing LIMIT is preserved."""
    sql = "SELECT * FROM companies LIMIT 10"
    result = add_limit_if_missing(sql, default_limit=50)

    assert "LIMIT 10" in result
    assert "LIMIT 50" not in result


def test_add_limit_with_semicolon():
    """Test that LIMIT is added before semicolon."""
    sql = "SELECT * FROM companies;"
    result = add_limit_if_missing(sql, default_limit=100)

    assert "LIMIT 100" in result
    # Should be at the end (semicolon removed by strip)


def test_format_results_for_llm_empty():
    """Test formatting empty results."""
    results = []
    formatted = format_results_for_llm(results)

    assert "No results" in formatted


def test_format_results_for_llm_basic():
    """Test formatting basic results."""
    results = [
        {"id": 1, "name": "Company A", "ticker": "AAA"},
        {"id": 2, "name": "Company B", "ticker": "BBB"},
    ]
    formatted = format_results_for_llm(results)

    assert "id" in formatted
    assert "name" in formatted
    assert "ticker" in formatted
    assert "Company A" in formatted
    assert "AAA" in formatted


def test_format_results_for_llm_max_rows():
    """Test that max_rows limit is respected."""
    results = [{"id": i, "value": f"val_{i}"} for i in range(100)]
    formatted = format_results_for_llm(results, max_rows=10)

    # Should only have 10 rows plus header and separator
    lines = formatted.split('\n')
    # Header + separator + 10 data rows = 12 lines
    assert len(lines) == 12


def test_format_results_for_llm_handles_none():
    """Test formatting results with None values."""
    results = [
        {"id": 1, "value": None},
        {"id": 2, "value": "present"},
    ]
    formatted = format_results_for_llm(results)

    assert "None" in formatted or "" in formatted


@pytest.mark.parametrize("dangerous_keyword", [
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
    "TRUNCATE", "GRANT", "REVOKE"
])
def test_dangerous_keywords_blocked(dangerous_keyword):
    """Test that all dangerous keywords are blocked."""
    sql = f"{dangerous_keyword} something"
    is_safe, error = is_safe_sql(sql)

    assert is_safe is False
    assert error is not None


def test_sql_case_insensitive_blocking():
    """Test that dangerous keywords are blocked regardless of case."""
    test_cases = [
        "DELETE FROM companies",
        "delete from companies",
        "DeLeTe FrOm companies",
    ]

    for sql in test_cases:
        is_safe, error = is_safe_sql(sql)
        assert is_safe is False, f"Should block: {sql}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
