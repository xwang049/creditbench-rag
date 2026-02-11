"""Pytest configuration and fixtures for creditbench tests."""

import pytest
from sqlalchemy.orm import Session

from src.db.session import get_session


@pytest.fixture(scope="session")
def db_session() -> Session:
    """Provide a database session for the entire test session.

    This uses the actual database connection, so tests will run against
    the real data. Make sure to run the seed script before running tests.
    """
    with get_session() as session:
        yield session


@pytest.fixture(scope="function")
def db_session_function() -> Session:
    """Provide a fresh database session for each test function.

    Use this if you need transaction isolation between tests.
    """
    with get_session() as session:
        yield session
        # Rollback any changes made during the test
        session.rollback()
