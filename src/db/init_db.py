"""Initialize database tables."""

import logging
from sqlalchemy import text

from src.db.models import Base
from src.db.session import engine, get_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_pgvector() -> None:
    """Initialize pgvector extension in PostgreSQL."""
    logger.info("Initializing pgvector extension...")
    try:
        with get_session() as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            session.commit()
        logger.info("✓ pgvector extension enabled")
    except Exception as e:
        logger.error(f"Failed to initialize pgvector: {e}")
        raise


def create_all() -> None:
    """Create all database tables.

    This is the main function to call for setting up the database schema.
    It will:
    1. Enable pgvector extension
    2. Create all tables defined in models.py
    3. Create indexes including vector indexes
    """
    logger.info("=" * 60)
    logger.info("Creating CreditBench database schema...")
    logger.info("=" * 60)

    # Step 1: Initialize pgvector
    init_pgvector()

    # Step 2: Create all tables
    logger.info("Creating database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✓ All tables created successfully!")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise

    # Step 3: List created tables
    logger.info("\nCreated tables:")
    for table_name in Base.metadata.tables.keys():
        logger.info(f"  - {table_name}")

    logger.info("=" * 60)
    logger.info("Database initialization complete!")
    logger.info("=" * 60)


def init_db() -> None:
    """Alias for create_all() for backward compatibility."""
    create_all()


def drop_all() -> None:
    """Drop all database tables (use with caution!)."""
    logger.warning("=" * 60)
    logger.warning("⚠️  DROPPING ALL DATABASE TABLES")
    logger.warning("=" * 60)

    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("✓ All tables dropped!")
    except Exception as e:
        logger.error(f"Failed to drop tables: {e}")
        raise


def reset_db() -> None:
    """Reset database by dropping and recreating all tables."""
    logger.warning("⚠️  Resetting database (drop + recreate)")
    drop_all()
    create_all()


def init_database() -> None:
    """Initialize database with pgvector extension and create tables.

    This is a convenience function that combines pgvector initialization
    and table creation. Safe to call multiple times (idempotent).
    """
    create_all()


def create_tables() -> None:
    """Create all tables without dropping existing ones.

    This function only creates tables that don't exist yet.
    Useful for migrations or partial schema updates.
    """
    logger.info("Creating missing tables...")
    Base.metadata.create_all(bind=engine, checkfirst=True)
    logger.info("✓ Tables created (if they didn't exist)")


if __name__ == "__main__":
    # When run directly, create all tables
    create_all()
