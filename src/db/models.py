"""SQLAlchemy ORM models for CreditBench data."""

from datetime import datetime
from typing import Optional

from sqlalchemy import String, Float, DateTime, Text, Integer, ForeignKey, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class Company(Base):
    """Company master data."""

    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(primary_key=True)
    company_code: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    company_name: Mapped[str] = mapped_column(String(255))
    industry: Mapped[Optional[str]] = mapped_column(String(100))
    sector: Mapped[Optional[str]] = mapped_column(String(100))
    country: Mapped[Optional[str]] = mapped_column(String(50))
    listing_status: Mapped[Optional[str]] = mapped_column(String(50))

    # Embedding for semantic search
    embedding: Mapped[Optional[list]] = mapped_column(Vector(1536))

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    credit_events: Mapped[list["CreditEvent"]] = relationship(back_populates="company", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Company(code={self.company_code}, name={self.company_name})>"


class CreditEvent(Base):
    """Credit events (defaults, downgrades, etc.)."""

    __tablename__ = "credit_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    event_date: Mapped[datetime] = mapped_column(DateTime, index=True)
    event_type: Mapped[str] = mapped_column(String(50), index=True)  # 'default', 'downgrade', 'upgrade'
    rating_before: Mapped[Optional[str]] = mapped_column(String(10))
    rating_after: Mapped[Optional[str]] = mapped_column(String(10))
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Financial metrics at the time of event
    total_assets: Mapped[Optional[float]] = mapped_column(Float)
    total_liabilities: Mapped[Optional[float]] = mapped_column(Float)
    revenue: Mapped[Optional[float]] = mapped_column(Float)
    ebitda: Mapped[Optional[float]] = mapped_column(Float)

    # Embedding for semantic search
    embedding: Mapped[Optional[list]] = mapped_column(Vector(1536))

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    company: Mapped["Company"] = relationship(back_populates="credit_events")

    def __repr__(self) -> str:
        return f"<CreditEvent(company_id={self.company_id}, type={self.event_type}, date={self.event_date})>"


class MacroIndicator(Base):
    """Macroeconomic indicators."""

    __tablename__ = "macro_indicators"

    id: Mapped[int] = mapped_column(primary_key=True)
    indicator_date: Mapped[datetime] = mapped_column(DateTime, index=True)
    country: Mapped[str] = mapped_column(String(50), index=True)

    # Economic indicators
    gdp_growth: Mapped[Optional[float]] = mapped_column(Float)
    inflation_rate: Mapped[Optional[float]] = mapped_column(Float)
    unemployment_rate: Mapped[Optional[float]] = mapped_column(Float)
    interest_rate: Mapped[Optional[float]] = mapped_column(Float)
    exchange_rate: Mapped[Optional[float]] = mapped_column(Float)

    # Market indicators
    stock_index: Mapped[Optional[float]] = mapped_column(Float)
    credit_spread: Mapped[Optional[float]] = mapped_column(Float)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<MacroIndicator(country={self.country}, date={self.indicator_date})>"


# Create indexes for vector similarity search
Index("idx_companies_embedding", Company.embedding, postgresql_using="ivfflat")
Index("idx_credit_events_embedding", CreditEvent.embedding, postgresql_using="ivfflat")
