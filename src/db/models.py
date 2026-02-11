"""SQLAlchemy ORM models for CreditBench data."""

from datetime import date, datetime
from typing import Optional

from sqlalchemy import String, Float, Date, DateTime, Text, Integer, ForeignKey, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class IndustryMapping(Base):
    """Industry classification mapping (BICS hierarchy)."""

    __tablename__ = "industry_mapping"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    industry_sector: Mapped[Optional[str]] = mapped_column(String(50))
    industry_sector_num: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    industry_group: Mapped[Optional[str]] = mapped_column(String(100))
    industry_group_num: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    industry_subgroup: Mapped[Optional[str]] = mapped_column(String(100))
    industry_subgroup_num: Mapped[Optional[int]] = mapped_column(Integer, index=True, unique=True)

    def __repr__(self) -> str:
        return f"<IndustryMapping(sector={self.industry_sector}, group={self.industry_group}, subgroup={self.industry_subgroup})>"


class Company(Base):
    """Company master data from Bloomberg."""

    __tablename__ = "companies"

    # Primary key
    u3_company_number: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Bloomberg identifiers
    id_bb_unique: Mapped[Optional[str]] = mapped_column(String(30))
    id_bb_company: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    ticker: Mapped[Optional[str]] = mapped_column(String(30), index=True)

    # Company information
    company_name: Mapped[Optional[str]] = mapped_column(String(200), index=True)
    country_name: Mapped[Optional[str]] = mapped_column(String(100))
    security_type: Mapped[Optional[str]] = mapped_column(String(50))
    market_status: Mapped[Optional[str]] = mapped_column(String(10))  # ACTV, PRNA, ACQU, DLST, MERG, LIQU, RCNA, HAAI
    prime_exchange: Mapped[Optional[str]] = mapped_column(String(50))
    domicile: Mapped[Optional[str]] = mapped_column(String(100))

    # Industry classification (no FK constraint due to non-unique sector_num)
    industry_sector_num: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    industry_group_num: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    industry_subgroup_num: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Additional identifiers
    id_isin: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, index=True)
    id_cusip: Mapped[Optional[str]] = mapped_column(String(15), nullable=True)

    # Embedding for semantic search
    embedding: Mapped[Optional[list]] = mapped_column(Vector(1536), nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    credit_events: Mapped[list["CreditEvent"]] = relationship(back_populates="company", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Company(u3={self.u3_company_number}, ticker={self.ticker}, name={self.company_name})>"


class CreditEvent(Base):
    """Credit events (defaults, bankruptcies, delistings, etc.)."""

    __tablename__ = "credit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Company reference
    u3_company_number: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("companies.u3_company_number"),
        index=True
    )
    id_bb_company: Mapped[Optional[int]] = mapped_column(Integer)

    # Event dates
    announcement_date: Mapped[Optional[date]] = mapped_column(Date, index=True)
    effective_date: Mapped[Optional[date]] = mapped_column(Date)

    # Event details
    event_type: Mapped[Optional[int]] = mapped_column(Integer)  # 208=Delisting, 301=Default, 110=Bankruptcy Filing
    action_name: Mapped[Optional[str]] = mapped_column(String(100), index=True)  # Delisting, Default Corp Action, Bankruptcy Filing, etc.
    subcategory: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Embedding for semantic search
    embedding: Mapped[Optional[list]] = mapped_column(Vector(1536), nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    company: Mapped["Company"] = relationship(back_populates="credit_events")

    def __repr__(self) -> str:
        return f"<CreditEvent(u3={self.u3_company_number}, type={self.action_name}, date={self.announcement_date})>"


class MacroCommodities(Base):
    """Commodity prices and Kansas financial stress index."""

    __tablename__ = "macro_commodities"

    date: Mapped[date] = mapped_column(Date, primary_key=True)

    # Energy commodities
    wti_crude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    brent_crude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gasoline: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    heating_oil: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gasoil: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    natural_gas: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Metals
    aluminum: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    copper: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lead: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    nickel: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    zinc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    silver: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Agricultural commodities
    wheat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    corn: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    soybeans: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cotton: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sugar: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    coffee: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cocoa: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Financial stress index
    kansas_financial_stress: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Other commodities
    iron_ore: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    coal: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    palm_oil: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rubber: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:
        return f"<MacroCommodities(date={self.date}, wti={self.wti_crude}, gold={self.gold})>"


class MacroBondYields(Base):
    """US Treasury bond yields across different maturities."""

    __tablename__ = "macro_bond_yields"

    data_date: Mapped[date] = mapped_column(Date, primary_key=True)

    # US Treasury yields
    us_1m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    us_3m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    us_6m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    us_1y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    us_2y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    us_3y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    us_5y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    us_7y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    us_10y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    us_30y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:
        return f"<MacroBondYields(date={self.data_date}, 10y={self.us_10y}, 2y={self.us_2y})>"


class MacroUS(Base):
    """US macroeconomic indicators (mixed frequency)."""

    __tablename__ = "macro_us"

    date: Mapped[date] = mapped_column(Date, primary_key=True)

    # Commodities and market indices
    sp_gsci: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # S&P GSCI Commodity Spot
    sp500: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    nasdaq: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vix: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Economic indicators (quarterly data stored at quarter end)
    gdp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Quarterly
    unemployment: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cpi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ppi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Financial indicators
    effective_exchange_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    interbank_3m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Real estate
    house_price_index: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Quarterly

    # Balance of payments
    current_account: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Quarterly

    def __repr__(self) -> str:
        return f"<MacroUS(date={self.date}, sp500={self.sp500}, vix={self.vix}, gdp={self.gdp})>"


class MacroFX(Base):
    """Foreign exchange rates (USD pairs)."""

    __tablename__ = "macro_fx"

    date: Mapped[date] = mapped_column(Date, primary_key=True)

    # Asia-Pacific currencies
    audusd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    usdcny: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    usdhkd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    usdinr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    usdidr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    usdjpy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    usdmyr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    usdphp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    usdsgd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    usdkrw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    usdtwd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    usdthb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # European currencies
    eurusd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gbpusd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    usdchf: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    usdzar: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    usdnok: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    usdsek: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Americas currencies
    usdbrl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    usdmxn: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    usdcad: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:
        return f"<MacroFX(date={self.date}, eurusd={self.eurusd}, usdjpy={self.usdjpy}, usdcny={self.usdcny})>"


# Create indexes for vector similarity search
Index("idx_companies_embedding", Company.embedding, postgresql_using="ivfflat")
Index("idx_credit_events_embedding", CreditEvent.embedding, postgresql_using="ivfflat")

# Create composite indexes for common queries
Index("idx_companies_ticker_status", Company.ticker, Company.market_status)
Index("idx_credit_events_date_type", CreditEvent.announcement_date, CreditEvent.event_type)
Index("idx_credit_events_action", CreditEvent.action_name, CreditEvent.announcement_date)
