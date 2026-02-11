"""Interactive database query script for exploring CreditBench data."""

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from src.db.models import (
    Company, CreditEvent, IndustryMapping,
    MacroCommodities, MacroBondYields, MacroUS, MacroFX
)

# Database connection
DATABASE_URL = "postgresql://creditbench:creditbench@localhost:5432/creditbench"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)


def show_table_info():
    """Show all tables and their row counts."""
    print("\n" + "="*60)
    print("DATABASE TABLES")
    print("="*60)

    session = Session()
    tables = [
        ("companies", Company),
        ("credit_events", CreditEvent),
        ("industry_mapping", IndustryMapping),
        ("macro_commodities", MacroCommodities),
        ("macro_bond_yields", MacroBondYields),
        ("macro_us", MacroUS),
        ("macro_fx", MacroFX),
    ]

    for name, model in tables:
        count = session.query(model).count()
        print(f"  {name:<25} {count:>10,} rows")

    session.close()


def show_sample_companies():
    """Show sample companies with credit events."""
    print("\n" + "="*60)
    print("SAMPLE COMPANIES (with credit events)")
    print("="*60)

    session = Session()
    results = session.query(
        Company.company_name,
        Company.ticker,
        Company.country_name,
        Company.market_status
    ).filter(
        Company.ticker.isnot(None)
    ).limit(10).all()

    for name, ticker, country, status in results:
        print(f"  {ticker:>6} | {name:<40} | {country:<15} | {status}")

    session.close()


def show_recent_credit_events():
    """Show recent credit events."""
    print("\n" + "="*60)
    print("RECENT CREDIT EVENTS")
    print("="*60)

    session = Session()
    results = session.query(
        CreditEvent.announcement_date,
        CreditEvent.action_name,
        Company.company_name,
        Company.ticker
    ).join(
        Company
    ).filter(
        CreditEvent.announcement_date.isnot(None)
    ).order_by(
        CreditEvent.announcement_date.desc()
    ).limit(10).all()

    for date, action, company, ticker in results:
        ticker_str = f"({ticker})" if ticker else ""
        print(f"  {date} | {action:<25} | {company} {ticker_str}")

    session.close()


def show_macro_sample():
    """Show sample macro data."""
    print("\n" + "="*60)
    print("SAMPLE MACRO DATA (2008 Financial Crisis)")
    print("="*60)

    from datetime import date
    session = Session()

    # VIX during crisis
    vix_data = session.query(
        MacroUS.date,
        MacroUS.vix,
        MacroUS.sp500
    ).filter(
        MacroUS.date >= date(2008, 9, 1),
        MacroUS.date <= date(2008, 9, 30),
        MacroUS.vix.isnot(None)
    ).order_by(MacroUS.date).limit(10).all()

    print("\n  VIX and S&P 500 (Sep 2008):")
    for dt, vix, sp500 in vix_data:
        sp500_str = f"${sp500:.2f}" if sp500 else "N/A"
        print(f"    {dt} | VIX: {vix:>6.2f} | S&P 500: {sp500_str}")

    # Bond yields
    bond_data = session.query(
        MacroBondYields.data_date,
        MacroBondYields.us_10y,
        MacroBondYields.us_2y
    ).filter(
        MacroBondYields.data_date >= date(2008, 9, 1),
        MacroBondYields.data_date <= date(2008, 9, 15),
        MacroBondYields.us_10y.isnot(None)
    ).order_by(MacroBondYields.data_date).limit(5).all()

    print("\n  Treasury Yields (Sep 2008):")
    for dt, y10, y2 in bond_data:
        y2_str = f"{y2:.3f}%" if y2 else "N/A"
        print(f"    {dt} | 10Y: {y10:.3f}% | 2Y: {y2_str}")

    session.close()


def run_custom_query():
    """Run a custom SQL query."""
    print("\n" + "="*60)
    print("CUSTOM SQL QUERY")
    print("="*60)

    query = """
    SELECT
        c.ticker,
        c.company_name,
        COUNT(ce.id) as event_count,
        STRING_AGG(DISTINCT ce.action_name, ', ') as event_types
    FROM companies c
    JOIN credit_events ce ON c.u3_company_number = ce.u3_company_number
    WHERE c.ticker IS NOT NULL
    GROUP BY c.ticker, c.company_name
    HAVING COUNT(ce.id) > 5
    ORDER BY event_count DESC
    LIMIT 10;
    """

    print("\nTop 10 companies by credit event count:\n")

    session = Session()
    result = session.execute(text(query))

    for row in result:
        ticker, name, count, types = row
        print(f"  {ticker:>6} | {name:<40} | {count:>3} events")
        print(f"         Types: {types[:80]}")
        print()

    session.close()


def main():
    """Main function to run all demonstrations."""
    print("\n" + "="*60)
    print("CREDITBENCH DATABASE EXPLORER")
    print("="*60)

    show_table_info()
    show_sample_companies()
    show_recent_credit_events()
    show_macro_sample()
    run_custom_query()

    print("\n" + "="*60)
    print("DIRECT DATABASE ACCESS")
    print("="*60)
    print("\nYou can also query directly using psql:")
    print(f"  docker exec creditbench-postgres psql -U creditbench -d creditbench")
    print("\nOr use Python with SQLAlchemy:")
    print(f"  from src.db.models import *")
    print(f"  session.query(Company).filter(Company.ticker == 'AAPL').first()")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
