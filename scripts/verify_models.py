"""Verify database models are correctly defined."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.models import (
    Base,
    IndustryMapping,
    Company,
    CreditEvent,
    MacroCommodities,
    MacroBondYields,
    MacroUS,
    MacroFX,
)


def verify_models():
    """Verify all models are properly defined."""
    print("=" * 60)
    print("Verifying CreditBench Database Models")
    print("=" * 60)
    print()

    # Check all tables
    tables = Base.metadata.tables
    print(f"Found {len(tables)} tables:")
    for table_name in sorted(tables.keys()):
        table = tables[table_name]
        print(f"\n[OK] {table_name}")
        print(f"  Primary keys: {[col.name for col in table.primary_key]}")
        print(f"  Columns: {len(table.columns)}")

        # Show foreign keys
        if table.foreign_keys:
            print(f"  Foreign keys:")
            for fk in table.foreign_keys:
                print(f"    - {fk.parent.name} -> {fk.column}")

        # Show indexes
        if table.indexes:
            print(f"  Indexes: {len(table.indexes)}")

    print()
    print("=" * 60)

    # Test model instantiation
    print("\nTesting model instantiation:")
    print()

    try:
        # Test IndustryMapping
        industry = IndustryMapping(
            industry_sector="Energy",
            industry_sector_num=10,
            industry_group="Oil & Gas",
            industry_group_num=1010,
            industry_subgroup="Oil & Gas Exploration",
            industry_subgroup_num=101010
        )
        print(f"[OK] IndustryMapping: {industry}")

        # Test Company
        company = Company(
            u3_company_number=1,
            ticker="AAPL",
            company_name="Apple Inc.",
            country_name="United States",
            market_status="ACTV"
        )
        print(f"[OK] Company: {company}")

        # Test CreditEvent
        from datetime import date
        event = CreditEvent(
            u3_company_number=1,
            announcement_date=date(2024, 1, 1),
            event_type=301,
            action_name="Default Corp Action"
        )
        print(f"[OK] CreditEvent: {event}")

        # Test MacroCommodities
        commodities = MacroCommodities(
            date=date(2024, 1, 1),
            wti_crude=75.5,
            gold=2000.0
        )
        print(f"[OK] MacroCommodities: {commodities}")

        # Test MacroBondYields
        bonds = MacroBondYields(
            data_date=date(2024, 1, 1),
            us_10y=4.5,
            us_2y=4.2
        )
        print(f"[OK] MacroBondYields: {bonds}")

        # Test MacroUS
        macro_us = MacroUS(
            date=date(2024, 1, 1),
            sp500=4500.0,
            vix=15.5
        )
        print(f"[OK] MacroUS: {macro_us}")

        # Test MacroFX
        fx = MacroFX(
            date=date(2024, 1, 1),
            eurusd=1.08,
            usdjpy=145.0
        )
        print(f"[OK] MacroFX: {fx}")

        print()
        print("=" * 60)
        print("✓ All models verified successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error during model verification: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(verify_models())
