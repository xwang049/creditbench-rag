"""Tests for data import validation and query functionality."""

import pytest
from datetime import datetime, date
from sqlalchemy import func, extract, and_
from sqlalchemy.orm import Session

from src.db.models import (
    Company, CreditEvent, IndustryMapping,
    MacroBondYields, MacroCommodities, MacroUS, MacroFX
)


class TestBasicStatistics:
    """Test basic statistics to validate data import."""

    def test_companies_count(self, db_session):
        """Test that companies table has expected number of records."""
        count = db_session.query(func.count(Company.u3_company_number)).scalar()

        # Should be around 29,118 companies
        assert count > 25000, f"Expected >25k companies, got {count}"
        assert count < 35000, f"Expected <35k companies, got {count}"
        print(f"[OK] Companies count: {count:,}")

    def test_credit_events_count(self, db_session):
        """Test that credit_events table has expected number of records."""
        count = db_session.query(func.count(CreditEvent.id)).scalar()

        # Should be around 40,936 credit events
        assert count > 35000, f"Expected >35k credit events, got {count}"
        assert count < 50000, f"Expected <50k credit events, got {count}"
        print(f"[OK] Credit events count: {count:,}")

    def test_industry_mapping_count(self, db_session):
        """Test that industry_mapping table has expected number of records."""
        count = db_session.query(func.count(IndustryMapping.id)).scalar()

        # Should be exactly 64 industry mappings
        assert count == 64, f"Expected exactly 64 industry mappings, got {count}"
        print(f"[OK] Industry mappings count: {count}")

    def test_distinct_action_names(self, db_session):
        """Test distinct credit event action names."""
        count = db_session.query(
            func.count(func.distinct(CreditEvent.action_name))
        ).scalar()

        # Should have multiple distinct action types
        assert count >= 3, f"Expected at least 3 distinct action names, got {count}"

        # Get the actual action names
        action_names = db_session.query(
            CreditEvent.action_name,
            func.count(CreditEvent.id).label('count')
        ).group_by(CreditEvent.action_name).all()

        print(f"[OK] Distinct action names: {count}")
        for name, cnt in action_names:
            print(f"  - {name}: {cnt:,}")

    def test_macro_bond_yields_count(self, db_session):
        """Test that macro_bond_yields table has data."""
        count = db_session.query(func.count(MacroBondYields.data_date)).scalar()

        assert count > 100, f"Expected >100 bond yield records, got {count}"
        print(f"[OK] Bond yields count: {count:,}")

    def test_macro_commodities_count(self, db_session):
        """Test that macro_commodities table has data."""
        count = db_session.query(func.count(MacroCommodities.date)).scalar()

        assert count > 100, f"Expected >100 commodity records, got {count}"
        print(f"[OK] Commodity records count: {count:,}")

    def test_macro_us_count(self, db_session):
        """Test that macro_us table has data."""
        count = db_session.query(func.count(MacroUS.date)).scalar()

        assert count > 100, f"Expected >100 US macro records, got {count}"
        print(f"[OK] US macro records count: {count:,}")

    def test_macro_fx_count(self, db_session):
        """Test that macro_fx table has data."""
        count = db_session.query(func.count(MacroFX.date)).scalar()

        assert count > 100, f"Expected >100 FX records, got {count}"
        print(f"[OK] FX records count: {count:,}")


class TestRelationalQueries:
    """Test relational queries across tables."""

    def test_top_companies_with_defaults(self, db_session):
        """Find top 10 companies with most 'Default Corp Action' events."""
        results = db_session.query(
            Company.company_name,
            Company.ticker,
            func.count(CreditEvent.id).label('default_count')
        ).join(
            CreditEvent,
            Company.u3_company_number == CreditEvent.u3_company_number
        ).filter(
            CreditEvent.action_name == 'Default Corp Action'
        ).group_by(
            Company.u3_company_number,
            Company.company_name,
            Company.ticker
        ).order_by(
            func.count(CreditEvent.id).desc()
        ).limit(10).all()

        assert len(results) > 0, "Expected at least some companies with defaults"

        print(f"[OK] Top 10 companies with 'Default Corp Action':")
        for company_name, ticker, count in results:
            ticker_str = f"({ticker})" if ticker else ""
            print(f"  - {company_name} {ticker_str}: {count} defaults")

    def test_monthly_defaults_2008_2009(self, db_session):
        """Count default events per month from Jan 2008 to Dec 2009."""
        start_date = date(2008, 1, 1)
        end_date = date(2009, 12, 31)

        results = db_session.query(
            extract('year', CreditEvent.announcement_date).label('year'),
            extract('month', CreditEvent.announcement_date).label('month'),
            func.count(CreditEvent.id).label('count')
        ).filter(
            and_(
                CreditEvent.action_name == 'Default Corp Action',
                CreditEvent.announcement_date >= start_date,
                CreditEvent.announcement_date <= end_date
            )
        ).group_by(
            'year', 'month'
        ).order_by(
            'year', 'month'
        ).all()

        assert len(results) > 0, "Expected defaults during 2008-2009 crisis"

        print(f"[OK] Monthly defaults 2008-2009:")
        for year, month, count in results:
            print(f"  - {int(year)}-{int(month):02d}: {count} defaults")

    def test_energy_sector_credit_events(self, db_session):
        """Find Energy sector companies with credit events."""
        results = db_session.query(
            Company.company_name,
            Company.ticker,
            IndustryMapping.industry_sector,
            func.count(CreditEvent.id).label('event_count')
        ).join(
            IndustryMapping,
            Company.industry_sector_num == IndustryMapping.industry_sector_num
        ).join(
            CreditEvent,
            Company.u3_company_number == CreditEvent.u3_company_number
        ).filter(
            IndustryMapping.industry_sector.ilike('%Energy%')
        ).group_by(
            Company.u3_company_number,
            Company.company_name,
            Company.ticker,
            IndustryMapping.industry_sector
        ).order_by(
            func.count(CreditEvent.id).desc()
        ).limit(10).all()

        # May or may not have results depending on data
        print(f"[OK] Energy sector companies with credit events: {len(results)}")
        for company_name, ticker, sector, event_count in results[:5]:
            ticker_str = f"({ticker})" if ticker else ""
            print(f"  - {company_name} {ticker_str}: {event_count} events")

    def test_vix_around_lehman_collapse(self, db_session):
        """Check VIX values around Lehman Brothers collapse (Sep 15, 2008)."""
        lehman_date = date(2008, 9, 15)
        start_date = date(2008, 8, 15)  # 30 days before
        end_date = date(2008, 10, 15)   # 30 days after

        results = db_session.query(
            MacroUS.date,
            MacroUS.vix
        ).filter(
            and_(
                MacroUS.date >= start_date,
                MacroUS.date <= end_date,
                MacroUS.vix.isnot(None)
            )
        ).order_by(
            MacroUS.date
        ).all()

        if len(results) > 0:
            print(f"[OK] VIX around Lehman collapse ({start_date} to {end_date}):")
            for dt, vix in results[:10]:
                marker = " <-- Lehman collapse" if dt == lehman_date else ""
                print(f"  - {dt}: VIX = {vix:.2f}{marker}")

            # VIX should spike during this period
            max_vix = max(vix for _, vix in results)
            assert max_vix > 30, f"Expected VIX spike >30 during crisis, got max {max_vix}"
        else:
            pytest.skip("No VIX data available for this period")

    def test_10y_treasury_yield_2008_q3(self, db_session):
        """Check 10Y Treasury yields during 2008 Q3."""
        start_date = date(2008, 7, 1)
        end_date = date(2008, 9, 30)

        results = db_session.query(
            MacroBondYields.data_date,
            MacroBondYields.us_10y
        ).filter(
            and_(
                MacroBondYields.data_date >= start_date,
                MacroBondYields.data_date <= end_date,
                MacroBondYields.us_10y.isnot(None)
            )
        ).order_by(
            MacroBondYields.data_date
        ).all()

        assert len(results) > 0, "Expected 10Y yield data for 2008 Q3"

        print(f"[OK] 10Y Treasury yields during 2008 Q3:")
        for dt, yield_val in results[:15]:
            print(f"  - {dt}: {yield_val:.2f}%")

        # Calculate average
        avg_yield = sum(y for _, y in results) / len(results)
        print(f"  Average 10Y yield in 2008 Q3: {avg_yield:.2f}%")


class TestDataIntegrity:
    """Test data integrity and relationships."""

    def test_foreign_key_companies_to_industry(self, db_session):
        """Verify foreign key relationship: companies -> industry_mapping."""
        # Count companies with industry sector numbers
        companies_with_sector = db_session.query(
            func.count(Company.u3_company_number)
        ).filter(
            Company.industry_sector_num.isnot(None)
        ).scalar()

        # All those sector numbers should exist in industry_mapping
        valid_sectors = db_session.query(
            func.count(func.distinct(Company.industry_sector_num))
        ).join(
            IndustryMapping,
            Company.industry_sector_num == IndustryMapping.industry_sector_num
        ).filter(
            Company.industry_sector_num.isnot(None)
        ).scalar()

        print(f"[OK] Companies with sector: {companies_with_sector:,}")
        print(f"[OK] Valid sector mappings: {valid_sectors}")

        # Most companies should have valid sector mappings
        assert valid_sectors > 0, "No valid sector mappings found"

    def test_foreign_key_events_to_companies(self, db_session):
        """Verify foreign key relationship: credit_events -> companies."""
        # Count credit events
        total_events = db_session.query(func.count(CreditEvent.id)).scalar()

        # Count events with valid company references
        valid_events = db_session.query(
            func.count(CreditEvent.id)
        ).join(
            Company,
            CreditEvent.u3_company_number == Company.u3_company_number
        ).scalar()

        print(f"[OK] Total credit events: {total_events:,}")
        print(f"[OK] Events with valid companies: {valid_events:,}")

        # All events should have valid company references
        assert valid_events == total_events, \
            f"Some events have invalid company references: {total_events - valid_events}"

    def test_date_ranges(self, db_session):
        """Verify date ranges across different tables."""
        # Credit events date range
        event_dates = db_session.query(
            func.min(CreditEvent.announcement_date).label('min_date'),
            func.max(CreditEvent.announcement_date).label('max_date')
        ).first()

        # Macro data date range
        macro_dates = db_session.query(
            func.min(MacroUS.date).label('min_date'),
            func.max(MacroUS.date).label('max_date')
        ).first()

        print(f"[OK] Credit events date range: {event_dates.min_date} to {event_dates.max_date}")
        print(f"[OK] Macro US data range: {macro_dates.min_date} to {macro_dates.max_date}")

        # Dates should be reasonable (after 1980, before current date)
        assert event_dates.min_date.year >= 1980
        assert event_dates.max_date <= date.today()
        assert macro_dates.min_date.year >= 1990


class TestSampleQueries:
    """Test sample analytical queries."""

    def test_bankruptcy_filings_by_year(self, db_session):
        """Count bankruptcy filings by year."""
        results = db_session.query(
            extract('year', CreditEvent.announcement_date).label('year'),
            func.count(CreditEvent.id).label('count')
        ).filter(
            CreditEvent.action_name == 'Bankruptcy Filing'
        ).group_by(
            'year'
        ).order_by(
            'year'
        ).all()

        if len(results) > 0:
            print(f"[OK] Bankruptcy filings by year:")
            for year, count in results[-10:]:  # Last 10 years
                if year is not None:
                    print(f"  - {int(year)}: {count} filings")
                else:
                    print(f"  - NULL: {count} filings")

    def test_market_status_distribution(self, db_session):
        """Check distribution of company market statuses."""
        results = db_session.query(
            Company.market_status,
            func.count(Company.u3_company_number).label('count')
        ).group_by(
            Company.market_status
        ).order_by(
            func.count(Company.u3_company_number).desc()
        ).all()

        assert len(results) > 0, "Expected various market statuses"

        print(f"[OK] Market status distribution:")
        for status, count in results:
            status_str = status if status else "NULL"
            print(f"  - {status_str}: {count:,}")

    def test_commodity_price_sample(self, db_session):
        """Sample commodity prices for a specific date."""
        # Try to get data from 2008 (financial crisis)
        target_date = date(2008, 9, 15)

        result = db_session.query(MacroCommodities).filter(
            MacroCommodities.date == target_date
        ).first()

        if result:
            print(f"[OK] Commodity prices on {target_date}:")
            if result.wti_crude:
                print(f"  - WTI Crude: ${result.wti_crude:.2f}")
            if result.gold:
                print(f"  - Gold: ${result.gold:.2f}")
            if result.copper:
                print(f"  - Copper: ${result.copper:.2f}")
        else:
            pytest.skip(f"No commodity data for {target_date}")



if __name__ == "__main__":
    # Run tests with: pytest tests/test_queries.py -v -s
    pytest.main([__file__, "-v", "-s"])
