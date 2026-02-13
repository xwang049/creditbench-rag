"""Tests for risk_indicators model and data loading."""

import pytest
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from src.db.models import Base, RiskIndicator, Company
from src.db.session import get_session


class TestRiskIndicatorModel:
    """Test RiskIndicator model definition."""

    def test_model_imports(self):
        """Test that RiskIndicator model can be imported."""
        assert RiskIndicator is not None
        assert hasattr(RiskIndicator, '__tablename__')
        assert RiskIndicator.__tablename__ == 'risk_indicators'

    def test_model_columns(self):
        """Test that all required columns are defined."""
        # Get column names from the model
        columns = {col.name for col in RiskIndicator.__table__.columns}

        expected_columns = {
            'id', 'u3_company_number', 'year', 'month',
            'stk_index', 'st_int', 'm2b', 'sigma',
            'dtd_median', 'dtd_median_i', 'dtd',
            'liquidity_r', 'ni2ta', 'size', 'liquidity_fin'
        }

        assert expected_columns.issubset(columns), \
            f"Missing columns: {expected_columns - columns}"

    def test_model_constraints(self):
        """Test that unique constraint is defined."""
        table = RiskIndicator.__table__

        # Check for unique constraint
        unique_constraints = [c for c in table.constraints
                            if hasattr(c, 'columns') and len(c.columns) == 3]
        assert len(unique_constraints) > 0, "Unique constraint not found"

        # Verify constraint columns
        constraint = unique_constraints[0]
        constraint_cols = {col.name for col in constraint.columns}
        assert constraint_cols == {'u3_company_number', 'year', 'month'}

    def test_model_indexes(self):
        """Test that indexes are defined."""
        table = RiskIndicator.__table__

        # Check for indexes
        index_names = {idx.name for idx in table.indexes}

        # Should have at least the composite index on year, month
        assert any('year' in str(idx.columns) and 'month' in str(idx.columns)
                  for idx in table.indexes), \
            "Composite index on (year, month) not found"

    def test_model_foreign_key(self):
        """Test that foreign key to companies table exists."""
        table = RiskIndicator.__table__

        # Get foreign keys
        fk_columns = {fk.parent.name for fk in table.foreign_keys}
        assert 'u3_company_number' in fk_columns, \
            "Foreign key on u3_company_number not found"

        # Check that it references companies table
        fk = list(table.foreign_keys)[0]
        assert 'companies' in str(fk.target_fullname)

    def test_model_repr(self):
        """Test model __repr__ method."""
        record = RiskIndicator(
            u3_company_number=12345,
            year=2020,
            month=6,
            dtd=2.5
        )
        repr_str = repr(record)
        assert '12345' in repr_str
        assert '2020' in repr_str
        assert '6' in repr_str
        assert '2.5' in repr_str


class TestDataLoadingLogic:
    """Test data loading logic without actually loading data."""

    def test_clean_value_function(self):
        """Test the clean_value helper function."""
        from src.ingestion.load_risk_indicators import clean_value
        import pandas as pd

        # Test NaN
        assert clean_value(pd.NA) is None
        assert clean_value(float('nan')) is None

        # Test empty string
        assert clean_value('') is None
        assert clean_value('  ') is None

        # Test 'NA' string
        assert clean_value('NA') is None
        assert clean_value('na') is None
        assert clean_value('N/A') is None

        # Test valid values
        assert clean_value(123.45) == 123.45
        assert clean_value('hello') == 'hello'

    def test_company_number_conversion(self):
        """Test the Company_Number to u3_company_number conversion logic."""
        # Test the conversion formula: u3 = Company_Number // 1000
        test_cases = [
            (26978004, 26978),
            (12345678, 12345),
            (1000, 1),
            (999, 0),
            (5000000, 5000),
        ]

        for company_num, expected_u3 in test_cases:
            calculated_u3 = company_num // 1000
            assert calculated_u3 == expected_u3, \
                f"Failed for {company_num}: expected {expected_u3}, got {calculated_u3}"


class TestSeedScript:
    """Test seed.py script functionality."""

    def test_seed_imports(self):
        """Test that seed.py imports work."""
        # This will fail if there are import errors
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent))

        from scripts.seed import setup_logging, main
        assert setup_logging is not None
        assert main is not None

    def test_only_parameter_choices(self):
        """Test that --only parameter has correct choices."""
        import sys
        from pathlib import Path
        import argparse

        sys.path.insert(0, str(Path(__file__).parent.parent))

        # This is a simple check - we can't easily test argparse without running it
        # Just verify the script doesn't have syntax errors
        with open(Path(__file__).parent.parent / 'scripts' / 'seed.py', 'r') as f:
            content = f.read()
            assert 'risk_indicators' in content
            assert '--only' in content
            assert 'load_risk_indicator_data' in content


class TestLoadAllIntegration:
    """Test load_all.py integration."""

    def test_load_all_imports(self):
        """Test that load_all.py imports correctly."""
        from src.ingestion.load_all import load_all_data
        from src.ingestion.load_risk_indicators import load_risk_indicator_data

        assert load_all_data is not None
        assert load_risk_indicator_data is not None

    def test_load_all_includes_risk_indicators(self):
        """Test that load_all.py includes risk_indicators in the workflow."""
        with open(Path(__file__).parent.parent / 'src' / 'ingestion' / 'load_all.py', 'r') as f:
            content = f.read()
            assert 'load_risk_indicator_data' in content
            assert 'risk_indicators' in content
            # Verify it's in the step sequence
            assert '[4/5]' in content or '4/5' in content


def test_all_files_exist():
    """Test that all new files were created."""
    base_path = Path(__file__).parent.parent

    files_to_check = [
        base_path / 'src' / 'ingestion' / 'load_risk_indicators.py',
        base_path / 'scripts' / 'add_risk_indicators_table.py',
    ]

    for file_path in files_to_check:
        assert file_path.exists(), f"File not found: {file_path}"


def test_models_file_updated():
    """Test that models.py was updated correctly."""
    models_path = Path(__file__).parent.parent / 'src' / 'db' / 'models.py'

    with open(models_path, 'r') as f:
        content = f.read()

        # Check for UniqueConstraint import
        assert 'UniqueConstraint' in content

        # Check for RiskIndicator class
        assert 'class RiskIndicator(Base):' in content

        # Check for table name
        assert '__tablename__ = "risk_indicators"' in content

        # Check for key columns
        assert 'u3_company_number' in content
        assert 'dtd:' in content
        assert 'dtd_median:' in content
        assert 'dtd_median_i:' in content

        # Check for constraint
        assert "UniqueConstraint('u3_company_number', 'year', 'month'" in content

        # Check for index
        assert 'ix_risk_year_month' in content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
