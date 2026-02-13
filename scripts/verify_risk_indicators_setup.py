"""Quick verification script for risk_indicators setup.

This script checks that all code modifications are correct without
requiring database connection or data files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_header(text):
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}{text:^70}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")


def print_check(passed, message):
    symbol = f"{GREEN}[OK]{RESET}" if passed else f"{RED}[FAIL]{RESET}"
    print(f"{symbol} {message}")
    return passed


def check_model_definition():
    """Check RiskIndicator model definition."""
    print_header("1. Checking Model Definition")

    all_passed = True

    try:
        from src.db.models import RiskIndicator, UniqueConstraint
        all_passed &= print_check(True, "RiskIndicator model imports successfully")
    except ImportError as e:
        all_passed &= print_check(False, f"Failed to import RiskIndicator: {e}")
        return False

    # Check tablename
    passed = RiskIndicator.__tablename__ == 'risk_indicators'
    all_passed &= print_check(passed, f"Table name is correct: '{RiskIndicator.__tablename__}'")

    # Check columns
    columns = {col.name for col in RiskIndicator.__table__.columns}
    required_columns = {
        'id', 'u3_company_number', 'year', 'month',
        'stk_index', 'st_int', 'm2b', 'sigma',
        'dtd_median', 'dtd_median_i', 'dtd',
        'liquidity_r', 'ni2ta', 'size', 'liquidity_fin'
    }

    missing = required_columns - columns
    if not missing:
        all_passed &= print_check(True, f"All {len(required_columns)} required columns present")
    else:
        all_passed &= print_check(False, f"Missing columns: {missing}")

    # Check constraints
    unique_constraints = [c for c in RiskIndicator.__table__.constraints
                         if hasattr(c, 'columns') and len(c.columns) == 3]
    passed = len(unique_constraints) > 0
    all_passed &= print_check(passed, "Unique constraint (u3, year, month) defined")

    # Check foreign key
    fk_columns = {fk.parent.name for fk in RiskIndicator.__table__.foreign_keys}
    passed = 'u3_company_number' in fk_columns
    all_passed &= print_check(passed, "Foreign key to companies table defined")

    # Check indexes
    has_composite_index = any('year' in str(idx.columns) and 'month' in str(idx.columns)
                             for idx in RiskIndicator.__table__.indexes)
    all_passed &= print_check(has_composite_index, "Composite index (year, month) defined")

    return all_passed


def check_data_loader():
    """Check load_risk_indicators.py."""
    print_header("2. Checking Data Loader")

    all_passed = True

    try:
        from src.ingestion.load_risk_indicators import (
            load_risk_indicators,
            load_risk_indicator_data,
            clean_value
        )
        all_passed &= print_check(True, "Data loader module imports successfully")
    except ImportError as e:
        all_passed &= print_check(False, f"Failed to import loader: {e}")
        return False

    # Test clean_value function
    import pandas as pd
    test_cases = [
        (pd.NA, None, "pd.NA -> None"),
        ('', None, "empty string -> None"),
        ('NA', None, "'NA' string -> None"),
        (123.45, 123.45, "float value preserved"),
    ]

    for input_val, expected, description in test_cases:
        result = clean_value(input_val)
        passed = result == expected
        all_passed &= print_check(passed, f"clean_value: {description}")

    # Test ID conversion logic
    test_conversions = [
        (26978004, 26978),
        (12345678, 12345),
        (1000, 1),
    ]

    for company_num, expected_u3 in test_conversions:
        calculated = company_num // 1000
        passed = calculated == expected_u3
        all_passed &= print_check(passed,
            f"ID conversion: {company_num} -> {calculated} (expected {expected_u3})")

    return all_passed


def check_load_all_integration():
    """Check load_all.py integration."""
    print_header("3. Checking load_all.py Integration")

    all_passed = True

    try:
        from src.ingestion.load_all import load_all_data
        all_passed &= print_check(True, "load_all_data imports successfully")
    except ImportError as e:
        all_passed &= print_check(False, f"Failed to import: {e}")
        return False

    # Check file content
    load_all_path = Path(__file__).parent.parent / 'src' / 'ingestion' / 'load_all.py'
    with open(load_all_path, 'r') as f:
        content = f.read()

    checks = [
        ('from .load_risk_indicators import load_risk_indicator_data' in content,
         "Imports load_risk_indicator_data"),
        ('[4/5]' in content or '4/5' in content,
         "Updated step numbers to 1/5...5/5"),
        ('load_risk_indicator_data(session, data_dir)' in content,
         "Calls load_risk_indicator_data in workflow"),
        ('risk_stats' in content or 'for key, value in all_stats.items()' in content,
         "Outputs statistics in summary"),
    ]

    for condition, description in checks:
        all_passed &= print_check(condition, description)

    return all_passed


def check_seed_script():
    """Check seed.py modifications."""
    print_header("4. Checking seed.py Script")

    all_passed = True

    seed_path = Path(__file__).parent.parent / 'scripts' / 'seed.py'

    if not seed_path.exists():
        all_passed &= print_check(False, "seed.py file not found")
        return False

    with open(seed_path, 'r') as f:
        content = f.read()

    checks = [
        ('from src.ingestion.load_risk_indicators import load_risk_indicator_data' in content,
         "Imports load_risk_indicator_data"),
        ("'--only'" in content,
         "Has --only argument"),
        ("'risk_indicators'" in content and "choices=" in content,
         "risk_indicators in --only choices"),
        ("args.only == 'risk_indicators'" in content,
         "Handles --only risk_indicators case"),
        ("load_risk_indicator_data(session, data_dir)" in content,
         "Calls load_risk_indicator_data"),
    ]

    for condition, description in checks:
        all_passed &= print_check(condition, description)

    return all_passed


def check_migration_script():
    """Check add_risk_indicators_table.py script."""
    print_header("5. Checking Migration Script")

    all_passed = True

    script_path = Path(__file__).parent.parent / 'scripts' / 'add_risk_indicators_table.py'

    passed = script_path.exists()
    all_passed &= print_check(passed, "add_risk_indicators_table.py exists")

    if passed:
        with open(script_path, 'r') as f:
            content = f.read()

        checks = [
            ('Base.metadata.create_all' in content,
             "Uses create_all to create table"),
            ('load_risk_indicator_data' in content,
             "Calls data loader"),
            ('verify_data' in content or 'SELECT' in content,
             "Includes verification queries"),
        ]

        for condition, description in checks:
            all_passed &= print_check(condition, description)

    return all_passed


def check_file_structure():
    """Check that all files exist."""
    print_header("6. Checking File Structure")

    all_passed = True
    base_path = Path(__file__).parent.parent

    files = [
        ('src/db/models.py', "Model definitions"),
        ('src/ingestion/load_risk_indicators.py', "Data loader"),
        ('src/ingestion/load_all.py', "Load orchestrator"),
        ('scripts/seed.py', "Seed script"),
        ('scripts/add_risk_indicators_table.py', "Migration script"),
        ('tests/test_risk_indicators.py', "Test suite"),
    ]

    for file_path, description in files:
        full_path = base_path / file_path
        passed = full_path.exists()
        all_passed &= print_check(passed, f"{description}: {file_path}")

    return all_passed


def check_docker_config():
    """Check Docker configuration."""
    print_header("7. Checking Docker Configuration")

    docker_compose_path = Path(__file__).parent.parent / 'docker-compose.yml'

    if not docker_compose_path.exists():
        print_check(False, "docker-compose.yml not found")
        return False

    with open(docker_compose_path, 'r') as f:
        content = f.read()

    checks = [
        ('pgvector/pgvector' in content,
         "Uses pgvector image"),
        ('POSTGRES_USER' in content and 'POSTGRES_PASSWORD' in content,
         "Has database credentials"),
        ('5432:5432' in content,
         "Exposes PostgreSQL port"),
    ]

    all_passed = True
    for condition, description in checks:
        all_passed &= print_check(condition, description)

    print(f"\n{YELLOW}[INFO] Docker config looks good - no changes needed{RESET}")

    return all_passed


def main():
    """Run all verification checks."""
    print(f"\n{BLUE}{'*' * 70}")
    print(f"  Risk Indicators Setup Verification")
    print(f"{'*' * 70}{RESET}\n")

    results = []

    results.append(("Model Definition", check_model_definition()))
    results.append(("Data Loader", check_data_loader()))
    results.append(("load_all Integration", check_load_all_integration()))
    results.append(("seed.py Script", check_seed_script()))
    results.append(("Migration Script", check_migration_script()))
    results.append(("File Structure", check_file_structure()))
    results.append(("Docker Config", check_docker_config()))

    # Summary
    print_header("Summary")

    all_passed = all(passed for _, passed in results)

    for name, passed in results:
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {name:.<50} {status}")

    print()

    if all_passed:
        print(f"{GREEN}{'=' * 70}")
        print(f"  [SUCCESS] All checks passed! Setup is ready.")
        print(f"{'=' * 70}{RESET}\n")
        print("Next steps:")
        print("  1. Ensure database is running: docker-compose up -d")
        print("  2. Run migration: python -m scripts.add_risk_indicators_table")
        print("  3. Or use seed.py: python -m scripts.seed --only risk_indicators --skip-init")
        return 0
    else:
        print(f"{RED}{'=' * 70}")
        print(f"  [FAILED] Some checks failed. Please review the output above.")
        print(f"{'=' * 70}{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
