"""Configuration dataclass for benchmark experiments."""

from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    """Parameters controlling the benchmark experiment.

    Attributes:
        prediction_horizon_months: n — how many months into the future the LLM predicts.
            cutoff_date = event_date - prediction_horizon_months for Group 1.
        lookback_months: How many months of historical data to expose to the LLM.
        n_default_cases: Max number of default/bankruptcy cases (Group 1).
        n_control_cases: Max number of control cases (Group 2).
        llm_model: Claude model ID.
        include_risk_indicators: Include DTD/sigma/etc. from risk_indicators table.
            NOTE: data only covers 1988–2003; disabled by default for modern events.
        include_market_data: Include daily price/marketcap/volume from market_data.
            Disabled by default per researcher data-scope decision.
        include_macro: Include macro context (VIX, yields, SP500, etc.).
        include_fundamentals: Include financial statements from DuckDB parquet.
            Skipped automatically if parquet files are not found locally.
        anonymize_company: Replace company name and ticker with u3_company_number.
            Reduces look-up-table contamination from LLM training memory.
        blur_year: Replace actual calendar years in dates with relative labels
            (Y0 = cutoff year, Y-1 = one year prior, etc.).  Month/day and all
            numeric macro levels (VIX, yields, prices) are preserved unchanged.
        output_dir: Directory to write JSONL and summary files.
        random_seed: Seed for reproducible control-group sampling.
    """
    prediction_horizon_months: int = 6
    lookback_months: int = 12
    n_default_cases: int = 100
    n_control_cases: int = 100
    llm_model: str = "claude-sonnet-4-6"
    # Data-type toggles
    include_risk_indicators: bool = False   # coverage ends 2003; off by default
    include_market_data: bool = False        # excluded per data-scope decision
    include_macro: bool = True
    include_fundamentals: bool = True
    # Contamination-prevention
    anonymize_company: bool = True          # replace name/ticker with u3 ID
    blur_year: bool = True                  # map YYYY → Y0/Y-1/… in all dates
    # Output
    output_dir: str = "results/benchmark"
    random_seed: int = 42
