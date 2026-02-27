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
            Note: data only covers 1988–2003; may be missing for modern events.
        include_market_data: Include daily price/marketcap/volume from market_data.
        include_macro: Include macro context (VIX, yields, etc.).
        include_fundamentals: Include financial statements from DuckDB parquet.
            Skipped automatically if parquet files are not found locally.
        output_dir: Directory to write JSONL and summary files.
        random_seed: Seed for reproducible control-group sampling.
    """
    prediction_horizon_months: int = 6
    lookback_months: int = 12
    n_default_cases: int = 100
    n_control_cases: int = 100
    llm_model: str = "claude-sonnet-4-6"
    include_risk_indicators: bool = True
    include_market_data: bool = True
    include_macro: bool = True
    include_fundamentals: bool = True
    output_dir: str = "results/benchmark"
    random_seed: int = 42
