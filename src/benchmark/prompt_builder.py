"""Build the LLM prompt for a benchmark case.

Prepends a task header (cutoff_date, horizon, ground-truth suppression reminder)
to the standard context produced by context_builder.build_context().
"""

from dateutil.relativedelta import relativedelta

from src.benchmark.case_builder import CaseRecord
from src.benchmark.config import BenchmarkConfig
from src.pipeline.context_builder import build_context


def build_prompt(
    data: dict,
    case: CaseRecord,
    cfg: BenchmarkConfig,
) -> str:
    """Build the full prompt string for a benchmark case.

    Args:
        data:  Dict returned by benchmark.data_fetcher.fetch_for_case()
        case:  CaseRecord (used for cutoff_date and horizon)
        cfg:   BenchmarkConfig

    Returns:
        Complete prompt text to send as the user message to the LLM.
    """
    cutoff_str = case.cutoff_date.isoformat()
    end_date = case.cutoff_date + relativedelta(months=cfg.prediction_horizon_months)
    end_str = end_date.isoformat()

    task_header = (
        f"=== PREDICTION TASK ===\n"
        f"Data cutoff date : {cutoff_str}\n"
        f"Prediction window: {cfg.prediction_horizon_months} months "
        f"({cutoff_str} → {end_str})\n"
        f"\n"
        f"Predict whether this company will experience a significant credit event\n"
        f"(default, bankruptcy filing, or equivalent) within the prediction window.\n"
        f"\n"
        f"IMPORTANT: You must NOT use any information about events that occurred\n"
        f"after {cutoff_str}. Base your assessment only on the data below.\n"
        f"{'=' * 55}\n\n"
    )

    company_context = build_context(data)

    return (
        "Analyse the following company and predict its credit event risk.\n\n"
        "---\n"
        f"{task_header}"
        f"{company_context}\n"
        "---\n\n"
        "Return your assessment as JSON."
    )
