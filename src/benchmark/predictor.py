"""Benchmark predictor — thin wrapper around pipeline.predictor.predict()."""

import logging

from src.benchmark.case_builder import CaseRecord
from src.benchmark.config import BenchmarkConfig
from src.pipeline.predictor import predict as _pipeline_predict

logger = logging.getLogger(__name__)


def predict(
    prompt: str,
    case: CaseRecord,
    cfg: BenchmarkConfig,
) -> dict:
    """Call Claude with a benchmark prompt and return parsed prediction.

    Args:
        prompt: Full user message (from prompt_builder.build_prompt)
        case:   CaseRecord — used to pass cutoff_date and horizon to predictor
        cfg:    BenchmarkConfig

    Returns:
        Prediction dict with keys:
            prediction, confidence, probability_pct,
            key_risk_factors, protective_factors, reasoning, data_gaps,
            _model, _usage
        On failure: {"error": "...", "probability_pct": None}
    """
    logger.info(
        f"  [{case.group}] {case.company_name} cutoff={case.cutoff_date} → {cfg.llm_model}"
    )

    result = _pipeline_predict(
        context=prompt,
        model=cfg.llm_model,
        cutoff_date=case.cutoff_date,
        horizon_months=cfg.prediction_horizon_months,
    )

    # Ensure probability_pct key exists for evaluator compatibility
    if "error" in result and "probability_pct" not in result:
        result["probability_pct"] = None

    return result
