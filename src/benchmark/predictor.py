"""Benchmark predictor: calls Claude and returns a structured prediction dict.

Re-uses src.pipeline.predictor internals but passes cutoff_date and
horizon_months so the system prompt is properly time-bounded.
"""

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
    # The pipeline predictor accepts a context string in its user message.
    # We bypass build_context() here because prompt_builder already embedded
    # the full context + task header into `prompt`.
    # We call _pipeline_predict directly with the pre-built prompt by
    # temporarily monkey-patching — instead, we call the Anthropic client
    # directly to avoid double-wrapping.

    import json
    import re
    from anthropic import Anthropic
    from src.config import config
    from src.pipeline.predictor import _SYSTEM_PROMPT_TEMPLATE

    if not config.ANTHROPIC_API_KEY:
        return {"error": "ANTHROPIC_API_KEY not configured", "probability_pct": None}

    cutoff_str = case.cutoff_date.isoformat()
    from dateutil.relativedelta import relativedelta
    end_date = case.cutoff_date + relativedelta(months=cfg.prediction_horizon_months)
    horizon_clause = (
        f"within the next {cfg.prediction_horizon_months} months "
        f"(i.e., between {cutoff_str} and {end_date.isoformat()})"
    )
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        horizon_clause=horizon_clause,
        cutoff_date=cutoff_str,
    )

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

    logger.info(
        f"  [{case.group}] {case.company_name} cutoff={cutoff_str} → calling Claude"
    )

    try:
        response = client.messages.create(
            model=cfg.llm_model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
    except Exception as e:
        logger.error(f"Claude API error for {case.company_name}: {e}")
        return {"error": str(e), "probability_pct": None}

    raw = response.content[0].text.strip()
    logger.debug(f"Raw LLM response:\n{raw}")

    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"JSON parse failed for {case.company_name}")
        result = {"raw_response": raw, "error": "JSON parse failed", "probability_pct": None}

    result["_model"] = cfg.llm_model
    result["_usage"] = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return result
