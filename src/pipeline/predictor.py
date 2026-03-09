"""LLM-based credit event prediction using OpenAI."""

import json
import logging
import re
from datetime import date
from typing import Optional

from openai import OpenAI

from src.config import config

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_TEMPLATE = """\
You are a senior credit risk analyst. You will be given a structured financial \
profile of a company — including historical financial statements, market data, \
risk indicators, past credit events, and macro context.

Your task: assess whether this company is likely to experience a **credit event** \
(bankruptcy filing, debt default, delisting, or severe rating downgrade) \
{horizon_clause}.

IMPORTANT: All data provided is as of {cutoff_date}. Do NOT use any knowledge \
about events that occurred after {cutoff_date}. Base your assessment solely on \
the data provided.

Respond in **exactly** this JSON format (no markdown fences):
{{
  "prediction": "HIGH" | "MEDIUM" | "LOW",
  "confidence": <0–100>,
  "probability_pct": <0–100>,
  "key_risk_factors": ["...", "..."],
  "protective_factors": ["...", "..."],
  "reasoning": "2-3 paragraph analysis",
  "data_gaps": ["any missing data that limits your analysis"]
}}

Guidelines:
- HIGH = >30% probability of credit event in the prediction horizon
- MEDIUM = 10–30% probability
- LOW = <10% probability
- confidence reflects how certain you are in your assessment given the available data
- Be specific: cite numbers from the data (revenue trends, leverage ratios, DTD levels, etc.)
- If critical data is missing (e.g. no recent financials), note it in data_gaps and \
  adjust confidence downward accordingly
- Consider both company-specific factors AND the macro environment
"""

# Default system prompt (no time constraint — for ad-hoc queries)
SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE.format(
    horizon_clause="within the next 12–24 months",
    cutoff_date="the most recent available data",
)


def predict(
    context: str,
    model: str = "gpt-4o-mini",
    cutoff_date: Optional[date] = None,
    horizon_months: Optional[int] = None,
) -> dict:
    """Run credit event prediction given a formatted company context.

    Args:
        context:        Structured text from context_builder.build_context()
        model:          OpenAI model ID
        cutoff_date:    Data cutoff date (for benchmark experiments, prevents look-ahead)
        horizon_months: Prediction horizon in months (default: 12-24 generic)

    Returns:
        Parsed prediction dict, or a dict with 'error' key on failure.
    """
    if not config.OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY not configured"}

    # Build time-aware system prompt if cutoff_date is specified
    if cutoff_date is not None:
        cutoff_str = cutoff_date.isoformat()
        if horizon_months:
            from dateutil.relativedelta import relativedelta
            end_date = cutoff_date + relativedelta(months=horizon_months)
            horizon_clause = (
                f"within the next {horizon_months} months "
                f"(i.e., between {cutoff_str} and {end_date.isoformat()})"
            )
        else:
            horizon_clause = "within the next 12–24 months"
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            horizon_clause=horizon_clause,
            cutoff_date=cutoff_str,
        )
    else:
        system_prompt = SYSTEM_PROMPT

    client = OpenAI(api_key=config.OPENAI_API_KEY)

    user_message = (
        "Analyse the following company and predict its credit event risk.\n\n"
        "---\n"
        f"{context}\n"
        "---\n\n"
        "Return your assessment as JSON."
    )

    logger.info("Calling OpenAI for credit prediction...")

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return {"error": str(e)}

    raw = response.choices[0].message.content.strip()
    logger.debug(f"Raw LLM response:\n{raw}")

    # Parse JSON — strip markdown fences if the model wraps them anyway
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM JSON; returning raw text")
        result = {"raw_response": raw, "error": "JSON parse failed"}

    result["_model"] = model
    result["_usage"] = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return result


def predict_company(
    session,
    identifier: str,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """End-to-end: identifier → data fetch → context → LLM prediction.

    This is the single-call convenience entry point.
    """
    from src.pipeline.company_lookup import lookup
    from src.pipeline.data_fetcher import fetch_all
    from src.pipeline.context_builder import build_context

    company = lookup(session, identifier)
    data = fetch_all(session, company)
    context = build_context(data)
    prediction = predict(context, model=model)

    return {
        "company": company,
        "prediction": prediction,
        "context_length": len(context),
    }
