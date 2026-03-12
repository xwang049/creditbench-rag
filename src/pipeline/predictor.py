"""LLM-based credit event prediction using OpenAI."""

import json
import logging
import re
from datetime import date
from typing import Optional

from openai import OpenAI

from src.config import config

logger = logging.getLogger(__name__)

# Module-level client — created once, reused across all calls
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not configured")
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client

# System prompt: role definition, output schema, scoring guidelines.
# Time constraints (cutoff date, horizon) belong in the user message task header
# (built by prompt_builder.build_prompt), not here — avoids sending the same
# information twice and keeps system prompt stable across experiments.
SYSTEM_PROMPT = """\
You are a senior credit risk analyst. You will be given a structured financial \
profile of a company — including pre-computed credit-risk ratios, historical \
financial statements, earnings call excerpts, past credit events, and macro context.

Your task: assess the probability that this company will experience a \
**credit event** (bankruptcy filing, debt default, delisting, or severe rating \
downgrade) within the prediction window defined in the task header below.

The task header specifies the data cutoff date and prediction window. \
Do NOT use any knowledge about events that occurred after the cutoff date. \
Base your assessment solely on the data provided.

Respond in **exactly** this JSON format (no markdown fences):
{
  "prediction": "HIGH" | "MEDIUM" | "LOW",
  "confidence": <0–100>,
  "probability_pct": <0–100>,
  "key_risk_factors": ["...", "..."],
  "protective_factors": ["...", "..."],
  "reasoning": "2-3 paragraph analysis",
  "data_gaps": ["any missing data that limits your analysis"]
}

Guidelines:
- HIGH = >30% probability of credit event in the prediction horizon
- MEDIUM = 10–30% probability
- LOW = <10% probability
- confidence reflects how certain you are given the available data
- Be specific: cite numbers (Altman Z, interest coverage, revenue trend, etc.)
- If critical data is missing, note it in data_gaps and adjust confidence down
- Consider both company-specific signals AND the macro environment
"""


def predict(
    context: str,
    model: str = "gpt-4o-mini",
    cutoff_date: Optional[date] = None,   # kept for API compat, no longer used here
    horizon_months: Optional[int] = None,  # kept for API compat, no longer used here
) -> dict:
    """Run credit event prediction given a fully-assembled prompt.

    Args:
        context:        Complete user message — either the raw context string
                        (predict_company path) or the full prompt from
                        prompt_builder.build_prompt() (benchmark path).
                        Time constraints must be in this string; they are NOT
                        duplicated into the system prompt.
        model:          OpenAI model ID
        cutoff_date:    Unused — kept for backward compatibility
        horizon_months: Unused — kept for backward compatibility

    Returns:
        Parsed prediction dict, or a dict with 'error' key on failure.
    """
    try:
        client = _get_client()
    except RuntimeError as e:
        return {"error": str(e)}

    logger.info("Calling OpenAI for credit prediction...")

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context},
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
    model: str = "gpt-4o-mini",
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
    user_message = (
        "Analyse the following company and predict its credit event risk.\n\n"
        "---\n"
        f"{context}\n"
        "---\n\n"
        "Return your assessment as JSON."
    )
    prediction = predict(user_message, model=model)

    return {
        "company": company,
        "prediction": prediction,
        "context_length": len(context),
    }
