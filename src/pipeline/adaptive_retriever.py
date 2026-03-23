"""Layer 2: Adaptive retrieval — build transcript query from financial signals.

Uses Layer 1 signals to construct a semantically targeted query, then calls
parquet + numpy cosine search to retrieve the most relevant earnings call excerpts.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from sqlalchemy.orm import Session

from src.rag.transcript_retriever import search_transcripts_parquet

logger = logging.getLogger(__name__)


def build_adaptive_query(signals: dict, industry: Optional[str]) -> str:
    """Construct a semantic search query from credit-risk signals.

    Always starts with a management outlook anchor; adds distress-specific
    terms when signals are in stressed territory.
    """
    latest = signals.get("latest", {})
    trends = signals.get("trends", {})

    parts: list[str] = ["management outlook guidance financial performance"]

    if industry:
        parts.append(industry.lower())

    # ── Leverage stress ──────────────────────────────────────────────────────
    ic = latest.get("interest_coverage")
    if ic is not None and ic < 2.0:
        parts.append("debt refinancing covenant waiver credit facility")

    nde = latest.get("net_debt_ebitda")
    if nde is not None and nde > 4.0:
        parts.append("leverage deleveraging debt reduction capital structure")

    # ── Revenue / margin deterioration ───────────────────────────────────────
    if trends.get("revenue") in ("↓", "↓↓"):
        parts.append("revenue decline demand weakness volume pricing pressure")

    if trends.get("ebit_margin") in ("↓", "↓↓"):
        parts.append("margin compression cost cutting efficiency restructuring")

    # ── Liquidity stress ─────────────────────────────────────────────────────
    cr = latest.get("current_ratio")
    if cr is not None and cr < 1.0:
        parts.append("liquidity working capital cash runway short-term obligations")

    cash_r = latest.get("cash_ratio")
    if cash_r is not None and cash_r < 0.2:
        parts.append("cash position burn rate liquidity management")

    # ── Profitability stress ─────────────────────────────────────────────────
    nm = latest.get("net_margin")
    if nm is not None and nm < 0:
        parts.append("losses cost reduction turnaround profitability")

    # ── Altman Z distress zone ───────────────────────────────────────────────
    zone = latest.get("altman_zone")
    if zone == "distress":
        parts.append("financial distress going concern bankruptcy risk viability")
    elif zone == "grey":
        parts.append("credit risk financial challenges debt obligations covenant")

    query = " ".join(parts)
    logger.debug(f"Adaptive query ({len(query.split())} terms): {query[:120]}…")
    return query


def retrieve_transcripts(
    session: Session,
    signals: dict,
    u3_company_number: int,
    industry: Optional[str] = None,
    as_of: Optional[date] = None,
    k: int = 8,
) -> list[dict]:
    """Retrieve the most contextually relevant transcript chunks for a case.

    Args:
        session:            Active DB session
        signals:            Output of feature_engineer.compute_financial_signals()
        u3_company_number:  Company filter
        industry:           Industry sector string (used in query construction)
        as_of:              Data cutoff — only return transcripts on/before this date
        k:                  Number of chunks to return

    Returns:
        List of transcript chunk dicts with similarity scores
    """
    year_to = as_of.year if as_of else None
    call_date_to = as_of  # strict cutoff prevents look-ahead within the year

    query = build_adaptive_query(signals, industry)

    try:
        chunks = search_transcripts_parquet(
            query=query,
            u3_company_number=u3_company_number,
            k=k,
            call_date_to=call_date_to,
            year_to=year_to,
        )
    except Exception as e:
        logger.warning(f"Adaptive retrieval failed (u3={u3_company_number}): {e}")
        return []

    logger.info(
        f"  Adaptive retrieval: {len(chunks)} chunks "
        f"(top similarity: {chunks[0]['similarity'] if chunks else 'n/a'})"
    )
    return chunks
