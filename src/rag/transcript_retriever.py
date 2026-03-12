"""Semantic search over earnings call transcript chunks using pgvector."""

import logging
import os
from typing import Optional

from openai import OpenAI
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.config import config

logger = logging.getLogger(__name__)

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536

# Module-level client — created once, reused across all search calls
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.getenv("OPENAI_API_KEY") or config.OPENAI_API_KEY
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set in environment / .env")
        _client = OpenAI(api_key=key)
    return _client


def embed_query(query: str) -> list[float]:
    """Embed a single query string for retrieval."""
    client = _get_client()
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=query,
    )
    return response.data[0].embedding


def search_transcripts(
    session: Session,
    query: str,
    k: int = 10,
    u3_company_number: Optional[int] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    call_date_to=None,          # date | str — hard cutoff on call_date (prevents look-ahead)
    quarter: Optional[int] = None,
    speaker_type: Optional[str] = None,
) -> list[dict]:
    """Semantic search over transcript_chunks.

    Args:
        session:            DB session
        query:              Natural language query
        k:                  Number of results to return
        u3_company_number:  Filter to a specific company
        year_from:          Filter calls from this year (inclusive)
        year_to:            Filter calls up to this year (inclusive)
        call_date_to:       Strict date cutoff — only transcripts with
                            call_date <= this value (use as_of from benchmark
                            to prevent look-ahead bias within a calendar year)
        quarter:            Filter to a specific quarter (1-4)
        speaker_type:       Filter by speaker type ('Executives', 'Analysts', ...)

    Returns:
        List of dicts with chunk metadata + similarity score
    """
    vec = embed_query(query)
    # vec contains float32 values from OpenAI — safe to format as a literal
    # pgvector string.  CAST(:vec AS vector) is the recommended parameterised
    # pattern; the string is never interpolated into SQL directly.
    vec_str = "[" + ",".join(str(x) for x in vec) + "]"

    filters = ["tc.embedding IS NOT NULL"]
    params: dict = {"k": k, "vec": vec_str}

    if u3_company_number is not None:
        filters.append("tc.u3_company_number = :u3")
        params["u3"] = u3_company_number
    if year_from is not None:
        filters.append("tc.year >= :year_from")
        params["year_from"] = year_from
    if year_to is not None:
        filters.append("tc.year <= :year_to")
        params["year_to"] = year_to
    if call_date_to is not None:
        filters.append("(tc.call_date IS NULL OR tc.call_date <= :call_date_to)")
        params["call_date_to"] = call_date_to
    if quarter is not None:
        filters.append("tc.quarter = :quarter")
        params["quarter"] = quarter
    if speaker_type is not None:
        filters.append("tc.speaker_type = :speaker_type")
        params["speaker_type"] = speaker_type

    where_clause = " AND ".join(filters)

    sql = text(f"""
        SELECT
            tc.id,
            tc.paragraph_id,
            tc.u3_company_number,
            c.company_name,
            tc.call_date,
            tc.year,
            tc.quarter,
            tc.event_title,
            tc.speaker_type,
            tc.utterance_type,
            tc.speaker_name,
            tc.speaker_title,
            tc.text_content,
            1 - (tc.embedding <=> CAST(:vec AS vector)) AS similarity
        FROM transcript_chunks tc
        LEFT JOIN companies c ON c.u3_company_number = tc.u3_company_number
        WHERE {where_clause}
        ORDER BY tc.embedding <=> CAST(:vec AS vector)
        LIMIT :k
    """)

    rows = session.execute(sql, params).fetchall()

    return [
        {
            "id": r.id,
            "paragraph_id": r.paragraph_id,
            "u3_company_number": r.u3_company_number,
            "company_name": r.company_name,
            "call_date": str(r.call_date) if r.call_date else None,
            "year": r.year,
            "quarter": r.quarter,
            "event_title": r.event_title,
            "speaker_type": r.speaker_type,
            "utterance_type": r.utterance_type,
            "speaker_name": r.speaker_name,
            "speaker_title": r.speaker_title,
            "text_content": r.text_content,
            "similarity": round(float(r.similarity), 4),
        }
        for r in rows
    ]
