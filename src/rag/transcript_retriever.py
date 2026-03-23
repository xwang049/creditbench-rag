"""Semantic search over earnings call transcript chunks.

Two backends:
  - search_transcripts_parquet(): DuckDB + numpy brute-force (primary)
  - search_transcripts():         pgvector cosine search (legacy fallback)
"""

import logging
import os
from datetime import date
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
from openai import OpenAI
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.config import config

logger = logging.getLogger(__name__)

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536

# Directory containing per-batch embedding parquet files.
# Defaults to OneDrive shared library; override with EMBED_DIR env var.
EMBED_PARQUET_DIR = Path(os.getenv(
    "EMBED_DIR",
    os.path.expanduser(
        "~/Library/CloudStorage/OneDrive-共享的库-NationalUniversityofSingapore"
        "/Li Shuping - Data/embeddings"
    ),
))

# Module-level singletons
_client: OpenAI | None = None
_embed_con: duckdb.DuckDBPyConnection | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.getenv("OPENAI_API_KEY") or config.OPENAI_API_KEY
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set in environment / .env")
        _client = OpenAI(api_key=key)
    return _client


def _get_embed_con() -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection with the embeddings view registered."""
    global _embed_con
    if _embed_con is None:
        if not EMBED_PARQUET_DIR.exists() or not list(EMBED_PARQUET_DIR.glob("*.parquet")):
            raise FileNotFoundError(
                f"No embedding parquet files found in {EMBED_PARQUET_DIR}. "
                "Run `scripts/batch_embed.py ingest` first."
            )
        _embed_con = duckdb.connect()
        _embed_con.execute(
            f"CREATE VIEW embeddings AS SELECT * FROM read_parquet('{EMBED_PARQUET_DIR}/*.parquet')"
        )
    return _embed_con


def embed_query(query: str) -> list[float]:
    """Embed a single query string for retrieval."""
    client = _get_client()
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=query,
    )
    return response.data[0].embedding


def search_transcripts_parquet(
    query: str,
    u3_company_number: int,
    k: int = 8,
    call_date_to: Optional[date] = None,
    year_to: Optional[int] = None,
) -> list[dict]:
    """Semantic search over embedding parquet files using DuckDB + numpy.

    Steps:
      1. DuckDB filters rows by company + date (pushes predicates into parquet stats)
      2. numpy brute-force cosine similarity on the filtered candidate set
      3. Returns top-k results sorted by similarity

    Args:
        query:              Natural language query to embed
        u3_company_number:  Company filter (required — keeps candidate set small)
        k:                  Number of results
        call_date_to:       Hard cutoff on call_date (prevents look-ahead bias)
        year_to:            Optional year upper bound (inclusive)

    Returns:
        List of dicts matching the shape returned by search_transcripts()
    """
    try:
        con = _get_embed_con()
    except FileNotFoundError as e:
        logger.warning(str(e))
        return []

    # Build DuckDB filter
    filters = ["u3_company_number = ?"]
    params: list = [u3_company_number]
    if call_date_to is not None:
        filters.append("(call_date IS NULL OR call_date <= ?)")
        params.append(call_date_to)
    if year_to is not None:
        filters.append("(year IS NULL OR year <= ?)")
        params.append(year_to)

    where = " AND ".join(filters)
    df = con.execute(
        f"SELECT paragraph_id, call_date, year, quarter, speaker_type, "
        f"speaker_name, text_content, embedding "
        f"FROM embeddings WHERE {where}",
        params,
    ).df()

    if df.empty:
        return []

    # Embed query and compute cosine similarity
    # OpenAI text-embedding-3-small vectors are already L2-normalised,
    # so dot product == cosine similarity
    query_vec = np.array(embed_query(query), dtype=np.float32)
    embeddings = np.stack(df["embedding"].values).astype(np.float32)  # (N, 1536)
    similarities = embeddings @ query_vec                               # (N,)

    top_idx = np.argsort(similarities)[::-1][:k]

    results = []
    for idx in top_idx.tolist():
        row = df.iloc[idx]
        results.append({
            "paragraph_id":      int(row["paragraph_id"]),
            "u3_company_number": u3_company_number,
            "call_date":         str(row["call_date"]) if row["call_date"] is not None else None,
            "year":              int(row["year"]) if row["year"] is not None else None,
            "quarter":           int(row["quarter"]) if row["quarter"] is not None else None,
            "speaker_type":      row["speaker_type"],
            "speaker_name":      row["speaker_name"],
            "text_content":      row["text_content"],
            "similarity":        round(float(similarities[idx]), 4),
        })

    return results


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
