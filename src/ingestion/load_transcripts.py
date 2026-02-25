"""Load earnings call transcripts into transcript_chunks table with Gemini embeddings.

Parquet columns (unnamed → mapped):
  _COL_0  company_name        _COL_1  bb_company_id
  _COL_6  event_title         _COL_7  call_start (datetime)
  _COL_9  year                _COL_10 quarter
  _COL_13 utterance_type      _COL_14 speaker_name
  _COL_15 speaker_type        _COL_17 paragraph_id (unique)
  _COL_18 text_content        _COL_19 speaker_title
  _COL_20 u3_company_number

Filter: skip Operator rows and texts shorter than MIN_TEXT_LEN chars.
Embedding: gemini-embedding-001, 768 dims, batch size EMBED_BATCH_SIZE.
Incremental: uses paragraph_id as unique key — already-embedded rows are skipped.
"""

import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.config import config
from src.db.models import TranscriptChunk

logger = logging.getLogger(__name__)

PARQUET_FILENAME = "earnings_call_transcripts/SP1500_earnings_call_transcripts.parquet"
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMS = 768
EMBED_BATCH_SIZE = 50      # texts per Gemini API call (free tier: 100 RPM)
READ_CHUNK_SIZE = 200_000  # rows read from parquet at a time
MIN_TEXT_LEN = 50          # skip very short utterances ("Yes.", "Thank you.")
SKIP_SPEAKER_TYPES = {"Operator"}
REQUEST_INTERVAL = 0.65    # seconds between API calls to stay under 100 RPM
MAX_RETRIES = 5


def _get_gemini_client() -> genai.Client:
    key = os.getenv("GOOGLE_API_KEY") or config.GOOGLE_API_KEY
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set in environment / .env")
    return genai.Client(api_key=key)


def _parse_retry_delay(error_str: str) -> float:
    """Extract retryDelay seconds from a 429 error message, default 60s."""
    m = re.search(r"retryDelay.*?(\d+)s", str(error_str))
    return float(m.group(1)) + 2.0 if m else 62.0


def _embed_batch(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with retry on 429. Returns list of 768-dim vectors."""
    for attempt in range(MAX_RETRIES):
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(output_dimensionality=EMBED_DIMS),
            )
            time.sleep(REQUEST_INTERVAL)
            return [e.values for e in result.embeddings]
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = _parse_retry_delay(err)
                logger.warning(f"  Rate limit hit — waiting {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Embedding failed after {MAX_RETRIES} retries")


def _get_existing_paragraph_ids(session: Session) -> set[int]:
    """Return the set of paragraph_ids already in the DB."""
    rows = session.execute(text("SELECT paragraph_id FROM transcript_chunks")).fetchall()
    return {r[0] for r in rows}


def load_transcripts(session: Session, data_dir: Path) -> dict:
    """Incrementally embed and load transcript chunks.

    Only rows whose paragraph_id is not yet in transcript_chunks are processed.
    """
    parquet_path = data_dir / PARQUET_FILENAME
    if not parquet_path.exists():
        raise FileNotFoundError(f"Transcripts parquet not found: {parquet_path}")

    client = _get_gemini_client()

    logger.info("Loading existing paragraph IDs from DB...")
    existing_ids = _get_existing_paragraph_ids(session)
    logger.info(f"{len(existing_ids):,} chunks already in DB")

    inserted = 0
    skipped_existing = 0
    skipped_short = 0
    skipped_operator = 0
    api_errors = 0

    logger.info(f"Reading parquet in chunks of {READ_CHUNK_SIZE:,} rows...")
    pf = pd.read_parquet(parquet_path, engine="pyarrow")
    total_rows = len(pf)
    logger.info(f"Total parquet rows: {total_rows:,}")

    for chunk_start in range(0, total_rows, READ_CHUNK_SIZE):
        chunk = pf.iloc[chunk_start: chunk_start + READ_CHUNK_SIZE].copy()

        # Build candidate rows
        pending: list[dict] = []
        for _, row in chunk.iterrows():
            pid = int(row["_COL_17"])

            if pid in existing_ids:
                skipped_existing += 1
                continue

            speaker_type = str(row["_COL_15"]) if pd.notna(row["_COL_15"]) else ""
            if speaker_type in SKIP_SPEAKER_TYPES:
                skipped_operator += 1
                continue

            text_content = str(row["_COL_18"]) if pd.notna(row["_COL_18"]) else ""
            if len(text_content) < MIN_TEXT_LEN:
                skipped_short += 1
                continue

            call_ts = row["_COL_7"]
            call_date = call_ts.date() if pd.notna(call_ts) else None

            year_val = row["_COL_9"]
            year = int(year_val) if pd.notna(year_val) else None

            quarter_val = row["_COL_10"]
            quarter = int(quarter_val) if pd.notna(quarter_val) else None

            u3_val = row["_COL_20"]
            u3 = int(u3_val) if pd.notna(u3_val) else None

            pending.append({
                "paragraph_id": pid,
                "u3_company_number": u3,
                "call_date": call_date,
                "year": year,
                "quarter": quarter,
                "event_title": str(row["_COL_6"])[:500] if pd.notna(row["_COL_6"]) else None,
                "speaker_type": speaker_type[:50] if speaker_type else None,
                "utterance_type": str(row["_COL_13"])[:80] if pd.notna(row["_COL_13"]) else None,
                "speaker_name": str(row["_COL_14"])[:200] if pd.notna(row["_COL_14"]) else None,
                "speaker_title": str(row["_COL_19"])[:200] if pd.notna(row["_COL_19"]) else None,
                "text_content": text_content,
            })

        if not pending:
            logger.info(
                f"  Chunk {chunk_start:,}–{chunk_start + len(chunk):,}: all skipped"
            )
            continue

        # Embed in sub-batches
        for batch_start in range(0, len(pending), EMBED_BATCH_SIZE):
            batch = pending[batch_start: batch_start + EMBED_BATCH_SIZE]
            texts = [r["text_content"] for r in batch]

            try:
                vectors = _embed_batch(client, texts)
            except Exception as e:
                logger.warning(f"  Embedding error (skipping batch of {len(batch)}): {e}")
                api_errors += len(batch)
                time.sleep(2)
                continue

            now = datetime.now(timezone.utc)
            records = []
            for rec, vec in zip(batch, vectors):
                rec["embedding"] = vec
                rec["embedding_model"] = EMBED_MODEL
                rec["embedded_at"] = now
                records.append(rec)

            stmt = (
                insert(TranscriptChunk)
                .values(records)
                .on_conflict_do_nothing(index_elements=["paragraph_id"])
            )
            session.execute(stmt)
            session.commit()

            inserted += len(records)
            # Track inserted ids so we don't re-process within the same run
            for r in records:
                existing_ids.add(r["paragraph_id"])

        logger.info(
            f"  Progress: {chunk_start + len(chunk):,}/{total_rows:,} rows scanned | "
            f"{inserted:,} inserted | {skipped_existing:,} existed | "
            f"{skipped_operator:,} operator | {skipped_short:,} short"
        )

    logger.info(
        f"[OK] Transcripts done — {inserted:,} new chunks, "
        f"{skipped_existing:,} existed, {skipped_operator:,} operator, "
        f"{skipped_short:,} too short, {api_errors:,} api errors"
    )
    return {
        "transcripts_new": inserted,
        "transcripts_skipped_existing": skipped_existing,
        "transcripts_skipped_operator": skipped_operator,
        "transcripts_skipped_short": skipped_short,
        "transcripts_api_errors": api_errors,
    }
