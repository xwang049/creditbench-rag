"""Load earnings call transcripts into transcript_chunks table with Gemini embeddings.

Data sources (all in earnings_call_transcripts/):
  1. SP1500_earnings_call_transcripts.parquet  — unnamed cols (_COL_*), has u3_company_number
  2. Credit_ECC_combined_renamed.parquet       — named cols, 3133 credit-event companies
  3. credit_transcript_missing.csv             — named cols, 8 additional companies

Unique key across all three: TRANSCRIPTCOMPONENTID (= paragraph_id in DB).
u3_company_number resolution: directly from SP1500 parquet; for the other two, looked up
via ID_BB_COMPANY → companies.id_bb_company.

Filter: skip Operator rows and texts shorter than MIN_TEXT_LEN chars.
Embedding: gemini-embedding-001, 768 dims, batch size EMBED_BATCH_SIZE.
Incremental: paragraph_id unique key — already-embedded rows are skipped.
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

TRANSCRIPT_DIR = "earnings_call_transcripts"
SP1500_FILE = "SP1500_earnings_call_transcripts.parquet"
ECC_FILE = "Credit_ECC_combined_renamed.parquet"
MISSING_FILE = "credit_transcript_missing.csv"

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMS = 768
EMBED_BATCH_SIZE = 50      # texts per Gemini API call (free tier: 100 RPM)
READ_CHUNK_SIZE = 200_000  # rows read from parquet at a time
MIN_TEXT_LEN = 50          # skip very short utterances ("Yes.", "Thank you.")
SKIP_SPEAKER_TYPES = {"Operator"}
REQUEST_INTERVAL = 0.65    # seconds between API calls to stay under 100 RPM
MAX_RETRIES = 5


# ── Column mapping for named-column files (ECC parquet + CSV) ──────────────
NAMED_COL_MAP = {
    "TRANSCRIPTCOMPONENTID": "paragraph_id",
    "ID_BB_COMPANY":         "bb_company_id",   # used to look up u3_company_number
    "MOSTIMPORTANTDATEUTC":  "call_start",
    "FISCALYEAR":            "year",
    "FISCALQUARTER":         "quarter",
    "HEADLINE":              "event_title",
    "TRANSCRIPTCOMPONENTTYPENAME": "utterance_type",
    "TRANSCRIPTPERSONNAME":  "speaker_name",
    "SPEAKERTYPENAME":       "speaker_type",
    "COMPONENTTEXT":         "text_content",
    "TITLE":                 "speaker_title",
}


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
                logger.warning(f"  Rate limit — waiting {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Embedding failed after {MAX_RETRIES} retries")


def _get_existing_paragraph_ids(session: Session) -> set[int]:
    rows = session.execute(text("SELECT paragraph_id FROM transcript_chunks")).fetchall()
    return {r[0] for r in rows}


def _build_bb_to_u3_map(session: Session) -> dict[int, int]:
    """Build ID_BB_COMPANY → u3_company_number mapping from companies table."""
    rows = session.execute(
        text("SELECT id_bb_company, u3_company_number FROM companies WHERE id_bb_company IS NOT NULL")
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def _load_sp1500(data_dir: Path) -> pd.DataFrame | None:
    """Load SP1500 parquet (unnamed columns) into normalised DataFrame."""
    path = data_dir / TRANSCRIPT_DIR / SP1500_FILE
    if not path.exists():
        logger.warning(f"SP1500 parquet not found: {path}")
        return None
    logger.info(f"Reading {SP1500_FILE}...")
    df = pd.read_parquet(path, engine="pyarrow")
    df = df.rename(columns={
        "_COL_17": "paragraph_id",
        "_COL_1":  "bb_company_id",
        "_COL_7":  "call_start",
        "_COL_9":  "year",
        "_COL_10": "quarter",
        "_COL_6":  "event_title",
        "_COL_13": "utterance_type",
        "_COL_14": "speaker_name",
        "_COL_15": "speaker_type",
        "_COL_18": "text_content",
        "_COL_19": "speaker_title",
        "_COL_20": "u3_company_number",
    })
    df["_source"] = "sp1500"
    logger.info(f"  {len(df):,} rows")
    return df


def _load_named(path: Path, source_label: str) -> pd.DataFrame | None:
    """Load ECC parquet or missing CSV (named columns) into normalised DataFrame."""
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None
    logger.info(f"Reading {path.name}...")
    if path.suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
    else:
        df = pd.read_parquet(path, engine="pyarrow")
    df = df.rename(columns=NAMED_COL_MAP)
    df["u3_company_number"] = None  # will be filled from BB→u3 map
    df["_source"] = source_label
    logger.info(f"  {len(df):,} rows")
    return df


def _normalise_row(row: pd.Series, bb_to_u3: dict[int, int]) -> dict | None:
    """Convert a normalised row to a DB record dict. Returns None if row should be skipped."""
    speaker_type = str(row["speaker_type"]) if pd.notna(row.get("speaker_type")) else ""
    if speaker_type in SKIP_SPEAKER_TYPES:
        return None

    text_content = str(row["text_content"]) if pd.notna(row.get("text_content")) else ""
    if len(text_content) < MIN_TEXT_LEN:
        return None

    pid = row.get("paragraph_id")
    if pd.isna(pid):
        return None
    pid = int(pid)

    # Resolve u3_company_number
    u3 = row.get("u3_company_number")
    if pd.isna(u3) if u3 is not None else True:
        bb = row.get("bb_company_id")
        u3 = bb_to_u3.get(int(bb)) if bb is not None and not pd.isna(bb) else None
    else:
        u3 = int(u3)

    # call_date from call_start
    call_ts = row.get("call_start")
    call_date = None
    if call_ts is not None and not (isinstance(call_ts, float) and pd.isna(call_ts)):
        try:
            call_date = pd.to_datetime(call_ts, utc=True).date()
        except Exception:
            call_date = None

    year_val = row.get("year")
    year = int(year_val) if year_val is not None and not pd.isna(year_val) else None

    quarter_val = row.get("quarter")
    quarter = int(quarter_val) if quarter_val is not None and not pd.isna(quarter_val) else None

    return {
        "paragraph_id": pid,
        "u3_company_number": u3,
        "call_date": call_date,
        "year": year,
        "quarter": quarter,
        "event_title": str(row["event_title"])[:500] if pd.notna(row.get("event_title")) else None,
        "speaker_type": speaker_type[:50] if speaker_type else None,
        "utterance_type": str(row["utterance_type"])[:80] if pd.notna(row.get("utterance_type")) else None,
        "speaker_name": str(row["speaker_name"])[:200] if pd.notna(row.get("speaker_name")) else None,
        "speaker_title": str(row["speaker_title"])[:200] if pd.notna(row.get("speaker_title")) else None,
        "text_content": text_content,
    }


def load_transcripts(
    session: Session,
    data_dir: Path,
    year_from: int | None = None,
    year_to: int | None = None,
) -> dict:
    """Incrementally embed and load transcript chunks from all three sources.

    Args:
        year_from: Only process calls from this year onwards (inclusive).
        year_to:   Only process calls up to this year (inclusive).
    """
    year_label = f" [{year_from or ''}–{year_to or ''}]" if (year_from or year_to) else ""
    logger.info(f"Loading transcripts{year_label} from all sources...")

    client = _get_gemini_client()
    bb_to_u3 = _build_bb_to_u3_map(session)
    logger.info(f"BB→u3 map: {len(bb_to_u3):,} entries")

    existing_ids = _get_existing_paragraph_ids(session)
    logger.info(f"{len(existing_ids):,} chunks already in DB")

    # Load all sources into one combined DataFrame
    frames = []
    sp1500 = _load_sp1500(data_dir)
    if sp1500 is not None:
        frames.append(sp1500)

    ecc_path = data_dir / TRANSCRIPT_DIR / ECC_FILE
    ecc = _load_named(ecc_path, "ecc")
    if ecc is not None:
        frames.append(ecc)

    csv_path = data_dir / TRANSCRIPT_DIR / MISSING_FILE
    missing = _load_named(csv_path, "missing")
    if missing is not None:
        frames.append(missing)

    # Deduplicate across sources by paragraph_id (keep first occurrence)
    combined = pd.concat(frames, ignore_index=True)
    before_dedup = len(combined)
    combined["paragraph_id"] = combined["paragraph_id"].apply(
        lambda x: int(x) if x is not None and not pd.isna(x) else None
    )
    combined = combined.dropna(subset=["paragraph_id"])
    combined = combined.drop_duplicates(subset=["paragraph_id"], keep="first")
    logger.info(
        f"Combined: {before_dedup:,} rows → {len(combined):,} after dedup across sources"
    )

    inserted = 0
    skipped_existing = 0
    skipped_short = 0
    skipped_operator = 0
    skipped_year = 0
    api_errors = 0
    total_rows = len(combined)

    for chunk_start in range(0, total_rows, READ_CHUNK_SIZE):
        chunk = combined.iloc[chunk_start: chunk_start + READ_CHUNK_SIZE]

        pending: list[dict] = []
        for _, row in chunk.iterrows():
            pid = int(row["paragraph_id"])

            if pid in existing_ids:
                skipped_existing += 1
                continue

            speaker_type = str(row.get("speaker_type", "")) if pd.notna(row.get("speaker_type")) else ""
            if speaker_type in SKIP_SPEAKER_TYPES:
                skipped_operator += 1
                continue

            text_content = str(row.get("text_content", "")) if pd.notna(row.get("text_content")) else ""
            if len(text_content) < MIN_TEXT_LEN:
                skipped_short += 1
                continue

            year_val = row.get("year")
            year = int(year_val) if year_val is not None and not pd.isna(year_val) else None
            if year_from is not None and (year is None or year < year_from):
                skipped_year += 1
                continue
            if year_to is not None and (year is None or year > year_to):
                skipped_year += 1
                continue

            rec = _normalise_row(row, bb_to_u3)
            if rec is None:
                continue
            pending.append(rec)

        if not pending:
            logger.info(f"  Chunk {chunk_start:,}–{chunk_start + len(chunk):,}: all skipped")
            continue

        # Embed in sub-batches
        for batch_start in range(0, len(pending), EMBED_BATCH_SIZE):
            batch = pending[batch_start: batch_start + EMBED_BATCH_SIZE]
            texts = [r["text_content"] for r in batch]

            try:
                vectors = _embed_batch(client, texts)
            except Exception as e:
                logger.warning(f"  Embedding error (skipping {len(batch)} rows): {e}")
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
            for r in records:
                existing_ids.add(r["paragraph_id"])

        logger.info(
            f"  Progress {chunk_start + len(chunk):,}/{total_rows:,} | "
            f"inserted={inserted:,} existed={skipped_existing:,} "
            f"operator={skipped_operator:,} short={skipped_short:,} "
            f"year-filter={skipped_year:,}"
        )

    logger.info(
        f"[OK] Done — {inserted:,} new, {skipped_existing:,} existed, "
        f"{skipped_operator:,} operator, {skipped_short:,} short, "
        f"{skipped_year:,} year-filtered, {api_errors:,} api errors"
    )
    return {
        "transcripts_new": inserted,
        "transcripts_skipped_existing": skipped_existing,
        "transcripts_skipped_operator": skipped_operator,
        "transcripts_skipped_short": skipped_short,
        "transcripts_skipped_year": skipped_year,
        "transcripts_api_errors": api_errors,
    }
