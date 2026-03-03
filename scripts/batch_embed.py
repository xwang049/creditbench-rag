"""OpenAI Batch API embedding pipeline for transcript chunks.

4-phase workflow:
    python scripts/batch_embed.py prepare [--data-dir PATH] [--year-from N] [--year-to N] [--out-dir PATH] [--force]
    python scripts/batch_embed.py submit  [--out-dir PATH]
    python scripts/batch_embed.py poll    [--out-dir PATH] [--wait]
    python scripts/batch_embed.py ingest  [--out-dir PATH] [--batch-size 5000]

Phase 1 (prepare): Read parquet sources, filter, write JSONL files (≤50k rows each) + metadata sidecars.
Phase 2 (submit):  Upload JSONLs to OpenAI, create batch jobs.
Phase 3 (poll):    Check batch statuses. --wait loops until all done.
Phase 4 (ingest):  Download results from OpenAI, parse embeddings, INSERT into PostgreSQL.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from openai import OpenAI
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

from src.config import config
from src.db.models import TranscriptChunk
from src.db.session import get_session
from src.ingestion.load_transcripts import (
    TRANSCRIPT_DIR,
    SP1500_FILE,
    ECC_FILE,
    MISSING_FILE,
    MIN_TEXT_LEN,
    SKIP_SPEAKER_TYPES,
    _load_sp1500,
    _load_named,
    _normalise_row,
    _build_bb_to_u3_map,
    _get_existing_paragraph_ids,
)

logger = logging.getLogger(__name__)

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536
MAX_BATCH_ROWS = 50_000  # OpenAI limit per batch file
DEFAULT_OUT_DIR = "./batch_embed_output"

TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


# ── Manifest helpers ─────────────────────────────────────────────────────────

def _load_manifest(out_dir: Path) -> dict:
    path = out_dir / "manifest.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBED_MODEL,
        "embedding_dims": EMBED_DIMS,
        "total_paragraphs": 0,
        "prepare": None,
        "batches": [],
    }


def _save_manifest(out_dir: Path, manifest: dict):
    """Atomic write: write to .tmp then rename."""
    path = out_dir / "manifest.json"
    tmp = out_dir / "manifest.json.tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY") or config.OPENAI_API_KEY
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in environment / .env")
    return OpenAI(api_key=key)


# ── Phase 1: PREPARE ─────────────────────────────────────────────────────────

def cmd_prepare(args):
    """Read parquets, filter, generate JSONL + metadata sidecar files."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(out_dir)

    # Check if already prepared (unless --force)
    if manifest.get("prepare") and manifest["prepare"].get("completed_at"):
        if not args.force:
            n_files = len(manifest["prepare"]["jsonl_files"])
            logger.info(
                f"Already prepared ({n_files} files). Use --force to re-generate."
            )
            return
        logger.info("--force: re-generating all JSONL files")
        # Clean up old files
        for entry in manifest["prepare"].get("jsonl_files", []):
            for key in ("filename", "meta_file"):
                p = out_dir / entry.get(key, "")
                if p.exists():
                    p.unlink()
        manifest["prepare"] = None
        manifest["batches"] = []

    data_dir = Path(args.data_dir) if args.data_dir else Path(config.DATA_DIR)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    logger.info(f"Data source: {data_dir}")
    logger.info(f"Output dir:  {out_dir}")

    # Load sources
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

    if not frames:
        logger.error("No data sources found")
        sys.exit(1)

    # Combine and deduplicate
    combined = pd.concat(frames, ignore_index=True)
    combined["paragraph_id"] = combined["paragraph_id"].apply(
        lambda x: int(x) if x is not None and not pd.isna(x) else None
    )
    combined = combined.dropna(subset=["paragraph_id"])
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["paragraph_id"], keep="first")
    logger.info(f"Combined: {before_dedup:,} → {len(combined):,} after dedup")

    # Get existing paragraph_ids from DB (for incremental)
    with get_session() as session:
        bb_to_u3 = _build_bb_to_u3_map(session)
        existing_ids = _get_existing_paragraph_ids(session)
        # Only exclude paragraphs that already have embeddings
        if existing_ids:
            rows_with_emb = session.execute(
                text("SELECT paragraph_id FROM transcript_chunks WHERE embedding IS NOT NULL")
            ).fetchall()
            embedded_ids = {r[0] for r in rows_with_emb}
        else:
            embedded_ids = set()
    logger.info(f"BB→u3 map: {len(bb_to_u3):,} | Already embedded: {len(embedded_ids):,}")

    # Filter and collect pending rows
    pending = []
    stats = {"operator": 0, "short": 0, "year": 0, "embedded": 0}

    for _, row in combined.iterrows():
        pid = int(row["paragraph_id"])

        if pid in embedded_ids:
            stats["embedded"] += 1
            continue

        speaker_type = str(row.get("speaker_type", "")) if pd.notna(row.get("speaker_type")) else ""
        if speaker_type in SKIP_SPEAKER_TYPES:
            stats["operator"] += 1
            continue

        text_content = str(row.get("text_content", "")) if pd.notna(row.get("text_content")) else ""
        if len(text_content) < MIN_TEXT_LEN:
            stats["short"] += 1
            continue

        year_val = row.get("year")
        year = int(year_val) if year_val is not None and not pd.isna(year_val) else None
        if args.year_from is not None and (year is None or year < args.year_from):
            stats["year"] += 1
            continue
        if args.year_to is not None and (year is None or year > args.year_to):
            stats["year"] += 1
            continue

        rec = _normalise_row(row, bb_to_u3)
        if rec is None:
            continue
        pending.append(rec)

    logger.info(
        f"Pending: {len(pending):,} | Skipped: "
        f"embedded={stats['embedded']:,} operator={stats['operator']:,} "
        f"short={stats['short']:,} year={stats['year']:,}"
    )

    if not pending:
        logger.info("Nothing to embed. All rows already processed or filtered out.")
        return

    # Write JSONL + metadata sidecar files
    jsonl_entries = []
    batch_idx = 0

    for start in range(0, len(pending), MAX_BATCH_ROWS):
        chunk = pending[start: start + MAX_BATCH_ROWS]

        jsonl_name = f"batch_{batch_idx:03d}.jsonl"
        meta_name = f"meta_{batch_idx:03d}.jsonl"

        with open(out_dir / jsonl_name, "w") as jf, open(out_dir / meta_name, "w") as mf:
            for rec in chunk:
                # OpenAI Batch API request line
                request = {
                    "custom_id": f"pid_{rec['paragraph_id']}",
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": EMBED_MODEL,
                        "input": rec["text_content"],
                    },
                }
                jf.write(json.dumps(request, ensure_ascii=False) + "\n")

                # Metadata sidecar line (everything except text_content to save space,
                # but we do include text_content because it's needed for the DB record)
                meta = {k: v for k, v in rec.items()}
                # Convert date objects to strings for JSON
                if meta.get("call_date") and isinstance(meta["call_date"], date):
                    meta["call_date"] = meta["call_date"].isoformat()
                mf.write(json.dumps(meta, ensure_ascii=False) + "\n")

        jsonl_entries.append({
            "filename": jsonl_name,
            "meta_file": meta_name,
            "num_requests": len(chunk),
        })
        logger.info(f"  Wrote {jsonl_name}: {len(chunk):,} requests")
        batch_idx += 1

    manifest["total_paragraphs"] = len(pending)
    manifest["prepare"] = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "year_from": args.year_from,
        "year_to": args.year_to,
        "jsonl_files": jsonl_entries,
    }
    _save_manifest(out_dir, manifest)

    logger.info(
        f"[OK] Prepared {len(pending):,} paragraphs → {batch_idx} batch files"
    )


# ── Phase 2: SUBMIT ──────────────────────────────────────────────────────────

def cmd_submit(args):
    """Upload JSONL files to OpenAI and create batch jobs."""
    out_dir = Path(args.out_dir)
    manifest = _load_manifest(out_dir)

    if not manifest.get("prepare") or not manifest["prepare"].get("completed_at"):
        logger.error("No prepared data found. Run `prepare` first.")
        sys.exit(1)

    client = _get_openai_client()
    jsonl_files = manifest["prepare"]["jsonl_files"]

    # Build lookup: jsonl_file -> batch entry
    batch_lookup = {b["jsonl_file"]: b for b in manifest.get("batches", [])}

    n_submitted = 0
    n_skipped = 0

    for entry in jsonl_files:
        fname = entry["filename"]

        # Skip if already submitted
        existing = batch_lookup.get(fname)
        if existing and existing.get("openai_batch_id"):
            n_skipped += 1
            continue

        jsonl_path = out_dir / fname
        if not jsonl_path.exists():
            logger.warning(f"JSONL file missing: {fname}, skipping")
            continue

        logger.info(f"Submitting {fname} ({entry['num_requests']:,} requests)...")

        try:
            # Upload file
            with open(jsonl_path, "rb") as f:
                file_obj = client.files.create(file=f, purpose="batch")
            file_id = file_obj.id
            logger.info(f"  Uploaded → {file_id}")

            # Create batch
            batch_obj = client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/embeddings",
                completion_window="24h",
            )
            batch_id = batch_obj.id
            logger.info(f"  Batch created → {batch_id}")

            batch_entry = {
                "jsonl_file": fname,
                "openai_file_id": file_id,
                "openai_batch_id": batch_id,
                "status": batch_obj.status,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": None,
                "result_file_id": None,
                "ingested": False,
                "error_count": 0,
            }

            # Update or add in manifest
            if fname in batch_lookup:
                idx = next(i for i, b in enumerate(manifest["batches"]) if b["jsonl_file"] == fname)
                manifest["batches"][idx] = batch_entry
            else:
                manifest["batches"].append(batch_entry)
                batch_lookup[fname] = batch_entry

            _save_manifest(out_dir, manifest)
            n_submitted += 1

        except Exception as e:
            logger.error(f"  Failed to submit {fname}: {e}")
            # Save partial progress
            _save_manifest(out_dir, manifest)
            continue

    logger.info(f"[OK] Submitted: {n_submitted}, Skipped (already submitted): {n_skipped}")


# ── Phase 3: POLL ────────────────────────────────────────────────────────────

def cmd_poll(args):
    """Check batch statuses. With --wait, loop until all complete."""
    out_dir = Path(args.out_dir)
    manifest = _load_manifest(out_dir)

    if not manifest.get("batches"):
        logger.error("No batches found. Run `submit` first.")
        sys.exit(1)

    client = _get_openai_client()
    poll_interval = 60  # seconds
    max_interval = 300

    while True:
        counts = {"completed": 0, "failed": 0, "in_progress": 0, "other": 0}

        for batch_entry in manifest["batches"]:
            status = batch_entry.get("status", "unknown")

            # Skip terminal statuses
            if status in TERMINAL_STATUSES:
                counts[status if status in counts else "other"] += 1
                continue

            batch_id = batch_entry.get("openai_batch_id")
            if not batch_id:
                counts["other"] += 1
                continue

            try:
                batch_obj = client.batches.retrieve(batch_id)
                batch_entry["status"] = batch_obj.status

                if batch_obj.status == "completed":
                    batch_entry["completed_at"] = datetime.now(timezone.utc).isoformat()
                    batch_entry["result_file_id"] = batch_obj.output_file_id
                    if batch_obj.request_counts:
                        batch_entry["error_count"] = batch_obj.request_counts.failed or 0
                    counts["completed"] += 1
                    logger.info(f"  ✓ {batch_entry['jsonl_file']} completed")
                elif batch_obj.status == "failed":
                    batch_entry["completed_at"] = datetime.now(timezone.utc).isoformat()
                    errors = []
                    if batch_obj.errors and batch_obj.errors.data:
                        errors = [e.message for e in batch_obj.errors.data[:3]]
                    logger.warning(
                        f"  ✗ {batch_entry['jsonl_file']} FAILED: {errors}"
                    )
                    counts["failed"] += 1
                elif batch_obj.status in ("expired", "cancelled"):
                    batch_entry["completed_at"] = datetime.now(timezone.utc).isoformat()
                    counts["other"] += 1
                else:
                    counts["in_progress"] += 1

            except Exception as e:
                logger.warning(f"  Error polling {batch_id}: {e}")
                counts["other"] += 1

        _save_manifest(out_dir, manifest)

        total = len(manifest["batches"])
        logger.info(
            f"Status: {counts['completed']}/{total} completed, "
            f"{counts['in_progress']} in progress, "
            f"{counts['failed']} failed, "
            f"{counts['other']} other"
        )

        # Check if all done
        all_terminal = all(
            b.get("status", "") in TERMINAL_STATUSES
            for b in manifest["batches"]
        )

        if all_terminal or not args.wait:
            break

        logger.info(f"Waiting {poll_interval}s before next poll...")
        time.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.5, max_interval)

    if all(b.get("status") in TERMINAL_STATUSES for b in manifest["batches"]):
        logger.info("[OK] All batches reached terminal status")
    else:
        logger.info("Some batches still in progress. Run `poll --wait` to keep checking.")


# ── Phase 4: INGEST ──────────────────────────────────────────────────────────

def cmd_ingest(args):
    """Download results from OpenAI and INSERT embeddings into PostgreSQL."""
    out_dir = Path(args.out_dir)
    manifest = _load_manifest(out_dir)
    batch_size = args.batch_size

    if not manifest.get("batches"):
        logger.error("No batches found. Run `submit` and `poll` first.")
        sys.exit(1)

    client = _get_openai_client()

    # Find batches ready for ingest
    ready = [
        b for b in manifest["batches"]
        if b.get("status") == "completed"
        and not b.get("ingested")
        and b.get("result_file_id")
    ]

    if not ready:
        already_done = sum(1 for b in manifest["batches"] if b.get("ingested"))
        logger.info(
            f"No batches ready for ingest. "
            f"({already_done} already ingested, "
            f"{len(manifest['batches']) - already_done} pending/failed)"
        )
        return

    logger.info(f"{len(ready)} batches ready for ingest")

    total_inserted = 0
    total_errors = 0

    with get_session() as session:
        for batch_entry in ready:
            fname = batch_entry["jsonl_file"]
            result_file_id = batch_entry["result_file_id"]

            # Find corresponding meta file
            meta_file = None
            for jf in manifest["prepare"]["jsonl_files"]:
                if jf["filename"] == fname:
                    meta_file = jf.get("meta_file")
                    break

            if not meta_file:
                logger.warning(f"No meta file for {fname}, skipping")
                continue

            meta_path = out_dir / meta_file
            if not meta_path.exists():
                logger.warning(f"Meta file missing: {meta_file}, skipping")
                continue

            logger.info(f"Ingesting {fname} (result: {result_file_id})...")

            # Load metadata sidecar
            meta_lookup: dict[int, dict] = {}
            with open(meta_path) as mf:
                for line in mf:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    meta_lookup[rec["paragraph_id"]] = rec

            # Download result from OpenAI (streaming)
            try:
                result_content = client.files.content(result_file_id)
                result_text = result_content.text
            except Exception as e:
                logger.error(f"  Failed to download result for {fname}: {e}")
                total_errors += 1
                continue

            # Parse results and batch insert
            records = []
            batch_errors = 0
            now = datetime.now(timezone.utc)

            for line in result_text.strip().split("\n"):
                if not line.strip():
                    continue

                try:
                    result_obj = json.loads(line)
                except json.JSONDecodeError:
                    batch_errors += 1
                    continue

                custom_id = result_obj.get("custom_id", "")
                response = result_obj.get("response", {})

                # Check for errors
                if response.get("status_code") != 200:
                    batch_errors += 1
                    error_body = response.get("body", {}).get("error", {})
                    logger.debug(
                        f"  Error for {custom_id}: {error_body.get('message', 'unknown')}"
                    )
                    continue

                # Extract paragraph_id from custom_id: "pid_12345"
                try:
                    pid = int(custom_id.split("_", 1)[1])
                except (IndexError, ValueError):
                    logger.warning(f"  Bad custom_id: {custom_id}")
                    batch_errors += 1
                    continue

                # Extract embedding
                body = response.get("body", {})
                data_list = body.get("data", [])
                if not data_list:
                    batch_errors += 1
                    continue
                embedding = data_list[0].get("embedding")
                if not embedding:
                    batch_errors += 1
                    continue

                # Get metadata
                meta = meta_lookup.get(pid)
                if not meta:
                    logger.debug(f"  No metadata for pid={pid}, inserting with embedding only")
                    meta = {"paragraph_id": pid}

                # Build DB record
                record = {
                    "paragraph_id": pid,
                    "u3_company_number": meta.get("u3_company_number"),
                    "call_date": meta.get("call_date"),
                    "year": meta.get("year"),
                    "quarter": meta.get("quarter"),
                    "event_title": meta.get("event_title"),
                    "speaker_type": meta.get("speaker_type"),
                    "utterance_type": meta.get("utterance_type"),
                    "speaker_name": meta.get("speaker_name"),
                    "speaker_title": meta.get("speaker_title"),
                    "text_content": meta.get("text_content", ""),
                    "embedding": embedding,
                    "embedding_model": EMBED_MODEL,
                    "embedded_at": now,
                }
                records.append(record)

                # Flush in batches
                if len(records) >= batch_size:
                    _upsert_batch(session, records)
                    total_inserted += len(records)
                    records = []

            # Flush remaining
            if records:
                _upsert_batch(session, records)
                total_inserted += len(records)

            batch_entry["ingested"] = True
            batch_entry["error_count"] = batch_errors
            total_errors += batch_errors
            _save_manifest(out_dir, manifest)

            logger.info(
                f"  ✓ {fname}: ingested, errors={batch_errors}"
            )

    logger.info(
        f"[OK] Ingest complete: {total_inserted:,} rows inserted/updated, "
        f"{total_errors:,} errors"
    )


def _upsert_batch(session, records: list[dict]):
    """Batch upsert: INSERT ... ON CONFLICT DO UPDATE for embedding columns."""
    if not records:
        return
    stmt = insert(TranscriptChunk).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["paragraph_id"],
        set_={
            "embedding": stmt.excluded.embedding,
            "embedding_model": stmt.excluded.embedding_model,
            "embedded_at": stmt.excluded.embedded_at,
            # Also update metadata in case it was missing before
            "u3_company_number": stmt.excluded.u3_company_number,
            "call_date": stmt.excluded.call_date,
            "year": stmt.excluded.year,
            "quarter": stmt.excluded.quarter,
            "event_title": stmt.excluded.event_title,
            "speaker_type": stmt.excluded.speaker_type,
            "utterance_type": stmt.excluded.utterance_type,
            "speaker_name": stmt.excluded.speaker_name,
            "speaker_title": stmt.excluded.speaker_title,
            "text_content": stmt.excluded.text_content,
        },
    )
    session.execute(stmt)
    session.commit()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OpenAI Batch API embedding pipeline for transcript chunks"
    )
    sub = parser.add_subparsers(dest="command", help="Sub-command")

    # prepare
    p_prepare = sub.add_parser("prepare", help="Generate JSONL files from parquet sources")
    p_prepare.add_argument("--data-dir", type=str, default=None, help="Override DATA_DIR")
    p_prepare.add_argument("--year-from", type=int, default=None, help="Filter: year >= N")
    p_prepare.add_argument("--year-to", type=int, default=None, help="Filter: year <= N")
    p_prepare.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory")
    p_prepare.add_argument("--force", action="store_true", help="Re-generate even if already prepared")

    # submit
    p_submit = sub.add_parser("submit", help="Upload JSONLs and create OpenAI batch jobs")
    p_submit.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)

    # poll
    p_poll = sub.add_parser("poll", help="Check batch statuses")
    p_poll.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p_poll.add_argument("--wait", action="store_true", help="Loop until all batches complete")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Download results and INSERT into PostgreSQL")
    p_ingest.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p_ingest.add_argument("--batch-size", type=int, default=5000, help="DB insert batch size")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "poll":
        cmd_poll(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
