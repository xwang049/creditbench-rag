"""CLI entry point for running the benchmark experiment.

Usage examples:
    python -m src.benchmark.run_experiment --horizon 6 --lookback 12 --n-default 100 --n-control 100
    python -m src.benchmark.run_experiment --horizon 6 --n-default 10 --n-control 10 --dry-run
    python -m src.benchmark.run_experiment --horizon 6 --no-fundamentals --resume --output results/exp1
    python -m src.benchmark.evaluator results/exp1/cases.jsonl
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running as a module from the project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.benchmark.case_builder import build_cases
from src.benchmark.config import BenchmarkConfig
from src.benchmark.evaluator import evaluate, print_report
from src.benchmark.predictor import predict
from src.benchmark.prompt_builder import build_prompt
from src.db.session import get_session
from src.pipeline.data_fetcher import fetch_all


def _load_done_keys(jsonl_path: Path) -> set[tuple]:
    """Return set of (u3_company_number, cutoff_date_str) already in JSONL."""
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["u3_company_number"], rec["cutoff_date"]))
            except Exception:
                pass
    return done


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM credit default prediction benchmark"
    )
    parser.add_argument("--horizon", type=int, default=6,
                        help="Prediction horizon in months (default: 6)")
    parser.add_argument("--lookback", type=int, default=12,
                        help="Historical lookback window in months (default: 12)")
    parser.add_argument("--n-default", type=int, default=100,
                        help="Max number of default cases (Group 1)")
    parser.add_argument("--n-control", type=int, default=100,
                        help="Max number of control cases (Group 2)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI model ID")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: results/exp_h{horizon}_lb{lookback})")
    parser.add_argument("--no-fundamentals", action="store_true",
                        help="Skip DuckDB fundamentals (use when parquet files are unavailable)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip cases already in the output JSONL (for resuming interrupted runs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build cases and fetch data, but do NOT call the LLM")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for control group sampling")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Build config
    cfg = BenchmarkConfig(
        prediction_horizon_months=args.horizon,
        lookback_months=args.lookback,
        n_default_cases=args.n_default,
        n_control_cases=args.n_control,
        llm_model=args.model,
        include_fundamentals=not args.no_fundamentals,
        random_seed=args.seed,
        output_dir=args.output or f"results/exp_h{args.horizon}_lb{args.lookback}",
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "cases.jsonl"

    logger.info(f"Benchmark config: horizon={cfg.prediction_horizon_months}m, "
                f"lookback={cfg.lookback_months}m, "
                f"n_default={cfg.n_default_cases}, n_control={cfg.n_control_cases}")
    logger.info(f"Output: {jsonl_path}")

    with get_session() as session:
        # --- Step 1: Build cases ---
        logger.info("Building experiment cases...")
        cases = build_cases(session, cfg)

        if not cases:
            logger.error("No cases built — exiting")
            sys.exit(1)

        logger.info(f"Total cases: {len(cases)}")

        if args.dry_run:
            logger.info("Dry-run mode: printing case summary, no LLM calls")
            for case in cases[:5]:
                print(f"  [{case.group}] {case.company_name} ({case.ticker}) "
                      f"cutoff={case.cutoff_date} label={case.label}")
            if len(cases) > 5:
                print(f"  ... and {len(cases) - 5} more")
            print(f"\nTotal: {len(cases)} cases ({sum(c.label for c in cases)} default, "
                  f"{sum(not c.label for c in cases)} control)")
            return

        # --- Step 2: Load done keys for resume ---
        done_keys = _load_done_keys(jsonl_path) if args.resume else set()
        if done_keys:
            logger.info(f"Resuming: {len(done_keys)} cases already processed")

        # --- Step 3: Run predictions ---
        n_processed = 0
        n_skipped = 0
        n_errors = 0

        with open(jsonl_path, "a") as out_f:
            for i, case in enumerate(cases, 1):
                key = (case.u3_company_number, case.cutoff_date.isoformat())
                if key in done_keys:
                    n_skipped += 1
                    continue

                logger.info(
                    f"[{i}/{len(cases)}] {case.group:7s} | {case.company_name[:35]:35s} | "
                    f"cutoff={case.cutoff_date}"
                )

                # Fetch data
                try:
                    company = {
                        "u3_company_number": case.u3_company_number,
                        "id_bb_company": case.id_bb_company,
                        "ticker": case.ticker,
                        "company_name": case.company_name,
                        "country_name": "United States",
                        "market_status": "—",
                        "prime_exchange": "—",
                        "id_isin": None,
                        "industry_sector": case.industry_sector,
                        "industry_group": None,
                        "industry_subgroup": None,
                    }
                    data = fetch_all(
                        session, company,
                        as_of=case.cutoff_date,
                        include_fundamentals=cfg.include_fundamentals,
                        include_risk_indicators=cfg.include_risk_indicators,
                        include_market_data=cfg.include_market_data,
                        include_macro=cfg.include_macro,
                        include_transcripts=False,
                        lookback_months=cfg.lookback_months,
                    )
                except Exception as e:
                    logger.error(f"Data fetch failed for {case.company_name}: {e}")
                    n_errors += 1
                    continue

                # Build prompt
                try:
                    prompt = build_prompt(data, case, cfg)
                except Exception as e:
                    logger.error(f"Prompt build failed for {case.company_name}: {e}")
                    n_errors += 1
                    continue

                # Call LLM
                try:
                    prediction = predict(prompt, case, cfg)
                except Exception as e:
                    logger.error(f"Prediction failed for {case.company_name}: {e}")
                    prediction = {"error": str(e), "probability_pct": None}
                    n_errors += 1

                # Record
                record = {
                    "u3_company_number": case.u3_company_number,
                    "id_bb_company": case.id_bb_company,
                    "ticker": case.ticker,
                    "company_name": case.company_name,
                    "industry_sector": case.industry_sector,
                    "cutoff_date": case.cutoff_date.isoformat(),
                    "event_date": case.event_date.isoformat() if case.event_date else None,
                    "label": case.label,
                    "group": case.group,
                    "prediction": prediction,
                    "prompt_chars": len(prompt),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

                n_processed += 1

                # Rate limit protection
                time.sleep(0.5)

        logger.info(
            f"Done: {n_processed} processed, {n_skipped} skipped (resume), "
            f"{n_errors} errors"
        )

        # --- Step 4: Evaluate ---
        if n_processed > 0 or (args.resume and jsonl_path.exists()):
            logger.info("Computing evaluation metrics...")
            metrics = evaluate(jsonl_path)
            metrics["config"] = {
                "prediction_horizon_months": cfg.prediction_horizon_months,
                "lookback_months": cfg.lookback_months,
                "llm_model": cfg.llm_model,
                "run_at": datetime.utcnow().isoformat() + "Z",
            }

            print_report(metrics)

            summary_path = output_dir / "results_summary.json"
            with open(summary_path, "w") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            logger.info(f"Summary saved to {summary_path}")
        else:
            logger.warning("No new cases processed — skipping evaluation")


if __name__ == "__main__":
    main()
