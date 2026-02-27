"""CLI entry point for end-to-end credit event prediction.

Usage:
    python scripts/predict_credit_risk.py AAPL
    python scripts/predict_credit_risk.py "AAPL US"
    python scripts/predict_credit_risk.py US0378331005      # ISIN
    python scripts/predict_credit_risk.py AAPL --model claude-sonnet-4-20250514
    python scripts/predict_credit_risk.py AAPL --context-only   # print context, skip LLM
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.session import get_session
from src.pipeline.company_lookup import lookup, CompanyNotFoundError
from src.pipeline.data_fetcher import fetch_all
from src.pipeline.context_builder import build_context
from src.pipeline.predictor import predict


def main():
    parser = argparse.ArgumentParser(description="Credit event prediction for a company")
    parser.add_argument("identifier", help="Ticker (e.g. AAPL), Bloomberg ticker (AAPL US), or ISIN")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model ID")
    parser.add_argument("--context-only", action="store_true",
                        help="Print the data context without calling the LLM")
    parser.add_argument("--json", action="store_true", help="Output prediction as raw JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    with get_session() as session:
        # Step 1: Resolve company
        try:
            company = lookup(session, args.identifier)
        except CompanyNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"Company: {company['company_name']} ({company['ticker']})")
        print(f"  Sector: {company.get('industry_sector') or '—'} / {company.get('industry_subgroup') or '—'}")
        print(f"  Status: {company['market_status']}")
        print()

        # Step 2: Fetch data
        data = fetch_all(session, company)

        # Step 3: Build context
        context = build_context(data)

        if args.context_only:
            print(context)
            return

        print(f"Context: {len(context):,} chars")
        print()

        # Step 4: LLM prediction
        result = predict(context, model=args.model)

        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return

        # Pretty-print
        if "error" in result:
            print(f"ERROR: {result['error']}")
            if "raw_response" in result:
                print(result["raw_response"])
            return

        pred = result.get("prediction", "?")
        conf = result.get("confidence", "?")
        prob = result.get("probability_pct", "?")
        print(f"{'='*60}")
        print(f"  CREDIT EVENT RISK:  {pred}")
        print(f"  Probability:        {prob}%")
        print(f"  Confidence:         {conf}/100")
        print(f"{'='*60}")
        print()

        risks = result.get("key_risk_factors", [])
        if risks:
            print("Key Risk Factors:")
            for r in risks:
                print(f"  - {r}")
            print()

        protective = result.get("protective_factors", [])
        if protective:
            print("Protective Factors:")
            for p in protective:
                print(f"  + {p}")
            print()

        reasoning = result.get("reasoning", "")
        if reasoning:
            print("Analysis:")
            print(reasoning)
            print()

        gaps = result.get("data_gaps", [])
        if gaps:
            print("Data Gaps:")
            for g in gaps:
                print(f"  ! {g}")
            print()

        usage = result.get("_usage", {})
        if usage:
            print(f"[Model: {result.get('_model', '?')}, "
                  f"tokens: {usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out]")


if __name__ == "__main__":
    main()
