"""Build the LLM prompt for a benchmark case.

Prepends a task header (cutoff_date, horizon, ground-truth suppression reminder)
to the standard context produced by context_builder.build_context().

Two contamination-prevention strategies are applied when enabled in cfg:

anonymize_company (cfg.anonymize_company=True)
    Company name and ticker are replaced with a neutral ID derived from
    u3_company_number (e.g. "Company-12345" / "U3-12345").  Sector, group, and
    country are kept because they are needed for risk assessment and are not
    individually identifying without the company name.

blur_year (cfg.blur_year=True)
    All calendar years that appear in date strings and fiscal-period labels are
    replaced with relative labels anchored to the cutoff year:
        cutoff year          → Y0
        cutoff year − 1      → Y-1
        cutoff year + 1      → Y+1   (should never appear — data is time-bounded)
        …
    Numeric *levels* (VIX, yield %, S&P500 price, revenue figures, etc.) are
    NEVER touched — only 4-digit sequences that look like a year inside a date
    context are replaced.  Specifically, the regex targets:
        YYYY-MM  / YYYY-MM-DD  (ISO dates and year-month risk-indicator labels)
        YYYYA / YYYYQ<n>       (fiscal-period suffixes used by DuckDB fundamentals)
"""

import copy
import re

from dateutil.relativedelta import relativedelta

from src.benchmark.case_builder import CaseRecord
from src.benchmark.config import BenchmarkConfig
from src.pipeline.context_builder import build_context


# ---------------------------------------------------------------------------
# Year-blurring helpers
# ---------------------------------------------------------------------------

def _build_year_map(cutoff_year: int) -> dict[int, str]:
    """Return a dict mapping calendar year → anonymised label.

    Labels:  Y0 (cutoff year), Y-1, Y-2, …  Y+1, Y+2, …
    Covers ±40 years around the cutoff so all historical fundamentals and any
    edge-case future dates (e.g. end of prediction window) are mapped.
    """
    year_map: dict[int, str] = {}
    for offset in range(-40, 5):
        actual = cutoff_year + offset
        if actual < 1900 or actual > 2100:
            continue
        if offset == 0:
            label = "Y0"
        elif offset > 0:
            label = f"Y+{offset}"
        else:
            label = f"Y{offset}"      # e.g. "Y-1", "Y-2"
        year_map[actual] = label
    return year_map


def _blur_years_in_text(text: str, year_map: dict[int, str]) -> str:
    """Replace calendar years in date / fiscal-period contexts only.

    Two patterns are replaced (in order):

    1. ISO date prefix: YYYY followed by a hyphen and two digits
       e.g.  2020-06-01  →  Y0-06-01
             2019-12     →  Y-1-12

    2. Fiscal period suffix: YYYY followed by A or Q (+ optional digit)
       e.g.  2020A   →  Y0A
             2019Q3  →  Y-1Q3

    Numeric values that happen to fall in the year range (e.g. S&P500 ≈ 2000)
    are NOT replaced because they are not followed by the date / period patterns.
    """
    def _sub(pattern: str, text: str) -> str:
        def replacer(m: re.Match) -> str:
            y = int(m.group("year"))
            return year_map.get(y, m.group("year"))
        return re.sub(pattern, replacer, text)

    # Pattern 1: ISO date prefix  YYYY-DD (2-digit month or day follows)
    text = _sub(r"(?P<year>(19|20)\d{2})(?=-\d{2})", text)

    # Pattern 2: Fiscal period    YYYYA or YYYYQ[0-9]?
    text = _sub(r"(?P<year>(19|20)\d{2})(?=[AQ]\d?(?:\b|$))", text)

    return text


# ---------------------------------------------------------------------------
# Company anonymisation helper
# ---------------------------------------------------------------------------

def _anonymize_company_dict(company: dict) -> dict:
    """Return a copy of company dict with name and ticker replaced by u3 ID."""
    anon = copy.copy(company)
    u3 = anon.get("u3_company_number", "unknown")
    anon["company_name"] = f"Company-{u3}"
    anon["ticker"] = f"U3-{u3}"
    return anon


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_prompt(
    data: dict,
    case: CaseRecord,
    cfg: BenchmarkConfig,
) -> str:
    """Build the full prompt string for a benchmark case.

    Args:
        data:  Dict returned by pipeline.data_fetcher.fetch_all()
        case:  CaseRecord (used for cutoff_date and horizon)
        cfg:   BenchmarkConfig

    Returns:
        Complete prompt text to send as the user message to the LLM.
        Anonymisation and year-blurring are applied when the corresponding
        cfg flags are True (both default to True).
    """
    # --- Apply anonymisation to data dict (before context is built) ----------
    effective_data = data
    if cfg.anonymize_company:
        effective_data = dict(data)                    # shallow copy of top level
        effective_data["company"] = _anonymize_company_dict(data["company"])

    # --- Build cutoff and end-date strings ------------------------------------
    cutoff_str = case.cutoff_date.isoformat()          # e.g. "2020-06-01"
    end_date = case.cutoff_date + relativedelta(months=cfg.prediction_horizon_months)
    end_str = end_date.isoformat()                     # e.g. "2020-12-01"

    # --- Task header ----------------------------------------------------------
    task_header = (
        f"=== PREDICTION TASK ===\n"
        f"Data cutoff date : {cutoff_str}\n"
        f"Prediction window: {cfg.prediction_horizon_months} months "
        f"({cutoff_str} → {end_str})\n"
        f"\n"
        f"Predict whether this company will experience a significant credit event\n"
        f"(default, bankruptcy filing, or equivalent) within the prediction window.\n"
        f"\n"
        f"IMPORTANT: You must NOT use any information about events that occurred\n"
        f"after {cutoff_str}. Base your assessment only on the data below.\n"
        f"{'=' * 55}\n\n"
    )

    # --- Structured company context -------------------------------------------
    company_context = build_context(effective_data)

    # --- Assemble full prompt --------------------------------------------------
    full_prompt = (
        "Analyse the following company and predict its credit event risk.\n\n"
        "---\n"
        f"{task_header}"
        f"{company_context}\n"
        "---\n\n"
        "Return your assessment as JSON."
    )

    # --- Apply year blurring (post-processing on the assembled text) ----------
    if cfg.blur_year:
        year_map = _build_year_map(case.cutoff_date.year)
        full_prompt = _blur_years_in_text(full_prompt, year_map)

    return full_prompt
