"""Format fetched company data into a structured text context for LLM prediction."""


def _fmt(val, decimals=1, suffix=""):
    """Format a numeric value, returning '—' for None."""
    if val is None:
        return "—"
    try:
        v = float(val)
        if abs(v) >= 1e9:
            return f"{v/1e9:,.{decimals}f}B{suffix}"
        if abs(v) >= 1e6:
            return f"{v/1e6:,.{decimals}f}M{suffix}"
        return f"{v:,.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(val)


def _section(title: str, body: str) -> str:
    return f"## {title}\n{body}\n"


def _company_profile(company: dict) -> str:
    lines = [
        f"Name:     {company['company_name']}",
        f"Ticker:   {company['ticker']}",
        f"Status:   {company['market_status']}",
        f"Exchange: {company['prime_exchange']}",
        f"Country:  {company['country_name']}",
        f"Sector:   {company.get('industry_sector') or '—'}",
        f"Group:    {company.get('industry_group') or '—'}",
        f"Subgroup: {company.get('industry_subgroup') or '—'}",
    ]
    return _section("Company Profile", "\n".join(lines))


def _income_statement(records: list[dict]) -> str:
    if not records:
        return _section("Income Statement", "(no data)")
    header = f"{'Period':<12} {'Revenue':>12} {'EBIT':>12} {'EBITDA':>12} {'NetIncome':>12} {'IntExp':>12} {'EPS':>8}"
    rows = [header, "-" * len(header)]
    for r in records:
        rows.append(
            f"{(r.get('FISCAL_YEAR_PERIOD') or '?'):<12} "
            f"{_fmt(r.get('SALES_REV_TURN')):>12} "
            f"{_fmt(r.get('EBIT')):>12} "
            f"{_fmt(r.get('EBITDA')):>12} "
            f"{_fmt(r.get('NET_INCOME')):>12} "
            f"{_fmt(r.get('IS_INT_EXPENSE') or r.get('TOT_INT_EXP')):>12} "
            f"{_fmt(r.get('IS_DILUTED_EPS'), 2):>8}"
        )
    return _section("Income Statement (Annual)", "\n".join(rows))


def _balance_sheet(records: list[dict]) -> str:
    if not records:
        return _section("Balance Sheet", "(no data)")
    header = f"{'Period':<12} {'TotalAsset':>12} {'Cash':>12} {'CurAsset':>12} {'CurLiab':>12} {'STDebt':>12} {'LTDebt':>12}"
    rows = [header, "-" * len(header)]
    for r in records:
        rows.append(
            f"{(r.get('FISCAL_YEAR_PERIOD') or '?'):<12} "
            f"{_fmt(r.get('BS_TOT_ASSET')):>12} "
            f"{_fmt(r.get('BS_CASH_NEAR_CASH_ITEM')):>12} "
            f"{_fmt(r.get('BS_CUR_ASSET_REPORT')):>12} "
            f"{_fmt(r.get('BS_CUR_LIAB')):>12} "
            f"{_fmt(r.get('BS_ST_BORROW')):>12} "
            f"{_fmt(r.get('BS_LT_BORROW')):>12}"
        )
    return _section("Balance Sheet (Annual)", "\n".join(rows))


def _cash_flow(records: list[dict]) -> str:
    if not records:
        return _section("Cash Flow", "(no data)")
    header = f"{'Period':<12} {'OperCF':>12} {'Capex':>12} {'InvestCF':>12} {'FinCF':>12} {'IntPaid':>12}"
    rows = [header, "-" * len(header)]
    for r in records:
        rows.append(
            f"{(r.get('FISCAL_YEAR_PERIOD') or '?'):<12} "
            f"{_fmt(r.get('CF_CASH_FROM_OPER')):>12} "
            f"{_fmt(r.get('CF_CAP_EXPEND_INC_FIX_ASSET')):>12} "
            f"{_fmt(r.get('CF_CASH_FROM_INV_ACT')):>12} "
            f"{_fmt(r.get('CF_CASH_FROM_FNC_ACT')):>12} "
            f"{_fmt(r.get('CF_ACT_CASH_PAID_FOR_INT_DEBT')):>12}"
        )
    return _section("Cash Flow (Annual)", "\n".join(rows))


def _risk_indicators(records: list[dict]) -> str:
    if not records:
        return _section("Risk Indicators (Merton Model)", "(no data — coverage ends 2003)")
    # Show a summary: first (most recent) and trend
    latest = records[0]
    header = f"{'Year-Mo':<10} {'DTD':>8} {'Sigma':>8} {'M/B':>8} {'NI/TA':>8} {'Size':>8} {'LiqR':>8}"
    rows = [header, "-" * len(header)]
    for r in records[:12]:  # last 12 months
        rows.append(
            f"{r['year']}-{r['month']:02d}   "
            f"{_fmt(r.get('dtd'), 2):>8} "
            f"{_fmt(r.get('sigma'), 3):>8} "
            f"{_fmt(r.get('m2b'), 2):>8} "
            f"{_fmt(r.get('ni2ta'), 3):>8} "
            f"{_fmt(r.get('size'), 2):>8} "
            f"{_fmt(r.get('liquidity_r'), 3):>8}"
        )
    return _section("Risk Indicators (Merton Model, monthly)", "\n".join(rows))


def _market_data(records: list[dict]) -> str:
    if not records:
        return _section("Market Data", "(no data)")
    # Summarise: latest, 1m ago, 3m ago, 6m ago, 1y ago
    latest = records[0]
    lines = [f"Latest ({latest['date']}): price={_fmt(latest.get('price'), 2)}, "
             f"mktcap={_fmt(latest.get('marketcap'))}, vol={_fmt(latest.get('volume'))}"]

    checkpoints = {"1m": 21, "3m": 63, "6m": 126, "1y": 252}
    for label, offset in checkpoints.items():
        if offset < len(records):
            r = records[offset]
            chg = None
            if latest.get("price") and r.get("price") and r["price"]:
                chg = (latest["price"] / r["price"] - 1) * 100
            chg_str = f" ({chg:+.1f}%)" if chg is not None else ""
            lines.append(
                f"  {label} ago ({r['date']}): price={_fmt(r.get('price'), 2)}{chg_str}, "
                f"mktcap={_fmt(r.get('marketcap'))}"
            )

    return _section("Market Data", "\n".join(lines))


def _credit_events(records: list[dict]) -> str:
    if not records:
        return _section("Historical Credit Events", "None on record.")
    rows = []
    for r in records:
        date = r.get("announcement_date") or r.get("effective_date") or "?"
        action = r.get("action_name") or f"type={r.get('event_type')}"
        sub = r.get("subcategory") or ""
        rows.append(f"  {date}  {action}  {sub}")
    return _section("Historical Credit Events", "\n".join(rows))


def _macro_context(macro: dict) -> str:
    if not macro:
        return _section("Macro Environment", "(no data)")
    lines = []
    us = macro.get("us")
    if us:
        lines.append(
            f"As of {us.get('date')}: S&P500={_fmt(us.get('sp500'), 0)}, "
            f"VIX={_fmt(us.get('vix'), 1)}, Unemp={_fmt(us.get('unemployment'), 1)}%, "
            f"CPI={_fmt(us.get('cpi'), 1)}"
        )
    by = macro.get("bond_yields")
    if by:
        lines.append(
            f"Treasury yields ({by.get('date')}): "
            f"2Y={_fmt(by.get('us_2y'), 2)}%, 5Y={_fmt(by.get('us_5y'), 2)}%, "
            f"10Y={_fmt(by.get('us_10y'), 2)}%, 30Y={_fmt(by.get('us_30y'), 2)}%"
        )
    return _section("Macro Environment", "\n".join(lines) if lines else "(no data)")


def _transcripts(records: list[dict]) -> str:
    if not records:
        return _section("Earnings Call Excerpts", "(no transcript data available)")
    chunks = []
    for r in records:
        meta = f"[{r.get('call_date') or '?'} Q{r.get('quarter') or '?'} — {r.get('speaker_name') or '?'} ({r.get('speaker_type') or '?'})]"
        chunks.append(f"{meta}\n{r['text_content']}")
    return _section("Earnings Call Excerpts", "\n\n".join(chunks))


def build_context(data: dict) -> str:
    """Build a structured text context from fetched data for the LLM.

    Args:
        data: Dict returned by data_fetcher.fetch_all()

    Returns:
        Multi-section formatted text ready for LLM consumption.
    """
    fundamentals = data.get("fundamentals", {})

    sections = [
        _company_profile(data["company"]),
        _income_statement(fundamentals.get("income_statement", [])),
        _balance_sheet(fundamentals.get("balance_sheet", [])),
        _cash_flow(fundamentals.get("cash_flow", [])),
        _risk_indicators(data.get("risk_indicators", [])),
        _market_data(data.get("market_data", [])),
        _credit_events(data.get("credit_events", [])),
        _macro_context(data.get("macro")),
        _transcripts(data.get("transcripts", [])),
    ]

    return "\n".join(sections)
