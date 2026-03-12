"""Layer 1: Feature engineering — compute credit-relevant ratios from raw fundamentals.

Produces a multi-year signals table (Y0 = most recent, Y-1, Y-2, …) plus
trend arrows and risk warnings.  Structured data does the arithmetic so
the LLM can focus on interpretation rather than calculation.

Trend convention:
  ↑↑ = improved >15%   ↑ = improved 5-15%   → = flat ±5%
  ↓  = worsened 5-15%  ↓↓ = worsened >15%   n/a = insufficient data
"""

from __future__ import annotations

from typing import Optional


# ── Arithmetic helpers ────────────────────────────────────────────────────────

def _f(val) -> Optional[float]:
    if val is None or val == "N.A.":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _div(a, b) -> Optional[float]:
    a, b = _f(a), _f(b)
    if a is None or b is None or b == 0:
        return None
    return a / b


def _add(*args) -> Optional[float]:
    vals = [_f(v) for v in args]
    if all(v is None for v in vals):
        return None
    return sum(v for v in vals if v is not None)


def _sub(a, b) -> Optional[float]:
    a, b = _f(a), _f(b)
    if a is None or b is None:
        return None
    return a - b


def _trend(recent: Optional[float], old: Optional[float]) -> str:
    """Return trend arrow comparing recent vs old value."""
    if recent is None or old is None or old == 0:
        return "n/a"
    chg = (recent - old) / abs(old)
    if chg > 0.15:
        return "↑↑"
    if chg > 0.05:
        return "↑"
    if chg < -0.15:
        return "↓↓"
    if chg < -0.05:
        return "↓"
    return "→"


def _altman_z(is_r: dict, bs_r: dict) -> Optional[float]:
    """Altman Z-Score (public company, 1968 formula).

    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    X1 = Working Capital / Total Assets
    X2 = Net Income / Total Assets  (retained earnings proxy)
    X3 = EBIT / Total Assets
    X4 = Book Equity / Total Liabilities
    X5 = Revenue / Total Assets
    """
    assets   = _f(bs_r.get("BS_TOT_ASSET"))
    cur_a    = _f(bs_r.get("BS_CUR_ASSET_REPORT"))
    cur_l    = _f(bs_r.get("BS_CUR_LIAB"))
    ni       = _f(is_r.get("NET_INCOME"))
    ebit     = _f(is_r.get("EBIT"))
    rev      = _f(is_r.get("SALES_REV_TURN"))
    st_debt  = _f(bs_r.get("BS_ST_BORROW"))
    lt_debt  = _f(bs_r.get("BS_LT_BORROW"))

    if any(v is None for v in [assets, cur_a, cur_l, ni, ebit, rev]) or assets == 0:
        return None

    total_debt  = (st_debt or 0) + (lt_debt or 0)
    book_equity = max(assets - total_debt, 0)
    total_liab  = max(total_debt, assets * 0.01)   # avoid div-by-zero

    x1 = (cur_a - cur_l) / assets
    x2 = ni / assets
    x3 = ebit / assets
    x4 = book_equity / total_liab
    x5 = rev / assets

    return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5


def _altman_zone(z: Optional[float]) -> Optional[str]:
    if z is None:
        return None
    if z > 2.99:
        return "safe"
    if z >= 1.81:
        return "grey"
    return "distress"


# ── Per-period signal computation ─────────────────────────────────────────────

def _period_signals(is_r: dict, bs_r: dict, cf_r: dict) -> dict:
    """Compute all signals for a single annual period."""
    revenue   = _f(is_r.get("SALES_REV_TURN"))
    ebit      = _f(is_r.get("EBIT"))
    ebitda    = _f(is_r.get("EBITDA"))
    ni        = _f(is_r.get("NET_INCOME"))
    int_exp   = _f(is_r.get("IS_INT_EXPENSE")) or _f(is_r.get("TOT_INT_EXP"))

    assets    = _f(bs_r.get("BS_TOT_ASSET"))
    cash      = _f(bs_r.get("BS_CASH_NEAR_CASH_ITEM"))
    cur_a     = _f(bs_r.get("BS_CUR_ASSET_REPORT"))
    cur_l     = _f(bs_r.get("BS_CUR_LIAB"))
    st_debt   = _f(bs_r.get("BS_ST_BORROW"))
    lt_debt   = _f(bs_r.get("BS_LT_BORROW"))

    oper_cf   = _f(cf_r.get("CF_CASH_FROM_OPER"))

    total_debt = _add(st_debt, lt_debt)
    net_debt   = _sub(total_debt, cash)     # total_debt - cash

    z = _altman_z(is_r, bs_r)

    return {
        "period":             is_r.get("FISCAL_YEAR_PERIOD"),
        # Raw (for key balance sheet summary)
        "revenue":            revenue,
        "lt_debt":            lt_debt,
        "cash":               cash,
        "total_assets":       assets,
        # Profitability
        "ebit_margin":        _div(ebit, revenue),
        "net_margin":         _div(ni, revenue),
        "roa":                _div(ni, assets),
        "ocf_margin":         _div(oper_cf, revenue),
        # Leverage
        "interest_coverage":  _div(ebit, int_exp),
        "net_debt_ebitda":    _div(net_debt, ebitda),
        "ocf_debt_ratio":     _div(oper_cf, total_debt),
        # Liquidity
        "current_ratio":      _div(cur_a, cur_l),
        "cash_ratio":         _div(cash, cur_l),
        # Altman
        "altman_z":           z,
        "altman_zone":        _altman_zone(z),
    }


# ── Trend computation ─────────────────────────────────────────────────────────

def _compute_trends(years: list[dict]) -> dict:
    """Compare Y0 vs Y-2 for the main metrics."""
    if len(years) < 3:
        return {}
    y0, y2 = years[0], years[2]
    metrics = [
        "revenue", "ebit_margin", "net_margin", "interest_coverage",
        "net_debt_ebitda", "current_ratio", "ocf_margin", "altman_z",
    ]
    # For debt-like metrics, higher is worse — invert trend direction
    invert = {"net_debt_ebitda"}

    result = {}
    for m in metrics:
        t = _trend(y0.get(m), y2.get(m))
        if m in invert and t != "n/a":
            # flip arrows: ↑↑→↓↓, ↑→↓, ↓→↑, ↓↓→↑↑
            flip = {"↑↑": "↓↓", "↑": "↓", "↓": "↑", "↓↓": "↑↑", "→": "→"}
            t = flip.get(t, t)
        result[m] = t
    return result


# ── Risk warnings ─────────────────────────────────────────────────────────────

def _compute_warnings(latest: dict) -> list[str]:
    """Generate human-readable warning strings from latest-year signals."""
    warnings = []

    ic = latest.get("interest_coverage")
    if ic is not None and ic < 2.0:
        warnings.append(f"⚠ Interest Coverage {ic:.1f}x < 2.0 — debt service pressure")

    nde = latest.get("net_debt_ebitda")
    if nde is not None and nde > 5.0:
        warnings.append(f"⚠ Net Debt/EBITDA {nde:.1f}x > 5.0 — high leverage")

    cr = latest.get("current_ratio")
    if cr is not None and cr < 1.0:
        warnings.append(f"⚠ Current Ratio {cr:.2f}x < 1.0 — liquidity pressure")

    nm = latest.get("net_margin")
    if nm is not None and nm < 0:
        warnings.append(f"⚠ Net Margin {nm*100:.1f}% — operating at a loss")

    z = latest.get("altman_z")
    zone = latest.get("altman_zone")
    if zone == "distress":
        warnings.append(f"⚠ Altman Z {z:.2f} < 1.81 — distress zone")
    elif zone == "grey":
        warnings.append(f"⚠ Altman Z {z:.2f} in grey zone (1.81–2.99)")

    return warnings


# ── Public API ────────────────────────────────────────────────────────────────

def compute_financial_signals(fundamentals: dict) -> dict:
    """Compute multi-year credit-risk signals from raw fundamentals.

    Args:
        fundamentals: dict with keys income_statement, balance_sheet, cash_flow
                      (each is a list of annual dicts, most-recent first)

    Returns:
        {
          signals_by_year: list of per-year dicts (Y0 first),
          trends:          dict of metric → trend arrow (Y0 vs Y-2),
          warnings:        list of warning strings from latest year,
          latest:          convenience alias for signals_by_year[0],
        }
    """
    is_list = fundamentals.get("income_statement") or []
    bs_list = fundamentals.get("balance_sheet") or []
    cf_list = fundamentals.get("cash_flow") or []

    # Zip all three; fall back to empty dicts where a statement is shorter
    n = max(len(is_list), len(bs_list), len(cf_list))
    signals_by_year: list[dict] = []

    for i in range(n):
        is_r = is_list[i] if i < len(is_list) else {}
        bs_r = bs_list[i] if i < len(bs_list) else {}
        cf_r = cf_list[i] if i < len(cf_list) else {}
        signals_by_year.append(_period_signals(is_r, bs_r, cf_r))

    latest = signals_by_year[0] if signals_by_year else {}
    trends = _compute_trends(signals_by_year)
    warnings = _compute_warnings(latest)

    return {
        "signals_by_year": signals_by_year,
        "trends":          trends,
        "warnings":        warnings,
        "latest":          latest,
    }
