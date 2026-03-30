"""Feature definitions and column mappings for the credit benchmark feature matrix."""

from pathlib import Path
import os

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv(
    "DATA_DIR",
    os.path.expanduser(
        "~/Library/CloudStorage/OneDrive-共享的库-NationalUniversityofSingapore"
        "/Li Shuping - Data"
    ),
))
FUND_DIR = DATA_DIR / "foundamentals"
MKTCAP_PATH = DATA_DIR / "mktcap.parquet"
CRI_PATH = DATA_DIR / "risk_indicators.csv"

# ── CR2 features (56) — direct columns from cr2_history.parquet ──────────────
CR2_FEATURES = [
    # Size & valuation
    "EARN_YLD", "HISTORICAL_MARKET_CAP",
    # Profitability & margins
    "COGS_TO_NET_SALES", "EARN_FOR_COM_TO_TOT_REV", "EBITDA_MARGIN",
    "EBIT_TO_NET_SALES", "INC_BEF_XO_ITEMS_TO_NET_SALES",
    "OPERATING_EXPENSES_TO_NET_SALES", "OPER_INC_TO_INT_EXP",
    "OPER_INC_TO_NET_SALES", "TRAIL_12M_GROSS_MARGIN",
    "TRAIL_12M_OPER_MARGIN", "TRAIL_12M_PROF_MARGIN",
    # Cash flow & coverage
    "CASH_FLOW_TO_NET_INC", "CFO_BEF_NON_CASH_WC", "CFO_TO_SALES",
    "CF_FREE_CASH_FLOW", "EBITDA_TO_NET_INTEREST", "EXCESS_CASH_FLOW",
    "INT_EXP_TO_TOT_DEBT",
    # Liquidity
    "ACCT_PAY_TO_TOT_LIAB_AND_EQY", "ASSETS_CUR_LIAB",
    "CASH_AND_EQUIV_PER_TOT_LIAB", "CUR_LIAB_TO_TOT_LIAB_AND_EQY",
    "INVENT_TO_CURRENT_ASSETS", "NON_CASH_WORKING_CAPITAL",
    # Leverage & capital structure
    "COM_EQY_TO_TOT_CAP", "LIAB_GROWTH",
    "LT_BORROW_TO_TOT_LIAB_AND_EQY", "LT_LIAB_LESS_DEF_TAX",
    "NET_CHNG_IN_LT_DEBT", "OTHER_LIAB_TOT_MKT_VAL",
    "RETAIN_EARN_TO_TOT_LIAB_AND_EQY", "SHRHLDR_EQY_TO_TOT_LIAB_AND_EQY",
    "ST_BORROW_TO_TOT_LIAB_AND_EQY", "TOT_LIAB_TO_TOT_LIAB_AND_EQY",
    # Asset structure / recovery
    "CAP_EXPEND_TO_SALES", "CAP_EXPEND_TO_TOT_ASSET",
    "SALES_TO_ACCT_RCV", "SALES_TO_CUR_ASSET", "SALES_TO_INVENT",
    "SALES_TO_NET_FIX_ASSET", "SALES_TO_TOT_ASSET",
    # Asset structure / investment / efficiency
    "ACCUM_DEPR_TO_GROSS_FA", "ACCUM_DEPR_TO_TOT_ASSET",
    "GROSS_FIX_ASSET_TO_TOT_ASSET", "NET_CAP_EXP",
    "NET_FIX_ASSET_TO_TOT_ASSET", "TOT_INVEST_TO_TOT_ASSET",
    # Payout & shareholder policy
    "REINVEST_RT", "TOT_COM_DVD_TO_NET_SALES",
    # Liquidity & working capital
    "CHNG_WORK_CAP", "CUR_ASSET_TO_TOT_ASSET", "FONDS_DE_ROULEMENT",
    "MKT_SEC_TO_ASSET", "TRAIL_12M_CHNG_IN_WORK_CAP",
]

# ── CRI features (13) — from risk_indicators.csv ─────────────────────────────
# Column mapping: feature_name → csv column
# "level" features map directly; "trend" features are computed as 12-month diff
CRI_DIRECT = {
    "DTDmedianNonFin": "DTDmedianNonFin",
    "STInt": "STInt",
    "StkIndx": "StkIndx",
    "dtdlevel": "dtd",
    "sigma": "sigma",
    "m2b": "m2b",
    "sizelevel": "size",
    "ni2talevel": "ni2ta",
    "liqnonfinlevel": "liquidity_nonfin",
}
CRI_TREND = {
    # feature_name → base column (trend = current - 12 months ago)
    "dtdtrend": "dtd",
    "sizetrend": "size",
    "ni2tatrend": "ni2ta",
    "liqnonfintrend": "liquidity_nonfin",
}

# ── GPT-proposed features (36) — computed from raw fundamentals + market ─────
# These are defined by their computation formulas in gpt_features.py
GPT_MARKET_FEATURES = [
    "MARKET_CAP_RETURN_1M",
    "MARKET_CAP_RETURN_3M",
    "MARKET_CAP_RETURN_6M",
    "MARKET_CAP_RETURN_12M",
]

GPT_FINANCIAL_FEATURES = [
    # Profitability
    "EBITDA_TO_TOTAL_ASSETS", "EBIT_TO_TOTAL_ASSETS", "ROE",
    # Coverage
    "EBIT_TO_INTEREST_EXPENSE", "CFO_TO_INTEREST_EXPENSE",
    "CFO_TO_TOTAL_ASSETS", "CFO_TO_TOTAL_DEBT",
    "NET_DEBT_TO_EBITDA", "TOTAL_DEBT_TO_EBITDA",
    "FCF_TO_TOTAL_ASSETS", "FCF_TO_TOTAL_DEBT",
    # Liquidity
    "CASH_TO_CURRENT_LIABILITIES",
    "OPERATING_CASH_FLOW_TO_CURRENT_LIABILITIES",
    "QUICK_RATIO", "WORKING_CAPITAL_TO_TOTAL_ASSETS",
    # Leverage
    "MARKET_LEVERAGE_RATIO", "NET_DEBT_TO_CAPITAL",
    "NET_DEBT_TO_MARKET_CAP", "NET_DEBT_TO_TOTAL_ASSETS",
    "RETAINED_EARNINGS_TO_TOTAL_ASSETS", "TOTAL_DEBT_TO_CAPITAL",
    # Valuation (needs market cap)
    "ENTERPRISE_VALUE_TO_ASSETS", "ENTERPRISE_VALUE_TO_SALES",
    # Asset structure
    "CAPEX_TO_DEPRECIATION", "INTANGIBLE_ASSETS_TO_TOTAL_ASSETS",
    "INVENTORY_TO_TOTAL_ASSETS", "PPE_TO_TOTAL_DEBT",
    "RECEIVABLES_TO_TOTAL_ASSETS", "TANGIBLE_ASSETS_TO_TOTAL_ASSETS",
    # Payout
    "DIVIDEND_CUT_FLAG", "DIVIDEND_PAYOUT_RATIO", "SHARE_COUNT_GROWTH_1Y",
]

# All 105 features in order
ALL_FEATURES = (
    CR2_FEATURES
    + GPT_MARKET_FEATURES
    + GPT_FINANCIAL_FEATURES
    + list(CRI_DIRECT.keys())
    + list(CRI_TREND.keys())
)
