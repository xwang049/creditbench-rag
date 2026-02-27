"""Resolve a ticker or ISIN to a company record from the companies table."""

from sqlalchemy import text
from sqlalchemy.orm import Session


class CompanyNotFoundError(Exception):
    pass


def lookup(session: Session, identifier: str) -> dict:
    """Look up a company by ticker (e.g. 'AAPL' or 'AAPL US') or ISIN.

    Returns a dict with keys:
        u3_company_number, id_bb_company, ticker, company_name,
        country_name, industry_sector, industry_group, industry_subgroup,
        market_status, prime_exchange
    """
    identifier = identifier.strip()

    # Normalise: if it looks like a bare ticker (no space), try both bare and " US" suffix
    candidates = [identifier]
    if " " not in identifier and not identifier.startswith("US") and len(identifier) <= 10:
        candidates.append(f"{identifier} US")

    sql = text("""
        SELECT
            c.u3_company_number,
            c.id_bb_company,
            c.ticker,
            c.company_name,
            c.country_name,
            c.market_status,
            c.prime_exchange,
            c.id_isin,
            im.industry_sector,
            im.industry_group,
            im.industry_subgroup
        FROM companies c
        LEFT JOIN industry_mapping im
               ON im.industry_subgroup_num = c.industry_subgroup_num
        WHERE c.ticker ILIKE ANY(:tickers)
           OR c.id_isin  ILIKE :id
        LIMIT 1
    """)

    row = session.execute(sql, {
        "tickers": candidates,
        "id": identifier,
    }).fetchone()

    if row is None:
        raise CompanyNotFoundError(
            f"No company found for '{identifier}'. "
            "Try the Bloomberg ticker format e.g. 'AAPL US' or the full ISIN."
        )

    return {
        "u3_company_number": row.u3_company_number,
        "id_bb_company": row.id_bb_company,
        "ticker": row.ticker,
        "company_name": row.company_name,
        "country_name": row.country_name,
        "market_status": row.market_status,
        "prime_exchange": row.prime_exchange,
        "id_isin": row.id_isin,
        "industry_sector": row.industry_sector,
        "industry_group": row.industry_group,
        "industry_subgroup": row.industry_subgroup,
    }
