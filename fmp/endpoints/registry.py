from __future__ import annotations

from .analyst_estimates import build as build_analyst_estimates
from .balance_sheet import build as build_balance_sheet
from .balance_sheet_growth import build as build_balance_sheet_growth
from .cash_flow import build as build_cash_flow
from .cash_flow_growth import build as build_cash_flow_growth
from .dividends import build as build_dividends
from .earnings import build as build_earnings
from .etf_holders import build as build_etf_holders
from .financial_growth import build as build_financial_growth
from .grades import build as build_grades
from .grades_historical import build as build_grades_historical
from .income_statement import build as build_income_statement
from .income_statement_growth import build as build_income_statement_growth
from .insider_trading import build as build_insider_trading
from .institutional_holders import build as build_institutional_holders
from .key_executives import build as build_key_executives
from .key_metrics import build as build_key_metrics
from .mutual_fund_holders import build as build_mutual_fund_holders
from .news import build as build_news
from .peer_symbols import build as build_peer_symbols
from .prices_div_adj import build as build_prices_div_adj
from .prices_unadjusted import build as build_prices_unadjusted
from .profile import build as build_profile
from .quote import build as build_quote
from .ratings_historical import build as build_ratings_historical
from .ratings_snapshot import build as build_ratings_snapshot
from .ratios import build as build_ratios
from .revenue_geographic_segmentation import build as build_revenue_geographic_segmentation
from .revenue_product_segmentation import build as build_revenue_product_segmentation
from .sec_filings import build as build_sec_filings
from .splits import build as build_splits


_BUILDERS = (
    build_prices_div_adj,
    build_prices_unadjusted,
    build_key_metrics,
    build_ratios,
    build_profile,
    build_quote,
    build_dividends,
    build_splits,
    build_earnings,
    build_analyst_estimates,
    build_ratings_snapshot,
    build_ratings_historical,
    build_grades,
    build_grades_historical,
    build_income_statement,
    build_income_statement_growth,
    build_balance_sheet,
    build_balance_sheet_growth,
    build_cash_flow,
    build_cash_flow_growth,
    build_financial_growth,
    build_key_executives,
    build_insider_trading,
    build_institutional_holders,
    build_etf_holders,
    build_mutual_fund_holders,
    build_sec_filings,
    build_news,
    build_revenue_product_segmentation,
    build_revenue_geographic_segmentation,
    build_peer_symbols,
)


def get_symbol_endpoint_definitions(symbol_obj):
    return [builder(symbol_obj) for builder in _BUILDERS]
