FEATURE_FAMILY_DEFINITIONS = (
    ("prices_div_adj", "Prices Div Adj", 73),
    ("technical_candles", "Technical Candles", 12),
    ("technical_cycles", "Technical Cycles", 8),
    ("technical_math", "Technical Math", 8),
    ("technical_momentum", "Technical Momentum", 20),
    ("technical_overlap", "Technical Overlap", 22),
    ("technical_performance", "Technical Performance", 9),
    ("time_calendar", "Time Calendar", 27),
    ("key_metrics", "Key Metrics", 42),
    ("ratios", "Ratios", 59),
    ("income_statement", "Income Statement", 31),
    ("income_statement_growth", "Income Statement Growth", 29),
    ("cash_flow", "Cash Flow", 39),
    ("cash_flow_growth", "Cash Flow Growth", 37),
    ("balance_sheet", "Balance Sheet", 53),
    ("balance_sheet_growth", "Balance Sheet Growth", 51),
    ("financial_growth", "Financial Growth", 39),
    ("earnings", "Earnings", 5),
    ("analyst_estimates", "Analyst Estimates", 4),
    ("ratings_historical", "Ratings Historical", 3),
    ("grades_historical", "Grades Historical", 4),
    ("insider_trading", "Insider Trading", 5),
    ("economic_indicators", "Economic Indicators", 24),
    ("treasury_rates", "Treasury Rates", 12),
    ("representation_embedding", "Representation Embedding", 384),
)

FEATURE_FAMILY_LABELS = {
    key: label for key, label, _count in FEATURE_FAMILY_DEFINITIONS
}

FEATURE_FAMILY_CHOICES = tuple(
    (key, f"{label} ({count})") for key, label, count in FEATURE_FAMILY_DEFINITIONS
)
