from __future__ import annotations

from typing import Sequence


FUNDAMENTAL_PREFIXES = {
    "key_metrics_ttm": ("km_ttm__",),
    "ratios_ttm": ("rt_ttm__",),
    "key_metrics": ("km__",),
    "ratios": ("ratio__", "rt__"),
}

STATEMENT_PREFIXES = {
    "income_statement_ttm": ("is_ttm__",),
    "cash_flow_ttm": ("cf_ttm__",),
    "balance_sheet_ttm": ("bs_ttm__",),
    "income_statement": ("is__",),
    "income_statement_growth": ("isg__",),
    "cash_flow": ("cf__",),
    "cash_flow_growth": ("cfg__",),
    "balance_sheet": ("bs__",),
    "balance_sheet_growth": ("bsg__",),
    "financial_growth": ("fg__",),
}

EVENT_PREFIXES = {
    "earnings": ("evt__earn_",),
    "analyst_estimates": ("evt__ae_",),
    "ratings_historical": ("evt__rating_",),
    "grades_historical": ("evt__grade_",),
}

TIME_CALENDAR_PREFIXES = ("time__",)

TA_CLASSIC_PREFIXES = {
    "technical_candles": ("ta_candle__",),
    "technical_cycles": ("ta_cycle__",),
    "technical_math": ("ta_math__",),
    "technical_momentum": ("ta_momentum__",),
    "technical_overlap": ("ta_overlap__",),
    "technical_performance": ("ta_performance__",),
}

TECHNICAL_PREFIXES = (
    "sma_",
    "ema_",
    "vol_",
    "rsi_",
    "macd_",
    "bb_",
    "atr_",
    "stoch_",
    "adx_",
    "roc_",
    "mom_",
    "px__",
)

PRICE_FAMILY_COLUMNS = {"close", "ret_1", "adj_close", "adj_open", "adj_high", "adj_low", "volume"}


def infer_feature_family_columns(feature_cols: Sequence[str]) -> dict[str, list[str]]:
    """Infer feature families from canonical column prefixes."""

    grouped: dict[str, list[str]] = {
        "prices_div_adj": [],
        "technical_candles": [],
        "technical_cycles": [],
        "technical_math": [],
        "technical_momentum": [],
        "technical_overlap": [],
        "technical_performance": [],
        "time_calendar": [],
        "key_metrics": [],
        "ratios": [],
        "key_metrics_ttm": [],
        "ratios_ttm": [],
        "income_statement_ttm": [],
        "cash_flow_ttm": [],
        "balance_sheet_ttm": [],
        "income_statement": [],
        "income_statement_growth": [],
        "cash_flow": [],
        "cash_flow_growth": [],
        "balance_sheet": [],
        "balance_sheet_growth": [],
        "financial_growth": [],
        "earnings": [],
        "analyst_estimates": [],
        "ratings_historical": [],
        "grades_historical": [],
        "insider_trading": [],
        "economic_indicators": [],
        "treasury_rates": [],
        "representation_embedding": [],
    }
    for col in list(feature_cols):
        name = str(col or "").strip()
        if not name:
            continue
        assigned = False
        if name.startswith(("embedding_", "repr_emb_")):
            grouped["representation_embedding"].append(name)
            assigned = True
        if name in PRICE_FAMILY_COLUMNS or name.startswith(TECHNICAL_PREFIXES):
            grouped["prices_div_adj"].append(name)
            assigned = True
        for family, prefixes in TA_CLASSIC_PREFIXES.items():
            if name.startswith(prefixes):
                grouped[family].append(name)
                assigned = True
        if name.startswith(TIME_CALENDAR_PREFIXES):
            grouped["time_calendar"].append(name)
            assigned = True
        for family, prefixes in FUNDAMENTAL_PREFIXES.items():
            if name.startswith(prefixes):
                grouped[family].append(name)
                assigned = True
        for family, prefixes in STATEMENT_PREFIXES.items():
            if name.startswith(prefixes):
                grouped[family].append(name)
                assigned = True
        for family, prefixes in EVENT_PREFIXES.items():
            if name.startswith(prefixes):
                grouped[family].append(name)
                assigned = True
        if name.startswith("own__insider_"):
            grouped["insider_trading"].append(name)
            assigned = True
        if name.startswith(("econ__", "economic__", "fred__", "macro__")):
            grouped["economic_indicators"].append(name)
            assigned = True
        if name.startswith(("tr__", "treasury__", "yield__", "rate__")):
            grouped["treasury_rates"].append(name)
            assigned = True
        if not assigned and "__" not in name:
            grouped["prices_div_adj"].append(name)
    return {key: list(dict.fromkeys(values)) for key, values in grouped.items() if values}
