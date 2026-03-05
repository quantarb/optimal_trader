from __future__ import annotations

import re

from modules.data.feature_name_map import PRETTY_NAME_MAP


_ACRONYM_MAP = {
    "Adj": "Adjusted",
    "Atr": "ATR",
    "Capex": "CAPEX",
    "Cfo": "CFO",
    "Dcf": "DCF",
    "Etf": "ETF",
    "Eps": "EPS",
    "Ebit": "EBIT",
    "Ebitda": "EBITDA",
    "Ev": "EV",
    "Fcf": "FCF",
    "Macd": "MACD",
    "Pe": "PE",
    "P E": "P/E",
    "P S": "P/S",
    "P B": "P/B",
    "Roe": "ROE",
    "Roa": "ROA",
    "Roi": "ROI",
    "Rsi": "RSI",
    "Pct": "Pct",
    "Ust": "UST",
    "Vwap": "VWAP",
}


def _split_feature_name(name: str) -> tuple[str | None, str]:
    value = str(name or "").strip()
    if not value:
        return None, ""
    if "__" not in value:
        return None, value
    prefix, remainder = value.split("__", 1)
    return prefix, remainder


def _humanize_token(token: str) -> str:
    token = str(token or "").strip()
    if not token:
        return ""
    mapped = PRETTY_NAME_MAP.get(token.lower(), token)
    with_spaces = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", mapped.replace("_", " "))
    pretty = with_spaces.strip().title()
    for src, target in _ACRONYM_MAP.items():
        pretty = re.sub(r"\b" + re.escape(src) + r"\b", target, pretty)
    return pretty


def feature_display_name(name: str) -> str:
    prefix, remainder = _split_feature_name(name)
    if prefix is None:
        return _humanize_token(name)

    if prefix in {"km", "rt", "is", "isg", "cf", "cfg", "bs", "bsg", "fg", "earn", "ae", "rating", "grade", "mcap", "float", "insider"}:
        return _humanize_token(remainder)

    if prefix in {"evt", "own"} and "_" in remainder:
        _, _, detail = remainder.partition("_")
        if detail:
            return _humanize_token(detail)

    return _humanize_token(remainder)
