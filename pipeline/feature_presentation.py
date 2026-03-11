from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from ml.execution import infer_feature_family_columns
from ml.feature_families import FEATURE_FAMILY_LABELS


DEFAULT_DECIMALS = 2

FAMILY_DISPLAY_NAMES = {
    **FEATURE_FAMILY_LABELS,
    "prices_div_adj": "Price / Technical",
    "model_signals": "Model Signals",
    "novelty": "Novelty",
    "other": "Other",
    "market_situations": "Market Situations",
    "oracle_cluster": "Oracle Clusters",
}

ACRONYM_MAP = {
    "ae": "Analyst Estimates",
    "adj": "Adj",
    "adx": "ADX",
    "amd": "AMD",
    "api": "API",
    "atr": "ATR",
    "avg": "Average",
    "bb": "Bollinger Band",
    "bps": "BPS",
    "cf": "Cash Flow",
    "cfg": "Cash Flow Growth",
    "clv": "CLV",
    "cvar": "CVaR",
    "ebit": "EBIT",
    "ebitda": "EBITDA",
    "ema": "EMA",
    "eps": "EPS",
    "ev": "EV",
    "fcf": "FCF",
    "fred": "FRED",
    "fx": "FX",
    "hl": "High-Low",
    "hh": "High",
    "is": "Income Statement",
    "isg": "Income Statement Growth",
    "ll": "Low",
    "ltm": "LTM",
    "macd": "MACD",
    "mae": "MAE",
    "mfe": "MFE",
    "meta": "Meta",
    "mom": "Momentum",
    "mtl": "MTL",
    "netincome": "Net Income",
    "nvda": "NVDA",
    "obv": "OBV",
    "oc": "Open-Close",
    "pct": "%",
    "pe": "P / E",
    "ppo": "PPO",
    "ps": "P / Sales",
    "px": "Price",
    "qoq": "QoQ",
    "rd": "R&D",
    "ret": "Return",
    "rev": "Revision",
    "rl": "RL",
    "roc": "ROC",
    "rsi": "RSI",
    "sga": "SG&A",
    "shs": "Shares",
    "sma": "SMA",
    "std": "Std Dev",
    "stoch": "Stochastic",
    "surprise": "Surprise",
    "tr": "Treasury Rate",
    "tsla": "TSLA",
    "vol": "Volatility",
    "wfo": "WFO",
    "yoy": "YoY",
    "z": "Z-Score",
}

EXPLICIT_FEATURES = {
    "ev_dividedby_ebitda": ("EV / EBITDA", "ratios", "ratio", 2, None),
    "revenue_growth": ("Revenue Growth", "income_statement_growth", "percent", 2, None),
    "eps_revision_30d": ("EPS Revision (30D)", "analyst_estimates", "percent", 2, None),
    "ret_5d": ("Return (5D)", "prices_div_adj", "percent", 2, None),
    "ret_20d": ("Return (20D)", "prices_div_adj", "percent", 2, None),
    "ret_60d": ("Return (60D)", "prices_div_adj", "percent", 2, None),
    "ret_90d": ("Return (90D)", "prices_div_adj", "percent", 2, None),
    "ret_180d": ("Return (180D)", "prices_div_adj", "percent", 2, None),
    "prob_buy": ("Buy Probability", "model_signals", "percent", 2, None),
    "ranking": ("Ranking Score", "model_signals", "float", 2, None),
    "combined_score": ("Combined Score", "model_signals", "float", 2, None),
    "strategy_score": ("Strategy Score", "model_signals", "float", 2, None),
    "signal_score": ("Signal Score", "model_signals", "float", 2, None),
    "prediction_score": ("Prediction Score", "model_signals", "float", 2, None),
    "ae_familiarity": ("Market Familiarity", "novelty", "float", 2, None),
}


@dataclass(frozen=True)
class FeatureDefinition:
    internal_name: str
    display_name: str
    family: str
    format: str = "float"
    decimals: int = DEFAULT_DECIMALS
    unit: str | None = None


def _family_display_name(family: str) -> str:
    return str(FAMILY_DISPLAY_NAMES.get(str(family), str(family).replace("_", " ").title()))


def _infer_family(name: str) -> str:
    inferred = infer_feature_family_columns([name])
    if inferred:
        return next(iter(inferred.keys()))
    lowered = str(name).lower()
    if lowered.startswith(("predictions_", "classifier_predictions_", "regressor_predictions_", "mtl_predictions_")):
        return "model_signals"
    if lowered.startswith(("ae_", "autoencoder_")):
        return "novelty"
    return "other"


def _infer_format(name: str, family: str) -> tuple[str, int]:
    lowered = str(name).lower()
    if lowered.endswith(("_flag", "_bool")) or "selected_" in lowered or lowered.startswith("is_"):
        return "boolean", 0
    if any(token in lowered for token in ("days_since", "hold_days", "cluster_code")) or lowered in {"k"}:
        return "integer", 0
    if any(token in lowered for token in ("return", "ret_", "growth", "margin", "revision", "surprise", "yield", "ratio_", "bb_pos", "dist_", "gap", "change", "vol_", "pct")):
        return "percent", 2
    if "dividedby" in lowered or lowered.endswith(("ratio", "_multiple")):
        return "ratio", 2
    if family in {
        "income_statement",
        "cash_flow",
        "balance_sheet",
        "financial_growth",
        "key_metrics",
    } and not any(token in lowered for token in ("ratio", "margin", "yield", "growth", "eps")):
        return "currency", 2
    if lowered.endswith(("volume", "_count", "shares", "rows")):
        return "integer", 0
    return "float", 2


def _title_token(token: str) -> str:
    raw = str(token).strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if lowered in ACRONYM_MAP:
        return ACRONYM_MAP[lowered]
    if lowered.isdigit():
        return lowered
    return lowered.replace("netincome", "net income").replace("and", "and").title()


def _heuristic_display_name(name: str) -> str:
    lowered = str(name).strip()
    core = lowered
    for prefix in (
        "px__",
        "is__",
        "isg__",
        "cf__",
        "cfg__",
        "bs__",
        "bsg__",
        "fg__",
        "km__",
        "ratio__",
        "evt__",
        "econ__",
        "fred__",
        "tr__",
        "mtl_predictions_",
        "classifier_predictions_",
        "regressor_predictions_",
    ):
        if core.startswith(prefix):
            core = core.split("__", 1)[1] if "__" in core else core[len(prefix):]
            break
    horizon_match = re.match(r"^(.*?)(?:_)?(\d+)d$", core)
    if horizon_match:
        base = horizon_match.group(1).strip("_")
        horizon = horizon_match.group(2)
        if base in {"ret", "return"}:
            return f"Return ({horizon}D)"
        if base in {"eps_revision", "revenue_revision"}:
            return f"{_title_token(base)} ({horizon}D)"
        if base:
            return f"{_title_token(base)} ({horizon}D)"
    if "_dividedby_" in core:
        left, right = core.split("_dividedby_", 1)
        return f"{_title_token(left)} / {_title_token(right)}"
    if core.endswith("_growth"):
        return f"{_title_token(core[:-7])} Growth"
    if core.endswith("_margin"):
        return f"{_title_token(core[:-7])} Margin"
    if core.endswith("_ratio"):
        return f"{_title_token(core[:-6])} Ratio"
    if core.endswith("_score"):
        return f"{_title_token(core[:-6])} Score"
    if core.endswith("_days_since"):
        return f"{_title_token(core[:-11])} Days Since"
    if core.endswith("_flag"):
        return f"{_title_token(core[:-5])} Flag"
    tokens = [_title_token(token) for token in core.replace("__", "_").split("_") if token]
    display = " ".join(token for token in tokens if token)
    display = re.sub(r"\s+%", "%", display).strip()
    return display or str(name)


def get_feature_definition(feature_name: str) -> FeatureDefinition:
    name = str(feature_name or "").strip()
    if not name:
        return FeatureDefinition("", "-", "other")
    if name in EXPLICIT_FEATURES:
        display_name, family, fmt, decimals, unit = EXPLICIT_FEATURES[name]
        return FeatureDefinition(
            internal_name=name,
            display_name=str(display_name),
            family=str(family),
            format=str(fmt),
            decimals=int(decimals),
            unit=unit,
        )
    family = _infer_family(name)
    fmt, decimals = _infer_format(name, family)
    return FeatureDefinition(
        internal_name=name,
        display_name=_heuristic_display_name(name),
        family=family,
        format=fmt,
        decimals=decimals,
        unit=None,
    )


def _format_compact_number(value: float, decimals: int) -> str:
    number = float(value)
    sign = "-" if number < 0 else ""
    abs_value = abs(number)
    suffixes = (
        (1_000_000_000_000.0, "T"),
        (1_000_000_000.0, "B"),
        (1_000_000.0, "M"),
        (1_000.0, "K"),
    )
    for threshold, suffix in suffixes:
        if abs_value >= threshold:
            return f"{sign}{abs_value / threshold:.{int(decimals)}f}{suffix}"
    if abs_value and abs_value < 0.0001:
        return f"{number:.1e}"
    if float(number).is_integer():
        return f"{int(number)}"
    return f"{number:.{int(decimals)}f}"


def format_feature_value(feature_name: str, value: Any) -> str:
    definition = get_feature_definition(feature_name)
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        parsed = value
    else:
        parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        text = str(value)
        return text if text else "-"
    number = float(parsed)
    if definition.format == "boolean":
        return "Yes" if bool(round(number)) else "No"
    if definition.format == "integer":
        return str(int(round(number)))
    if definition.format == "percent":
        scaled = number * 100.0
        if abs(scaled) < 0.01 and scaled != 0.0:
            return f"{scaled:.1e}%"
        return f"{scaled:.{int(definition.decimals)}f}%"
    if definition.format == "currency":
        return f"${_format_compact_number(number, definition.decimals)}"
    if definition.format == "ratio":
        return f"{number:.{int(definition.decimals)}f}"
    if definition.format == "float":
        if abs(number) < 0.0001 and number != 0.0:
            return f"{number:.1e}"
        if float(number).is_integer():
            return str(int(number))
        return f"{number:.{int(definition.decimals)}f}"
    return str(value)


def render_feature(feature_name: str, value: Any, mode: str = "canonical") -> str:
    definition = get_feature_definition(feature_name)
    rendered_value = format_feature_value(feature_name, value)
    if str(mode or "canonical").strip().lower() == "value":
        return rendered_value
    if str(mode).strip().lower() == "label":
        return definition.display_name
    return f"{definition.display_name}: {rendered_value}"


def group_features_by_family(
    feature_dict: Mapping[str, Any],
    *,
    feature_family_map: Mapping[str, Sequence[str]] | None = None,
) -> OrderedDict[str, list[tuple[FeatureDefinition, Any]]]:
    grouped: OrderedDict[str, list[tuple[FeatureDefinition, Any]]] = OrderedDict()
    if not feature_dict:
        return grouped
    if all(isinstance(value, Mapping) for value in feature_dict.values()):
        for raw_family, values in feature_dict.items():
            family_name = _family_display_name(str(raw_family))
            for internal_name, value in dict(values or {}).items():
                definition = get_feature_definition(str(internal_name))
                grouped.setdefault(family_name, []).append((definition, value))
        return grouped

    family_lookup: dict[str, str] = {}
    for family_name, columns in dict(feature_family_map or {}).items():
        for column in list(columns or []):
            family_lookup[str(column)] = str(family_name)
    items = []
    for internal_name, value in dict(feature_dict or {}).items():
        definition = get_feature_definition(str(internal_name))
        family_key = family_lookup.get(str(internal_name), definition.family)
        items.append((family_key, definition, value))
    items.sort(key=lambda item: (_family_display_name(item[0]), item[1].display_name))
    for family_key, definition, value in items:
        grouped.setdefault(_family_display_name(family_key), []).append((definition, value))
    return grouped


def serialize_features_for_embedding(
    feature_dict: Mapping[str, Any],
    *,
    feature_family_map: Mapping[str, Sequence[str]] | None = None,
) -> str:
    grouped = group_features_by_family(feature_dict, feature_family_map=feature_family_map)
    sections: list[str] = []
    for family_name, rows in grouped.items():
        sections.append(str(family_name))
        for definition, value in rows:
            sections.append(render_feature(definition.internal_name, value, mode="canonical"))
        sections.append("")
    return "\n".join(sections).strip()


def render_feature_family_name(family_name: str) -> str:
    return _family_display_name(str(family_name or "other"))


def render_feature_family_signature(signature: str) -> str:
    raw = str(signature or "").strip()
    if not raw:
        return "-"
    if "+" in raw:
        parts = [part.strip() for part in raw.split("+") if part.strip()]
        return " + ".join(render_feature_family_name(part) for part in parts)
    if "," in raw:
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        return ", ".join(render_feature_family_name(part) for part in parts)
    return render_feature_family_name(raw)
