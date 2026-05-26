from __future__ import annotations

from datetime import date, timedelta
import json
import math
from pathlib import Path
import sys
from typing import Any
import numpy as np 

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.optimal_trade_lookup import (
    OptimalTradeQuery,
    bootstrap_django,
    find_nearest_optimal_trades,
)


st.set_page_config(
    page_title="Optimal Trade Finder",
    page_icon="OT",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --page-top: #f5fbf6;
        --page-bottom: #eef7f0;
        --panel-bg: rgba(255, 255, 255, 0.92);
        --panel-strong: rgba(255, 255, 255, 0.98);
        --ink: #101714;
        --ink-soft: #244133;
        --accent: #00c805;
        --accent-deep: #009e04;
        --accent-soft: rgba(0, 200, 5, 0.10);
        --sage: #0f8f3b;
        --muted: #5d6f66;
        --line: rgba(20, 44, 29, 0.10);
        --shadow: 0 18px 42px rgba(16, 23, 20, 0.07);
    }
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(0, 200, 5, 0.10), transparent 24%),
            radial-gradient(circle at left 20%, rgba(15, 143, 59, 0.08), transparent 22%),
            linear-gradient(180deg, var(--page-top) 0%, var(--page-bottom) 100%);
        color: var(--ink);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(247, 251, 248, 0.98));
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        box-shadow: var(--shadow);
    }
    div[data-testid="stMetricValue"] {
        color: var(--ink);
    }
    div[data-testid="stMetric"] label {
        color: var(--muted);
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(248, 252, 248, 0.98), rgba(239, 247, 241, 0.98));
        border-right: 1px solid rgba(16, 23, 20, 0.08);
    }
    div[data-testid="stSidebar"] * {
        color: var(--ink);
    }
    .stTextInput input,
    .stNumberInput input,
    .stMultiSelect div[data-baseweb="select"],
    .stSelectbox div[data-baseweb="select"] {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 14px;
    }
    .stDateInput input {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 14px;
    }
    .stButton > button {
        background: linear-gradient(135deg, var(--accent), var(--accent-deep));
        color: #f8fff8;
        border: none;
        border-radius: 999px;
        font-weight: 700;
        min-height: 3rem;
        box-shadow: 0 12px 24px rgba(0, 158, 4, 0.18);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #18d80d, #00a904);
        color: #ffffff;
    }
    div[data-testid="stTabs"] button {
        border-radius: 999px;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 18px;
        overflow: hidden;
        box-shadow: var(--shadow);
        background: rgba(255, 255, 255, 0.92);
    }
    .hero {
        background:
            radial-gradient(circle at top right, rgba(0, 200, 5, 0.12), transparent 28%),
            linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(245, 251, 246, 0.96));
        border: 1px solid var(--line);
        border-radius: 28px;
        padding: 1.45rem 1.55rem;
        margin-bottom: 1.1rem;
        box-shadow: var(--shadow);
    }
    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.32rem 0.72rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--sage);
        font-size: 0.82rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-weight: 700;
    }
    .hero h1 {
        margin: 0.75rem 0 0 0;
        color: var(--ink);
        font-size: 2.5rem;
        line-height: 1.02;
        font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif;
        font-weight: 800;
    }
    .hero p {
        margin: 0.65rem 0 0 0;
        color: var(--muted);
        max-width: 46rem;
        font-size: 1.02rem;
        line-height: 1.55;
    }
    .sidebar-note {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        margin: 0.35rem 0 1rem 0;
        color: var(--ink-soft);
        font-size: 0.94rem;
        line-height: 1.45;
    }
    .results-ribbon {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin: 0 0 1rem 0;
    }
    .results-chip {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.5rem 0.8rem;
        color: var(--ink-soft);
        font-size: 0.92rem;
        box-shadow: 0 8px 18px rgba(16, 23, 20, 0.05);
    }
    .section-title {
        margin: 0.15rem 0 0.35rem 0;
        color: var(--ink);
        font-size: 1.15rem;
        font-weight: 700;
    }
    .section-copy {
        margin: 0 0 0.8rem 0;
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _indicator_column_config() -> dict[str, st.column_config.Column]:
    return {
        "value": st.column_config.NumberColumn("Value", format="%.2f"),
        "reference_mean": st.column_config.NumberColumn("Reference Mean", format="%.2f"),
        "zscore": st.column_config.NumberColumn("Z-Score", format="%.2f"),
    }


def _nearest_trade_column_config() -> dict[str, st.column_config.Column]:
    return {
        "Signed Trade Return": st.column_config.NumberColumn("Signed Trade Return", format="%.2f%%"),
        "Hold Days": st.column_config.NumberColumn("Hold Days", format="%d"),
        "Entry Price": st.column_config.NumberColumn("Entry Price", format="$%.2f"),
        "Exit Price": st.column_config.NumberColumn("Exit Price", format="$%.2f"),
        "AE Familiarity": st.column_config.NumberColumn("Match Score", format="%.2f"),
    }


def _options_payoff_column_config() -> dict[str, st.column_config.Column]:
    return {
        "Last Trade Date (EDT)": st.column_config.TextColumn("Last Trade Date (EDT)"),
        "Strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
        "Implied Volatility": st.column_config.NumberColumn("Implied Volatility", format="%.1f%%"),
        "Last Price": st.column_config.NumberColumn("Last Price", format="$%.2f"),
        "Bid": st.column_config.NumberColumn("Bid", format="$%.2f"),
        "Ask": st.column_config.NumberColumn("Ask", format="$%.2f"),
        "Breakeven": st.column_config.NumberColumn("Breakeven", format="$%.2f"),
        "Entry Cost": st.column_config.NumberColumn("Entry Cost", format="$%.0f"),
        "Projected Value": st.column_config.NumberColumn("Projected Value", format="$%.0f"),
        "Estimated Payoff": st.column_config.NumberColumn("Estimated Payoff", format="$%.0f"),
        "Expected Return On Premium": st.column_config.NumberColumn("Expected Return On Premium", format="%.1f%%"),
        "Return Probability": st.column_config.NumberColumn("Return Probability", format="%.1f%%"),
        "Profit Probability": st.column_config.NumberColumn("Profit Probability", format="%.1f%%"),
        "Payoff Std Dev": st.column_config.NumberColumn("Payoff Std Dev", format="$%.0f"),
        "Downside Case": st.column_config.NumberColumn("Downside Case", format="$%.0f"),
        "Upside Case": st.column_config.NumberColumn("Upside Case", format="$%.0f"),
        "Return On Premium": st.column_config.NumberColumn("Return On Premium", format="%.1f%%"),
    }


def _bucket_confidence(probability: float) -> str:
    value = float(probability)
    if value >= 0.75:
        return "High"
    if value >= 0.60:
        return "Medium"
    return "Low"


def _bucket_opportunity_size(predicted_return_pct: float) -> str:
    value = abs(float(predicted_return_pct))
    if value >= 20.0:
        return "Large"
    if value >= 8.0:
        return "Medium"
    return "Small"


def _bucket_setup_quality(familiarity: float) -> str:
    value = float(familiarity)
    if value >= 0.85:
        return "Very Strong"
    if value >= 0.65:
        return "Solid"
    if value >= 0.45:
        return "Mixed"
    return "Unusual"


def _bucket_hold_time(predicted_hold_days: float) -> str:
    value = float(predicted_hold_days)
    if value >= 45:
        return "Long"
    if value >= 15:
        return "Medium"
    return "Short"


def _format_prediction_metric(predictions: dict[str, dict[str, object]]) -> list[tuple[str, str, str]]:
    cards: list[tuple[str, str, str]] = []
    classifier = predictions.get("classifier") or {}
    probability = pd.to_numeric(pd.Series([classifier.get("probability")]), errors="coerce").iloc[0]
    predicted_class = str(classifier.get("predicted_class") or "").strip()
    if predicted_class:
        cards.append(
            (
                "Oracle Direction",
                predicted_class,
                f"{_bucket_confidence(float(probability))} confidence" if pd.notna(probability) else "Model direction",
            )
        )

    regressor = predictions.get("regressor") or {}
    predicted_return_pct = pd.to_numeric(pd.Series([regressor.get("predicted_trade_return_pct")]), errors="coerce").iloc[0]
    if pd.notna(predicted_return_pct):
        cards.append(
            (
                "Opportunity Size",
                _bucket_opportunity_size(float(predicted_return_pct)),
                "Based on the model's expected move",
            )
        )

    autoencoder = predictions.get("autoencoder") or {}
    familiarity = pd.to_numeric(pd.Series([autoencoder.get("familiarity")]), errors="coerce").iloc[0]
    if pd.notna(familiarity):
        cards.append(
            (
                "Setup Quality",
                _bucket_setup_quality(float(familiarity)),
                "How clean this setup looks versus past examples",
            )
        )

    duration_regressor = predictions.get("duration_regressor") or {}
    predicted_hold_days = pd.to_numeric(pd.Series([duration_regressor.get("predicted_hold_days")]), errors="coerce").iloc[0]
    if pd.notna(predicted_hold_days):
        cards.append(
            (
                "Expected Hold",
                _bucket_hold_time(float(predicted_hold_days)),
                "Short, medium, or long trade length",
            )
        )

    return cards


def _format_similar_trade_summary(nearest_trades: pd.DataFrame) -> list[tuple[str, str, str]]:
    stats = _compute_similar_trade_summary_stats(nearest_trades)
    if not stats:
        return []

    cards: list[tuple[str, str, str]] = []
    top_side = str(stats.get("top_side") or "").strip().title()
    top_side_probability = pd.to_numeric(pd.Series([stats.get("top_side_probability_pct")]), errors="coerce").iloc[0]
    if top_side and pd.notna(top_side_probability):
        cards.append((f"Most Common Side: {top_side}", f"{float(top_side_probability):.2f}%", "Share of similar past trades"))

    average_signed_return = pd.to_numeric(pd.Series([stats.get("average_signed_return_pct")]), errors="coerce").iloc[0]
    if pd.notna(average_signed_return):
        cards.append(("Average Signed Return", f"{float(average_signed_return):.2f}%", "Average outcome from similar trades"))

    average_hold_days = pd.to_numeric(pd.Series([stats.get("average_hold_days")]), errors="coerce").iloc[0]
    if pd.notna(average_hold_days):
        cards.append(("Average Hold Time", f"{float(average_hold_days):.1f}d", "Average days in the trade"))

    return cards


def _compute_similar_trade_summary_stats(nearest_trades: pd.DataFrame) -> dict[str, Any]:
    if nearest_trades.empty:
        return {}

    signed_trade_returns = pd.to_numeric(nearest_trades.get("Signed Trade Return"), errors="coerce")
    if signed_trade_returns.isna().all():
        trade_returns = pd.to_numeric(nearest_trades.get("Trade Return"), errors="coerce")
        sides = nearest_trades.get("Side", pd.Series([""] * len(nearest_trades), index=nearest_trades.index)).astype(str).str.strip().str.lower()
        side_sign = sides.map({"long": 1.0, "short": -1.0}).fillna(1.0)
        signed_trade_returns = trade_returns * side_sign
    hold_days = pd.to_numeric(nearest_trades.get("Hold Days"), errors="coerce")
    sides = nearest_trades.get("Side", pd.Series([""] * len(nearest_trades), index=nearest_trades.index)).astype(str).str.strip().str.lower()

    valid_sides = sides[sides.isin(["long", "short"])]
    top_side = ""
    top_side_probability_pct = float("nan")
    if not valid_sides.empty:
        side_share = valid_sides.value_counts(normalize=True)
        top_side = str(side_share.index[0]).strip().lower()
        top_side_probability_pct = float(side_share.iloc[0]) * 100.0

    average_signed_return_pct = pd.to_numeric(pd.Series([signed_trade_returns.mean()]), errors="coerce").iloc[0]
    signed_return_std_pct = pd.to_numeric(pd.Series([signed_trade_returns.std(ddof=0)]), errors="coerce").iloc[0]
    average_hold_days = pd.to_numeric(pd.Series([hold_days.mean()]), errors="coerce").iloc[0]
    hold_days_std = pd.to_numeric(pd.Series([hold_days.std(ddof=0)]), errors="coerce").iloc[0]
    return {
        "top_side": top_side,
        "top_side_probability_pct": None if pd.isna(top_side_probability_pct) else float(top_side_probability_pct),
        "average_signed_return_pct": None if pd.isna(average_signed_return_pct) else float(average_signed_return_pct),
        "signed_return_std_pct": None if pd.isna(signed_return_std_pct) else float(signed_return_std_pct),
        "average_hold_days": None if pd.isna(average_hold_days) else float(average_hold_days),
        "hold_days_std": None if pd.isna(hold_days_std) else float(hold_days_std),
    }


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def _black_scholes_price(
    *,
    spot_price: float,
    strike_price: float,
    days_to_expiry: float,
    sigma: float,
    option_type: str,
    rate: float = 0.0,
) -> float:
    spot = max(float(spot_price), 1e-6)
    strike = max(float(strike_price), 1e-6)
    tau = max(float(days_to_expiry) / 252.0, 1.0 / 252.0)
    vol = max(float(sigma), 1e-6)
    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(spot / strike) + (float(rate) + 0.5 * vol * vol) * tau) / (vol * sqrt_tau)
    d2 = d1 - vol * sqrt_tau
    if str(option_type).strip().lower() == "put":
        price = strike * math.exp(-float(rate) * tau) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
        intrinsic = max(strike - spot, 0.0)
    else:
        price = spot * _norm_cdf(d1) - strike * math.exp(-float(rate) * tau) * _norm_cdf(d2)
        intrinsic = max(spot - strike, 0.0)
    if not math.isfinite(price):
        price = intrinsic
    return max(float(price), 0.25)


def _norm_pdf(value: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * float(value) * float(value))


def _black_scholes_greeks(
    *,
    spot_price: float,
    strike_price: float,
    days_to_expiry: float,
    sigma: float,
    option_type: str,
    rate: float = 0.0,
) -> dict[str, float]:
    spot = max(float(spot_price), 1e-6)
    strike = max(float(strike_price), 1e-6)
    tau = max(float(days_to_expiry) / 252.0, 1.0 / 252.0)
    vol = max(float(sigma), 1e-6)
    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(spot / strike) + (float(rate) + 0.5 * vol * vol) * tau) / (vol * sqrt_tau)
    d2 = d1 - vol * sqrt_tau
    pdf_d1 = _norm_pdf(d1)
    option_type_value = str(option_type).strip().lower()
    if option_type_value == "put":
        delta = _norm_cdf(d1) - 1.0
        theta = (
            -(spot * pdf_d1 * vol) / (2.0 * sqrt_tau)
            + float(rate) * strike * math.exp(-float(rate) * tau) * _norm_cdf(-d2)
        ) / 252.0
    else:
        delta = _norm_cdf(d1)
        theta = (
            -(spot * pdf_d1 * vol) / (2.0 * sqrt_tau)
            - float(rate) * strike * math.exp(-float(rate) * tau) * _norm_cdf(d2)
        ) / 252.0
    gamma = pdf_d1 / (spot * vol * sqrt_tau)
    vega = spot * pdf_d1 * sqrt_tau / 100.0
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
    }


def _option_exit_value(
    *,
    spot_price: float,
    strike_price: float,
    remaining_days_to_expiry: float,
    sigma: float,
    option_type: str,
    rate: float = 0.0,
) -> float:
    spot = max(float(spot_price), 0.01)
    strike = max(float(strike_price), 0.01)
    option_type_value = str(option_type).strip().lower()
    if float(remaining_days_to_expiry) <= 0:
        if option_type_value == "put":
            return max(strike - spot, 0.0)
        return max(spot - strike, 0.0)

    tau = max(float(remaining_days_to_expiry) / 252.0, 1.0 / 252.0)
    vol = max(float(sigma), 1e-6)
    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(spot / strike) + (float(rate) + 0.5 * vol * vol) * tau) / (vol * sqrt_tau)
    d2 = d1 - vol * sqrt_tau
    if option_type_value == "put":
        price = strike * math.exp(-float(rate) * tau) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
        intrinsic = max(strike - spot, 0.0)
    else:
        price = spot * _norm_cdf(d1) - strike * math.exp(-float(rate) * tau) * _norm_cdf(d2)
        intrinsic = max(spot - strike, 0.0)
    if not math.isfinite(price):
        price = intrinsic
    return max(float(price), float(intrinsic), 0.0)


def _resolve_similarity_weights(nearest_trades: pd.DataFrame) -> np.ndarray:
    familiarity = pd.to_numeric(nearest_trades.get("AE Familiarity"), errors="coerce")
    if familiarity.notna().any():
        weights = familiarity.clip(lower=0.0).to_numpy(dtype=float, copy=False)
    else:
        weights = np.ones(len(nearest_trades), dtype=float)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return np.full(len(nearest_trades), 1.0 / max(len(nearest_trades), 1), dtype=float)
    return weights / total_weight


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.average(values, weights=weights))


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    mean_value = _weighted_mean(values, weights)
    variance = float(np.average((values - mean_value) ** 2, weights=weights))
    return math.sqrt(max(variance, 0.0))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        return float("nan")
    q = min(max(float(quantile), 0.0), 1.0)
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(sorted_weights)
    if cumulative.size == 0 or cumulative[-1] <= 0:
        return float("nan")
    threshold = q * cumulative[-1]
    idx = int(np.searchsorted(cumulative, threshold, side="left"))
    idx = min(max(idx, 0), len(sorted_values) - 1)
    return float(sorted_values[idx])


def _normal_interval_probability(*, mean: float, std_dev: float, low: float, high: float) -> float:
    if not math.isfinite(mean):
        return float("nan")
    if not math.isfinite(std_dev) or std_dev <= 1e-9:
        return 1.0 if float(low) <= float(mean) <= float(high) else 0.0
    z_low = (float(low) - float(mean)) / float(std_dev)
    z_high = (float(high) - float(mean)) / float(std_dev)
    return max(min(_norm_cdf(z_high) - _norm_cdf(z_low), 1.0), 0.0)


def _softmax_weights(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    temp = max(float(temperature), 1e-9)
    scaled = arr / temp
    scaled = scaled - np.nanmax(scaled)
    exp_values = np.exp(np.nan_to_num(scaled, nan=-1e9))
    total = float(exp_values.sum())
    if total <= 0:
        return np.full(len(arr), 1.0 / max(len(arr), 1), dtype=float)
    return exp_values / total


def _resolve_classifier_long_probability(model_predictions: dict[str, dict[str, Any]]) -> float | None:
    classifier = dict(model_predictions.get("classifier") or {})
    class_probabilities = dict(classifier.get("class_probabilities") or {})
    normalized = {str(key).strip().lower(): float(value) for key, value in class_probabilities.items() if pd.notna(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])}
    if "long" in normalized:
        return float(normalized["long"])
    if "short" in normalized:
        return float(1.0 - normalized["short"])
    predicted_class = str(classifier.get("predicted_class") or "").strip().lower()
    probability = pd.to_numeric(pd.Series([classifier.get("probability")]), errors="coerce").iloc[0]
    if pd.isna(probability):
        return None
    if predicted_class == "long":
        return float(probability)
    if predicted_class == "short":
        return float(1.0 - float(probability))
    return None


def _resolve_model_expected_return_decimal(model_predictions: dict[str, dict[str, Any]], option_type: str) -> float | None:
    regressor = dict(model_predictions.get("regressor") or {})
    predicted_trade_return_pct = pd.to_numeric(pd.Series([regressor.get("predicted_trade_return_pct")]), errors="coerce").iloc[0]
    if pd.isna(predicted_trade_return_pct):
        return None
    classifier = dict(model_predictions.get("classifier") or {})
    predicted_class = str(classifier.get("predicted_class") or "").strip().lower()
    signed_return_decimal = float(predicted_trade_return_pct) / 100.0
    if predicted_class == "short":
        signed_return_decimal *= -1.0
    option_type_value = str(option_type).strip().lower()
    if option_type_value == "put":
        signed_return_decimal *= -1.0
    return float(signed_return_decimal)


def _build_option_scenarios(
    *,
    nearest_trades: pd.DataFrame,
    option_type: str,
) -> pd.DataFrame:
    work = nearest_trades.copy()
    signed_returns = pd.to_numeric(work.get("Signed Trade Return"), errors="coerce") / 100.0
    hold_days = pd.to_numeric(work.get("Hold Days"), errors="coerce")
    familiarity = pd.to_numeric(work.get("AE Familiarity"), errors="coerce")
    distance = 1.0 - familiarity.clip(lower=0.0, upper=1.0)
    scenario_frame = pd.DataFrame(
        {
            "stock_return": signed_returns,
            "hold_days": hold_days,
            "distance": distance,
            "ae_familiarity": familiarity,
        }
    ).dropna(subset=["stock_return", "hold_days"])
    if scenario_frame.empty:
        return scenario_frame
    if str(option_type).strip().lower() == "put":
        scenario_frame["stock_return"] = -scenario_frame["stock_return"].astype(float)
    return scenario_frame.reset_index(drop=True)


def _compute_option_scenario_weights(
    scenarios: pd.DataFrame,
    *,
    model_expected_return: float | None,
    ae_familiarity: float | None,
    distance_temperature: float = 0.25,
    return_alignment_temperature: float = 0.10,
) -> pd.DataFrame:
    if scenarios.empty:
        out = scenarios.copy()
        out["scenario_weight"] = pd.Series(dtype=float)
        return out

    df = scenarios.copy()
    similarity_score = -pd.to_numeric(df["distance"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    final_weight = _softmax_weights(similarity_score, temperature=distance_temperature)

    if model_expected_return is not None and np.isfinite(float(model_expected_return)):
        scenario_returns = pd.to_numeric(df["stock_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        alignment_error = np.abs(scenario_returns - float(model_expected_return))
        alignment_score = -alignment_error
        alignment_weight = _softmax_weights(alignment_score, temperature=return_alignment_temperature)
        final_weight *= alignment_weight

    if ae_familiarity is not None and np.isfinite(float(ae_familiarity)):
        setup_quality = float(np.clip(float(ae_familiarity), 0.0, 1.0))
        uniform_weight = np.full(len(final_weight), 1.0 / max(len(final_weight), 1), dtype=float)
        final_weight = setup_quality * final_weight + (1.0 - setup_quality) * uniform_weight

    total_weight = float(final_weight.sum())
    if total_weight <= 0:
        final_weight = np.full(len(final_weight), 1.0 / max(len(final_weight), 1), dtype=float)
    else:
        final_weight = final_weight / total_weight
    df["scenario_weight"] = final_weight
    return df


def _resolve_nearest_option_expiry(as_of_date: object, target_hold_days: float) -> tuple[pd.Timestamp, int]:
    base_date = pd.Timestamp(as_of_date).normalize()
    target_days = max(int(round(float(target_hold_days))), 7)
    target_date = base_date + pd.Timedelta(days=target_days)
    friday_candidates = [
        (base_date + pd.Timedelta(days=offset)).normalize()
        for offset in range(1, 370)
        if (base_date + pd.Timedelta(days=offset)).weekday() == 4
    ]
    if not friday_candidates:
        expiry_date = target_date
    else:
        expiry_date = min(
            friday_candidates,
            key=lambda candidate: (abs((candidate - target_date).days), candidate),
        )
    days_to_expiry = max(int((expiry_date - base_date).days), 1)
    return expiry_date, days_to_expiry


def _build_options_payoff_snapshot(
    *,
    symbol: str,
    as_of_date: object,
    nearest_trades: pd.DataFrame,
) -> dict[str, Any]:
    stats = _compute_similar_trade_summary_stats(nearest_trades)
    top_side = str(stats.get("top_side") or "").strip().lower()
    average_signed_return_pct = pd.to_numeric(pd.Series([stats.get("average_signed_return_pct")]), errors="coerce").iloc[0]
    average_hold_days = pd.to_numeric(pd.Series([stats.get("average_hold_days")]), errors="coerce").iloc[0]
    if top_side not in {"long", "short"} or pd.isna(average_signed_return_pct) or pd.isna(average_hold_days):
        return {}

    as_of_ts = pd.Timestamp(as_of_date).normalize()
    lookback_start = (as_of_ts - pd.Timedelta(days=220)).strftime("%Y-%m-%d")
    ohlcv = _load_symbol_ohlcv(str(symbol).strip().upper(), lookback_start, as_of_ts.strftime("%Y-%m-%d"))
    if ohlcv.empty:
        return {}

    price_frame = ohlcv.copy()
    price_frame["Date"] = pd.to_datetime(price_frame["Date"], errors="coerce").dt.normalize()
    price_frame = price_frame.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    price_frame = price_frame.loc[price_frame["Date"] <= as_of_ts].copy()
    if price_frame.empty:
        return {}

    spot_price = pd.to_numeric(pd.Series([price_frame["Close"].iloc[-1]]), errors="coerce").iloc[0]
    if pd.isna(spot_price) or float(spot_price) <= 0:
        return {}

    returns = pd.to_numeric(price_frame["Close"], errors="coerce").pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    realized_vol = float(returns.tail(60).std(ddof=0) * math.sqrt(252.0)) if not returns.empty else 0.0
    realized_vol = min(max(realized_vol, 0.15), 1.50)

    expiry_date, days_to_expiry = _resolve_nearest_option_expiry(as_of_ts, float(average_hold_days))
    option_type = "call" if top_side == "long" else "put"
    contract_label = "Long Call" if option_type == "call" else "Long Put"
    strike_price = float(spot_price)
    projected_underlying_price = float(spot_price) * (1.0 + float(average_signed_return_pct) / 100.0)
    projected_underlying_price = max(projected_underlying_price, 0.01)
    premium_per_share = _black_scholes_price(
        spot_price=float(spot_price),
        strike_price=float(strike_price),
        days_to_expiry=float(days_to_expiry),
        sigma=float(realized_vol),
        option_type=option_type,
        rate=0.0,
    )
    if option_type == "call":
        value_at_expiry_per_share = max(projected_underlying_price - strike_price, 0.0)
    else:
        value_at_expiry_per_share = max(strike_price - projected_underlying_price, 0.0)
    net_payoff_per_share = float(value_at_expiry_per_share) - float(premium_per_share)
    contract_multiplier = 100.0
    return_on_premium_pct = (net_payoff_per_share / premium_per_share) * 100.0 if premium_per_share > 0 else float("nan")
    return {
        "contract_label": contract_label,
        "option_type": option_type.title(),
        "expiry_date": expiry_date.strftime("%Y-%m-%d"),
        "days_to_expiry": int(days_to_expiry),
        "spot_price": float(spot_price),
        "strike_price": float(strike_price),
        "projected_underlying_price": float(projected_underlying_price),
        "premium_per_share": float(premium_per_share),
        "premium_per_contract": float(premium_per_share * contract_multiplier),
        "value_at_expiry_per_share": float(value_at_expiry_per_share),
        "value_at_expiry_per_contract": float(value_at_expiry_per_share * contract_multiplier),
        "net_payoff_per_share": float(net_payoff_per_share),
        "net_payoff_per_contract": float(net_payoff_per_share * contract_multiplier),
        "return_on_premium_pct": None if pd.isna(return_on_premium_pct) else float(return_on_premium_pct),
        "realized_vol_pct": float(realized_vol * 100.0),
    }


def _build_options_payoff_table(
    *,
    symbol: str,
    as_of_date: object,
    nearest_trades: pd.DataFrame,
    model_predictions: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    snapshot = _build_options_payoff_snapshot(
        symbol=symbol,
        as_of_date=as_of_date,
        nearest_trades=nearest_trades,
    )
    if not snapshot:
        return pd.DataFrame(), {}

    option_type = str(snapshot.get("option_type") or "").strip().lower()
    expiry_date = str(snapshot.get("expiry_date") or "")
    days_to_expiry = int(snapshot.get("days_to_expiry") or 0)
    spot_price = float(snapshot.get("spot_price") or 0.0)
    projected_underlying_price = float(snapshot.get("projected_underlying_price") or 0.0)
    realized_vol = float(snapshot.get("realized_vol_pct") or 0.0) / 100.0
    setup_quality = pd.to_numeric(pd.Series([(model_predictions or {}).get("autoencoder", {}).get("familiarity")]), errors="coerce").iloc[0]
    setup_quality = float(setup_quality) if pd.notna(setup_quality) else 0.5
    symbol_upper = str(symbol).strip().upper()
    if spot_price <= 0 or projected_underlying_price <= 0 or days_to_expiry <= 0:
        return pd.DataFrame(), {}

    if option_type == "put":
        strike_multipliers = [1.15, 1.10, 1.05, 1.00, 0.95, 0.90, 0.85]
        right_code = "P"
    else:
        strike_multipliers = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
        right_code = "C"

    model_expected_return = _resolve_model_expected_return_decimal(dict(model_predictions or {}), option_type)
    scenario_frame = _build_option_scenarios(nearest_trades=nearest_trades, option_type=option_type)
    if scenario_frame.empty:
        return pd.DataFrame(), snapshot
    weighted_scenarios = _compute_option_scenario_weights(
        scenario_frame,
        model_expected_return=model_expected_return,
        ae_familiarity=setup_quality,
    )
    scenario_weights = pd.to_numeric(weighted_scenarios["scenario_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
    scenario_return_values = pd.to_numeric(weighted_scenarios["stock_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
    scenario_hold_values = pd.to_numeric(weighted_scenarios["hold_days"], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
    scenario_projected_prices = np.maximum(spot_price * (1.0 + scenario_return_values), 0.01)
    terminal_price_mean = _weighted_mean(scenario_projected_prices, scenario_weights)
    terminal_price_std = _weighted_std(scenario_projected_prices, scenario_weights)
    classifier_long_probability = _resolve_classifier_long_probability(dict(model_predictions or {}))
    if classifier_long_probability is None:
        classifier_long_probability = 0.5
    option_side_probability = float(classifier_long_probability) if option_type == "call" else float(1.0 - classifier_long_probability)
    duration_regressor = dict((model_predictions or {}).get("duration_regressor") or {})
    predicted_hold_days = pd.to_numeric(pd.Series([duration_regressor.get("predicted_hold_days")]), errors="coerce").iloc[0]
    blended_hold_days = float(predicted_hold_days) if pd.notna(predicted_hold_days) else float(snapshot["days_to_expiry"])

    rows: list[dict[str, Any]] = []
    for multiplier in strike_multipliers:
        strike_price = max(round(float(spot_price) * float(multiplier), 2), 0.01)
        moneyness_gap = abs(math.log(max(strike_price / max(spot_price, 0.01), 1e-6)))
        model_return_scale = abs(float(model_expected_return)) if model_expected_return is not None else abs(float(np.average(scenario_return_values, weights=scenario_weights)))
        implied_vol = realized_vol * (1.0 + 0.20 * model_return_scale + 0.28 * moneyness_gap)
        implied_vol = min(max(implied_vol, 0.15), 2.00)
        mid_price = _black_scholes_price(
            spot_price=spot_price,
            strike_price=strike_price,
            days_to_expiry=float(days_to_expiry),
            sigma=implied_vol,
            option_type=option_type,
            rate=0.0,
        )
        greeks = _black_scholes_greeks(
            spot_price=spot_price,
            strike_price=strike_price,
            days_to_expiry=float(days_to_expiry),
            sigma=implied_vol,
            option_type=option_type,
            rate=0.0,
        )
        spread = max(mid_price * 0.06, 0.05)
        bid_price = max(mid_price - spread / 2.0, 0.01)
        ask_price = max(mid_price + spread / 2.0, bid_price)
        breakeven_price = strike_price - ask_price if option_type == "put" else strike_price + ask_price
        if option_type == "put":
            projected_value_per_share = max(strike_price - projected_underlying_price, 0.0)
        else:
            projected_value_per_share = max(projected_underlying_price - strike_price, 0.0)
        entry_cost = ask_price * 100.0
        scenario_remaining_days = np.maximum(float(days_to_expiry) - scenario_hold_values, 0.0)
        scenario_exit_values = np.asarray(
            [
                _option_exit_value(
                    spot_price=float(next_spot),
                    strike_price=float(strike_price),
                    remaining_days_to_expiry=float(next_remaining_days),
                    sigma=float(implied_vol),
                    option_type=option_type,
                    rate=0.0,
                )
                for next_spot, next_remaining_days in zip(scenario_projected_prices, scenario_remaining_days)
            ],
            dtype=float,
        ) * 100.0
        scenario_option_returns = np.where(entry_cost > 0, (scenario_exit_values - float(entry_cost)) / float(entry_cost), 0.0)
        scenario_payoffs = scenario_exit_values - float(entry_cost)
        projected_value = _weighted_mean(scenario_exit_values, scenario_weights)
        estimated_payoff = _weighted_mean(scenario_payoffs, scenario_weights)
        payoff_std = _weighted_std(scenario_payoffs, scenario_weights)
        downside_case = _weighted_quantile(scenario_payoffs, scenario_weights, 0.25)
        upside_case = _weighted_quantile(scenario_payoffs, scenario_weights, 0.75)
        profit_probability = float(scenario_weights[scenario_payoffs > 0].sum() * 100.0)
        return_band_half_width = max(float(ask_price), float(terminal_price_std) * 0.20, float(spot_price) * 0.02)
        return_probability = _normal_interval_probability(
            mean=float(terminal_price_mean),
            std_dev=float(terminal_price_std),
            low=float(breakeven_price) - float(return_band_half_width),
            high=float(breakeven_price) + float(return_band_half_width),
        ) * 100.0
        return_on_premium = (estimated_payoff / entry_cost) * 100.0 if entry_cost > 0 else float("nan")
        greek_theta_drag = min(max(abs(float(greeks["theta"])) * min(blended_hold_days, float(days_to_expiry)) * 100.0 / max(entry_cost, 1.0), 0.0), 0.95)
        gamma_scale = min(max(abs(float(greeks["gamma"])) * float(spot_price) * float(spot_price) * 0.01, 0.0), 1.0)
        delta_scale = min(max(abs(float(greeks["delta"])), 0.0), 1.0)
        greek_efficiency = max(0.20, min(1.40, (0.55 * delta_scale + 0.45 * gamma_scale) * (1.0 - 0.50 * greek_theta_drag)))
        expected_return_on_premium_raw = _weighted_mean(scenario_option_returns, scenario_weights)
        expected_return_on_premium = float(expected_return_on_premium_raw * 100.0 * option_side_probability * greek_efficiency)
        rows.append(
            {
                "Contract Name": f"{symbol_upper} {expiry_date} {strike_price:.2f} {right_code}",
                "Last Trade Date (EDT)": expiry_date,
                "Strike": float(strike_price),
                "Implied Volatility": float(implied_vol * 100.0),
                "Last Price": float(mid_price),
                "Bid": float(bid_price),
                "Ask": float(ask_price),
                "Breakeven": float(breakeven_price),
                "Entry Cost": float(entry_cost),
                "Projected Value": float(projected_value),
                "Estimated Payoff": float(estimated_payoff),
                "Expected Return On Premium": float(expected_return_on_premium),
                "Return Probability": float(return_probability),
                "Profit Probability": float(profit_probability),
                "Payoff Std Dev": float(payoff_std),
                "Downside Case": float(downside_case),
                "Upside Case": float(upside_case),
                "Return On Premium": None if pd.isna(return_on_premium) else float(return_on_premium),
                "__distance_to_money": abs(float(strike_price) - float(spot_price)),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(), snapshot
    table = table.sort_values(["Strike"], ascending=[True]).reset_index(drop=True)
    return table, snapshot


def _build_trade_label(row: pd.Series | dict[str, Any]) -> str:
    payload = row if isinstance(row, pd.Series) else pd.Series(dict(row or {}))
    side = str(payload.get("Side") or payload.get("side") or "").strip().title() or "Trade"
    entry_date = str(payload.get("Entry Date") or payload.get("entry_date") or "").strip()
    exit_date = str(payload.get("Exit Date") or payload.get("exit_date") or "").strip()
    if entry_date and exit_date:
        return f"{side} • {entry_date} to {exit_date}"
    if entry_date:
        return f"{side} • {entry_date}"
    return side


def _prepare_trade_focus_table(nearest_trades: pd.DataFrame) -> pd.DataFrame:
    if nearest_trades.empty:
        return nearest_trades.copy()
    table = nearest_trades.copy()
    table["__trade_label"] = table.apply(_build_trade_label, axis=1)
    preferred_columns = [
        "Symbol",
        "Side",
        "Entry Date",
        "Entry Price",
        "Exit Date",
        "Exit Price",
        "Signed Trade Return",
        "Hold Days",
        "AE Familiarity",
    ]
    keep_columns = [column for column in preferred_columns if column in table.columns]
    remainder = [column for column in table.columns if column not in keep_columns]
    return table[keep_columns + remainder]


def _format_trade_table_cell(column: str, value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")

    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric_value):
        numeric_value = float(numeric_value)
        if column == "Signed Trade Return":
            return f"{numeric_value:.2f}%"
        if column in {"Entry Price", "Exit Price"}:
            return f"${numeric_value:.2f}"
        if column == "Hold Days":
            return f"{int(round(numeric_value))}"
        if column == "AE Familiarity":
            return f"{numeric_value:.2f}"
        return f"{numeric_value:.2f}"
    return str(value)


@st.cache_data(show_spinner=False)
def _load_symbol_ohlcv(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    bootstrap_django()
    from trading.live_trade import build_technical_dataframe_from_django

    technical_df, _technical_cols = build_technical_dataframe_from_django(
        symbols=[str(symbol).strip().upper()],
        start_date=str(start_date),
        end_date=str(end_date),
    )
    if technical_df.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume", "Symbol"])

    ohlcv = technical_df.reset_index()[["date", "symbol", "open", "high", "low", "close", "volume"]].copy()
    ohlcv = ohlcv.rename(
        columns={
            "date": "Date",
            "symbol": "Symbol",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    ohlcv["Date"] = pd.to_datetime(ohlcv["Date"], errors="coerce")
    ohlcv = ohlcv.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return ohlcv


def _payload_value(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload.get(key) not in (None, ""):
            return payload.get(key)
    return None


def _price_payload_to_row(symbol: str, event_label: str, target_date: pd.Timestamp, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "Event": event_label,
        "Symbol": str(symbol).strip().upper(),
        "Date": pd.Timestamp(target_date).strftime("%Y-%m-%d"),
        "Open": pd.to_numeric(pd.Series([_payload_value(payload, "open", "adjOpen")]), errors="coerce").iloc[0],
        "High": pd.to_numeric(pd.Series([_payload_value(payload, "high", "adjHigh")]), errors="coerce").iloc[0],
        "Low": pd.to_numeric(pd.Series([_payload_value(payload, "low", "adjLow")]), errors="coerce").iloc[0],
        "Close": pd.to_numeric(pd.Series([_payload_value(payload, "close", "adjClose")]), errors="coerce").iloc[0],
        "Volume": pd.to_numeric(pd.Series([_payload_value(payload, "volume", "adjVolume")]), errors="coerce").iloc[0],
    }


def _news_payload_to_row(symbol: str, event_label: str, target_date: pd.Timestamp, payload: dict[str, Any]) -> dict[str, Any]:
    published = _payload_value(payload, "publishedDate", "publishedAt", "date")
    return {
        "Event": event_label,
        "Symbol": str(symbol).strip().upper(),
        "Date": pd.Timestamp(target_date).strftime("%Y-%m-%d"),
        "Published": str(published or "")[:19],
        "Title": str(_payload_value(payload, "title", "headline") or ""),
        "Publisher": str(_payload_value(payload, "site", "publisher", "source") or ""),
        "URL": str(_payload_value(payload, "url", "link") or ""),
    }


@st.cache_data(show_spinner=False)
def _load_trade_event_details(symbol: str, entry_date: str, exit_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    bootstrap_django()
    from fmp.models import Symbol, SymbolSectionHistorical
    from trading.live_trade import resolve_fmp_api_key
    from data import FMPClient

    symbol_upper = str(symbol).strip().upper()
    date_pairs = [
        ("Entry", pd.to_datetime(entry_date, errors="coerce")),
        ("Exit", pd.to_datetime(exit_date, errors="coerce")),
    ]
    date_pairs = [(label, pd.Timestamp(value).normalize()) for label, value in date_pairs if pd.notna(value)]
    if not symbol_upper or not date_pairs:
        return pd.DataFrame(), pd.DataFrame()

    symbol_obj = Symbol.objects.filter(symbol__iexact=symbol_upper).first()
    if symbol_obj is None:
        return pd.DataFrame(), pd.DataFrame()

    wanted_dates = {value.date() for _, value in date_pairs}
    price_rows: list[dict[str, Any]] = []
    news_rows: list[dict[str, Any]] = []

    price_qs = SymbolSectionHistorical.objects.filter(
        symbol=symbol_obj,
        section_key="prices_div_adj",
        record_date__in=sorted(wanted_dates),
    ).only("record_date", "payload")
    price_by_date = {
        row.record_date: row.payload if isinstance(row.payload, dict) else {}
        for row in price_qs
        if row.record_date is not None
    }
    for event_label, target_ts in date_pairs:
        payload = price_by_date.get(target_ts.date())
        if payload:
            price_rows.append(_price_payload_to_row(symbol_upper, event_label, target_ts, payload))

    news_qs = SymbolSectionHistorical.objects.filter(
        symbol=symbol_obj,
        section_key="news",
        record_date__in=sorted(wanted_dates),
    ).only("record_date", "payload").order_by("record_date", "updated_at")
    for row in news_qs:
        if row.record_date is None:
            continue
        payload = row.payload if isinstance(row.payload, dict) else {}
        event_label = "Entry" if row.record_date == date_pairs[0][1].date() else "Exit"
        news_rows.append(_news_payload_to_row(symbol_upper, event_label, pd.Timestamp(row.record_date), payload))

    missing_news_dates = sorted(wanted_dates - {pd.Timestamp(row["Date"]).date() for row in news_rows})
    api_key = resolve_fmp_api_key(required=False)
    if missing_news_dates and api_key:
        client = FMPClient(api_key=api_key, timeout_s=15.0, max_retries=1)
        for missing_date in missing_news_dates:
            try:
                fetched = client.get_df(
                    "/stable/news/stock",
                    params={
                        "symbols": symbol_upper,
                        "from": missing_date.isoformat(),
                        "to": missing_date.isoformat(),
                        "limit": 50,
                    },
                )
            except Exception:
                fetched = pd.DataFrame()
            if fetched is None or fetched.empty:
                continue
            for payload in fetched.to_dict("records"):
                published = pd.to_datetime(_payload_value(payload, "publishedDate", "publishedAt", "date"), errors="coerce")
                if pd.isna(published) or pd.Timestamp(published).date() != missing_date:
                    continue
                event_label = next((label for label, ts in date_pairs if ts.date() == missing_date), "")
                news_rows.append(_news_payload_to_row(symbol_upper, event_label, pd.Timestamp(missing_date), payload))

    return pd.DataFrame(price_rows), pd.DataFrame(news_rows)


def _resolve_trade_chart_point(
    ohlcv: pd.DataFrame,
    target_date: pd.Timestamp,
    *,
    prefer: str,
) -> tuple[pd.Timestamp | None, float | None]:
    if ohlcv.empty or pd.isna(target_date):
        return None, None

    price_frame = ohlcv.copy()
    price_frame["Date"] = pd.to_datetime(price_frame["Date"], errors="coerce").dt.normalize()
    price_frame = price_frame.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if price_frame.empty:
        return None, None

    target = pd.Timestamp(target_date).normalize()
    dates = pd.DatetimeIndex(price_frame["Date"])
    exact_matches = price_frame.loc[dates == target]
    if not exact_matches.empty:
        row = exact_matches.iloc[0]
        close_value = pd.to_numeric(pd.Series([row.get("Close")]), errors="coerce").iloc[0]
        if pd.notna(close_value):
            return pd.Timestamp(row["Date"]).normalize(), float(close_value)

    if prefer == "previous":
        candidates = price_frame.loc[price_frame["Date"] <= target]
        if candidates.empty:
            candidates = price_frame.loc[price_frame["Date"] >= target]
    else:
        candidates = price_frame.loc[price_frame["Date"] >= target]
        if candidates.empty:
            candidates = price_frame.loc[price_frame["Date"] <= target]

    if candidates.empty:
        return None, None

    row = candidates.iloc[-1] if prefer == "previous" and not candidates.empty else candidates.iloc[0]
    close_value = pd.to_numeric(pd.Series([row.get("Close")]), errors="coerce").iloc[0]
    if pd.isna(close_value):
        return None, None
    return pd.Timestamp(row["Date"]).normalize(), float(close_value)


def _build_similar_trade_ohlcv_chart_data(nearest_trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if nearest_trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    trades = nearest_trades.copy()
    trades["Entry Date"] = pd.to_datetime(trades.get("Entry Date"), errors="coerce")
    trades["Exit Date"] = pd.to_datetime(trades.get("Exit Date"), errors="coerce")
    trades = trades.dropna(subset=["Entry Date", "Exit Date", "Symbol"]).copy()
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    symbol = str(trades["Symbol"].iloc[0]).strip().upper()
    start_date = (trades["Entry Date"].min() - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    end_date = (trades["Exit Date"].max() + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    ohlcv = _load_symbol_ohlcv(symbol, start_date, end_date)
    if ohlcv.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    marker_rows: list[dict[str, Any]] = []
    line_rows: list[dict[str, Any]] = []
    for _, trade in trades.iterrows():
        side = str(trade.get("Side") or "").strip().title() or "Long"
        signed_trade_return = pd.to_numeric(pd.Series([trade.get("Signed Trade Return")]), errors="coerce").iloc[0]
        entry_date = pd.Timestamp(trade["Entry Date"]).normalize()
        exit_date = pd.Timestamp(trade["Exit Date"]).normalize()
        label = _build_trade_label(
            {
                "Side": side,
                "Entry Date": entry_date.strftime("%Y-%m-%d"),
                "Exit Date": exit_date.strftime("%Y-%m-%d"),
            }
        )
        entry_price = pd.to_numeric(pd.Series([trade.get("Entry Price")]), errors="coerce").iloc[0]
        exit_price = pd.to_numeric(pd.Series([trade.get("Exit Price")]), errors="coerce").iloc[0]
        chart_entry_date = entry_date
        chart_exit_date = exit_date
        if pd.isna(entry_price):
            chart_entry_date, entry_price = _resolve_trade_chart_point(ohlcv, entry_date, prefer="next")
        else:
            resolved_entry_date, resolved_entry_price = _resolve_trade_chart_point(ohlcv, entry_date, prefer="next")
            if resolved_entry_date is not None:
                chart_entry_date = resolved_entry_date
            if pd.isna(entry_price) and resolved_entry_price is not None:
                entry_price = resolved_entry_price
        if pd.isna(exit_price):
            chart_exit_date, exit_price = _resolve_trade_chart_point(ohlcv, exit_date, prefer="previous")
        else:
            resolved_exit_date, resolved_exit_price = _resolve_trade_chart_point(ohlcv, exit_date, prefer="previous")
            if resolved_exit_date is not None:
                chart_exit_date = resolved_exit_date
            if pd.isna(exit_price) and resolved_exit_price is not None:
                exit_price = resolved_exit_price
        if chart_entry_date is not None and pd.notna(entry_price):
            marker_rows.append(
                {
                    "Date": chart_entry_date,
                    "Price": float(entry_price),
                    "Marker Type": "Entry",
                    "Side": side,
                    "Trade Label": label,
                    "Signed Trade Return": None if pd.isna(signed_trade_return) else float(signed_trade_return),
                }
            )
        if chart_exit_date is not None and pd.notna(exit_price):
            marker_rows.append(
                {
                    "Date": chart_exit_date,
                    "Price": float(exit_price),
                    "Marker Type": "Exit",
                    "Side": side,
                    "Trade Label": label,
                    "Signed Trade Return": None if pd.isna(signed_trade_return) else float(signed_trade_return),
                }
            )
        if chart_entry_date is not None and chart_exit_date is not None and pd.notna(entry_price) and pd.notna(exit_price):
            line_rows.append(
                {
                    "Entry Date": chart_entry_date,
                    "Exit Date": chart_exit_date,
                    "Entry Price": float(entry_price),
                    "Exit Price": float(exit_price),
                    "Side": side,
                    "Trade Label": label,
                    "Signed Trade Return": None if pd.isna(signed_trade_return) else float(signed_trade_return),
                }
            )
    return ohlcv, pd.DataFrame(marker_rows), pd.DataFrame(line_rows)


def _render_trade_event_expanders(nearest_trades: pd.DataFrame) -> None:
    if nearest_trades.empty:
        return
    trades = _prepare_trade_focus_table(nearest_trades).copy()
    if trades.empty:
        return

    st.markdown('<div class="section-title">Trade Event Details</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Expand a trade to load only its entry-date and exit-date stock rows and news.</div>',
        unsafe_allow_html=True,
    )

    for idx, row in trades.reset_index(drop=True).iterrows():
        symbol = str(row.get("Symbol") or "").strip().upper()
        entry_date = pd.to_datetime(row.get("Entry Date"), errors="coerce")
        exit_date = pd.to_datetime(row.get("Exit Date"), errors="coerce")
        if not symbol or pd.isna(entry_date) or pd.isna(exit_date):
            continue
        signed_return = pd.to_numeric(pd.Series([row.get("Signed Trade Return")]), errors="coerce").iloc[0]
        return_label = f" | {float(signed_return):.2f}%" if pd.notna(signed_return) else ""
        label = (
            f"{idx + 1}. {symbol} {str(row.get('Side') or '').strip().title() or 'Trade'} "
            f"{pd.Timestamp(entry_date).strftime('%Y-%m-%d')} to {pd.Timestamp(exit_date).strftime('%Y-%m-%d')}{return_label}"
        )
        with st.expander(label, expanded=False):
            detail_key = (
                f"similar_trade_detail::{symbol}::"
                f"{pd.Timestamp(entry_date).strftime('%Y-%m-%d')}::"
                f"{pd.Timestamp(exit_date).strftime('%Y-%m-%d')}::{idx}"
            )
            load_clicked = st.button("Load Entry/Exit Details", key=f"{detail_key}::button")
            if load_clicked or detail_key in st.session_state:
                if detail_key not in st.session_state:
                    with st.spinner(f"Loading {symbol} entry/exit details..."):
                        price_rows, news_rows = _load_trade_event_details(
                            symbol,
                            pd.Timestamp(entry_date).strftime("%Y-%m-%d"),
                            pd.Timestamp(exit_date).strftime("%Y-%m-%d"),
                        )
                    st.session_state[detail_key] = {
                        "price_rows": price_rows,
                        "news_rows": news_rows,
                    }
                else:
                    cached = st.session_state.get(detail_key) or {}
                    price_rows = cached.get("price_rows", pd.DataFrame())
                    news_rows = cached.get("news_rows", pd.DataFrame())
            else:
                st.caption("No data fetched yet. Click the button above to retrieve only this trade's entry/exit dates.")
                continue

            price_rows = st.session_state[detail_key]["price_rows"]
            news_rows = st.session_state[detail_key]["news_rows"]

            if price_rows.empty:
                st.info("No exact-date adjusted stock rows were found for the entry/exit dates.")
            else:
                st.markdown("**Stock Rows**")
                st.dataframe(price_rows, use_container_width=True, hide_index=True)

            if news_rows.empty:
                st.info("No news was found on the exact entry or exit date.")
            else:
                st.markdown("**News On Entry/Exit Dates Only**")
                display_news = news_rows.copy()
                column_config = {}
                if "URL" in display_news.columns:
                    column_config["URL"] = st.column_config.LinkColumn("URL")
                st.dataframe(display_news, use_container_width=True, hide_index=True, column_config=column_config)


def _render_similar_trade_ohlcv_chart(
    symbol: str,
    candle_df: pd.DataFrame,
    trade_table_df: pd.DataFrame,
    marker_df: pd.DataFrame,
    line_df: pd.DataFrame,
    *,
    selected_trade_label: str = "",
) -> None:
    if candle_df.empty:
        return
    labels = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in candle_df["Date"].tolist()]
    opens = [None if pd.isna(v) else float(v) for v in candle_df["Open"].tolist()]
    highs = [None if pd.isna(v) else float(v) for v in candle_df["High"].tolist()]
    lows = [None if pd.isna(v) else float(v) for v in candle_df["Low"].tolist()]
    closes = [None if pd.isna(v) else float(v) for v in candle_df["Close"].tolist()]
    volumes = [None if pd.isna(v) else float(v) for v in candle_df["Volume"].tolist()]

    entry_markers = []
    exit_markers = []
    if not marker_df.empty:
        for _, row in marker_df.iterrows():
            payload = {
                "x": pd.Timestamp(row["Date"]).strftime("%Y-%m-%d"),
                "y": None if pd.isna(row.get("Price")) else float(row["Price"]),
                "type": f"{row.get('Side', '')} {row.get('Marker Type', '')}".strip(),
                "details": [
                    row.get("Trade Label", ""),
                    f"Signed Return: {float(row['Signed Trade Return']):.2f}%"
                    if pd.notna(pd.to_numeric(pd.Series([row.get('Signed Trade Return')]), errors='coerce').iloc[0])
                    else "",
                ],
            }
            if str(row.get("Marker Type") or "").strip().lower() == "entry":
                entry_markers.append(payload)
            else:
                exit_markers.append(payload)

    trade_lines = []
    selected_trade = str(selected_trade_label or "").strip()
    if not line_df.empty:
        for _, row in line_df.iterrows():
            trade_label = str(row.get("Trade Label") or "")
            trade_lines.append(
                {
                    "entry_x": pd.Timestamp(row["Entry Date"]).strftime("%Y-%m-%d"),
                    "entry_y": None if pd.isna(row.get("Entry Price")) else float(row["Entry Price"]),
                    "exit_x": pd.Timestamp(row["Exit Date"]).strftime("%Y-%m-%d"),
                    "exit_y": None if pd.isna(row.get("Exit Price")) else float(row["Exit Price"]),
                    "side": str(row.get("Side") or ""),
                    "label": trade_label,
                    "ret_pct": (
                        f"{float(row['Signed Trade Return']):.2f}%"
                        if pd.notna(pd.to_numeric(pd.Series([row.get('Signed Trade Return')]), errors='coerce').iloc[0])
                        else ""
                    ),
                }
            )

    table_columns = [
        str(column)
        for column in trade_table_df.columns.tolist()
        if not str(column).startswith("__")
    ]
    table_rows: list[dict[str, Any]] = []
    if not trade_table_df.empty:
        for _, row in trade_table_df.iterrows():
            trade_label = str(row.get("__trade_label") or _build_trade_label(row))
            entry_date = pd.to_datetime(row.get("Entry Date"), errors="coerce")
            exit_date = pd.to_datetime(row.get("Exit Date"), errors="coerce")
            payload = {
                "__trade_label": trade_label,
                "__entry_date": entry_date.strftime("%Y-%m-%d") if pd.notna(entry_date) else "",
                "__exit_date": exit_date.strftime("%Y-%m-%d") if pd.notna(exit_date) else "",
            }
            for column in table_columns:
                payload[column] = _format_trade_table_cell(column, row.get(column))
            table_rows.append(payload)

    if not selected_trade and table_rows:
        selected_trade = str(table_rows[0]["__trade_label"])

    component_height = 720 + min(len(table_rows), 12) * 34

    html = f"""
    <style>
      .trade-explorer-wrap {{
        background: rgba(255,255,255,0.95);
        border: 1px solid rgba(20, 44, 29, 0.10);
        border-radius: 20px;
        overflow: hidden;
      }}
      .trade-toolbar {{
        margin: 8px 0 6px;
        padding: 0 8px;
        display:flex;
        gap:8px;
        align-items:center;
      }}
      .trade-table-wrap {{
        border-top: 1px solid rgba(20, 44, 29, 0.08);
        max-height: 360px;
        overflow-y: auto;
        overflow-x: auto;
        background: #ffffff;
      }}
      .trade-table {{
        width: 100%;
        border-collapse: collapse;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-size: 13px;
      }}
      .trade-table thead th {{
        position: sticky;
        top: 0;
        z-index: 2;
        background: #f7faf8;
        color: #335142;
        text-align: left;
        padding: 10px 12px;
        border-bottom: 1px solid rgba(20, 44, 29, 0.10);
        white-space: nowrap;
      }}
      .trade-table tbody td {{
        padding: 10px 12px;
        border-bottom: 1px solid rgba(20, 44, 29, 0.07);
        white-space: nowrap;
        color: #17231c;
      }}
      .trade-table tbody tr {{
        cursor: pointer;
        transition: background 120ms ease;
      }}
      .trade-table tbody tr:hover {{
        background: rgba(0, 200, 5, 0.05);
      }}
      .trade-table tbody tr.is-active {{
        background: rgba(0, 200, 5, 0.10);
      }}
      .trade-table tbody tr.is-active td {{
        font-weight: 600;
      }}
    </style>
    <div class="trade-explorer-wrap">
    <div style="margin: 8px 0 6px; display:flex; gap:8px; align-items:center;">
      <label for="priceChartType" style="font-size:0.86rem;color:#425062;">Price Type</label>
      <select id="priceChartType" style="border:1px solid #c7d2df;border-radius:6px;padding:4px 8px;background:#fff;">
        <option value="candlestick" selected>Candlestick</option>
        <option value="ohlc">OHLC</option>
        <option value="line">Line (Close)</option>
      </select>
    </div>
    <div id="ohlcStockChart" style="width:100%;height:560px;"></div>
    <div id="chartError" style="display:none;margin-top:10px;padding:10px;border:1px solid #f3c4c4;background:#fff1f1;color:#9e2020;border-radius:8px;"></div>
    <div class="trade-table-wrap">
      <table class="trade-table" id="tradeExplorerTable"></table>
    </div>
    </div>
    <script>
      const labels = {json.dumps(labels)};
      const opens = {json.dumps(opens)};
      const highs = {json.dumps(highs)};
      const lows = {json.dumps(lows)};
      const closes = {json.dumps(closes)};
      const volumes = {json.dumps(volumes)};
      const entryMarkers = {json.dumps(entry_markers)};
      const exitMarkers = {json.dumps(exit_markers)};
      const tradeLines = {json.dumps(trade_lines)};
      const tableColumns = {json.dumps(table_columns)};
      const tableRows = {json.dumps(table_rows)};
      const selectedTradeLabel = {json.dumps(selected_trade)};
      let activeTradeLabel = selectedTradeLabel || "";
      let chartRef = null;
      const tradeSeriesByLabel = {{}};

      function toTs(dateStr) {{
        return Date.parse(String(dateStr || "") + "T00:00:00Z");
      }}

      function htmlEscape(value) {{
        return String(value == null ? "" : value)
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#39;");
      }}

      function getTradeRow(label) {{
        return (tableRows || []).find((row) => String(row.__trade_label || "") === String(label || ""));
      }}

      const ohlc = [];
      const closeLine = [];
      const volume = [];
      for (let i = 0; i < labels.length; i += 1) {{
        const ts = toTs(labels[i]);
        const o = opens[i];
        const h = highs[i];
        const l = lows[i];
        const c = closes[i];
        if (!Number.isFinite(ts) || o == null || h == null || l == null || c == null) continue;
        ohlc.push([ts, Number(o), Number(h), Number(l), Number(c)]);
        closeLine.push([ts, Number(c)]);
        if (volumes[i] != null) volume.push([ts, Number(volumes[i])]);
      }}

      const entryPts = entryMarkers
        .map((m) => ({{ x: toTs(m.x), y: Number(m.y), custom: {{ type: m.type, details: m.details || [] }} }}))
        .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));

      const exitPts = exitMarkers
        .map((m) => ({{ x: toTs(m.x), y: Number(m.y), custom: {{ type: m.type, details: m.details || [] }} }}))
        .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));

      const tradeLineSeries = [];
      for (const t of tradeLines || []) {{
        const x1 = toTs(t.entry_x);
        const y1 = Number(t.entry_y);
        const x2 = toTs(t.exit_x);
        const y2 = Number(t.exit_y);
        if (!Number.isFinite(x1) || !Number.isFinite(y1) || !Number.isFinite(x2) || !Number.isFinite(y2)) continue;
        const isLong = String(t.side || "").toLowerCase() === "long";
        const isSelected = activeTradeLabel && String(t.label || "") === activeTradeLabel;
        tradeLineSeries.push({{
          type: "line",
          name: "Trade Line",
          id: "trade-line-" + String(t.label || "").replaceAll(/[^a-zA-Z0-9_-]/g, "_"),
          yAxis: 0,
          data: [
            {{ x: x1, y: y1 }},
            {{ x: x2, y: y2, custom: {{ type: (t.side || "Trade") + " Trade", details: [t.ret_pct ? ("Return: " + t.ret_pct) : ""].filter(Boolean) }} }},
          ],
          color: isLong ? "rgba(22, 163, 74, 0.8)" : "rgba(220, 38, 38, 0.8)",
          lineWidth: isSelected ? 4.0 : 2.4,
          marker: {{ enabled: false }},
          enableMouseTracking: true,
          showInLegend: false,
          zIndex: isSelected ? 8 : 6,
          custom: {{
            tradeLabel: String(t.label || ""),
            entryX: x1,
            exitX: x2
          }}
        }});
      }}

      function updateActiveTableRow() {{
        const rows = document.querySelectorAll("#tradeExplorerTable tbody tr[data-trade-label]");
        rows.forEach((rowEl) => {{
          const isActive = String(rowEl.dataset.tradeLabel || "") === String(activeTradeLabel || "");
          rowEl.classList.toggle("is-active", isActive);
        }});
      }}

      function zoomToTrade(label) {{
        const tradeRow = getTradeRow(label);
        if (!tradeRow || !chartRef || !chartRef.xAxis || !chartRef.xAxis[0]) return;
        const startTs = toTs(tradeRow.__entry_date);
        const endTs = toTs(tradeRow.__exit_date);
        if (!Number.isFinite(startTs) || !Number.isFinite(endTs)) return;
        const padMs = 30 * 24 * 60 * 60 * 1000;
        chartRef.xAxis[0].setExtremes(startTs - padMs, endTs + padMs, true, false);
      }}

      function updateTradeLineStyles() {{
        if (!chartRef || !chartRef.series) return;
        for (const series of chartRef.series) {{
          const tradeLabel = series && series.options && series.options.custom ? series.options.custom.tradeLabel : "";
          if (!tradeLabel) continue;
          const isActive = String(tradeLabel || "") === String(activeTradeLabel || "");
          series.update({{
            lineWidth: isActive ? 4.0 : 2.4,
            zIndex: isActive ? 8 : 6
          }}, false);
        }}
        chartRef.redraw();
      }}

      function focusTrade(label) {{
        if (!label) return;
        activeTradeLabel = String(label);
        updateActiveTableRow();
        updateTradeLineStyles();
        zoomToTrade(activeTradeLabel);
      }}

      function renderTradeTable() {{
        const tableEl = document.getElementById("tradeExplorerTable");
        if (!tableEl) return;
        const headerHtml = "<thead><tr>" + tableColumns.map((column) => "<th>" + htmlEscape(column) + "</th>").join("") + "</tr></thead>";
        const bodyHtml = "<tbody>" + tableRows.map((row) => {{
          const cells = tableColumns.map((column) => "<td>" + htmlEscape(row[column]) + "</td>").join("");
          return "<tr data-trade-label=\\"" + htmlEscape(row.__trade_label) + "\\">" + cells + "</tr>";
        }}).join("") + "</tbody>";
        tableEl.innerHTML = headerHtml + bodyHtml;
        tableEl.querySelectorAll("tbody tr[data-trade-label]").forEach((rowEl) => {{
          rowEl.addEventListener("click", () => {{
            const nextLabel = String(rowEl.dataset.tradeLabel || "");
            focusTrade(nextLabel);
          }});
        }});
        if (!activeTradeLabel && tableRows.length > 0) {{
          activeTradeLabel = String(tableRows[0].__trade_label || "");
        }}
        updateActiveTableRow();
      }}

      function showChartError(msg) {{
        const el = document.getElementById("chartError");
        if (!el) return;
        el.style.display = "block";
        el.textContent = msg;
      }}

      function loadScript(src) {{
        return new Promise((resolve, reject) => {{
          const s = document.createElement("script");
          s.src = src;
          s.async = true;
          s.onload = () => resolve(true);
          s.onerror = () => reject(new Error("Failed to load " + src));
          document.head.appendChild(s);
        }});
      }}

      async function ensureHighchartsLoaded() {{
        if (typeof Highcharts !== "undefined" && typeof Highcharts.stockChart === "function") return true;
        const urls = [
          "https://code.highcharts.com/stock/highstock.js",
          "https://cdn.jsdelivr.net/npm/highcharts@12.4.0/highstock.js",
          "https://unpkg.com/highcharts@12.4.0/highstock.js"
        ];
        for (const u of urls) {{
          try {{
            await loadScript(u);
            if (typeof Highcharts !== "undefined" && typeof Highcharts.stockChart === "function") return true;
          }} catch (_) {{}}
        }}
        return false;
      }}

      function renderChart() {{
        try {{
          chartRef = Highcharts.stockChart("ohlcStockChart", {{
            chart: {{ backgroundColor: "#fbfdff" }},
            rangeSelector: {{
              selected: 4,
              buttons: [
                {{ type: "month", count: 1, text: "1M" }},
                {{ type: "month", count: 6, text: "6M" }},
                {{ type: "ytd", text: "YTD" }},
                {{ type: "year", count: 1, text: "1Y" }},
                {{ type: "year", count: 5, text: "5Y" }},
                {{ type: "all", text: "Max" }}
              ]
            }},
            navigator: {{ enabled: true }},
            scrollbar: {{ enabled: true }},
            title: {{ text: "{str(symbol).strip().upper()} OHLC" }},
            yAxis: [
              {{ title: {{ text: "Price" }}, height: "72%", resize: {{ enabled: true }} }},
              {{ title: {{ text: "Volume" }}, top: "76%", height: "24%", offset: 0 }}
            ],
            tooltip: {{ split: true, xDateFormat: "%Y-%m-%d" }},
            plotOptions: {{
              series: {{ dataGrouping: {{ enabled: false }} }},
              scatter: {{
                tooltip: {{
                  pointFormatter: function () {{
                    const details = (this.custom && this.custom.details) ? this.custom.details.join(" | ") : "";
                    const t = (this.custom && this.custom.type) ? this.custom.type : this.series.name;
                    return "<span style=\\"color:" + this.color + "\\">●</span> " + t +
                      ": <b>" + Highcharts.numberFormat(this.y, 4) + "</b>" + (details ? " | " + details : "") + "<br/>";
                  }}
                }}
              }}
            }},
            series: [
              {{
                type: "candlestick",
                name: "OHLC",
                id: "price",
                data: ohlc,
                upColor: "#0f9d58",
                color: "#d44d3a",
                lineColor: "#d44d3a",
                upLineColor: "#0f9d58"
              }},
              {{
                type: "scatter",
                name: "Entries",
                yAxis: 0,
                data: entryPts,
                color: "#15803d",
                marker: {{ symbol: "triangle", radius: 6 }}
              }},
              {{
                type: "scatter",
                name: "Exits",
                yAxis: 0,
                data: exitPts,
                color: "#b91c1c",
                marker: {{ symbol: "diamond", radius: 6 }}
              }},
              {{
                type: "column",
                name: "Volume",
                yAxis: 1,
                data: volume,
                color: "rgba(13,106,109,0.45)"
              }},
              ...tradeLineSeries
            ]
          }});

          const typeSelect = document.getElementById("priceChartType");
          if (typeSelect) {{
            typeSelect.addEventListener("change", function () {{
              const nextType = String(typeSelect.value || "candlestick");
              const priceSeries = chartRef.get("price");
              if (!priceSeries) return;
              if (nextType === "line") {{
                priceSeries.update({{
                  type: "line",
                  name: "Close",
                  data: closeLine,
                  color: "#3366cc",
                  lineWidth: 1.5
                }}, true);
              }} else if (nextType === "ohlc") {{
                priceSeries.update({{
                  type: "ohlc",
                  name: "OHLC",
                  data: ohlc,
                  color: "#224f8f",
                  lineWidth: 1.2
                }}, true);
              }} else {{
                priceSeries.update({{
                  type: "candlestick",
                  name: "OHLC",
                  data: ohlc,
                  upColor: "#0f9d58",
                  color: "#d44d3a",
                  lineColor: "#d44d3a",
                  upLineColor: "#0f9d58"
                }}, true);
              }}
            }});
          }}
          renderTradeTable();
          updateTradeLineStyles();
          if (activeTradeLabel) {{
            zoomToTrade(activeTradeLabel);
          }}
        }} catch (e) {{
          showChartError("Chart render failed: " + (e && e.message ? e.message : "unknown error"));
        }}
      }}

      ensureHighchartsLoaded().then((ok) => {{
        if (!ok) {{
          showChartError("Highcharts failed to load from all CDNs. Network or CSP is blocking external scripts.");
          return;
        }}
        renderChart();
      }});
    </script>
    """
    components.html(html, height=component_height, scrolling=False)


def _style_feature_attribution(frame: pd.DataFrame) -> pd.io.formats.style.Styler | pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "Feature Name",
                "Current Value",
                "Attribution Percentage",
                "Direction",
            ]
        )

    display = frame.copy()
    importance_values = display.get("tree_importance_pct")
    if isinstance(importance_values, pd.Series):
        attribution_percentage = pd.to_numeric(importance_values, errors="coerce").fillna(0.0)
    else:
        scalar_value = pd.to_numeric(pd.Series([importance_values]), errors="coerce").fillna(0.0).iloc[0]
        attribution_percentage = pd.Series([scalar_value] * len(display), index=display.index, dtype=float)
    display["Attribution Percentage"] = attribution_percentage
    display["Pretty Feature Name"] = display.get("feature", "").apply(
        _get_pretty_feature_name
    )
    display["Direction"] = display.get("direction_label", "").astype(str).str.strip().str.title()
    display = display.rename(
        columns={
            "Pretty Feature Name": "Feature Name",
            "current_value": "Current Value",
        }
    )[
        [
            "Feature Name",
            "Current Value",
            "Attribution Percentage",
            "Direction",
        ]
    ]

    def _format_feature_value(value: object) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return ""
        numeric = float(numeric)
        magnitude = abs(numeric)
        if magnitude >= 1000:
            return f"{numeric:,.2f}"
        if magnitude >= 1:
            return f"{numeric:,.2f}"
        if magnitude >= 0.01:
            return f"{numeric:,.4f}"
        if magnitude == 0.0:
            return "0.0000"
        return f"{numeric:,.6f}"

    def _row_styles(row: pd.Series) -> list[str]:
        direction_value = str(row["Direction"] or "").strip().lower()
        if not direction_value:
            highlight = ""
        elif direction_value == "long":
            highlight = "color: #216e39; font-weight: 700;"
        else:
            highlight = "color: #b42318; font-weight: 700;"
        return [
            highlight if column in {"Attribution Percentage", "Direction"} else ""
            for column in row.index
        ]

    return (
        display.style.hide(axis="index")
        .format(
            {
                "Current Value": _format_feature_value,
                "Attribution Percentage": "{:,.2f}%",
            }
        )
        .apply(_row_styles, axis=1)
    )


def _get_pretty_feature_name(value: object) -> str:
    feature_name = str(value or "").strip()
    if not feature_name:
        return ""
    try:
        bootstrap_django()
        from pipeline.feature_presentation import get_feature_definition

        return str(get_feature_definition(feature_name).display_name or feature_name)
    except Exception:
        return feature_name


st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Past Setups</div>
      <h1>Find Similar Stock Setups</h1>
      <p>
        Enter a stock symbol and date, and we will look for past setups that looked the most like it.
        Then we show two views: what similar setups did in the past, and what our model thinks is most likely next.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Search")
    st.markdown(
        """
        <div class="sidebar-note">
          We use the latest full data available on or before your selected date,
          then compare that setup with the same stock's past trades to find the closest matches.
        </div>
        """,
        unsafe_allow_html=True,
    )
    query_params = st.query_params
    default_symbol = str(query_params.get("symbol", "AAPL")).strip().upper() or "AAPL"
    symbol = st.text_input("Symbol", value=default_symbol).strip().upper() or "AAPL"
    default_query_date = date.today() - timedelta(days=1)
    query_date = st.date_input("Query Date", value=default_query_date, max_value=date.today())

    st.subheader("Settings")
    top_k = st.slider("How many past trades to compare", min_value=3, max_value=25, value=10)
    label_k_values = st.multiselect("Trade windows", options=[1, 2, 3, 4, 6, 8, 12], default=[1, 2, 4, 8])
    download_missing_prices = True


current_query_signature = {
    "symbol": str(symbol),
    "as_of_date": pd.Timestamp(query_date).strftime("%Y-%m-%d"),
    "top_k": int(top_k),
    "label_k_values": tuple(int(value) for value in label_k_values),
    "download_missing_prices": bool(download_missing_prices),
}

if not label_k_values:
    st.warning("Choose at least one trade window before running the search.")
else:
    st.query_params["symbol"] = symbol
    st.query_params["date"] = pd.Timestamp(query_date).strftime("%Y-%m-%d")
    st.query_params["top_k"] = str(int(top_k))

    stored_signature = st.session_state.get("optimal_trade_query_signature")
    result = st.session_state.get("optimal_trade_result") if stored_signature == current_query_signature else None
    needs_lookup = result is None

    if needs_lookup:
        query = OptimalTradeQuery(
            symbol=symbol,
            as_of_date=pd.Timestamp(query_date).strftime("%Y-%m-%d"),
            query_lookback_years=0,
            reference_symbols=(symbol,),
            reference_start_date="1900-01-01",
            reference_end_date=None,
            top_k=int(top_k),
            label_freq="YE",
            label_k_values=tuple(int(value) for value in label_k_values),
            download_missing_prices=bool(download_missing_prices),
        )

        try:
            with st.spinner("Loading similar past setups..."):
                result = find_nearest_optimal_trades(query)
        except Exception as exc:
            st.error(str(exc))
            st.session_state.pop("optimal_trade_result", None)
            st.session_state.pop("optimal_trade_query_signature", None)
            result = None
        else:
            st.session_state["optimal_trade_result"] = result
            st.session_state["optimal_trade_query_signature"] = dict(current_query_signature)
    if result is not None:
        summary = result.query_summary.iloc[0]
        st.markdown(
            f"""
            <div class="results-ribbon">
              <div class="results-chip"><strong>Ticker</strong> {summary["symbol"]}</div>
              <div class="results-chip"><strong>Date Used</strong> {summary["as_of_date"]}</div>
              <div class="results-chip"><strong>Past Trades Compared</strong> {int(top_k)}</div>
              <div class="results-chip"><strong>Trade Windows</strong> {", ".join(str(value) for value in label_k_values)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        metric_cols = st.columns(4)
        metric_cols[0].metric("Ticker", str(summary["symbol"]))
        metric_cols[1].metric("Date Used", str(summary["as_of_date"]))
        metric_cols[2].metric("Match Score", f"{float(summary['ae_familiarity']):.3f}")
        metric_cols[3].metric("Past Trades Found", f"{int(summary['reference_trade_count'])}")

        prediction_cards = _format_prediction_metric(result.model_predictions)
        if prediction_cards:
            st.markdown('<div class="section-title">Oracle View</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-copy">This is the model\'s simple read on what may happen next. We keep it plain-English on purpose.</div>',
                unsafe_allow_html=True,
            )
            prediction_cols = st.columns(len(prediction_cards))
            for idx, (label, value, help_text) in enumerate(prediction_cards):
                prediction_cols[idx].metric(label, value, help_text)

        similar_trade_summary_cards = _format_similar_trade_summary(result.nearest_trades)
        if similar_trade_summary_cards:
            st.markdown('<div class="section-title">What Happened In Similar Setups</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-copy">This is the historical view. It shows what the closest past setups actually did. Short trades are shown as negative returns.</div>',
                unsafe_allow_html=True,
            )
            summary_cols = st.columns(len(similar_trade_summary_cards))
            for idx, (label, value, help_text) in enumerate(similar_trade_summary_cards):
                summary_cols[idx].metric(label, value, help_text)

            payoff_table, payoff_snapshot = _build_options_payoff_table(
                symbol=str(summary["symbol"]),
                as_of_date=summary["as_of_date"],
                nearest_trades=result.nearest_trades,
                model_predictions=result.model_predictions,
            )
            if payoff_snapshot and not payoff_table.empty:
                st.markdown('<div class="section-title">Options Payoff</div>', unsafe_allow_html=True)
                display_payoff_table = payoff_table.drop(columns=["__distance_to_money"], errors="ignore")
                move_std = pd.to_numeric(pd.Series([_compute_similar_trade_summary_stats(result.nearest_trades).get("signed_return_std_pct")]), errors="coerce").iloc[0]
                hold_std = pd.to_numeric(pd.Series([_compute_similar_trade_summary_stats(result.nearest_trades).get("hold_days_std")]), errors="coerce").iloc[0]
                if pd.notna(move_std) or pd.notna(hold_std):
                    move_text = f"move std dev {float(move_std):.2f}%" if pd.notna(move_std) else ""
                    hold_text = f"hold-time std dev {float(hold_std):.1f}d" if pd.notna(hold_std) else ""
                    joiner = " | " if move_text and hold_text else ""
                    st.caption(f"Expected payoff now uses the full similar-trade distribution, weighted by match score. {move_text}{joiner}{hold_text}")
                st.dataframe(
                    display_payoff_table,
                    use_container_width=True,
                    hide_index=True,
                    column_config=_options_payoff_column_config(),
                )

        trade_focus_table = _prepare_trade_focus_table(result.nearest_trades)
        trade_block = st.container()
        with trade_block:
            st.markdown('<div class="section-title">Trade Explorer</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-copy">Click any trade row below the chart and it will zoom around that entry and exit window.</div>',
                unsafe_allow_html=True,
            )
            chart_ohlcv_df, chart_marker_df, chart_line_df = _build_similar_trade_ohlcv_chart_data(result.nearest_trades)
            if not chart_ohlcv_df.empty:
                _render_similar_trade_ohlcv_chart(
                    summary["symbol"],
                    chart_ohlcv_df,
                    trade_focus_table,
                    chart_marker_df,
                    chart_line_df,
                )
            else:
                st.dataframe(
                    trade_focus_table,
                    use_container_width=True,
                    hide_index=True,
                    height=420,
                    column_config=_nearest_trade_column_config(),
                )
            _render_trade_event_expanders(result.nearest_trades)

        with st.expander("Search Metadata", expanded=False):
            st.markdown(
                '<div class="section-copy">Technical details for debugging the search.</div>',
                unsafe_allow_html=True,
            )
            st.json(result.metadata)
    else:
        st.info(
            "Pick a stock and date, then run the search to see what the most similar past setups looked like."
        )
