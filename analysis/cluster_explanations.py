from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from .market_state import feature_family_map_from_columns


FAMILY_PHRASES = {
    "prices_div_adj": "strong price momentum",
    "analyst_estimates": "rising earnings revisions",
    "economic_indicators": "positive liquidity environment",
    "treasury_rates": "supportive rate backdrop",
    "income_statement": "fundamental trend support",
    "income_statement_growth": "accelerating growth",
    "key_metrics": "quality metrics improving",
    "ratios": "valuation and efficiency support",
    "model_signals": "internal model rank support",
    "oracle_cluster": "state aligns with profitable oracle pockets",
    "market_situations": "state closely matches known situation clusters",
}


def _cluster_title(side: str, top_families: Sequence[str]) -> str:
    family_set = {str(value) for value in list(top_families or [])}
    side_value = str(side or "long").strip().lower()
    if side_value == "short":
        if {"economic_indicators", "treasury_rates"} & family_set:
            return "Macro Tightening Short Regime"
        if "prices_div_adj" in family_set:
            return "Momentum Breakdown Regime"
        return "Short Opportunity Regime"
    if {"prices_div_adj", "analyst_estimates"} <= family_set:
        return "Earnings Revision Breakout"
    if {"prices_div_adj", "economic_indicators"} <= family_set:
        return "Liquidity-Driven Rally"
    if {"prices_div_adj", "income_statement_growth"} <= family_set:
        return "Momentum Growth Regime"
    if {"income_statement", "income_statement_growth"} & family_set:
        return "Fundamental Expansion Regime"
    return "Long Opportunity Regime"


def build_cluster_feature_explanations(
    *,
    scaled_feature_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    feature_columns: Sequence[str],
    top_n: int = 3,
) -> dict[str, dict[str, Any]]:
    family_map = feature_family_map_from_columns(feature_columns)
    results: dict[str, dict[str, Any]] = {}
    if scaled_feature_df.empty or assignments_df.empty or not family_map:
        return results
    for cluster_id, index_rows in assignments_df.groupby("cluster_id", observed=True):
        cluster_scaled = scaled_feature_df.loc[index_rows.index]
        if cluster_scaled.empty:
            continue
        family_rows: list[dict[str, Any]] = []
        for family, columns in family_map.items():
            usable = [column for column in columns if column in cluster_scaled.columns]
            if not usable:
                continue
            contribution = float(cluster_scaled[usable].abs().mean(axis=1).mean())
            if contribution <= 0.0:
                continue
            family_rows.append(
                {
                    "family": str(family),
                    "score": round(contribution, 6),
                    "phrase": FAMILY_PHRASES.get(str(family), str(family).replace("_", " ")),
                }
            )
        family_rows.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
        top_rows = family_rows[: max(int(top_n), 1)]
        side = str(index_rows["side"].iloc[0] if "side" in index_rows.columns and not index_rows.empty else "long")
        results[str(cluster_id)] = {
            "feature_signature": [str(row["family"]) for row in top_rows],
            "family_rows": top_rows,
            "description": _cluster_title(side, [row["family"] for row in top_rows]),
            "typical_features": [str(row["phrase"]) for row in top_rows],
        }
    return results
