#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from app.trading_app_v2_runtime import (
    alpaca_client_from_env,
    build_alpaca_equity_orders,
    build_latest_equity_leaderboard,
    build_llm_ranked_option_orders,
    build_ranked_alpaca_option_orders,
    build_robinhood_option_orders,
    build_score_date_option_ml_ranking_table,
    build_symbol_score_table,
    default_paths,
    load_equity_artifacts,
    save_live_artifacts,
    select_optionable_leaderboard,
    option_contract_quantity,
    write_streamlit_leaderboard_app,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-market-cap", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    load_dotenv(repo / ".env", override=False)
    paths = default_paths(repo)
    strategy_scores = load_equity_artifacts(paths.equity_artifact_dir)["strategy_scores"]
    leaderboard = build_latest_equity_leaderboard(strategy_scores, top_k=args.top_k, min_long_score=0.5)
    score_date = pd.to_datetime(leaderboard["score_date"], errors="coerce").max().strftime("%Y-%m-%d")
    option_leaderboard = select_optionable_leaderboard(leaderboard, score_date=score_date, top_k=args.top_k)
    option_rankings = build_score_date_option_ml_ranking_table(
        paths.option_artifact_dir,
        leaderboard=option_leaderboard,
        score_date=score_date,
        symbols=option_leaderboard["symbol"].tolist(),
        target_dte=90,
        min_market_cap=args.min_market_cap,
        start_date="1900-01-01",
        equity_family_scores=strategy_scores,
    )

    orders: dict[str, pd.DataFrame] = {}
    orders["alpaca_equity_paper"] = build_alpaca_equity_orders(
        leaderboard=leaderboard,
        account_prefix="EQUITY",
    )
    orders["alpaca_option_paper"] = build_ranked_alpaca_option_orders(
        option_rankings=option_rankings,
        decisions=option_leaderboard[["symbol", "direction"]],
        account_prefix="OPTION",
        strategy_allocation=100_000.0,
    )
    llm_orders, llm_reviews = build_llm_ranked_option_orders(
        leaderboard=option_leaderboard,
        option_rankings=option_rankings,
        account_prefix="LLM",
        top_k=args.top_k,
        as_of_date=score_date,
        strategy_allocation=100_000.0,
    )
    orders["alpaca_llm_paper"] = llm_orders

    robinhood_targets = option_rankings.loc[
        option_rankings["selected_by_option_ensemble"].astype(bool)
    ].merge(option_leaderboard[["symbol", "direction"]], on="symbol", how="inner")
    robinhood_targets = robinhood_targets.loc[
        ((robinhood_targets["direction"] == "long") & (robinhood_targets["option_type"] == "call"))
        | ((robinhood_targets["direction"] == "short") & (robinhood_targets["option_type"] == "put"))
    ].copy()
    robinhood_targets["expiry_date"] = robinhood_targets["expiration"]
    robinhood_targets["strike_price"] = robinhood_targets["strike"]
    robinhood_targets["combined_score"] = robinhood_targets["pred_meta_stack_rank"]
    robinhood_targets["bid_price"] = pd.to_numeric(robinhood_targets.get("bid"), errors="coerce")
    robinhood_targets["ask_price"] = pd.to_numeric(robinhood_targets.get("ask"), errors="coerce")
    robinhood_targets["quantity"] = robinhood_targets["bid_price"].map(
        lambda bid: option_contract_quantity(
            account_value=100_000.0,
            option_price=(float(bid) * 0.10) if pd.notna(bid) and float(bid) > 0 else None,
            max_underlyings=args.top_k,
        )
    )
    robinhood_plan = build_robinhood_option_orders(
        target_contracts=robinhood_targets,
        gate_discount_pct=90.0,
        current_option_positions=pd.DataFrame(),
        pending_option_orders=pd.DataFrame(),
    )
    orders["robinhood_option_real"] = robinhood_plan["actionable_orders"]

    output_dir = args.output_dir.resolve()
    symbol_scores = build_symbol_score_table(strategy_scores, leaderboard)
    save_live_artifacts(
        live_dir=output_dir,
        leaderboard=leaderboard,
        symbol_scores=symbol_scores,
        option_ml_rankings=option_rankings,
        orders=orders,
    )
    llm_reviews.to_csv(output_dir / "trading_agents_reviews.csv", index=False)
    write_streamlit_leaderboard_app(
        live_dir=output_dir,
        leaderboard=leaderboard,
        symbol_scores=symbol_scores,
        option_ml_rankings=option_rankings,
        orders=orders,
    )
    print(
        {
            "score_date": score_date,
            "equity_targets": int(leaderboard["selected"].sum()),
            "option_targets": len(option_leaderboard),
            "orders": {name: len(frame) for name, frame in orders.items()},
            "output_dir": str(output_dir),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
