from __future__ import annotations

from datetime import timedelta
import os
from pathlib import Path
import sys
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[2]
repo_root_text = str(REPO_ROOT)
sys.path = [entry for entry in sys.path if entry != repo_root_text]
sys.path.insert(0, repo_root_text)

from app.leaderboard_ui import (
    LEADERBOARD_CSS,
    render_leaderboard_pager,
    render_leaderboard_ribbon,
    render_leaderboard_table,
)
from app.moe_paper_trading import (
    build_alpaca_option_trade_plan,
    build_moe_ranked_scores,
    default_artifact_dir,
    load_moe_paper_artifacts,
    select_alpaca_option_contract,
)
from app.trading_notebook import bootstrap_repo
from trading.alpaca_paper import AlpacaPaperClient


bootstrap_repo(REPO_ROOT)
if os.environ.get("MOE_STREAMLIT_EMBEDDED_PAGE") != "1":
    st.set_page_config(page_title="MoE Paper Trading", page_icon="OT", layout="wide")


def _expert_score_name(column: str) -> str:
    family = str(column).removesuffix("__prob_buy")
    return f"{family.replace('_', ' ').title()} Score"


def _enrich_option_records(
    client: AlpacaPaperClient,
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    contract_cache: dict[str, dict[str, object]] = {}
    for raw_record in records:
        record = dict(raw_record)
        asset_class = str(record.get("asset_class") or "").lower()
        symbol = str(record.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        if asset_class and asset_class not in {"us_option", "option"}:
            continue
        try:
            if symbol not in contract_cache:
                contract_cache[symbol] = client.get_option_contract(symbol)
            contract = contract_cache[symbol]
        except RuntimeError:
            if asset_class not in {"us_option", "option"}:
                continue
            contract = {}
        record["underlying_symbol"] = str(contract.get("underlying_symbol") or "").upper()
        record["option_type"] = contract.get("type")
        record["expiry_date"] = contract.get("expiration_date")
        record["strike_price"] = contract.get("strike_price")
        enriched.append(record)
    return enriched


st.markdown(LEADERBOARD_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="leaderboard-hero">
      <div class="leaderboard-kicker">MoE Paper Trading</div>
      <h1>Ranked Trade Setups</h1>
      <p>
        This board ranks every stock scored by the feature-family mixture of experts.
        The option portfolio is computed from the current raw scores and reconciled with live Alpaca paper option positions and orders.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

artifacts = load_moe_paper_artifacts(default_artifact_dir(REPO_ROOT))
metadata = dict(artifacts.metadata)
scored = artifacts.latest_scored.copy()
scored.index = pd.Index(scored.index.astype(str).str.strip().str.upper(), name="symbol")

top_k_default = min(int(metadata.get("top_k") or 40), max(len(scored), 1))
score_threshold = 0.50
ranked_scores = build_moe_ranked_scores(scored, top_k=top_k_default, threshold=score_threshold)

expert_columns = [column for column in ranked_scores.columns if str(column).endswith("__prob_buy")]
leaderboard = pd.DataFrame(index=ranked_scores.index)
leaderboard["Scored Date"] = str(metadata.get("strategy_date") or "")
leaderboard["Symbol"] = leaderboard.index
leaderboard["Direction"] = "Long"
for expert_column in expert_columns:
    leaderboard[_expert_score_name(str(expert_column))] = pd.to_numeric(
        ranked_scores[expert_column], errors="coerce"
    )
leaderboard["Classifier Score"] = pd.to_numeric(ranked_scores["prob_buy"], errors="coerce")
leaderboard["Combined Score"] = leaderboard["Classifier Score"]
leaderboard["Similar Trades"] = [f"/?symbol={quote_plus(symbol)}" for symbol in leaderboard.index]
leaderboard["Eligible"] = leaderboard["Combined Score"].gt(score_threshold).fillna(False)
leaderboard = leaderboard.sort_values("Combined Score", ascending=False, kind="stable").reset_index(drop=True)
leaderboard.insert(0, "Rank", range(1, len(leaderboard) + 1))

total_rows = int(len(leaderboard))
eligible_count = int(leaderboard["Eligible"].sum())
universe_size = int(metadata.get("universe_size") or total_rows)
inactive_count = int(metadata.get("inactive_symbol_count") or max(universe_size - total_rows, 0))
render_leaderboard_ribbon(
    st,
    as_of_date=metadata.get("strategy_date"),
    universe_size=universe_size,
    scored_symbols=total_rows,
    inactive=inactive_count,
    eligible=eligible_count,
    not_eligible=total_rows - eligible_count,
)

st.markdown(
    """
    <div class="section-card">
      <div class="table-title">Alpaca Option Automation</div>
      <div class="table-copy">Open long calls for the current top MoE names and close option positions that are no longer selected. Targets are recomputed from raw scores every time.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

option_cols = st.columns(4)
top_k = option_cols[0].number_input(
    "Alpaca Top K", min_value=1, max_value=max(total_rows, 1), value=top_k_default, step=1
)
option_bucket_default = str(metadata.get("instrument") or "otm_option")
option_buckets = ["atm_option", "otm_option", "ditm_option"]
option_bucket = option_cols[1].selectbox(
    "Option Bucket",
    options=option_buckets,
    index=option_buckets.index(option_bucket_default) if option_bucket_default in option_buckets else 1,
)
tenor_days = option_cols[2].number_input(
    "Option Tenor (Days)", min_value=7, max_value=365,
    value=int(metadata.get("option_tenor_days") or 60), step=7,
)
max_contracts = option_cols[3].number_input(
    "Max Contracts / Position", min_value=0, max_value=10000, value=0, step=1,
    help="Use 0 to size by budget with no contract-count cap.",
)

env_key_ready = bool(str(os.getenv("ALPACA_PAPER_API_KEY") or "").strip())
env_secret_ready = bool(str(os.getenv("ALPACA_PAPER_API_SECRET") or "").strip())
st.caption(
    "Alpaca paper credentials are loaded from `.env`. "
    f"API key: {'loaded' if env_key_ready else 'missing'} | "
    f"API secret: {'loaded' if env_secret_ready else 'missing'}"
)

plan_cols = st.columns(2)
equity_override = plan_cols[0].number_input(
    "Account Equity Override ($)", min_value=0.0, value=0.0, step=1000.0,
    help="Leave at 0 to use live Alpaca paper account equity.",
)
strategy_allocation = plan_cols[1].number_input(
    "Option Allocation Budget ($)", min_value=0.0, value=100000.0, step=1000.0,
    help="Leave at 0 to use the full account equity.",
)
st.caption("This flow manages Alpaca option positions only. Common-stock positions are not included in the plan.")

build_plan = st.button("Generate Alpaca Target Portfolio", type="primary", use_container_width=True)
if build_plan:
    with st.status("Connecting to Alpaca and generating the option portfolio...", expanded=True) as status:
        try:
            client = AlpacaPaperClient.from_env()
            status.write("Loading Alpaca paper account, option positions, and outstanding orders.")
            account = client.get_account()
            current_positions = _enrich_option_records(client, client.get_positions())
            open_orders = _enrich_option_records(client, client.get_open_orders())
            account_equity = float(equity_override) if float(equity_override) > 0 else float(
                account.get("equity") or account.get("portfolio_value") or 0.0
            )
            allocation = float(strategy_allocation) if float(strategy_allocation) > 0 else account_equity
            current_ranked_scores = build_moe_ranked_scores(
                scored, top_k=int(top_k), threshold=score_threshold
            )
            selected_symbols = current_ranked_scores.loc[current_ranked_scores["selected"]].index.astype(str)
            as_of = pd.Timestamp.today().date()
            target_expiration = as_of + timedelta(days=int(tenor_days))
            expiration_lte = target_expiration + timedelta(days=45)
            option_contracts: dict[str, list[dict[str, object]]] = {}
            selected_contract_symbols: list[str] = []
            status.write(f"Loading active Alpaca call contracts for {len(selected_symbols):,} selected symbols.")
            for symbol in selected_symbols:
                contracts = client.get_option_contracts(
                    symbol,
                    option_type="call",
                    expiration_date_gte=str(as_of),
                    expiration_date_lte=str(expiration_lte),
                )
                option_contracts[symbol] = contracts
                selected_contract = select_alpaca_option_contract(
                    contracts,
                    underlying_price=float(current_ranked_scores.loc[symbol, "close"]),
                    target_expiration=target_expiration,
                    option_bucket=str(option_bucket),
                )
                if selected_contract:
                    selected_contract_symbols.append(str(selected_contract.get("symbol") or ""))
            status.write("Loading quotes for the selected option contracts.")
            option_snapshots = client.get_option_snapshots(selected_contract_symbols)
            plan = build_alpaca_option_trade_plan(
                ranked_scores=current_ranked_scores,
                current_option_positions=current_positions,
                open_orders=open_orders,
                option_contracts=option_contracts,
                option_snapshots=option_snapshots,
                strategy_allocation=allocation,
                as_of_date=as_of,
                option_bucket=str(option_bucket),
                tenor_days=int(tenor_days),
                max_contracts_per_position=int(max_contracts) if int(max_contracts) > 0 else None,
            )
            plan["client"] = client
            plan["account"] = account
            st.session_state["moe_alpaca_option_plan"] = plan
            st.session_state.pop("moe_alpaca_option_results", None)
        except Exception:
            status.update(label="Alpaca option portfolio generation failed", state="error", expanded=True)
            raise
        status.update(label="Alpaca option portfolio ready", state="complete", expanded=False)

plan = st.session_state.get("moe_alpaca_option_plan")
if isinstance(plan, dict):
    summary = plan.get("summary", pd.DataFrame()).iloc[0]
    summary_cols = st.columns(5)
    summary_cols[0].metric("Target Positions", f"{int(summary.get('target_positions', 0)):,}")
    summary_cols[1].metric("Calls To Open", f"{int(summary.get('calls_to_open', 0)):,}")
    summary_cols[2].metric("Puts To Open", f"{int(summary.get('puts_to_open', 0)):,}")
    summary_cols[3].metric("Contracts To Close", f"{int(summary.get('contracts_to_close', 0)):,}")
    summary_cols[4].metric("Option Allocation", f"${float(summary.get('strategy_allocation', 0.0)):,.0f}")
    capacity_cols = st.columns(3)
    capacity_cols[0].metric("Occupied Slots", f"{int(summary.get('occupied_slots', 0)):,}")
    capacity_cols[1].metric("Pending Buy Slots", f"{int(summary.get('pending_buy_underlyings', 0)):,}")
    capacity_cols[2].metric("Remaining Buy Slots", f"{int(summary.get('remaining_buy_slots', 0)):,}")

    for title, key in (
        ("New Entries", "desired_contracts"),
        ("Current Alpaca Option Positions", "current_option_positions"),
        ("Outstanding Alpaca Option Orders", "pending_option_orders"),
        ("Target Portfolio Actions", "actions"),
        ("Skipped Symbols", "skipped_symbols"),
    ):
        frame = plan.get(key, pd.DataFrame())
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            st.markdown(f"**{title}**")
            st.dataframe(frame, use_container_width=True, hide_index=True)

    preview_orders = plan.get("actionable_orders", pd.DataFrame())
    st.markdown(
        """
        <div class="section-card">
          <div class="table-title">Order Preview</div>
          <div class="table-copy">These are the Alpaca paper option orders produced from the current scores, positions, and open orders.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if isinstance(preview_orders, pd.DataFrame) and not preview_orders.empty:
        preview_cols = st.columns(4)
        preview_cols[0].metric("Orders", f"{len(preview_orders):,}")
        preview_cols[1].metric("Buy To Open", f"{preview_orders['action'].astype(str).str.startswith('buy_to_open').sum():,}")
        preview_cols[2].metric("Sell To Close", f"{preview_orders['action'].astype(str).str.startswith('sell_to_close').sum():,}")
        preview_cols[3].metric("Cancel", f"{preview_orders['action'].astype(str).str.startswith('cancel_').sum():,}")
        st.dataframe(preview_orders, use_container_width=True, hide_index=True)
        if st.button("Submit Alpaca Option Orders", type="primary", use_container_width=True):
            with st.status("Submitting Alpaca paper option orders...", expanded=True) as status:
                try:
                    results = plan["client"].submit_orders(preview_orders.to_dict(orient="records"))
                    st.session_state["moe_alpaca_option_results"] = pd.DataFrame(results)
                except Exception:
                    status.update(label="Alpaca option order submission failed", state="error", expanded=True)
                    raise
                status.update(label="Alpaca option order submission complete", state="complete", expanded=False)
    else:
        st.info("There are no actionable Alpaca option orders for the current portfolio.")

results = st.session_state.get("moe_alpaca_option_results")
if isinstance(results, pd.DataFrame) and not results.empty:
    st.markdown("**Alpaca Option Order Results**")
    st.dataframe(results, use_container_width=True, hide_index=True)

page_size = 50
selected_page, page_start, page_end, total_pages = render_leaderboard_pager(
    st,
    total_rows=total_rows,
    session_page_key="moe_leaderboard_current_page",
    selectbox_key="moe_leaderboard_page_select",
    page_size=page_size,
)
leaderboard_page = leaderboard.iloc[page_start:page_end].copy()
st.markdown(
    f"""
    <div class="section-card">
      <div class="table-header">
        <div>
          <p class="table-title">Leaderboard Table</p>
          <p class="table-copy">Showing rows {page_start + 1:,}-{page_end:,} of {total_rows:,}, sorted strictly by Combined Score from highest to lowest.</p>
        </div>
        <div class="pager-summary">
          Trained Families: {len(metadata.get('trained_feature_families') or []):,}<br>
          Instrument: Alpaca {str(option_bucket).replace('_', ' ').title()} Calls
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(render_leaderboard_table(leaderboard_page), unsafe_allow_html=True)

with st.expander("Saved Artifacts", expanded=False):
    st.json(metadata)
