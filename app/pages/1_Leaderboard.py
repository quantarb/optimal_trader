from __future__ import annotations

import importlib
import json
from datetime import date
import html
import os
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.live_trade_leaderboard import (
    default_live_trade_config,
    latest_scored_staleness_reason,
    load_saved_leaderboard,
    load_saved_latest_scored,
    run_live_trade_leaderboard_build,
)


@st.cache_data(show_spinner=False)
def _resolve_universe_size_from_config(config: dict[str, object]) -> int | None:
    try:
        from app.optimal_trade_lookup import bootstrap_django
        bootstrap_django()
        from pipeline.universe_selection import resolve_symbol_universe

        universe = resolve_symbol_universe(
            min_market_cap=float(config.get("min_market_cap", 10_000_000_000.0)),
            country=str(config.get("country", "US")),
            exchanges=list(config.get("exchanges", ["NASDAQ", "NYSE", "AMEX"])),
            exclude_pooled_vehicles=bool(config.get("exclude_pooled_vehicles", True)),
            limit=config.get("size"),
        )
        return int(len(tuple(universe)))
    except Exception:
        return None


def _recompute_eligibility(frame: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    out = frame.copy()
    component_cols = [column for column in out.columns if str(column).startswith("__component__")]
    if component_cols:
        eligible = pd.Series(True, index=out.index, dtype=bool)
        for component_col in component_cols:
            eligible &= pd.to_numeric(out[component_col], errors="coerce").gt(float(threshold)).fillna(False)
        combined_score = pd.to_numeric(out.get("Combined Score"), errors="coerce")
        out["Eligible"] = eligible & combined_score.notna()
        return out

    required_cols = ["Classifier Score", "Regressor Score", "Autoencoder Score"]
    missing_cols = [column for column in required_cols if column not in out.columns]
    if not missing_cols:
        classifier = pd.to_numeric(out["Classifier Score"], errors="coerce")
        regressor = pd.to_numeric(out["Regressor Score"], errors="coerce")
        autoencoder = pd.to_numeric(out["Autoencoder Score"], errors="coerce")
        out["Eligible"] = (
            classifier.gt(float(threshold))
            & regressor.gt(float(threshold))
            & autoencoder.gt(float(threshold))
        )
    return out


def _style_leaderboard_rows(frame: pd.DataFrame):
    if frame.empty or "Eligible" not in frame.columns:
        return frame

    display = frame.copy()
    display["Status"] = display["Eligible"].map(lambda value: "Eligible" if bool(value) else "Not Eligible")
    ordered_columns = ["Rank", "Status", "Scored Date", "Symbol", "Direction", "Eligible", "Classifier Score", "Regressor Score", "Autoencoder Score", "Combined Score", "Similar Trades"]
    existing_columns = [column for column in ordered_columns if column in display.columns]
    remaining_columns = [column for column in display.columns if column not in existing_columns]
    display = display[existing_columns + remaining_columns]

    def _row_style(row: pd.Series) -> list[str]:
        is_eligible = bool(row.get("Eligible"))
        if is_eligible:
            base_style = "background-color: rgba(0, 200, 5, 0.14); color: #216e39;"
            highlight_style = "background-color: rgba(0, 200, 5, 0.22); color: #14532d; font-weight: 700;"
        else:
            base_style = "background-color: rgba(220, 38, 38, 0.12); color: #b42318;"
            highlight_style = "background-color: rgba(220, 38, 38, 0.20); color: #7f1d1d; font-weight: 700;"
        styles: list[str] = []
        for column in row.index:
            if column in {"Status", "Symbol", "Direction", "Eligible"}:
                styles.append(highlight_style)
            else:
                styles.append(base_style)
        return styles

    return display.style.apply(_row_style, axis=1)


def _format_leaderboard_cell(column: str, value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if column in {"Classifier Score", "Regressor Score", "Autoencoder Score", "Combined Score"}:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return "" if pd.isna(numeric) else f"{float(numeric):.4f}"
    if column == "Eligible":
        return "Yes" if bool(value) else "No"
    return str(value)


def _render_leaderboard_html_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<div class="leaderboard-empty">No leaderboard rows available.</div>'

    hidden_columns = {"Eligible", "Status"}
    columns = [
        str(column)
        for column in frame.columns
        if str(column) not in hidden_columns and not str(column).startswith("__")
    ]
    header_html = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    row_html_parts: list[str] = []
    for _, row in frame.iterrows():
        is_eligible = bool(row.get("Eligible"))
        row_class = "eligible-row" if is_eligible else "ineligible-row"
        cell_html: list[str] = []
        for column in columns:
            value = row.get(column)
            if column == "Similar Trades":
                href = str(value or "").strip()
                cell_html.append(
                    f'<td><a class="trade-link" href="{html.escape(href)}" target="_self">Open</a></td>'
                )
                continue
            formatted = _format_leaderboard_cell(column, value)
            cell_html.append(f"<td>{html.escape(formatted)}</td>")
        row_html_parts.append(f'<tr class="{row_class}">{"".join(cell_html)}</tr>')

    return f"""
    <div class="leaderboard-table-wrap">
      <table class="leaderboard-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>
          {''.join(row_html_parts)}
        </tbody>
      </table>
    </div>
    """


def _format_display_object(value: object, *, max_length: int = 240) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, (dict, list, tuple, set)):
        text = json.dumps(value, default=str, sort_keys=True)
    else:
        text = str(value)
    text = " ".join(text.split())
    if len(text) > max_length:
        return text[: max_length - 1] + "..."
    return text


def _prepare_skipped_symbols_display(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    display = display.replace([float("inf"), -float("inf")], pd.NA)
    column_order = [
        "symbol",
        "direction",
        "candidate_rank",
        "reason",
        "replacement_status",
        "contract_value",
        "target_dollars",
        "quote_contract_value",
        "contract_price_source",
        "quote_contract_price_source",
        "error",
    ]
    existing_columns = [column for column in column_order if column in display.columns]
    remaining_columns = [column for column in display.columns if column not in existing_columns]
    display = display[existing_columns + remaining_columns]

    numeric_columns = {"candidate_rank", "contract_value", "target_dollars", "quote_contract_value"}
    for column in display.columns:
        if column in numeric_columns:
            display[column] = pd.to_numeric(display[column], errors="coerce")
        else:
            display[column] = display[column].map(_format_display_object)

    return display.rename(
        columns={
            "symbol": "Symbol",
            "direction": "Direction",
            "candidate_rank": "Candidate Rank",
            "reason": "Reason",
            "replacement_status": "Replacement Status",
            "contract_value": "Contract Value",
            "target_dollars": "Slot Budget",
            "quote_contract_value": "Quote Contract Value",
            "contract_price_source": "Limit Price Source",
            "quote_contract_price_source": "Quote Price Source",
            "error": "Error",
        }
    )


def _render_plain_html_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<div class="leaderboard-empty">No rows available.</div>'
    columns = [str(column) for column in frame.columns]
    header_html = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    row_parts: list[str] = []
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row.get(column)
            cells.append(f"<td>{html.escape(_format_display_object(value, max_length=500))}</td>")
        row_parts.append(f"<tr>{''.join(cells)}</tr>")
    return f"""
    <div class="leaderboard-table-wrap">
      <table class="leaderboard-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{''.join(row_parts)}</tbody>
      </table>
    </div>
    """


def _request_rerun() -> None:
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()


st.set_page_config(
    page_title="Leaderboard",
    page_icon="OT",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --page-top: #f5fbf6;
        --page-bottom: #eef7f0;
        --panel-bg: rgba(255, 255, 255, 0.94);
        --ink: #101714;
        --muted: #5d6f66;
        --line: rgba(20, 44, 29, 0.10);
        --accent: #00c805;
        --accent-deep: #009e04;
        --accent-soft: rgba(0, 200, 5, 0.10);
        --danger-soft: rgba(220, 38, 38, 0.08);
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
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(248, 252, 248, 0.98), rgba(239, 247, 241, 0.98));
        border-right: 1px solid rgba(16, 23, 20, 0.08);
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(247, 251, 248, 0.98));
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        box-shadow: var(--shadow);
    }
    .stButton > button {
        background: linear-gradient(135deg, var(--accent), var(--accent-deep));
        color: #f8fff8;
        border: none;
        border-radius: 999px;
        font-weight: 700;
        min-height: 2.8rem;
        box-shadow: 0 12px 24px rgba(0, 158, 4, 0.18);
    }
    .stButton > button[kind="secondary"] {
        background: #ffffff;
        color: #173224;
        border: 1px solid var(--line);
        box-shadow: none;
    }
    .leaderboard-hero {
        background:
            radial-gradient(circle at top right, rgba(0, 200, 5, 0.12), transparent 28%),
            linear-gradient(135deg, rgba(255, 255, 255, 0.97), rgba(245, 251, 246, 0.96));
        border: 1px solid var(--line);
        border-radius: 28px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow);
    }
    .leaderboard-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.32rem 0.72rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: #0f8f3b;
        font-size: 0.82rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-weight: 700;
    }
    .leaderboard-hero h1 {
        margin: 0.75rem 0 0 0;
        font-size: 2.35rem;
        line-height: 1.02;
        font-weight: 800;
    }
    .leaderboard-hero p {
        margin: 0.65rem 0 0 0;
        color: var(--muted);
        max-width: 48rem;
        font-size: 1.02rem;
        line-height: 1.55;
    }
    .section-card {
        background: var(--panel-bg);
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1rem 1.05rem;
        box-shadow: var(--shadow);
        margin-bottom: 0.95rem;
    }
    .table-header {
        display: flex;
        justify-content: space-between;
        align-items: end;
        gap: 1rem;
        margin-bottom: 0.9rem;
    }
    .table-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--ink);
        margin: 0;
    }
    .table-copy {
        color: var(--muted);
        font-size: 0.93rem;
        margin: 0.2rem 0 0 0;
    }
    .pager-summary {
        text-align: right;
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.45;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 18px;
        overflow: hidden;
        box-shadow: var(--shadow);
        background: rgba(255, 255, 255, 0.94);
    }
    .leaderboard-table-wrap {
        border: 1px solid var(--line);
        border-radius: 18px;
        overflow: auto;
        box-shadow: var(--shadow);
        background: rgba(255, 255, 255, 0.94);
    }
    .leaderboard-table {
        width: 100%;
        border-collapse: collapse;
        min-width: 980px;
        font-size: 0.94rem;
    }
    .leaderboard-table thead th {
        position: sticky;
        top: 0;
        background: #f7faf8;
        color: #335142;
        text-align: left;
        padding: 0.85rem 0.9rem;
        border-bottom: 1px solid var(--line);
        white-space: nowrap;
    }
    .leaderboard-table tbody td {
        padding: 0.85rem 0.9rem;
        border-bottom: 1px solid rgba(20, 44, 29, 0.06);
        white-space: nowrap;
    }
    .leaderboard-table tbody tr.eligible-row {
        background: rgba(0, 200, 5, 0.14);
        color: #14532d;
    }
    .leaderboard-table tbody tr.ineligible-row {
        background: rgba(220, 38, 38, 0.14);
        color: #7f1d1d;
    }
    .leaderboard-table tbody tr:hover {
        filter: brightness(0.985);
    }
    .leaderboard-table tbody tr td:first-child,
    .leaderboard-table tbody tr td:nth-child(4),
    .leaderboard-table tbody tr td:nth-child(5) {
        font-weight: 700;
    }
    .trade-link {
        color: inherit;
        font-weight: 700;
        text-decoration: underline;
        text-underline-offset: 2px;
    }
    .leaderboard-empty {
        border: 1px dashed var(--line);
        border-radius: 18px;
        padding: 1rem;
        color: var(--muted);
        background: rgba(255, 255, 255, 0.94);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="leaderboard-hero">
      <div class="leaderboard-kicker">Leaderboard</div>
      <h1>Ranked Trade Setups</h1>
      <p>
        This board ranks every stock we scored for the latest completed market day.
        Eligible names rise to the top first, then everything else stays sorted by combined score.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

default_cfg = default_live_trade_config()

with st.sidebar:
    st.subheader("Build")
    data_start = st.date_input(
        "Data Start",
        value=pd.Timestamp(default_cfg["dates"]["data_start"]).date(),
        max_value=date.today(),
    )
    min_market_cap_b = st.number_input(
        "Min Market Cap ($B)",
        min_value=1.0,
        value=float(default_cfg["universe"]["min_market_cap"]) / 1_000_000_000.0,
        step=1.0,
    )
    refresh_fmp = st.toggle("Download Missing FMP Data First", value=True)
    run_build = st.button("Train Models And Build Leaderboard", type="primary", use_container_width=True)


build_cfg = {
    "dates": {
        "data_start": str(pd.Timestamp(data_start).date()),
        "data_end": pd.Timestamp.today().strftime("%Y-%m-%d"),
    },
    "universe": {
        "min_market_cap": float(min_market_cap_b) * 1_000_000_000.0,
    },
    "fmp_refresh": {
        "enabled": bool(refresh_fmp),
        "refresh_symbol_sections_before_build": bool(refresh_fmp),
        "refresh_macro_before_build": bool(refresh_fmp),
        "existing_historical_sections_only": True,
        "verbose": False,
    },
}

leaderboard_payload = st.session_state.get("live_trade_leaderboard_payload")
stale_score_reason = None
saved = None
if leaderboard_payload is not None:
    payload_meta = dict(leaderboard_payload.get("meta") or {})
    payload_artifact_dir = (
        payload_meta.get("config", {}).get("runtime", {}).get("artifact_dir")
        or payload_meta.get("artifact_dir")
        or default_live_trade_config()["runtime"]["artifact_dir"]
    )
    stale_score_reason = latest_scored_staleness_reason(artifact_dir=payload_artifact_dir)
    if stale_score_reason:
        leaderboard_payload = None
        st.session_state.pop("live_trade_leaderboard_payload", None)
if leaderboard_payload is None:
    saved = load_saved_leaderboard()
    if saved is not None:
        _, saved_meta = saved
        saved_artifact_dir = (
            (saved_meta or {}).get("config", {}).get("runtime", {}).get("artifact_dir")
            or (saved_meta or {}).get("artifact_dir")
            or default_live_trade_config()["runtime"]["artifact_dir"]
        )
        stale_score_reason = latest_scored_staleness_reason(artifact_dir=saved_artifact_dir)

if stale_score_reason:
    st.warning(
        f"{stale_score_reason} Rerun the leaderboard refresh notebook to regenerate the saved scores."
    )

should_build = bool(run_build)
if should_build:
    build_log_lines: list[str] = []
    with st.status("Starting leaderboard build...", expanded=True) as status:
        def _progress_logger(message: str) -> None:
            build_log_lines.append(str(message))
            status.write(str(message))

        try:
            result = run_live_trade_leaderboard_build(
                config=build_cfg,
                progress_logger=_progress_logger,
            )
        except Exception:
            status.update(label="Leaderboard build failed", state="error", expanded=True)
            raise
        else:
            status.update(label="Leaderboard build complete", state="complete", expanded=False)

    leaderboard_payload = {
        "leaderboard": result.leaderboard.copy(),
        "latest_scored": result.latest_scored.copy(),
        "meta": {
            "latest_date": str(pd.Timestamp(result.latest_date).date()),
            "artifact_dir": str(result.artifact_dir),
            "universe_size": int(len(result.universe)),
            "scored_symbol_count": int(len(result.leaderboard)),
            "reference_trade_count": int(result.reference_trade_count),
            "vector_backend": str(result.vector_metadata.get("backend") or ""),
            "leaderboard_scope": "all_scored",
            "config": dict(result.config),
            "build_log_lines": list(build_log_lines),
        },
    }
    st.session_state["live_trade_leaderboard_payload"] = leaderboard_payload

if leaderboard_payload is None:
    if saved is not None:
        saved_leaderboard, saved_meta = saved
        saved_latest_scored = load_saved_latest_scored(
            artifact_dir=(saved_meta or {}).get("config", {}).get("runtime", {}).get("artifact_dir")
            or (saved_meta or {}).get("artifact_dir")
            or default_live_trade_config()["runtime"]["artifact_dir"]
        )
        leaderboard_payload = {
            "leaderboard": saved_leaderboard,
            "latest_scored": saved_latest_scored,
            "meta": saved_meta,
        }

if leaderboard_payload is None:
    st.info("Run the build once to train the models, save the artifacts, and generate the latest leaderboard.")
else:
    leaderboard = _recompute_eligibility(leaderboard_payload["leaderboard"].copy(), threshold=0.50)
    if "Combined Score" in leaderboard.columns:
        leaderboard["Combined Score"] = pd.to_numeric(leaderboard["Combined Score"], errors="coerce")
    if "Eligible" in leaderboard.columns:
        leaderboard = leaderboard.sort_values(
            ["Eligible", "Combined Score"],
            ascending=[False, False],
            kind="stable",
        ).reset_index(drop=True)
    if "Rank" in leaderboard.columns:
        leaderboard["Rank"] = range(1, len(leaderboard) + 1)
    meta = dict(leaderboard_payload.get("meta") or {})
    config = dict(meta.get("config") or {})
    universe_cfg = dict(config.get("universe") or {})
    legacy_top_k = pd.to_numeric(pd.Series([((meta.get("config") or {}).get("strategy") or {}).get("top_k")]), errors="coerce").iloc[0]
    leaderboard_scope = str(meta.get("leaderboard_scope") or "").strip().lower()
    legacy_capped_artifact = (
        leaderboard_scope != "all_scored"
        and pd.notna(legacy_top_k)
        and int(len(leaderboard)) == int(legacy_top_k)
    )
    page_size = 50
    total_rows = int(len(leaderboard))
    total_pages = max((total_rows - 1) // page_size + 1, 1)
    eligible_count = int(pd.Series(leaderboard.get("Eligible"), dtype="boolean").fillna(False).sum()) if "Eligible" in leaderboard.columns else 0
    ineligible_count = max(total_rows - eligible_count, 0)
    universe_size_value = pd.to_numeric(pd.Series([meta.get("universe_size")]), errors="coerce").iloc[0]
    if pd.isna(universe_size_value):
        universe_size_value = pd.to_numeric(pd.Series([meta.get("resolved_universe_size")]), errors="coerce").iloc[0]
    if pd.isna(universe_size_value):
        universe_size_value = pd.to_numeric(pd.Series([universe_cfg.get("size")]), errors="coerce").iloc[0]
    if pd.isna(universe_size_value):
        resolved_universe_size = _resolve_universe_size_from_config(universe_cfg)
        universe_size_value = float(resolved_universe_size) if resolved_universe_size is not None else float("nan")
    scored_symbol_count = pd.to_numeric(pd.Series([meta.get("scored_symbol_count")]), errors="coerce").iloc[0]
    if pd.isna(scored_symbol_count):
        scored_symbol_count = float(total_rows)
    session_page_key = "leaderboard_current_page"
    if session_page_key not in st.session_state:
        st.session_state[session_page_key] = 1
    st.session_state[session_page_key] = max(1, min(int(st.session_state[session_page_key]), total_pages))

    ribbon_cols = st.columns(5)
    ribbon_cols[0].metric("As Of Date", str(meta.get("latest_date") or ""))
    ribbon_cols[1].metric("Universe Size", f"{int(universe_size_value) if pd.notna(universe_size_value) else 0:,}")
    ribbon_cols[2].metric("Scored Symbols", f"{int(scored_symbol_count) if pd.notna(scored_symbol_count) else total_rows:,}")
    ribbon_cols[3].metric("Eligible", f"{eligible_count:,}")
    ribbon_cols[4].metric("Not Eligible", f"{ineligible_count:,}")

    if legacy_capped_artifact:
        st.warning(
            "This saved leaderboard appears to come from an older capped build that kept only 100 rows. "
            "Run `Train Models And Build Leaderboard` once more to regenerate the full scored universe."
        )

    latest_scored_for_robinhood = leaderboard_payload.get("latest_scored")
    if latest_scored_for_robinhood is not None and not getattr(latest_scored_for_robinhood, "empty", True):
        st.markdown(
            """
            <div class="section-card">
              <div class="table-title">Robinhood Option Automation</div>
              <div class="table-copy">Open long calls for bullish names and long puts for bearish names, using the same shared top-k capacity and classifier-driven exits as the options notebook.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        strategy_cfg = dict((meta.get("config") or {}).get("strategy") or {})
        rh_option_cols = st.columns(4)
        rh_top_k = rh_option_cols[0].number_input("Robinhood Top K", min_value=1, max_value=max(total_rows, 1), value=min(max(total_rows, 1), 20), step=1)
        rh_option_bucket = rh_option_cols[1].selectbox("Option Bucket", options=["atm_option", "otm_option", "ditm_option"], index=1)
        rh_tenor_days = rh_option_cols[2].number_input("Option Tenor (Days)", min_value=7, max_value=365, value=60, step=7)
        rh_max_contracts = rh_option_cols[3].number_input(
            "Max Contracts / Position",
            min_value=0,
            max_value=10000,
            value=0,
            step=1,
            help="Use 0 to size by budget with no contract-count cap.",
        )

        env_username_ready = bool(str(os.getenv("ROBINHOOD_USERNAME") or "").strip())
        env_password_ready = bool(str(os.getenv("ROBINHOOD_PASSWORD") or "").strip())
        env_mfa_ready = bool(str(os.getenv("ROBINHOOD_MFA_CODE") or "").strip())
        st.caption(
            "Robinhood login credentials are loaded from `.env`. "
            f"Username: {'loaded' if env_username_ready else 'missing'} | "
            f"Password: {'loaded' if env_password_ready else 'missing'} | "
            f"MFA code: {'loaded from .env' if env_mfa_ready else 'enter below if Robinhood asks for one'}"
        )

        rh_plan_cols = st.columns(2)
        rh_equity_override = rh_plan_cols[0].number_input(
            "Account Equity Override ($)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            help="Leave at 0 to use the live Robinhood equity from the connected account.",
        )
        rh_strategy_allocation = rh_plan_cols[1].number_input(
            "Option Allocation Budget ($)",
            min_value=0.0,
            value=100000.0,
            step=1000.0,
            help="Only this dollar amount will be sized into the option strategy. Leave at 0 to use the full account equity.",
        )
        st.caption("This Robinhood flow only manages option positions from the strategy. It does not submit sell orders for your QQQ stock or other common-stock holdings.")

        build_rh_plan = st.button("Generate Robinhood Target Portfolio", type="primary", use_container_width=True)
        if build_rh_plan:
            rh_plan_log_lines: list[str] = []
            with st.status("Connecting to Robinhood and generating the target portfolio...", expanded=True) as status:
                try:
                    import trading.robinhood as robinhood_module

                    robinhood_module = importlib.reload(robinhood_module)
                    from trading.robinhood import (
                        build_robinhood_option_trade_plan,
                        enrich_robinhood_option_prices,
                        annotate_robinhood_option_limit_savings,
                        load_robinhood_account_snapshot,
                        load_robinhood_open_option_orders,
                        load_robinhood_option_positions,
                        robinhood_login,
                    )

                    rh_plan_log_lines.append(
                        f"Loaded trading.robinhood from {getattr(robinhood_module, '__file__', '<unknown>')}."
                    )
                    status.write(rh_plan_log_lines[-1])
                    rh_plan_log_lines.append("Step 0: logging in to Robinhood.")
                    status.write(rh_plan_log_lines[-1])
                    robinhood_login(
                        store_session=True,
                    )
                    rh_plan_log_lines.append("Robinhood login succeeded")
                    status.write(rh_plan_log_lines[-1])
                    rh_plan_log_lines.append("Step 1: loading account snapshot and current option positions.")
                    status.write(rh_plan_log_lines[-1])
                    account_snapshot = load_robinhood_account_snapshot(
                        account_number=None,
                    )
                    current_option_positions = load_robinhood_option_positions(
                        account_number=None,
                    )
                    rh_plan_log_lines.append(f"Loaded {int(len(current_option_positions))} current option position row(s).")
                    status.write(rh_plan_log_lines[-1])
                    rh_plan_log_lines.append("Loading outstanding Robinhood option orders for capacity checks.")
                    status.write(rh_plan_log_lines[-1])
                    pending_option_orders = load_robinhood_open_option_orders(
                        account_number=None,
                    )
                    pending_option_orders = annotate_robinhood_option_limit_savings(pending_option_orders)
                    rh_plan_log_lines.append(f"Loaded {int(len(pending_option_orders))} outstanding option order row(s).")
                    status.write(rh_plan_log_lines[-1])

                    snapshot_equity_candidates = [
                        ((account_snapshot.get("portfolio") or {}).get("equity")),
                        ((account_snapshot.get("phoenix") or {}).get("portfolio_equity")),
                        ((account_snapshot.get("phoenix") or {}).get("total_equity")),
                    ]
                    snapshot_equity = pd.to_numeric(pd.Series(snapshot_equity_candidates), errors="coerce").dropna()
                    resolved_account_equity = float(rh_equity_override) if float(rh_equity_override) > 0 else (float(snapshot_equity.iloc[0]) if not snapshot_equity.empty else 0.0)
                    rh_plan_log_lines.append(f"Using account equity ${resolved_account_equity:,.2f}")
                    status.write(rh_plan_log_lines[-1])
                    resolved_strategy_allocation = float(rh_strategy_allocation) if float(rh_strategy_allocation) > 0 else float(resolved_account_equity)
                    rh_plan_log_lines.append(f"Using option allocation budget ${resolved_strategy_allocation:,.2f}")
                    status.write(rh_plan_log_lines[-1])
                    plan = build_robinhood_option_trade_plan(
                        latest_scored_df=latest_scored_for_robinhood,
                        current_option_positions=current_option_positions,
                        pending_option_orders=pending_option_orders,
                        top_k=int(rh_top_k),
                        score_col=str(strategy_cfg.get("score_col") or "buy_score_mean_raw_pct6"),
                        component_threshold=float(strategy_cfg.get("component_threshold") or 0.50),
                        account_equity=float(resolved_account_equity),
                        strategy_allocation=float(resolved_strategy_allocation),
                        as_of_date=str(meta.get("latest_date") or pd.Timestamp.today().date()),
                        option_bucket=str(rh_option_bucket),
                        tenor_days=int(rh_tenor_days),
                        max_contracts_per_position=int(rh_max_contracts) if int(rh_max_contracts) > 0 else None,
                    )
                    plan["actions"] = enrich_robinhood_option_prices(plan.get("actions", pd.DataFrame()))
                    enriched_actions = plan.get("actions", pd.DataFrame())
                    if isinstance(enriched_actions, pd.DataFrame) and not enriched_actions.empty:
                        actionable_mask = enriched_actions["action"].isin(
                            [
                                "cancel_buy_to_open_call",
                                "cancel_buy_to_open_put",
                                "sell_to_close_call",
                                "sell_to_close_put",
                                "buy_to_open_call",
                                "buy_to_open_put",
                            ]
                        )
                        actionable_orders = enriched_actions.loc[actionable_mask].copy()
                        cancel_mask = actionable_orders["action"].astype(str).str.startswith("cancel_")
                        actionable_orders["skip_submit"] = (
                            pd.to_numeric(actionable_orders.get("quantity"), errors="coerce").fillna(0.0).le(0.0)
                            & ~cancel_mask
                        )
                        plan["actionable_orders"] = actionable_orders
                    else:
                        plan["actionable_orders"] = pd.DataFrame()
                    plan["current_option_positions"] = current_option_positions
                    plan["pending_option_orders"] = pending_option_orders
                    plan["account_snapshot"] = account_snapshot
                    plan["account_number"] = ""
                    plan["store_session"] = True
                    build_log_lines = list(plan.get("plan_log_lines") or [])
                    for line in build_log_lines:
                        status.write(line)
                    plan["plan_log_lines"] = list(rh_plan_log_lines) + build_log_lines
                    st.session_state["robinhood_option_plan"] = plan
                    st.session_state.pop("robinhood_option_order_results", None)
                    st.session_state.pop("robinhood_option_preview_requested", None)
                except Exception:
                    status.update(label="Robinhood target portfolio generation failed", state="error", expanded=True)
                    raise
                else:
                    status.update(label="Robinhood target portfolio ready", state="complete", expanded=False)

        robinhood_option_plan = st.session_state.get("robinhood_option_plan")
        if isinstance(robinhood_option_plan, dict) and not robinhood_option_plan.get("code_version"):
            st.session_state.pop("robinhood_option_plan", None)
            st.warning("Cleared an older Robinhood target portfolio plan. Generate a fresh target portfolio to use the latest option lookup code.")
            robinhood_option_plan = None
        if robinhood_option_plan:
            plan_log_lines = list(robinhood_option_plan.get("plan_log_lines") or [])
            if plan_log_lines:
                st.success("\n\n".join(plan_log_lines))
            plan_summary = robinhood_option_plan.get("summary", pd.DataFrame())
            if isinstance(plan_summary, pd.DataFrame) and not plan_summary.empty:
                summary_row = plan_summary.iloc[0]
                rh_summary_cols = st.columns(5)
                rh_summary_cols[0].metric("Target Positions", f"{int(summary_row.get('target_positions', 0)):,}")
                rh_summary_cols[1].metric("Calls To Open", f"{int(summary_row.get('calls_to_open', 0)):,}")
                rh_summary_cols[2].metric("Puts To Open", f"{int(summary_row.get('puts_to_open', 0)):,}")
                rh_summary_cols[3].metric("Contracts To Close", f"{int(summary_row.get('contracts_to_close', 0)):,}")
                rh_summary_cols[4].metric("Option Allocation", f"${float(summary_row.get('strategy_allocation', summary_row.get('account_equity', 0.0))):,.0f}")
                rh_capacity_cols = st.columns(3)
                rh_capacity_cols[0].metric("Occupied Slots", f"{int(summary_row.get('occupied_slots', 0)):,}")
                rh_capacity_cols[1].metric("Pending Buy Slots", f"{int(summary_row.get('pending_buy_underlyings', 0)):,}")
                rh_capacity_cols[2].metric("Remaining Buy Slots", f"{int(summary_row.get('remaining_buy_slots', 0)):,}")

            desired_contracts = robinhood_option_plan.get("desired_contracts", pd.DataFrame())
            if isinstance(desired_contracts, pd.DataFrame) and not desired_contracts.empty:
                st.markdown("**New Entries**")
                st.dataframe(desired_contracts, use_container_width=True, hide_index=True)

            current_option_positions = robinhood_option_plan.get("current_option_positions", pd.DataFrame())
            if isinstance(current_option_positions, pd.DataFrame) and not current_option_positions.empty:
                st.markdown("**Current Robinhood Option Positions**")
                st.dataframe(current_option_positions, use_container_width=True, hide_index=True)

            pending_option_orders = robinhood_option_plan.get("pending_option_orders", pd.DataFrame())
            if isinstance(pending_option_orders, pd.DataFrame) and not pending_option_orders.empty:
                st.markdown("**Outstanding Robinhood Option Orders**")
                missed_move_total = (
                    pd.to_numeric(pending_option_orders.get("missed_move_total"), errors="coerce")
                    .dropna()
                    .sum()
                    if "missed_move_total" in pending_option_orders.columns
                    else 0.0
                )
                pending_buy_count = int(
                    pending_option_orders.get("action", pd.Series(dtype=str)).astype(str).str.startswith("buy_to_open").sum()
                )
                saving_loss_total = -float(missed_move_total)
                pending_cols = st.columns(2)
                pending_cols[0].metric("Pending Buy Orders", f"{pending_buy_count:,}")
                pending_cols[1].metric("Saving / Loss", f"${saving_loss_total:,.2f}")
                pending_display = pending_option_orders.copy()
                if "missed_move_total" in pending_display.columns:
                    pending_display["saving_loss"] = -pd.to_numeric(
                        pending_display["missed_move_total"],
                        errors="coerce",
                    )
                if "limit_price" not in pending_display.columns:
                    pending_display["limit_price"] = pd.NA
                pending_display["limit_price"] = pd.to_numeric(
                    pending_display["limit_price"],
                    errors="coerce",
                )
                if "price" in pending_display.columns:
                    pending_display["limit_price"] = pending_display["limit_price"].fillna(
                        pd.to_numeric(pending_display["price"], errors="coerce")
                    )
                for price_column in ("limit_order_price", "premium", "processed_premium", "pending_premium"):
                    if price_column in pending_display.columns:
                        pending_display["limit_price"] = pending_display["limit_price"].fillna(
                            pd.to_numeric(pending_display[price_column], errors="coerce")
                        )
                if "current_qty" not in pending_display.columns:
                    pending_display["current_qty"] = pd.NA
                pending_display["current_qty"] = pd.to_numeric(
                    pending_display["current_qty"],
                    errors="coerce",
                )
                for qty_column in ("contract_quantity", "quantity"):
                    if qty_column in pending_display.columns:
                        pending_display["current_qty"] = pending_display["current_qty"].fillna(
                            pd.to_numeric(pending_display[qty_column], errors="coerce")
                        )
                discount_rate = 0.05
                if "discount_rate" in pending_display.columns:
                    rate_values = pd.to_numeric(pending_display["discount_rate"], errors="coerce")
                    if not rate_values.dropna().empty:
                        discount_rate = float(rate_values.dropna().iloc[0])
                if "original_strategy_price" not in pending_display.columns:
                    pending_display["original_strategy_price"] = pd.NA
                pending_display["original_strategy_price"] = pd.to_numeric(
                    pending_display["original_strategy_price"],
                    errors="coerce",
                ).fillna(pending_display["limit_price"] / discount_rate)
                if "original_strategy_qty" not in pending_display.columns:
                    pending_display["original_strategy_qty"] = pd.NA
                pending_display["original_strategy_qty"] = pd.to_numeric(
                    pending_display["original_strategy_qty"],
                    errors="coerce",
                ).fillna(pending_display["current_qty"] * discount_rate)
                pending_display_cols = {
                    "symbol": "Symbol",
                    "action": "Action",
                    "limit_price": "Limit Order Price",
                    "limit_price_source": "Limit Price Source",
                    "current_qty": "Limit Order Qty",
                    "bid_price": "Current Bid Price",
                    "original_strategy_price": "Original Strategy Price",
                    "original_strategy_qty": "Original Strategy Qty",
                    "saving_loss": "Saving / Loss",
                }
                for column in pending_display_cols:
                    if column not in pending_display.columns:
                        pending_display[column] = pd.NA
                st.dataframe(
                    pending_display[list(pending_display_cols)].rename(columns=pending_display_cols),
                    use_container_width=True,
                    hide_index=True,
                )

            actions_df = robinhood_option_plan.get("actions", pd.DataFrame())
            if isinstance(actions_df, pd.DataFrame) and not actions_df.empty:
                st.markdown("**Target Portfolio Actions**")
                st.dataframe(actions_df, use_container_width=True, hide_index=True)

            skipped_df = robinhood_option_plan.get("skipped_symbols", pd.DataFrame())
            if isinstance(skipped_df, pd.DataFrame) and not skipped_df.empty:
                st.markdown("**Skipped Symbols**")
                skipped_display = _prepare_skipped_symbols_display(skipped_df)
                try:
                    st.dataframe(skipped_display, use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.warning(f"Could not render the interactive skipped-symbols table: {type(exc).__name__}: {exc}")
                    st.markdown(_render_plain_html_table(skipped_display), unsafe_allow_html=True)

            st.caption(
                "Use `Generate Robinhood Target Portfolio` first to refresh entries and exits from the current leaderboard. "
                "Use the submit button only when you're ready to send the option orders."
            )

            submit_cols = st.columns([1.2, 1.2])
            submit_option_orders = submit_cols[0].button(
                "Submit Robinhood Option Orders",
                type="primary",
                use_container_width=True,
            )
            if submit_option_orders:
                st.session_state["robinhood_option_preview_requested"] = True

            preview_requested = bool(st.session_state.get("robinhood_option_preview_requested"))
            preview_orders = robinhood_option_plan.get("actionable_orders", pd.DataFrame())
            if preview_requested:
                st.markdown(
                    """
                    <div class="section-card">
                      <div class="table-title">Order Preview</div>
                      <div class="table-copy">Review these Robinhood option orders before anything gets submitted.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if isinstance(preview_orders, pd.DataFrame) and not preview_orders.empty:
                    preview_summary_cols = st.columns(4)
                    preview_summary_cols[0].metric("Orders", f"{int(len(preview_orders)):,}")
                    preview_summary_cols[1].metric(
                        "Buy To Open",
                        f"{int(preview_orders['action'].astype(str).str.startswith('buy_to_open').sum()):,}",
                    )
                    preview_summary_cols[2].metric(
                        "Sell To Close",
                        f"{int(preview_orders['action'].astype(str).str.startswith('sell_to_close').sum()):,}",
                    )
                    preview_summary_cols[3].metric(
                        "Cancel",
                        f"{int(preview_orders['action'].astype(str).str.startswith('cancel_').sum()):,}",
                    )
                    bid_limit_columns = [
                        column
                        for column in preview_orders.columns
                        if str(column).startswith("bid_price_x_")
                    ]
                    preview_columns = [
                        column
                        for column in [
                            "symbol",
                            "action",
                            "quantity",
                            "order_type",
                            "option_type",
                            "expiry_date",
                            "strike_price",
                            "limit_order_price",
                            "price",
                            "limit_price_source",
                            "bid_price",
                            *bid_limit_columns,
                            "previous_close_price",
                            "ask_price",
                            "mark_price",
                            "contract_value",
                            "quote_contract_value",
                            "order_id",
                            "cancel_url",
                            "breakeven_price",
                            "breakeven_move_pct",
                            "reason",
                            "combined_score",
                            "direction",
                        ]
                        if column in preview_orders.columns
                    ]
                    st.dataframe(
                        preview_orders[preview_columns] if preview_columns else preview_orders,
                        use_container_width=True,
                        hide_index=True,
                    )
                    preview_action_cols = st.columns([1, 1.4, 1.6])
                    cancel_preview = preview_action_cols[0].button(
                        "Cancel Preview",
                        type="secondary",
                        use_container_width=True,
                    )
                    confirm_submit = preview_action_cols[1].button(
                        "Confirm And Submit",
                        type="primary",
                        use_container_width=True,
                    )
                    if cancel_preview:
                        st.session_state["robinhood_option_preview_requested"] = False
                        _request_rerun()
                    if confirm_submit:
                        with st.status("Submitting Robinhood option orders...", expanded=True) as status:
                            try:
                                import trading.robinhood as robinhood_module

                                importlib.reload(robinhood_module)
                                from trading.robinhood import robinhood_login, submit_robinhood_option_orders

                                robinhood_login(
                                    store_session=bool(robinhood_option_plan.get("store_session", True)),
                                )
                                order_results = submit_robinhood_option_orders(
                                    orders_df=preview_orders,
                                    account_number=str(robinhood_option_plan.get("account_number") or "").strip() or None,
                                    time_in_force="gtc",
                                )
                                st.session_state["robinhood_option_order_results"] = order_results
                                st.session_state["robinhood_option_preview_requested"] = False
                            except Exception:
                                status.update(label="Robinhood option order submission failed", state="error", expanded=True)
                                raise
                            else:
                                unconfirmed_count = (
                                    int(order_results["robinhood_state"].astype(str).str.lower().eq("unconfirmed").sum())
                                    if isinstance(order_results, pd.DataFrame) and "robinhood_state" in order_results.columns
                                    else 0
                                )
                                failed_count = (
                                    int((~order_results["submitted"].fillna(False).astype(bool)).sum())
                                    if isinstance(order_results, pd.DataFrame) and "submitted" in order_results.columns
                                    else 0
                                )
                                if unconfirmed_count:
                                    status.update(
                                        label=f"Robinhood returned {unconfirmed_count} unconfirmed option order(s)",
                                        state="error",
                                        expanded=True,
                                    )
                                    st.warning(
                                        "Robinhood created order IDs, but marked them unconfirmed. "
                                        "Check Robinhood before submitting replacements."
                                    )
                                elif failed_count:
                                    status.update(label="Some Robinhood option orders were not accepted", state="error", expanded=True)
                                else:
                                    status.update(label="Robinhood option order submission complete", state="complete", expanded=False)
                else:
                    st.info("There are no actionable Robinhood option orders to preview for the current target portfolio.")
                    if st.button("Close Preview", type="secondary", use_container_width=False):
                        st.session_state["robinhood_option_preview_requested"] = False
                        _request_rerun()

            order_results = st.session_state.get("robinhood_option_order_results")
            if isinstance(order_results, pd.DataFrame) and not order_results.empty:
                st.markdown("**Robinhood Order Results**")
                st.dataframe(order_results, use_container_width=True, hide_index=True)

    with st.container():
        st.markdown(
            """
            <div class="section-card">
              <div class="table-title">Page Controls</div>
              <div class="table-copy">Move through the ranked list without losing your place.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        pager_cols = st.columns([1, 1, 2.2, 1.8])
        if pager_cols[0].button("Previous", use_container_width=True, disabled=st.session_state[session_page_key] <= 1, type="secondary"):
            st.session_state[session_page_key] -= 1
        if pager_cols[1].button("Next", use_container_width=True, disabled=st.session_state[session_page_key] >= total_pages, type="secondary"):
            st.session_state[session_page_key] += 1
        selected_page = pager_cols[2].selectbox(
            "Page",
            options=list(range(1, total_pages + 1)),
            index=st.session_state[session_page_key] - 1,
            key="leaderboard_page_select",
            label_visibility="collapsed",
        )
        st.session_state[session_page_key] = int(selected_page)
        pager_cols[3].markdown(
            f'<div class="pager-summary"><strong>Page {st.session_state[session_page_key]} of {total_pages}</strong><br>50 names per page</div>',
            unsafe_allow_html=True,
        )

    current_page = int(st.session_state[session_page_key])
    page_start = (int(current_page) - 1) * page_size
    page_end = min(page_start + page_size, total_rows)
    leaderboard_page = leaderboard.iloc[page_start:page_end].copy()

    st.markdown(
        f"""
        <div class="section-card">
          <div class="table-header">
            <div>
              <p class="table-title">Leaderboard Table</p>
              <p class="table-copy">Showing rows {page_start + 1:,}-{page_end:,} of {total_rows:,}. Eligible names are highlighted green. Non-eligible names are highlighted red.</p>
            </div>
            <div class="pager-summary">
              Historical Entry Trades: {int(meta.get('reference_trade_count') or 0):,}<br>
              Vector Backend: {str(meta.get("vector_backend") or "")}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    leaderboard_display = leaderboard_page.copy()
    if "Status" not in leaderboard_display.columns and "Eligible" in leaderboard_display.columns:
        insert_at = 1 if "Rank" in leaderboard_display.columns else 0
        leaderboard_display.insert(insert_at, "Status", leaderboard_display["Eligible"].map(lambda value: "Eligible" if bool(value) else "Not Eligible"))
    st.markdown(_render_leaderboard_html_table(leaderboard_display), unsafe_allow_html=True)

    with st.expander("Saved Artifacts", expanded=False):
        st.json(meta)

    build_log_lines = list(meta.get("build_log_lines") or [])
    if build_log_lines:
        with st.expander("Build Steps", expanded=False):
            st.code("\n".join(build_log_lines), language="text")
