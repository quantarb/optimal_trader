from __future__ import annotations

import html

import pandas as pd


LEADERBOARD_CSS = """
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
    --shadow: 0 18px 42px rgba(16, 23, 20, 0.07);
}
.stApp {
    background:
        radial-gradient(circle at top right, rgba(0, 200, 5, 0.10), transparent 24%),
        radial-gradient(circle at left 20%, rgba(15, 143, 59, 0.08), transparent 22%),
        linear-gradient(180deg, var(--page-top) 0%, var(--page-bottom) 100%);
    color: var(--ink);
}
.block-container { padding-top: 2rem; padding-bottom: 2.5rem; }
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
    padding: 0.32rem 0.72rem;
    border-radius: 999px;
    background: var(--accent-soft);
    color: #0f8f3b;
    font-size: 0.82rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 700;
}
.leaderboard-hero h1 { margin: 0.75rem 0 0 0; font-size: 2.35rem; line-height: 1.02; font-weight: 800; }
.leaderboard-hero p { margin: 0.65rem 0 0 0; color: var(--muted); max-width: 48rem; font-size: 1.02rem; line-height: 1.55; }
.section-card {
    background: var(--panel-bg);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 1rem 1.05rem;
    box-shadow: var(--shadow);
    margin-bottom: 0.95rem;
}
.table-header { display: flex; justify-content: space-between; align-items: end; gap: 1rem; margin-bottom: 0.9rem; }
.table-title { font-size: 1.1rem; font-weight: 700; color: var(--ink); margin: 0; }
.table-copy { color: var(--muted); font-size: 0.93rem; margin: 0.2rem 0 0 0; }
.pager-summary { text-align: right; color: var(--muted); font-size: 0.92rem; line-height: 1.45; }
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
.leaderboard-table { width: 100%; border-collapse: collapse; min-width: 980px; font-size: 0.94rem; }
.leaderboard-table thead th {
    position: sticky; top: 0; background: #f7faf8; color: #335142; text-align: left;
    padding: 0.85rem 0.9rem; border-bottom: 1px solid var(--line); white-space: nowrap;
}
.leaderboard-table tbody td { padding: 0.85rem 0.9rem; border-bottom: 1px solid rgba(20, 44, 29, 0.06); white-space: nowrap; }
.leaderboard-table tbody tr.eligible-row { background: rgba(0, 200, 5, 0.14); color: #14532d; }
.leaderboard-table tbody tr.ineligible-row { background: rgba(220, 38, 38, 0.14); color: #7f1d1d; }
.leaderboard-table tbody tr:hover { filter: brightness(0.985); }
.leaderboard-table tbody tr td:first-child,
.leaderboard-table tbody tr td:nth-child(4),
.leaderboard-table tbody tr td:nth-child(5) { font-weight: 700; }
.trade-link {
    color: inherit;
    font-weight: 700;
    text-decoration: underline;
    text-underline-offset: 2px;
}
</style>
"""


def render_leaderboard_table(frame: pd.DataFrame) -> str:
    columns = [
        str(column)
        for column in frame.columns
        if str(column) not in {"Eligible", "Status"} and not str(column).startswith("__")
    ]
    header_html = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    rows = []
    for _, row in frame.iterrows():
        row_class = "eligible-row" if bool(row.get("Eligible")) else "ineligible-row"
        cells = []
        for column in columns:
            value = row.get(column)
            if column == "Similar Trades":
                href = str(value or "").strip()
                cells.append(
                    f'<td><a class="trade-link" href="{html.escape(href)}" target="_self">Open</a></td>'
                )
                continue
            if value is None or pd.isna(value):
                formatted = ""
            elif column.endswith(" Score"):
                formatted = f"{float(value):.4f}"
            elif column == "Close":
                formatted = f"${float(value):,.2f}"
            else:
                formatted = str(value)
            cells.append(f"<td>{html.escape(formatted)}</td>")
        rows.append(f'<tr class="{row_class}">{"".join(cells)}</tr>')
    return (
        '<div class="leaderboard-table-wrap"><table class="leaderboard-table">'
        f'<thead><tr>{header_html}</tr></thead><tbody>{"".join(rows)}</tbody>'
        "</table></div>"
    )


def render_leaderboard_ribbon(
    st,
    *,
    as_of_date: object,
    universe_size: int,
    scored_symbols: int,
    inactive: int,
    eligible: int,
    not_eligible: int,
) -> None:
    columns = st.columns(6)
    columns[0].metric("As Of Date", str(as_of_date or ""))
    columns[1].metric("Universe Size", f"{int(universe_size):,}")
    columns[2].metric("Scored Symbols", f"{int(scored_symbols):,}")
    columns[3].metric("Inactive", f"{int(inactive):,}")
    columns[4].metric("Eligible", f"{int(eligible):,}")
    columns[5].metric("Not Eligible", f"{int(not_eligible):,}")


def render_leaderboard_pager(
    st,
    *,
    total_rows: int,
    session_page_key: str,
    selectbox_key: str,
    page_size: int = 50,
) -> tuple[int, int, int, int]:
    total_pages = max((int(total_rows) - 1) // int(page_size) + 1, 1)
    st.session_state.setdefault(session_page_key, 1)
    st.session_state[session_page_key] = max(
        1, min(int(st.session_state[session_page_key]), total_pages)
    )
    st.markdown(
        """
        <div class="section-card">
          <div class="table-title">Page Controls</div>
          <div class="table-copy">Move through the ranked list without losing your place.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns([1, 1, 2.2, 1.8])
    if columns[0].button(
        "Previous",
        use_container_width=True,
        disabled=st.session_state[session_page_key] <= 1,
        type="secondary",
        key=f"{selectbox_key}_previous",
    ):
        st.session_state[session_page_key] -= 1
    if columns[1].button(
        "Next",
        use_container_width=True,
        disabled=st.session_state[session_page_key] >= total_pages,
        type="secondary",
        key=f"{selectbox_key}_next",
    ):
        st.session_state[session_page_key] += 1
    selected_page = columns[2].selectbox(
        "Page",
        options=list(range(1, total_pages + 1)),
        index=st.session_state[session_page_key] - 1,
        key=selectbox_key,
        label_visibility="collapsed",
    )
    st.session_state[session_page_key] = int(selected_page)
    columns[3].markdown(
        f'<div class="pager-summary"><strong>Page {int(selected_page)} of {total_pages}</strong>'
        f'<br>{int(page_size)} names per page</div>',
        unsafe_allow_html=True,
    )
    page_start = (int(selected_page) - 1) * int(page_size)
    page_end = min(page_start + int(page_size), int(total_rows))
    return int(selected_page), page_start, page_end, total_pages


__all__ = [
    "LEADERBOARD_CSS",
    "render_leaderboard_pager",
    "render_leaderboard_ribbon",
    "render_leaderboard_table",
]
