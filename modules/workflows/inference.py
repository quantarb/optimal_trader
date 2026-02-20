from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from modules.api import (
    build_technical_dataframe,
    build_fundamental_dataframe,
    build_macro_dataframe,
)
from modules.analysis import add_cluster_explanations
from modules.analysis.alpha_flavors import score_trade_flavor


def _class_probs_for_buy_short(clf_wrapper: Any, X: pd.DataFrame) -> tuple[float, float]:
    model = getattr(clf_wrapper, "model", clf_wrapper)
    proba = np.asarray(model.predict_proba(X), dtype=float)
    classes = list(getattr(model, "classes_", []))
    class_mapping = getattr(clf_wrapper, "_class_mapping", {}) or {}

    labels = [str(class_mapping.get(c, c)).strip().lower() for c in classes]

    def find_idx(cands: set[str]):
        for i, lab in enumerate(labels):
            if lab in cands:
                return i
        return None

    i_buy = find_idx({"buy", "long", "1"})
    i_short = find_idx({"short", "-1"})

    p_buy = float(proba[:, i_buy][0]) if i_buy is not None else float(np.max(proba[0]))
    p_short = float(proba[:, i_short][0]) if i_short is not None else float(np.min(proba[0]))
    return p_buy, p_short


def predict_symbol_fresh(
    symbol: str,
    *,
    ctx,
    clf_raw: Any,
    reg_raw: Any,
    ae_raw: Any,
    flavor_space: Any,
    start_date: str = "1980-01-01",
    end_date: str | None = None,
) -> tuple[pd.Timestamp, pd.Series]:
    """Pull fresh data+features for one symbol and run full scoring stack."""
    sym = symbol.upper().strip()
    end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")

    technical_df, _ = build_technical_dataframe(
        ctx=ctx,
        symbols=[sym],
        start_date=start_date,
        end_date=end_date,
        verbose_data=False,
        skip_on_error=False,
    )
    if technical_df.empty:
        raise RuntimeError(f"No technical data fetched for {sym}.")

    fund_df, _ = build_fundamental_dataframe(
        ctx=ctx,
        symbols=[sym],
        start_date=start_date,
        end_date=end_date,
        target_index=technical_df.index,
        daily_prices=technical_df,
        verbose=False,
    )

    macro_df, _ = build_macro_dataframe(
        ctx=ctx,
        start_date=start_date,
        end_date=end_date,
        target_index=technical_df.index,
        verbose=False,
    )

    features_df = pd.concat([technical_df, fund_df, macro_df], axis=1)

    latest_date = features_df.index.get_level_values("date").max()
    row = features_df.xs(latest_date, level="date").copy()
    if sym not in row.index:
        raise RuntimeError(f"{sym} not present at latest date {latest_date.date()}.")
    row = row.loc[[sym]].copy()

    clf_feats = list(getattr(clf_raw, "_used_features", []))
    reg_feats = list(getattr(reg_raw, "_used_features", []))
    flavor_numeric = list(getattr(flavor_space, "numeric_cols", []))
    required = sorted(set(clf_feats + reg_feats + flavor_numeric))

    for c in required:
        if c not in row.columns:
            row[c] = 0.0

    if "market_position" in clf_feats and "market_position" not in row.columns:
        row["market_position"] = 0

    row_num = row.copy()
    for c in required:
        row_num[c] = pd.to_numeric(row_num[c], errors="coerce")
    row_num[required] = row_num[required].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_clf = row_num[clf_feats]
    X_reg = row_num[reg_feats]

    p_buy, p_short = _class_probs_for_buy_short(clf_raw, X_clf)

    reg_model = getattr(reg_raw, "model", reg_raw)
    ranking = float(np.asarray(reg_model.predict(X_reg), dtype=float).reshape(-1)[0])

    flavor_scored = score_trade_flavor(
        flavor_space=flavor_space,
        entries_df=row_num,
        ae_model=ae_raw,
    )
    cluster_col = str(flavor_space.cluster_col)
    min_dist_col = f"{cluster_col}__min_dist"
    fam_col = f"{cluster_col}__familiarity"

    out = row_num.copy()
    out["clf__prob_buy"] = p_buy
    out["clf__prob_short"] = p_short
    out["ranking"] = ranking

    for c in [cluster_col, min_dist_col, fam_col, "cluster_familiarity"]:
        if c in flavor_scored.columns:
            out[c] = flavor_scored[c]

    if "cluster_familiarity" not in out.columns and fam_col in out.columns:
        out["cluster_familiarity"] = out[fam_col]
    if "cluster_familiarity" not in out.columns:
        out["cluster_familiarity"] = 0.0

    out["buy_score"] = out["clf__prob_buy"] * out["ranking"] * out["cluster_familiarity"]
    out["short_score"] = out["clf__prob_short"] * out["ranking"] * out["cluster_familiarity"]

    out = add_cluster_explanations(
        out,
        flavor_space=flavor_space,
        top_matches=5,
        top_deviations=3,
    )

    return latest_date, out.loc[sym]


def pretty_print_symbol_prediction(symbol: str, latest_dt: pd.Timestamp, pred: pd.Series) -> None:
    s = symbol.upper()

    def get_num(k: str, default=np.nan) -> float:
        v = pred.get(k, default)
        vv = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        return float(vv) if pd.notna(vv) else float(default)

    p_buy = get_num("clf__prob_buy", 0.0)
    p_short = get_num("clf__prob_short", 0.0)
    ranking = get_num("ranking", 0.0)
    fam = get_num("cluster_familiarity", 0.0)
    buy_score = get_num("buy_score", 0.0)
    short_score = get_num("short_score", 0.0)

    cmr = get_num("cluster_mean_return", np.nan)
    csh = get_num("cluster_sharpe", np.nan)
    cmd = get_num("cluster_mean_duration", np.nan)

    matches = pred.get("top_matching_features", []) or []
    devs = pred.get("top_deviating_features", []) or []

    cluster_candidates = [c for c in pred.index if c.endswith("cluster_id") or c == "cluster_id" or c.startswith("cluster_")]
    cluster_val = None
    if "cluster_id" in pred.index:
        cluster_val = pred["cluster_id"]
    elif cluster_candidates:
        cluster_val = pred[cluster_candidates[0]]

    print(f"{s} | {pd.Timestamp(latest_dt).date()}")
    print("-" * 72)
    if cluster_val is not None and pd.notna(cluster_val):
        print(f"Cluster: {cluster_val}")
    print(f"BUY prob:   {p_buy:.3f}")
    print(f"SHORT prob: {p_short:.3f}")
    print(f"Ranking:    {ranking:.3f}")
    print(f"Familiarity:{fam:.3f}")
    print(f"BUY score:  {buy_score:.3f}")
    print(f"SHORT score:{short_score:.3f}")
    print()

    print("Cluster history:")
    print(f"  Mean return:  {cmr:.3f}" if pd.notna(cmr) else "  Mean return:  N/A")
    print(f"  Sharpe:       {csh:.3f}" if pd.notna(csh) else "  Sharpe:       N/A")
    print(f"  Mean duration:{cmd:.1f} days" if pd.notna(cmd) else "  Mean duration:N/A")
    print()

    print("Top matches:")
    if matches:
        for feat, val in list(matches)[:5]:
            print(f"  + {feat}: {val}")
    else:
        print("  (none)")
    print()

    print("Top deviations:")
    if devs:
        for feat, val in list(devs)[:3]:
            print(f"  - {feat}: {val}")
    else:
        print("  (none)")
