from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from modules.api import (
    build_technical_dataframe,
    build_fundamental_dataframe,
    build_macro_dataframe,
)


def _ae_familiarity(ae_model: Any, row_num: pd.DataFrame) -> float:
    artifact = getattr(ae_model, "_artifact", None)
    if artifact is None:
        return 1.0

    numeric_cols = list(getattr(artifact, "numeric_cols", []) or [])
    categorical_cols = list(getattr(artifact, "cat_cols", []) or [])
    if not numeric_cols:
        return 1.0

    for c in numeric_cols:
        if c not in row_num.columns:
            row_num[c] = 0.0
        row_num[c] = pd.to_numeric(row_num[c], errors="coerce")
    row_num[numeric_cols] = row_num[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for c in categorical_cols:
        if c not in row_num.columns:
            row_num[c] = ""

    if hasattr(ae_model, "familiarity"):
        fam = np.asarray(
            ae_model.familiarity(
                row_num,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                quantile=99.9,
                mode="latent_reciprocal_soft",
            ),
            dtype=float,
        ).reshape(-1)
        if len(fam):
            return float(np.clip(fam[0], 0.0, 1.0))
    return 1.0


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
    reg_raw: Any | None = None,
    reg_trade_return_raw: Any | None = None,
    reg_duration_raw: Any | None = None,
    ae_raw: Any,
    flavor_space: Any | None = None,
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
    primary_reg = reg_trade_return_raw if reg_trade_return_raw is not None else reg_raw
    if primary_reg is None:
        raise ValueError("predict_symbol_fresh requires reg_trade_return_raw (or legacy reg_raw).")

    reg_feats = list(getattr(primary_reg, "_used_features", []))
    required = sorted(set(clf_feats + reg_feats))

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

    reg_model = getattr(primary_reg, "model", primary_reg)
    predicted_trade_return = float(np.asarray(reg_model.predict(X_reg), dtype=float).reshape(-1)[0])
    ranking = predicted_trade_return
    predicted_duration_days = np.nan
    if reg_duration_raw is not None:
        reg_dur_feats = list(getattr(reg_duration_raw, "_used_features", []))
        for c in reg_dur_feats:
            if c not in row_num.columns:
                row_num[c] = 0.0
        X_reg_dur = row_num[reg_dur_feats]
        reg_duration_model = getattr(reg_duration_raw, "model", reg_duration_raw)
        predicted_duration_days = float(np.asarray(reg_duration_model.predict(X_reg_dur), dtype=float).reshape(-1)[0])
        predicted_duration_days = max(0.0, predicted_duration_days)

    out = row_num.copy()
    out["clf__prob_buy"] = p_buy
    out["clf__prob_short"] = p_short
    out["ranking"] = ranking
    out["predicted_trade_return"] = predicted_trade_return
    out["predicted_duration_days"] = predicted_duration_days
    out["ae_familiarity"] = _ae_familiarity(ae_raw, row_num)

    if pd.notna(predicted_trade_return):
        out["expected_buy_return"] = out["clf__prob_buy"] * out["predicted_trade_return"] * out["ae_familiarity"]
        out["expected_short_return"] = out["clf__prob_short"] * out["predicted_trade_return"] * out["ae_familiarity"]
        # Backward-compatible aliases.
        out["buy_score"] = out["expected_buy_return"]
        out["short_score"] = out["expected_short_return"]
    else:
        out["buy_score"] = out["clf__prob_buy"] * out["ranking"] * out["ae_familiarity"]
        out["short_score"] = out["clf__prob_short"] * out["ranking"] * out["ae_familiarity"]

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
    predicted_trade_return = get_num("predicted_trade_return", np.nan)
    predicted_duration_days = get_num("predicted_duration_days", np.nan)
    fam = get_num("ae_familiarity", 0.0)
    buy_score = get_num("buy_score", 0.0)
    short_score = get_num("short_score", 0.0)
    expected_buy_return = get_num("expected_buy_return", np.nan)
    expected_short_return = get_num("expected_short_return", np.nan)

    print(f"{s} | {pd.Timestamp(latest_dt).date()}")
    print("-" * 72)
    print(f"BUY prob:   {p_buy:.3f}")
    print(f"SHORT prob: {p_short:.3f}")
    print(f"Ranking:    {ranking:.3f}")
    if pd.notna(predicted_trade_return):
        print(f"Pred ret:   {predicted_trade_return:.4f}")
    if pd.notna(predicted_duration_days):
        print(f"Pred dur:   {predicted_duration_days:.1f} days")
    print(f"AE fam:     {fam:.3f}")
    if pd.notna(expected_buy_return):
        print(f"Exp BUY ret:{expected_buy_return:.4f}")
    print(f"BUY score:  {buy_score:.3f}")
    if pd.notna(expected_short_return):
        print(f"Exp SHRT ret:{expected_short_return:.4f}")
    print(f"SHORT score:{short_score:.3f}")
