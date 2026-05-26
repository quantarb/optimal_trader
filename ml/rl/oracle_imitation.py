from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score


LONG_ONLY_ACTION_TO_ID = {"hold": 0, "buy": 1, "sell": 2}
LONG_ONLY_ID_TO_ACTION = {value: key for key, value in LONG_ONLY_ACTION_TO_ID.items()}
LONG_ONLY_ACTION_LABELS = ["hold", "buy", "sell"]
FLAT_POLICY_ACTION_TO_ID = {"hold": 0, "buy": 1}
FLAT_POLICY_ID_TO_ACTION = {value: key for key, value in FLAT_POLICY_ACTION_TO_ID.items()}
LONG_POLICY_ACTION_TO_ID = {"hold": 0, "sell": 1}
LONG_POLICY_ID_TO_ACTION = {value: key for key, value in LONG_POLICY_ACTION_TO_ID.items()}
LONG_ONLY_POSITION_TO_ID = {"flat": 0, "long": 1}
LONG_ONLY_ID_TO_POSITION = {value: key for key, value in LONG_ONLY_POSITION_TO_ID.items()}
LONG_ONLY_POSITION_LABELS = ["flat", "long"]
POSITION_TO_ID = {"flat": 0, "long": 1, "short": 2}
ID_TO_POSITION = {value: key for key, value in POSITION_TO_ID.items()}
POSITION_LABELS = ["flat", "long", "short"]
POSITION_LABEL_TO_SIGN = {"flat": 0, "long": 1, "short": -1}
POSITION_SIGN_TO_LABEL = {value: key for key, value in POSITION_LABEL_TO_SIGN.items()}
POSITION_POLICY_ACTION_TO_ID = {"hold": 0, "buy": 1, "sell": 2, "short": 3, "cover": 4}
POSITION_POLICY_ID_TO_ACTION = {value: key for key, value in POSITION_POLICY_ACTION_TO_ID.items()}


@dataclass(frozen=True)
class OracleBehaviorCloningResult:
    model: Any
    scored_train_df: pd.DataFrame
    scored_oos_df: pd.DataFrame
    summary_df: pd.DataFrame
    report_df: pd.DataFrame
    feature_importance_df: pd.DataFrame


@dataclass(frozen=True)
class OraclePositionAwareBehaviorCloningResult:
    flat_model: Any
    long_model: Any
    scored_train_df: pd.DataFrame
    scored_oos_df: pd.DataFrame
    summary_df: pd.DataFrame
    report_df: pd.DataFrame
    feature_importance_df: pd.DataFrame


@dataclass(frozen=True)
class OraclePositionCloningResult:
    model: Any
    scored_train_df: pd.DataFrame
    scored_oos_df: pd.DataFrame
    summary_df: pd.DataFrame
    report_df: pd.DataFrame
    feature_importance_df: pd.DataFrame


def _ensure_timestamp_series(frame: pd.DataFrame, *, date_col: str, date_text_col: str) -> pd.Series:
    if date_col in frame.columns:
        out = pd.to_datetime(frame[date_col], errors="coerce")
    elif date_text_col in frame.columns:
        out = pd.to_datetime(frame[date_text_col], errors="coerce")
    else:
        raise KeyError(f"Expected either '{date_col}' or '{date_text_col}' in frame columns.")
    return out


def _resolve_trade_date_text(frame: pd.DataFrame, *, timestamp_col: str, text_col: str) -> pd.Series:
    if text_col in frame.columns:
        return frame[text_col].astype(str)
    if timestamp_col in frame.columns:
        return pd.to_datetime(frame[timestamp_col], errors="coerce").dt.strftime("%Y-%m-%d")
    raise KeyError(f"Expected either '{text_col}' or '{timestamp_col}' in trade frame columns.")


def build_long_only_expert_action_panel(
    daily_state_df: pd.DataFrame,
    trade_pair_df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
    date_text_col: str = "date_text",
) -> pd.DataFrame:
    panel = daily_state_df.copy()
    panel[symbol_col] = panel[symbol_col].astype(str).str.strip().str.upper()
    panel[date_col] = _ensure_timestamp_series(panel, date_col=date_col, date_text_col=date_text_col)
    if date_text_col not in panel.columns:
        panel[date_text_col] = panel[date_col].dt.strftime("%Y-%m-%d")
    panel = panel.dropna(subset=[date_col]).sort_values([symbol_col, date_col]).reset_index(drop=True)
    panel["expert_action"] = "hold"
    panel["expert_action_id"] = LONG_ONLY_ACTION_TO_ID["hold"]

    trades = trade_pair_df.copy()
    trades[symbol_col] = trades[symbol_col].astype(str).str.strip().str.upper()
    trades["side"] = trades["side"].fillna("").astype(str).str.strip().str.lower()
    trades = trades[trades["side"] == "long"].copy()
    if trades.empty:
        panel["expert_position_before"] = 0
        panel["expert_position_after"] = 0
        return panel

    trades["entry_date_text"] = _resolve_trade_date_text(trades, timestamp_col="entry_date", text_col="entry_date_text")
    trades["exit_date_text"] = _resolve_trade_date_text(trades, timestamp_col="exit_date", text_col="exit_date_text")

    event_rows: list[dict[str, Any]] = []
    for row in trades.to_dict(orient="records"):
        trade_id = str(row.get("trade_id") or "").strip()
        symbol = str(row.get(symbol_col) or "").strip().upper()
        entry_date_text = str(row.get("entry_date_text") or "").strip()
        exit_date_text = str(row.get("exit_date_text") or "").strip()
        if not symbol or not entry_date_text or not exit_date_text:
            continue
        event_rows.append(
            {
                symbol_col: symbol,
                date_text_col: entry_date_text,
                "event_trade_id": trade_id,
                "event_action": "buy",
                "action_priority": 1,
            }
        )
        event_rows.append(
            {
                symbol_col: symbol,
                date_text_col: exit_date_text,
                "event_trade_id": trade_id,
                "event_action": "sell",
                "action_priority": 2,
            }
        )

    event_df = pd.DataFrame(event_rows)
    if not event_df.empty:
        event_df = (
            event_df.sort_values([symbol_col, date_text_col, "action_priority"])
            .groupby([symbol_col, date_text_col], as_index=False)
            .agg(
                event_action=("event_action", "last"),
                event_trade_ids=("event_trade_id", lambda s: "|".join(sorted({str(v) for v in s if str(v).strip()}))),
                action_count=("event_action", "size"),
            )
        )
        panel = panel.merge(event_df, on=[symbol_col, date_text_col], how="left")
        has_event = panel["event_action"].notna()
        panel.loc[has_event, "expert_action"] = panel.loc[has_event, "event_action"].astype(str)
        panel["expert_action_id"] = panel["expert_action"].map(LONG_ONLY_ACTION_TO_ID).astype(int)
    else:
        panel["event_trade_ids"] = ""
        panel["action_count"] = 0

    position_before: list[int] = []
    position_after: list[int] = []
    for _, group in panel.groupby(symbol_col, sort=False):
        current_position = 0
        for action in group["expert_action"].tolist():
            position_before.append(current_position)
            if action == "buy":
                current_position = 1
            elif action == "sell":
                current_position = 0
            position_after.append(current_position)
    panel["expert_position_before"] = position_before
    panel["expert_position_after"] = position_after
    return panel


def build_long_only_expert_position_panel(
    daily_state_df: pd.DataFrame,
    trade_pair_df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
    date_text_col: str = "date_text",
) -> pd.DataFrame:
    panel = build_long_only_expert_action_panel(
        daily_state_df,
        trade_pair_df,
        symbol_col=symbol_col,
        date_col=date_col,
        date_text_col=date_text_col,
    ).copy()
    panel["expert_position_label"] = np.where(panel["expert_position_after"].to_numpy(dtype=int) > 0, "long", "flat")
    panel["expert_position_id"] = panel["expert_position_label"].map(LONG_ONLY_POSITION_TO_ID).astype(int)
    panel["entry_signal"] = (panel["expert_action"] == "buy").astype(int)
    panel["exit_signal"] = (panel["expert_action"] == "sell").astype(int)
    return panel


def build_expert_position_panel(
    daily_state_df: pd.DataFrame,
    trade_pair_df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
    date_text_col: str = "date_text",
) -> pd.DataFrame:
    panel = daily_state_df.copy()
    panel[symbol_col] = panel[symbol_col].astype(str).str.strip().str.upper()
    panel[date_col] = _ensure_timestamp_series(panel, date_col=date_col, date_text_col=date_text_col)
    if date_text_col not in panel.columns:
        panel[date_text_col] = panel[date_col].dt.strftime("%Y-%m-%d")
    panel = panel.dropna(subset=[date_col]).sort_values([symbol_col, date_col]).reset_index(drop=True)
    panel["expert_position_label"] = "flat"
    panel["entry_signal"] = 0
    panel["exit_signal"] = 0
    panel["entry_action"] = ""
    panel["exit_action"] = ""
    panel["entry_trade_id"] = ""
    panel["exit_trade_id"] = ""

    trades = trade_pair_df.copy()
    trades[symbol_col] = trades[symbol_col].astype(str).str.strip().str.upper()
    trades["side"] = trades["side"].fillna("").astype(str).str.strip().str.lower()
    trades = trades[trades["side"].isin(["long", "short"])].copy()
    if trades.empty:
        panel["expert_position_id"] = panel["expert_position_label"].map(POSITION_TO_ID).astype(int)
        panel["expert_position_before"] = 0
        panel["expert_position_after"] = 0
        panel["expert_action"] = "hold"
        return panel

    trades["entry_date"] = _ensure_timestamp_series(trades, date_col="entry_date", date_text_col="entry_date_text")
    trades["exit_date"] = _ensure_timestamp_series(trades, date_col="exit_date", date_text_col="exit_date_text")
    if "trade_id" in trades.columns:
        trades["trade_id"] = trades["trade_id"].astype(str)
    else:
        trades["trade_id"] = ""
    trades = trades.dropna(subset=["entry_date", "exit_date"]).sort_values([symbol_col, "entry_date", "exit_date"]).reset_index(drop=True)

    symbol_indices = {symbol: group.index.to_numpy() for symbol, group in panel.groupby(symbol_col, sort=False)}
    symbol_dates = {
        symbol: panel.loc[idx, date_col].to_numpy(dtype="datetime64[ns]")
        for symbol, idx in symbol_indices.items()
    }

    for trade in trades.to_dict(orient="records"):
        symbol = str(trade.get(symbol_col) or "").strip().upper()
        side = str(trade.get("side") or "").strip().lower()
        if symbol not in symbol_indices or side not in {"long", "short"}:
            continue
        entry_date = pd.Timestamp(trade["entry_date"])
        exit_date = pd.Timestamp(trade["exit_date"])
        trade_id = str(trade.get("trade_id") or "").strip()
        date_values = symbol_dates[symbol]
        row_indices = symbol_indices[symbol]
        start = int(np.searchsorted(date_values, entry_date.to_datetime64(), side="left"))
        stop = int(np.searchsorted(date_values, exit_date.to_datetime64(), side="left"))
        if stop <= start:
            stop = min(start + 1, len(row_indices))
        if start < len(row_indices):
            panel.loc[row_indices[start:stop], "expert_position_label"] = side
            panel.loc[row_indices[start], "entry_signal"] = 1
            panel.loc[row_indices[start], "entry_action"] = "buy" if side == "long" else "short"
            panel.loc[row_indices[start], "entry_trade_id"] = trade_id
        if 0 <= stop < len(row_indices):
            panel.loc[row_indices[stop], "exit_signal"] = 1
            panel.loc[row_indices[stop], "exit_action"] = "sell" if side == "long" else "cover"
            panel.loc[row_indices[stop], "exit_trade_id"] = trade_id

    panel["expert_position_id"] = panel["expert_position_label"].map(POSITION_TO_ID).astype(int)
    panel["expert_position_after"] = panel["expert_position_label"].map(POSITION_LABEL_TO_SIGN).astype(int)
    panel["expert_position_before"] = panel.groupby(symbol_col, sort=False)["expert_position_after"].shift(1).fillna(0).astype(int)

    action_values: list[str] = []
    for before_sign, after_sign in panel[["expert_position_before", "expert_position_after"]].itertuples(index=False):
        if before_sign == 0 and after_sign == 1:
            action_values.append("buy")
        elif before_sign == 1 and after_sign == 0:
            action_values.append("sell")
        elif before_sign == 0 and after_sign == -1:
            action_values.append("short")
        elif before_sign == -1 and after_sign == 0:
            action_values.append("cover")
        else:
            action_values.append("hold")
    panel["expert_action"] = action_values
    return panel


def build_transition_sample_weights(
    expert_panel_df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    transition_col: str = "expert_action",
    output_distance_col: str = "transition_distance_steps",
    output_weight_col: str = "sample_weight",
    transition_actions: Sequence[str] = ("buy", "sell", "short", "cover"),
    near_window: int = 3,
    transition_weight: float = 1.0,
    near_weight: float = 0.5,
    interior_weight: float = 0.1,
) -> pd.DataFrame:
    if transition_col not in expert_panel_df.columns:
        raise KeyError(f"Expected '{transition_col}' in expert_panel_df.")

    frame = expert_panel_df.copy()
    frame[symbol_col] = frame[symbol_col].astype(str).str.strip().str.upper()
    action_set = {str(value).strip().lower() for value in transition_actions}
    frame["_transition_flag"] = frame[transition_col].fillna("").astype(str).str.strip().str.lower().isin(action_set)

    all_distances: list[int] = []
    all_weights: list[float] = []
    for _, group in frame.groupby(symbol_col, sort=False):
        transition_idx = np.flatnonzero(group["_transition_flag"].to_numpy(dtype=bool))
        if len(transition_idx) == 0:
            distances = np.full(len(group), near_window + 1, dtype=int)
        else:
            row_idx = np.arange(len(group), dtype=int)
            distances = np.min(np.abs(row_idx[:, None] - transition_idx[None, :]), axis=1)
        weights = np.where(
            distances == 0,
            float(transition_weight),
            np.where(distances <= int(near_window), float(near_weight), float(interior_weight)),
        )
        all_distances.extend(distances.tolist())
        all_weights.extend(weights.tolist())

    frame[output_distance_col] = all_distances
    frame[output_weight_col] = all_weights
    frame.drop(columns=["_transition_flag"], inplace=True)
    return frame


def build_transition_training_mask(
    expert_panel_df: pd.DataFrame,
    *,
    output_keep_col: str = "keep_for_training",
    distance_col: str = "transition_distance_steps",
    symbol_col: str = "symbol",
    transition_col: str = "expert_action",
    transition_actions: Sequence[str] = ("buy", "sell", "short", "cover"),
    near_window: int = 3,
    transition_keep_prob: float = 1.0,
    near_keep_prob: float = 0.5,
    interior_keep_prob: float = 0.1,
    random_state: int = 1337,
) -> pd.DataFrame:
    frame = expert_panel_df.copy()
    if distance_col not in frame.columns:
        frame = build_transition_sample_weights(
            frame,
            symbol_col=symbol_col,
            transition_col=transition_col,
            output_distance_col=distance_col,
            transition_actions=transition_actions,
            near_window=near_window,
        )

    distance_values = pd.to_numeric(frame[distance_col], errors="coerce").fillna(near_window + 1).to_numpy(dtype=float)
    keep_probs = np.where(
        distance_values == 0,
        float(transition_keep_prob),
        np.where(distance_values <= int(near_window), float(near_keep_prob), float(interior_keep_prob)),
    )
    rng = np.random.default_rng(int(random_state))
    keep_mask = rng.random(len(frame)) < keep_probs
    # Never drop true transition rows.
    keep_mask = np.where(distance_values == 0, True, keep_mask)
    frame[output_keep_col] = keep_mask.astype(bool)
    frame["train_keep_probability"] = keep_probs
    return frame


def _build_report_df(y_true: Sequence[int], y_pred: Sequence[int]) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=[LONG_ONLY_ACTION_TO_ID[label] for label in LONG_ONLY_ACTION_LABELS],
        target_names=LONG_ONLY_ACTION_LABELS,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})


def _build_binary_report_df(y_true: Sequence[int], y_pred: Sequence[int], *, labels: list[str], mapping: dict[str, int]) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=[mapping[label] for label in labels],
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})


def _build_position_report_df(y_true: Sequence[int], y_pred: Sequence[int]) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=[LONG_ONLY_POSITION_TO_ID[label] for label in LONG_ONLY_POSITION_LABELS],
        target_names=LONG_ONLY_POSITION_LABELS,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})


def _build_multiclass_report_df(y_true: Sequence[int], y_pred: Sequence[int], *, labels: list[str], mapping: dict[str, int]) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=[mapping[label] for label in labels],
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})


def _default_rf_kwargs() -> dict[str, Any]:
    return {
        "n_estimators": 200,
        "max_depth": 12,
        "max_features": "sqrt",
        "min_samples_leaf": 5,
        "min_samples_split": 2,
        "class_weight": "balanced",
        "random_state": 1337,
        "n_jobs": -1,
    }


def _default_hgb_kwargs() -> dict[str, Any]:
    return {
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "max_depth": 6,
        "min_samples_leaf": 50,
        "l2_regularization": 0.0,
        "random_state": 1337,
        "early_stopping": False,
    }


def train_long_only_behavior_cloning_rf(
    expert_panel_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    train_cutoff: str | pd.Timestamp = "2020-01-01",
    rf_kwargs: dict[str, Any] | None = None,
    symbol_col: str = "symbol",
    date_col: str = "date",
    date_text_col: str = "date_text",
) -> OracleBehaviorCloningResult:
    if "expert_action_id" not in expert_panel_df.columns:
        raise KeyError("expert_panel_df must include 'expert_action_id'.")

    frame = expert_panel_df.copy()
    frame[date_col] = _ensure_timestamp_series(frame, date_col=date_col, date_text_col=date_text_col)
    frame = frame.dropna(subset=[date_col]).reset_index(drop=True)
    for col in feature_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    train_mask = frame[date_col] < pd.Timestamp(train_cutoff)
    train_df = frame.loc[train_mask].reset_index(drop=True)
    oos_df = frame.loc[~train_mask].reset_index(drop=True)
    if train_df.empty:
        raise ValueError("Training panel is empty after applying train_cutoff.")
    if oos_df.empty:
        raise ValueError("Out-of-sample panel is empty after applying train_cutoff.")

    model_kwargs = _default_rf_kwargs()
    if rf_kwargs:
        model_kwargs.update(dict(rf_kwargs))
    model = RandomForestClassifier(**model_kwargs)
    model.fit(train_df[list(feature_cols)], train_df["expert_action_id"])

    def _score(split_name: str, split_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        pred_ids = model.predict(split_df[list(feature_cols)])
        proba = model.predict_proba(split_df[list(feature_cols)])
        class_order = list(model.classes_)
        scored_df = split_df.copy()
        scored_df["pred_action_id"] = pred_ids
        scored_df["pred_action"] = pd.Series(pred_ids, index=scored_df.index).map(LONG_ONLY_ID_TO_ACTION)
        for action_label, action_id in LONG_ONLY_ACTION_TO_ID.items():
            col_name = f"prob_{action_label}"
            if action_id in class_order:
                scored_df[col_name] = proba[:, class_order.index(action_id)]
            else:
                scored_df[col_name] = 0.0
        summary_df = pd.DataFrame(
            [
                {
                    "split": split_name,
                    "rows": int(len(split_df)),
                    "symbols": int(split_df[symbol_col].nunique()) if symbol_col in split_df.columns else int(len(split_df)),
                    "accuracy": float(accuracy_score(split_df["expert_action_id"], pred_ids)),
                    "macro_f1": float(f1_score(split_df["expert_action_id"], pred_ids, average="macro")),
                    "buy_rate": float((scored_df["pred_action"] == "buy").mean()),
                    "sell_rate": float((scored_df["pred_action"] == "sell").mean()),
                    "hold_rate": float((scored_df["pred_action"] == "hold").mean()),
                }
            ]
        )
        return scored_df, summary_df

    scored_train_df, train_summary_df = _score("train", train_df)
    scored_oos_df, oos_summary_df = _score("out_of_sample", oos_df)
    report_df = _build_report_df(oos_df["expert_action_id"], scored_oos_df["pred_action_id"])
    feature_importance_df = pd.DataFrame(
        {
            "feature": list(feature_cols),
            "importance": getattr(model, "feature_importances_", np.zeros(len(feature_cols), dtype=float)),
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    return OracleBehaviorCloningResult(
        model=model,
        scored_train_df=scored_train_df,
        scored_oos_df=scored_oos_df,
        summary_df=pd.concat([train_summary_df, oos_summary_df], ignore_index=True),
        report_df=report_df,
        feature_importance_df=feature_importance_df,
    )


def train_long_only_position_cloning_rf(
    expert_panel_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    train_cutoff: str | pd.Timestamp = "2020-01-01",
    rf_kwargs: dict[str, Any] | None = None,
    symbol_col: str = "symbol",
    date_col: str = "date",
    date_text_col: str = "date_text",
) -> OraclePositionCloningResult:
    if "expert_position_id" not in expert_panel_df.columns:
        raise KeyError("expert_panel_df must include 'expert_position_id'.")

    frame = expert_panel_df.copy()
    frame[date_col] = _ensure_timestamp_series(frame, date_col=date_col, date_text_col=date_text_col)
    frame = frame.dropna(subset=[date_col]).reset_index(drop=True)
    for col in feature_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    train_mask = frame[date_col] < pd.Timestamp(train_cutoff)
    train_df = frame.loc[train_mask].reset_index(drop=True)
    oos_df = frame.loc[~train_mask].reset_index(drop=True)
    if train_df.empty:
        raise ValueError("Training panel is empty after applying train_cutoff.")
    if oos_df.empty:
        raise ValueError("Out-of-sample panel is empty after applying train_cutoff.")

    model_kwargs = _default_rf_kwargs()
    if rf_kwargs:
        model_kwargs.update(dict(rf_kwargs))
    model = RandomForestClassifier(**model_kwargs)
    model.fit(train_df[list(feature_cols)], train_df["expert_position_id"])

    def _score(split_name: str, split_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        pred_ids = model.predict(split_df[list(feature_cols)])
        proba = model.predict_proba(split_df[list(feature_cols)])
        class_order = list(model.classes_)
        scored_df = split_df.copy()
        scored_df["pred_position_id"] = pred_ids
        scored_df["pred_position_label"] = pd.Series(pred_ids, index=scored_df.index).map(LONG_ONLY_ID_TO_POSITION)
        scored_df["prob_flat"] = 0.0
        scored_df["prob_long"] = 0.0
        if LONG_ONLY_POSITION_TO_ID["flat"] in class_order:
            scored_df["prob_flat"] = proba[:, class_order.index(LONG_ONLY_POSITION_TO_ID["flat"])]
        if LONG_ONLY_POSITION_TO_ID["long"] in class_order:
            scored_df["prob_long"] = proba[:, class_order.index(LONG_ONLY_POSITION_TO_ID["long"])]
        summary_df = pd.DataFrame(
            [
                {
                    "split": split_name,
                    "rows": int(len(split_df)),
                    "symbols": int(split_df[symbol_col].nunique()) if symbol_col in split_df.columns else int(len(split_df)),
                    "accuracy": float(accuracy_score(split_df["expert_position_id"], pred_ids)),
                    "macro_f1": float(f1_score(split_df["expert_position_id"], pred_ids, average="macro")),
                    "pred_long_rate": float((scored_df["pred_position_label"] == "long").mean()),
                    "true_long_rate": float((split_df["expert_position_label"] == "long").mean()),
                }
            ]
        )
        return scored_df, summary_df

    scored_train_df, train_summary_df = _score("train", train_df)
    scored_oos_df, oos_summary_df = _score("out_of_sample", oos_df)
    report_df = _build_position_report_df(oos_df["expert_position_id"], scored_oos_df["pred_position_id"])
    feature_importance_df = pd.DataFrame(
        {
            "feature": list(feature_cols),
            "importance": getattr(model, "feature_importances_", np.zeros(len(feature_cols), dtype=float)),
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    return OraclePositionCloningResult(
        model=model,
        scored_train_df=scored_train_df,
        scored_oos_df=scored_oos_df,
        summary_df=pd.concat([train_summary_df, oos_summary_df], ignore_index=True),
        report_df=report_df,
        feature_importance_df=feature_importance_df,
    )


def train_position_cloning_rf(
    expert_panel_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    train_cutoff: str | pd.Timestamp = "2020-01-01",
    rf_kwargs: dict[str, Any] | None = None,
    sample_weight_col: str | None = None,
    train_keep_col: str | None = None,
    symbol_col: str = "symbol",
    date_col: str = "date",
    date_text_col: str = "date_text",
) -> OraclePositionCloningResult:
    if "expert_position_id" not in expert_panel_df.columns:
        raise KeyError("expert_panel_df must include 'expert_position_id'.")

    frame = expert_panel_df.copy()
    frame[date_col] = _ensure_timestamp_series(frame, date_col=date_col, date_text_col=date_text_col)
    frame = frame.dropna(subset=[date_col]).reset_index(drop=True)
    for col in feature_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    train_mask = frame[date_col] < pd.Timestamp(train_cutoff)
    train_df = frame.loc[train_mask].reset_index(drop=True)
    oos_df = frame.loc[~train_mask].reset_index(drop=True)
    if train_keep_col:
        if train_keep_col not in train_df.columns:
            raise KeyError(f"Expected training keep column '{train_keep_col}' in training frame.")
        train_df = train_df[train_df[train_keep_col].fillna(False).astype(bool)].reset_index(drop=True)
    if train_df.empty:
        raise ValueError("Training panel is empty after applying train_cutoff.")
    if oos_df.empty:
        raise ValueError("Out-of-sample panel is empty after applying train_cutoff.")

    model_kwargs = _default_rf_kwargs()
    if rf_kwargs:
        model_kwargs.update(dict(rf_kwargs))
    model = RandomForestClassifier(**model_kwargs)
    fit_kwargs: dict[str, Any] = {}
    if sample_weight_col:
        if sample_weight_col not in train_df.columns:
            raise KeyError(f"Expected sample weight column '{sample_weight_col}' in training frame.")
        fit_kwargs["sample_weight"] = pd.to_numeric(train_df[sample_weight_col], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    model.fit(train_df[list(feature_cols)], train_df["expert_position_id"], **fit_kwargs)

    def _score(split_name: str, split_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        pred_ids = model.predict(split_df[list(feature_cols)])
        proba = model.predict_proba(split_df[list(feature_cols)])
        class_order = list(model.classes_)
        scored_df = split_df.copy()
        scored_df["pred_position_id"] = pred_ids
        scored_df["pred_position_label"] = pd.Series(pred_ids, index=scored_df.index).map(ID_TO_POSITION)
        for position_label in POSITION_LABELS:
            position_id = POSITION_TO_ID[position_label]
            col_name = f"prob_{position_label}"
            if position_id in class_order:
                scored_df[col_name] = proba[:, class_order.index(position_id)]
            else:
                scored_df[col_name] = 0.0
        summary_df = pd.DataFrame(
            [
                {
                    "split": split_name,
                    "rows": int(len(split_df)),
                    "symbols": int(split_df[symbol_col].nunique()) if symbol_col in split_df.columns else int(len(split_df)),
                    "accuracy": float(accuracy_score(split_df["expert_position_id"], pred_ids)),
                    "macro_f1": float(f1_score(split_df["expert_position_id"], pred_ids, average="macro")),
                    "fit_rows": int(len(train_df)) if split_name == "train" else int(len(train_df)),
                    "pred_long_rate": float((scored_df["pred_position_label"] == "long").mean()),
                    "pred_short_rate": float((scored_df["pred_position_label"] == "short").mean()),
                    "true_long_rate": float((split_df["expert_position_label"] == "long").mean()),
                    "true_short_rate": float((split_df["expert_position_label"] == "short").mean()),
                    "true_flat_rate": float((split_df["expert_position_label"] == "flat").mean()),
                    "mean_sample_weight": float(pd.to_numeric(split_df[sample_weight_col], errors="coerce").fillna(1.0).mean()) if sample_weight_col and sample_weight_col in split_df.columns else 1.0,
                }
            ]
        )
        return scored_df, summary_df

    scored_train_df, train_summary_df = _score("train", train_df)
    scored_oos_df, oos_summary_df = _score("out_of_sample", oos_df)
    report_df = _build_multiclass_report_df(
        oos_df["expert_position_id"],
        scored_oos_df["pred_position_id"],
        labels=POSITION_LABELS,
        mapping=POSITION_TO_ID,
    )
    feature_importance_df = pd.DataFrame(
        {
            "feature": list(feature_cols),
            "importance": getattr(model, "feature_importances_", np.zeros(len(feature_cols), dtype=float)),
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    return OraclePositionCloningResult(
        model=model,
        scored_train_df=scored_train_df,
        scored_oos_df=scored_oos_df,
        summary_df=pd.concat([train_summary_df, oos_summary_df], ignore_index=True),
        report_df=report_df,
        feature_importance_df=feature_importance_df,
    )


def train_position_cloning_hgb(
    expert_panel_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    train_cutoff: str | pd.Timestamp = "2020-01-01",
    hgb_kwargs: dict[str, Any] | None = None,
    sample_weight_col: str | None = None,
    train_keep_col: str | None = None,
    symbol_col: str = "symbol",
    date_col: str = "date",
    date_text_col: str = "date_text",
) -> OraclePositionCloningResult:
    if "expert_position_id" not in expert_panel_df.columns:
        raise KeyError("expert_panel_df must include 'expert_position_id'.")

    frame = expert_panel_df.copy()
    frame[date_col] = _ensure_timestamp_series(frame, date_col=date_col, date_text_col=date_text_col)
    frame = frame.dropna(subset=[date_col]).reset_index(drop=True)
    for col in feature_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    train_mask = frame[date_col] < pd.Timestamp(train_cutoff)
    train_df = frame.loc[train_mask].reset_index(drop=True)
    oos_df = frame.loc[~train_mask].reset_index(drop=True)
    if train_keep_col:
        if train_keep_col not in train_df.columns:
            raise KeyError(f"Expected training keep column '{train_keep_col}' in training frame.")
        train_df = train_df[train_df[train_keep_col].fillna(False).astype(bool)].reset_index(drop=True)
    if train_df.empty:
        raise ValueError("Training panel is empty after applying train_cutoff.")
    if oos_df.empty:
        raise ValueError("Out-of-sample panel is empty after applying train_cutoff.")

    model_kwargs = _default_hgb_kwargs()
    if hgb_kwargs:
        model_kwargs.update(dict(hgb_kwargs))
    model = HistGradientBoostingClassifier(**model_kwargs)
    fit_kwargs: dict[str, Any] = {}
    if sample_weight_col:
        if sample_weight_col not in train_df.columns:
            raise KeyError(f"Expected sample weight column '{sample_weight_col}' in training frame.")
        fit_kwargs["sample_weight"] = pd.to_numeric(train_df[sample_weight_col], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    model.fit(train_df[list(feature_cols)], train_df["expert_position_id"], **fit_kwargs)

    def _score(split_name: str, split_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        pred_ids = model.predict(split_df[list(feature_cols)])
        proba = model.predict_proba(split_df[list(feature_cols)])
        class_order = list(model.classes_)
        scored_df = split_df.copy()
        scored_df["pred_position_id"] = pred_ids
        scored_df["pred_position_label"] = pd.Series(pred_ids, index=scored_df.index).map(ID_TO_POSITION)
        for position_label in POSITION_LABELS:
            position_id = POSITION_TO_ID[position_label]
            col_name = f"prob_{position_label}"
            if position_id in class_order:
                scored_df[col_name] = proba[:, class_order.index(position_id)]
            else:
                scored_df[col_name] = 0.0
        summary_df = pd.DataFrame(
            [
                {
                    "split": split_name,
                    "rows": int(len(split_df)),
                    "symbols": int(split_df[symbol_col].nunique()) if symbol_col in split_df.columns else int(len(split_df)),
                    "accuracy": float(accuracy_score(split_df["expert_position_id"], pred_ids)),
                    "macro_f1": float(f1_score(split_df["expert_position_id"], pred_ids, average="macro")),
                    "fit_rows": int(len(train_df)),
                    "pred_long_rate": float((scored_df["pred_position_label"] == "long").mean()),
                    "pred_short_rate": float((scored_df["pred_position_label"] == "short").mean()),
                    "true_long_rate": float((split_df["expert_position_label"] == "long").mean()),
                    "true_short_rate": float((split_df["expert_position_label"] == "short").mean()),
                    "true_flat_rate": float((split_df["expert_position_label"] == "flat").mean()),
                    "mean_sample_weight": float(pd.to_numeric(split_df[sample_weight_col], errors="coerce").fillna(1.0).mean()) if sample_weight_col and sample_weight_col in split_df.columns else 1.0,
                }
            ]
        )
        return scored_df, summary_df

    scored_train_df, train_summary_df = _score("train", train_df)
    scored_oos_df, oos_summary_df = _score("out_of_sample", oos_df)
    report_df = _build_multiclass_report_df(
        oos_df["expert_position_id"],
        scored_oos_df["pred_position_id"],
        labels=POSITION_LABELS,
        mapping=POSITION_TO_ID,
    )
    feature_importance_df = pd.DataFrame(columns=["feature", "importance"])

    return OraclePositionCloningResult(
        model=model,
        scored_train_df=scored_train_df,
        scored_oos_df=scored_oos_df,
        summary_df=pd.concat([train_summary_df, oos_summary_df], ignore_index=True),
        report_df=report_df,
        feature_importance_df=feature_importance_df,
    )


def train_position_aware_long_only_behavior_cloning_rf(
    expert_panel_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    train_cutoff: str | pd.Timestamp = "2020-01-01",
    rf_kwargs: dict[str, Any] | None = None,
    symbol_col: str = "symbol",
    date_col: str = "date",
    date_text_col: str = "date_text",
) -> OraclePositionAwareBehaviorCloningResult:
    required_cols = {"expert_action", "expert_action_id", "expert_position_before"}
    missing = required_cols.difference(expert_panel_df.columns)
    if missing:
        raise KeyError(f"expert_panel_df is missing required columns: {sorted(missing)}")

    frame = expert_panel_df.copy()
    frame[date_col] = _ensure_timestamp_series(frame, date_col=date_col, date_text_col=date_text_col)
    frame = frame.dropna(subset=[date_col]).reset_index(drop=True)
    for col in feature_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    train_mask = frame[date_col] < pd.Timestamp(train_cutoff)
    train_df = frame.loc[train_mask].reset_index(drop=True)
    oos_df = frame.loc[~train_mask].reset_index(drop=True)
    if train_df.empty:
        raise ValueError("Training panel is empty after applying train_cutoff.")
    if oos_df.empty:
        raise ValueError("Out-of-sample panel is empty after applying train_cutoff.")

    flat_train_df = train_df[
        (train_df["expert_position_before"] == 0) & (train_df["expert_action"].isin(["hold", "buy"]))
    ].copy()
    long_train_df = train_df[
        (train_df["expert_position_before"] == 1) & (train_df["expert_action"].isin(["hold", "sell"]))
    ].copy()
    if flat_train_df.empty or long_train_df.empty:
        raise ValueError("Need both flat and long expert states to train the position-aware policy.")

    model_kwargs = _default_rf_kwargs()
    if rf_kwargs:
        model_kwargs.update(dict(rf_kwargs))

    flat_target = flat_train_df["expert_action"].map(FLAT_POLICY_ACTION_TO_ID)
    long_target = long_train_df["expert_action"].map(LONG_POLICY_ACTION_TO_ID)
    flat_model = RandomForestClassifier(**model_kwargs)
    long_model = RandomForestClassifier(**model_kwargs)
    flat_model.fit(flat_train_df[list(feature_cols)], flat_target)
    long_model.fit(long_train_df[list(feature_cols)], long_target)

    def _score(split_name: str, split_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        scored_df = split_df.copy()
        scored_df["pred_action"] = "hold"
        scored_df["pred_action_id"] = LONG_ONLY_ACTION_TO_ID["hold"]
        scored_df["prob_hold"] = 1.0
        scored_df["prob_buy"] = 0.0
        scored_df["prob_sell"] = 0.0
        scored_df["policy_state"] = np.where(scored_df["expert_position_before"].to_numpy(dtype=int) == 0, "flat", "long")

        flat_mask = scored_df["expert_position_before"] == 0
        if flat_mask.any():
            flat_pred = flat_model.predict(scored_df.loc[flat_mask, list(feature_cols)])
            flat_proba = flat_model.predict_proba(scored_df.loc[flat_mask, list(feature_cols)])
            flat_classes = list(flat_model.classes_)
            scored_df.loc[flat_mask, "pred_action"] = pd.Series(flat_pred, index=scored_df.index[flat_mask]).map(FLAT_POLICY_ID_TO_ACTION)
            scored_df.loc[flat_mask, "pred_action_id"] = scored_df.loc[flat_mask, "pred_action"].map(LONG_ONLY_ACTION_TO_ID).astype(int)
            if FLAT_POLICY_ACTION_TO_ID["hold"] in flat_classes:
                scored_df.loc[flat_mask, "prob_hold"] = flat_proba[:, flat_classes.index(FLAT_POLICY_ACTION_TO_ID["hold"])]
            if FLAT_POLICY_ACTION_TO_ID["buy"] in flat_classes:
                scored_df.loc[flat_mask, "prob_buy"] = flat_proba[:, flat_classes.index(FLAT_POLICY_ACTION_TO_ID["buy"])]

        long_mask = scored_df["expert_position_before"] == 1
        if long_mask.any():
            long_pred = long_model.predict(scored_df.loc[long_mask, list(feature_cols)])
            long_proba = long_model.predict_proba(scored_df.loc[long_mask, list(feature_cols)])
            long_classes = list(long_model.classes_)
            scored_df.loc[long_mask, "pred_action"] = pd.Series(long_pred, index=scored_df.index[long_mask]).map(LONG_POLICY_ID_TO_ACTION)
            scored_df.loc[long_mask, "pred_action_id"] = scored_df.loc[long_mask, "pred_action"].map(LONG_ONLY_ACTION_TO_ID).astype(int)
            if LONG_POLICY_ACTION_TO_ID["hold"] in long_classes:
                scored_df.loc[long_mask, "prob_hold"] = long_proba[:, long_classes.index(LONG_POLICY_ACTION_TO_ID["hold"])]
            if LONG_POLICY_ACTION_TO_ID["sell"] in long_classes:
                scored_df.loc[long_mask, "prob_sell"] = long_proba[:, long_classes.index(LONG_POLICY_ACTION_TO_ID["sell"])]

        summary_rows = []
        report_frames = []
        for policy_state, labels, mapping in (
            ("flat", ["hold", "buy"], FLAT_POLICY_ACTION_TO_ID),
            ("long", ["hold", "sell"], LONG_POLICY_ACTION_TO_ID),
        ):
            state_df = scored_df[scored_df["policy_state"] == policy_state].copy()
            if state_df.empty:
                continue
            y_true = state_df["expert_action"].map(mapping)
            y_pred = state_df["pred_action"].map(mapping)
            valid_mask = y_true.notna() & y_pred.notna()
            state_df = state_df.loc[valid_mask].copy()
            y_true = y_true.loc[valid_mask].astype(int)
            y_pred = y_pred.loc[valid_mask].astype(int)
            if state_df.empty:
                continue
            summary_rows.append(
                {
                    "split": split_name,
                    "policy_state": policy_state,
                    "rows": int(len(state_df)),
                    "symbols": int(state_df[symbol_col].nunique()) if symbol_col in state_df.columns else int(len(state_df)),
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                    "act_rate": float((state_df["pred_action"] != "hold").mean()),
                }
            )
            report_df = _build_binary_report_df(y_true, y_pred, labels=labels, mapping=mapping)
            report_df.insert(0, "policy_state", policy_state)
            report_df.insert(0, "split", split_name)
            report_frames.append(report_df)
        return scored_df, pd.DataFrame(summary_rows), pd.concat(report_frames, ignore_index=True)

    scored_train_df, train_summary_df, train_report_df = _score("train", train_df)
    scored_oos_df, oos_summary_df, oos_report_df = _score("out_of_sample", oos_df)
    feature_importance_df = pd.concat(
        [
            pd.DataFrame({"model": "flat_policy", "feature": list(feature_cols), "importance": getattr(flat_model, "feature_importances_", np.zeros(len(feature_cols), dtype=float))}),
            pd.DataFrame({"model": "long_policy", "feature": list(feature_cols), "importance": getattr(long_model, "feature_importances_", np.zeros(len(feature_cols), dtype=float))}),
        ],
        ignore_index=True,
    ).sort_values(["model", "importance"], ascending=[True, False]).reset_index(drop=True)

    return OraclePositionAwareBehaviorCloningResult(
        flat_model=flat_model,
        long_model=long_model,
        scored_train_df=scored_train_df,
        scored_oos_df=scored_oos_df,
        summary_df=pd.concat([train_summary_df, oos_summary_df], ignore_index=True),
        report_df=pd.concat([train_report_df, oos_report_df], ignore_index=True),
        feature_importance_df=feature_importance_df,
    )


def rollout_position_aware_long_only_policy(
    daily_state_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    flat_model: Any,
    long_model: Any,
    symbol_col: str = "symbol",
    date_col: str = "date",
    date_text_col: str = "date_text",
) -> pd.DataFrame:
    frame = daily_state_df.copy()
    frame[date_col] = _ensure_timestamp_series(frame, date_col=date_col, date_text_col=date_text_col)
    if date_text_col not in frame.columns:
        frame[date_text_col] = frame[date_col].dt.strftime("%Y-%m-%d")
    frame = frame.dropna(subset=[date_col]).sort_values([symbol_col, date_col]).reset_index(drop=True)
    for col in feature_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    sim_position_before: list[int] = []
    sim_position_after: list[int] = []
    pred_action_values: list[str] = []
    pred_action_ids: list[int] = []
    prob_hold_values: list[float] = []
    prob_buy_values: list[float] = []
    prob_sell_values: list[float] = []
    for symbol, group in frame.groupby(symbol_col, sort=False):
        current_position = 0
        ordered = group.sort_values(date_col).reset_index(drop=True)
        for row in ordered.to_dict(orient="records"):
            features_df = pd.DataFrame([{col: row.get(col, 0.0) for col in feature_cols}])
            sim_position_before.append(current_position)
            if current_position == 0:
                pred_id = int(flat_model.predict(features_df)[0])
                proba = flat_model.predict_proba(features_df)[0]
                classes = list(flat_model.classes_)
                pred_action = FLAT_POLICY_ID_TO_ACTION[pred_id]
                prob_hold = float(proba[classes.index(FLAT_POLICY_ACTION_TO_ID["hold"])]) if FLAT_POLICY_ACTION_TO_ID["hold"] in classes else 0.0
                prob_buy = float(proba[classes.index(FLAT_POLICY_ACTION_TO_ID["buy"])]) if FLAT_POLICY_ACTION_TO_ID["buy"] in classes else 0.0
                prob_sell = 0.0
                if pred_action == "buy":
                    current_position = 1
            else:
                pred_id = int(long_model.predict(features_df)[0])
                proba = long_model.predict_proba(features_df)[0]
                classes = list(long_model.classes_)
                pred_action = LONG_POLICY_ID_TO_ACTION[pred_id]
                prob_hold = float(proba[classes.index(LONG_POLICY_ACTION_TO_ID["hold"])]) if LONG_POLICY_ACTION_TO_ID["hold"] in classes else 0.0
                prob_buy = 0.0
                prob_sell = float(proba[classes.index(LONG_POLICY_ACTION_TO_ID["sell"])]) if LONG_POLICY_ACTION_TO_ID["sell"] in classes else 0.0
                if pred_action == "sell":
                    current_position = 0
            sim_position_after.append(current_position)
            pred_action_values.append(pred_action)
            pred_action_ids.append(int(LONG_ONLY_ACTION_TO_ID[pred_action]))
            prob_hold_values.append(prob_hold)
            prob_buy_values.append(prob_buy)
            prob_sell_values.append(prob_sell)

    rollout_df = frame.copy().reset_index(drop=True)
    rollout_df["sim_position_before"] = sim_position_before
    rollout_df["sim_position_after"] = sim_position_after
    rollout_df["pred_action"] = pred_action_values
    rollout_df["pred_action_id"] = pred_action_ids
    rollout_df["prob_hold"] = prob_hold_values
    rollout_df["prob_buy"] = prob_buy_values
    rollout_df["prob_sell"] = prob_sell_values
    return rollout_df


def rollout_long_only_position_policy(
    daily_state_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    model: Any,
    symbol_col: str = "symbol",
    date_col: str = "date",
    date_text_col: str = "date_text",
) -> pd.DataFrame:
    frame = daily_state_df.copy()
    frame[date_col] = _ensure_timestamp_series(frame, date_col=date_col, date_text_col=date_text_col)
    if date_text_col not in frame.columns:
        frame[date_text_col] = frame[date_col].dt.strftime("%Y-%m-%d")
    frame = frame.dropna(subset=[date_col]).sort_values([symbol_col, date_col]).reset_index(drop=True)
    for col in feature_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    pred_ids = model.predict(frame[list(feature_cols)])
    proba = model.predict_proba(frame[list(feature_cols)])
    class_order = list(model.classes_)
    rollout_df = frame.copy()
    rollout_df["pred_position_id"] = pred_ids
    rollout_df["pred_position_label"] = pd.Series(pred_ids, index=rollout_df.index).map(LONG_ONLY_ID_TO_POSITION)
    rollout_df["prob_flat"] = 0.0
    rollout_df["prob_long"] = 0.0
    if LONG_ONLY_POSITION_TO_ID["flat"] in class_order:
        rollout_df["prob_flat"] = proba[:, class_order.index(LONG_ONLY_POSITION_TO_ID["flat"])]
    if LONG_ONLY_POSITION_TO_ID["long"] in class_order:
        rollout_df["prob_long"] = proba[:, class_order.index(LONG_ONLY_POSITION_TO_ID["long"])]

    sim_position_before: list[int] = []
    sim_position_after: list[int] = []
    pred_action_values: list[str] = []
    pred_action_ids: list[int] = []
    for _, group in rollout_df.groupby(symbol_col, sort=False):
        current_position = 0
        ordered = group.sort_values(date_col)
        for _, row in ordered.iterrows():
            desired_position = 1 if str(row["pred_position_label"]) == "long" else 0
            sim_position_before.append(current_position)
            if current_position == 0 and desired_position == 1:
                pred_action = "buy"
                current_position = 1
            elif current_position == 1 and desired_position == 0:
                pred_action = "sell"
                current_position = 0
            else:
                pred_action = "hold"
            sim_position_after.append(current_position)
            pred_action_values.append(pred_action)
            pred_action_ids.append(int(LONG_ONLY_ACTION_TO_ID[pred_action]))

    rollout_df["sim_position_before"] = sim_position_before
    rollout_df["sim_position_after"] = sim_position_after
    rollout_df["pred_action"] = pred_action_values
    rollout_df["pred_action_id"] = pred_action_ids
    return rollout_df


def rollout_position_policy(
    daily_state_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    model: Any,
    symbol_col: str = "symbol",
    date_col: str = "date",
    date_text_col: str = "date_text",
) -> pd.DataFrame:
    frame = daily_state_df.copy()
    frame[date_col] = _ensure_timestamp_series(frame, date_col=date_col, date_text_col=date_text_col)
    if date_text_col not in frame.columns:
        frame[date_text_col] = frame[date_col].dt.strftime("%Y-%m-%d")
    frame = frame.dropna(subset=[date_col]).sort_values([symbol_col, date_col]).reset_index(drop=True)
    for col in feature_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    pred_ids = model.predict(frame[list(feature_cols)])
    proba = model.predict_proba(frame[list(feature_cols)])
    class_order = list(model.classes_)
    rollout_df = frame.copy()
    rollout_df["pred_position_id"] = pred_ids
    rollout_df["pred_position_label"] = pd.Series(pred_ids, index=rollout_df.index).map(ID_TO_POSITION)
    for position_label in POSITION_LABELS:
        position_id = POSITION_TO_ID[position_label]
        col_name = f"prob_{position_label}"
        if position_id in class_order:
            rollout_df[col_name] = proba[:, class_order.index(position_id)]
        else:
            rollout_df[col_name] = 0.0

    sim_position_before: list[int] = []
    sim_position_after: list[int] = []
    pred_action_values: list[str] = []
    pred_action_ids: list[int] = []
    for _, group in rollout_df.groupby(symbol_col, sort=False):
        current_position = 0
        ordered = group.sort_values(date_col)
        for _, row in ordered.iterrows():
            desired_position = int(POSITION_LABEL_TO_SIGN.get(str(row["pred_position_label"]), 0))
            sim_position_before.append(current_position)
            if current_position == 0 and desired_position == 1:
                pred_action = "buy"
                current_position = 1
            elif current_position == 0 and desired_position == -1:
                pred_action = "short"
                current_position = -1
            elif current_position == 1 and desired_position == 0:
                pred_action = "sell"
                current_position = 0
            elif current_position == -1 and desired_position == 0:
                pred_action = "cover"
                current_position = 0
            elif current_position == 1 and desired_position == -1:
                pred_action = "short"
                current_position = -1
            elif current_position == -1 and desired_position == 1:
                pred_action = "buy"
                current_position = 1
            else:
                pred_action = "hold"
            sim_position_after.append(current_position)
            pred_action_values.append(pred_action)
            pred_action_ids.append(int(POSITION_POLICY_ACTION_TO_ID[pred_action]))

    rollout_df["sim_position_before"] = sim_position_before
    rollout_df["sim_position_after"] = sim_position_after
    rollout_df["sim_position_label_after"] = pd.Series(sim_position_after, index=rollout_df.index).map(POSITION_SIGN_TO_LABEL)
    rollout_df["pred_action"] = pred_action_values
    rollout_df["pred_action_id"] = pred_action_ids
    return rollout_df


def backtest_position_policy_equal_weight(
    frame: pd.DataFrame,
    *,
    position_col: str,
    close_col: str = "adj_close",
    symbol_col: str = "symbol",
    date_col: str = "date",
    date_text_col: str = "date_text",
    initial_balance: float = 1_000_000.0,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if position_col not in frame.columns:
        raise KeyError(f"Expected '{position_col}' in frame columns.")

    working = frame.copy()
    working[date_col] = _ensure_timestamp_series(working, date_col=date_col, date_text_col=date_text_col)
    working = working.dropna(subset=[date_col, symbol_col, close_col]).copy()
    working[symbol_col] = working[symbol_col].astype(str).str.strip().str.upper()
    working[close_col] = pd.to_numeric(working[close_col], errors="coerce")
    working = working.dropna(subset=[close_col]).sort_values([symbol_col, date_col]).reset_index(drop=True)

    if pd.api.types.is_numeric_dtype(working[position_col]):
        working["position_sign"] = pd.to_numeric(working[position_col], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    else:
        working["position_sign"] = working[position_col].astype(str).map(POSITION_LABEL_TO_SIGN).fillna(0).astype(float)

    working["next_close"] = working.groupby(symbol_col, sort=False)[close_col].shift(-1)
    working["next_return"] = np.where(
        working["next_close"].notna() & (working[close_col].abs() > 1e-12),
        (working["next_close"] / working[close_col]) - 1.0,
        np.nan,
    )
    working["prev_position_sign"] = working.groupby(symbol_col, sort=False)["position_sign"].shift(1).fillna(0.0)
    working["turnover_units"] = (working["position_sign"] - working["prev_position_sign"]).abs()
    trade_cost_rate = float(fee_bps + slippage_bps) / 10000.0
    working["trade_cost_return"] = working["turnover_units"] * trade_cost_rate
    working["strategy_return"] = (working["position_sign"] * working["next_return"]) - working["trade_cost_return"]

    valid_df = working[working["next_return"].notna()].copy()
    daily_return_s = valid_df.groupby(date_col, sort=True)["strategy_return"].mean()
    equity_s = (1.0 + daily_return_s).cumprod() * float(initial_balance)
    return valid_df, equity_s, daily_return_s


__all__ = [
    "LONG_ONLY_ACTION_LABELS",
    "LONG_ONLY_ACTION_TO_ID",
    "LONG_ONLY_ID_TO_ACTION",
    "FLAT_POLICY_ACTION_TO_ID",
    "FLAT_POLICY_ID_TO_ACTION",
    "LONG_POLICY_ACTION_TO_ID",
    "LONG_POLICY_ID_TO_ACTION",
    "LONG_ONLY_POSITION_TO_ID",
    "LONG_ONLY_ID_TO_POSITION",
    "LONG_ONLY_POSITION_LABELS",
    "POSITION_TO_ID",
    "ID_TO_POSITION",
    "POSITION_LABELS",
    "POSITION_LABEL_TO_SIGN",
    "POSITION_SIGN_TO_LABEL",
    "POSITION_POLICY_ACTION_TO_ID",
    "POSITION_POLICY_ID_TO_ACTION",
    "OracleBehaviorCloningResult",
    "OraclePositionCloningResult",
    "OraclePositionAwareBehaviorCloningResult",
    "build_long_only_expert_action_panel",
    "build_long_only_expert_position_panel",
    "build_expert_position_panel",
    "build_transition_sample_weights",
    "build_transition_training_mask",
    "backtest_position_policy_equal_weight",
    "rollout_long_only_position_policy",
    "rollout_position_policy",
    "rollout_position_aware_long_only_policy",
    "train_long_only_behavior_cloning_rf",
    "train_long_only_position_cloning_rf",
    "train_position_cloning_hgb",
    "train_position_cloning_rf",
    "train_position_aware_long_only_behavior_cloning_rf",
]
