# Notebook-parity helpers for latest-day inference.

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from modules.utils.llm_prompts import build_llm_guardrail_prompt_from_results


def _safe_label_token(v: object) -> str:
    s = str(v).strip().lower()
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "_")
    token = "".join(out).strip("_")
    return token or "class"


def make_autoencoder_familiarity_predictor(
    numeric_cols: list[str],
    *,
    quantile: float = 99.9,
    mode: str = "latent_reciprocal_soft",
) -> Callable[[pd.DataFrame, Any], np.ndarray]:
    """Factory for AE familiarity predictor used by the generic latest-day scorer."""

    def _predict(latest_df: pd.DataFrame, ae_model: Any) -> np.ndarray:
        cat_cols = list(getattr(ae_model, "_artifact", None).cat_cols) if getattr(ae_model, "_artifact", None) is not None else []
        if hasattr(ae_model, "familiarity"):
            return np.asarray(
                ae_model.familiarity(
                    latest_df,
                    numeric_cols=numeric_cols,
                    categorical_cols=cat_cols,
                    quantile=quantile,
                    mode=mode,
                ),
                dtype=float,
            )

        # Backward-compatible fallback for older AE wrappers.
        artifact = getattr(ae_model, "_artifact", None)
        raw_model = None
        if artifact is not None:
            for attr in ("model", "net", "nn", "module"):
                candidate = getattr(artifact, attr, None)
                if isinstance(candidate, nn.Module):
                    raw_model = candidate
                    break
        if raw_model is None:
            raise AttributeError("Could not locate torch model.")

        x_num = torch.tensor(
            latest_df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values,
            dtype=torch.float32,
        )
        raw_model.eval()
        with torch.no_grad():
            reconstructed = raw_model(x_num).numpy()
        raw_mse = np.mean((x_num.numpy() - reconstructed) ** 2, axis=1)
        return (1.0 - pd.Series(raw_mse).rank(pct=True).values).astype(float)

    return _predict


def _get_class_probability_columns(
    model_clf,
    clf_wrapper,
    x: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Return ({prob_<label>: values}, ordered probability columns)."""
    proba = np.asarray(model_clf.predict_proba(x), dtype=float)
    classes = list(getattr(model_clf, "classes_", []))
    class_mapping = getattr(clf_wrapper, "_class_mapping", None)

    class_to_label: dict[object, str] = {}
    if class_mapping and classes:
        for cls in classes:
            class_to_label[cls] = _safe_label_token(class_mapping.get(cls, cls))
    elif classes:
        for cls in classes:
            class_to_label[cls] = _safe_label_token(cls)
    else:
        for i in range(proba.shape[1]):
            class_to_label[i] = f"class{i}"
        classes = list(class_to_label.keys())

    prob_cols: dict[str, np.ndarray] = {}
    ordered_cols: list[str] = []
    for i, cls in enumerate(classes):
        col = f"prob_{class_to_label.get(cls, _safe_label_token(cls))}"
        if col in prob_cols:
            col = f"{col}_{i}"
        prob_cols[col] = proba[:, i]
        ordered_cols.append(col)
    return prob_cols, ordered_cols


def _as_array(values: Any, n: int) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError("predict_fn returned DataFrame with multiple columns; return Series/array or map explicitly.")
        values = values.iloc[:, 0]
    if isinstance(values, pd.Series):
        arr = values.to_numpy()
    else:
        arr = np.asarray(values)
    arr = arr.reshape(-1)
    if len(arr) != n:
        raise ValueError(f"Prediction length mismatch: got {len(arr)} rows, expected {n}.")
    return arr.astype(float)


def _date_level_name(panel_df: pd.DataFrame) -> str:
    if not isinstance(panel_df.index, pd.MultiIndex):
        raise TypeError("train_data must have MultiIndex (date, symbol).")
    return "date" if "date" in panel_df.index.names else panel_df.index.names[0]


def _symbol_level_name(panel_df: pd.DataFrame) -> str:
    if not isinstance(panel_df.index, pd.MultiIndex):
        raise TypeError("train_data must have MultiIndex (date, symbol).")
    return "symbol" if "symbol" in panel_df.index.names else panel_df.index.names[-1]


def _normalize_ts(v: Any) -> pd.Timestamp:
    return pd.Timestamp(v)


def _apply_market_position(
    df_slice: pd.DataFrame,
    *,
    market_position_value: int | Mapping[Any, Any] | pd.Series | None,
    market_position_col: str,
) -> pd.DataFrame:
    out = df_slice.copy()
    if market_position_value is None:
        return out
    if isinstance(market_position_value, Mapping):
        out[market_position_col] = out.index.map(lambda s: market_position_value.get(s, 0)).astype(int)
        return out
    if isinstance(market_position_value, pd.Series):
        out[market_position_col] = out.index.map(lambda s: market_position_value.get(s, 0)).astype(int)
        return out
    out[market_position_col] = int(market_position_value)
    return out


def _apply_model_specs(latest_df: pd.DataFrame, model_specs: list[tuple[Any, str] | Mapping[str, Any]]) -> pd.DataFrame:
    out = latest_df.copy()
    n = len(out)
    for spec in model_specs:
        if isinstance(spec, tuple):
            if len(spec) != 2:
                raise ValueError("Tuple model spec must be (model, prediction_column).")
            model_obj, pred_col = spec
            cfg: dict[str, Any] = {"model": model_obj, "pred_col": str(pred_col)}
        else:
            cfg = dict(spec)

        model_obj = cfg.get("model")
        pred_col = str(cfg.get("pred_col"))
        if model_obj is None or not pred_col:
            raise ValueError("Each model spec must provide 'model' and 'pred_col'.")

        predict_fn = cfg.get("predict_fn")
        feature_cols = cfg.get("feature_cols")
        include_class_probs = bool(cfg.get("include_class_probs", True))
        raw_model = getattr(model_obj, "model", model_obj)

        if callable(predict_fn):
            out[pred_col] = _as_array(predict_fn(out.copy(), model_obj), n)
            continue

        if feature_cols is None:
            feature_cols = list(getattr(model_obj, "_used_features", out.columns))
        x = out[list(feature_cols)]

        if hasattr(raw_model, "predict_proba"):
            prob_cols, ordered_cols = _get_class_probability_columns(raw_model, model_obj, x)
            if include_class_probs:
                for c, v in prob_cols.items():
                    out[f"{pred_col}__{c}"] = v
            out[pred_col] = _select_primary_probability(ordered_cols, pd.DataFrame(prob_cols))
        elif hasattr(raw_model, "predict"):
            out[pred_col] = _as_array(raw_model.predict(x), n)
        else:
            raise ValueError(
                f"Model spec for '{pred_col}' has no predict_fn and model has neither predict_proba nor predict."
            )
    return out


def _select_primary_probability(prob_cols: list[str], latest_df: pd.DataFrame) -> np.ndarray:
    """Choose a generic classifier probability for legacy combined_score."""
    if not prob_cols:
        return np.full(len(latest_df), np.nan, dtype=float)
    if len(prob_cols) == 1:
        return latest_df[prob_cols[0]].to_numpy(dtype=float)
    if len(prob_cols) == 2:
        if "prob_1" in latest_df.columns:
            return latest_df["prob_1"].to_numpy(dtype=float)
        return latest_df[prob_cols[1]].to_numpy(dtype=float)
    return latest_df[prob_cols].max(axis=1).to_numpy(dtype=float)


def predict_latest_day_with_all_models(
    train_data: pd.DataFrame,
    clf,
    reg,
    ae_model,
    spec_ae,
    numeric_cols,
    categorical_cols,
    confidence_cutoff=0.50,
    build_llm_guardrail_prompt=True,
    llm_top_k=15,
    known_events="NONE",
    market_position_value: int | None = None,
    market_position_col: str = "market_position",
):
    """Score the latest date in a (date, symbol) panel with all available models."""
    if not isinstance(train_data.index, pd.MultiIndex):
        raise TypeError("train_data must have MultiIndex (date, symbol).")

    date_level = "date" if "date" in train_data.index.names else train_data.index.names[0]
    latest_date = train_data.index.get_level_values(date_level).max()
    latest_df = train_data.xs(latest_date, level=date_level).copy()
    if market_position_value is not None:
        latest_df[market_position_col] = int(market_position_value)

    # Keep these args for notebook/API compatibility.
    _ = (spec_ae, categorical_cols)

    # RF Classifier
    prob_cols: list[str] = []
    if clf is not None:
        used_clf = list(getattr(clf, "_used_features", latest_df.columns))
        model_clf = getattr(clf, "model", clf)
        all_prob_cols, prob_cols = _get_class_probability_columns(model_clf, clf, latest_df[used_clf])
        for col, arr in all_prob_cols.items():
            latest_df[col] = arr
        latest_df["pred_rf_cls_proba"] = _select_primary_probability(prob_cols, latest_df)
    else:
        latest_df["pred_rf_cls_proba"] = np.nan

    # RF Regressor
    if reg is not None:
        used_reg = list(getattr(reg, "_used_features", latest_df.columns))
        model_reg = getattr(reg, "model", reg)
        latest_df["pred_rf_reg"] = model_reg.predict(latest_df[used_reg])
    else:
        latest_df["pred_rf_reg"] = np.nan

    # Autoencoder familiarity (model-native transform + calibration path)
    if ae_model is not None:
        ae_predict = make_autoencoder_familiarity_predictor(numeric_cols)
        latest_df["ae_familiarity"] = ae_predict(latest_df, ae_model)
    else:
        latest_df["ae_familiarity"] = 1.0

    latest_df["combined_score"] = (
        latest_df["pred_rf_cls_proba"] * latest_df["pred_rf_reg"] * latest_df["ae_familiarity"]
    )
    for pcol in prob_cols:
        latest_df[f"score_{pcol[5:]}"] = latest_df[pcol] * latest_df["pred_rf_reg"] * latest_df["ae_familiarity"]
    final_results = latest_df[latest_df["ae_familiarity"] >= confidence_cutoff].copy()

    llm_prompt = None
    if build_llm_guardrail_prompt:
        llm_prompt = build_llm_guardrail_prompt_from_results(
            as_of_date=latest_date,
            results_df=final_results,
            confidence_cutoff=confidence_cutoff,
            llm_top_k=llm_top_k,
            known_events=known_events,
        )

    return latest_date, final_results, llm_prompt


def run_latest_prediction_and_llm_prompt(
    *,
    train_data: pd.DataFrame,
    clf,
    reg,
    ae_model,
    spec_ae,
    numeric_cols: list[str],
    categorical_cols: list[str],
    confidence_cutoff: float = 0.50,
    llm_top_k: int = 15,
    known_events: str = "NONE",
    round_decimals: int = 2,
    market_position_value: int | None = None,
    market_position_col: str = "market_position",
) -> tuple[pd.Timestamp, pd.DataFrame, str | None]:
    """Notebook wrapper; returns (latest_date, results_df, llm_prompt)."""
    latest_date, results, llm_prompt = predict_latest_day_with_all_models(
        train_data=train_data,
        clf=clf,
        reg=reg,
        ae_model=ae_model,
        spec_ae=spec_ae,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        confidence_cutoff=confidence_cutoff,
        build_llm_guardrail_prompt=True,
        llm_top_k=llm_top_k,
        known_events=known_events,
        market_position_value=market_position_value,
        market_position_col=market_position_col,
    )

    if round_decimals is not None and len(results):
        numeric_cols_to_round = [
            "pred_rf_cls_proba",
            "pred_rf_reg",
            "ae_familiarity",
            "combined_score",
        ]
        numeric_cols_to_round.extend([c for c in results.columns if c.startswith("prob_") or c.startswith("score_")])
        present = [c for c in numeric_cols_to_round if c in results.columns]
        if present:
            results.loc[:, present] = results[present].round(round_decimals)

    return latest_date, results, llm_prompt


def run_latest_prediction_custom(
    *,
    train_data: pd.DataFrame,
    model_specs: list[tuple[Any, str] | Mapping[str, Any]],
    combine_scores_fn: Callable[[pd.DataFrame], pd.Series | np.ndarray],
    row_filter_fn: Callable[[pd.DataFrame], pd.Series | np.ndarray] | None = None,
    market_position_value: int | Mapping[Any, Any] | pd.Series | None = None,
    market_position_col: str = "market_position",
    round_decimals: int | None = 2,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Generic latest-day scorer with no assumptions about number/type of models.

    model_specs:
      - tuple: (model, prediction_column)
      - mapping: {
            "model": model,
            "pred_col": "name",
            "feature_cols": [...],           # optional
            "predict_fn": callable,          # optional; overrides predict/predict_proba
            "include_class_probs": True,     # optional for predict_proba models
        }
    """
    date_level = _date_level_name(train_data)
    latest_date = _normalize_ts(train_data.index.get_level_values(date_level).max())
    scored_latest = run_panel_prediction_custom(
        train_data=train_data,
        model_specs=model_specs,
        combine_scores_fn=combine_scores_fn,
        row_filter_fn=row_filter_fn,
        market_position_value=market_position_value,
        market_position_col=market_position_col,
        round_decimals=round_decimals,
        as_of_date=latest_date,
    )
    if isinstance(scored_latest.index, pd.MultiIndex):
        out = scored_latest.xs(latest_date, level=date_level, drop_level=False)
        out = out.droplevel(date_level)
    else:
        out = scored_latest
    return latest_date, out


def score_latest_with_cluster_familiarity(
    *,
    train_data: pd.DataFrame,
    clf: Any,
    reg: Any,
    ae_model: Any,
    flavor_space: Any,
    market_position_value: int | Mapping[Any, Any] | pd.Series | None = 0,
    market_position_col: str = "market_position",
    round_decimals: int | None = None,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """
    Score latest-date rows with classifier/regressor and multiply by cluster familiarity.

    Returns dataframe indexed by symbol with:
      - clf probabilities, ranking
      - closest cluster + min distance + familiarity
      - buy_score / short_score
    """
    if flavor_space is None:
        raise RuntimeError("flavor_space is required.")
    if not isinstance(train_data.index, pd.MultiIndex) or "date" not in (train_data.index.names or []):
        raise TypeError("train_data must have MultiIndex with a 'date' level.")

    model_specs = [
        {"model": clf, "pred_col": "clf", "include_class_probs": True},
        {"model": reg, "pred_col": "ranking"},
    ]
    latest_date, scored = run_latest_prediction_custom(
        train_data=train_data,
        model_specs=model_specs,
        market_position_value=market_position_value,
        market_position_col=market_position_col,
        combine_scores_fn=lambda df: df["clf"] * 0.0,
        row_filter_fn=None,
        round_decimals=round_decimals,
    )

    from modules.analysis.alpha_flavors import score_trade_flavor  # local import to keep module coupling light

    entry_rows = train_data.xs(latest_date, level="date").copy()
    entry_flavor = score_trade_flavor(
        flavor_space=flavor_space,
        entries_df=entry_rows,
        ae_model=ae_model,
    )

    cluster_col = str(flavor_space.cluster_col)
    min_dist_col = f"{cluster_col}__min_dist"
    fam_col = f"{cluster_col}__familiarity"
    attach_cols = [c for c in (cluster_col, min_dist_col, fam_col, "cluster_familiarity") if c in entry_flavor.columns]
    scored = scored.join(entry_flavor[attach_cols], how="left")

    if "cluster_familiarity" not in scored.columns and fam_col in scored.columns:
        scored["cluster_familiarity"] = scored[fam_col]
    if "cluster_familiarity" not in scored.columns:
        scored["cluster_familiarity"] = 0.0

    scored["buy_score"] = scored["clf__prob_buy"] * scored["ranking"] * scored["cluster_familiarity"]
    scored["short_score"] = scored["clf__prob_short"] * scored["ranking"] * scored["cluster_familiarity"]
    return latest_date, scored


def run_panel_prediction_custom(
    *,
    train_data: pd.DataFrame,
    model_specs: list[tuple[Any, str] | Mapping[str, Any]],
    combine_scores_fn: Callable[[pd.DataFrame], pd.Series | np.ndarray],
    row_filter_fn: Callable[[pd.DataFrame], pd.Series | np.ndarray] | None = None,
    market_position_value: int | Mapping[Any, Any] | pd.Series | None = None,
    market_position_col: str = "market_position",
    round_decimals: int | None = 2,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Generic panel scorer.

    - If `as_of_date` is set: score only that date.
    - If `as_of_date` is None: score all dates in panel.
    Returns a DataFrame indexed by (date, symbol).
    """
    if not isinstance(train_data.index, pd.MultiIndex):
        raise TypeError("train_data must have MultiIndex (date, symbol).")
    if not callable(combine_scores_fn):
        raise TypeError("combine_scores_fn must be callable.")

    date_level = _date_level_name(train_data)
    symbol_level = _symbol_level_name(train_data)

    if as_of_date is not None:
        dates = [_normalize_ts(as_of_date)]
    else:
        dates = sorted(pd.Index(train_data.index.get_level_values(date_level)).unique())

    frames: list[pd.DataFrame] = []
    for dt in dates:
        try:
            day_df = train_data.xs(dt, level=date_level).copy()
        except KeyError:
            continue
        day_df = _apply_market_position(
            day_df,
            market_position_value=market_position_value,
            market_position_col=market_position_col,
        )
        scored = _apply_model_specs(day_df, model_specs)
        scored["combined_score"] = _as_array(combine_scores_fn(scored.copy()), len(scored))

        if row_filter_fn is not None:
            mask = np.asarray(row_filter_fn(scored.copy())).reshape(-1).astype(bool)
            if len(mask) != len(scored):
                raise ValueError(f"row_filter_fn returned {len(mask)} rows, expected {len(scored)}.")
            scored = scored.loc[mask].copy()

        scored[date_level] = _normalize_ts(dt)
        scored[symbol_level] = scored.index
        scored = scored.set_index([date_level, symbol_level]).sort_index()
        frames.append(scored)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, axis=0).sort_index()
    if round_decimals is not None and len(panel):
        ncols = panel.select_dtypes(include=[np.number]).columns.tolist()
        if ncols:
            panel.loc[:, ncols] = panel[ncols].round(round_decimals)
    return panel


def select_prediction_slice(
    prediction_panel: pd.DataFrame,
    *,
    as_of_date: str | pd.Timestamp | None = None,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Filter panel predictions by optional date and/or symbol."""
    if prediction_panel is None or len(prediction_panel) == 0:
        return pd.DataFrame()
    if not isinstance(prediction_panel.index, pd.MultiIndex):
        raise TypeError("prediction_panel must have MultiIndex (date, symbol).")

    out = prediction_panel
    date_level = _date_level_name(out)
    symbol_level = _symbol_level_name(out)

    if as_of_date is not None:
        ts = _normalize_ts(as_of_date)
        try:
            out = out.xs(ts, level=date_level, drop_level=False)
        except KeyError:
            return pd.DataFrame(columns=prediction_panel.columns)

    if symbol is not None:
        sym = str(symbol)
        mask = out.index.get_level_values(symbol_level).astype(str) == sym
        out = out.loc[mask]
    return out.copy()
