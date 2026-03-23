from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from analysis.oracle_entry_exit_dataset import build_state_text
from data.historical_prices import load_adjusted_price_frames
from domain.labels.specs import LabelBuildSpec
from features.feature_builders import build_fundamental_change_features, build_price_technical_features
from features.financial_growth_features import build_financial_growth_features
from features.macro import EconomicDataConfig, broadcast_series_to_daily, fetch_economic_data_series
from features.section_utils import prime_section_record_cache
from fmp.models import EconomicIndicatorSeries, Symbol, TreasuryRateSeries
from ml.frameworks.transformers import ContextFamilyStateModel, load_local_first_tokenizer, resolve_torch_device
from pipeline.universe_selection import DEFAULT_US_EXCHANGES, resolve_symbol_universe
from workflows.labels import build_oracle_labels


@dataclass(frozen=True)
class ClassifierBacktestConfig:
    min_market_cap: float = 100_000_000_000.0
    start_date: str = "1900-01-01"
    train_end_date: str = "2019-12-31"
    backtest_start_date: str = "2020-01-01"
    end_date: str | None = None
    k_params: dict[str, list[int]] | None = None
    min_profit_pct: float = 0.01
    buy_execution: str = "adj_high"
    sell_execution: str = "adj_low"
    short_execution: str = "adj_low"
    cover_execution: str = "adj_high"
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    text_projection_dim: int = 192
    family_embedding_dim: int = 64
    family_num_heads: int = 4
    family_num_layers: int = 2
    fusion_dim: int = 256
    bottleneck_hidden_1: int = 288
    bottleneck_hidden_2: int = 240
    dropout: float = 0.10
    model_name: str = "answerdotai/ModernBERT-base"

    def resolved_k_params(self) -> dict[str, list[int]]:
        return dict(self.k_params or {"YE": [1, 2, 4, 8, 16]})


def _frame_batches(frame: pd.DataFrame, batch_size: int, *, shuffle: bool, seed: int = 42):
    indices = np.arange(len(frame))
    if shuffle and len(indices) > 0:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield frame.iloc[batch_idx].reset_index(drop=True)


def _compute_feature_stats(frame: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, float]]:
    if not cols:
        return {"mean": {}, "std": {}}
    subset = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    mean = subset.mean(axis=0)
    std = subset.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    return {"mean": mean.to_dict(), "std": std.to_dict()}


def _build_universe(min_market_cap: float) -> tuple[tuple[str, ...], pd.DataFrame]:
    universe = tuple(
        resolve_symbol_universe(
            min_market_cap=float(min_market_cap),
            country="US",
            exchanges=list(DEFAULT_US_EXCHANGES),
            exclude_pooled_vehicles=True,
            limit=None,
        )
    )
    symbol_order = {symbol: idx for idx, symbol in enumerate(universe)}
    universe_rows = list(
        Symbol.objects.filter(symbol__in=universe)
        .values("symbol", "company_name", "sector", "industry", "exchange", "country", "market_cap", "payload")
    )
    universe_df = pd.DataFrame(universe_rows)
    if not universe_df.empty:
        universe_df["symbol"] = universe_df["symbol"].astype(str).str.strip().str.upper()
        universe_df["sector"] = universe_df["sector"].fillna("").astype(str).str.strip().replace("", "Unknown")
        universe_df["industry"] = universe_df["industry"].fillna("").astype(str).str.strip().replace("", "Unknown")
        universe_df["sort_order"] = universe_df["symbol"].map(symbol_order)
        universe_df = universe_df.sort_values(["sort_order", "symbol"]).drop(columns=["sort_order", "payload"])
    return universe, universe_df


def _build_price_lookup(price_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, frame in price_frames.items():
        if frame.empty:
            continue
        working = frame.reset_index().copy()
        working["symbol"] = symbol
        working["date_text"] = pd.to_datetime(working["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for row in working.to_dict(orient="records"):
            rows.append(
                {
                    "symbol": str(row.get("symbol") or "").strip().upper(),
                    "date_text": str(row.get("date_text") or ""),
                    "adj_open": row.get("adj_open"),
                    "adj_high": row.get("adj_high"),
                    "adj_low": row.get("adj_low"),
                    "adj_close": row.get("adj_close"),
                    "volume": row.get("volume"),
                }
            )
    price_lookup_df = pd.DataFrame(rows)
    if not price_lookup_df.empty:
        for col in ["adj_open", "adj_high", "adj_low", "adj_close", "volume"]:
            price_lookup_df[col] = pd.to_numeric(price_lookup_df[col], errors="coerce")
        price_lookup_df = price_lookup_df.drop_duplicates(subset=["symbol", "date_text"], keep="last").reset_index(drop=True)
    return price_lookup_df


def _build_label_df(universe: tuple[str, ...], price_frames: dict[str, pd.DataFrame], cfg: ClassifierBacktestConfig) -> pd.DataFrame:
    label_spec = LabelBuildSpec(
        k_params=cfg.resolved_k_params(),
        min_profit_pct=float(cfg.min_profit_pct),
        buy_execution=str(cfg.buy_execution),
        sell_execution=str(cfg.sell_execution),
        short_execution=str(cfg.short_execution),
        cover_execution=str(cfg.cover_execution),
        trade_dedup_mode="exact",
        start_date=str(cfg.start_date),
        end_date=str(cfg.train_end_date),
        download_missing_prices=True,
    )
    oracle_result = build_oracle_labels(list(universe), spec=label_spec, price_frames=price_frames)
    label_df = pd.DataFrame(oracle_result.label_rows)
    if label_df.empty:
        raise ValueError("No oracle label rows were generated for the pre-2020 training window.")
    label_df["date"] = pd.to_datetime(label_df["date"], errors="coerce")
    label_df["date_text"] = label_df["date"].dt.strftime("%Y-%m-%d")
    if "action_label" in label_df.columns:
        label_df["action_label"] = label_df["action_label"].fillna("").astype(str).str.strip().str.lower()
    elif "label" in label_df.columns:
        label_df["action_label"] = label_df["label"].fillna("").astype(str).str.strip().str.lower()
    if "event" in label_df.columns:
        label_df["event"] = label_df["event"].fillna("").astype(str).str.strip().str.lower()
        label_df = label_df[label_df["event"] == "entry"].copy()
    label_df["symbol"] = label_df["symbol"].astype(str).str.strip().str.upper()
    label_df = label_df[label_df["action_label"].isin(["buy", "short"])].copy()
    return label_df.reset_index(drop=True)


def _build_daily_state_panel(
    universe_df: pd.DataFrame,
    price_frames: dict[str, pd.DataFrame],
    *,
    start_date: str,
    end_date: str | None,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    state_rows: list[dict[str, Any]] = []
    for row in universe_df.to_dict(orient="records"):
        symbol = str(row.get("symbol") or "").strip().upper()
        company_name = str(row.get("company_name") or "Unknown")
        sector = str(row.get("sector") or "Unknown")
        industry = str(row.get("industry") or "Unknown")
        frame = price_frames.get(symbol, pd.DataFrame())
        if frame.empty:
            continue
        working = frame.reset_index().copy()
        working["date_text"] = pd.to_datetime(working["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for item in working.to_dict(orient="records"):
            date_text = str(item.get("date_text") or "")
            if not date_text:
                continue
            state_rows.append(
                {
                    "date_text": date_text,
                    "symbol": symbol,
                    "company_name": company_name,
                    "sector": sector,
                    "industry": industry,
                    "adj_open": item.get("adj_open"),
                    "adj_high": item.get("adj_high"),
                    "adj_low": item.get("adj_low"),
                    "adj_close": item.get("adj_close"),
                    "volume": item.get("volume"),
                    "text": build_state_text(date_text, symbol, company_name, sector, industry),
                }
            )
    state_df = pd.DataFrame(state_rows)
    if state_df.empty:
        raise ValueError("No daily state rows were built from price history.")
    state_df["date"] = pd.to_datetime(state_df["date_text"], errors="coerce")
    state_df = state_df.dropna(subset=["date"]).copy()
    state_df = state_df[state_df["date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        state_df = state_df[state_df["date"] <= pd.Timestamp(end_date)].copy()
    state_df = state_df.sort_values(["symbol", "date"]).reset_index(drop=True)
    for col in ["adj_open", "adj_high", "adj_low", "adj_close", "volume"]:
        state_df[col] = pd.to_numeric(state_df[col], errors="coerce")

    unique_symbols = sorted(state_df["symbol"].dropna().astype(str).str.upper().unique().tolist())
    symbol_obj_rows = list(Symbol.objects.filter(symbol__in=unique_symbols).only("id", "symbol"))
    symbol_obj_map = {str(item.symbol).strip().upper(): item for item in symbol_obj_rows}
    prime_section_record_cache(symbol_obj_rows, ["key_metrics", "ratios", "financial_growth"])

    fundamental_frames = []
    for symbol in unique_symbols:
        symbol_obj = symbol_obj_map.get(symbol)
        if symbol_obj is None:
            continue
        symbol_dates = (
            pd.DatetimeIndex(pd.to_datetime(state_df.loc[state_df["symbol"] == symbol, "date_text"], errors="coerce"))
            .dropna()
            .normalize()
            .unique()
            .sort_values()
        )
        if len(symbol_dates) == 0:
            continue
        target_index = pd.MultiIndex.from_arrays([symbol_dates, [symbol] * len(symbol_dates)], names=["date", "symbol"])
        df_prices_symbol = price_frames.get(symbol, pd.DataFrame()).copy()
        if not df_prices_symbol.empty and "adj_close" in df_prices_symbol.columns and "close" not in df_prices_symbol.columns:
            df_prices_symbol = df_prices_symbol.rename(columns={"adj_close": "close"})
        merged_symbol_features = pd.DataFrame(index=target_index)
        selected_feature_cols: list[str] = []
        technical_built = build_price_technical_features(symbol, df_prices_symbol) if not df_prices_symbol.empty else None
        if technical_built is not None and not technical_built.df.empty and technical_built.feature_cols:
            technical_aligned = technical_built.df.reindex(target_index)
            merged_symbol_features = merged_symbol_features.join(technical_aligned[technical_built.feature_cols], how="left")
            selected_feature_cols.extend(list(technical_built.feature_cols))
        fundamental_built = build_fundamental_change_features(
            symbol_obj,
            target_index,
            df_prices=df_prices_symbol if not df_prices_symbol.empty else None,
        )
        if not fundamental_built.df.empty and fundamental_built.feature_cols:
            merged_symbol_features = merged_symbol_features.join(fundamental_built.df[fundamental_built.feature_cols], how="left")
            selected_feature_cols.extend(list(fundamental_built.feature_cols))
        financial_growth_built = build_financial_growth_features(symbol_obj, target_index)
        if not financial_growth_built.df.empty and financial_growth_built.feature_cols:
            merged_symbol_features = merged_symbol_features.join(
                financial_growth_built.df[financial_growth_built.feature_cols],
                how="left",
            )
            selected_feature_cols.extend(list(financial_growth_built.feature_cols))
        selected_feature_cols = list(dict.fromkeys(selected_feature_cols))
        if not selected_feature_cols:
            continue
        symbol_feature_df = merged_symbol_features[selected_feature_cols].reset_index()
        symbol_feature_df["date_text"] = pd.to_datetime(symbol_feature_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        fundamental_frames.append(symbol_feature_df[["symbol", "date_text", *selected_feature_cols]])

    if fundamental_frames:
        combined_symbol_feature_df = pd.concat(fundamental_frames, ignore_index=True)
        technical_feature_cols = sorted([col for col in combined_symbol_feature_df.columns if str(col).startswith("px__")])
        technical_feature_cols = [
            col for col in technical_feature_cols if col not in {"px__adj_open", "px__adj_high", "px__adj_low", "px__adj_close"}
        ]
        fundamental_feature_cols = sorted(
            [col for col in combined_symbol_feature_df.columns if str(col).startswith(("km__", "rt__", "fg__"))]
        )
        symbol_feature_lookup_df = combined_symbol_feature_df[
            ["symbol", "date_text", *technical_feature_cols, *fundamental_feature_cols]
        ].copy()
        for col in technical_feature_cols + fundamental_feature_cols:
            symbol_feature_lookup_df[col] = pd.to_numeric(symbol_feature_lookup_df[col], errors="coerce")
    else:
        technical_feature_cols = []
        fundamental_feature_cols = []
        symbol_feature_lookup_df = pd.DataFrame(columns=["symbol", "date_text"])

    all_macro_dates = pd.to_datetime(state_df["date_text"], errors="coerce").dropna()
    macro_start = all_macro_dates.min().date().isoformat() if len(all_macro_dates) > 0 else None
    macro_end = all_macro_dates.max().date().isoformat() if len(all_macro_dates) > 0 else None
    if macro_start and macro_end:
        economic_series_codes = tuple(
            str(code) for code in EconomicIndicatorSeries.objects.order_by("code").values_list("code", flat=True)
        )
        treasury_series_codes = tuple(
            str(code) for code in TreasuryRateSeries.objects.order_by("code").values_list("code", flat=True)
        )
        economic_df = fetch_economic_data_series(
            api_key="",
            start_date=macro_start,
            end_date=macro_end,
            config=EconomicDataConfig(economic_indicator_series=economic_series_codes, include_treasury_rates=False),
        )
        treasury_df = fetch_economic_data_series(
            api_key="",
            start_date=macro_start,
            end_date=macro_end,
            config=EconomicDataConfig(economic_indicator_series=treasury_series_codes, include_treasury_rates=False),
        )
    else:
        economic_df = pd.DataFrame()
        treasury_df = pd.DataFrame()
    macro_target_index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(state_df["date_text"], errors="coerce"), state_df["symbol"].astype(str)],
        names=["date", "symbol"],
    )
    economic_daily = broadcast_series_to_daily(economic_df, macro_target_index) if not economic_df.empty else pd.DataFrame(index=macro_target_index)
    treasury_daily = broadcast_series_to_daily(treasury_df, macro_target_index) if not treasury_df.empty else pd.DataFrame(index=macro_target_index)
    economic_feature_cols = [str(col) for col in list(economic_daily.columns)]
    treasury_feature_cols = [str(col) for col in list(treasury_daily.columns)]
    macro_feature_cols = economic_feature_cols + treasury_feature_cols
    if macro_feature_cols:
        macro_lookup_df = pd.concat([economic_daily, treasury_daily], axis=1).reset_index()
        macro_lookup_df["date_text"] = pd.to_datetime(macro_lookup_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        macro_lookup_df = macro_lookup_df[["date_text", *macro_feature_cols]].copy()
        for col in macro_feature_cols:
            macro_lookup_df[col] = pd.to_numeric(macro_lookup_df[col], errors="coerce")
        macro_lookup_df = macro_lookup_df.groupby("date_text", as_index=False).last()
    else:
        macro_lookup_df = pd.DataFrame(columns=["date_text"])

    state_df = state_df.merge(symbol_feature_lookup_df, on=["symbol", "date_text"], how="left")
    for col in technical_feature_cols + fundamental_feature_cols:
        state_df[col] = pd.to_numeric(state_df[col], errors="coerce")
    state_df = state_df.merge(macro_lookup_df, on="date_text", how="left")
    for col in macro_feature_cols:
        state_df[col] = pd.to_numeric(state_df[col], errors="coerce")
    return state_df, technical_feature_cols, fundamental_feature_cols, macro_feature_cols


def build_context_family_daily_state_panel(
    universe_df: pd.DataFrame,
    price_frames: dict[str, pd.DataFrame],
    *,
    start_date: str,
    end_date: str | None,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    return _build_daily_state_panel(
        universe_df,
        price_frames,
        start_date=start_date,
        end_date=end_date,
    )


def _train_entry_classifier(
    train_df: pd.DataFrame,
    *,
    model_name: str,
    tokenizer,
    tokenizer_max_length: int | None,
    market_cols: list[str],
    fundamental_cols: list[str],
    macro_cols: list[str],
    cfg: ClassifierBacktestConfig,
    device: torch.device,
) -> ContextFamilyStateModel:
    model = ContextFamilyStateModel(
        model_name=model_name,
        tokenizer=tokenizer,
        tokenizer_max_length=tokenizer_max_length,
        market_input_dim=len(market_cols),
        fundamental_input_dim=len(fundamental_cols),
        macro_input_dim=len(macro_cols),
        text_projection_dim=cfg.text_projection_dim,
        family_embedding_dim=cfg.family_embedding_dim,
        family_num_heads=cfg.family_num_heads,
        family_num_layers=cfg.family_num_layers,
        fusion_dim=cfg.fusion_dim,
        bottleneck_hidden_1=cfg.bottleneck_hidden_1,
        bottleneck_hidden_2=cfg.bottleneck_hidden_2,
        action_classes=2,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    action_to_id = {"buy": 0, "short": 1}
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        for batch in _frame_batches(train_df, int(cfg.batch_size), shuffle=True, seed=42 + epoch):
            optimizer.zero_grad()
            market_tensor = torch.tensor(batch[market_cols].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
            fundamental_tensor = torch.tensor(batch[fundamental_cols].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
            macro_tensor = torch.tensor(batch[macro_cols].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
            state_embeddings, _ = model.forward_state(batch["text"].tolist(), market_tensor, fundamental_tensor, macro_tensor)
            logits, _return_pred, _signed_return_pred, _duration_pred = model.predict_entry_outcomes(state_embeddings)
            targets = torch.tensor(
                [action_to_id[value] for value in batch["action_label"].astype(str).str.strip().str.lower().tolist()],
                dtype=torch.long,
                device=device,
            )
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            steps += 1
        print(f"classifier epoch={epoch}/{cfg.epochs} train_action_loss={running_loss / max(steps, 1):.4f}", flush=True)
    return model


@torch.no_grad()
def _score_entry_classifier(
    model: ContextFamilyStateModel,
    score_df: pd.DataFrame,
    *,
    market_cols: list[str],
    fundamental_cols: list[str],
    macro_cols: list[str],
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    id_to_action = {0: "buy", 1: "short"}
    model.eval()
    for batch in _frame_batches(score_df, int(batch_size), shuffle=False):
        market_tensor = torch.tensor(batch[market_cols].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
        fundamental_tensor = torch.tensor(batch[fundamental_cols].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
        macro_tensor = torch.tensor(batch[macro_cols].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
        state_embeddings, _ = model.forward_state(batch["text"].tolist(), market_tensor, fundamental_tensor, macro_tensor)
        logits, _return_pred, _signed_return_pred, _duration_pred = model.predict_entry_outcomes(state_embeddings)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        pred_ids = probs.argmax(axis=1)
        for row, pred_id, prob_values in zip(batch.to_dict(orient="records"), pred_ids, probs):
            rows.append(
                {
                    "date": row["date"],
                    "date_text": row["date_text"],
                    "symbol": row["symbol"],
                    "adj_close": row["adj_close"],
                    "pred_action": id_to_action[int(pred_id)],
                    "prob_buy": float(prob_values[0]),
                    "prob_short": float(prob_values[1]),
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def _score_entry_exit_actions(
    model: ContextFamilyStateModel,
    score_df: pd.DataFrame,
    *,
    market_cols: list[str],
    fundamental_cols: list[str],
    macro_cols: list[str],
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    entry_id_to_action = {0: "buy", 1: "short"}
    exit_id_to_action = {0: "sell", 1: "cover"}
    model.eval()
    for batch in _frame_batches(score_df, int(batch_size), shuffle=False):
        market_tensor = torch.tensor(batch[market_cols].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
        fundamental_tensor = torch.tensor(batch[fundamental_cols].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
        macro_tensor = torch.tensor(batch[macro_cols].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
        state_embeddings, _ = model.forward_state(batch["text"].tolist(), market_tensor, fundamental_tensor, macro_tensor)
        entry_logits, _entry_return_pred, _entry_signed_return_pred, _entry_duration_pred = model.predict_entry_outcomes(state_embeddings)
        exit_logits, _exit_return_pred, _exit_signed_return_pred, _exit_duration_pred = model.predict_exit_outcomes(state_embeddings)
        entry_probs = torch.softmax(entry_logits, dim=1).detach().cpu().numpy()
        exit_probs = torch.softmax(exit_logits, dim=1).detach().cpu().numpy()
        entry_pred_ids = entry_probs.argmax(axis=1)
        exit_pred_ids = exit_probs.argmax(axis=1)
        for row, entry_pred_id, exit_pred_id, entry_prob_values, exit_prob_values in zip(
            batch.to_dict(orient="records"),
            entry_pred_ids,
            exit_pred_ids,
            entry_probs,
            exit_probs,
        ):
            rows.append(
                {
                    "date": row["date"],
                    "date_text": row["date_text"],
                    "symbol": row["symbol"],
                    "adj_close": row["adj_close"],
                    "entry_pred_action": entry_id_to_action[int(entry_pred_id)],
                    "exit_pred_action": exit_id_to_action[int(exit_pred_id)],
                    "entry_prob_buy": float(entry_prob_values[0]),
                    "entry_prob_short": float(entry_prob_values[1]),
                    "exit_prob_sell": float(exit_prob_values[0]),
                    "exit_prob_cover": float(exit_prob_values[1]),
                }
            )
    return pd.DataFrame(rows)


def score_context_family_entry_classifier(
    model: ContextFamilyStateModel,
    score_df: pd.DataFrame,
    *,
    market_cols: list[str],
    fundamental_cols: list[str],
    macro_cols: list[str],
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    return _score_entry_classifier(
        model,
        score_df,
        market_cols=market_cols,
        fundamental_cols=fundamental_cols,
        macro_cols=macro_cols,
        device=device,
        batch_size=batch_size,
    )


def score_context_family_entry_exit_actions(
    model: ContextFamilyStateModel,
    score_df: pd.DataFrame,
    *,
    market_cols: list[str],
    fundamental_cols: list[str],
    macro_cols: list[str],
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    return _score_entry_exit_actions(
        model,
        score_df,
        market_cols=market_cols,
        fundamental_cols=fundamental_cols,
        macro_cols=macro_cols,
        device=device,
        batch_size=batch_size,
    )


def _build_strategy_returns(scored_df: pd.DataFrame, price_frames: dict[str, pd.DataFrame], backtest_start_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_rows: list[pd.DataFrame] = []
    for symbol, frame in price_frames.items():
        if frame.empty or "adj_close" not in frame.columns:
            continue
        working = frame[["adj_close"]].copy().reset_index()
        working["symbol"] = symbol
        working["next_adj_close"] = working["adj_close"].shift(-1)
        working["next_return"] = (pd.to_numeric(working["next_adj_close"], errors="coerce") / pd.to_numeric(working["adj_close"], errors="coerce")) - 1.0
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
        price_rows.append(working[["date", "symbol", "adj_close", "next_return"]])
    return_frame = pd.concat(price_rows, ignore_index=True)
    return_frame["symbol"] = return_frame["symbol"].astype(str).str.strip().str.upper()
    return_frame = return_frame[return_frame["date"] >= pd.Timestamp(backtest_start_date)].copy()
    merged = scored_df.merge(return_frame, on=["date", "symbol"], how="inner", suffixes=("", "_price"))
    merged = merged.dropna(subset=["next_return"]).copy()
    merged["position"] = np.where(merged["pred_action"].astype(str).str.lower() == "buy", 1.0, -1.0)
    merged["strategy_return"] = merged["position"] * pd.to_numeric(merged["next_return"], errors="coerce").fillna(0.0)
    merged["buy_hold_return"] = pd.to_numeric(merged["next_return"], errors="coerce").fillna(0.0)
    per_symbol = (
        merged.groupby("symbol", as_index=False)
        .agg(
            observations=("date", "count"),
            strategy_cum_return=("strategy_return", lambda s: float((1.0 + pd.Series(s, dtype=float)).prod() - 1.0)),
            buy_hold_cum_return=("buy_hold_return", lambda s: float((1.0 + pd.Series(s, dtype=float)).prod() - 1.0)),
            mean_prob_buy=("prob_buy", "mean"),
            mean_prob_short=("prob_short", "mean"),
        )
    )
    per_symbol["outperform_buy_hold"] = per_symbol["strategy_cum_return"] > per_symbol["buy_hold_cum_return"]
    daily_equal_weight = (
        merged.groupby("date", as_index=False)
        .agg(
            strategy_return=("strategy_return", "mean"),
            buy_hold_return=("buy_hold_return", "mean"),
            symbols=("symbol", "nunique"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily_equal_weight["strategy_equity"] = (1.0 + daily_equal_weight["strategy_return"]).cumprod()
    daily_equal_weight["buy_hold_equity"] = (1.0 + daily_equal_weight["buy_hold_return"]).cumprod()
    return per_symbol, daily_equal_weight


def _build_stateful_strategy_returns(
    scored_df: pd.DataFrame,
    price_frames: dict[str, pd.DataFrame],
    *,
    backtest_start_date: str,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mode_normalized = str(mode).strip().lower()
    if mode_normalized not in {"long_only", "short_only"}:
        raise ValueError(f"Unsupported stateful backtest mode: {mode}")

    price_rows: list[pd.DataFrame] = []
    for symbol, frame in price_frames.items():
        if frame.empty or "adj_close" not in frame.columns:
            continue
        working = frame[["adj_close"]].copy().reset_index()
        working["symbol"] = symbol
        working["next_adj_close"] = working["adj_close"].shift(-1)
        working["next_return"] = (pd.to_numeric(working["next_adj_close"], errors="coerce") / pd.to_numeric(working["adj_close"], errors="coerce")) - 1.0
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
        price_rows.append(working[["date", "symbol", "adj_close", "next_return"]])
    return_frame = pd.concat(price_rows, ignore_index=True)
    return_frame["symbol"] = return_frame["symbol"].astype(str).str.strip().str.upper()
    return_frame = return_frame[return_frame["date"] >= pd.Timestamp(backtest_start_date)].copy()

    merged = scored_df.merge(return_frame, on=["date", "symbol"], how="inner", suffixes=("", "_price"))
    merged = merged.dropna(subset=["next_return"]).copy()
    merged = merged.sort_values(["symbol", "date"]).reset_index(drop=True)

    row_outputs: list[dict[str, Any]] = []
    for symbol, symbol_frame in merged.groupby("symbol", sort=True):
        current_position = 0.0
        for row in symbol_frame.to_dict(orient="records"):
            entry_action = str(row.get("entry_pred_action") or "").strip().lower()
            exit_action = str(row.get("exit_pred_action") or "").strip().lower()
            if mode_normalized == "long_only":
                if current_position == 0.0 and entry_action == "buy":
                    next_position = 1.0
                    trade_signal = "buy"
                elif current_position == 1.0 and exit_action == "sell":
                    next_position = 0.0
                    trade_signal = "sell"
                else:
                    next_position = current_position
                    trade_signal = "hold" if current_position == 1.0 else "flat"
            else:
                if current_position == 0.0 and entry_action == "short":
                    next_position = -1.0
                    trade_signal = "short"
                elif current_position == -1.0 and exit_action == "cover":
                    next_position = 0.0
                    trade_signal = "cover"
                else:
                    next_position = current_position
                    trade_signal = "hold" if current_position == -1.0 else "flat"

            next_return = float(pd.to_numeric(row.get("next_return"), errors="coerce") or 0.0)
            strategy_return = next_position * next_return
            row_outputs.append(
                {
                    "date": row["date"],
                    "date_text": row["date_text"],
                    "symbol": symbol,
                    "entry_pred_action": entry_action,
                    "exit_pred_action": exit_action,
                    "trade_signal": trade_signal,
                    "position": float(next_position),
                    "next_return": next_return,
                    "strategy_return": strategy_return,
                    "buy_hold_return": next_return,
                    "entry_prob_buy": row.get("entry_prob_buy"),
                    "entry_prob_short": row.get("entry_prob_short"),
                    "exit_prob_sell": row.get("exit_prob_sell"),
                    "exit_prob_cover": row.get("exit_prob_cover"),
                }
            )
            current_position = next_position

    stateful_df = pd.DataFrame(row_outputs)
    per_symbol = (
        stateful_df.groupby("symbol", as_index=False)
        .agg(
            observations=("date", "count"),
            active_days=("position", lambda s: int((pd.Series(s, dtype=float) != 0.0).sum())),
            strategy_cum_return=("strategy_return", lambda s: float((1.0 + pd.Series(s, dtype=float)).prod() - 1.0)),
            buy_hold_cum_return=("buy_hold_return", lambda s: float((1.0 + pd.Series(s, dtype=float)).prod() - 1.0)),
            mean_entry_prob_buy=("entry_prob_buy", "mean"),
            mean_entry_prob_short=("entry_prob_short", "mean"),
            mean_exit_prob_sell=("exit_prob_sell", "mean"),
            mean_exit_prob_cover=("exit_prob_cover", "mean"),
        )
    )
    per_symbol["outperform_buy_hold"] = per_symbol["strategy_cum_return"] > per_symbol["buy_hold_cum_return"]
    daily_equal_weight = (
        stateful_df.groupby("date", as_index=False)
        .agg(
            strategy_return=("strategy_return", "mean"),
            buy_hold_return=("buy_hold_return", "mean"),
            active_positions=("position", lambda s: int((pd.Series(s, dtype=float) != 0.0).sum())),
            symbols=("symbol", "nunique"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily_equal_weight["strategy_equity"] = (1.0 + daily_equal_weight["strategy_return"]).cumprod()
    daily_equal_weight["buy_hold_equity"] = (1.0 + daily_equal_weight["buy_hold_return"]).cumprod()
    return stateful_df, per_symbol, daily_equal_weight


def backtest_scored_entry_exit_actions_against_buy_hold(
    scored_df: pd.DataFrame,
    price_frames: dict[str, pd.DataFrame],
    *,
    backtest_start_date: str,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _build_stateful_strategy_returns(
        scored_df,
        price_frames,
        backtest_start_date=backtest_start_date,
        mode=mode,
    )


def backtest_scored_entry_actions_against_buy_hold(
    scored_df: pd.DataFrame,
    price_frames: dict[str, pd.DataFrame],
    *,
    backtest_start_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _build_strategy_returns(scored_df, price_frames, backtest_start_date)


def run_pretrained_context_family_entry_exit_symbol_backtests(
    *,
    model: ContextFamilyStateModel,
    universe_df: pd.DataFrame,
    price_frames: dict[str, pd.DataFrame],
    backtest_start_date: str,
    market_cols: list[str],
    fundamental_cols: list[str],
    macro_cols: list[str],
    device: torch.device | None = None,
    batch_size: int = 16,
    end_date: str | None = None,
) -> dict[str, Any]:
    resolved_device = device or resolve_torch_device()
    daily_state_df, technical_feature_cols, built_fundamental_cols, built_macro_cols = build_context_family_daily_state_panel(
        universe_df,
        price_frames,
        start_date=backtest_start_date,
        end_date=end_date,
    )
    expected_cols = ["adj_open", "adj_high", "adj_low", "adj_close", "volume", *technical_feature_cols, *built_fundamental_cols, *built_macro_cols]
    for col in expected_cols:
        if col in daily_state_df.columns:
            daily_state_df[col] = pd.to_numeric(daily_state_df[col], errors="coerce").fillna(0.0)
    scored_df = score_context_family_entry_exit_actions(
        model,
        daily_state_df,
        market_cols=market_cols,
        fundamental_cols=fundamental_cols,
        macro_cols=macro_cols,
        device=resolved_device,
        batch_size=batch_size,
    )

    payload: dict[str, Any] = {"scored_df": scored_df}
    for mode_name in ("long_only", "short_only"):
        stateful_df, per_symbol_df, daily_equal_weight_df = backtest_scored_entry_exit_actions_against_buy_hold(
            scored_df,
            price_frames,
            backtest_start_date=backtest_start_date,
            mode=mode_name,
        )
        payload[f"{mode_name}_stateful_df"] = stateful_df
        payload[f"{mode_name}_per_symbol_df"] = per_symbol_df.sort_values("strategy_cum_return", ascending=False).reset_index(drop=True)
        payload[f"{mode_name}_daily_equal_weight_df"] = daily_equal_weight_df
        payload[f"{mode_name}_summary"] = {
            "mode": mode_name,
            "scored_rows": int(len(scored_df)),
            "backtest_symbols": int(per_symbol_df["symbol"].nunique()),
            "strategy_equal_weight_cum_return": float(daily_equal_weight_df["strategy_equity"].iloc[-1] - 1.0) if not daily_equal_weight_df.empty else float("nan"),
            "buy_hold_equal_weight_cum_return": float(daily_equal_weight_df["buy_hold_equity"].iloc[-1] - 1.0) if not daily_equal_weight_df.empty else float("nan"),
            "symbol_outperform_rate": float(per_symbol_df["outperform_buy_hold"].mean()) if not per_symbol_df.empty else float("nan"),
            "mean_active_positions": float(daily_equal_weight_df["active_positions"].mean()) if not daily_equal_weight_df.empty else float("nan"),
        }
    return payload


def run_pretrained_context_family_classifier_symbol_backtest(
    *,
    model: ContextFamilyStateModel,
    universe_df: pd.DataFrame,
    price_frames: dict[str, pd.DataFrame],
    backtest_start_date: str,
    market_cols: list[str],
    fundamental_cols: list[str],
    macro_cols: list[str],
    device: torch.device | None = None,
    batch_size: int = 16,
    end_date: str | None = None,
) -> dict[str, Any]:
    resolved_device = device or resolve_torch_device()
    daily_state_df, technical_feature_cols, built_fundamental_cols, built_macro_cols = build_context_family_daily_state_panel(
        universe_df,
        price_frames,
        start_date=backtest_start_date,
        end_date=end_date,
    )
    expected_cols = ["adj_open", "adj_high", "adj_low", "adj_close", "volume", *technical_feature_cols, *built_fundamental_cols, *built_macro_cols]
    for col in expected_cols:
        if col in daily_state_df.columns:
            daily_state_df[col] = pd.to_numeric(daily_state_df[col], errors="coerce").fillna(0.0)
    scored_df = score_context_family_entry_classifier(
        model,
        daily_state_df,
        market_cols=market_cols,
        fundamental_cols=fundamental_cols,
        macro_cols=macro_cols,
        device=resolved_device,
        batch_size=batch_size,
    )
    per_symbol_df, daily_equal_weight_df = backtest_scored_entry_actions_against_buy_hold(
        scored_df,
        price_frames,
        backtest_start_date=backtest_start_date,
    )
    summary = {
        "scored_rows": int(len(scored_df)),
        "backtest_symbols": int(per_symbol_df["symbol"].nunique()),
        "strategy_equal_weight_cum_return": float(daily_equal_weight_df["strategy_equity"].iloc[-1] - 1.0) if not daily_equal_weight_df.empty else float("nan"),
        "buy_hold_equal_weight_cum_return": float(daily_equal_weight_df["buy_hold_equity"].iloc[-1] - 1.0) if not daily_equal_weight_df.empty else float("nan"),
        "symbol_outperform_rate": float(per_symbol_df["outperform_buy_hold"].mean()) if not per_symbol_df.empty else float("nan"),
    }
    return {
        "summary": summary,
        "per_symbol_df": per_symbol_df.sort_values("strategy_cum_return", ascending=False).reset_index(drop=True),
        "daily_equal_weight_df": daily_equal_weight_df,
        "scored_df": scored_df,
    }


def run_context_family_classifier_symbol_backtest(
    cfg: ClassifierBacktestConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or ClassifierBacktestConfig()
    device = resolve_torch_device()
    universe, universe_df = _build_universe(cfg.min_market_cap)
    if not universe:
        raise ValueError("Universe screen returned no symbols.")
    price_frames = load_adjusted_price_frames(list(universe), start_date=cfg.start_date, end_date=cfg.end_date)
    label_df = _build_label_df(universe, price_frames, cfg)
    daily_state_df, technical_cols, fundamental_cols, macro_cols = _build_daily_state_panel(
        universe_df,
        price_frames,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
    )
    for col in ["adj_open", "adj_high", "adj_low", "adj_close", "volume", *technical_cols, *fundamental_cols, *macro_cols]:
        if col in daily_state_df.columns:
            daily_state_df[col] = pd.to_numeric(daily_state_df[col], errors="coerce").fillna(0.0)

    train_df = label_df.merge(
        daily_state_df[
            [
                "date_text",
                "symbol",
                "text",
                "adj_open",
                "adj_high",
                "adj_low",
                "adj_close",
                "volume",
                *technical_cols,
                *fundamental_cols,
                *macro_cols,
            ]
        ],
        on=["symbol", "date_text"],
        how="inner",
    )
    train_df = train_df[train_df["date"] <= pd.Timestamp(cfg.train_end_date)].copy()
    if train_df.empty:
        raise ValueError("No pre-2020 entry training rows were available after feature joins.")

    tokenizer = load_local_first_tokenizer(cfg.model_name)
    model_max_length = getattr(tokenizer, "model_max_length", None)
    resolved_model_max_length = (
        int(model_max_length) if isinstance(model_max_length, int) and model_max_length < 1_000_000 else None
    )

    model = _train_entry_classifier(
        train_df,
        model_name=cfg.model_name,
        tokenizer=tokenizer,
        tokenizer_max_length=resolved_model_max_length,
        market_cols=["adj_open", "adj_high", "adj_low", "adj_close", "volume", *technical_cols],
        fundamental_cols=list(fundamental_cols),
        macro_cols=list(macro_cols),
        cfg=cfg,
        device=device,
    )

    score_df = daily_state_df[daily_state_df["date"] >= pd.Timestamp(cfg.backtest_start_date)].copy()
    scored_df = _score_entry_classifier(
        model,
        score_df,
        market_cols=["adj_open", "adj_high", "adj_low", "adj_close", "volume", *technical_cols],
        fundamental_cols=list(fundamental_cols),
        macro_cols=list(macro_cols),
        device=device,
        batch_size=cfg.batch_size,
    )
    per_symbol_df, daily_equal_weight_df = _build_strategy_returns(scored_df, price_frames, cfg.backtest_start_date)
    summary = {
        "universe_size": int(len(universe)),
        "train_rows": int(len(train_df)),
        "scored_rows": int(len(scored_df)),
        "backtest_symbols": int(per_symbol_df["symbol"].nunique()),
        "strategy_equal_weight_cum_return": float(daily_equal_weight_df["strategy_equity"].iloc[-1] - 1.0) if not daily_equal_weight_df.empty else float("nan"),
        "buy_hold_equal_weight_cum_return": float(daily_equal_weight_df["buy_hold_equity"].iloc[-1] - 1.0) if not daily_equal_weight_df.empty else float("nan"),
        "symbol_outperform_rate": float(per_symbol_df["outperform_buy_hold"].mean()) if not per_symbol_df.empty else float("nan"),
    }
    return {
        "config": cfg,
        "summary": summary,
        "per_symbol_df": per_symbol_df.sort_values("strategy_cum_return", ascending=False).reset_index(drop=True),
        "daily_equal_weight_df": daily_equal_weight_df,
        "scored_df": scored_df,
    }
