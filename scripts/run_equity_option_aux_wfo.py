"""Paired equity WFO: original equity MTL versus equity MTL + option tasks.

The trading signal always comes from the original equity graph heads.  The
option document task is auxiliary training only, so this runner measures
negative transfer on the equity strategy rather than trading option labels.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]

def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

equity = _load("equity_transformer", ROOT / "scripts" / "run_symbol_year_transformer_mtl.py")
option = _load("option_documents", ROOT / "scripts" / "run_equity_conditioned_option_document.py")
option.DEVICE = equity.DEVICE

OPTION_LOSS_WEIGHT = float(os.getenv("TRANSFORMER_OPTION_LOSS_WEIGHT", "0.10"))
OPTION_BATCH = int(os.getenv("TRANSFORMER_OPTION_BATCH", "256"))
BASELINE_ONLY = os.getenv("TRANSFORMER_BASELINE_ONLY", "0") == "1"
OPTION_ONLY = os.getenv("TRANSFORMER_OPTION_ONLY", "0") == "1"
OUT = ROOT / "artifacts" / "equity_option_aux_wfo"
OUT.mkdir(parents=True, exist_ok=True)


class SharedOptionTask(nn.Module):
    """Pools option rows after conditioning them on the shared issuer state."""

    def __init__(self, option_dim: int):
        super().__init__()
        self.row_encoder = nn.Sequential(
            nn.Linear(equity.HIDDEN + option_dim, equity.HIDDEN),
            nn.LayerNorm(equity.HIDDEN), nn.GELU(),
            nn.Linear(equity.HIDDEN, equity.HIDDEN),
            nn.LayerNorm(equity.HIDDEN), nn.GELU(),
        )
        self.overall = nn.Linear(equity.HIDDEN, 1)
        self.call = nn.Linear(equity.HIDDEN, 1)
        self.put = nn.Linear(equity.HIDDEN, 1)

    def forward(self, issuer_state: torch.Tensor, option_x: torch.Tensor, ids: torch.Tensor):
        row_state = issuer_state.index_select(0, ids)
        z = self.row_encoder(torch.cat([row_state, option_x], dim=-1))
        pooled = torch.zeros((issuer_state.shape[0], equity.HIDDEN), device=z.device, dtype=z.dtype)
        pooled.index_add_(0, ids, z)
        counts = torch.bincount(ids, minlength=issuer_state.shape[0]).to(z.device, z.dtype).unsqueeze(1)
        pooled = pooled / counts.clamp_min(1)
        return self.overall(pooled).squeeze(-1), self.call(pooled).squeeze(-1), self.put(pooled).squeeze(-1)


def _issuer_states(model, batch, use_option_task):
    """Run one equity token through the transformer's shared encoder."""
    x = torch.from_numpy(batch["equity_states"]).to(equity.DEVICE).unsqueeze(1)
    padding = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=equity.DEVICE)
    h = model.input(x) + model.position[:1].unsqueeze(0)
    trunks = [encoder(h, src_key_padding_mask=padding) for encoder in model.encoders]
    routed = model._task_state("option", trunks) if use_option_task else torch.stack(trunks, dim=0).mean(dim=0)
    return routed[:, 0, :]


def _equity_epoch(model, docs, optimizer):
    """Train every equity batch once; this avoids the legacy empty-kind path."""
    model.train(); losses = []
    for kind in ("symbol_year", "cross_sectional"):
        indices = [i for i, item in enumerate(docs) if item["kind"] == kind]
        if not indices:
            continue
        order = np.random.permutation(indices)
        for offset in range(0, len(order), equity.BATCH_SIZE):
            selected = [docs[int(i)] for i in order[offset:offset + equity.BATCH_SIZE]]
            x, padding, target = equity._batch(selected)
            x = x.to(equity.DEVICE); padding = padding.to(equity.DEVICE)
            target = {name: value.to(equity.DEVICE) for name, value in target.items()}
            optimizer.zero_grad(set_to_none=True)
            graph_hat, speed_hat, event_logits, _, document_aux_logits, macro_logits = model(
                x, padding, causal=kind == "symbol_year"
            )
            valid = ~padding; task_valid = target["task_mask"]
            graph_mask = target["graph_mask"] * valid.unsqueeze(-1) * task_valid[:, 0, None, None]
            graph_error = nn.functional.smooth_l1_loss(graph_hat, target["graph"], reduction="none")
            graph_loss = (graph_error * graph_mask).sum() / graph_mask.sum().clamp_min(1.0)
            speed_valid = valid * task_valid[:, 1, None].bool()
            speed_error = nn.functional.smooth_l1_loss(speed_hat, target["speed"], reduction="none")
            speed_loss = (speed_error * speed_valid.unsqueeze(-1)).sum() / speed_valid.sum().clamp_min(1.0)
            event_valid = valid * task_valid[:, 2, None].bool()
            event_loss = equity.gnn.event_loss_from_logits(event_logits[event_valid], target["events"][event_valid]) if event_valid.any() else graph_hat.new_zeros(())
            aux_loss = graph_hat.new_zeros(())
            for index, name in enumerate(equity.AUX_COLS):
                aux_target = target["aux"][:, 0, index]
                mask = task_valid[:, 3].bool() & aux_target.ge(0)
                if mask.any():
                    aux_loss = aux_loss + nn.functional.cross_entropy(document_aux_logits[name][mask], aux_target[mask])
            macro_loss = graph_hat.new_zeros(())
            macro_valid = valid * task_valid[:, 4, None].bool()
            if macro_logits is not None and equity.MACRO_EVENT_COLS and macro_valid.any():
                macro_loss = equity.gnn.event_loss_from_logits(
                    macro_logits if kind == "cross_sectional" else macro_logits[macro_valid],
                    target["macro_document_events"] if kind == "cross_sectional" else target["macro_events"][macro_valid],
                )
            task_losses = {"graph": graph_loss, "speed": speed_loss, "event": event_loss, "aux": aux_loss, "macro": macro_loss}
            weights = {"graph": 1.0, "speed": float(os.getenv("TRANSFORMER_SPEED_LOSS_WEIGHT", "0.10")), "event": 1.0, "aux": float(os.getenv("TRANSFORMER_AUX_LOSS_WEIGHT", "0.10")), "macro": float(os.getenv("TRANSFORMER_MACRO_LOSS_WEIGHT", "1.0"))}
            loss = sum(weights[name] * task_losses[name] for name in model.gradnorm_task_names) + model.routing_regularization()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else 0.0


def _option_loss(model, option_head, batches, optimizer, train: bool):
    model.train(train); option_head.train(train)
    losses = []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in batches:
            option_x = torch.from_numpy(batch["options"]).to(equity.DEVICE)
            ids = torch.from_numpy(batch["document_ids"]).to(equity.DEVICE)
            with torch.autocast(device_type=equity.DEVICE.type, enabled=False):
                issuer_state = _issuer_states(model, batch, True)
                overall_hat, call_hat, put_hat = option_head(issuer_state, option_x, ids)
                overall_target = torch.from_numpy(batch["targets"][:, 0]).to(equity.DEVICE)
                call_target = torch.from_numpy(np.nan_to_num(batch["call_targets"], nan=0.0)).to(equity.DEVICE)
                put_target = torch.from_numpy(np.nan_to_num(batch["put_targets"], nan=0.0)).to(equity.DEVICE)
                loss = nn.functional.smooth_l1_loss(overall_hat, overall_target)
                call_mask = torch.from_numpy(np.isfinite(batch["call_targets"])).to(equity.DEVICE)
                put_mask = torch.from_numpy(np.isfinite(batch["put_targets"])).to(equity.DEVICE)
                if call_mask.any():
                    loss = loss + 0.25 * nn.functional.smooth_l1_loss(call_hat[call_mask], call_target[call_mask])
                if put_mask.any():
                    loss = loss + 0.25 * nn.functional.smooth_l1_loss(put_hat[put_mask], put_target[put_mask])
                loss = OPTION_LOSS_WEIGHT * loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(option_head.parameters()), 1.0)
                optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else 0.0


def _equity_scores(model, docs):
    model.eval(); rows = []
    with torch.no_grad():
        for start in range(0, len(docs), equity.BATCH_SIZE):
            selected = docs[start:start + equity.BATCH_SIZE]
            x, padding, _ = equity._batch(selected)
            graph_hat, _, _, _, _, _ = model(x.to(equity.DEVICE), padding.to(equity.DEVICE), causal=True)
            values = graph_hat.cpu().numpy()
            for row, doc in enumerate(selected):
                n = len(doc["x"])
                frame = pd.DataFrame({"symbol": doc["symbol"], "date": doc["date"]})
                frame[equity.TARGET_COLS] = values[row, :n]
                rows.append(frame)
    pred = pd.concat(rows, ignore_index=True)
    for column in equity.TARGET_COLS:
        pred[column] = pred.groupby("date")[column].rank(pct=True, method="average")
    pred["long_score"] = pred["long_hub"]
    pred["short_score"] = pred["short_hub"]
    pred["long_exit_score"] = pred["long_authority"]
    pred["short_exit_score"] = pred["short_authority"]
    pred["long_agree_count"] = (pred.long_score >= pred.short_score).astype(int)
    pred["short_agree_count"] = (pred.short_score > pred.long_score).astype(int)
    pred["model_count"] = 1
    return pred


def _trade(pred, prices, tier, year, model_name):
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    next_returns = close.pct_change().shift(-1)
    dates = pd.DatetimeIndex(next_returns.index[
        (next_returns.index >= f"{year}-01-01") & (next_returns.index <= f"{year}-12-31")
    ])
    summary, _, _ = equity.gnn.run_shared_book_framework_comparison(
        scores=pred[["symbol", "date", "long_score", "short_score", "long_exit_score", "short_exit_score", "long_agree_count", "short_agree_count", "model_count"]],
        next_returns=next_returns, symbols=tuple(close.columns), dates=dates,
        variants=("long_only",), top_k_values=(min(20, len(close.columns)),),
        entry_threshold=0.5, exit_threshold=0.5,
        cost_models={"family_common": equity.gnn.SharedBookCostModel(0.5, 5.0)},
    )
    if summary.empty:
        return summary
    summary["tier"] = tier; summary["year"] = year; summary["model"] = model_name
    return summary


def main():
    tier = os.getenv("OPTION_DOCUMENT_TIER", "1T").upper()
    base, prices, feature_cols = equity._prepare_data(tier)
    base = base.copy(); base["symbol"] = base.symbol.astype(str).str.upper()
    base["date"] = pd.to_datetime(base.date).dt.normalize()
    symbols = sorted(base.symbol.unique())
    equity_dates = set(zip(base.symbol, base.date))
    docs = [] if BASELINE_ONLY else option.load_or_build_option_documents(symbols, equity_dates)
    all_results = []
    for year in range(equity.FIRST_TEST_YEAR, equity.LAST_TEST_YEAR + 1):
        normalized = equity._normalize(base, feature_cols, year)
        train_docs = equity._make_docs(normalized, year, True)
        test_docs = equity._make_docs(normalized, year, False)
        cross_enabled = equity.CROSS_SECTIONAL_ENABLED and (
            bool(equity.MACRO_EVENT_COLS) or "year_target" in equity.AUX_COLS
        )
        cross_train_docs = equity._make_cross_sectional_docs(normalized, year, True) if cross_enabled else []
        train_idx = sorted(i for i, doc in enumerate(docs) if doc["date"].year < year)
        test_idx = [i for i, doc in enumerate(docs) if doc["date"].year == year]
        if not train_docs or not test_docs or (not BASELINE_ONLY and not test_idx):
            continue
        equity_map = {(symbol, date): np.asarray(value, dtype=np.float32)
                      for (symbol, date), value in normalized.set_index(["symbol", "date"])["__x__"].items()}
        if not BASELINE_ONLY and train_idx:
            option.normalize_fold(docs, equity_map, train_idx)
        if OPTION_ONLY:
            train_batches = option.prepare_fold_batches(docs, train_idx, np.array([])) if train_idx else []
            model_specs = [("equity_plus_option_tasks", True)] if train_idx else [("equity_plus_option_tasks_no_option_history", False)]
        elif BASELINE_ONLY:
            train_batches = []
            model_specs = [("equity_baseline", False)]
        elif train_idx:
            train_batches = option.prepare_fold_batches(docs, train_idx, np.array([]))
            model_specs = [("equity_baseline", False), ("equity_plus_option_tasks", True)]
        else:
            # Equity history exists before the first option year.  Preserve
            # that equity-only test year instead of dropping it from WFO.
            train_batches = []
            model_specs = [
                ("equity_baseline", False),
                ("equity_plus_option_tasks_no_option_history", False),
            ]
        training_docs = train_docs + cross_train_docs
        for model_name, use_option in model_specs:
            torch.manual_seed(equity.SEED); np.random.seed(equity.SEED)
            extra_tasks = ("option",) if use_option else ()
            model = equity.TransformerMTL(
                len(feature_cols),
                {name: int(normalized[name].max()) + 1 for name in equity.AUX_COLS},
                extra_task_names=extra_tasks,
            ).to(equity.DEVICE)
            option_head = SharedOptionTask(len(option.OPTION_FEATURES)).to(equity.DEVICE)
            parameters = list(model.parameters()) + (list(option_head.parameters()) if use_option else [])
            optimizer = torch.optim.AdamW(parameters, lr=equity.LR, weight_decay=1e-4)
            print({"tier": tier, "year": year, "model": model_name, "option_tasks": use_option, "option_loss_weight": OPTION_LOSS_WEIGHT}, flush=True)
            for epoch in range(equity.EPOCHS):
                equity_loss = _equity_epoch(model, training_docs, optimizer)
                option_loss = _option_loss(model, option_head, train_batches, optimizer, True) if use_option else 0.0
                if epoch == 0 or epoch == equity.EPOCHS - 1:
                    print({"year": year, "model": model_name, "epoch": epoch + 1, "equity_loss": equity_loss, "option_loss": option_loss}, flush=True)
            if use_option:
                print({"year": year, "model": model_name, "existing_task_router_option_weights": model.routing_weights()["option"]}, flush=True)
            pred = _equity_scores(model, test_docs)
            result = _trade(pred, prices, tier, year, model_name)
            if not result.empty:
                print({"year": year, "model": model_name, "equity_trading": result.to_dict("records")}, flush=True)
                all_results.append(result)
    result = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    path = OUT / f"{tier.lower()}_equity_vs_option_aux_wfo.csv"
    result.to_csv(path, index=False)
    print(result.to_string(index=False) if not result.empty else result, flush=True)
    print({"output": str(path), "rows": len(result)}, flush=True)


if __name__ == "__main__":
    main()
