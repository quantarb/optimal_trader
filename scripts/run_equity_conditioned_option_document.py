"""Equity-conditioned pooled option-document WFO experiment."""
from __future__ import annotations
import importlib.util
import os
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = ROOT.parent / "quant-warehouse"
sys.path.insert(0, str(WAREHOUSE))
TRANSFORMER_PATH = ROOT / "scripts" / "run_symbol_year_transformer_mtl.py"
spec = importlib.util.spec_from_file_location("equity_transformer", TRANSFORMER_PATH)
assert spec and spec.loader
equity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(equity)
from quant_warehouse.platforms.data_providers.thetadata.feature_engineering import filter_option_instrument_rows
from quant_warehouse.platforms.data_providers.thetadata.options import read_thetadata_eod_option_chain

TIER = os.getenv("OPTION_DOCUMENT_TIER", "1T").upper()
FIRST_YEAR = int(os.getenv("OPTION_DOCUMENT_FIRST_YEAR", "2021"))
LAST_YEAR = int(os.getenv("OPTION_DOCUMENT_LAST_YEAR", "2025"))
EPOCHS = int(os.getenv("OPTION_DOCUMENT_EPOCHS", "3"))
HIDDEN = int(os.getenv("OPTION_DOCUMENT_HIDDEN", "64"))
N_BINS = int(os.getenv("OPTION_DOCUMENT_BINS", "1"))
if N_BINS < 1:
    raise ValueError("OPTION_DOCUMENT_BINS must be at least 1")
DEVICE = torch.device(os.getenv("OPTION_DOCUMENT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
DOC_BATCH = int(os.getenv("OPTION_DOCUMENT_DOC_BATCH", "256"))
CACHE_DIR = Path(os.getenv("OPTION_DOCUMENT_CACHE_DIR", str(ROOT / ".cache")))
CACHE_VERSION = "filtered_v2"
AMP_ENABLED = DEVICE.type == "cuda" and os.getenv("OPTION_DOCUMENT_AMP", "1") != "0"
SEED = 20260722
torch.manual_seed(SEED); np.random.seed(SEED)

OPTION_FEATURES = [
    "underlying_price", "strike", "dte", "option_type_id",
    "bid", "ask", "mid", "iv", "volume", "open_interest",
    "delta", "vega", "theta", "rho", "gamma", "vanna", "charm",
    "vomma", "volga", "speed", "color", "zomma", "ultima",
]
LABEL = "change_percent"

class PooledOptionDocument(nn.Module):
    def __init__(self, equity_dim: int, option_dim: int):
        super().__init__()
        self.row_encoder = nn.Sequential(
            nn.Linear(equity_dim + option_dim, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU(),
        )
        self.mean_heads = nn.ModuleList([nn.Linear(HIDDEN, 1) for _ in range(N_BINS)])
        self.call_head = nn.Linear(HIDDEN, 1)
        self.put_head = nn.Linear(HIDDEN, 1)

    def forward(self, equity_x: torch.Tensor, option_x: torch.Tensor):
        state = equity_x.expand(option_x.shape[0], -1)
        z = self.row_encoder(torch.cat([state, option_x], dim=-1))
        pooled = z.mean(dim=0, keepdim=True)
        prototype_means = torch.cat([head(pooled) for head in self.mean_heads], dim=0).squeeze(-1)
        return prototype_means, self.call_head(pooled).squeeze(), self.put_head(pooled).squeeze()

def load_equity():
    base, prices, feature_cols = equity._prepare_data(TIER)
    base = base.copy()
    base["symbol"] = base.symbol.astype(str).str.upper()
    base["date"] = pd.to_datetime(base.date, errors="coerce").dt.normalize()
    return base, prices, feature_cols, sorted(base.symbol.unique())

def load_option_documents(symbols, equity_dates):
    requested = ["snapshot_date", "underlying_symbol", "contract_symbol", "option_type", *OPTION_FEATURES, LABEL]
    docs = []
    for number, symbol in enumerate(symbols, 1):
        chain = read_thetadata_eod_option_chain(
            symbol, start_date=f"{FIRST_YEAR}-01-01", end_date=f"{LAST_YEAR}-12-31",
            columns=requested, require_rich_columns=True,
        )
        if chain.empty:
            continue
        chain = filter_option_instrument_rows(chain).copy()
        chain["date"] = pd.to_datetime(chain.snapshot_date, errors="coerce").dt.normalize()
        chain["symbol"] = symbol
        chain["option_type_id"] = chain.option_type.astype(str).str.lower().str.startswith("p").astype(np.int64)
        numeric = chain[[LABEL, "bid", "ask"]].apply(pd.to_numeric, errors="coerce")
        keep = numeric[LABEL].notna() & numeric[LABEL].ne(0) & numeric.bid.notna() & numeric.ask.notna()
        chain = chain.loc[keep].copy()
        chain = chain.loc[list(zip(chain.symbol, chain.date)).__class__([key in equity_dates for key in zip(chain.symbol, chain.date)])].copy()
        if chain.empty:
            continue
        chain[LABEL] = pd.to_numeric(chain[LABEL], errors="coerce")
        for date, frame in chain.groupby("date", sort=True):
            values = frame[LABEL].to_numpy(np.float32)
            call = frame.loc[frame.option_type_id.eq(0), LABEL].to_numpy(np.float32)
            put = frame.loc[frame.option_type_id.eq(1), LABEL].to_numpy(np.float32)
            raw = frame[OPTION_FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            docs.append({
                "symbol": symbol, "date": pd.Timestamp(date), "option": raw.to_numpy(np.float32),
                "target_rows": values,
                "target": float(np.mean(values)),
                "call_target": float(np.mean(call)) if len(call) else np.nan,
                "put_target": float(np.mean(put)) if len(put) else np.nan,
                "rows": len(frame),
            })
        print({"symbol": symbol, "symbol_number": number, "filtered_rows": len(chain), "documents": int(chain.date.nunique())}, flush=True)
    docs.sort(key=lambda item: (item["date"], item["symbol"]))
    return docs

def load_or_build_option_documents(symbols, equity_dates):
    """Load the expensive filtered chains once and reuse them across WFO folds."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / (
        f"option_documents_{TIER}_{FIRST_YEAR}_{LAST_YEAR}_{CACHE_VERSION}.pkl"
    )
    if os.getenv("OPTION_DOCUMENT_REBUILD_CACHE", "0") != "1" and cache_path.exists():
        with cache_path.open("rb") as handle:
            docs = pickle.load(handle)
        print({"option_document_cache": str(cache_path), "status": "loaded", "documents": len(docs)}, flush=True)
        return docs
    docs = load_option_documents(symbols, equity_dates)
    with cache_path.open("wb") as handle:
        pickle.dump(docs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print({"option_document_cache": str(cache_path), "status": "written", "documents": len(docs)}, flush=True)
    return docs

def normalize_fold(docs, equity_map, train):
    train_rows = np.concatenate([docs[i]["option"] for i in train], axis=0)
    train_df = pd.DataFrame(train_rows, columns=OPTION_FEATURES).replace([np.inf, -np.inf], np.nan)
    med = train_df.median().fillna(0.0)
    filled = train_df.fillna(med)
    mean = filled.mean().fillna(0.0)
    std = filled.std().replace(0, 1).fillna(1.0)
    for doc in docs:
        raw = pd.DataFrame(doc["option"], columns=OPTION_FEATURES).replace([np.inf, -np.inf], np.nan)
        doc["option_norm"] = ((raw.fillna(med).fillna(0.0) - mean) / std).clip(-8, 8).to_numpy(np.float32)
        doc["equity_norm"] = equity_map[(doc["symbol"], doc["date"])]

def prepare_fold_batches(docs, indices, bin_cuts):
    """Build reusable flattened minibatches once instead of once per epoch."""
    batches = []
    for start in range(0, len(indices), DOC_BATCH):
        batch_indices = indices[start:start + DOC_BATCH]
        options = np.concatenate([docs[i]["option_norm"] for i in batch_indices], axis=0)
        document_ids = np.concatenate([
            np.full(len(docs[i]["option_norm"]), position, dtype=np.int64)
            for position, i in enumerate(batch_indices)
        ])
        equity_states = np.stack([docs[i]["equity_norm"] for i in batch_indices]).astype(np.float32)
        targets = []
        call_targets = []
        put_targets = []
        actual = []
        for i in batch_indices:
            doc = docs[i]
            row_bins = np.digitize(doc["target_rows"], bin_cuts)
            targets.append([
                doc["target_rows"][row_bins == index].mean()
                if np.any(row_bins == index) else doc["target"]
                for index in range(N_BINS)
            ])
            call_targets.append(doc["call_target"])
            put_targets.append(doc["put_target"])
            actual.append(doc["target"])
        batches.append({
            "doc_indices": list(batch_indices),
            "options": options,
            "document_ids": document_ids,
            "equity_states": equity_states,
            "targets": np.asarray(targets, dtype=np.float32),
            "call_targets": np.asarray(call_targets, dtype=np.float32),
            "put_targets": np.asarray(put_targets, dtype=np.float32),
            "actual": np.asarray(actual, dtype=np.float32),
        })
    return batches

def predict_equity_scores(model, batches, docs):
    """Turn option-document outputs into issuer-date scores for equity WFO."""
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in batches:
            with torch.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
                prototype_hat, call_hat, put_hat = _batched_forward(model, batch)
            prototype = prototype_hat[:, 0].float().cpu().numpy()
            call = call_hat.float().cpu().numpy()
            put = put_hat.float().cpu().numpy()
            for position, doc_index in enumerate(batch["doc_indices"]):
                doc = docs[doc_index]
                rows.append({
                    "symbol": doc["symbol"], "date": doc["date"],
                    "option_overall_score": float(prototype[position]),
                    "option_call_score": float(call[position]),
                    "option_put_score": float(put[position]),
                })
    return pd.DataFrame(rows)

def backtest_equity_scores(scores, prices, test_year):
    """Evaluate option predictions with the same next-day equity backtest."""
    if scores.empty:
        return pd.DataFrame()
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    next_returns = close.pct_change().shift(-1)
    effective_top_k = min(20, len(close.columns))
    dates = pd.DatetimeIndex(next_returns.index[
        (next_returns.index >= f"{test_year}-01-01") &
        (next_returns.index <= f"{test_year}-12-31")
    ])
    summaries = []
    for signal in ("option_overall_score", "option_call_score", "option_put_score"):
        ranked = scores[["symbol", "date", signal]].copy()
        ranked["score"] = ranked.groupby("date")[signal].rank(pct=True, method="average")
        ranked["long_score"] = ranked["score"]
        ranked["short_score"] = 1.0 - ranked["score"]
        ranked["long_exit_score"] = ranked["long_score"]
        ranked["short_exit_score"] = ranked["short_score"]
        ranked["long_agree_count"] = (ranked.long_score >= ranked.short_score).astype(int)
        ranked["short_agree_count"] = (ranked.short_score > ranked.long_score).astype(int)
        ranked["model_count"] = 1
        summary, _, _ = equity.gnn.run_shared_book_framework_comparison(
            scores=ranked[["symbol", "date", "long_score", "short_score", "long_exit_score", "short_exit_score", "long_agree_count", "short_agree_count", "model_count"]],
            next_returns=next_returns, symbols=tuple(close.columns), dates=dates,
            variants=("long_only",), top_k_values=(effective_top_k,), entry_threshold=0.5,
            exit_threshold=0.5, cost_models={"family_common": equity.gnn.SharedBookCostModel(0.5, 5.0)},
        )
        if not summary.empty:
            summary["signal"] = signal
            summary["year"] = test_year
            summaries.append(summary)
    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()

def _batched_forward(model, batch):
    options = batch["options"]
    document_ids = batch["document_ids"]
    equity_states = batch["equity_states"]
    option_x = torch.from_numpy(options).to(DEVICE)
    ids = torch.from_numpy(document_ids).to(DEVICE)
    equity_x = torch.from_numpy(equity_states).to(DEVICE)
    row_equity = equity_x.index_select(0, ids)
    z = model.row_encoder(torch.cat([row_equity, option_x], dim=-1))
    pooled = torch.zeros((len(equity_states), HIDDEN), device=DEVICE, dtype=z.dtype)
    pooled.index_add_(0, ids, z)
    counts = torch.bincount(ids, minlength=len(equity_states)).to(device=DEVICE, dtype=z.dtype).unsqueeze(1)
    pooled = pooled / counts.clamp_min(1)
    prototype_means = torch.cat([head(pooled) for head in model.mean_heads], dim=1)
    return prototype_means, model.call_head(pooled).squeeze(-1), model.put_head(pooled).squeeze(-1)

def run_fold(model, batches, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    losses = []; actual = []; predicted = []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in batches:
            target = torch.from_numpy(batch["targets"]).to(DEVICE)
            call_target_np = np.nan_to_num(batch["call_targets"], nan=0.0)
            put_target_np = np.nan_to_num(batch["put_targets"], nan=0.0)
            call_mask = torch.from_numpy(np.isfinite(batch["call_targets"])).to(DEVICE)
            put_mask = torch.from_numpy(np.isfinite(batch["put_targets"])).to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
                prototype_hat, call_hat, put_hat = _batched_forward(model, batch)
                loss = nn.functional.smooth_l1_loss(prototype_hat, target)
                call_target = torch.from_numpy(call_target_np).to(DEVICE)
                put_target = torch.from_numpy(put_target_np).to(DEVICE)
                if call_mask.any():
                    loss = loss + 0.25 * nn.functional.smooth_l1_loss(call_hat[call_mask], call_target[call_mask])
                if put_mask.any():
                    loss = loss + 0.25 * nn.functional.smooth_l1_loss(put_hat[put_mask], put_target[put_mask])
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach()))
            actual.extend(batch["actual"].tolist())
            predicted.extend(prototype_hat.detach().mean(dim=1).cpu().numpy().tolist())
    actual = np.asarray(actual); predicted = np.asarray(predicted)
    return {
        "loss": float(np.mean(losses)),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "mae": float(np.mean(np.abs(actual - predicted))),
        "spearman": float(pd.Series(actual).corr(pd.Series(predicted), method="spearman") or 0.0),
        "documents": len(actual), "mean_target": float(actual.mean()), "mean_prediction": float(predicted.mean()),
    }

def main():
    base, prices, feature_cols, symbols = load_equity()
    equity_dates = set(zip(base.symbol, base.date))
    docs = load_or_build_option_documents(symbols, equity_dates)
    if not docs:
        raise RuntimeError("No filtered option documents found")
    results = []
    for test_year in range(FIRST_YEAR, LAST_YEAR + 1):
        train_idx = {i for i, d in enumerate(docs) if d["date"].year < test_year}
        test_idx = [i for i, d in enumerate(docs) if d["date"].year == test_year]
        if not train_idx or not test_idx:
            continue
        equity_norm = equity._normalize(base, feature_cols, test_year)
        equity_map = {
            (symbol, date): np.asarray(value, dtype=np.float32)
            for (symbol, date), value in equity_norm.set_index(["symbol", "date"])["__x__"].items()
        }
        normalize_fold(docs, equity_map, train_idx)
        train_targets = np.concatenate([docs[i]["target_rows"] for i in train_idx]).astype(float)
        bin_cuts = np.quantile(train_targets, np.linspace(0.2, 0.8, N_BINS - 1))
        train_batches = prepare_fold_batches(docs, sorted(train_idx), bin_cuts)
        test_batches = prepare_fold_batches(docs, test_idx, bin_cuts)
        model = PooledOptionDocument(len(feature_cols), len(OPTION_FEATURES)).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
        print({"tier": TIER, "test_year": test_year, "train_documents": len(train_idx), "test_documents": len(test_idx), "n_bins": N_BINS, "document_batch": DOC_BATCH, "bin_cutpoints": bin_cuts.tolist(), "device": str(DEVICE), "amp": AMP_ENABLED, "filtered_rule": "change_percent not null and != 0; bid and ask not null", "architecture": "prebatched_amp_equity_conditioned_per_option_encoder_scatter_mean_pool"}, flush=True)
        for epoch in range(EPOCHS):
            print({"year": test_year, "epoch": epoch + 1, "train": run_fold(model, train_batches, optimizer, scaler)}, flush=True)
        metric = run_fold(model, test_batches)
        metric["year"] = test_year
        results.append(metric)
        print({"year": test_year, "evaluation": metric}, flush=True)
        scores = predict_equity_scores(model, test_batches, docs)
        equity_summary = backtest_equity_scores(scores, prices, test_year)
        if not equity_summary.empty:
            equity_summary["tier"] = TIER
            equity_summary["architecture"] = "equity_conditioned_option_document"
            print({"year": test_year, "equity_trading": equity_summary.to_dict("records")}, flush=True)
            equity_results_path = CACHE_DIR / f"option_document_equity_wfo_{TIER.lower()}.csv"
            prior = pd.read_csv(equity_results_path) if equity_results_path.exists() else pd.DataFrame()
            pd.concat([prior, equity_summary], ignore_index=True).drop_duplicates(
                subset=["year", "signal", "variant", "top_k"], keep="last"
            ).to_csv(equity_results_path, index=False)
    print(pd.DataFrame(results).to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
