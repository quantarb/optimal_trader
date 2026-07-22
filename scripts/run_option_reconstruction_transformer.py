"""Causal issuer transformer reconstructing sparse raw call/put prototypes.

Option targets exist only when ThetaData has an observed chain for a specific
``(date, symbol)``.  No option dates are forward-filled or treated as a
continuous option time series.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "scripts" / "run_option_instrument_mtl_smoke.py"
spec = importlib.util.spec_from_file_location("option_proto", SMOKE)
assert spec and spec.loader
option_proto = importlib.util.module_from_spec(spec); spec.loader.exec_module(option_proto)
gnn = option_proto.gnn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 20260716
LOOKBACK = 20
EPOCHS = 3
BATCH_SIZE = 256
ISSUER_FEATURES = ["prev_close", "ret_1d", "ret_5d", "ret_20d"]
TARGET_FEATURES = [*option_proto.RAW_FEATURES, "target_change_percent", "prototype_contract_count"]


class ReconstructionTransformer(nn.Module):
    def __init__(self, issuer_dim: int, target_dim: int, aux_dims: dict[str, int], event_dim: int, macro_dim: int, hidden: int = 64):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(issuer_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=hidden * 2, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.position = nn.Parameter(torch.zeros(1, LOOKBACK, hidden))
        self.call_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, target_dim))
        self.put_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, target_dim))
        self.aux_heads = nn.ModuleDict({name: option_proto.PrototypeHead(hidden, size) for name, size in aux_dims.items()})
        self.event_head = gnn.EventPrototypeHead(hidden, event_dim)
        self.macro_head = gnn.EventPrototypeHead(hidden, macro_dim)
        self.graph_head = nn.Linear(hidden, len(option_proto.GRAPH_FEATURE_TARGETS))

    def forward(self, x: torch.Tensor):
        h = self.encoder(self.input(x) + self.position[:, :x.shape[1]])
        state = h[:, -1]
        aux = {name: head(state) for name, head in self.aux_heads.items()}
        return self.call_head(state), self.put_head(state), aux, self.event_head(state), self.macro_head(state), self.graph_head(state)


def make_issuer_panel(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices.copy(); out["symbol"] = out.symbol.astype(str).str.upper(); out["date"] = pd.to_datetime(out.date).dt.normalize()
    out = out.sort_values(["symbol", "date"])
    out["prev_close"] = out.groupby("symbol")["close"].shift(1)
    for horizon in (1, 5, 20):
        returns = out.groupby("symbol")["close"].pct_change(horizon)
        out[f"ret_{horizon}d"] = returns.groupby(out.symbol).shift(1)
    return out[["symbol", "date", *ISSUER_FEATURES]]


def build_sparse_targets(rows: pd.DataFrame) -> pd.DataFrame:
    """Pivot only observed call/put prototypes; absent sides remain masked."""
    base_cols = ["symbol", "date", *option_proto.GRAPH_FEATURE_TARGETS, *gnn.AUX_TARGET_COLS, *gnn.ALL_EVENT_TARGETS]
    macro_cols = [c for c in rows.columns if str(c).startswith("is_macro_")]
    base_cols += macro_cols
    meta = rows.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"])[[c for c in base_cols if c in rows.columns]].copy()
    parts = []
    for side in ("call", "put"):
        part = rows.loc[rows.prototype_type.eq(side), ["symbol", "date", *TARGET_FEATURES]].copy()
        part = part.rename(columns={column: f"{side}_{column}" for column in TARGET_FEATURES})
        parts.append(part)
    out = meta.merge(parts[0], on=["symbol", "date"], how="left").merge(parts[1], on=["symbol", "date"], how="left")
    out["has_call"] = out[[f"call_{c}" for c in TARGET_FEATURES]].notna().any(axis=1)
    out["has_put"] = out[[f"put_{c}" for c in TARGET_FEATURES]].notna().any(axis=1)
    return out


def evaluate_vector_store(meta: pd.DataFrame, call_query: np.ndarray, put_query: np.ndarray,
                          call_vectors: np.ndarray, put_vectors: np.ndarray,
                          call_observed: np.ndarray, put_observed: np.ndarray,
                          call_change: np.ndarray, put_change: np.ndarray, top_k: int = 5) -> dict[str, float]:
    """Search same-date call/put stores and measure proxy outcomes."""
    results: dict[str, float] = {}
    rng = np.random.default_rng(SEED)
    for side, queries, vectors, observed, changes in (
        ("call", call_query, call_vectors, call_observed, call_change),
        ("put", put_query, put_vectors, put_observed, put_change),
    ):
        top_values=[]; random_values=[]; bottom_values=[]; all_scores=[]; all_changes=[]; query_count=0
        for date, group in meta.groupby("date", sort=False):
            query_indices = group.index.to_numpy()
            candidates = query_indices[observed[query_indices]]
            if len(candidates) < 2: continue
            candidate_vectors = vectors[candidates]
            candidate_vectors = candidate_vectors / np.linalg.norm(candidate_vectors, axis=1, keepdims=True).clip(min=1e-8)
            for query_index in query_indices:
                if not observed[query_index]: continue
                eligible = candidates[candidates != query_index]
                if len(eligible) < 1: continue
                query_count += 1
                eligible_vectors = vectors[eligible]
                eligible_vectors = eligible_vectors / np.linalg.norm(eligible_vectors, axis=1, keepdims=True).clip(min=1e-8)
                query_vector = queries[query_index] / max(np.linalg.norm(queries[query_index]), 1e-8)
                scores = eligible_vectors @ query_vector
                order = np.argsort(-scores)
                k = min(top_k, len(order))
                top_indices = eligible[order[:k]]
                bottom_indices = eligible[order[-k:]]
                random_indices = rng.choice(eligible, size=k, replace=False)
                top_values.extend(changes[top_indices].tolist())
                bottom_values.extend(changes[bottom_indices].tolist())
                random_values.extend(changes[random_indices].tolist())
                all_scores.extend(scores.tolist()); all_changes.extend(changes[eligible].tolist())
        results[f"{side}_queries"] = float(query_count)
        results[f"{side}_top{k if 'k' in locals() else top_k}_change_percent"] = float(np.mean(top_values)) if top_values else float("nan")
        results[f"{side}_random_change_percent"] = float(np.mean(random_values)) if random_values else float("nan")
        results[f"{side}_bottom_change_percent"] = float(np.mean(bottom_values)) if bottom_values else float("nan")
        results[f"{side}_top_uplift_vs_random"] = results[f"{side}_top{k if 'k' in locals() else top_k}_change_percent"] - results[f"{side}_random_change_percent"]
        results[f"{side}_similarity_change_spearman"] = float(pd.Series(all_scores).corr(pd.Series(all_changes), method="spearman")) if all_scores else float("nan")
    return results


def main() -> None:
    torch.manual_seed(SEED); np.random.seed(SEED)
    index = pd.read_csv(gnn.feature_dir("1T") / "index.csv")
    first = pd.read_parquet(index.iloc[0].panel_path)
    symbols = sorted(first.symbol.astype(str).str.upper().unique())
    option_symbols = [s for s in symbols if s not in {"BRK-A", "BRK-B"}]
    prices, labels = gnn.build_price_and_labels(symbols, "1T")
    labels = labels.copy(); labels["symbol"] = labels.symbol.astype(str).str.upper(); labels["date"] = pd.to_datetime(labels.date).dt.normalize()
    macro_panel, _ = option_proto.transformer_module._load_macro_event_panel(labels.date)
    labels = labels.merge(macro_panel, on="date", how="left")
    rows = option_proto.load_rows(option_symbols, labels)
    sparse = build_sparse_targets(rows).merge(make_issuer_panel(prices), on=["symbol", "date"], how="inner")
    sparse = sparse.loc[sparse[ISSUER_FEATURES].notna().all(axis=1)].sort_values(["symbol", "date"]).reset_index(drop=True)
    train_date = sparse.date.dt.year < 2025
    issuer = sparse[ISSUER_FEATURES].apply(pd.to_numeric, errors="coerce")
    issuer_mean = issuer.loc[train_date].mean().fillna(0.0); issuer_std = issuer.loc[train_date].std().replace(0, 1).fillna(1.0)
    issuer = ((issuer.fillna(issuer_mean) - issuer_mean) / issuer_std).clip(-8, 8)
    target = sparse[[f"{side}_{column}" for side in ("call", "put") for column in TARGET_FEATURES]].apply(pd.to_numeric, errors="coerce")
    train_target = target.loc[train_date]
    target_fill = train_target.median().fillna(0.0)
    target_mean = train_target.fillna(target_fill).mean().fillna(0.0); target_std = train_target.fillna(target_fill).std().replace(0, 1).fillna(1.0)
    target_scaled = ((target.fillna(target_fill) - target_mean) / target_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Only observed option dates are samples.  The sequence is equity history
    # ending immediately before the target date, so the target is not visible.
    sequences=[]; call_y=[]; put_y=[]; call_mask=[]; put_mask=[]; metas=[]
    for symbol, group in sparse.groupby("symbol", sort=False):
        group = group.sort_values("date"); idxs = group.index.to_list()
        for position, row_index in enumerate(idxs):
            if position < LOOKBACK: continue
            history = issuer.loc[idxs[position-LOOKBACK:position]].to_numpy(np.float32)
            sequences.append(history)
            call_y.append(target_scaled.loc[row_index, [f"call_{c}" for c in TARGET_FEATURES]].to_numpy(np.float32))
            put_y.append(target_scaled.loc[row_index, [f"put_{c}" for c in TARGET_FEATURES]].to_numpy(np.float32))
            call_mask.append(bool(sparse.loc[row_index, "has_call"])); put_mask.append(bool(sparse.loc[row_index, "has_put"])); metas.append(row_index)
    x=np.asarray(sequences,np.float32); call_y=np.asarray(call_y,np.float32); put_y=np.asarray(put_y,np.float32); call_mask=np.asarray(call_mask); put_mask=np.asarray(put_mask)
    meta=sparse.loc[metas].reset_index(drop=True); train=np.asarray(meta.date.dt.year < 2025); train_idx=np.flatnonzero(train); valid_idx=np.flatnonzero(~train)
    aux_dims={name:int(pd.to_numeric(sparse[name],errors="coerce").max())+1 for name in gnn.AUX_TARGET_COLS}
    event_y=meta[list(gnn.ALL_EVENT_TARGETS)].fillna(0).to_numpy(np.float32); macro_cols=[c for c in meta.columns if str(c).startswith("is_macro_")]; macro_y=meta[macro_cols].fillna(0).to_numpy(np.float32); graph_y=meta[option_proto.GRAPH_FEATURE_TARGETS].fillna(0).to_numpy(np.float32); aux_y=meta[list(gnn.AUX_TARGET_COLS)].fillna(-1).to_numpy(np.int64)
    model=ReconstructionTransformer(len(ISSUER_FEATURES),len(TARGET_FEATURES),aux_dims,len(gnn.ALL_EVENT_TARGETS),len(macro_cols)).to(DEVICE); optimizer=torch.optim.AdamW(model.parameters(),lr=0.002,weight_decay=1e-4)
    print({"observed_issuer_dates":len(sparse),"sequence_samples":len(x),"train_samples":len(train_idx),"validation_samples":len(valid_idx),"observed_call_targets":int(call_mask.sum()),"observed_put_targets":int(put_mask.sum()),"device":str(DEVICE)},flush=True)
    for epoch in range(EPOCHS):
        model.train(); order=np.random.permutation(train_idx); losses=[]
        for start in range(0,len(order),BATCH_SIZE):
            idx=order[start:start+BATCH_SIZE]; call_hat,put_hat,aux_hat,event_hat,macro_hat,graph_hat=model(torch.from_numpy(x[idx]).to(DEVICE)); loss=torch.tensor(0.,device=DEVICE)
            cy=torch.from_numpy(call_y[idx]).to(DEVICE); py=torch.from_numpy(put_y[idx]).to(DEVICE); cm=torch.from_numpy(call_mask[idx]).to(DEVICE); pm=torch.from_numpy(put_mask[idx]).to(DEVICE)
            if cm.any(): loss=loss+nn.functional.smooth_l1_loss(call_hat[cm],cy[cm])
            if pm.any(): loss=loss+nn.functional.smooth_l1_loss(put_hat[pm],py[pm])
            loss=loss+0.1*gnn.event_loss_from_logits(event_hat,torch.from_numpy(event_y[idx]).to(DEVICE))+0.1*nn.functional.smooth_l1_loss(graph_hat,torch.from_numpy(graph_y[idx]).to(DEVICE))
            if macro_cols: loss=loss+0.1*gnn.event_loss_from_logits(macro_hat,torch.from_numpy(macro_y[idx]).to(DEVICE))
            for ci,name in enumerate(gnn.AUX_TARGET_COLS):
                t=torch.from_numpy(aux_y[idx,ci]).to(DEVICE); m=t.ge(0)
                if m.any(): loss=loss+0.05*nn.functional.cross_entropy(aux_hat[name][m],t[m])
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();losses.append(float(loss.detach()))
        print({"epoch":epoch+1,"train_loss":round(float(np.mean(losses)),5)},flush=True)
    model.eval();
    with torch.no_grad():
        call_hat,put_hat,_,_,_,_=model(torch.from_numpy(x[valid_idx]).to(DEVICE)); call_hat=call_hat.cpu().numpy();put_hat=put_hat.cpu().numpy()
    metrics={"call_reconstruction_mae":float(np.nanmean(np.abs(call_hat-call_y[valid_idx])[call_mask[valid_idx]])),"put_reconstruction_mae":float(np.nanmean(np.abs(put_hat-put_y[valid_idx])[put_mask[valid_idx]]))}
    change_index=TARGET_FEATURES.index("target_change_percent")
    metrics["call_change_percent_mae_scaled"]=float(np.nanmean(np.abs(call_hat[:,change_index]-call_y[valid_idx,:,][...,change_index])[call_mask[valid_idx]]))
    metrics["put_change_percent_mae_scaled"]=float(np.nanmean(np.abs(put_hat[:,change_index]-put_y[valid_idx,:,][...,change_index])[put_mask[valid_idx]]))
    call_vectors = call_y[valid_idx]; put_vectors = put_y[valid_idx]
    call_change = call_vectors[:, change_index] * float(target_std["call_target_change_percent"]) + float(target_mean["call_target_change_percent"])
    put_change = put_vectors[:, change_index] * float(target_std["put_target_change_percent"]) + float(target_mean["put_target_change_percent"])
    retrieval = evaluate_vector_store(meta.iloc[valid_idx].reset_index(drop=True), call_hat, put_hat,
                                      call_vectors, put_vectors, call_mask[valid_idx], put_mask[valid_idx],
                                      call_change, put_change, top_k=5)
    print({**metrics, **retrieval, "status":"complete"},flush=True)


if __name__ == "__main__": main()
