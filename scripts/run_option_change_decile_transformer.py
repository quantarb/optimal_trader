"""Causal transformer with ten call/put change-percent prototype classes."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
RECON_PATH = ROOT / "scripts" / "run_option_reconstruction_transformer.py"
spec = importlib.util.spec_from_file_location("option_recon", RECON_PATH)
assert spec and spec.loader
recon = importlib.util.module_from_spec(spec); spec.loader.exec_module(recon)
option_proto = recon.option_proto; gnn = recon.gnn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 20260716; LOOKBACK = 20; EPOCHS = 3; BATCH_SIZE = 256; N_BINS = 10
ISSUER_FEATURES: list[str] = []


class DecileTransformer(nn.Module):
    def __init__(self, issuer_dim: int, aux_dims: dict[str, int], event_dim: int, macro_dim: int, hidden: int = 64):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(issuer_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=hidden * 2, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.position = nn.Parameter(torch.zeros(1, LOOKBACK, hidden))
        self.call_head = option_proto.PrototypeHead(hidden, N_BINS)
        self.put_head = option_proto.PrototypeHead(hidden, N_BINS)
        self.aux_heads = nn.ModuleDict({name: option_proto.PrototypeHead(hidden, size) for name, size in aux_dims.items()})
        self.event_head = gnn.EventPrototypeHead(hidden, event_dim)
        self.macro_head = gnn.EventPrototypeHead(hidden, macro_dim)
        self.graph_head = nn.Linear(hidden, len(option_proto.GRAPH_FEATURE_TARGETS))

    def forward(self, x: torch.Tensor):
        h = self.encoder(self.input(x) + self.position[:, :x.shape[1]])
        state = h[:, -1]
        aux = {name: head(state) for name, head in self.aux_heads.items()}
        return self.call_head(state), self.put_head(state), aux, self.event_head(state), self.macro_head(state), self.graph_head(state)


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
    sparse = recon.build_sparse_targets(rows)
    issuer_base, _, feature_cols = option_proto.transformer_module._prepare_data("1T")
    issuer_frame = issuer_base[["symbol", "date", *feature_cols]].copy(); issuer_frame["symbol"] = issuer_frame.symbol.astype(str).str.upper(); issuer_frame["date"] = pd.to_datetime(issuer_frame.date).dt.normalize()
    sparse = sparse.merge(issuer_frame, on=["symbol", "date"], how="inner").sort_values(["symbol", "date"]).reset_index(drop=True)
    global ISSUER_FEATURES; ISSUER_FEATURES = feature_cols
    train_date = sparse.date.dt.year < 2025
    issuer = sparse[ISSUER_FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    issuer_fill = issuer.loc[train_date].median().fillna(0.0); issuer_mean = issuer.loc[train_date].fillna(issuer_fill).mean().fillna(0.0); issuer_std = issuer.loc[train_date].fillna(issuer_fill).std().replace(0, 1).fillna(1.0)
    issuer = ((issuer.fillna(issuer_fill) - issuer_mean) / issuer_std).clip(-8, 8)
    call_change = pd.to_numeric(sparse["call_target_change_percent"], errors="coerce")
    put_change = pd.to_numeric(sparse["put_target_change_percent"], errors="coerce")
    call_cuts = np.nanquantile(call_change.loc[train_date], np.linspace(.1, .9, 9)); put_cuts = np.nanquantile(put_change.loc[train_date], np.linspace(.1, .9, 9))
    call_bin = np.digitize(call_change.fillna(call_change.loc[train_date].median()), call_cuts).astype(np.int64)
    put_bin = np.digitize(put_change.fillna(put_change.loc[train_date].median()), put_cuts).astype(np.int64)
    sequences=[]; call_y=[]; put_y=[]; call_mask=[]; put_mask=[]; metas=[]
    for _, group in sparse.groupby("symbol", sort=False):
        group = group.sort_values("date"); indices = group.index.to_list()
        for pos, row_index in enumerate(indices):
            if pos < LOOKBACK: continue
            sequences.append(issuer.loc[indices[pos-LOOKBACK:pos]].to_numpy(np.float32)); call_y.append(call_bin[row_index]); put_y.append(put_bin[row_index]); call_mask.append(bool(sparse.loc[row_index,"has_call"])); put_mask.append(bool(sparse.loc[row_index,"has_put"])); metas.append(row_index)
    x=np.asarray(sequences,np.float32); call_y=np.asarray(call_y,np.int64); put_y=np.asarray(put_y,np.int64); call_mask=np.asarray(call_mask); put_mask=np.asarray(put_mask); sample_call_change=call_change.iloc[metas].to_numpy(); sample_put_change=put_change.iloc[metas].to_numpy(); meta=sparse.loc[metas].reset_index(drop=True); train=np.asarray(meta.date.dt.year<2025); train_idx=np.flatnonzero(train); valid_idx=np.flatnonzero(~train)
    aux_dims={name:max(1,int(pd.to_numeric(rows[name],errors="coerce").max())+1) if pd.to_numeric(rows[name],errors="coerce").notna().any() else 1 for name in gnn.AUX_TARGET_COLS}
    event_y=meta[list(gnn.ALL_EVENT_TARGETS)].fillna(0).to_numpy(np.float32); macro_cols=[c for c in meta.columns if str(c).startswith("is_macro_")]; macro_y=meta[macro_cols].fillna(0).to_numpy(np.float32); graph_y=meta[option_proto.GRAPH_FEATURE_TARGETS].fillna(0).to_numpy(np.float32); aux_y=meta[list(gnn.AUX_TARGET_COLS)].fillna(-1).to_numpy(np.int64)
    model=DecileTransformer(len(ISSUER_FEATURES),aux_dims,len(gnn.ALL_EVENT_TARGETS),len(macro_cols)).to(DEVICE); optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4)
    print({"issuer_features":len(ISSUER_FEATURES),"observed_issuer_dates":len(sparse),"samples":len(x),"train_samples":len(train_idx),"validation_samples":len(valid_idx),"device":str(DEVICE)},flush=True)
    print({"call_cutpoints":np.round(call_cuts,4).tolist(),"put_cutpoints":np.round(put_cuts,4).tolist()},flush=True)
    for epoch in range(EPOCHS):
        model.train(); order=np.random.permutation(train_idx); losses=[]
        for start in range(0,len(order),BATCH_SIZE):
            idx=order[start:start+BATCH_SIZE]; ch,ph,aux,event,macro,graph=model(torch.from_numpy(x[idx]).to(DEVICE)); loss=torch.tensor(0.,device=DEVICE); cm=torch.from_numpy(call_mask[idx]).to(DEVICE); pm=torch.from_numpy(put_mask[idx]).to(DEVICE)
            if cm.any(): loss=loss+nn.functional.cross_entropy(ch[cm],torch.from_numpy(call_y[idx]).to(DEVICE)[cm])
            if pm.any(): loss=loss+nn.functional.cross_entropy(ph[pm],torch.from_numpy(put_y[idx]).to(DEVICE)[pm])
            loss=loss+.1*gnn.event_loss_from_logits(event,torch.from_numpy(event_y[idx]).to(DEVICE))+.1*nn.functional.smooth_l1_loss(graph,torch.from_numpy(graph_y[idx]).to(DEVICE))
            if macro_cols: loss=loss+.1*gnn.event_loss_from_logits(macro,torch.from_numpy(macro_y[idx]).to(DEVICE))
            for ci,name in enumerate(gnn.AUX_TARGET_COLS):
                t=torch.from_numpy(aux_y[idx,ci]).to(DEVICE); m=t.ge(0)
                if m.any(): loss=loss+.05*nn.functional.cross_entropy(aux[name][m],t[m])
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        print({"epoch":epoch+1,"train_loss":round(float(np.mean(losses)),5)},flush=True)
    model.eval();
    with torch.no_grad(): ch,ph,_,_,_,_=model(torch.from_numpy(x[valid_idx]).to(DEVICE)); call_pred=ch.argmax(1).cpu().numpy(); put_pred=ph.argmax(1).cpu().numpy()
    result={}
    for side,pred,actual,change,mask in (("call",call_pred,call_y[valid_idx],sample_call_change[valid_idx],call_mask[valid_idx]),("put",put_pred,put_y[valid_idx],sample_put_change[valid_idx],put_mask[valid_idx])):
        valid=mask & np.isfinite(change); result[f"{side}_accuracy"]=float(np.mean(pred[valid]==actual[valid])); result[f"{side}_change_spearman"]=float(pd.Series(pred[valid]).corr(pd.Series(change[valid]),method="spearman")); result[f"{side}_predicted_bin0_change"] = float(np.mean(change[valid][pred[valid]==0])) if np.any(valid & (pred==0)) else float("nan"); result[f"{side}_predicted_bin9_change"] = float(np.mean(change[valid][pred[valid]==9])) if np.any(valid & (pred==9)) else float("nan")
    print({**result,"status":"complete"},flush=True)


if __name__ == "__main__": main()
