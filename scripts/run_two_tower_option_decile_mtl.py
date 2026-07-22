"""Two-tower ten-class call/put prototype compatibility experiment."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
RECON = ROOT / "scripts" / "run_option_reconstruction_transformer.py"
spec = importlib.util.spec_from_file_location("recon", RECON); assert spec and spec.loader
recon = importlib.util.module_from_spec(spec); spec.loader.exec_module(recon)
option_proto = recon.option_proto; gnn = recon.gnn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 20260716; LOOKBACK = 20; EPOCHS = 3; BATCH_SIZE = 512; N_CLASSES = 10
# The candidate embedding cannot contain the outcome being ranked.  The
# prototype's change_percent remains the supervised target/metadata only.
OPTION_FEATURES = [*option_proto.RAW_FEATURES, "prototype_contract_count"]


class DecileTwoTower(nn.Module):
    def __init__(self, issuer_dim: int, option_dim: int, aux_dims: dict[str, int], event_dim: int, macro_dim: int, metric_dim: int = 32):
        super().__init__()
        self.issuer = nn.Sequential(nn.Linear(issuer_dim, 96), nn.LayerNorm(96), nn.GELU(), nn.Linear(96, metric_dim))
        self.option = nn.Sequential(nn.Linear(option_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, metric_dim))
        self.option_type = nn.Embedding(2, metric_dim)
        self.call_prototypes = nn.Parameter(torch.randn(N_CLASSES, metric_dim) * .02)
        self.put_prototypes = nn.Parameter(torch.randn(N_CLASSES, metric_dim) * .02)
        self.aux_heads = nn.ModuleDict({name: option_proto.PrototypeHead(metric_dim, size) for name, size in aux_dims.items()})
        self.event_head = gnn.EventPrototypeHead(metric_dim, event_dim); self.macro_head = gnn.EventPrototypeHead(metric_dim, macro_dim)
        self.graph_head = nn.Linear(metric_dim, len(option_proto.GRAPH_FEATURE_TARGETS))

    def forward(self, issuer_x, option_x, option_type):
        # The sequence has already been constructed causally; use its latest
        # issuer state as the query representation.
        if issuer_x.ndim == 3:
            issuer_x = issuer_x[:, -1]
        q = nn.functional.normalize(self.issuer(issuer_x), dim=-1)
        v = nn.functional.normalize(self.option(option_x) + self.option_type(option_type), dim=-1)
        call_p = nn.functional.normalize(self.call_prototypes, dim=-1); put_p = nn.functional.normalize(self.put_prototypes, dim=-1)
        call_logits_q = q @ call_p.T; put_logits_q = q @ put_p.T
        call_logits_v = v @ call_p.T; put_logits_v = v @ put_p.T
        aux_q = {name: head(q) for name, head in self.aux_heads.items()}; aux_v = {name: head(v) for name, head in self.aux_heads.items()}
        tasks_q = (aux_q, self.event_head(q), self.macro_head(q), self.graph_head(q)); tasks_v = (aux_v, self.event_head(v), self.macro_head(v), self.graph_head(v))
        return q, v, call_logits_q, put_logits_q, call_logits_v, put_logits_v, tasks_q, tasks_v


def task_loss(tasks, aux_target, event_target, macro_target, graph_target):
    aux, event, macro, graph = tasks; loss=.1*gnn.event_loss_from_logits(event,event_target)+.1*nn.functional.smooth_l1_loss(graph,graph_target)
    if macro_target.shape[1]: loss=loss+.1*gnn.event_loss_from_logits(macro,macro_target)
    for ci,name in enumerate(gnn.AUX_TARGET_COLS):
        t=aux_target[:,ci]; m=t.ge(0)
        if m.any(): loss=loss+.05*nn.functional.cross_entropy(aux[name][m],t[m])
    return loss


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    index=pd.read_csv(gnn.feature_dir("1T")/"index.csv"); first=pd.read_parquet(index.iloc[0].panel_path); symbols=sorted(first.symbol.astype(str).str.upper().unique()); option_symbols=[s for s in symbols if s not in {"BRK-A","BRK-B"}]
    prices,labels=gnn.build_price_and_labels(symbols,"1T"); labels=labels.copy(); labels["symbol"]=labels.symbol.astype(str).str.upper(); labels["date"]=pd.to_datetime(labels.date).dt.normalize(); macro_panel,_=option_proto.transformer_module._load_macro_event_panel(labels.date); labels=labels.merge(macro_panel,on="date",how="left")
    rows=option_proto.load_rows(option_symbols,labels); sparse=recon.build_sparse_targets(rows); issuer_base,_,issuer_cols=option_proto.transformer_module._prepare_data("1T"); issuer_frame=issuer_base[["symbol","date",*issuer_cols]].copy(); issuer_frame["symbol"]=issuer_frame.symbol.astype(str).str.upper(); issuer_frame["date"]=pd.to_datetime(issuer_frame.date).dt.normalize(); sparse=sparse.merge(issuer_frame,on=["symbol","date"],how="inner").sort_values(["symbol","date"]).reset_index(drop=True)
    train_date=sparse.date.dt.year<2025; issuer=sparse[issuer_cols].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan); fill=issuer.loc[train_date].median().fillna(0); mean=issuer.loc[train_date].fillna(fill).mean().fillna(0); std=issuer.loc[train_date].fillna(fill).std().replace(0,1).fillna(1); issuer=((issuer.fillna(fill)-mean)/std).clip(-8,8)
    call_change=pd.to_numeric(sparse.call_target_change_percent,errors="coerce"); put_change=pd.to_numeric(sparse.put_target_change_percent,errors="coerce"); call_cuts=np.nanquantile(call_change.loc[train_date],np.linspace(.1,.9,9)); put_cuts=np.nanquantile(put_change.loc[train_date],np.linspace(.1,.9,9)); call_bin=np.digitize(call_change.fillna(call_change.loc[train_date].median()),call_cuts); put_bin=np.digitize(put_change.fillna(put_change.loc[train_date].median()),put_cuts)
    option_raw=sparse[[f"{side}_{c}" for side in ("call","put") for c in OPTION_FEATURES]].apply(pd.to_numeric,errors="coerce"); tf=option_raw.loc[train_date].median().fillna(0); tm=option_raw.loc[train_date].fillna(tf).mean().fillna(0); ts=option_raw.loc[train_date].fillna(tf).std().replace(0,1).fillna(1); option_scaled=((option_raw.fillna(tf)-tm)/ts).fillna(0)
    xs=[]; os=[]; types=[]; bins=[]; changes=[]; metas=[]
    for _,group in sparse.groupby("symbol",sort=False):
        group=group.sort_values("date"); inds=group.index.to_list()
        for pos,ri in enumerate(inds):
            if pos<LOOKBACK: continue
            # Same-date state alignment: the issuer query and its observed
            # call/put prototype are both from this exact (date, symbol).
            # This is a state-matching experiment, not a pre-close forecast.
            history=issuer.loc[ri].to_numpy(np.float32)
            for side,side_id in (("call",0),("put",1)):
                if not bool(sparse.loc[ri,f"has_{side}"]): continue
                xs.append(history); os.append(option_scaled.loc[ri,[f"{side}_{c}" for c in OPTION_FEATURES]].to_numpy(np.float32)); types.append(side_id); bins.append((call_bin if side=="call" else put_bin)[ri]); changes.append((call_change if side=="call" else put_change).iloc[ri]); metas.append(ri)
    x=np.asarray(xs,np.float32); ox=np.asarray(os,np.float32); types=np.asarray(types,np.int64); bins=np.asarray(bins,np.int64); changes=np.asarray(changes,np.float32); meta=sparse.loc[metas].reset_index(drop=True); train=np.asarray(meta.date.dt.year<2025); tr=np.flatnonzero(train); va=np.flatnonzero(~train)
    aux_dims={n:max(1,int(pd.to_numeric(rows[n],errors="coerce").max())+1) if pd.to_numeric(rows[n],errors="coerce").notna().any() else 1 for n in gnn.AUX_TARGET_COLS}; event_y=meta[list(gnn.ALL_EVENT_TARGETS)].fillna(0).to_numpy(np.float32); macro_cols=[c for c in meta.columns if str(c).startswith("is_macro_")]; macro_y=meta[macro_cols].fillna(0).to_numpy(np.float32); graph_y=meta[option_proto.GRAPH_FEATURE_TARGETS].fillna(0).to_numpy(np.float32); aux_y=meta[list(gnn.AUX_TARGET_COLS)].fillna(-1).to_numpy(np.int64)
    model=DecileTwoTower(len(issuer_cols),len(OPTION_FEATURES),aux_dims,len(gnn.ALL_EVENT_TARGETS),len(macro_cols)).to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4); print({"issuer_features":len(issuer_cols),"samples":len(x),"train_samples":len(tr),"validation_samples":len(va),"device":str(DEVICE)},flush=True)
    for epoch in range(EPOCHS):
        model.train(); order=np.random.permutation(tr); losses=[]
        for start in range(0,len(order),BATCH_SIZE):
            idx=order[start:start+BATCH_SIZE]; out=model(torch.from_numpy(x[idx]).to(DEVICE),torch.from_numpy(ox[idx]).to(DEVICE),torch.from_numpy(types[idx]).to(DEVICE)); q,v,cq,pq,cv,pv,tq,tv=out; target_bin=torch.from_numpy(bins[idx]).to(DEVICE); typ=torch.from_numpy(types[idx]).to(DEVICE); issuer_logits=torch.where(typ[:,None].eq(0),cq,pq); option_logits=torch.where(typ[:,None].eq(0),cv,pv); change_t=torch.from_numpy(changes[idx]).to(DEVICE); change_z=(change_t-torch.from_numpy(changes[tr]).to(DEVICE).mean())/torch.from_numpy(changes[tr]).to(DEVICE).std().clamp_min(1e-4); score=(q*v).sum(1); loss=nn.functional.cross_entropy(issuer_logits,target_bin)+nn.functional.cross_entropy(option_logits,target_bin)+.5*nn.functional.mse_loss(score,change_z)+task_loss(tq,torch.from_numpy(aux_y[idx]).to(DEVICE),torch.from_numpy(event_y[idx]).to(DEVICE),torch.from_numpy(macro_y[idx]).to(DEVICE),torch.from_numpy(graph_y[idx]).to(DEVICE))+task_loss(tv,torch.from_numpy(aux_y[idx]).to(DEVICE),torch.from_numpy(event_y[idx]).to(DEVICE),torch.from_numpy(macro_y[idx]).to(DEVICE),torch.from_numpy(graph_y[idx]).to(DEVICE)); opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
        print({"epoch":epoch+1,"train_loss":round(float(np.mean(losses)),5)},flush=True)
    model.eval();
    with torch.no_grad(): out=model(torch.from_numpy(x[va]).to(DEVICE),torch.from_numpy(ox[va]).to(DEVICE),torch.from_numpy(types[va]).to(DEVICE)); q,v,cq,pq,cv,pv,_,_=out; typ=torch.from_numpy(types[va]).to(DEVICE); logits=torch.where(typ[:,None].eq(0),cq,pq); pred=logits.argmax(1).cpu().numpy(); q=q.cpu().numpy(); v=v.cpu().numpy()
    result={"call_accuracy":float(np.mean(pred[types[va]==0]==bins[va][types[va]==0])),"put_accuracy":float(np.mean(pred[types[va]==1]==bins[va][types[va]==1]))}
    rng=np.random.default_rng(SEED)
    validation_meta=meta.iloc[va].reset_index(drop=True)
    for side,side_id in (("call",0),("put",1)):
        top_values=[]; random_values=[]; query_count=0
        side_indices=np.flatnonzero(types[va]==side_id)
        for date, date_group in validation_meta.iloc[side_indices].groupby("date",sort=False):
            date_indices=date_group.index.to_numpy()
            if len(date_indices)<2: continue
            for query_index in date_indices:
                query_symbol=validation_meta.iloc[query_index].symbol
                eligible=np.asarray([candidate for candidate in date_indices if validation_meta.iloc[candidate].symbol != query_symbol],dtype=int)
                if not len(eligible): continue
                candidate_scores=v[eligible] @ q[query_index]
                k=min(5,len(eligible)); order=np.argsort(-candidate_scores)[:k]; top_indices=eligible[order]; random_indices=rng.choice(eligible,size=k,replace=False)
                top_values.extend(changes[va][top_indices].tolist()); random_values.extend(changes[va][random_indices].tolist()); query_count+=1
        result[f"{side}_top5_queries"]=query_count; result[f"{side}_top5_change_percent"]=float(np.mean(top_values)) if top_values else float("nan"); result[f"{side}_random5_change_percent"]=float(np.mean(random_values)) if random_values else float("nan"); result[f"{side}_top5_uplift"]=result[f"{side}_top5_change_percent"]-result[f"{side}_random5_change_percent"]
    print({**result,"status":"complete"},flush=True)


if __name__ == "__main__": main()
