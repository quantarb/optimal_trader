"""1T feature-family GNN smoke test: train on 2024, trade on 2025.

Each family gets an independent causal temporal GNN.  The shared encoder uses
the long/short HITS graph as its message-passing graph and predicts six
direction-specific targets: long/short hub, authority, and PageRank.  Only
the hub/authority outputs are used by the current trading backtest; PageRank
is trained as an additional graph target.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
for repo in (REPO_ROOT, WORKSPACE_ROOT / "quant-warehouse", WORKSPACE_ROOT / "quant-orchestrator"):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

from quant_warehouse.platforms.data_providers.fmp.target_engineering import (
    HitsLabelSpec,
    build_hits_labels,
    build_inverse_holding_time_hits_labels,
)
from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs.store import (
    build_event_pairs_from_historical_data,
)
from quant_warehouse.warehouse.api import Warehouse
from quant_orchestrator.platforms.backtesting_frameworks.shared_book import (
    SharedBookCostModel,
    run_shared_book_framework_comparison,
)

FIRST_TEST_YEAR = int(os.getenv("GNN_FIRST_TEST_YEAR", "2021"))
LAST_TEST_YEAR = int(os.getenv("GNN_LAST_TEST_YEAR", "2025"))
# Use the full available warehouse history by default.  Set GNN_DATA_START only
# when intentionally running a restricted historical experiment.
DATA_START = pd.Timestamp(os.getenv("GNN_DATA_START", "1900-01-01"))
DATA_END = pd.Timestamp(f"{LAST_TEST_YEAR}-12-31")
MAX_HOLD = int(os.getenv("GNN_MAX_HOLD", "120"))
HITS_ITERATIONS = int(os.getenv("GNN_HITS_ITERATIONS", "50"))
HITS_TAIL_QUANTILE = float(os.getenv("GNN_HITS_TAIL_QUANTILE", "0.20"))
SPEED_HORIZONS = tuple(
    int(value) for value in os.getenv("GNN_SPEED_HORIZONS", "5,20,60,120").split(",") if value.strip()
)
SPEED_TARGET_COLS = [
    "speed_long_hub", "speed_long_authority", "speed_short_hub",
    "speed_short_authority", "speed_long_pagerank", "speed_short_pagerank",
]
GNN_VARIANT = os.getenv("GNN_VARIANT", "long_only").strip().lower()
LOOKBACK = int(os.getenv("GNN_LOOKBACK", "10"))
EPOCHS = int(os.getenv("GNN_EPOCHS", "12"))
HIDDEN = int(os.getenv("GNN_HIDDEN", "48"))
MOE_EXPERTS = int(os.getenv("GNN_MOE_EXPERTS", "0"))
MOE_TOP_K = max(1, int(os.getenv("GNN_MOE_TOP_K", "2")))
PAIR_PER_SOURCE = int(os.getenv("GNN_PAIR_PER_SOURCE", "8"))
GNN_DEVICE_NAME = os.getenv("GNN_DEVICE", "auto").strip().lower()
if GNN_DEVICE_NAME == "auto":
    GNN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    GNN_DEVICE = torch.device(GNN_DEVICE_NAME)
if GNN_DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")
SEED = 20260716
torch.manual_seed(SEED)
np.random.seed(SEED)

OUT = REPO_ROOT / "artifacts" / "graph_oracle_feature_family_gnn_improved_hits"
OUT.mkdir(parents=True, exist_ok=True)
CACHE_VERSION = "wfo_full_history_targets_v5_single_inverse_time_speed_graph"
GRAPH_CACHE: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]] = {}
TIER_CONFIGS = {
    "1T": (1_000_000_000_000, "equity_meta_model_1t"),
    "100B": (100_000_000_000, "equity_meta_model_100b"),
    "10B": (10_000_000_000, "equity_meta_model_10b"),
}


def feature_dir(tier: str) -> Path:
    cap, cache = TIER_CONFIGS[tier]
    return REPO_ROOT / "artifacts" / "trading_app_v2" / cache / f"mcap_{cap}_train_2020-12-31_seed_20260707" / "feature_family_panels"


def normalize_prices(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy().reset_index() if "date" not in raw.columns else raw.copy()
    frame.columns = [str(c).lower() for c in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return frame[["date", "open", "high", "low", "close"]].dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _pagerank(weights: np.ndarray, damping: float = 0.85, iterations: int = 100) -> np.ndarray:
    n = len(weights)
    if n == 0:
        return np.zeros(0, dtype=float)
    score = np.full(n, 1.0 / n, dtype=float)
    out = weights.sum(axis=1)
    for _ in range(iterations):
        incoming = np.zeros(n, dtype=float)
        for i in range(n):
            if out[i] > 0:
                incoming += score[i] * weights[i] / out[i]
            else:
                incoming += score[i] / n
        score = (1.0 - damping) / n + damping * incoming
    return score / (score.max() or 1.0)


def add_pagerank_labels(price_frames: dict[str, pd.DataFrame], labels: pd.DataFrame) -> pd.DataFrame:
    """Add long/short PageRank targets computed from the same HITS graphs."""
    additions: list[pd.DataFrame] = []
    for symbol, raw in price_frames.items():
        frame = normalize_prices(raw)
        for _, year_frame in frame.groupby(frame.date.dt.year, sort=True):
            year_frame = year_frame.reset_index(drop=True)
            n = len(year_frame)
            if n < 2:
                continue
            high = year_frame.high.to_numpy(float)
            low = year_frame.low.to_numpy(float)
            valid = np.triu(np.ones((n, n), dtype=bool), 1)
            valid &= (np.arange(n)[None, :] - np.arange(n)[:, None]) <= MAX_HOLD
            long_w = np.where(valid, np.maximum(low[None, :] / high[:, None] - 1.0, 0.0), 0.0)
            short_w = np.where(valid, np.maximum(low[:, None] / high[None, :] - 1.0, 0.0), 0.0)
            additions.append(pd.DataFrame({
                "symbol": symbol.upper(), "date": year_frame.date,
                "long_pagerank": _pagerank(long_w), "short_pagerank": _pagerank(short_w),
            }))
    if not additions:
        return labels
    extra = pd.concat(additions, ignore_index=True)
    return labels.merge(extra, on=["symbol", "date"], how="left")


def add_speed_pagerank_labels(price_frames: dict[str, pd.DataFrame], labels: pd.DataFrame) -> pd.DataFrame:
    """Add PageRank targets from the single inverse-holding-time graph."""
    additions: list[pd.DataFrame] = []
    for symbol, raw in price_frames.items():
        frame = normalize_prices(raw)
        for _, year_frame in frame.groupby(frame.date.dt.year, sort=True):
            year_frame = year_frame.reset_index(drop=True)
            n = len(year_frame)
            if n < 2:
                continue
            high = year_frame.high.to_numpy(float)
            low = year_frame.low.to_numpy(float)
            index = np.arange(n)
            valid = np.triu(np.ones((n, n), dtype=bool), 1)
            valid &= (index[None, :] - index[:, None]) <= MAX_HOLD
            long_return = low[None, :] / high[:, None] - 1.0
            short_return = low[:, None] / high[None, :] - 1.0
            holding_days = index[None, :] - index[:, None]
            long_w = np.zeros((n, n), dtype=float)
            short_w = np.zeros((n, n), dtype=float)
            np.divide(1.0, holding_days, out=long_w, where=valid & (long_return > 0))
            np.divide(1.0, holding_days, out=short_w, where=valid & (short_return > 0))
            additions.append(pd.DataFrame({
                "symbol": symbol.upper(), "date": year_frame.date,
                "speed_long_pagerank": _pagerank(long_w),
                "speed_short_pagerank": _pagerank(short_w),
            }))
    if not additions:
        return labels
    return labels.merge(pd.concat(additions, ignore_index=True), on=["symbol", "date"], how="left")


EVENT_TARGETS = {
    "is_congressman_buy": ("congress", "congressman_buy", None),
    "is_congressman_sell": ("congress", "congressman_sell", None),
    "is_senator_buy": ("congress", "senator_buy", None),
    "is_senator_sell": ("congress", "senator_sell", None),
    "is_insider_buy": ("insider", "insider_buy", None),
    "is_insider_sell": ("insider", "insider_sell", None),
    "is_analyst_upgrade": ("analyst_rating", "analyst_upgrade", None),
    "is_analyst_downgrade": ("analyst_rating", "analyst_downgrade", None),
    "is_analyst_estimate_raise": ("analyst_estimate", "analyst_estimate_raise", None),
    "is_analyst_estimate_cut": ("analyst_estimate", "analyst_estimate_cut", None),
    "is_price_target_raise": ("price_target", "price_target_raise", None),
    "is_price_target_cut": ("price_target", "price_target_cut", None),
    "is_institutional_add": ("institutional", "institutional_add", None),
    "is_institutional_reduce": ("institutional", "institutional_reduce", None),
    "is_buyback_authorization": ("capital_action", "buyback_authorization", None),
    "is_equity_offering": ("capital_action", "equity_offering", None),
    "is_dividend_increase": ("dividend", "dividend_increase", None),
    "is_dividend_cut": ("dividend", "dividend_cut", None),
    "is_forward_split": ("split", "forward_split", None),
    "is_reverse_split": ("split", "reverse_split", None),
    "is_earnings_reported": ("earnings", "earnings_reported", None),
    "is_eps_beat": ("earnings", "eps_beat", None),
    "is_eps_miss": ("earnings", "eps_miss", None),
    "is_revenue_beat": ("earnings", "revenue_beat", None),
    "is_revenue_miss": ("earnings", "revenue_miss", None),
    "is_dividend_declared": ("dividend", "dividend_declared", None),
    "is_dividend_ex_date": ("dividend", "dividend_ex_date", None),
    "is_dividend_record_date": ("dividend", "dividend_record_date", None),
    "is_dividend_payment_date": ("dividend", "dividend_payment_date", None),
    "is_forward_split": ("split", "forward_split", None),
    "is_reverse_split": ("split", "reverse_split", None),
    "is_ipo_trading_started": ("profile", "ipo_trading_started", None),
    "is_sec_8k_filed": ("filing", "sec_8k_filed", None),
    "is_sec_10q_filed": ("filing", "sec_10q_filed", None),
    "is_sec_10k_filed": ("filing", "sec_10k_filed", None),
    "is_sec_form4_filed": ("filing", "sec_form4_filed", None),
}
GRAPH_EVENT_TARGETS = {
    "is_long_hub_event": "long_hub",
    "is_long_authority_event": "long_authority",
    "is_long_pagerank_event": "long_pagerank",
    "is_short_hub_event": "short_hub",
    "is_short_authority_event": "short_authority",
    "is_short_pagerank_event": "short_pagerank",
}
# The baseline MTL configuration uses one shared prototypical head for the
# company/event targets.  The six continuous graph targets remain node
# regression targets below; graph percentile events are intentionally not
# separate event tasks in this comparison.
ALL_EVENT_TARGETS = EVENT_TARGETS
AUX_TARGET_COLS = ("sector_target", "industry_target", "year_target")
AUX_CLASS_DIMS: dict[str, int] = {"sector_target": 1, "industry_target": 1, "year_target": 1}


def add_auxiliary_labels(symbols: list[str], labels: pd.DataFrame) -> pd.DataFrame:
    """Add categorical sector, industry, and calendar-year targets."""
    warehouse = Warehouse()
    profiles: dict[str, tuple[str, str]] = {}
    for symbol in symbols:
        try:
            profile = warehouse.read_profile(str(symbol).upper(), provider="fmp")
        except Exception:
            profile = None
        sector = str(getattr(profile, "sector", "") or "").strip() if profile is not None else ""
        industry = str(getattr(profile, "industry", "") or "").strip() if profile is not None else ""
        profiles[str(symbol).upper()] = (sector, industry)
    sectors = sorted({value[0] for value in profiles.values() if value[0]})
    industries = sorted({value[1] for value in profiles.values() if value[1]})
    sector_codes = {value: i for i, value in enumerate(sectors)}
    industry_codes = {value: i for i, value in enumerate(industries)}
    out = labels.copy()
    symbols_upper = out.symbol.astype(str).str.upper()
    out["sector_target"] = symbols_upper.map(lambda value: sector_codes.get(profiles.get(value, ("", ""))[0], -1)).astype("int64")
    out["industry_target"] = symbols_upper.map(lambda value: industry_codes.get(profiles.get(value, ("", ""))[1], -1)).astype("int64")
    dates = pd.to_datetime(out.date, errors="coerce")
    out["year_target"] = dates.dt.year.fillna(-1).astype("int64")
    # Year labels are made contiguous so the classification head can use CE.
    years = sorted(value for value in out["year_target"].unique() if value >= 0)
    year_codes = {value: i for i, value in enumerate(years)}
    out["year_target"] = out["year_target"].map(lambda value: year_codes.get(int(value), -1)).astype("int64")
    AUX_CLASS_DIMS.update({
        "sector_target": max(1, len(sectors)), "industry_target": max(1, len(industries)),
        "year_target": max(1, len(years)),
    })
    return out


def add_event_labels(symbols: list[str], labels: pd.DataFrame) -> pd.DataFrame:
    """Add all requested event targets at the symbol-date level."""
    warehouse = Warehouse()
    rows: list[pd.DataFrame] = []
    families = tuple(sorted({spec[0] for spec in EVENT_TARGETS.values()}))
    for symbol in symbols:
        events = build_event_pairs_from_historical_data(
            symbol,
            fundamentals=warehouse.fundamentals,
            event_families=families,
            provider="fmp",
            start_date=str(DATA_START.date()),
            end_date=str(DATA_END.date()),
        )
        if events is None or events.empty:
            continue
        frame = events.copy()
        frame["date"] = pd.to_datetime(frame["event_date"], errors="coerce").dt.normalize()
        frame = frame.loc[frame.date.notna()]
        if frame.empty:
            continue
        chamber = frame["actor_chamber"].astype(str).str.lower()
        for target, (family, event_type, chamber_name) in EVENT_TARGETS.items():
            mask = frame["event_family"].eq(family) & frame["event_type"].eq(event_type)
            if chamber_name is not None:
                mask &= chamber.eq(chamber_name)
            frame[target] = mask.astype("float32")
        grouped = frame.groupby("date", as_index=False)[list(EVENT_TARGETS)].max()
        grouped.insert(0, "symbol", symbol.upper())
        rows.append(grouped)
    if not rows:
        for target in EVENT_TARGETS:
            labels[target] = 0.0
        return labels
    extra = pd.concat(rows, ignore_index=True)
    return labels.merge(extra, on=["symbol", "date"], how="left")


def add_graph_event_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Add symbol-level expanding top-decile graph events without lookahead."""
    out = labels.copy()
    for target, source in GRAPH_EVENT_TARGETS.items():
        values = pd.to_numeric(out[source], errors="coerce")
        prior_values = values.groupby(out.symbol, sort=False).shift(1)
        thresholds = prior_values.groupby(out.symbol, sort=False).transform(
            lambda series: series.expanding(min_periods=20).quantile(0.90)
        )
        out[target] = values.ge(thresholds).fillna(False).astype("float32")
    return out


def build_price_and_labels(symbols: list[str], tier: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = OUT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"{CACHE_VERSION}_{tier.lower()}_{DATA_START:%Y%m%d}_{DATA_END:%Y%m%d}_hold{MAX_HOLD}"
    prices_path = cache_dir / f"{cache_key}_prices.parquet"
    labels_path = cache_dir / f"{cache_key}_labels.parquet"
    if prices_path.exists() and labels_path.exists():
        prices = pd.read_parquet(prices_path)
        labels = pd.read_parquet(labels_path)
        required = set(EVENT_TARGETS) | {"long_hub", "long_authority", "short_hub", "short_authority", "long_pagerank", "short_pagerank"}
        if required.issubset(labels.columns) and set(SPEED_TARGET_COLS).issubset(labels.columns):
            if not set(AUX_TARGET_COLS).issubset(labels.columns):
                labels = add_auxiliary_labels(symbols, labels)
            print({"tier": tier, "cache": "hit", "prices": str(prices_path), "labels": str(labels_path)}, flush=True)
            for column in AUX_TARGET_COLS:
                AUX_CLASS_DIMS[column] = max(1, int(pd.to_numeric(labels[column], errors="coerce").max()) + 1)
            return prices, labels
    print({"tier": tier, "cache": "miss", "prices": str(prices_path), "labels": str(labels_path)}, flush=True)
    warehouse = Warehouse()
    node_rows: list[pd.DataFrame] = []
    price_frames: dict[str, pd.DataFrame] = {}
    for i, symbol in enumerate(symbols, 1):
        raw = warehouse.read_prices(symbol, provider="fmp", start=str(DATA_START.date()), end=str(DATA_END.date()))
        if raw is None or raw.empty:
            continue
        try:
            prices = normalize_prices(raw)
        except Exception:
            continue
        prices = prices.loc[prices.date.between(DATA_START, DATA_END)].copy()
        if len(prices) < 30:
            continue
        price_frames[symbol] = prices.copy()
        prices.insert(0, "symbol", symbol)
        node_rows.append(prices)
        if i % 100 == 0:
            print({"prices_loaded": i, "usable_symbols": len(node_rows)}, flush=True)
    prices = pd.concat(node_rows, ignore_index=True) if node_rows else pd.DataFrame()
    labels = build_hits_labels(
        price_frames,
        spec=HitsLabelSpec(
            max_hold=MAX_HOLD,
            iterations=HITS_ITERATIONS,
            tail_quantile=HITS_TAIL_QUANTILE,
            start_date=str(DATA_START.date()),
            end_date=str(DATA_END.date()),
        ),
    )
    speed_labels = build_inverse_holding_time_hits_labels(
        price_frames,
        spec=HitsLabelSpec(
            max_hold=MAX_HOLD,
            iterations=HITS_ITERATIONS,
            tail_quantile=HITS_TAIL_QUANTILE,
            start_date=str(DATA_START.date()),
            end_date=str(DATA_END.date()),
        ),
    )
    speed_labels = speed_labels.rename(columns={
        "long_hub": "speed_long_hub",
        "long_authority": "speed_long_authority",
        "short_hub": "speed_short_hub",
        "short_authority": "speed_short_authority",
    })
    speed_labels = add_speed_pagerank_labels(price_frames, speed_labels)
    labels = labels.merge(speed_labels, on=["symbol", "date"], how="left")
    labels = add_pagerank_labels(price_frames, labels)
    labels = add_event_labels(symbols, labels)
    labels = add_auxiliary_labels(symbols, labels)
    prices.to_parquet(prices_path, index=False)
    labels.to_parquet(labels_path, index=False)
    return prices, labels


def make_temporal_edges(nodes: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Past-to-current edges using the same long/short HITS return formulas."""
    src: list[int] = []
    dst: list[int] = []
    edge_ret: list[float] = []
    for _, group in nodes.groupby("symbol", sort=False):
        ids = group.index.to_numpy()
        high = group.high.to_numpy(float)
        low = group.low.to_numpy(float)
        for j in range(len(ids)):
            # Keep message passing local and causal.  The HITS/PageRank label
            # graph still uses MAX_HOLD; the GNN sees only recent history.
            lo = max(0, j - LOOKBACK)
            for k in range(lo, j):
                if np.isfinite(high[k] + low[k] + high[j] + low[j]) and min(high[k], low[k], high[j], low[j]) > 0:
                    src.append(int(ids[k])); dst.append(int(ids[j]))
                    edge_ret.append((
                        float(np.clip(low[j] / high[k] - 1.0, 0.0, 2.0)),
                        float(np.clip(low[k] / high[j] - 1.0, 0.0, 2.0)),
                    ))
    return torch.tensor([src, dst], dtype=torch.long), torch.tensor(edge_ret, dtype=torch.float32)


def make_pair_batch(nodes: pd.DataFrame, max_pairs: int = 250_000) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample historical same-symbol entry/exit edges for auxiliary learning."""
    rng = np.random.default_rng(SEED)
    src, dst, long_y, short_y = [], [], [], []
    for _, group in nodes.groupby("symbol", sort=False):
        ids = group.index.to_numpy()
        high = group.high.to_numpy(float)
        low = group.low.to_numpy(float)
        n = len(ids)
        for i in range(n):
            hi = min(n, i + MAX_HOLD + 1)
            candidates = np.arange(i + 1, hi)
            if len(candidates) > PAIR_PER_SOURCE:
                candidates = rng.choice(candidates, PAIR_PER_SOURCE, replace=False)
            for j in candidates:
                if min(high[i], low[i], high[j], low[j]) <= 0 or not np.isfinite(high[i] + low[i] + high[j] + low[j]):
                    continue
                src.append(int(ids[i])); dst.append(int(ids[j]))
                long_y.append(float(np.clip(low[j] / high[i] - 1.0, 0.0, 2.0)))
                short_y.append(float(np.clip(low[i] / high[j] - 1.0, 0.0, 2.0)))
    if len(src) > max_pairs:
        keep = rng.choice(len(src), max_pairs, replace=False)
        src = [src[i] for i in keep]; dst = [dst[i] for i in keep]; long_y = [long_y[i] for i in keep]; short_y = [short_y[i] for i in keep]
    return torch.tensor(src), torch.tensor(dst), torch.tensor(np.column_stack([long_y, short_y]), dtype=torch.float32)


class TemporalGNN(nn.Module):
    def __init__(self, n_features: int, hidden: int = HIDDEN):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(n_features, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.message = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.update = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.edge_head = nn.Sequential(nn.Linear(hidden * 2 + 1, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.node_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 6), nn.Sigmoid())
        self.event_head = EventPrototypeHead(hidden, len(ALL_EVENT_TARGETS))

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        if edge_index.numel():
            messages = self.message(torch.cat([h[edge_index[0]], edge_attr], dim=1))
            agg = torch.zeros_like(h)
            count = torch.zeros((len(h), 1), device=h.device)
            agg.index_add_(0, edge_index[1], messages)
            count.index_add_(0, edge_index[1], torch.ones((len(messages), 1), device=h.device))
            h = self.update(torch.cat([h, agg / count.clamp_min(1.0)], dim=1))
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, pair_src: torch.Tensor | None = None, pair_dst: torch.Tensor | None = None):
        z = self.encode(x, edge_index, edge_attr)
        node_targets = self.node_head(z)
        event_logits = self.event_head(z)
        pair = None
        if pair_src is not None and pair_dst is not None:
            gap = (pair_dst - pair_src).float().unsqueeze(1) / max(1.0, MAX_HOLD)
            pair = self.edge_head(torch.cat([z[pair_src], z[pair_dst], gap], dim=1))
        return z, node_targets, event_logits, pair


class EventPrototypeHead(nn.Module):
    """Shared metric-learning head for independent binary event labels.

    Each event owns a positive prototype in the same learned embedding space.
    Events remain independent binary tasks: there is no softmax across events,
    and one node may activate multiple event labels at the same time.
    """

    def __init__(self, hidden: int, n_events: int, metric_dim: int | None = None):
        super().__init__()
        metric_dim = metric_dim or int(os.getenv("GNN_EVENT_METRIC_DIM", "24"))
        self.embedding = nn.Sequential(nn.Linear(hidden, metric_dim), nn.LayerNorm(metric_dim))
        self.prototypes = nn.Parameter(torch.randn(n_events, metric_dim) * 0.02)
        self.log_temperature = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(n_events))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        embedding = nn.functional.normalize(self.embedding(h), dim=-1)
        prototypes = nn.functional.normalize(self.prototypes, dim=-1)
        temperature = self.log_temperature.exp().clamp(1.0, 50.0)
        return temperature * embedding @ prototypes.T + self.bias


def event_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Class-balanced independent BCE for sparse event labels."""
    positives = targets.sum(dim=0)
    negative_count = targets.shape[0] - positives
    pos_weight = (negative_count / positives.clamp_min(1.0)).clamp_min(1.0).clamp_max(100.0)
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def combine_target_predictions(node_targets: torch.Tensor, event_logits: torch.Tensor) -> torch.Tensor:
    """Return the historical target-column layout for downstream ranking."""
    return torch.cat([node_targets, torch.sigmoid(event_logits)], dim=-1)


class CategoricalPrototypeHead(nn.Module):
    """Cosine-prototype classifier for dense categorical auxiliary tasks."""

    def __init__(self, hidden: int, n_classes: int, metric_dim: int | None = None):
        super().__init__()
        metric_dim = metric_dim or int(os.getenv("GNN_METRIC_DIM", "24"))
        self.embedding = nn.Sequential(nn.Linear(hidden, metric_dim), nn.LayerNorm(metric_dim))
        self.prototypes = nn.Parameter(torch.randn(n_classes, metric_dim) * 0.02)
        self.log_temperature = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        embedding = nn.functional.normalize(self.embedding(h), dim=-1)
        prototypes = nn.functional.normalize(self.prototypes, dim=-1)
        temperature = self.log_temperature.exp().clamp(1.0, 50.0)
        return temperature * embedding @ prototypes.T


def family_key(name: str) -> str:
    return "f_" + "".join(ch if ch.isalnum() else "_" for ch in str(name))


class SharedFamilyGNN(nn.Module):
    """One shared GNN with a family-specific input projection per family."""

    def __init__(self, feature_dims: dict[str, int], hidden: int = HIDDEN, aux_dims: dict[str, int] | None = None):
        super().__init__()
        self.inputs = nn.ModuleDict({family_key(name): nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.ReLU()) for name, dim in feature_dims.items()})
        self.message = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.update = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.edge_head = nn.Sequential(nn.Linear(hidden * 2 + 1, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.node_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 6), nn.Sigmoid())
        self.event_head = EventPrototypeHead(hidden, len(ALL_EVENT_TARGETS))

    def forward_batch(self, xs: list[torch.Tensor], families: list[str], edge_index: torch.Tensor, edge_attr: torch.Tensor, pair_src: torch.Tensor, pair_dst: torch.Tensor):
        h = torch.stack([self.inputs[family_key(name)](x) for x, name in zip(xs, families)])
        if edge_index.numel():
            source = h[:, edge_index[0], :]
            attrs = edge_attr.unsqueeze(0).expand(len(xs), -1, -1)
            messages = self.message(torch.cat([source, attrs], dim=2))
            agg = torch.zeros_like(h)
            destination = edge_index[1].view(1, -1, 1).expand(len(xs), -1, h.shape[2])
            agg.scatter_add_(1, destination, messages)
            count = torch.zeros((len(xs), h.shape[1], 1), device=h.device)
            count.scatter_add_(1, edge_index[1].view(1, -1, 1).expand(len(xs), -1, 1), torch.ones((len(xs), len(edge_index[1]), 1), device=h.device))
            h = self.update(torch.cat([h, agg / count.clamp_min(1.0)], dim=2))
        node_targets = self.node_head(h)
        event_logits = self.event_head(h)
        gap = (pair_dst - pair_src).float().unsqueeze(1) / max(1.0, MAX_HOLD)
        pair = self.edge_head(torch.cat([h[:, pair_src, :], h[:, pair_dst, :], gap.unsqueeze(0).expand(len(xs), -1, -1)], dim=2))
        return node_targets, event_logits, pair


class FusedFamilyGNN(nn.Module):
    """Fuse every feature family into one node state on fixed temporal edges."""

    def __init__(self, feature_dims: dict[str, int], hidden: int = HIDDEN, aux_dims: dict[str, int] | None = None):
        super().__init__()
        self.family_names = tuple(feature_dims)
        self.inputs = nn.ModuleDict({family_key(name): nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.ReLU()) for name, dim in feature_dims.items()})
        self.fusion = nn.Sequential(nn.Linear(hidden * len(self.family_names), hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.message = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.update = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.edge_head = nn.Sequential(nn.Linear(hidden * 2 + 1, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.node_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 6), nn.Sigmoid())
        self.event_head = EventPrototypeHead(hidden, len(ALL_EVENT_TARGETS))
        self.aux_heads = nn.ModuleDict({
            name: CategoricalPrototypeHead(hidden, int(size))
            for name, size in (aux_dims or {}).items()
            if name in AUX_TARGET_COLS
        })

    def forward(self, xs: list[torch.Tensor], edge_index: torch.Tensor, edge_attr: torch.Tensor, pair_src: torch.Tensor, pair_dst: torch.Tensor):
        h = self.fusion(torch.cat([self.inputs[family_key(name)](x) for x, name in zip(xs, self.family_names)], dim=1))
        if edge_index.numel():
            messages = self.message(torch.cat([h[edge_index[0]], edge_attr], dim=1))
            agg = torch.zeros_like(h); count = torch.zeros((len(h), 1), device=h.device)
            agg.index_add_(0, edge_index[1], messages)
            count.index_add_(0, edge_index[1], torch.ones((len(messages), 1), device=h.device))
            h = self.update(torch.cat([h, agg / count.clamp_min(1.0)], dim=1))
        node_targets = self.node_head(h)
        event_logits = self.event_head(h)
        gap = (pair_dst - pair_src).float().unsqueeze(1) / max(1.0, MAX_HOLD)
        pair = self.edge_head(torch.cat([h[pair_src], h[pair_dst], gap], dim=1))
        aux_logits = {name: head(h) for name, head in self.aux_heads.items()}
        return node_targets, event_logits, pair, aux_logits


class FusedFamilyMoEGNN(nn.Module):
    """Fused GNN with sqrt(families) shared experts on fixed temporal edges.

    The graph aggregation is performed once.  MoE routing is used for the
    family projections and the node update, so this is not one graph trunk
    per family or per expert.  Residual updates preserve the pre-message
    node state while temporal edges remain fixed and unweighted.
    """

    def __init__(self, feature_dims: dict[str, int], hidden: int = HIDDEN, aux_dims: dict[str, int] | None = None):
        super().__init__()
        self.family_names = tuple(feature_dims)
        self.n_experts = MOE_EXPERTS or max(1, int(np.ceil(np.sqrt(len(self.family_names)))))
        self.top_k = min(MOE_TOP_K, self.n_experts)
        self.aux_dims = dict(aux_dims or {})
        self.inputs = nn.ModuleDict({
            family_key(name): nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
            for name, dim in feature_dims.items()
        })
        self.family_router = nn.Linear(hidden, self.n_experts)
        self.family_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            for _ in range(self.n_experts)
        ])
        self.family_gate = nn.Linear(hidden, 1)
        self.message = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.update_router = nn.Linear(hidden * 2, self.n_experts)
        self.update_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            for _ in range(self.n_experts)
        ])
        self.update_norm = nn.LayerNorm(hidden)
        self.edge_head = nn.Sequential(nn.Linear(hidden * 2 + 1, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.node_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 6), nn.Sigmoid())
        self.event_head = EventPrototypeHead(hidden, len(ALL_EVENT_TARGETS))
        metric_dim = int(os.getenv("GNN_METRIC_DIM", "24"))
        self.sector_metric = nn.Sequential(nn.Linear(hidden, metric_dim), nn.LayerNorm(metric_dim)) if "sector_target" in self.aux_dims else None
        self.industry_metric = nn.Sequential(nn.Linear(hidden, metric_dim), nn.LayerNorm(metric_dim)) if "industry_target" in self.aux_dims else None
        self.year_metric = nn.Sequential(nn.Linear(hidden, metric_dim), nn.LayerNorm(metric_dim)) if "year_target" in self.aux_dims else None
        self.aux_prototypes = nn.ParameterDict({
            name: nn.Parameter(torch.randn(int(size), metric_dim) * 0.02)
            for name, size in self.aux_dims.items() if name in set(AUX_TARGET_COLS)
        })
        self.prototype_temperature = nn.Parameter(torch.tensor(10.0))

    def _sparse_experts(self, values: torch.Tensor, router: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply only top-k experts to flattened leading dimensions."""
        shape = values.shape
        flat = values.reshape(-1, shape[-1])
        probs = torch.softmax(router(flat), dim=-1)
        top_values, top_indices = torch.topk(probs, k=self.top_k, dim=-1)
        top_values = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        mixed = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.family_experts if router is self.family_router else self.update_experts):
            rows, slots = torch.where(top_indices == expert_id)
            if len(rows):
                mixed.index_add_(0, rows, expert(flat[rows]) * top_values[rows, slots].unsqueeze(1))
        return mixed.reshape(*shape), probs

    @staticmethod
    def _balance_loss(probs: torch.Tensor) -> torch.Tensor:
        mean_prob = probs.mean(dim=0)
        target = torch.full_like(mean_prob, 1.0 / mean_prob.numel())
        return (mean_prob - target).pow(2).sum()

    def forward(self, xs: list[torch.Tensor], edge_index: torch.Tensor, edge_attr: torch.Tensor, pair_src: torch.Tensor, pair_dst: torch.Tensor):
        family_h = torch.stack([self.inputs[family_key(name)](x) for x, name in zip(xs, self.family_names)], dim=1)
        family_z, family_probs = self._sparse_experts(family_h, self.family_router)
        family_mix = torch.softmax(self.family_gate(family_h).squeeze(-1), dim=1)
        h = (family_mix.unsqueeze(-1) * family_z).sum(dim=1)
        balance_loss = self._balance_loss(family_probs)
        if edge_index.numel():
            source = h[edge_index[0]]
            messages = self.message(torch.cat([source, edge_attr], dim=1))
            agg = torch.zeros_like(h)
            count = torch.zeros((len(h), 1), device=h.device)
            agg.index_add_(0, edge_index[1], messages)
            count.index_add_(0, edge_index[1], torch.ones((len(messages), 1), device=h.device))
            aggregate = agg / count.clamp_min(1.0)
            update_input = torch.cat([h, aggregate], dim=1)
            update_router_probs = torch.softmax(self.update_router(update_input), dim=-1)
            top_values, top_indices = torch.topk(update_router_probs, k=self.top_k, dim=-1)
            top_values = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            update = torch.zeros_like(h)
            for expert_id, expert in enumerate(self.update_experts):
                rows, slots = torch.where(top_indices == expert_id)
                if len(rows):
                    update.index_add_(0, rows, expert(update_input[rows]) * top_values[rows, slots].unsqueeze(1))
            h = self.update_norm(h + update)
            balance_loss = balance_loss + self._balance_loss(update_router_probs)
        node_targets = self.node_head(h)
        event_logits = self.event_head(h)
        gap = (pair_dst - pair_src).float().unsqueeze(1) / max(1.0, MAX_HOLD)
        pair = self.edge_head(torch.cat([h[pair_src], h[pair_dst], gap], dim=1))
        aux_logits = {}
        temperature = self.prototype_temperature.clamp(1.0, 50.0)
        if self.sector_metric is not None:
            sector_embedding = nn.functional.normalize(self.sector_metric(h), dim=1)
            sector_prototypes = nn.functional.normalize(self.aux_prototypes["sector_target"], dim=1)
            aux_logits["sector_target"] = temperature * sector_embedding @ sector_prototypes.T
        if self.industry_metric is not None:
            industry_embedding = nn.functional.normalize(self.industry_metric(h), dim=1)
            industry_prototypes = nn.functional.normalize(self.aux_prototypes["industry_target"], dim=1)
            aux_logits["industry_target"] = temperature * industry_embedding @ industry_prototypes.T
        if self.year_metric is not None:
            year_embedding = nn.functional.normalize(self.year_metric(h), dim=1)
            year_prototypes = nn.functional.normalize(self.aux_prototypes["year_target"], dim=1)
            aux_logits["year_target"] = temperature * year_embedding @ year_prototypes.T
        return node_targets, event_logits, pair, balance_loss, aux_logits


def prepare_family(panel: pd.DataFrame, price_map: pd.DataFrame, labels: pd.DataFrame, family: str, test_year: int) -> dict:
    metadata = pd.read_parquet(panel.attrs["metadata_path"])
    feature_cols = [c for c in metadata.feature.astype(str) if c in panel.columns]
    if not feature_cols:
        return {"family": family, "status": "no_features"}
    base = panel[["symbol", "date", *feature_cols]].copy()
    base["symbol"] = base.symbol.astype(str).str.upper()
    base["date"] = pd.to_datetime(base.date).dt.normalize()
    base = base.merge(price_map, on=["symbol", "date"], how="inner").sort_values(["symbol", "date"]).reset_index(drop=True)
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")
    base = base.loc[base.date.between(DATA_START, test_end)].reset_index(drop=True)
    if base.empty:
        return {"family": family, "status": "no_price_overlap"}
    y = labels.copy(); y.symbol = y.symbol.astype(str).str.upper(); y.date = pd.to_datetime(y.date).dt.normalize()
    base = base.merge(y, on=["symbol", "date"], how="left")
    target_cols = [
        "long_hub", "long_authority", "short_hub", "short_authority",
        "long_pagerank", "short_pagerank",
    ]
    event_cols = list(ALL_EVENT_TARGETS)
    all_target_cols = target_cols + event_cols
    base[list(AUX_TARGET_COLS)] = base[list(AUX_TARGET_COLS)].apply(pd.to_numeric, errors="coerce").fillna(-1).astype("int64")
    tail_cols = [f"{column}_tail" for column in target_cols]
    base[all_target_cols] = base[all_target_cols].fillna(0.0).astype("float32")
    for column in ("long_pagerank", "short_pagerank"):
        ranks = base.groupby(base.date.dt.year)[column].rank(method="first", pct=True)
        base[f"{column}_tail"] = ranks.le(HITS_TAIL_QUANTILE) | ranks.ge(1.0 - HITS_TAIL_QUANTILE)
    base[tail_cols] = base[tail_cols].fillna(False).astype(bool)
    train_mask = base.date < test_start
    med = base.loc[train_mask, feature_cols].apply(pd.to_numeric, errors="coerce").median().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    xdf = base[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    std = xdf.loc[train_mask].std().replace(0, 1).fillna(1.0)
    xdf = ((xdf - xdf.loc[train_mask].mean()) / std).clip(-8, 8).astype("float32")
    base[feature_cols] = xdf
    # Stable row order makes node IDs usable for temporal and pair edges.
    node_frame = base[["symbol", "date", "high", "low"]].reset_index(drop=True)
    node_hash = int(pd.util.hash_pandas_object(node_frame, index=False).sum())
    graph_key = (test_year, len(node_frame), node_hash)
    cached_graph = GRAPH_CACHE.get(graph_key)
    if cached_graph is None:
        temporal_edges, temporal_attr = make_temporal_edges(node_frame)
        train_nodes = base.index[train_mask].to_numpy()
        pair_src, pair_dst, pair_y = make_pair_batch(node_frame.loc[train_mask.to_numpy()].reset_index(drop=True))
        # Pair sampler returns local train IDs; translate to global node IDs.
        pair_global = train_nodes
        pair_src, pair_dst = pair_global[pair_src.numpy()], pair_global[pair_dst.numpy()]
        pair_src, pair_dst = torch.tensor(pair_src), torch.tensor(pair_dst)
        GRAPH_CACHE[graph_key] = (temporal_edges, temporal_attr, pair_src, pair_dst, train_nodes)
        GRAPH_CACHE[(graph_key[0], graph_key[1], graph_key[2], "pair_y")] = pair_y
    else:
        temporal_edges, temporal_attr, pair_src, pair_dst, train_nodes = cached_graph
        pair_y = GRAPH_CACHE[(graph_key[0], graph_key[1], graph_key[2], "pair_y")]
    x = torch.tensor(base[feature_cols].to_numpy(dtype=np.float32))
    ha_y = torch.tensor(base[all_target_cols].to_numpy(dtype=np.float32))
    ha_mask = torch.tensor(base[tail_cols].to_numpy(dtype=np.float32))
    aux_y = torch.tensor(base[list(AUX_TARGET_COLS)].to_numpy(dtype=np.int64))
    aux_mask = aux_y.ge(0)
    return {"family": family, "features": feature_cols, "base": base, "x": x, "ha_y": ha_y, "ha_mask": ha_mask,
            "train_nodes": train_nodes, "temporal_edges": temporal_edges, "temporal_attr": temporal_attr,
            "pair_src": pair_src, "pair_dst": pair_dst, "pair_y": pair_y, "test_start": test_start, "test_end": test_end,
            "target_cols": all_target_cols, "aux_y": aux_y, "aux_mask": aux_mask}


def fit_family(panel: pd.DataFrame, price_map: pd.DataFrame, labels: pd.DataFrame, family: str, test_year: int) -> tuple[pd.DataFrame, dict]:
    prepared = prepare_family(panel, price_map, labels, family, test_year)
    if prepared.get("status"):
        return pd.DataFrame(), prepared
    feature_cols = prepared["features"]
    base, x = prepared["base"], prepared["x"]
    train_nodes, temporal_edges, temporal_attr = prepared["train_nodes"], prepared["temporal_edges"], prepared["temporal_attr"]
    pair_src, pair_dst, pair_y = prepared["pair_src"], prepared["pair_dst"], prepared["pair_y"]
    target_cols, ha_y, ha_mask = prepared["target_cols"], prepared["ha_y"], prepared["ha_mask"]
    model = TemporalGNN(len(feature_cols)).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(os.getenv("GNN_LR", "0.002")), weight_decay=1e-4)
    edge_train_mask = torch.ones(len(pair_src), dtype=torch.bool)
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        _, node_hat, event_logits, edge_hat = model(x, temporal_edges, temporal_attr, pair_src, pair_dst)
        node_errors = nn.functional.smooth_l1_loss(node_hat[train_nodes], ha_y[train_nodes, :6], reduction="none")
        node_mask = ha_mask[train_nodes]
        node_loss = (node_errors * node_mask).sum() / node_mask.sum().clamp_min(1.0)
        event_true = ha_y[train_nodes, 6:]
        event_loss = event_loss_from_logits(event_logits[train_nodes], event_true)
        edge_loss = nn.functional.smooth_l1_loss(edge_hat[edge_train_mask], pair_y[edge_train_mask]) if edge_hat is not None and len(pair_src) else torch.tensor(0.0)
        loss = node_loss + float(os.getenv("GNN_EVENT_LOSS_WEIGHT", "1.0")) * event_loss + float(os.getenv("GNN_EDGE_LOSS_WEIGHT", "0.25")) * edge_loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if epoch == 0 or epoch == EPOCHS - 1:
            print({"family": family, "epoch": epoch + 1, "loss": round(float(loss.detach()), 5), "node_loss": round(float(node_loss.detach()), 5), "event_loss": round(float(event_loss.detach()), 5), "edge_loss": round(float(edge_loss.detach()), 5)}, flush=True)
    model.eval()
    with torch.no_grad():
        _, node_hat, event_logits, _ = model(x, temporal_edges, temporal_attr)
    pred = base[["symbol", "date"]].copy()
    pred[all_target_cols] = combine_target_predictions(node_hat, event_logits).numpy()
    pred = pred.loc[pred.date.between(test_start, test_end)].copy()
    # HITS is a relative score.  Calibrate predictions cross-sectionally so
    # the existing shared-book threshold has the same interpretation as the
    # prior feature-family experiments.
    for col in all_target_cols:
        pred[col] = pred.groupby("date")[col].rank(pct=True, method="average")
    pred["source"] = "gnn"
    pred["family"] = family
    pred["strategy_source"] = f"gnn.{family}"
    pred["long_score"] = pred["long_hub"]
    pred["long_exit_score"] = pred["long_authority"]
    pred["short_score"] = pred["short_hub"]
    pred["short_exit_score"] = pred["short_authority"]
    pred["long_agree_count"] = (pred.long_score >= pred.short_score).astype(int)
    pred["short_agree_count"] = (pred.short_score > pred.long_score).astype(int)
    pred["model_count"] = 1
    return pred, {"family": family, "status": "ok", "year": test_year, "nodes": len(base), "train_nodes": int(train_mask.sum()), "pairs": len(pair_src), "features": len(feature_cols), "epochs": EPOCHS}


def run_tier(tier: str) -> pd.DataFrame:
    if GNN_VARIANT not in {"long_only", "short_only"}:
        raise ValueError("GNN_VARIANT must be 'long_only' or 'short_only'")
    started = perf_counter()
    index = pd.read_csv(feature_dir(tier) / "index.csv")
    requested = tuple(x.strip() for x in os.getenv("GNN_FAMILIES", "").split(",") if x.strip())
    if requested:
        index = index.loc[index.family.astype(str).isin(requested)].copy()
    first = pd.read_parquet(index.iloc[0].panel_path)
    symbols = sorted(first.symbol.astype(str).str.upper().unique())
    price_map, labels = build_price_and_labels(symbols, tier)
    print({"tier": tier, "symbols": len(symbols), "price_rows": len(price_map), "label_rows": len(labels), "families": len(index)}, flush=True)
    close = price_map.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    next_returns = close.pct_change().shift(-1)
    effective_top_k = min(20, len(close.columns))
    print({"tier": tier, "symbols": len(close.columns), "requested_top_k": 20, "effective_top_k": effective_top_k, "weight": 1.0 / effective_top_k}, flush=True)
    summaries, predictions = [], []
    panels = {}
    for _, meta in index.iterrows():
        panel = pd.read_parquet(meta.panel_path)
        panel.attrs["metadata_path"] = meta.metadata_path
        panels[str(meta.family)] = panel
    for test_year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31")
        date_mask = (next_returns.index >= test_start) & (next_returns.index <= test_end)
        dates = pd.DatetimeIndex(next_returns.index[date_mask])
        for _, meta in index.iterrows():
            family = str(meta.family)
            panel = panels[family]
            pred, info = fit_family(panel, price_map, labels, family, test_year)
            print(info, flush=True)
            if pred.empty:
                continue
            predictions.append(pred)
            summary, _, _ = run_shared_book_framework_comparison(
                scores=pred[["symbol", "date", "long_score", "short_score", "long_exit_score", "short_exit_score", "long_agree_count", "short_agree_count", "model_count"]],
                next_returns=next_returns,
                symbols=tuple(close.columns),
                dates=dates,
                variants=(GNN_VARIANT,),
                top_k_values=(effective_top_k,),
                entry_threshold=0.5,
                exit_threshold=0.5,
                cost_models={"family_common": SharedBookCostModel(0.5, 5.0)},
            )
            if not summary.empty:
                summary["tier"] = tier; summary["year"] = test_year; summary["family"] = family; summary["label_source"] = "gnn_sparse_hits"; summaries.append(summary)
        if summaries:
            pd.concat(summaries, ignore_index=True).to_parquet(OUT / f"{tier.lower()}_{GNN_VARIANT}_through_{test_year}.parquet", index=False)
    result = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    result.to_csv(OUT / f"{tier.lower()}_{GNN_VARIANT}_wfo_results.csv", index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_parquet(OUT / f"{tier.lower()}_{GNN_VARIANT}_wfo_predictions.parquet", index=False)
    print(result.groupby(["year", "variant", "label_source"], as_index=False).agg(families=("family", "nunique"), mean_return=("total_return", "mean"), median_return=("total_return", "median"), mean_sharpe=("sharpe", "mean")).round(4) if not result.empty else result)
    print({"tier": tier, "seconds": round(perf_counter() - started, 1), "result_rows": len(result)}, flush=True)
    return result


def run_tier_shared(tier: str) -> pd.DataFrame:
    """Run one shared encoder per WFO fold with family-specific input layers."""
    if GNN_VARIANT not in {"long_only", "short_only"}:
        raise ValueError("GNN_VARIANT must be 'long_only' or 'short_only'")
    started = perf_counter()
    index = pd.read_csv(feature_dir(tier) / "index.csv")
    requested = tuple(x.strip() for x in os.getenv("GNN_FAMILIES", "").split(",") if x.strip())
    if requested:
        index = index.loc[index.family.astype(str).isin(requested)].copy()
    panels = {}
    for _, meta in index.iterrows():
        panel = pd.read_parquet(meta.panel_path)
        panel.attrs["metadata_path"] = meta.metadata_path
        panels[str(meta.family)] = panel
    first = next(iter(panels.values()))
    symbols = sorted(first.symbol.astype(str).str.upper().unique())
    price_map, labels = build_price_and_labels(symbols, tier)
    close = price_map.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    next_returns = close.pct_change().shift(-1)
    effective_top_k = min(20, len(close.columns))
    summaries = []
    chunk_size = int(os.getenv("GNN_SHARED_CHUNK", "4"))
    for test_year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):
        prepared = []
        for family, panel in panels.items():
            item = prepare_family(panel, price_map, labels, family, test_year)
            if not item.get("status"):
                prepared.append(item)
        if not prepared:
            continue
        feature_dims = {item["family"]: len(item["features"]) for item in prepared}
        model = SharedFamilyGNN(feature_dims).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(os.getenv("GNN_LR", "0.002")), weight_decay=1e-4)
        graph = prepared[0]
        print({"tier": tier, "year": test_year, "shared_families": len(prepared), "nodes": len(graph["base"]), "train_nodes": len(graph["train_nodes"]), "chunk_size": chunk_size}, flush=True)
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            for start in range(0, len(prepared), chunk_size):
                batch = prepared[start:start + chunk_size]
                optimizer.zero_grad()
                node_pred, event_logits, pair_pred = model.forward_batch(
                    [item["x"] for item in batch], [item["family"] for item in batch],
                    graph["temporal_edges"], graph["temporal_attr"], graph["pair_src"], graph["pair_dst"])
                losses = []
                for i, item in enumerate(batch):
                    train_nodes = item["train_nodes"]
                    target_cols = item["target_cols"]
                    graph_count = 6
                    node_errors = nn.functional.smooth_l1_loss(node_pred[i, train_nodes], item["ha_y"][train_nodes, :graph_count], reduction="none")
                    node_loss = (node_errors * item["ha_mask"][train_nodes]).sum() / item["ha_mask"][train_nodes].sum().clamp_min(1.0)
                    event_true = item["ha_y"][train_nodes, graph_count:]
                    event_loss = event_loss_from_logits(event_logits[i, train_nodes], event_true)
                    edge_loss = nn.functional.smooth_l1_loss(pair_pred[i], item["pair_y"])
                    losses.append(node_loss + float(os.getenv("GNN_EVENT_LOSS_WEIGHT", "1.0")) * event_loss + float(os.getenv("GNN_EDGE_LOSS_WEIGHT", "0.25")) * edge_loss)
                loss = torch.stack(losses).mean()
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                epoch_loss += float(loss.detach())
            if epoch == 0 or epoch == EPOCHS - 1:
                print({"tier": tier, "year": test_year, "epoch": epoch + 1, "shared_loss": round(epoch_loss, 5)}, flush=True)
        model.eval()
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31")
        dates = pd.DatetimeIndex(next_returns.index[(next_returns.index >= test_start) & (next_returns.index <= test_end)])
        with torch.no_grad():
            prediction_batches = []
            for start in range(0, len(prepared), chunk_size):
                batch = prepared[start:start + chunk_size]
                node_pred, event_logits, _ = model.forward_batch([item["x"] for item in batch], [item["family"] for item in batch], graph["temporal_edges"], graph["temporal_attr"], graph["pair_src"], graph["pair_dst"])
                prediction_batches.append(combine_target_predictions(node_pred, event_logits).cpu().numpy())
        for batch_start, values in zip(range(0, len(prepared), chunk_size), prediction_batches):
            for offset, item in enumerate(prepared[batch_start:batch_start + chunk_size]):
                pred = item["base"][["symbol", "date"]].copy()
                pred[item["target_cols"]] = values[offset]
                pred = pred.loc[pred.date.between(test_start, test_end)].copy()
                for col in item["target_cols"]:
                    pred[col] = pred.groupby("date")[col].rank(pct=True, method="average")
                pred["long_score"], pred["long_exit_score"] = pred["long_hub"], pred["long_authority"]
                pred["short_score"], pred["short_exit_score"] = pred["short_hub"], pred["short_authority"]
                pred["long_agree_count"] = (pred.long_score >= pred.short_score).astype(int); pred["short_agree_count"] = (pred.short_score > pred.long_score).astype(int); pred["model_count"] = 1
                summary, _, _ = run_shared_book_framework_comparison(
                    scores=pred[["symbol", "date", "long_score", "short_score", "long_exit_score", "short_exit_score", "long_agree_count", "short_agree_count", "model_count"]], next_returns=next_returns, symbols=tuple(close.columns), dates=dates, variants=(GNN_VARIANT,), top_k_values=(effective_top_k,), entry_threshold=0.5, exit_threshold=0.5, cost_models={"family_common": SharedBookCostModel(0.5, 5.0)})
                if not summary.empty:
                    summary["tier"] = tier; summary["year"] = test_year; summary["family"] = item["family"]; summary["label_source"] = "gnn_shared_sparse_hits"; summaries.append(summary)
        pd.concat(summaries, ignore_index=True).to_parquet(OUT / f"{tier.lower()}_shared_long_only_through_{test_year}.parquet", index=False)
    result = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    result.to_csv(OUT / f"{tier.lower()}_shared_wfo_results.csv", index=False)
    print(result.groupby(["tier", "year", "variant"], as_index=False).agg(families=("family", "nunique"), mean_return=("total_return", "mean"), mean_sharpe=("sharpe", "mean")).round(4).to_string(index=False) if not result.empty else result)
    print({"tier": tier, "shared_seconds": round(perf_counter() - started, 1), "result_rows": len(result)}, flush=True)
    return result


def run_tier_fused(tier: str) -> pd.DataFrame:
    """Run one fused multi-family GNN and one portfolio per WFO fold."""
    if GNN_VARIANT not in {"long_only", "short_only"}:
        raise ValueError("GNN_VARIANT must be 'long_only' or 'short_only'")
    use_moe = os.getenv("GNN_MOE_FAMILY", "0") == "1"
    run_label = "fused_moe" if use_moe else "fused"
    started = perf_counter()
    index = pd.read_csv(feature_dir(tier) / "index.csv")
    requested = tuple(x.strip() for x in os.getenv("GNN_FAMILIES", "").split(",") if x.strip())
    if requested:
        index = index.loc[index.family.astype(str).isin(requested)].copy()
    panels = {}
    for _, meta in index.iterrows():
        panel = pd.read_parquet(meta.panel_path); panel.attrs["metadata_path"] = meta.metadata_path; panels[str(meta.family)] = panel
    first = next(iter(panels.values())); symbols = sorted(first.symbol.astype(str).str.upper().unique())
    price_map, labels = build_price_and_labels(symbols, tier)
    close = price_map.pivot(index="date", columns="symbol", values="close").sort_index().ffill(); next_returns = close.pct_change().shift(-1)
    effective_top_k = min(20, len(close.columns)); summaries = []; predictions = []
    for test_year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):
        prepared = [prepare_family(panel, price_map, labels, family, test_year) for family, panel in panels.items()]
        prepared = [item for item in prepared if not item.get("status")]
        if not prepared: continue
        node_hashes = {int(pd.util.hash_pandas_object(item["base"][["symbol", "date"]], index=False).sum()) for item in prepared}
        if len(node_hashes) != 1: raise RuntimeError("Feature families do not share the same symbol/date node index")
        graph = prepared[0]; feature_dims = {item["family"]: len(item["features"]) for item in prepared}
        model = (FusedFamilyMoEGNN(feature_dims, aux_dims=AUX_CLASS_DIMS) if use_moe else FusedFamilyGNN(feature_dims, aux_dims=AUX_CLASS_DIMS)).to(GNN_DEVICE).train(); optimizer = torch.optim.AdamW(model.parameters(), lr=float(os.getenv("GNN_LR", "0.002")), weight_decay=1e-4)
        xs = [item["x"].to(GNN_DEVICE) for item in prepared]
        temporal_edges = graph["temporal_edges"].to(GNN_DEVICE)
        temporal_attr = graph["temporal_attr"].to(GNN_DEVICE)
        pair_src = graph["pair_src"].to(GNN_DEVICE)
        pair_dst = graph["pair_dst"].to(GNN_DEVICE)
        pair_y = graph["pair_y"].to(GNN_DEVICE)
        train_nodes = torch.as_tensor(graph["train_nodes"], dtype=torch.long, device=GNN_DEVICE)
        graph_y = graph["ha_y"].to(GNN_DEVICE)
        graph_mask = graph["ha_mask"].to(GNN_DEVICE)
        aux_y = graph["aux_y"].to(GNN_DEVICE)
        aux_mask = graph["aux_mask"].to(GNN_DEVICE)
        print({"tier": tier, "year": test_year, "fused_families": len(prepared), "nodes": len(graph["base"]), "train_nodes": len(graph["train_nodes"]), "architecture": run_label, "experts": getattr(model, "n_experts", 1), "device": str(GNN_DEVICE)}, flush=True)
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            model_output = model(xs, temporal_edges, temporal_attr, pair_src, pair_dst)
            if use_moe:
                node_pred, event_logits, pair_pred, balance_loss, aux_logits = model_output
            else:
                node_pred, event_logits, pair_pred, aux_logits = model_output
                balance_loss = torch.tensor(0.0, device=node_pred.device)
            graph_count = 6
            node_errors = nn.functional.smooth_l1_loss(node_pred[train_nodes], graph_y[train_nodes, :graph_count], reduction="none")
            node_loss = (node_errors * graph_mask[train_nodes]).sum() / graph_mask[train_nodes].sum().clamp_min(1.0)
            event_true = graph_y[train_nodes, graph_count:]
            event_loss = event_loss_from_logits(event_logits[train_nodes], event_true)
            edge_loss = nn.functional.smooth_l1_loss(pair_pred, pair_y)
            aux_loss = torch.tensor(0.0, device=node_pred.device)
            for aux_name, logits in aux_logits.items():
                aux_index = AUX_TARGET_COLS.index(aux_name)
                mask = aux_mask[train_nodes, aux_index]
                if mask.any():
                    aux_true = aux_y[train_nodes, aux_index][mask]
                    aux_loss = aux_loss + nn.functional.cross_entropy(logits[train_nodes][mask], aux_true)
            loss = node_loss + float(os.getenv("GNN_EVENT_LOSS_WEIGHT", "1.0")) * event_loss + float(os.getenv("GNN_EDGE_LOSS_WEIGHT", "0.25")) * edge_loss + float(os.getenv("GNN_MOE_BALANCE_WEIGHT", "0.01")) * balance_loss + float(os.getenv("GNN_AUX_LOSS_WEIGHT", "0.10")) * aux_loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            if epoch == 0 or epoch == EPOCHS - 1: print({"tier": tier, "year": test_year, "epoch": epoch + 1, "fused_loss": round(float(loss.detach()), 5), "moe_balance": round(float(balance_loss.detach()), 5)}, flush=True)
        model.eval(); test_start = pd.Timestamp(f"{test_year}-01-01"); test_end = pd.Timestamp(f"{test_year}-12-31")
        with torch.no_grad():
            model_output = model(xs, temporal_edges, temporal_attr, pair_src, pair_dst)
            values = combine_target_predictions(model_output[0], model_output[1]).cpu()
        pred = graph["base"][["symbol", "date"]].copy(); pred[graph["target_cols"]] = values.cpu().numpy(); pred = pred.loc[pred.date.between(test_start, test_end)].copy()
        for col in graph["target_cols"]: pred[col] = pred.groupby("date")[col].rank(pct=True, method="average")
        pred["long_score"], pred["long_exit_score"] = pred["long_hub"], pred["long_authority"]; pred["short_score"], pred["short_exit_score"] = pred["short_hub"], pred["short_authority"]
        pred["long_agree_count"] = (pred.long_score >= pred.short_score).astype(int); pred["short_agree_count"] = (pred.short_score > pred.long_score).astype(int); pred["model_count"] = 1; pred["tier"] = tier; pred["year"] = test_year
        predictions.append(pred)
        dates = pd.DatetimeIndex(next_returns.index[(next_returns.index >= test_start) & (next_returns.index <= test_end)])
        summary, _, _ = run_shared_book_framework_comparison(scores=pred[["symbol", "date", "long_score", "short_score", "long_exit_score", "short_exit_score", "long_agree_count", "short_agree_count", "model_count"]], next_returns=next_returns, symbols=tuple(close.columns), dates=dates, variants=(GNN_VARIANT,), top_k_values=(effective_top_k,), entry_threshold=0.5, exit_threshold=0.5, cost_models={"family_common": SharedBookCostModel(0.5, 5.0)})
        if not summary.empty:
            summary["tier"] = tier; summary["year"] = test_year; summary["family"] = "fused_all_families"; summary["label_source"] = f"gnn_{run_label}_sparse_hits"; summaries.append(summary)
        pd.concat(summaries, ignore_index=True).to_parquet(OUT / f"{tier.lower()}_{run_label}_long_only_through_{test_year}.parquet", index=False)
    result = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(); result.to_csv(OUT / f"{tier.lower()}_{run_label}_wfo_results.csv", index=False)
    if predictions: pd.concat(predictions, ignore_index=True).to_parquet(OUT / f"{tier.lower()}_{run_label}_wfo_predictions.parquet", index=False)
    print(result.to_string(index=False) if not result.empty else result); print({"tier": tier, "fused_seconds": round(perf_counter() - started, 1), "result_rows": len(result)}, flush=True)
    return result


def run_tier_fused_streaming(tier: str) -> pd.DataFrame:
    """Run fused MoE WFO by streaming complete per-symbol graph components.

    Every temporal and pair edge is within one symbol, so processing symbol
    batches is mathematically equivalent to processing the disjoint union of
    all symbol graphs, provided gradients are accumulated before optimizer
    steps.  This keeps all historical dates while bounding memory by the
    symbol batch size.
    """
    if GNN_VARIANT not in {"long_only", "short_only"}:
        raise ValueError("GNN_VARIANT must be 'long_only' or 'short_only'")
    started = perf_counter()
    chunk_size = max(1, int(os.getenv("GNN_SYMBOL_BATCH", "32")))
    index = pd.read_csv(feature_dir(tier) / "index.csv")
    requested = tuple(x.strip() for x in os.getenv("GNN_FAMILIES", "").split(",") if x.strip())
    if requested:
        index = index.loc[index.family.astype(str).isin(requested)].copy()
    if index.empty:
        return pd.DataFrame()
    first_panel = pd.read_parquet(str(index.iloc[0].panel_path), columns=["symbol"])
    symbols = sorted(first_panel.symbol.astype(str).str.upper().unique())
    del first_panel
    symbol_chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
    price_map, labels = build_price_and_labels(symbols, tier)
    close = price_map.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    next_returns = close.pct_change().shift(-1)
    effective_top_k = min(20, len(close.columns))
    summaries: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    for test_year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):
        feature_dims = {}
        for _, meta in index.iterrows():
            metadata = pd.read_parquet(str(meta.metadata_path))
            feature_dims[str(meta.family)] = int(len(metadata.feature.astype(str)))
        if not feature_dims:
            continue

        def prepare_stream_chunk(chunk_symbols: list[str]) -> list[dict]:
            chunk_prepared = []
            for _, meta in index.iterrows():
                try:
                    panel = pd.read_parquet(str(meta.panel_path), filters=[("symbol", "in", chunk_symbols)])
                except (NotImplementedError, ValueError, TypeError):
                    panel = pd.read_parquet(str(meta.panel_path))
                panel = panel.loc[panel.symbol.astype(str).str.upper().isin(chunk_symbols)].copy()
                panel.attrs["metadata_path"] = meta.metadata_path
                item = prepare_family(panel, price_map, labels, str(meta.family), test_year)
                if not item.get("status"):
                    chunk_prepared.append(item)
                del panel
            if chunk_prepared:
                names = tuple(item["family"] for item in chunk_prepared)
                if names != tuple(feature_dims):
                    raise RuntimeError(f"Streaming family mismatch for {tier} {test_year}: {names} != {tuple(feature_dims)}")
                node_hashes = {int(pd.util.hash_pandas_object(item["base"][['symbol', 'date']], index=False).sum()) for item in chunk_prepared}
                if len(node_hashes) != 1:
                    raise RuntimeError(f"Feature families do not share node index in streamed chunk for {tier} {test_year}")
            return chunk_prepared

        model = FusedFamilyMoEGNN(feature_dims, aux_dims=AUX_CLASS_DIMS).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(os.getenv("GNN_LR", "0.002")), weight_decay=1e-4)
        chunk_weights = {len(chunk_symbols): len(chunk_symbols) / max(1, len(symbols)) for chunk_symbols in symbol_chunks}
        print({"tier": tier, "year": test_year, "architecture": "fused_moe_streaming", "experts": model.n_experts, "symbols": len(symbols), "symbol_batch": chunk_size, "chunks": len(symbol_chunks)}, flush=True)
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            last_loss = torch.tensor(0.0)
            for chunk_symbols in symbol_chunks:
                prepared = prepare_stream_chunk(chunk_symbols)
                if not prepared:
                    continue
                graph = prepared[0]
                model_output = model([item["x"] for item in prepared], graph["temporal_edges"], graph["temporal_attr"], graph["pair_src"], graph["pair_dst"])
                node_pred, event_logits, pair_pred, balance_loss, aux_logits = model_output
                train_nodes = graph["train_nodes"]; graph_count = 6
                node_errors = nn.functional.smooth_l1_loss(node_pred[train_nodes], graph["ha_y"][train_nodes, :graph_count], reduction="none")
                node_loss = (node_errors * graph["ha_mask"][train_nodes]).sum() / graph["ha_mask"][train_nodes].sum().clamp_min(1.0)
                event_true = graph["ha_y"][train_nodes, graph_count:]
                event_loss = event_loss_from_logits(event_logits[train_nodes], event_true)
                edge_loss = nn.functional.smooth_l1_loss(pair_pred, graph["pair_y"])
                aux_loss = torch.tensor(0.0, device=node_pred.device)
                for aux_name, logits in aux_logits.items():
                    aux_index = AUX_TARGET_COLS.index(aux_name)
                    mask = graph["aux_mask"][train_nodes, aux_index]
                    if mask.any():
                        aux_true = graph["aux_y"][train_nodes, aux_index][mask]
                        aux_loss = aux_loss + nn.functional.cross_entropy(logits[train_nodes][mask], aux_true)
                node_weight = chunk_weights[len(chunk_symbols)]
                pair_weight = node_weight
                loss = node_weight * (node_loss + float(os.getenv("GNN_EVENT_LOSS_WEIGHT", "1.0")) * event_loss + float(os.getenv("GNN_MOE_BALANCE_WEIGHT", "0.01")) * balance_loss + float(os.getenv("GNN_AUX_LOSS_WEIGHT", "0.10")) * aux_loss) + pair_weight * float(os.getenv("GNN_EDGE_LOSS_WEIGHT", "0.25")) * edge_loss
                loss.backward()
                last_loss = loss.detach()
                del prepared
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            if epoch == 0 or epoch == EPOCHS - 1:
                print({"tier": tier, "year": test_year, "epoch": epoch + 1, "stream_loss": round(float(last_loss), 5)}, flush=True)
        model.eval(); test_start = pd.Timestamp(f"{test_year}-01-01"); test_end = pd.Timestamp(f"{test_year}-12-31")
        year_predictions = []
        target_cols = None
        for chunk_symbols in symbol_chunks:
            prepared = prepare_stream_chunk(chunk_symbols)
            if not prepared:
                continue
            graph = prepared[0]
            with torch.no_grad():
                model_output = model([item["x"] for item in prepared], graph["temporal_edges"], graph["temporal_attr"], graph["pair_src"], graph["pair_dst"])
                values = combine_target_predictions(model_output[0], model_output[1])
            pred = graph["base"][['symbol', 'date']].copy(); pred[graph["target_cols"]] = values.cpu().numpy(); pred = pred.loc[pred.date.between(test_start, test_end)].copy()
            year_predictions.append(pred); target_cols = graph["target_cols"]
            del prepared
        if not year_predictions:
            del model
            continue
        pred = pd.concat(year_predictions, ignore_index=True)
        predictions.extend(year_predictions)
        for col in target_cols:
            pred[col] = pred.groupby("date")[col].rank(pct=True, method="average")
        pred["long_score"], pred["long_exit_score"] = pred["long_hub"], pred["long_authority"]
        pred["short_score"], pred["short_exit_score"] = pred["short_hub"], pred["short_authority"]
        pred["long_agree_count"] = (pred.long_score >= pred.short_score).astype(int); pred["short_agree_count"] = (pred.short_score > pred.long_score).astype(int); pred["model_count"] = 1; pred["tier"] = tier; pred["year"] = test_year
        dates = pd.DatetimeIndex(next_returns.index[(next_returns.index >= test_start) & (next_returns.index <= test_end)])
        summary, _, _ = run_shared_book_framework_comparison(scores=pred[["symbol", "date", "long_score", "short_score", "long_exit_score", "short_exit_score", "long_agree_count", "short_agree_count", "model_count"]], next_returns=next_returns, symbols=tuple(close.columns), dates=dates, variants=(GNN_VARIANT,), top_k_values=(effective_top_k,), entry_threshold=0.5, exit_threshold=0.5, cost_models={"family_common": SharedBookCostModel(0.5, 5.0)})
        if not summary.empty:
            summary["tier"] = tier; summary["year"] = test_year; summary["family"] = "fused_all_families"; summary["label_source"] = "gnn_fused_moe_streaming_sparse_hits"; summaries.append(summary)
        pd.concat(summaries, ignore_index=True).to_parquet(OUT / f"{tier.lower()}_fused_moe_streaming_long_only_through_{test_year}.parquet", index=False)
        del year_predictions, model
    result = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    result.to_csv(OUT / f"{tier.lower()}_fused_moe_streaming_wfo_results.csv", index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_parquet(OUT / f"{tier.lower()}_fused_moe_streaming_wfo_predictions.parquet", index=False)
    print(result.to_string(index=False) if not result.empty else result)
    print({"tier": tier, "streaming_seconds": round(perf_counter() - started, 1), "result_rows": len(result)}, flush=True)
    return result


def main() -> None:
    requested = tuple(x.strip().upper() for x in os.getenv("GNN_TIERS", "1T").split(",") if x.strip())
    unknown = sorted(set(requested) - set(TIER_CONFIGS))
    if unknown:
        raise ValueError(f"unknown GNN_TIERS: {unknown}")
    if os.getenv("GNN_STREAM_SYMBOLS", "0") == "1":
        runner = run_tier_fused_streaming
    elif os.getenv("GNN_FUSED_FAMILY", "0") == "1" or os.getenv("GNN_MOE_FAMILY", "0") == "1":
        runner = run_tier_fused
    elif os.getenv("GNN_SHARED_FAMILY", "0") == "1":
        runner = run_tier_shared
    else:
        runner = run_tier
    all_results = [runner(tier) for tier in requested]
    combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    combined.to_csv(OUT / f"all_{GNN_VARIANT}_wfo_results.csv", index=False)
    if not combined.empty:
        print(combined.groupby(["tier", "year", "variant"], as_index=False).agg(families=("family", "nunique"), mean_return=("total_return", "mean"), median_return=("total_return", "median"), min_return=("total_return", "min"), max_return=("total_return", "max"), mean_sharpe=("sharpe", "mean")).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
