# ============================================================
# modules/config.py  (PATCH: make ranking defaults consistent + add rebalance/anchor)
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import copy
import os

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


MAG7: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

CONNORS_20_ETFS: List[str] = [
    "DIA","EEM","EFA","EWH","EWJ","EWT","EWZ","FXI","GLD","ILF",
    "IWM","IYR","QQQ","SPY","XHB","XLB","XLE","XLF","XLI","XLV"
]

DEFAULT_HORIZONS: List[str] = ["W", "ME", "QE", "YE"]
DEFAULT_K_PARAMS: Dict[str, int] = {"W": 1, "ME": 1, "QE": 1, "YE": 1}

DEFAULT_EXECUTION_PARAMS: Dict[str, Any] = dict(
    price_col="close",
    buy_threshold=0.65,
    sell_threshold=0.65,
    fee_bps=2.0,
    slippage_bps=2.0,
)

DEFAULT_RF_PARAMS: Dict[str, Any] = dict(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1,
)

DEFAULT_WEIGHTING: Dict[str, Any] = dict(
    use_sample_weight=True,
    r_clip=0.10,
    alpha=4.0,
    horizon_balance=True,
    horizon_balance_mode="mass",
    entry_only_weighting=True,
)

# ✅ PATCH: ranking defaults should match your current pipeline usage.
# - You were using selection_mode="top_pct" everywhere, but config said "topk".
# - Add rebalance + anchor so config can drive the spec consistently.
DEFAULT_RANKING: Dict[str, Any] = dict(
    top_k=10,
    selection_mode="top_pct",          # ✅ was "topk" (inconsistent)
    weight_scheme="equal",
    min_buy_prob=None,
    use_sell_exit=True,
    rebalance="W",                     # ✅ NEW
    anchor="period_start",             # ✅ NEW
)

DEFAULT_DATA_QUALITY: Dict[str, Any] = dict(
    price_floor=0.01,
    max_abs_ret_1d=2.0,
    max_bad_price_frac=0.01,
    max_bad_ret_frac=0.01,
    prefer_adj_close=False,
    drop_bad_bars=True,
)


def _get_env_required(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Set it in your shell or in a .env file (and install python-dotenv)."
        )
    return v


@dataclass(frozen=True)
class ExperimentConfig:
    api_key: str
    data_dir: str = "./data"
    db_path: str = "./data/quant.db"

    history_years: Optional[int] = None

    horizons: List[str] = None
    k_params: Dict[str, int] = None
    execution_params: Dict[str, Any] = None
    rf_params: Dict[str, Any] = None
    weighting: Dict[str, Any] = None
    ranking: Dict[str, Any] = None
    data_quality: Dict[str, Any] = None

    sleep_s: float = 1.0
    debug_first_symbol: bool = True
    debug_data_quality: bool = False


def deep_update(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not overrides:
        return copy.deepcopy(base)
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def make_config(
    *,
    api_key: Optional[str] = None,
    data_dir: str = "./data",
    db_path: Optional[str] = None,
    horizons: Optional[List[str]] = None,
    k_params_overrides: Optional[Dict[str, Any]] = None,
    execution_overrides: Optional[Dict[str, Any]] = None,
    rf_overrides: Optional[Dict[str, Any]] = None,
    weighting_overrides: Optional[Dict[str, Any]] = None,
    ranking_overrides: Optional[Dict[str, Any]] = None,
    data_quality_overrides: Optional[Dict[str, Any]] = None,
    debug_data_quality: bool = False,
    sleep_s: float = 1.0,
    debug_first_symbol: bool = True,
    history_years: Optional[int] = None,
) -> ExperimentConfig:
    os.makedirs(data_dir, exist_ok=True)

    if db_path is None:
        db_path = os.path.join(data_dir, "quant.db")

    if api_key is None:
        api_key = _get_env_required("FMP_API_KEY")

    return ExperimentConfig(
        api_key=api_key,
        data_dir=data_dir,
        db_path=db_path,
        history_years=history_years,
        horizons=list(horizons) if horizons is not None else list(DEFAULT_HORIZONS),
        k_params=deep_update(DEFAULT_K_PARAMS, k_params_overrides),
        execution_params=deep_update(DEFAULT_EXECUTION_PARAMS, execution_overrides),
        rf_params=deep_update(DEFAULT_RF_PARAMS, rf_overrides),
        weighting=deep_update(DEFAULT_WEIGHTING, weighting_overrides),
        ranking=deep_update(DEFAULT_RANKING, ranking_overrides),        # ✅ now carries rebalance/anchor
        data_quality=deep_update(DEFAULT_DATA_QUALITY, data_quality_overrides),
        debug_data_quality=debug_data_quality,
        sleep_s=sleep_s,
        debug_first_symbol=debug_first_symbol,
    )
