from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from app.live_trade_leaderboard import default_live_trade_config, load_saved_leaderboard
from app.optimal_trade_lookup import OptimalTradeQuery
from pipeline.notebook_universe import normalize_symbols
REQUIRED_SIMILARITY_ARTIFACTS = (
    "meta.json",
    "clf_raw.pkl",
    "reg_trade_return_raw.pkl",
    "ae_raw.pkl",
    "leaderboard_latest_meta.json",
    "leaderboard_latest.pkl",
)


def find_repo_root(start: Path) -> Path:
    start = Path(start).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "app").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repo root from cwd={start}")


def load_repo_env(repo_root: Path) -> None:
    env_path = Path(repo_root) / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ[key] = value.strip().strip('"').strip("'")


def bootstrap_repo(repo_root: Path) -> None:
    repo_root = Path(repo_root).resolve()
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    load_repo_env(repo_root)
    os.environ.setdefault("DJANGO_ALLOW_ASYNC_UNSAFE", "true")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")


def make_live_trade_notebook_config(
    repo_root: Path,
    *,
    data_start: str,
    min_market_cap: float,
    refresh_fmp_data: bool,
    skip_cached_inactive_symbols: bool,
    refresh_macro_data: bool,
    leaderboard_top_k: int,
    universe_source: str = "auto",
    symbols: list[str] | tuple[str, ...] | str | None = None,
) -> dict[str, Any]:
    cfg = default_live_trade_config()
    cfg["dates"]["data_start"] = str(data_start)
    cfg["dates"]["data_end"] = pd.Timestamp.today().strftime("%Y-%m-%d")
    cfg["universe"]["min_market_cap"] = float(min_market_cap)
    cfg["universe"]["source"] = str(universe_source or "auto")
    cfg["universe"]["symbols"] = symbols or []
    artifact_dir = (Path(repo_root) / "artifacts" / "raw_stack").resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cfg["runtime"]["artifact_dir"] = str(artifact_dir)
    cfg["fmp_refresh"].update(
        {
            "enabled": bool(refresh_fmp_data),
            "refresh_symbol_sections_before_build": bool(refresh_fmp_data),
            "refresh_macro_before_build": bool(refresh_macro_data),
            "repair_symbol_metadata_before_build": True,
            "skip_cached_inactive_symbols": bool(skip_cached_inactive_symbols),
            "verbose": False,
        }
    )
    cfg["strategy"]["top_k"] = int(leaderboard_top_k)
    return cfg


def live_trade_notebook_summary(
    cfg: dict[str, Any],
    *,
    query_symbol: str,
    similar_trades_top_k: int,
) -> dict[str, Any]:
    return {
        "query_symbol": str(query_symbol).strip().upper(),
        "universe_source": str(cfg["universe"].get("source") or "auto"),
        "explicit_symbol_count": len(normalize_symbols(cfg["universe"].get("symbols"))),
        "data_start": cfg["dates"]["data_start"],
        "data_end": cfg["dates"]["data_end"],
        "min_market_cap": cfg["universe"]["min_market_cap"],
        "download_missing_fmp_data": cfg["fmp_refresh"]["refresh_symbol_sections_before_build"],
        "skip_cached_inactive_symbols": cfg["fmp_refresh"]["skip_cached_inactive_symbols"],
        "refresh_macro_data": cfg["fmp_refresh"]["refresh_macro_before_build"],
        "leaderboard_top_k": cfg["strategy"]["top_k"],
        "similar_trades_top_k": int(similar_trades_top_k),
        "artifact_dir": cfg["runtime"]["artifact_dir"],
    }


def load_recent_similarity_build(
    cfg: dict[str, Any],
    *,
    max_age: pd.Timedelta = pd.Timedelta(days=1),
):
    from fmp.refresh import expected_latest_price_date_from_market_clock

    artifact_dir = Path(str(cfg["runtime"]["artifact_dir"])).expanduser().resolve()
    required_paths = [artifact_dir / name for name in REQUIRED_SIMILARITY_ARTIFACTS]
    missing = [path.name for path in required_paths if not path.exists()]
    if missing:
        return None, {"artifact_dir": str(artifact_dir), "age": pd.NaT, "reason": f"missing_files={','.join(missing)}"}

    latest_mtime = max(path.stat().st_mtime for path in required_paths)
    artifact_age = pd.Timestamp.now(tz="UTC") - pd.Timestamp(latest_mtime, unit="s", tz="UTC")
    if artifact_age > pd.Timedelta(max_age):
        return None, {"artifact_dir": str(artifact_dir), "age": artifact_age, "reason": f"older_than_{pd.Timedelta(max_age)}"}

    saved = load_saved_leaderboard(artifact_dir=str(artifact_dir))
    if saved is None:
        return None, {"artifact_dir": str(artifact_dir), "age": artifact_age, "reason": "missing_saved_leaderboard"}

    leaderboard, meta = saved
    meta = dict(meta or {})
    latest_date = pd.Timestamp(meta.get("latest_date") or pd.Timestamp.today().normalize()).normalize()
    requested_end_date = pd.Timestamp(str(cfg["dates"].get("data_end") or pd.Timestamp.today().date())).normalize()
    expected_score_date = min(
        requested_end_date,
        pd.Timestamp(expected_latest_price_date_from_market_clock()).normalize(),
    )
    if latest_date < expected_score_date:
        return None, {
            "artifact_dir": str(artifact_dir),
            "age": artifact_age,
            "reason": f"saved_latest_date_{latest_date.date().isoformat()}_lt_expected_{expected_score_date.date().isoformat()}",
            "saved_latest_date": latest_date.date().isoformat(),
            "expected_score_date": expected_score_date.date().isoformat(),
        }

    universe_size = int(pd.to_numeric(pd.Series([meta.get("universe_size")]), errors="coerce").fillna(0).iloc[0])
    vector_metadata = dict(meta.get("vector_metadata") or {})
    reference_trade_count = int(
        pd.to_numeric(pd.Series([vector_metadata.get("row_count")]), errors="coerce").fillna(0).iloc[0]
    )
    return SimpleNamespace(
        latest_date=latest_date,
        artifact_dir=artifact_dir,
        leaderboard=leaderboard,
        universe=tuple(range(universe_size)),
        reference_trade_count=reference_trade_count,
        vector_metadata=vector_metadata,
        source="artifacts",
    ), {"artifact_dir": str(artifact_dir), "age": artifact_age, "reason": "fresh_saved_build"}


def make_similarity_query(
    build_result,
    cfg: dict[str, Any],
    *,
    query_symbol: str,
    top_k: int,
) -> OptimalTradeQuery:
    latest_date = pd.Timestamp(build_result.latest_date).normalize()
    return OptimalTradeQuery(
        symbol=str(query_symbol).strip().upper(),
        as_of_date=latest_date.strftime("%Y-%m-%d"),
        query_lookback_years=0,
        reference_symbols=(),
        reference_start_date=cfg["dates"]["data_start"],
        reference_end_date=(latest_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        top_k=int(top_k),
        label_freq="YE",
        label_k_values=tuple(int(value) for value in cfg["labels"]["k_params"]["YE"]),
        min_profit_pct_points=0.0,
        download_missing_prices=False,
        artifact_dir=str(build_result.artifact_dir),
    )


__all__ = [
    "bootstrap_repo",
    "find_repo_root",
    "live_trade_notebook_summary",
    "load_recent_similarity_build",
    "load_repo_env",
    "make_live_trade_notebook_config",
    "make_similarity_query",
]
