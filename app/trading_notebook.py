from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from app.live_trade_leaderboard import default_live_trade_config, load_saved_leaderboard
from app.live_app_shared import inspect_saved_artifacts
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
    import django
    from django.apps import apps

    if not apps.ready:
        django.setup()


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
            "repair_symbol_metadata_before_build": False,
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
    from fmp.market_clock import expected_latest_price_date_from_market_clock

    artifact_dir = Path(str(cfg["runtime"]["artifact_dir"])).expanduser().resolve()
    required_paths = [artifact_dir / name for name in REQUIRED_SIMILARITY_ARTIFACTS]
    preliminary_status = inspect_saved_artifacts(
        required_paths=required_paths,
        artifact_dir=artifact_dir,
        max_age=max_age,
    )
    if not preliminary_status["reusable"]:
        return None, preliminary_status

    saved = load_saved_leaderboard(artifact_dir=str(artifact_dir))
    if saved is None:
        return None, {**preliminary_status, "reason": "missing_saved_leaderboard", "reusable": False}

    leaderboard, meta = saved
    meta = dict(meta or {})
    latest_date = pd.Timestamp(meta.get("latest_date") or pd.Timestamp.today().normalize()).normalize()
    requested_end_date = pd.Timestamp(str(cfg["dates"].get("data_end") or pd.Timestamp.today().date())).normalize()
    expected_score_date = min(
        requested_end_date,
        pd.Timestamp(expected_latest_price_date_from_market_clock()).normalize(),
    )
    artifact_status = inspect_saved_artifacts(
        required_paths=required_paths,
        artifact_dir=artifact_dir,
        max_age=max_age,
        saved_score_date=latest_date,
        expected_score_date=expected_score_date,
    )
    if not artifact_status["reusable"]:
        return None, artifact_status

    universe_size = int(pd.to_numeric(pd.Series([meta.get("universe_size")]), errors="coerce").fillna(0).iloc[0])
    scored_symbol_count = int(
        pd.to_numeric(pd.Series([meta.get("scored_symbol_count")]), errors="coerce").fillna(0).iloc[0]
    )
    if scored_symbol_count <= 0:
        scored_symbol_count = int(len(leaderboard))
    if universe_size > 0:
        coverage_ratio = float(scored_symbol_count) / float(universe_size)
        if coverage_ratio < 0.5:
            return None, {
                **artifact_status,
                "reusable": False,
                "reason": (
                    "partial_leaderboard_coverage"
                    f" | scored {scored_symbol_count:,}/{universe_size:,}"
                    f" ({coverage_ratio:.1%})"
                ),
            }
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
    ), artifact_status


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
