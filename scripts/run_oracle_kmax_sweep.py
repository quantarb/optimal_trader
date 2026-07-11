#!/usr/bin/env python3
"""Oracle-only YE k_max sweep: k=1..12 on 1T, 100B, 10B.

For each universe:
  1) Build/reuse feature panels once
  2) For k_max = 1,2,...,12: retrain with YE oracle labels using k in {1..k_max}
  3) Write a summary table of best Sharpe / return vs k_max

This does NOT reimplement the notebook — it drives run_equity_meta_notebook.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "run_equity_meta_notebook.py"
ART = REPO / "artifacts" / "trading_app_v2"

UNIVERSES = (
    # tag, min_market_cap, smoke_flags style
    ("1t", 1_000_000_000_000, True),
    ("100b", 100_000_000_000, False),
    ("10b", 10_000_000_000, False),
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _run_one(
    *,
    tag: str,
    mcap: int,
    k_max: int,
    smoke: bool,
    feature_cache_dir: Path | None,
) -> dict:
    out_root = ART / f"equity_meta_model_{tag}_oracle_only_kmax{k_max:02d}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "notebook_runner.log"

    cmd = [
        sys.executable,
        "-u",
        str(RUNNER),
        "--min-market-cap",
        str(mcap),
        "--tag",
        tag,
        "--label-mode",
        "oracle_only",
        "--oracle-ye-k-max",
        str(k_max),
        "--run-name-suffix",
        f"kmax{k_max:02d}",
        "--skip-anchored-wfo",
        "--skip-symbol-level-bt",
        "--rebuild-family-score-cache",
    ]
    if smoke:
        cmd.append("--smoke")
    if feature_cache_dir is not None:
        cmd.extend(["--feature-cache-dir", str(feature_cache_dir)])

    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    # Prefer local packages
    py_path = ":".join(
        [
            str(REPO),
            str(REPO.parent / "quant-warehouse"),
            str(REPO.parent / "quant-orchestrator"),
            env.get("PYTHONPATH", ""),
        ]
    )
    env["PYTHONPATH"] = py_path

    _log(f"\n===== {tag} oracle_only YE k_max={k_max} =====")
    _log("cmd: " + " ".join(cmd))
    t0 = perf_counter()
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO.parent),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = perf_counter() - t0
    if proc.returncode != 0:
        _log(f"FAILED {tag} k_max={k_max} rc={proc.returncode} log={log_path}")
        # tail log
        try:
            lines = log_path.read_text(errors="replace").splitlines()
            for line in lines[-40:]:
                _log(line)
        except Exception:
            pass
        raise RuntimeError(f"{tag} k_max={k_max} failed with rc={proc.returncode}")

    # Locate artifact dir written by runner
    # Pattern: equity_meta_model_{tag}_oracle_only_kmaxNN/mcap_...
    candidates = sorted(out_root.glob("mcap_*"))
    # Also possible nested name from dir_parts logic
    alt_root = ART / f"equity_meta_model_{tag}_oracle_only_kmax{k_max:02d}"
    if not candidates:
        # runner may have used equity_meta_model_{tag}_oracle_only_kmaxXX from suffix
        for p in ART.glob(f"equity_meta_model_{tag}_oracle_only*kmax{k_max:02d}*"):
            c = sorted(p.glob("mcap_*"))
            if c:
                candidates = c
                out_root = p
                break

    if not candidates:
        raise FileNotFoundError(f"No mcap artifact dir under {out_root}")

    run_dir = candidates[0]
    summary_path = run_dir / "notebook_runner_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary["wall_seconds"] = elapsed
    summary["log_path"] = str(log_path)
    summary["run_dir"] = str(run_dir)

    lb_path = run_dir / "trading_performance_leaderboard.csv"
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        summary["leaderboard_rows"] = len(lb)
        if not lb.empty:
            for _, row in lb.iterrows():
                src = str(row["strategy_source"])
                summary[f"{src}__total_return"] = float(row["total_return"])
                summary[f"{src}__sharpe"] = float(row["sharpe"])
                summary[f"{src}__max_drawdown"] = float(row["max_drawdown"])
                summary[f"{src}__ann_return"] = float(row.get("annualized_return", float("nan")))

    fe = run_dir / "feature_family_panels"
    summary["feature_family_panels"] = str(fe if fe.exists() else "")
    _log(
        f"OK {tag} k_max={k_max} labels={summary.get('oracle_label_rows')} "
        f"best={summary.get('best_strategy')} sharpe={summary.get('best_sharpe')} "
        f"ret={summary.get('best_total_return')} elapsed={elapsed:.0f}s"
    )
    return summary


def main() -> int:
    started = perf_counter()
    all_rows: list[dict] = []
    study_root = ART / "oracle_kmax_sweep"
    study_root.mkdir(parents=True, exist_ok=True)

    for tag, mcap, smoke in UNIVERSES:
        _log(f"\n########## UNIVERSE {tag} mcap={mcap} ##########")
        feature_cache: Path | None = None
        # Prefer existing FE from prior all-events run if present and complete.
        prior = ART / f"equity_meta_model_{tag}" / f"mcap_{mcap}_train_2020-12-31_seed_20260707" / "feature_family_panels"
        if prior.exists() and (prior / "index.csv").exists():
            idx = pd.read_csv(prior / "index.csv")
            if len(idx) >= 20 and idx["features"].gt(0).sum() >= 20:
                feature_cache = prior
                _log(f"Reusing prior FE cache: {feature_cache}")

        for k_max in range(1, 13):
            summary = _run_one(
                tag=tag,
                mcap=mcap,
                k_max=k_max,
                smoke=smoke,
                feature_cache_dir=feature_cache,
            )
            all_rows.append(summary)
            # After first successful k in this universe, lock FE for remaining k.
            fe = summary.get("feature_family_panels") or ""
            if feature_cache is None and fe and Path(fe).exists():
                feature_cache = Path(fe)
                _log(f"Locking FE cache for remaining k: {feature_cache}")

            # Incremental table
            df = pd.DataFrame(all_rows)
            df.to_csv(study_root / "oracle_kmax_sweep_results.csv", index=False)
            (study_root / "oracle_kmax_sweep_results.json").write_text(
                json.dumps(all_rows, indent=2, default=str), encoding="utf-8"
            )

        # Per-universe best-k table
        sub = [r for r in all_rows if r.get("tag") == tag]
        if sub:
            pdf = pd.DataFrame(sub).sort_values("oracle_ye_k_max")
            cols = [
                c
                for c in [
                    "oracle_ye_k_max",
                    "oracle_label_rows",
                    "n_symbols",
                    "best_strategy",
                    "best_total_return",
                    "best_sharpe",
                    "best_max_drawdown",
                    "ensemble_mean__total_return",
                    "ensemble_mean__sharpe",
                    "ensemble_rank_mean__total_return",
                    "ensemble_rank_mean__sharpe",
                    "stacked_meta__total_return",
                    "stacked_meta__sharpe",
                ]
                if c in pdf.columns
            ]
            pdf[cols].to_csv(study_root / f"{tag}_kmax_table.csv", index=False)
            _log(f"\n=== {tag} k_max table ===")
            _log(pdf[cols].to_string(index=False))

            # Recommend max k by best_sharpe plateau
            if "best_sharpe" in pdf.columns:
                best_row = pdf.loc[pdf["best_sharpe"].idxmax()]
                _log(
                    f"RECOMMEND {tag}: max best_sharpe at k_max={int(best_row['oracle_ye_k_max'])} "
                    f"sharpe={best_row['best_sharpe']:.4f} ret={best_row['best_total_return']:.4f} "
                    f"strategy={best_row.get('best_strategy')}"
                )

    elapsed = perf_counter() - started
    _log(f"\nALL SWEEPS DONE in {elapsed:.0f}s")
    _log(f"Results: {study_root / 'oracle_kmax_sweep_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
