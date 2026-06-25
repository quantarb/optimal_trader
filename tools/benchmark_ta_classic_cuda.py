from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class Timing:
    label: str
    seconds: float


def make_ohlcv_panel(symbols: int, rows_per_symbol: int, seed: int = 1337) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    symbol_values = np.repeat([f"S{i:04d}" for i in range(symbols)], rows_per_symbol)
    row_values = np.tile(np.arange(rows_per_symbol, dtype=np.int32), symbols)
    innovations = rng.normal(0.0002, 0.018, size=(symbols, rows_per_symbol))
    close = 100.0 * np.exp(np.cumsum(innovations, axis=1)).reshape(-1)
    spread = rng.uniform(0.001, 0.025, size=close.size)
    open_ = close * (1.0 + rng.normal(0.0, 0.004, size=close.size))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.lognormal(14.0, 0.8, size=close.size)
    return pd.DataFrame(
        {
            "symbol": symbol_values,
            "row": row_values,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _drop_group_level(series):
    return series.reset_index(level=0, drop=True)


def pandas_rolling_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("symbol", sort=False, observed=True)
    close = frame["close"]
    previous_close = grouped["close"].shift(1)
    mean_20 = _drop_group_level(grouped["close"].rolling(20, min_periods=20).mean())
    std_20 = _drop_group_level(grouped["close"].rolling(20, min_periods=20).std(ddof=1))
    high_20 = _drop_group_level(grouped["high"].rolling(20, min_periods=20).max())
    low_20 = _drop_group_level(grouped["low"].rolling(20, min_periods=20).min())
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    temp = frame[["symbol"]].copy()
    temp["true_range"] = true_range
    atr_14 = _drop_group_level(
        temp.groupby("symbol", sort=False, observed=True)["true_range"]
        .rolling(14, min_periods=14)
        .mean()
    )
    denominator = (high_20 - low_20).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "return_1": close / previous_close - 1.0,
            "sma_20": mean_20,
            "stdev_20": std_20,
            "zscore_20": (close - mean_20) / std_20,
            "bb_upper_20": mean_20 + 2.0 * std_20,
            "bb_lower_20": mean_20 - 2.0 * std_20,
            "donchian_high_20": high_20,
            "donchian_low_20": low_20,
            "stoch_20": 100.0 * (close - low_20) / denominator,
            "atr_sma_14": atr_14,
        }
    )


def cudf_rolling_candidates(frame):
    import cudf

    grouped = frame.groupby("symbol", sort=False)
    close = frame["close"]
    previous_close = grouped["close"].shift(1)
    mean_20 = _drop_group_level(grouped["close"].rolling(20, min_periods=20).mean())
    std_20 = _drop_group_level(grouped["close"].rolling(20, min_periods=20).std(ddof=1))
    high_20 = _drop_group_level(grouped["high"].rolling(20, min_periods=20).max())
    low_20 = _drop_group_level(grouped["low"].rolling(20, min_periods=20).min())
    true_range = cudf.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    temp = frame[["symbol"]].copy()
    temp["true_range"] = true_range
    atr_14 = _drop_group_level(
        temp.groupby("symbol", sort=False)["true_range"].rolling(14, min_periods=14).mean()
    )
    denominator = high_20 - low_20
    denominator = denominator.where(denominator != 0.0)
    return cudf.DataFrame(
        {
            "return_1": close / previous_close - 1.0,
            "sma_20": mean_20,
            "stdev_20": std_20,
            "zscore_20": (close - mean_20) / std_20,
            "bb_upper_20": mean_20 + 2.0 * std_20,
            "bb_lower_20": mean_20 - 2.0 * std_20,
            "donchian_high_20": high_20,
            "donchian_low_20": low_20,
            "stoch_20": 100.0 * (close - low_20) / denominator,
            "atr_sma_14": atr_14,
        }
    )


def synchronize_cuda() -> None:
    import cupy as cp

    cp.cuda.get_current_stream().synchronize()


def timed(label: str, function, repeats: int) -> tuple[Timing, object]:
    best = float("inf")
    result = None
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        result = function()
        elapsed = time.perf_counter() - started
        best = min(best, elapsed)
    return Timing(label, best), result


def profile_pandas_ta_classic(rows_per_symbol: int, repeats: int) -> tuple[list[Timing], list[Timing]]:
    import pandas_ta_classic as ta

    from domain.features.ta_classic_technical import (
        _compute_indicator,
        _indicator_specs,
        _prepare_price_frame,
    )

    prices = make_ohlcv_panel(1, rows_per_symbol).drop(columns=["symbol", "row"])
    prices.index = pd.date_range("2000-01-03", periods=len(prices), freq="B")
    prices = _prepare_price_frame(prices)
    family_timings: list[Timing] = []
    indicator_timings: list[Timing] = []
    for family_name, specs in _indicator_specs(ta).items():
        def run_family():
            return [_compute_indicator(ta, prices, spec) for spec in specs]

        family_timing, _ = timed(f"pandas-ta {family_name}", run_family, repeats)
        family_timings.append(family_timing)
        for position, spec in enumerate(specs):
            indicator_timing, _ = timed(
                f"{family_name}:{position:03d}:{spec.name}",
                lambda spec=spec: _compute_indicator(ta, prices, spec),
                repeats,
            )
            indicator_timings.append(indicator_timing)
    return family_timings, indicator_timings


def compare_results(cpu: pd.DataFrame, gpu: pd.DataFrame) -> tuple[float, float]:
    cpu_values = cpu.to_numpy(dtype=np.float64)
    gpu_values = gpu.to_numpy(dtype=np.float64)
    finite = np.isfinite(cpu_values) & np.isfinite(gpu_values)
    if not finite.any():
        return float("nan"), float("nan")
    absolute = np.abs(cpu_values[finite] - gpu_values[finite])
    scale = np.maximum(np.abs(cpu_values[finite]), 1e-12)
    return float(absolute.max()), float((absolute / scale).max())


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark batched pandas versus CUDA technical primitives.")
    parser.add_argument("--symbols", type=int, default=800)
    parser.add_argument("--rows", type=int, default=2500, help="Rows per symbol")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--profile-ta", action="store_true", help="Also time existing pandas-ta families for one symbol")
    args = parser.parse_args()

    frame = make_ohlcv_panel(args.symbols, args.rows)
    print(f"Input: {args.symbols:,} symbols x {args.rows:,} rows = {len(frame):,} rows")

    cpu_timing, cpu_result = timed(
        "pandas batched rolling candidates",
        lambda: pandas_rolling_candidates(frame),
        args.repeats,
    )
    print(f"{cpu_timing.label}: {cpu_timing.seconds:.3f}s")

    try:
        import cudf
        import cupy as cp

        print(f"CUDA device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        transfer_started = time.perf_counter()
        gpu_frame = cudf.from_pandas(frame)
        synchronize_cuda()
        host_to_device_seconds = time.perf_counter() - transfer_started

        def run_gpu():
            result = cudf_rolling_candidates(gpu_frame)
            synchronize_cuda()
            return result

        _ = run_gpu()
        gpu_timing, gpu_result = timed("cuDF compute only", run_gpu, args.repeats)
        transfer_back_started = time.perf_counter()
        gpu_result_pdf = gpu_result.to_pandas()
        synchronize_cuda()
        device_to_host_seconds = time.perf_counter() - transfer_back_started
        total_gpu = host_to_device_seconds + gpu_timing.seconds + device_to_host_seconds
        max_abs, max_rel = compare_results(cpu_result, gpu_result_pdf)
        print(f"host -> device: {host_to_device_seconds:.3f}s")
        print(f"{gpu_timing.label}: {gpu_timing.seconds:.3f}s")
        print(f"device -> host: {device_to_host_seconds:.3f}s")
        print(f"cuDF transfer-inclusive: {total_gpu:.3f}s")
        print(f"speedup compute-only: {cpu_timing.seconds / gpu_timing.seconds:.2f}x")
        print(f"speedup transfer-inclusive: {cpu_timing.seconds / total_gpu:.2f}x")
        print(f"agreement: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")
    except (ImportError, RuntimeError) as exc:
        print(f"CUDA benchmark unavailable: {type(exc).__name__}: {exc}")

    if args.profile_ta:
        print("\nExisting pandas-ta-classic family timings for one symbol:")
        family_timings, indicator_timings = profile_pandas_ta_classic(args.rows, args.repeats)
        total = sum(item.seconds for item in family_timings)
        for item in sorted(family_timings, key=lambda value: value.seconds, reverse=True):
            share = 100.0 * item.seconds / total if total else 0.0
            print(f"{item.label}: {item.seconds:.3f}s ({share:.1f}%)")
        print(f"pandas-ta family total: {total:.3f}s")
        print("\nSlowest individual pandas-ta-classic calls:")
        for item in sorted(indicator_timings, key=lambda value: value.seconds, reverse=True)[:20]:
            print(f"{item.label}: {item.seconds:.3f}s")


if __name__ == "__main__":
    main()
