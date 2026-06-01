from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    import cupy as cp
except Exception:
    cp = None

try:
    from cupyx.scipy.special import ndtr as _cupy_ndtr
except Exception:
    _cupy_ndtr = None

try:
    from scipy.special import ndtr as _scipy_ndtr
except Exception:
    _scipy_ndtr = None


def cuda_available() -> bool:
    if cp is None:
        return False
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


CUDA_AVAILABLE = cuda_available()


def _norm_cdf_cpu(x):
    x_arr = np.asarray(x, dtype=float)
    if _scipy_ndtr is not None:
        return _scipy_ndtr(x_arr)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x_arr / np.sqrt(2.0)))


def _norm_cdf_gpu(x_gpu):
    if _cupy_ndtr is not None:
        return _cupy_ndtr(x_gpu)
    if cp is None:
        raise RuntimeError("CuPy is not available.")
    return 0.5 * (1.0 + cp.erf(x_gpu / cp.sqrt(2.0)))


def _black_scholes_panel(
    close_df,
    realized_vol_df,
    *,
    strike_multiplier,
    tenor_days=30,
    rate=0.0,
    iv_multiplier=1.0,
    premium_floor=0.25,
    option_type="call",
):
    tau = max(float(tenor_days) / 252.0, 1.0 / 252.0)
    m = float(strike_multiplier)
    sqrt_tau = math.sqrt(tau)
    log_term = math.log(1.0 / m)
    discount = math.exp(-float(rate) * tau)
    premium_floor = float(premium_floor)

    if CUDA_AVAILABLE and cp is not None:
        try:
            spot_gpu = cp.asarray(close_df.to_numpy(dtype=np.float32, copy=False))
            sigma_gpu = cp.asarray(realized_vol_df.to_numpy(dtype=np.float32, copy=False)) * np.float32(iv_multiplier)
            denom = sigma_gpu * np.float32(sqrt_tau)
            denom = cp.where(denom == 0.0, cp.nan, denom)
            d1 = (np.float32(log_term) + (np.float32(rate) + np.float32(0.5) * sigma_gpu * sigma_gpu) * np.float32(tau)) / denom
            d2 = d1 - sigma_gpu * np.float32(sqrt_tau)
            if str(option_type).lower() == "put":
                price_gpu = (spot_gpu * np.float32(m)) * np.float32(discount) * _norm_cdf_gpu(-d2) - spot_gpu * _norm_cdf_gpu(-d1)
                intrinsic_gpu = cp.maximum((spot_gpu * np.float32(m)) - spot_gpu, 0.0)
            else:
                price_gpu = spot_gpu * _norm_cdf_gpu(d1) - (spot_gpu * np.float32(m)) * np.float32(discount) * _norm_cdf_gpu(d2)
                intrinsic_gpu = cp.maximum(spot_gpu - (spot_gpu * np.float32(m)), 0.0)
            price_gpu = cp.where(cp.isfinite(price_gpu), price_gpu, intrinsic_gpu)
            price_gpu = cp.maximum(price_gpu, np.float32(premium_floor))
            return pd.DataFrame(cp.asnumpy(price_gpu), index=close_df.index, columns=close_df.columns)
        except Exception as exc:
            print(f"[cuda] Black-Scholes GPU fallback to CPU: {type(exc).__name__}: {exc}", flush=True)

    spot = close_df.astype(float)
    sigma = realized_vol_df.astype(float) * float(iv_multiplier)
    denom = (sigma * sqrt_tau).replace(0.0, np.nan)
    d1 = (log_term + (float(rate) + 0.5 * sigma * sigma) * tau) / denom
    d2 = d1 - sigma * sqrt_tau
    if str(option_type).lower() == "put":
        n1 = pd.DataFrame(_norm_cdf_cpu((-d1).to_numpy(dtype=float)), index=d1.index, columns=d1.columns)
        n2 = pd.DataFrame(_norm_cdf_cpu((-d2).to_numpy(dtype=float)), index=d2.index, columns=d2.columns)
        price = (spot * m) * discount * n2 - spot * n1
        intrinsic = ((spot * m) - spot).clip(lower=0.0)
    else:
        n1 = pd.DataFrame(_norm_cdf_cpu(d1.to_numpy(dtype=float)), index=d1.index, columns=d1.columns)
        n2 = pd.DataFrame(_norm_cdf_cpu(d2.to_numpy(dtype=float)), index=d2.index, columns=d2.columns)
        price = spot * n1 - (spot * m) * discount * n2
        intrinsic = (spot - (spot * m)).clip(lower=0.0)
    price = price.where(np.isfinite(price), intrinsic)
    return price.clip(lower=premium_floor)


def build_constant_maturity_call_price_panel(close_df, realized_vol_df, *, strike_multiplier, tenor_days=30, rate=0.0, iv_multiplier=1.0, premium_floor=0.25):
    return _black_scholes_panel(
        close_df,
        realized_vol_df,
        strike_multiplier=strike_multiplier,
        tenor_days=tenor_days,
        rate=rate,
        iv_multiplier=iv_multiplier,
        premium_floor=premium_floor,
        option_type="call",
    )


def build_constant_maturity_put_price_panel(close_df, realized_vol_df, *, strike_multiplier, tenor_days=30, rate=0.0, iv_multiplier=1.0, premium_floor=0.25):
    return _black_scholes_panel(
        close_df,
        realized_vol_df,
        strike_multiplier=strike_multiplier,
        tenor_days=tenor_days,
        rate=rate,
        iv_multiplier=iv_multiplier,
        premium_floor=premium_floor,
        option_type="put",
    )
