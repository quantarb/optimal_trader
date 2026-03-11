from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

# =====================================================================
# AE MANIFOLD + EVENT DIAGNOSTICS (Notebook source of truth)
# =====================================================================

def _resolve_schema_cols(ae_model) -> tuple[list[str] | None, list[str] | None]:
    """Resolve numeric/categorical schema from the AE artifact if present."""
    if getattr(ae_model, "_artifact", None) is not None:
        numeric_cols = list(ae_model._artifact.numeric_cols)
        categorical_cols = list(ae_model._artifact.cat_cols)
        return numeric_cols, categorical_cols
    return None, None


def _resolve_numeric_cols(ae_model) -> list[str]:
    numeric_cols, _ = _resolve_schema_cols(ae_model)
    if numeric_cols is None:
        raise ValueError("AE artifact schema is unavailable. Fit the AE before running diagnostics.")
    return numeric_cols


def _resolve_cols_for_ae(ae_model) -> tuple[list[str], list[str]]:
    numeric_cols, categorical_cols = _resolve_schema_cols(ae_model)
    if numeric_cols is None or categorical_cols is None:
        raise ValueError("AE artifact schema is unavailable. Fit the AE before running diagnostics.")
    return numeric_cols, categorical_cols


def _row_recon_mse(ae_model, df: pd.DataFrame) -> np.ndarray:
    numeric_cols, categorical_cols = _resolve_cols_for_ae(ae_model)
    return ae_model.recon_error(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols)


def _row_recon_mse_percentile(ae_model, df: pd.DataFrame) -> np.ndarray:
    numeric_cols, categorical_cols = _resolve_cols_for_ae(ae_model)
    return ae_model.recon_error_percentile(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols)


def _row_latent_dist(ae_model, df: pd.DataFrame) -> np.ndarray:
    numeric_cols, categorical_cols = _resolve_cols_for_ae(ae_model)
    return ae_model.latent_distance(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols)


def _row_latent_dist_percentile(ae_model, df: pd.DataFrame) -> np.ndarray:
    numeric_cols, categorical_cols = _resolve_cols_for_ae(ae_model)
    return ae_model.latent_distance_percentile(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols)


def _row_latent_mahal(ae_model, df: pd.DataFrame) -> np.ndarray:
    numeric_cols, categorical_cols = _resolve_cols_for_ae(ae_model)
    return ae_model.latent_mahalanobis_distance(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols)


def _row_latent_mahal_percentile(ae_model, df: pd.DataFrame) -> np.ndarray:
    numeric_cols, categorical_cols = _resolve_cols_for_ae(ae_model)
    return ae_model.latent_mahalanobis_percentile(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols)


def _print_model_architecture(ae_model) -> None:
    if getattr(ae_model, "_artifact", None) is None or getattr(ae_model._artifact, "model", None) is None:
        print("Model architecture unavailable (AE not fit).")
        return
    model = ae_model._artifact.model
    enc = [layer for layer in model.encoder if isinstance(layer, nn.Linear)]
    dec = [layer for layer in model.decoder if isinstance(layer, nn.Linear)]
    enc_dims = [f"{layer.in_features}->{layer.out_features}" for layer in enc]
    dec_dims = [f"{layer.in_features}->{layer.out_features}" for layer in dec]
    print("AE architecture:")
    print(f"  - Encoder linear dims: {enc_dims}")
    print(f"  - Decoder linear dims: {dec_dims}")
    layer_dims = getattr(model, "layer_dims", None)
    if layer_dims is not None:
        print(f"  - Halving dims: {list(layer_dims)}")

# ------------------------------------------------------------
# Helpers: standardize + reconstruct
# ------------------------------------------------------------
def compute_global_and_feature_mse(
    ae_model,
    df: pd.DataFrame,
    label: str,
) -> tuple[float, pd.Series]:
    numeric_cols, categorical_cols = _resolve_cols_for_ae(ae_model)
    err_df = ae_model.recon_error_matrix(
        df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        space="standardized",
    )
    global_mse = float(err_df.to_numpy(dtype=np.float64, copy=False).mean())
    feat_mse = err_df.mean(axis=0).reindex(numeric_cols).sort_values(ascending=False)
    return global_mse, feat_mse

def compute_per_day_feature_error(
    ae_model,
    panel_df: pd.DataFrame,
) -> pd.DataFrame:
    # expects 'date' in index or column as in notebook
    df = panel_df.copy()
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"])
    elif isinstance(df.index, pd.MultiIndex) and "date" in df.index.names:
        dates = pd.to_datetime(df.index.get_level_values("date"))
    elif df.index.name == "date":
        dates = pd.to_datetime(df.index)
    else:
        raise ValueError("panel_df must have 'date' in index or as a column.")

    numeric_cols, categorical_cols = _resolve_cols_for_ae(ae_model)
    err_df = ae_model.recon_error_matrix(
        df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        space="standardized",
    ).reindex(columns=numeric_cols)
    err_df = err_df.copy()
    err_df["__date__"] = dates.values
    per_day = err_df.groupby("__date__").mean(numeric_only=True)
    per_day.index.name = "date"
    return per_day

def compute_regime_series(ae_model, panel_df: pd.DataFrame, smooth_window: int = 20) -> pd.Series:
    numeric_cols, categorical_cols = _resolve_schema_cols(ae_model)
    if numeric_cols is None or categorical_cols is None:
        raise ValueError("AE artifact schema is unavailable. Fit the AE before running diagnostics.")
    row_err = ae_model.recon_error(
        panel_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
    # aggregate by date
    if "date" in panel_df.columns:
        dates = pd.to_datetime(panel_df["date"])
    elif isinstance(panel_df.index, pd.MultiIndex) and "date" in panel_df.index.names:
        dates = pd.to_datetime(panel_df.index.get_level_values("date"))
    elif panel_df.index.name == "date":
        dates = pd.to_datetime(panel_df.index)
    else:
        raise ValueError("panel_df must have 'date' in index or as a column.")

    s = pd.Series(np.asarray(row_err).reshape(-1), index=dates)
    s = s.groupby(level=0).mean().sort_index()
    return s.rolling(smooth_window, min_periods=1).mean()

def plot_regime_full_and_zoom(regime_s: pd.Series, title: str = "AE Regime / Familiarity", zoom_years: int = 5) -> None:
    fig = plt.figure(figsize=(12, 3))
    plt.plot(regime_s.index, regime_s.values)
    plt.title(title)
    plt.tight_layout()
    plt.show()

    if zoom_years is not None and zoom_years > 0:
        cutoff = regime_s.index.max() - pd.DateOffset(years=int(zoom_years))
        zoom = regime_s[regime_s.index >= cutoff]
        fig = plt.figure(figsize=(12, 3))
        plt.plot(zoom.index, zoom.values)
        plt.title(f"{title} (last {zoom_years}y)")
        plt.tight_layout()
        plt.show()

def summarize_event_windows(
    per_day_feat_err: pd.DataFrame,
    events: dict[str, str],
    window_days: int = 20,
    topk: int = 25,
) -> pd.DataFrame:
    rows = []
    for date_str, label in events.items():
        center = pd.to_datetime(date_str)
        win = per_day_feat_err.loc[
            (per_day_feat_err.index >= center - pd.Timedelta(days=window_days)) &
            (per_day_feat_err.index <= center + pd.Timedelta(days=window_days))
        ]
        if len(win) == 0:
            continue

        avg = win.mean().sort_values(ascending=False).head(int(topk))
        tmp = avg.reset_index()
        tmp.columns = ["feature", "event_feat_mse"]
        tmp.insert(0, "event", label)
        tmp.insert(1, "event_date", center.date().isoformat())
        tmp.insert(2, "window_days_each_side", int(window_days))
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["event","event_date","window_days_each_side","feature","event_feat_mse"])
    return pd.concat(rows, axis=0, ignore_index=True)

def analyze_event_feature_breaks(
    per_day_feat_err: pd.DataFrame,
    baseline_feat_mse: pd.Series,
    events: dict[str, str],
    window_days: int = 20,
    topk: int = 25,
) -> pd.DataFrame:
    rows = []
    for date_str, label in events.items():
        center = pd.to_datetime(date_str)
        win = per_day_feat_err.loc[
            (per_day_feat_err.index >= center - pd.Timedelta(days=window_days)) &
            (per_day_feat_err.index <= center + pd.Timedelta(days=window_days))
        ]
        if len(win) == 0:
            continue

        event_mse = win.mean().sort_values(ascending=False).head(int(topk))
        base = baseline_feat_mse.reindex(event_mse.index)

        out = pd.DataFrame({
            "feature": event_mse.index,
            "event_feat_mse": event_mse.values,
            "baseline_feat_mse": base.values,
        })
        out["ratio_event_over_baseline"] = out["event_feat_mse"] / (out["baseline_feat_mse"] + 1e-9)
        out["diff_event_minus_baseline"] = out["event_feat_mse"] - out["baseline_feat_mse"]

        out = out.sort_values("ratio_event_over_baseline", ascending=False)

        out.insert(0, "event", label)
        out.insert(1, "event_date", center.date().isoformat())
        out.insert(2, "window_days_each_side", int(window_days))
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=[
            "event", "event_date", "window_days_each_side",
            "feature", "event_feat_mse", "baseline_feat_mse",
            "ratio_event_over_baseline", "diff_event_minus_baseline"
        ])
    return pd.concat(rows, axis=0, ignore_index=True)

def run_ae_manifold_event_diagnostics(
    *,
    ae_model,
    train_data: pd.DataFrame,
    X_non_optimal: pd.DataFrame,
    smooth_window: int = 20,
    window_days: int = 20,
    topk: int = 25,
    events: dict[str, str] | None = None,
) -> dict[str, object]:
    """Notebook cell wrapper: runs the AE manifold + event diagnostics and returns artifacts.

    Returns a dict with:
      - mse_non, mse_opt, mse_feat_non, mse_feat_opt
      - per_day_feat_err, regime_series, event_summary_df, event_breaks_df
    """
    print("\n--- AE MANIFOLD + EVENT DIAGNOSTICS (AE TRAINED ON OPTIMAL TRADES ONLY) ---")
    print(
        "High-level goal:\n"
        "1) Check whether the AE's learned feature space for OPTIMAL trades differs from 'normal' (non-optimal) days.\n"
        "2) If it differs, identify WHICH FEATURES differ most (the profitable manifold drivers).\n"
        "3) For extreme macro events, identify WHICH PARTS of feature space 'broke' relative to the optimal baseline.\n\n"
        "Context:\n"
        "- This autoencoder (AE) was trained ONLY on OPTIMAL-TRADE rows.\n"
        "- Therefore, reconstruction error is a 'familiarity score' vs profitable setups:\n"
        "  * Low error  -> today's feature relationships resemble historically profitable setups.\n"
        "  * High error -> today's feature relationships do NOT resemble historically profitable setups.\n"
        "- CRITICAL: We compute ALL errors in STANDARDIZED feature space to avoid unit/scale artifacts.\n"
    )

    if events is None:
        events = {
            "2000-03-10": "Dot-Com Peak",
            "2001-09-17": "Post-9/11 Reopen",
            "2008-09-15": "Lehman Collapse (GFC)",
            "2009-03-09": "GFC Market Bottom",
            "2010-05-06": "Flash Crash",
            "2011-08-08": "US Credit Downgrade",
            "2015-08-24": "China Devaluation Crash",
            "2018-02-05": "Volmageddon",
            "2020-03-20": "Covid Crash",
            "2020-11-09": "Vaccine Announcement",
            "2021-11-01": "Tech Top",
            "2022-01-01": "Rate Hike Regime",
            "2022-10-13": "Inflation Peak Pivot",
            "2023-01-01": "AI Rally",
            "2023-10-27": "Bond Yield Spike",
        }

    print("\n--- STEP 1) MODEL-NATIVE RECON ERROR PATH ---")
    print("Using ae_model internal fit-time transforms for both optimal and non-optimal sets.")
    _print_model_architecture(ae_model)
    n_opt = int(len(train_data))
    n_non = int(len(X_non_optimal))
    ratio_non_opt = (n_non / n_opt) if n_opt > 0 else np.nan
    print(
        f"Dataset sizes: optimal_train={n_opt:,} rows | "
        f"non_optimal={n_non:,} rows | non/opt={ratio_non_opt:.2f}x"
    )

    print("--- STEP 2) DOES OPTIMAL FEATURE SPACE DIFFER FROM NORMAL DAYS? ---")
    mse_non, mse_feat_non = compute_global_and_feature_mse(ae_model, X_non_optimal, "non_optimal")
    mse_opt, mse_feat_opt = compute_global_and_feature_mse(ae_model, train_data, "optimal_train")

    row_mse_non = _row_recon_mse(ae_model, X_non_optimal)
    row_mse_opt = _row_recon_mse(ae_model, train_data)
    row_pct_non = _row_recon_mse_percentile(ae_model, X_non_optimal)
    row_pct_opt = _row_recon_mse_percentile(ae_model, train_data)
    row_lat_non = _row_latent_dist(ae_model, X_non_optimal)
    row_lat_opt = _row_latent_dist(ae_model, train_data)
    row_lat_pct_non = _row_latent_dist_percentile(ae_model, X_non_optimal)
    row_lat_pct_opt = _row_latent_dist_percentile(ae_model, train_data)
    row_lat_mahal_non = _row_latent_mahal(ae_model, X_non_optimal)
    row_lat_mahal_opt = _row_latent_mahal(ae_model, train_data)
    row_lat_mahal_pct_non = _row_latent_mahal_percentile(ae_model, X_non_optimal)
    row_lat_mahal_pct_opt = _row_latent_mahal_percentile(ae_model, train_data)

    # Alpha-weighted MSE: weights from feature divergence ratio (non/opt).
    alpha_weights = (mse_feat_non / (mse_feat_opt + 1e-9)).clip(lower=0.0)
    numeric_cols, categorical_cols = _resolve_cols_for_ae(ae_model)
    row_alpha_non = ae_model.weighted_recon_error(
        X_non_optimal,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        feature_weights=alpha_weights,
    )
    row_alpha_opt = ae_model.weighted_recon_error(
        train_data,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        feature_weights=alpha_weights,
    )
    # Fracture score: top 5% largest feature errors per row.
    row_fracture_non = ae_model.fracture_score(
        X_non_optimal,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        top_frac=0.05,
    )
    row_fracture_opt = ae_model.fracture_score(
        train_data,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        top_frac=0.05,
    )
    # Precision-scaled MSE: inverse optimal residual variance weights.
    row_precision_non = ae_model.precision_scaled_recon_error(
        X_non_optimal,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
    row_precision_opt = ae_model.precision_scaled_recon_error(
        train_data,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    ratio_global = mse_non / (mse_opt + 1e-9)
    print(f"\nGlobal standardized recon MSE (non-optimal):   {mse_non:.6f}")
    print(f"Global standardized recon MSE (optimal-train): {mse_opt:.6f}")
    print(f"Ratio non/opt: {ratio_global:.3f}x")
    print(
        "Row recon-error score stats [p50/p90/p99] "
        f"non-opt={np.percentile(row_mse_non, [50, 90, 99])} | "
        f"opt={np.percentile(row_mse_opt, [50, 90, 99])}"
    )
    print(
        "Row recon-error percentile stats [p50/p90/p99] "
        f"non-opt={np.percentile(row_pct_non, [50, 90, 99])} | "
        f"opt={np.percentile(row_pct_opt, [50, 90, 99])}"
    )
    print(
        "Row latent-distance stats [p50/p90/p99] "
        f"non-opt={np.percentile(row_lat_non, [50, 90, 99])} | "
        f"opt={np.percentile(row_lat_opt, [50, 90, 99])}"
    )
    print(
        "Row latent-distance percentile stats [p50/p90/p99] "
        f"non-opt={np.percentile(row_lat_pct_non, [50, 90, 99])} | "
        f"opt={np.percentile(row_lat_pct_opt, [50, 90, 99])}"
    )
    print(
        "Row latent-Mahalanobis stats [p50/p90/p99] "
        f"non-opt={np.percentile(row_lat_mahal_non, [50, 90, 99])} | "
        f"opt={np.percentile(row_lat_mahal_opt, [50, 90, 99])}"
    )
    print(
        "Row latent-Mahalanobis percentile stats [p50/p90/p99] "
        f"non-opt={np.percentile(row_lat_mahal_pct_non, [50, 90, 99])} | "
        f"opt={np.percentile(row_lat_mahal_pct_opt, [50, 90, 99])}"
    )
    print(
        "Row alpha-weighted MSE stats [p50/p90/p99] "
        f"non-opt={np.percentile(row_alpha_non, [50, 90, 99])} | "
        f"opt={np.percentile(row_alpha_opt, [50, 90, 99])}"
    )
    print(
        "Row fracture score (top 5%) stats [p50/p90/p99] "
        f"non-opt={np.percentile(row_fracture_non, [50, 90, 99])} | "
        f"opt={np.percentile(row_fracture_opt, [50, 90, 99])}"
    )
    print(
        "Row precision-scaled MSE stats [p50/p90/p99] "
        f"non-opt={np.percentile(row_precision_non, [50, 90, 99])} | "
        f"opt={np.percentile(row_precision_opt, [50, 90, 99])}"
    )

    print("\n--- STEP 3) FEATURE-LEVEL DIFFERENCE (non-opt vs optimal) ---")
    compare = pd.DataFrame({
        "mse_non_opt": mse_feat_non,
        "mse_opt_train": mse_feat_opt.reindex(mse_feat_non.index),
    })
    compare["ratio_non_over_opt"] = compare["mse_non_opt"] / (compare["mse_opt_train"] + 1e-9)
    compare = compare.sort_values("ratio_non_over_opt", ascending=False)

    print(compare.head(25))

    print("\n--- STEP 4) PER-DAY FEATURE ERROR (for EVENT BREAKDOWN) ---")
    per_day_feat_err = compute_per_day_feature_error(ae_model, X_non_optimal)

    print("\n--- STEP 5) REGIME SERIES (SMOOTHED) ---")
    regime_s = compute_regime_series(ae_model, X_non_optimal, smooth_window=smooth_window)
    plot_regime_full_and_zoom(regime_s, title="AE Recon Error Regime (Smoothed)", zoom_years=5)

    print("\n--- STEP 6) EVENT WINDOW FEATURE SUMMARY ---")
    event_summary_df = summarize_event_windows(per_day_feat_err, events=events, window_days=window_days, topk=topk)
    display(event_summary_df.head(50))

    print("\n--- STEP 7) EVENT FEATURE BREAKS VS OPTIMAL BASELINE ---")
    event_breaks_df = analyze_event_feature_breaks(per_day_feat_err, baseline_feat_mse=mse_feat_opt, events=events, window_days=window_days, topk=topk)
    display(event_breaks_df.head(50))

    return dict(
        mse_non=mse_non,
        mse_opt=mse_opt,
        mse_feat_non=mse_feat_non,
        mse_feat_opt=mse_feat_opt,
        row_mse_non=row_mse_non,
        row_mse_opt=row_mse_opt,
        row_mse_pct_non=row_pct_non,
        row_mse_pct_opt=row_pct_opt,
        row_latent_non=row_lat_non,
        row_latent_opt=row_lat_opt,
        row_latent_pct_non=row_lat_pct_non,
        row_latent_pct_opt=row_lat_pct_opt,
        row_latent_mahal_non=row_lat_mahal_non,
        row_latent_mahal_opt=row_lat_mahal_opt,
        row_latent_mahal_pct_non=row_lat_mahal_pct_non,
        row_latent_mahal_pct_opt=row_lat_mahal_pct_opt,
        row_alpha_mse_non=row_alpha_non,
        row_alpha_mse_opt=row_alpha_opt,
        row_fracture_non=row_fracture_non,
        row_fracture_opt=row_fracture_opt,
        row_precision_mse_non=row_precision_non,
        row_precision_mse_opt=row_precision_opt,
        alpha_weights=alpha_weights,
        per_day_feat_err=per_day_feat_err,
        regime_series=regime_s,
        event_summary_df=event_summary_df,
        event_breaks_df=event_breaks_df,
        compare_df=compare,
    )


def build_non_optimal_set(
    *,
    final_df: pd.DataFrame,
    train_data: pd.DataFrame,
) -> pd.DataFrame:
    """Build non-optimal universe rows as final_df minus optimal-train index."""
    boring_indices = final_df.index.difference(train_data.index)
    return final_df.loc[boring_indices]


def fit_and_run_ae_diagnostics(
    *,
    ae_model,
    train_data: pd.DataFrame,
    final_df: pd.DataFrame,
    spec_ae,
    numeric_cols: list[str],
    categorical_cols: list[str] | tuple[str, ...] = (),
    cfg_updates: dict | None = None,
    verbose_fit: bool = False,
    smooth_window: int = 20,
    window_days: int = 20,
    topk: int = 25,
    events: dict[str, str] | None = None,
):
    """Refit AE from existing cfg + updates, build non-opt set, then run diagnostics."""
    from .config import AutoEncoderConfig
    from .adapter import TorchAutoEncoder

    old_cfg = ae_model.cfg.model_dump() if hasattr(ae_model.cfg, "model_dump") else dict(ae_model.cfg.__dict__)
    cfg = AutoEncoderConfig(**old_cfg)
    if cfg_updates:
        cfg = cfg.model_copy(update=dict(cfg_updates))

    new_ae = TorchAutoEncoder(cfg=cfg)
    new_ae.fit(
        train_data,
        spec_ae,
        numeric_cols=numeric_cols,
        categorical_cols=list(categorical_cols),
        verbose=verbose_fit,
    )

    x_non = build_non_optimal_set(final_df=final_df, train_data=train_data)
    diag = run_ae_manifold_event_diagnostics(
        ae_model=new_ae,
        train_data=train_data,
        X_non_optimal=x_non,
        smooth_window=smooth_window,
        window_days=window_days,
        topk=topk,
        events=events,
    )
    return new_ae, x_non, diag
