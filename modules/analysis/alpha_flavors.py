from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
try:
    import hdbscan as _hdbscan  # type: ignore
except Exception:
    _hdbscan = None
from modules.models.torch.autoencoder.vector_db import (
    LatentVectorDB,
    build_latent_vector_db,
    build_latent_vector_frame,
    explain_cluster_ae_feature_breaks,
    explain_cluster_feature_uniqueness,
)


@dataclass(frozen=True)
class AlphaFlavorClusterConfig:
    cluster_col: str = "cluster_sym_ym"
    ret_col: str = "trade_return"
    cluster_algo: str = "kmeans"  # "kmeans" | "hdbscan"
    auto_k: bool = True
    k_min: int = 10
    k_max: int = 50
    k_step: int = 5
    random_state: int = 1337
    sample_size_for_auto_k: int = 25_000
    auto_k_method: str = "elbow"   # "elbow" (cheap) or "silhouette" (expensive)
    # If <=1.0, interpreted as fraction of rows (e.g., 0.25 => 25%).
    # If >1.0, interpreted as absolute row count.
    fit_sample_size: float = 0.35  # fit kmeans on sample, then assign all rows by centroid distance
    batch_size: int = 4096
    top_feature_count: int = 25
    include_ae_breaks: bool = False  # expensive: computes recon-error explainability per cluster
    distance_temperature: float = 1.0
    distance_metric: str = "euclidean"
    hdbscan_min_cluster_size: int = 100
    hdbscan_min_samples: Optional[int] = None


@dataclass(frozen=True)
class AlphaFlavorSpace:
    vector_db: LatentVectorDB
    centroids: np.ndarray
    cluster_labels: pd.Series
    cluster_col: str
    latent_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    selected_k: int
    k_scores_df: pd.DataFrame
    clustered_df: pd.DataFrame
    summary_df: pd.DataFrame
    packets: list[dict[str, Any]]
    distance_metric: str
    distance_temperature: float
    cluster_feature_stats: dict[int, dict[str, dict[str, float]]]


@dataclass(frozen=True)
class AlphaFlavorClusterResult:
    df: pd.DataFrame
    packets: list[dict[str, Any]]
    summary_df: pd.DataFrame
    selected_k: int
    k_scores_df: pd.DataFrame
    flavor_space: Optional[AlphaFlavorSpace] = None


def _sanitize_x(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e3, neginf=-1e3)
    return np.clip(arr, a_min=-1e3, a_max=1e3).astype(np.float32, copy=False)


def _resolve_sample_count(n_rows: int, sample_spec: float) -> int:
    if n_rows <= 0:
        return 0
    try:
        s = float(sample_spec)
    except Exception:
        s = float(n_rows)
    if s <= 0:
        return min(n_rows, 1)
    if s <= 1.0:
        return int(max(1, min(n_rows, np.floor(n_rows * s))))
    return int(max(1, min(n_rows, np.floor(s))))


def _resolve_date_symbol(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    date_s: Optional[pd.Series] = None
    sym_s: Optional[pd.Series] = None

    if "date" in df.columns:
        date_s = pd.to_datetime(df["date"], errors="coerce")
    elif isinstance(df.index, pd.MultiIndex) and "date" in (df.index.names or []):
        date_s = pd.Series(pd.to_datetime(df.index.get_level_values("date"), errors="coerce"), index=df.index)
    elif isinstance(df.index, pd.DatetimeIndex):
        date_s = pd.Series(pd.to_datetime(df.index, errors="coerce"), index=df.index)

    if "symbol" in df.columns:
        sym_s = df["symbol"].astype(str)
    elif isinstance(df.index, pd.MultiIndex) and "symbol" in (df.index.names or []):
        sym_s = pd.Series(df.index.get_level_values("symbol").astype(str), index=df.index)

    if date_s is None:
        date_s = pd.Series(pd.NaT, index=df.index)
    if sym_s is None:
        sym_s = pd.Series("", index=df.index)
    return date_s, sym_s


def _extract_symbol_series(df: pd.DataFrame) -> pd.Series:
    if "symbol" in df.columns:
        return df["symbol"].astype(str)
    if isinstance(df.index, pd.MultiIndex) and "symbol" in (df.index.names or []):
        return pd.Series(df.index.get_level_values("symbol").astype(str), index=df.index)
    return pd.Series("", index=df.index, dtype="object")


def _extract_date_series(df: pd.DataFrame) -> pd.Series:
    if "date" in df.columns:
        return pd.to_datetime(df["date"], errors="coerce")
    if isinstance(df.index, pd.MultiIndex) and "date" in (df.index.names or []):
        return pd.Series(pd.to_datetime(df.index.get_level_values("date"), errors="coerce"), index=df.index)
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(pd.to_datetime(df.index, errors="coerce"), index=df.index)
    return pd.Series(pd.NaT, index=df.index)


def _auto_select_k(x: np.ndarray, cfg: AlphaFlavorClusterConfig, *, verbose: bool) -> tuple[int, pd.DataFrame]:
    n = x.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 rows to cluster.")

    lo = max(2, int(cfg.k_min))
    hi = max(lo, min(int(cfg.k_max), n - 1))
    step = max(1, int(cfg.k_step))
    k_grid = list(range(lo, hi + 1, step))
    if hi not in k_grid:
        k_grid.append(hi)
    k_grid = sorted(set(k_grid))

    if cfg.sample_size_for_auto_k and n > int(cfg.sample_size_for_auto_k):
        rng = np.random.default_rng(int(cfg.random_state))
        pick = rng.choice(n, size=int(cfg.sample_size_for_auto_k), replace=False)
        xs = x[pick]
    else:
        xs = x

    method = str(cfg.auto_k_method).lower().strip()
    if verbose:
        print(f"[AUTO_K] method={method} searching k={lo}..{hi} step={step} on sample n={len(xs):,}\n")

    rows: list[dict[str, Any]] = []
    if method == "silhouette":
        # Guardrail: silhouette is O(n^2) memory/time. Keep sample modest.
        max_sil_n = 5000
        if len(xs) > max_sil_n:
            rng = np.random.default_rng(int(cfg.random_state) + 7)
            pick = rng.choice(len(xs), size=max_sil_n, replace=False)
            xs_eval = xs[pick]
        else:
            xs_eval = xs
        best_k = k_grid[0]
        best_s = -np.inf
        for k in k_grid:
            km = MiniBatchKMeans(
                n_clusters=int(k),
                random_state=int(cfg.random_state),
                batch_size=int(max(256, cfg.batch_size)),
                n_init="auto",
                init="k-means++",
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*encountered in matmul.*")
                lab = km.fit_predict(xs_eval)
                if len(np.unique(lab)) < 2:
                    s = float("nan")
                else:
                    s = float(silhouette_score(xs_eval, lab))
            rows.append({"k": int(k), "silhouette": s, "inertia": float(km.inertia_)})
            if verbose:
                print(f"  - k={k} | silhouette={s:.5f}")
            if np.isfinite(s) and s > best_s:
                best_s = s
                best_k = int(k)
        if verbose:
            print(f"[AUTO_K] chose k={best_k} (best silhouette={best_s:.5f})")
        return best_k, pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

    # Cheap default: inertia elbow (MiniBatchKMeans only).
    for k in k_grid:
        km = MiniBatchKMeans(
            n_clusters=int(k),
            random_state=int(cfg.random_state),
            batch_size=int(max(256, cfg.batch_size)),
            n_init="auto",
            init="k-means++",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*encountered in matmul.*")
            km.fit(xs)
        rows.append({"k": int(k), "silhouette": np.nan, "inertia": float(km.inertia_)})
        if verbose:
            print(f"  - k={k} | inertia={float(km.inertia_):.4f}")

    scan = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    if len(scan) == 1:
        return int(scan.loc[0, "k"]), scan

    ks = scan["k"].to_numpy(dtype=np.float64)
    inertias = scan["inertia"].to_numpy(dtype=np.float64)
    x_norm = (ks - ks.min()) / max(1e-12, (ks.max() - ks.min()))
    y_norm = (inertias - inertias.min()) / max(1e-12, (inertias.max() - inertias.min()))
    d = y_norm - (1.0 - x_norm)  # kneedle-like score
    best_idx = int(np.argmax(d))
    best_k = int(scan.loc[best_idx, "k"])
    if verbose:
        print(f"[AUTO_K] chose k={best_k} (elbow on inertia)")
    return best_k, scan


def _cluster_once(x: np.ndarray, k: int, cfg: AlphaFlavorClusterConfig) -> np.ndarray:
    km = MiniBatchKMeans(
        n_clusters=int(k),
        random_state=int(cfg.random_state),
        batch_size=int(max(256, cfg.batch_size)),
        n_init="auto",
        init="k-means++",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*encountered in matmul.*")
        labels = km.fit_predict(x)
    return np.asarray(labels, dtype=int)


def _feature_distinctiveness(
    x_scaled_df: pd.DataFrame,
    labels: pd.Series,
    cluster_id: int,
    top_n: int,
) -> list[tuple[str, float]]:
    x = x_scaled_df.join(labels.rename("__cluster__"), how="inner")
    if x.empty:
        return []
    cols = [c for c in x_scaled_df.columns if c in x.columns and pd.api.types.is_numeric_dtype(x[c])]
    if not cols:
        return []
    in_mean = x.loc[x["__cluster__"] == int(cluster_id), cols].mean()
    global_mean = x[cols].mean()
    sigma_delta = (in_mean - global_mean).sort_values(key=lambda s: s.abs(), ascending=False)
    return [(str(c), float(v)) for c, v in sigma_delta.head(int(top_n)).items()]


def _pairwise_distance(
    x: np.ndarray,
    centroids: np.ndarray,
    *,
    metric: str,
) -> np.ndarray:
    m = str(metric).lower().strip()
    xa = np.asarray(x, dtype=np.float64)
    cb = np.asarray(centroids, dtype=np.float64)
    xa = np.nan_to_num(xa, nan=0.0, posinf=1e3, neginf=-1e3)
    cb = np.nan_to_num(cb, nan=0.0, posinf=1e3, neginf=-1e3)
    xa = np.clip(xa, a_min=-1e3, a_max=1e3)
    cb = np.clip(cb, a_min=-1e3, a_max=1e3)

    if m == "cosine":
        x_norm = np.linalg.norm(xa, axis=1, keepdims=True)
        c_norm = np.linalg.norm(cb, axis=1, keepdims=True).T
        denom = np.clip(x_norm * c_norm, a_min=1e-12, a_max=None)
        sim = (xa @ cb.T) / denom
        dist = 1.0 - sim
        return dist.astype(np.float32, copy=False)

    # default euclidean (stable, avoids large matmul warnings)
    n = xa.shape[0]
    k = cb.shape[0]
    d2 = np.empty((n, k), dtype=np.float64)
    for j in range(k):
        diff = xa - cb[j]
        d2[:, j] = np.sum(diff * diff, axis=1)
    d2 = np.clip(d2, a_min=0.0, a_max=None)
    return np.sqrt(d2).astype(np.float32, copy=False)


def _distance_signature(
    x: np.ndarray,
    centroids: np.ndarray,
    *,
    metric: str,
    temperature: float,
    cluster_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    d = _pairwise_distance(x, centroids, metric=metric)
    k = d.shape[1]
    dist_cols = [f"{cluster_col}__dist_{i}" for i in range(k)]
    prob_cols = [f"{cluster_col}__prob_{i}" for i in range(k)]

    t = max(1e-6, float(temperature))
    logits = -d / t
    logits = logits - np.max(logits, axis=1, keepdims=True)
    ex = np.exp(logits)
    probs = ex / np.clip(np.sum(ex, axis=1, keepdims=True), a_min=1e-12, a_max=None)

    dist_df = pd.DataFrame(d, columns=dist_cols)
    prob_df = pd.DataFrame(probs, columns=prob_cols)
    top = pd.Series(np.argmax(probs, axis=1).astype(int), name=cluster_col)
    return dist_df, prob_df, top


def cluster_alpha_flavors(
    *,
    df: pd.DataFrame,
    X_scaled_df: Optional[pd.DataFrame] = None,
    Z_latent: Any | None = None,
    flavor_space: Optional[AlphaFlavorSpace] = None,
    ae_model: Any | None = None,
    numeric_cols: Optional[list[str]] = None,
    categorical_cols: Optional[list[str]] = None,
    cfg: Optional[AlphaFlavorClusterConfig] = None,
    verbose: bool = True,
) -> AlphaFlavorClusterResult:
    """
    Compatibility implementation for old notebook alpha-flavor clustering API.
    """
    conf = cfg or AlphaFlavorClusterConfig()
    base = df.copy()
    cat_cols = list(categorical_cols or [])
    num_cols = list(numeric_cols or [])
    if flavor_space is not None:
        # Reuse already-fitted flavor space / vector DB; only enrich current dataframe.
        if verbose:
            print(f"[FLAVOR SPACE] Reusing pre-fit space with K={flavor_space.selected_k}.")
        cluster_col = str(flavor_space.cluster_col)
        k_sel = int(flavor_space.selected_k)
        k_scores = flavor_space.k_scores_df.copy()
        vector_db = flavor_space.vector_db
        centroids = _sanitize_x(flavor_space.centroids)
        latent_cols = list(flavor_space.latent_cols)
        num_cols = list(flavor_space.numeric_cols)
        cat_cols = list(flavor_space.categorical_cols)

        # Pull precomputed assignment where possible.
        scored_source = flavor_space.clustered_df
        assign_cols = [cluster_col, f"{cluster_col}__min_dist", f"{cluster_col}__margin"]
        assign_cols += [c for c in scored_source.columns if c.startswith(f"{cluster_col}__dist_")]
        assign_cols += [c for c in scored_source.columns if c.startswith(f"{cluster_col}__prob_")]
        assign_cols = [c for c in assign_cols if c in scored_source.columns]
        cached = pd.DataFrame(index=base.index)
        if assign_cols:
            if scored_source.index.equals(base.index):
                cached = scored_source[assign_cols].copy()
            elif scored_source.index.is_unique and base.index.is_unique:
                cached = scored_source[assign_cols].reindex(base.index).copy()
            elif len(scored_source) == len(base):
                # Positional fallback for duplicated MultiIndex shapes.
                cached = scored_source[assign_cols].copy()
                cached.index = base.index

        # For rows not present in cached space, score by distance signature using current AE.
        need = cached.empty or cached[cluster_col].isna().any() if cluster_col in cached.columns else True
        if need:
            if ae_model is None:
                raise ValueError("ae_model is required to score rows not present in flavor_space.")
            if cluster_col not in cached.columns:
                missing_idx = base.index
            else:
                missing_mask = cached[cluster_col].isna().to_numpy(dtype=bool)
                missing_idx = base.index[missing_mask]
            if len(missing_idx):
                scored_new = score_trade_flavor(
                    flavor_space=flavor_space,
                    entries_df=base.loc[missing_idx],
                    ae_model=ae_model,
                )
                if cached.empty:
                    cached = scored_new[[c for c in scored_new.columns if c.startswith(f"{cluster_col}__") or c == cluster_col]]
                else:
                    for c in [col for col in scored_new.columns if col in cached.columns or col.startswith(f"{cluster_col}__") or col == cluster_col]:
                        cached.loc[missing_idx, c] = scored_new[c]
        base = base.join(cached, how="left")
        if cluster_col not in base.columns:
            raise RuntimeError("Failed to assign cluster labels from flavor_space.")
        top_cluster = pd.Series(pd.to_numeric(base[cluster_col], errors="coerce").fillna(-1).astype(int), index=base.index, name=cluster_col)
    else:
        # Build latent vectors with model-native transform where possible.
        if ae_model is not None and num_cols:
            latent_df = build_latent_vector_frame(
                ae_model,
                base,
                numeric_cols=num_cols,
                categorical_cols=cat_cols,
                include_columns=[str(conf.ret_col), "trade_duration_days"],
                latent_prefix="z_",
            )
        else:
            if Z_latent is None:
                raise ValueError("Provide Z_latent, or provide ae_model + numeric_cols to compute embeddings.")
            z_raw = _sanitize_x(np.asarray(Z_latent))
            if z_raw.ndim != 2:
                raise ValueError(f"Z_latent must be 2D, got shape={z_raw.shape}.")
            if len(base) != z_raw.shape[0]:
                raise ValueError(f"Row mismatch: len(df)={len(base)} vs Z_latent rows={z_raw.shape[0]}.")
            latent_cols = [f"z_{i}" for i in range(z_raw.shape[1])]
            latent_df = pd.DataFrame(z_raw, index=base.index, columns=latent_cols)

        latent_cols = [c for c in latent_df.columns if str(c).startswith("z_")]
        z = _sanitize_x(latent_df[latent_cols].to_numpy(dtype=np.float32, copy=False))
        absmax = float(np.nanmax(np.abs(z))) if z.size else float("nan")
        q99abs = float(np.nanpercentile(np.abs(z), 99.0)) if z.size else float("nan")
        finite = float(np.isfinite(z).mean()) if z.size else float("nan")
        if verbose:
            print(f"[CLUSTER SANITY] X shape={z.shape} absmax={absmax:.4f} q99abs={q99abs:.4f} finite={finite:.6f}")

        vector_db = build_latent_vector_db(latent_df, latent_cols=latent_cols, metric="cosine", n_neighbors=50)

        cluster_col = str(conf.cluster_col)
        algo = str(conf.cluster_algo).lower().strip()
        z_fit = z
        fit_n = _resolve_sample_count(len(z), conf.fit_sample_size)
        if fit_n < len(z):
            rng = np.random.default_rng(int(conf.random_state) + 101)
            pick = rng.choice(len(z), size=int(fit_n), replace=False)
            z_fit = z[pick]

        if algo == "hdbscan":
            if _hdbscan is None:
                raise ImportError(
                    "hdbscan is not installed. Install with `pip install hdbscan` "
                    "or set cluster_algo='kmeans'."
                )
            if verbose:
                print(
                    f"[CLUSTER] Finalizing with HDBSCAN "
                    f"(min_cluster_size={int(conf.hdbscan_min_cluster_size)}, "
                    f"min_samples={conf.hdbscan_min_samples})...\n"
                )
            metric = str(conf.distance_metric).lower().strip()
            # sklearn BallTree backend used by default HDBSCAN does not support
            # cosine metric; force generic backend for compatibility.
            hdbscan_algorithm = "generic" if metric == "cosine" else "best"
            clusterer = _hdbscan.HDBSCAN(
                min_cluster_size=int(max(2, conf.hdbscan_min_cluster_size)),
                min_samples=None if conf.hdbscan_min_samples is None else int(max(1, conf.hdbscan_min_samples)),
                metric=metric,
                algorithm=hdbscan_algorithm,
                prediction_data=False,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*encountered in matmul.*")
                labels_fit = np.asarray(clusterer.fit_predict(z_fit), dtype=int)

            uniq = sorted([int(v) for v in np.unique(labels_fit) if int(v) >= 0])
            if len(uniq) == 0:
                # Fallback when HDBSCAN marks all points as noise.
                if verbose:
                    print("[CLUSTER] HDBSCAN returned all noise; falling back to KMeans.")
                k_sel = max(2, min(int(conf.k_max), 10))
                km = MiniBatchKMeans(
                    n_clusters=int(k_sel),
                    random_state=int(conf.random_state),
                    batch_size=int(max(256, conf.batch_size)),
                    n_init="auto",
                    init="k-means++",
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*encountered in matmul.*")
                    km.fit(z_fit)
                centroids = _sanitize_x(km.cluster_centers_)
                k_scores = pd.DataFrame([{"k": int(k_sel), "method": "kmeans_fallback", "silhouette": np.nan, "inertia": float(km.inertia_)}])
            else:
                centroids = _sanitize_x(np.vstack([z_fit[labels_fit == c].mean(axis=0) for c in uniq]))
                k_sel = int(len(uniq))
                noise_pct = float((labels_fit < 0).mean() * 100.0)
                k_scores = pd.DataFrame(
                    [
                        {
                            "k": int(k_sel),
                            "method": "hdbscan",
                            "silhouette": np.nan,
                            "inertia": np.nan,
                            "noise_pct": noise_pct,
                        }
                    ]
                )
        else:
            if conf.auto_k:
                k_sel, k_scores = _auto_select_k(z, conf, verbose=verbose)
            else:
                k_sel = int(conf.k_max)
                k_scores = pd.DataFrame([{"k": k_sel, "silhouette": np.nan, "inertia": np.nan}])

            if verbose:
                print(f"[CLUSTER] Finalizing with K={k_sel} using MiniBatchKMeans...\n")
            km = MiniBatchKMeans(
                n_clusters=int(k_sel),
                random_state=int(conf.random_state),
                batch_size=int(max(256, conf.batch_size)),
                n_init="auto",
                init="k-means++",
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*encountered in matmul.*")
                km.fit(z_fit)
            centroids = _sanitize_x(km.cluster_centers_)

        # Distance-signature representation for new flavor assignment.
        dist_df, prob_df, top_cluster = _distance_signature(
            z,
            centroids,
            metric=str(conf.distance_metric),
            temperature=float(conf.distance_temperature),
            cluster_col=cluster_col,
        )
        dist_df.index = base.index
        prob_df.index = base.index
        top_cluster.index = base.index
        base[cluster_col] = top_cluster.astype(int)
        base[f"{cluster_col}__min_dist"] = dist_df.min(axis=1).astype(float)

        p_sorted = np.sort(prob_df.to_numpy(dtype=float), axis=1)
        if p_sorted.shape[1] >= 2:
            margin = p_sorted[:, -1] - p_sorted[:, -2]
        else:
            margin = p_sorted[:, -1]
        base[f"{cluster_col}__margin"] = margin.astype(float)
        base = base.join(dist_df, how="left").join(prob_df, how="left")

    ret_col = str(conf.ret_col)
    has_ret = ret_col in base.columns
    n_total = max(1, len(base))
    date_s, sym_s = _resolve_date_symbol(base)
    cluster_assign = pd.to_numeric(base[str(cluster_col)], errors="coerce").fillna(-1).astype(int)

    # Align feature-scaled frame for distinctiveness.
    if X_scaled_df is None and ae_model is not None and num_cols:
        xs = ae_model.standardized_numeric(base, numeric_cols=num_cols, categorical_cols=cat_cols)
        X_scaled_df = pd.DataFrame(xs, index=base.index, columns=num_cols)
    if isinstance(X_scaled_df, pd.DataFrame):
        x_scaled = X_scaled_df.reindex(base.index) if len(X_scaled_df) != len(base) else X_scaled_df.copy()
    else:
        x_scaled = pd.DataFrame(index=base.index)

    packets: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    cluster_feature_stats: dict[int, dict[str, dict[str, float]]] = {}
    # IMPORTANT: compute explanation stats in RAW feature space so scored rows
    # (which are raw) are directly comparable.
    stats_cols: list[str] = []
    raw_num = pd.DataFrame(index=base.index)
    global_means = pd.Series(dtype=float)
    global_stds = pd.Series(dtype=float)
    if num_cols:
        stats_cols = [c for c in num_cols if c in base.columns and pd.api.types.is_numeric_dtype(base[c])]
        if stats_cols:
            raw_num = base[stats_cols].apply(pd.to_numeric, errors="coerce")
            global_means = raw_num.mean()
            global_stds = raw_num.std(ddof=0).replace(0.0, np.nan)

    valid_clusters = sorted([int(v) for v in np.unique(cluster_assign.to_numpy(dtype=int)) if int(v) >= 0])
    for c in valid_clusters:
        g = base.loc[cluster_assign == int(c)].copy()
        rows = int(len(g))
        pct = 100.0 * rows / float(n_total)

        if has_ret:
            r = pd.to_numeric(g[ret_col], errors="coerce")
            mean_r = float(r.mean())
            med_r = float(r.median())
            std_r = float(r.std(ddof=0))
            sharpe = float(mean_r / std_r) if std_r > 0 else float("nan")
            p10 = float(r.quantile(0.10))
            p90 = float(r.quantile(0.90))
        else:
            mean_r = med_r = std_r = sharpe = p10 = p90 = float("nan")

        dur = pd.to_numeric(g.get("trade_duration_days"), errors="coerce") if "trade_duration_days" in g.columns else pd.Series(dtype=float)
        mean_d = float(dur.mean()) if len(dur) else float("nan")
        std_d = float(dur.std(ddof=0)) if len(dur) else float("nan")

        g_sym = _extract_symbol_series(g)
        sym_gc = g_sym.value_counts()
        top_sym = str(sym_gc.index[0]) if len(sym_gc) else ""
        top_sym_n = int(sym_gc.iloc[0]) if len(sym_gc) else 0
        top_sym_pct = (100.0 * top_sym_n / rows) if rows else float("nan")
        sym_probs = (sym_gc / max(1, sym_gc.sum())).to_numpy(dtype=float)
        sym_hhi = float(np.sum(sym_probs ** 2)) if len(sym_probs) else float("nan")
        eff_syms = float(1.0 / sym_hhi) if np.isfinite(sym_hhi) and sym_hhi > 0 else float("nan")

        g_date = _extract_date_series(g)
        ym = pd.to_datetime(g_date, errors="coerce").dt.to_period("M").astype(str)
        ym_gc = ym.value_counts()
        top_ym = str(ym_gc.index[0]) if len(ym_gc) else ""
        top_ym_n = int(ym_gc.iloc[0]) if len(ym_gc) else 0
        top_ym_pct = (100.0 * top_ym_n / rows) if rows else float("nan")
        ym_probs = (ym_gc / max(1, ym_gc.sum())).to_numpy(dtype=float)
        ym_hhi = float(np.sum(ym_probs ** 2)) if len(ym_probs) else float("nan")
        eff_months = float(1.0 / ym_hhi) if np.isfinite(ym_hhi) and ym_hhi > 0 else float("nan")

        top_features = _feature_distinctiveness(x_scaled, cluster_assign, int(c), int(conf.top_feature_count))
        if stats_cols:
            cluster_means = raw_num.loc[g.index, stats_cols].mean()
            feat_stats: dict[str, dict[str, float]] = {}
            for feat in stats_cols:
                mu_k = float(cluster_means.get(feat, np.nan))
                mu_g = float(global_means.get(feat, np.nan))
                sd = float(global_stds.get(feat, np.nan))
                if np.isfinite(mu_k) and np.isfinite(mu_g) and np.isfinite(sd) and sd > 0:
                    feat_stats[str(feat)] = {
                        "cluster_mean": mu_k,
                        "global_mean": mu_g,
                        "std": sd,
                    }
            cluster_feature_stats[int(c)] = feat_stats
        ae_breaks_top: list[tuple[str, float]] = []
        if bool(conf.include_ae_breaks) and ae_model is not None and num_cols:
            try:
                ae_breaks_df = explain_cluster_ae_feature_breaks(
                    ae_model=ae_model,
                    source_df=base,
                    cluster_labels=cluster_assign.rename(str(conf.cluster_col)),
                    cluster_id=int(c),
                    numeric_cols=num_cols,
                    categorical_cols=cat_cols,
                    top_n=min(int(conf.top_feature_count), 10),
                )
                ae_breaks_top = [
                    (str(r["feature"]), float(r["ae_error_ratio"]))
                    for _, r in ae_breaks_df.iterrows()
                ]
            except Exception:
                ae_breaks_top = []

        packets.append(
            {
                "cluster": int(c),
                "rows": rows,
                "cluster_pct": pct,
                "mean_return": mean_r,
                "median_return": med_r,
                "std_return": std_r,
                "sharpe": sharpe,
                "p10": p10,
                "p90": p90,
                "mean_duration": mean_d,
                "std_duration": std_d,
                "top_symbol": top_sym,
                "top_symbol_rows": top_sym_n,
                "top_symbol_pct": top_sym_pct,
                "symbol_hhi": sym_hhi,
                "effective_symbols": eff_syms,
                "top_year_month": top_ym,
                "top_year_month_rows": top_ym_n,
                "top_year_month_pct": top_ym_pct,
                "time_hhi": ym_hhi,
                "effective_months": eff_months,
                "top_features": top_features,
                "top_ae_breaks": ae_breaks_top,
            }
        )

        feature_tuples = [
            (str(feat), f"{'+' if float(val) >= 0 else ''}{float(val):.2f}σ")
            for feat, val in top_features
        ]
        dominant_symbol_mode = (
            f"{top_sym} ({top_sym_n} rows, {top_sym_pct:.1f}%)"
            if top_sym
            else ""
        )
        symbol_concentration = (
            f"HHI={sym_hhi:.4f}  effective_symbols≈{eff_syms:.2f}"
            if np.isfinite(sym_hhi) and np.isfinite(eff_syms)
            else "HHI=N/A  effective_symbols≈N/A"
        )
        dominant_year_month_mode = (
            f"{top_ym} ({top_ym_n} rows, {top_ym_pct:.1f}%)"
            if top_ym
            else ""
        )
        time_concentration = (
            f"HHI={ym_hhi:.4f}  effective_months≈{eff_months:.2f}"
            if np.isfinite(ym_hhi) and np.isfinite(eff_months)
            else "HHI=N/A  effective_months≈N/A"
        )

        summary_rows.append(
            {
                "cluster": int(c),
                "rows": rows,
                "cluster_pct": pct,
                "mean_return": mean_r,
                "sharpe": sharpe,
                "std_return": std_r,
                "p10": p10,
                "p90": p90,
                "mean_duration": mean_d,
                "std_duration": std_d,
                "top_feature_tuples": feature_tuples,
                "dominant_symbol_mode": dominant_symbol_mode,
                "symbol_concentration": symbol_concentration,
                "dominant_year_month_mode": dominant_year_month_mode,
                "time_concentration": time_concentration,
            }
        )

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["rows", "sharpe", "mean_return"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    if verbose:
        print(f"[PACKETS] clusters={len(summary_df)} rows={len(base):,} has_returns={has_ret}\n")
    flavor_space_out = AlphaFlavorSpace(
        vector_db=vector_db,
        centroids=centroids,
        cluster_labels=cluster_assign.rename(str(cluster_col)),
        cluster_col=str(cluster_col),
        latent_cols=list(latent_cols),
        numeric_cols=list(num_cols),
        categorical_cols=list(cat_cols),
        selected_k=int(k_sel),
        k_scores_df=k_scores.copy(),
        clustered_df=base.copy(),
        summary_df=summary_df.copy(),
        packets=list(packets),
        distance_metric=str(conf.distance_metric if flavor_space is None else flavor_space.distance_metric),
        distance_temperature=float(conf.distance_temperature if flavor_space is None else flavor_space.distance_temperature),
        cluster_feature_stats=cluster_feature_stats,
    )
    return AlphaFlavorClusterResult(
        df=base,
        packets=packets,
        summary_df=summary_df,
        selected_k=int(k_sel),
        k_scores_df=k_scores,
        flavor_space=flavor_space_out,
    )


def print_packets(packets: list[dict[str, Any]], *, has_returns: bool = True) -> None:
    if not packets:
        print("[PACKETS] none")
        return

    pk = sorted(packets, key=lambda d: (float(d.get("sharpe", np.nan))), reverse=True)
    for p in pk:
        print("=" * 90)
        print(f"CLUSTER {p['cluster']}  n={p['rows']}  pct={p['cluster_pct']:.2f}%")
        if has_returns:
            print(
                "trade_return: "
                f"mean={p['mean_return']:.4f}  median={p['median_return']:.4f}  "
                f"std={p['std_return']:.4f}  Sharpe={p['sharpe']:.3f}  "
                f"p10={p['p10']:.4f}  p90={p['p90']:.4f}"
            )
        print(
            f"dominant symbol (mode): {p['top_symbol']} "
            f"({p['top_symbol_rows']} rows, {p['top_symbol_pct']:.1f}%)"
        )
        print(
            f"symbol concentration: HHI={p['symbol_hhi']:.4f}  "
            f"effective_symbols≈{p['effective_symbols']:.2f}"
        )
        print(
            f"dominant year-month (mode): {p['top_year_month']} "
            f"({p['top_year_month_rows']} rows, {p['top_year_month_pct']:.1f}%)"
        )
        print(
            f"time concentration: HHI={p['time_hhi']:.4f}  "
            f"effective_months≈{p['effective_months']:.2f}\n"
        )
        print("Top distinguishing features (cluster mean vs global mean):")
        for feat, val in p.get("top_features", []):
            sign = "+" if val >= 0 else ""
            print(f" - {feat}: {sign}{val:.2f}σ")
        if p.get("top_ae_breaks"):
            print("Top AE break features (error ratio in-cluster vs out-of-cluster):")
            for feat, ratio in p.get("top_ae_breaks", [])[:10]:
                print(f" - {feat}: x{float(ratio):.2f}")
        print()


def fit_flavor_space(
    *,
    df: pd.DataFrame,
    ae_model: Any,
    numeric_cols: list[str],
    categorical_cols: Optional[list[str]] = None,
    cfg: Optional[AlphaFlavorClusterConfig] = None,
    verbose: bool = True,
) -> AlphaFlavorSpace:
    """
    Build vector-db-backed alpha flavor space from AE embeddings.
    """
    res = cluster_alpha_flavors(
        df=df,
        ae_model=ae_model,
        numeric_cols=list(numeric_cols),
        categorical_cols=list(categorical_cols or []),
        cfg=cfg,
        verbose=verbose,
    )
    if res.flavor_space is None:
        raise RuntimeError("Failed to build flavor space.")
    return res.flavor_space


def score_trade_flavor(
    *,
    flavor_space: AlphaFlavorSpace,
    entries_df: pd.DataFrame,
    ae_model: Any,
) -> pd.DataFrame:
    """
    Score entry rows by distance/probability to learned alpha-flavor clusters.
    """
    z_new = ae_model.latent(
        entries_df,
        numeric_cols=flavor_space.numeric_cols,
        categorical_cols=flavor_space.categorical_cols,
    )
    z_new = _sanitize_x(np.asarray(z_new))
    dist_df, prob_df, top_cluster = _distance_signature(
        z_new,
        flavor_space.centroids,
        metric=flavor_space.distance_metric,
        temperature=flavor_space.distance_temperature,
        cluster_col=flavor_space.cluster_col,
    )
    dist_df.index = entries_df.index
    prob_df.index = entries_df.index
    top_cluster.index = entries_df.index

    out = entries_df.copy()
    cluster_col = str(flavor_space.cluster_col)
    min_dist_col = f"{cluster_col}__min_dist"
    fam_col = f"{cluster_col}__familiarity"

    out[cluster_col] = top_cluster.astype(int)
    out[min_dist_col] = dist_df.min(axis=1).astype(float)
    p_sorted = np.sort(prob_df.to_numpy(dtype=float), axis=1)
    out[f"{cluster_col}__margin"] = (
        (p_sorted[:, -1] - p_sorted[:, -2]) if p_sorted.shape[1] >= 2 else p_sorted[:, -1]
    ).astype(float)

    # Cluster familiarity: percentile-based confidence using each cluster's
    # training min-distance distribution. Higher means "more in-family".
    ref_df = flavor_space.clustered_df
    ref_cluster = pd.to_numeric(ref_df.get(cluster_col), errors="coerce")
    ref_dist = pd.to_numeric(ref_df.get(min_dist_col), errors="coerce")
    valid = ref_cluster.notna() & ref_dist.notna()

    familiarity = np.full(len(out), 0.0, dtype=np.float64)
    if bool(valid.any()):
        ref_stats = (
            pd.DataFrame({"cluster": ref_cluster[valid].astype(int), "dist": ref_dist[valid].astype(float)})
            .groupby("cluster")["dist"]
            .apply(lambda s: np.sort(s.to_numpy(dtype=np.float64)))
        )
        assigned = pd.to_numeric(out[cluster_col], errors="coerce").fillna(-1).astype(int).to_numpy()
        d_new = pd.to_numeric(out[min_dist_col], errors="coerce").fillna(np.nan).to_numpy(dtype=np.float64)
        for i, (c, d) in enumerate(zip(assigned, d_new)):
            arr = ref_stats.get(int(c), None)
            if arr is None or len(arr) == 0 or not np.isfinite(d):
                familiarity[i] = 0.0
                continue
            # familiarity = 1 - empirical CDF(distance)
            rank = float(np.searchsorted(arr, d, side="right")) / float(len(arr))
            familiarity[i] = float(np.clip(1.0 - rank, 0.0, 1.0))

    out[fam_col] = familiarity.astype(float)
    out["cluster_familiarity"] = out[fam_col].astype(float)
    out = out.join(dist_df, how="left").join(prob_df, how="left")
    return out


def explain_flavor(
    *,
    flavor_space: AlphaFlavorSpace,
    cluster_id: int,
    source_df: pd.DataFrame,
    ae_model: Any,
    top_n: int = 25,
) -> dict[str, Any]:
    """
    Return explainability bundle for a specific alpha flavor cluster.
    """
    labels = flavor_space.cluster_labels
    c = int(cluster_id)
    perf = flavor_space.summary_df[flavor_space.summary_df["cluster"] == c].copy()
    perf_row = perf.iloc[0].to_dict() if len(perf) else {}

    distinct_df = explain_cluster_feature_uniqueness(
        cluster_labels=labels,
        source_df=source_df,
        cluster_id=c,
        feature_cols=flavor_space.numeric_cols,
        top_n=int(top_n),
    )
    ae_breaks_df = explain_cluster_ae_feature_breaks(
        ae_model=ae_model,
        source_df=source_df,
        cluster_labels=labels,
        cluster_id=c,
        numeric_cols=flavor_space.numeric_cols,
        categorical_cols=flavor_space.categorical_cols,
        top_n=int(top_n),
    )
    return {
        "cluster_id": c,
        "performance": perf_row,
        "distinct_features": distinct_df,
        "ae_break_features": ae_breaks_df,
    }


def explain_trade_vs_cluster(
    *,
    row: pd.Series,
    cluster_id: int,
    flavor_space: AlphaFlavorSpace,
    top_matches: int = 5,
    top_deviations: int = 3,
) -> dict[str, Any]:
    """
    Explain one trade row relative to the assigned cluster.
    """
    c = int(cluster_id)
    stats_map = getattr(flavor_space, "cluster_feature_stats", {}) or {}
    feat_stats = stats_map.get(c, {})
    packet = next((p for p in flavor_space.packets if int(p.get("cluster", -1)) == c), {})
    preferred = [str(f) for f, _ in packet.get("top_features", [])]
    feature_pool = [f for f in preferred if f in feat_stats] or list(feat_stats.keys())

    recs: list[dict[str, Any]] = []
    for feat in feature_pool:
        if feat not in row.index:
            continue
        x = pd.to_numeric(pd.Series([row[feat]]), errors="coerce").iloc[0]
        s = feat_stats.get(feat, {})
        mu_k = float(s.get("cluster_mean", np.nan))
        mu_g = float(s.get("global_mean", np.nan))
        sd = float(s.get("std", np.nan))
        if not np.isfinite(x) or not np.isfinite(mu_k) or not np.isfinite(mu_g) or not np.isfinite(sd) or sd <= 0:
            continue
        expected_z = (mu_k - mu_g) / sd
        actual_z = (float(x) - mu_g) / sd
        delta_z = actual_z - expected_z
        recs.append(
            {
                "feature": feat,
                "actual_z": float(actual_z),
                "expected_z": float(expected_z),
                "delta_z": float(delta_z),
                "distance_contrib": float(delta_z * delta_z),
            }
        )

    if not recs:
        return {"matches": [], "deviations": [], "details": pd.DataFrame()}

    details = pd.DataFrame(recs)
    matches = (
        details.assign(_abs_delta=lambda d: d["delta_z"].abs(), _abs_expected=lambda d: d["expected_z"].abs())
        .sort_values(["_abs_delta", "_abs_expected"], ascending=[True, False])
        .head(max(1, int(top_matches)))
        .drop(columns=["_abs_delta", "_abs_expected"])
    )
    deviations = (
        details.assign(_abs_delta=lambda d: d["delta_z"].abs())
        .sort_values(["_abs_delta", "distance_contrib"], ascending=[False, False])
        .head(max(1, int(top_deviations)))
        .drop(columns=["_abs_delta"])
    )
    return {
        "matches": matches.to_dict(orient="records"),
        "deviations": deviations.to_dict(orient="records"),
        "details": details.sort_values("distance_contrib", ascending=False).reset_index(drop=True),
    }


def add_cluster_explanations(
    scored_df: pd.DataFrame,
    *,
    flavor_space: AlphaFlavorSpace,
    top_matches: int = 5,
    top_deviations: int = 3,
) -> pd.DataFrame:
    """
    Enrich scored rows with cluster narrative columns.
    """
    if scored_df is None or scored_df.empty:
        return scored_df

    out = scored_df.copy()
    cluster_col = str(flavor_space.cluster_col)
    if cluster_col not in out.columns:
        return out

    summary = flavor_space.summary_df.copy()
    summary_map = (
        summary.set_index("cluster").to_dict(orient="index")
        if not summary.empty and "cluster" in summary.columns
        else {}
    )

    out["cluster_top_feature_tuples"] = None
    out["top_matching_features"] = None
    out["top_deviating_features"] = None
    out["dominant_symbol_mode"] = ""
    out["symbol_concentration"] = ""
    out["dominant_year_month_mode"] = ""
    out["time_concentration"] = ""
    out["cluster_mean_return"] = np.nan
    out["cluster_sharpe"] = np.nan
    out["cluster_mean_duration"] = np.nan

    for idx, r in out.iterrows():
        c = pd.to_numeric(pd.Series([r.get(cluster_col)]), errors="coerce").iloc[0]
        if not np.isfinite(c):
            continue
        cid = int(c)
        ex = explain_trade_vs_cluster(
            row=r,
            cluster_id=cid,
            flavor_space=flavor_space,
            top_matches=top_matches,
            top_deviations=top_deviations,
        )
        out.at[idx, "top_matching_features"] = [
            (str(m["feature"]), f"{float(m['actual_z']):+.2f}σ")
            for m in ex.get("matches", [])
        ]
        out.at[idx, "top_deviating_features"] = [
            (str(d["feature"]), f"{float(d['delta_z']):+.2f}σ vs cluster")
            for d in ex.get("deviations", [])
        ]

        p = next((pk for pk in flavor_space.packets if int(pk.get("cluster", -1)) == cid), {})
        out.at[idx, "cluster_top_feature_tuples"] = [
            (str(f), f"{'+' if float(v) >= 0 else ''}{float(v):.2f}σ")
            for f, v in p.get("top_features", [])
        ]

        s = summary_map.get(cid, {})
        out.at[idx, "dominant_symbol_mode"] = str(s.get("dominant_symbol_mode", ""))
        out.at[idx, "symbol_concentration"] = str(s.get("symbol_concentration", ""))
        out.at[idx, "dominant_year_month_mode"] = str(s.get("dominant_year_month_mode", ""))
        out.at[idx, "time_concentration"] = str(s.get("time_concentration", ""))
        out.at[idx, "cluster_mean_return"] = pd.to_numeric(pd.Series([s.get("mean_return", np.nan)]), errors="coerce").iloc[0]
        out.at[idx, "cluster_sharpe"] = pd.to_numeric(pd.Series([s.get("sharpe", np.nan)]), errors="coerce").iloc[0]
        out.at[idx, "cluster_mean_duration"] = pd.to_numeric(pd.Series([s.get("mean_duration", np.nan)]), errors="coerce").iloc[0]

    return out
