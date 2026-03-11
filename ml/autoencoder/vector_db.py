from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class LatentVectorDB:
    """In-memory latent vector index + metadata."""

    latent_df: pd.DataFrame
    nn_index: NearestNeighbors
    nn_metric: str
    nn_neighbors: int


def build_latent_vector_frame(
    ae_model: Any,
    df: pd.DataFrame,
    *,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str] = (),
    include_columns: Optional[Sequence[str]] = None,
    latent_prefix: str = "z_",
) -> pd.DataFrame:
    """
    Encode rows into AE latent vectors and return a DataFrame indexed like input.
    """
    z = np.asarray(
        ae_model.latent(
            df,
            numeric_cols=list(numeric_cols),
            categorical_cols=list(categorical_cols),
        ),
        dtype=np.float32,
    )
    if z.ndim != 2:
        raise ValueError(f"Expected 2D latent matrix, got shape={z.shape}.")

    cols = [f"{latent_prefix}{i}" for i in range(z.shape[1])]
    out = pd.DataFrame(z, index=df.index, columns=cols)

    if include_columns:
        for c in include_columns:
            if c in df.columns and c not in out.columns:
                out[c] = df[c].values
    return out


def build_latent_vector_db(
    latent_df: pd.DataFrame,
    *,
    latent_cols: Optional[Sequence[str]] = None,
    metric: str = "cosine",
    n_neighbors: int = 50,
) -> LatentVectorDB:
    """
    Fit a nearest-neighbor index on latent vectors.
    """
    if latent_df is None or latent_df.empty:
        raise ValueError("latent_df is empty.")

    if latent_cols is None:
        latent_cols = [c for c in latent_df.columns if str(c).startswith("z_")]
    cols = list(latent_cols)
    if not cols:
        raise ValueError("No latent columns found. Pass latent_cols explicitly.")

    x = latent_df[cols].to_numpy(dtype=np.float32, copy=False)
    k = int(max(1, min(n_neighbors, len(latent_df))))
    nn = NearestNeighbors(n_neighbors=k, metric=str(metric))
    nn.fit(x)
    return LatentVectorDB(
        latent_df=latent_df.copy(),
        nn_index=nn,
        nn_metric=str(metric),
        nn_neighbors=k,
    )


def query_latent_neighbors(
    vector_db: LatentVectorDB,
    *,
    query_index: Any | None = None,
    query_vector: Optional[np.ndarray] = None,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Query nearest neighbors by row index key or explicit vector.
    """
    if (query_index is None) == (query_vector is None):
        raise ValueError("Provide exactly one of query_index or query_vector.")

    latent_cols = [c for c in vector_db.latent_df.columns if str(c).startswith("z_")]
    if not latent_cols:
        raise ValueError("vector_db has no latent columns.")

    if query_vector is not None:
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    else:
        if query_index not in vector_db.latent_df.index:
            raise KeyError("query_index not found in vector_db index.")
        q = vector_db.latent_df.loc[[query_index], latent_cols].to_numpy(dtype=np.float32, copy=False)

    k = int(max(1, min(top_k, len(vector_db.latent_df))))
    dists, idxs = vector_db.nn_index.kneighbors(q, n_neighbors=k, return_distance=True)

    nbrs = vector_db.latent_df.iloc[idxs[0]].copy()
    nbrs["nn_distance"] = dists[0]
    return nbrs


def select_natural_k_by_silhouette(
    latent_df: pd.DataFrame,
    *,
    latent_cols: Optional[Sequence[str]] = None,
    k_min: int = 2,
    k_max: int = 20,
    sample_size: int = 50_000,
    random_state: int = 1337,
) -> tuple[int, pd.DataFrame]:
    """
    Choose cluster count by maximizing silhouette score over a KMeans grid.
    """
    if latent_df is None or latent_df.empty:
        raise ValueError("latent_df is empty.")
    if latent_cols is None:
        latent_cols = [c for c in latent_df.columns if str(c).startswith("z_")]
    cols = list(latent_cols)
    if not cols:
        raise ValueError("No latent columns found. Pass latent_cols explicitly.")

    x_full = latent_df[cols].to_numpy(dtype=np.float32, copy=False)
    x_full = np.nan_to_num(x_full, nan=0.0, posinf=1e6, neginf=-1e6)
    n = x_full.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 rows for clustering.")

    if sample_size and n > int(sample_size):
        rng = np.random.default_rng(int(random_state))
        pick = rng.choice(n, size=int(sample_size), replace=False)
        x = x_full[pick]
    else:
        x = x_full

    hi = int(min(max(k_min, 2), max(2, n - 1)))
    k_upper = int(min(k_max, max(2, x.shape[0] - 1)))
    if k_upper < hi:
        hi = 2
        k_upper = min(10, x.shape[0] - 1)
    if k_upper < 2:
        raise ValueError("Not enough rows to evaluate silhouette over k>=2.")

    rows = []
    best_k = None
    best_s = -np.inf
    for k in range(int(hi), int(k_upper) + 1):
        km = KMeans(n_clusters=int(k), random_state=int(random_state), n_init="auto")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=r".*encountered in matmul.*",
            )
            labels = km.fit_predict(x)
            if len(np.unique(labels)) < 2:
                s = float("nan")
            else:
                s = float(silhouette_score(x, labels))
        rows.append({"k": int(k), "silhouette": s, "inertia": float(km.inertia_)})
        if np.isfinite(s) and s > best_s:
            best_s = s
            best_k = int(k)

    scan = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    if best_k is None:
        best_k = int(scan.loc[scan["inertia"].idxmin(), "k"])
    return best_k, scan


def select_natural_k_fast_elbow(
    latent_df: pd.DataFrame,
    *,
    latent_cols: Optional[Sequence[str]] = None,
    k_min: int = 2,
    k_max: int = 50,
    sample_size: int = 50_000,
    batch_size: int = 4096,
    random_state: int = 1337,
) -> tuple[int, pd.DataFrame]:
    """
    Fast/cheap cluster-count selection using MiniBatchKMeans inertia elbow.
    """
    if latent_df is None or latent_df.empty:
        raise ValueError("latent_df is empty.")
    if latent_cols is None:
        latent_cols = [c for c in latent_df.columns if str(c).startswith("z_")]
    cols = list(latent_cols)
    if not cols:
        raise ValueError("No latent columns found. Pass latent_cols explicitly.")

    x_full = latent_df[cols].to_numpy(dtype=np.float32, copy=False)
    x_full = np.nan_to_num(x_full, nan=0.0, posinf=1e6, neginf=-1e6)
    n = x_full.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 rows for clustering.")

    if sample_size and n > int(sample_size):
        rng = np.random.default_rng(int(random_state))
        pick = rng.choice(n, size=int(sample_size), replace=False)
        x = x_full[pick]
    else:
        x = x_full

    lo = int(max(2, k_min))
    hi = int(min(k_max, max(2, x.shape[0] - 1)))
    if hi < lo:
        hi = lo

    rows = []
    for k in range(lo, hi + 1):
        km = MiniBatchKMeans(
            n_clusters=int(k),
            random_state=int(random_state),
            batch_size=int(max(256, batch_size)),
            n_init="auto",
            init="k-means++",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=r".*encountered in matmul.*",
            )
            km.fit(x)
        rows.append({"k": int(k), "inertia": float(km.inertia_)})

    scan = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    if len(scan) == 1:
        return int(scan.loc[0, "k"]), scan

    # Kneedle-like elbow: max distance from line between first and last points.
    ks = scan["k"].to_numpy(dtype=np.float64)
    inertias = scan["inertia"].to_numpy(dtype=np.float64)
    x_norm = (ks - ks.min()) / max(1e-12, (ks.max() - ks.min()))
    y_norm = (inertias - inertias.min()) / max(1e-12, (inertias.max() - inertias.min()))
    d = y_norm - (1.0 - x_norm)
    best_idx = int(np.argmax(d))
    best_k = int(scan.loc[best_idx, "k"])
    return best_k, scan


def cluster_latent_kmeans(
    latent_df: pd.DataFrame,
    *,
    n_clusters: int,
    latent_cols: Optional[Sequence[str]] = None,
    random_state: int = 1337,
) -> pd.Series:
    """
    Cluster latent vectors with KMeans and return cluster labels aligned to latent_df index.
    """
    if latent_cols is None:
        latent_cols = [c for c in latent_df.columns if str(c).startswith("z_")]
    cols = list(latent_cols)
    if not cols:
        raise ValueError("No latent columns found. Pass latent_cols explicitly.")

    x = latent_df[cols].to_numpy(dtype=np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    km = KMeans(n_clusters=int(n_clusters), random_state=int(random_state), n_init="auto")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=r".*encountered in matmul.*",
        )
        lab = km.fit_predict(x)
    return pd.Series(lab, index=latent_df.index, name="cluster")


def summarize_cluster_performance(
    *,
    cluster_labels: pd.Series,
    source_df: pd.DataFrame,
    return_col: str = "trade_return",
    duration_col: str = "trade_duration_days",
    annualization_factor: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute cluster-level performance stats.

    Metrics:
      - n_obs
      - mean_return
      - std_return
      - mean_duration
      - std_duration
      - sharpe (mean_return/std_return)
      - sharpe_annualized (optional)
    """
    if cluster_labels is None or len(cluster_labels) == 0:
        raise ValueError("cluster_labels is empty.")
    if source_df is None or source_df.empty:
        raise ValueError("source_df is empty.")
    if return_col not in source_df.columns:
        raise KeyError(f"Missing return column: {return_col}")
    if duration_col not in source_df.columns:
        raise KeyError(f"Missing duration column: {duration_col}")

    aligned = source_df[[return_col, duration_col]].copy()
    aligned = aligned.join(cluster_labels.rename("cluster"), how="inner")
    if aligned.empty:
        raise ValueError("No overlapping rows between source_df and cluster_labels index.")

    aligned[return_col] = pd.to_numeric(aligned[return_col], errors="coerce")
    aligned[duration_col] = pd.to_numeric(aligned[duration_col], errors="coerce")

    g = aligned.groupby("cluster", sort=True)
    out = pd.DataFrame(
        {
            "n_obs": g.size(),
            "mean_return": g[return_col].mean(),
            "std_return": g[return_col].std(ddof=0),
            "mean_duration": g[duration_col].mean(),
            "std_duration": g[duration_col].std(ddof=0),
        }
    ).reset_index()
    total = max(1, int(out["n_obs"].sum()))
    out["pct_dataset"] = 100.0 * out["n_obs"] / float(total)

    out["sharpe"] = out["mean_return"] / out["std_return"].replace(0.0, np.nan)
    if annualization_factor is not None:
        out["sharpe_annualized"] = out["sharpe"] * float(np.sqrt(float(annualization_factor)))

    return out.sort_values("sharpe", ascending=False).reset_index(drop=True)


def build_ae_cluster_report(
    *,
    ae_model: Any,
    source_df: pd.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str] = (),
    return_col: str = "trade_return",
    duration_col: str = "trade_duration_days",
    k_min: int = 2,
    k_max: int = 50,
    silhouette_sample_size: int = 50_000,
    k_selection_method: str = "fast_elbow",
    random_state: int = 1337,
    nn_metric: str = "cosine",
    nn_neighbors: int = 50,
) -> Dict[str, Any]:
    """
    End-to-end helper:
      1) latent vectors
      2) vector index
      3) natural k via silhouette
      4) kmeans clusters
      5) per-cluster performance summary
    """
    latent_df = build_latent_vector_frame(
        ae_model,
        source_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        include_columns=[return_col, duration_col],
    )
    vector_db = build_latent_vector_db(
        latent_df,
        metric=nn_metric,
        n_neighbors=nn_neighbors,
    )
    method = str(k_selection_method).lower().strip()
    if method == "silhouette":
        best_k, k_scan = select_natural_k_by_silhouette(
            latent_df,
            k_min=k_min,
            k_max=k_max,
            sample_size=silhouette_sample_size,
            random_state=random_state,
        )
    elif method in {"fast_elbow", "elbow"}:
        best_k, k_scan = select_natural_k_fast_elbow(
            latent_df,
            k_min=k_min,
            k_max=k_max,
            sample_size=silhouette_sample_size,
            random_state=random_state,
        )
    else:
        raise ValueError("k_selection_method must be 'fast_elbow' or 'silhouette'.")
    cluster_labels = cluster_latent_kmeans(
        latent_df,
        n_clusters=best_k,
        random_state=random_state,
    )
    perf = summarize_cluster_performance(
        cluster_labels=cluster_labels,
        source_df=source_df,
        return_col=return_col,
        duration_col=duration_col,
    )

    clustered = latent_df.join(cluster_labels, how="left")
    return {
        "latent_df": latent_df,
        "vector_db": vector_db,
        "k_scan": k_scan,
        "k_selection_method": method,
        "best_k": int(best_k),
        "cluster_labels": cluster_labels,
        "clustered_df": clustered,
        "cluster_performance": perf,
    }


def explain_cluster_feature_uniqueness(
    *,
    cluster_labels: pd.Series,
    source_df: pd.DataFrame,
    cluster_id: int,
    feature_cols: Optional[Sequence[str]] = None,
    top_n: int = 30,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Rank features that are most unique for one cluster vs all other rows.

    Returns columns:
      - feature
      - n_in, n_out
      - mean_in, mean_out
      - std_in, std_out
      - delta_mean
      - effect_size (Cohen-like d using pooled std)
      - abs_effect_size
      - mean_ratio (mean_in / mean_out, signed-safe)
    """
    if cluster_labels is None or len(cluster_labels) == 0:
        raise ValueError("cluster_labels is empty.")
    if source_df is None or source_df.empty:
        raise ValueError("source_df is empty.")

    aligned = source_df.copy().join(cluster_labels.rename("cluster"), how="inner")
    if aligned.empty:
        raise ValueError("No overlapping rows between source_df and cluster_labels index.")

    if feature_cols is None:
        exclude = {"cluster", "label", "rank_y", "sample_weight"}
        feature_cols = [
            c for c in aligned.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(aligned[c])
        ]
    cols = [c for c in list(feature_cols) if c in aligned.columns]
    if not cols:
        raise ValueError("No usable numeric feature columns for explainability.")

    in_mask = aligned["cluster"] == int(cluster_id)
    out_mask = ~in_mask
    if int(in_mask.sum()) == 0:
        raise ValueError(f"cluster_id={cluster_id} has no rows.")
    if int(out_mask.sum()) == 0:
        raise ValueError("No out-of-cluster rows available for comparison.")

    a = aligned.loc[in_mask, cols].apply(pd.to_numeric, errors="coerce")
    b = aligned.loc[out_mask, cols].apply(pd.to_numeric, errors="coerce")

    n_in = a.notna().sum().astype(float)
    n_out = b.notna().sum().astype(float)
    mean_in = a.mean()
    mean_out = b.mean()
    std_in = a.std(ddof=0)
    std_out = b.std(ddof=0)

    pooled = np.sqrt((std_in.pow(2) + std_out.pow(2)) / 2.0).replace(0.0, np.nan)
    delta = mean_in - mean_out
    effect = delta / pooled

    denom = mean_out.where(mean_out.abs() > eps, np.nan)
    ratio = mean_in / denom

    out = pd.DataFrame(
        {
            "feature": cols,
            "n_in": n_in.reindex(cols).values,
            "n_out": n_out.reindex(cols).values,
            "mean_in": mean_in.reindex(cols).values,
            "mean_out": mean_out.reindex(cols).values,
            "std_in": std_in.reindex(cols).values,
            "std_out": std_out.reindex(cols).values,
            "delta_mean": delta.reindex(cols).values,
            "effect_size": effect.reindex(cols).values,
            "mean_ratio": ratio.reindex(cols).values,
        }
    )
    out["abs_effect_size"] = out["effect_size"].abs()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.sort_values("abs_effect_size", ascending=False).head(int(top_n)).reset_index(drop=True)
    return out


def explain_cluster_ae_feature_breaks(
    *,
    ae_model: Any,
    source_df: pd.DataFrame,
    cluster_labels: pd.Series,
    cluster_id: int,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str] = (),
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Explain cluster uniqueness using AE per-feature reconstruction error.

    Higher `error_ratio` means this feature is reconstructed much worse inside
    the cluster than outside, suggesting it is a distinctive manifold break.
    """
    if cluster_labels is None or len(cluster_labels) == 0:
        raise ValueError("cluster_labels is empty.")
    if source_df is None or source_df.empty:
        raise ValueError("source_df is empty.")

    aligned = source_df.copy().join(cluster_labels.rename("cluster"), how="inner")
    if aligned.empty:
        raise ValueError("No overlapping rows between source_df and cluster_labels index.")

    err_df = ae_model.recon_error_matrix(
        aligned,
        numeric_cols=list(numeric_cols),
        categorical_cols=list(categorical_cols),
        space="standardized",
    )
    err_df = err_df.join(aligned["cluster"], how="inner")

    in_mask = err_df["cluster"] == int(cluster_id)
    out_mask = ~in_mask
    if int(in_mask.sum()) == 0:
        raise ValueError(f"cluster_id={cluster_id} has no rows.")
    if int(out_mask.sum()) == 0:
        raise ValueError("No out-of-cluster rows available for comparison.")

    cols = [c for c in err_df.columns if c != "cluster"]
    in_mean = err_df.loc[in_mask, cols].mean()
    out_mean = err_df.loc[out_mask, cols].mean()
    ratio = in_mean / out_mean.replace(0.0, np.nan)
    delta = in_mean - out_mean

    out = pd.DataFrame(
        {
            "feature": cols,
            "ae_error_mean_in": in_mean.reindex(cols).values,
            "ae_error_mean_out": out_mean.reindex(cols).values,
            "ae_error_delta": delta.reindex(cols).values,
            "ae_error_ratio": ratio.reindex(cols).values,
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan)
    out["abs_ae_error_delta"] = out["ae_error_delta"].abs()
    out = out.sort_values("ae_error_ratio", ascending=False).head(int(top_n)).reset_index(drop=True)
    return out
