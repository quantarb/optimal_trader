from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import AutoEncoderConfig
from .model import DynamicHybridAutoEncoder, NumericAutoEncoder


@dataclass(frozen=True)
class AutoEncoderArtifact:
    # Schema
    numeric_cols: List[str]
    cat_cols: List[str]

    # Numeric transform (robust, fit on optimal-trade training rows only)
    # center_/scale_ are used as mean_/std_ equivalents throughout adapter code.
    mean_: np.ndarray
    std_: np.ndarray

    # Categorical mappings: col -> list of seen categories (index = code)
    cat_mappings: Dict[str, List[object]]
    cat_cardinalities: List[int]

    # Model + runtime
    model: torch.nn.Module
    device: str
    requires_cats: bool

    lower_: Optional[np.ndarray] = None
    upper_: Optional[np.ndarray] = None

    # Global calibration reference (trained on optimal trades).
    # Used to map reconstruction error -> percentile in [0,1] at inference.
    train_error_sorted: Optional[np.ndarray] = None
    train_feature_mse: Optional[np.ndarray] = None
    feature_weights: Optional[np.ndarray] = None
    score_topk_frac: float = 1.0

    # Latent-manifold calibration reference (trained on optimal trades).
    latent_ref: Optional[np.ndarray] = None
    latent_dist_train_sorted: Optional[np.ndarray] = None
    latent_mean: Optional[np.ndarray] = None
    latent_cov_inv: Optional[np.ndarray] = None
    latent_mahal_train_sorted: Optional[np.ndarray] = None


def _standardize_fit(
    X: np.ndarray,
    *,
    lo_pct: float = 0.1,
    hi_pct: float = 99.9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # RobustScaler-style transform:
    # 1) winsorize by lo/hi percentiles
    # 2) center by median
    # 3) scale by IQR (Q3-Q1)
    X = np.asarray(X, dtype=np.float64)
    try:
        lo = np.nanpercentile(X, float(lo_pct), axis=0)
        hi = np.nanpercentile(X, float(hi_pct), axis=0)
    except Exception:
        lo = np.full((X.shape[1],), -np.inf, dtype=np.float64)
        hi = np.full((X.shape[1],), np.inf, dtype=np.float64)
    lo = np.nan_to_num(lo, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    hi = np.nan_to_num(hi, nan=np.inf, posinf=np.inf, neginf=-np.inf)

    Xw = np.clip(X, a_min=lo, a_max=hi)

    center = np.nanmedian(Xw, axis=0)
    center = np.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0)
    q1 = np.nanpercentile(Xw, 25.0, axis=0)
    q3 = np.nanpercentile(Xw, 75.0, axis=0)
    scale = q3 - q1
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
    scale = np.where(scale > 1e-12, scale, 1.0)

    X_filled = np.where(np.isfinite(Xw), Xw, center)
    Xs = (X_filled - center) / scale
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = np.clip(Xs, a_min=-50.0, a_max=50.0)
    return (
        Xs.astype(np.float32, copy=False),
        center.astype(np.float32),
        scale.astype(np.float32),
        lo.astype(np.float32, copy=False),
        hi.astype(np.float32, copy=False),
    )


def standardize_apply(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if lower is not None and upper is not None and len(lower) and len(upper):
        lo = np.asarray(lower, dtype=np.float64)
        hi = np.asarray(upper, dtype=np.float64)
        X = np.clip(X, a_min=lo, a_max=hi)
    X_filled = np.where(np.isfinite(X), X, mean)
    Xs = (X_filled - mean) / std
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = np.clip(Xs, a_min=-50.0, a_max=50.0)
    return Xs.astype(np.float32, copy=False)


def train_autoencoder(
    X_num: np.ndarray,
    X_cat: Optional[np.ndarray],
    cat_cardinalities: List[int],
    numeric_cols: List[str],
    cat_cols: List[str],
    cat_mappings: Dict[str, List[object]],
    cfg: AutoEncoderConfig,
) -> AutoEncoderArtifact:
    lo_pct = float(getattr(cfg, "robust_clip_lo_pct", 0.1))
    hi_pct = float(getattr(cfg, "robust_clip_hi_pct", 99.9))
    n_layers = int(getattr(cfg, "n_layers", 2))
    min_layer_dim = int(getattr(cfg, "min_layer_dim", 2))
    denoise_std = float(getattr(cfg, "denoise_std", 0.0))
    latent_ref_max_points = int(getattr(cfg, "latent_ref_max_points", 50000))
    latent_ref_random_state = int(getattr(cfg, "latent_ref_random_state", 1337))

    Xs, mean, std, lower, upper = _standardize_fit(
        X_num,
        lo_pct=lo_pct,
        hi_pct=hi_pct,
    )

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    num_t = torch.tensor(Xs, dtype=torch.float32)
    if X_cat is None:
        X_cat = np.empty((Xs.shape[0], 0), dtype=np.int64)
    cat_t = torch.tensor(X_cat, dtype=torch.long)

    loader = DataLoader(TensorDataset(num_t, cat_t), batch_size=cfg.batch_size, shuffle=True)

    use_cats = (cfg.embed_dim is not None and cfg.embed_dim > 0 and len(cat_cols) > 0)
    if use_cats:
        model: torch.nn.Module = DynamicHybridAutoEncoder(
            in_dim=Xs.shape[1],
            cat_cardinalities=cat_cardinalities,
            embed_dim=int(cfg.embed_dim),
            n_layers=n_layers,
            min_layer_dim=min_layer_dim,
        ).to(device)
        requires_cats = True
    else:
        model = NumericAutoEncoder(
            in_dim=Xs.shape[1],
            n_layers=n_layers,
            min_layer_dim=min_layer_dim,
        ).to(device)
        requires_cats = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    loss_fn = torch.nn.MSELoss()
    model.train()
    for _epoch in range(cfg.epochs):
        for b_num, b_cat in loader:
            b_num = b_num.to(device)
            b_cat = b_cat.to(device)

            optimizer.zero_grad(set_to_none=True)

            if denoise_std > 0:
                noisy = b_num + (torch.randn_like(b_num) * denoise_std)
            else:
                noisy = b_num

            if requires_cats:
                recon = model(noisy, b_cat)
            else:
                recon = model(noisy)

            loss = loss_fn(recon, b_num)
            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

    model.eval()
    artifact_base = AutoEncoderArtifact(
        numeric_cols=list(numeric_cols),
        cat_cols=list(cat_cols),
        mean_=mean,
        std_=std,
        lower_=lower,
        upper_=upper,
        cat_mappings=dict(cat_mappings),
        cat_cardinalities=list(cat_cardinalities),
        model=model,
        device=str(device),
        requires_cats=requires_cats,
    )

    # Build train manifold calibration from plain row-wise reconstruction MSE.
    train_scores = reconstruction_error_rows(
        artifact_base,
        X_num=X_num,
        X_cat=X_cat,
        batch_size=cfg.batch_size,
    )
    feat_mse = reconstruction_error_feature_mse(
        artifact_base,
        X_num=X_num,
        X_cat=X_cat,
        batch_size=cfg.batch_size,
    )
    # Precision weights: stable optimal features (low residual variance) get higher weight.
    eps = 1e-8
    feature_weights = 1.0 / np.clip(feat_mse, a_min=eps, a_max=None)
    feature_weights = feature_weights / max(float(np.mean(feature_weights)), eps)

    # Build latent-manifold reference and calibration distances.
    z_train = latent_features(
        artifact_base,
        X_num=X_num,
        X_cat=X_cat,
        batch_size=cfg.batch_size,
    )
    z_train = np.asarray(z_train, dtype=np.float32)
    z_train = np.nan_to_num(z_train, nan=0.0, posinf=0.0, neginf=0.0)
    z_train = np.clip(z_train, a_min=-1e4, a_max=1e4)
    n = len(z_train)
    if n == 0:
        latent_ref = np.empty((0, 0), dtype=np.float32)
        latent_dist_sorted = np.empty((0,), dtype=np.float64)
        latent_mean = np.empty((0,), dtype=np.float64)
        latent_cov_inv = np.empty((0, 0), dtype=np.float64)
        latent_mahal_sorted = np.empty((0,), dtype=np.float64)
    else:
        rng = np.random.default_rng(latent_ref_random_state)
        m = min(latent_ref_max_points, n)
        if m < n:
            ref_idx = rng.choice(n, size=m, replace=False)
            latent_ref = z_train[ref_idx]
        else:
            latent_ref = z_train

        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(latent_ref)
        d_train, _ = nn.kneighbors(z_train, return_distance=True)
        latent_dist_sorted = np.sort(d_train[:, 0].astype(np.float64, copy=False))

        # Latent Mahalanobis calibration on optimal train manifold.
        z64 = z_train.astype(np.float64, copy=False)
        latent_mean = np.mean(z64, axis=0)
        centered = z64 - latent_mean
        cov = np.cov(centered, rowvar=False)
        if np.ndim(cov) == 0:
            cov = np.array([[float(cov)]], dtype=np.float64)
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        reg = 1e-6 * np.eye(cov.shape[0], dtype=np.float64)
        try:
            with np.errstate(divide="raise", over="raise", invalid="raise"):
                latent_cov_inv = np.linalg.pinv(cov + reg)
        except FloatingPointError:
            # Fallback to identity when covariance is numerically unstable.
            latent_cov_inv = np.eye(cov.shape[0], dtype=np.float64)
        latent_cov_inv = np.nan_to_num(latent_cov_inv, nan=0.0, posinf=0.0, neginf=0.0)
        # d^2 = (x-mu)^T S^-1 (x-mu)
        d2 = np.einsum("ij,jk,ik->i", centered, latent_cov_inv, centered, optimize=True)
        d2 = np.clip(d2, a_min=0.0, a_max=None)
        latent_mahal_sorted = np.sort(np.sqrt(d2).astype(np.float64, copy=False))

    return AutoEncoderArtifact(
        numeric_cols=list(numeric_cols),
        cat_cols=list(cat_cols),
        mean_=mean,
        std_=std,
        lower_=lower,
        upper_=upper,
        cat_mappings=dict(cat_mappings),
        cat_cardinalities=list(cat_cardinalities),
        model=model,
        device=str(device),
        requires_cats=requires_cats,
        train_error_sorted=np.sort(np.asarray(train_scores, dtype=np.float64)),
        train_feature_mse=feat_mse.astype(np.float64, copy=False),
        feature_weights=feature_weights.astype(np.float64, copy=False),
        score_topk_frac=1.0,
        latent_ref=latent_ref.astype(np.float32, copy=False),
        latent_dist_train_sorted=latent_dist_sorted,
        latent_mean=latent_mean,
        latent_cov_inv=latent_cov_inv,
        latent_mahal_train_sorted=latent_mahal_sorted,
    )


def reconstruction_error_rows(
    artifact: AutoEncoderArtifact,
    X_num: np.ndarray,
    X_cat: Optional[np.ndarray],
    batch_size: int = 8192,
) -> np.ndarray:
    Xs = standardize_apply(X_num, artifact.mean_, artifact.std_, artifact.lower_, artifact.upper_)
    device = torch.device(artifact.device)

    if X_cat is None:
        X_cat = np.empty((Xs.shape[0], 0), dtype=np.int64)

    n = Xs.shape[0]
    out = np.empty((n,), dtype=np.float64)
    artifact.model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            num_t = torch.tensor(Xs[start:end], dtype=torch.float32, device=device)
            if artifact.requires_cats:
                cat_t = torch.tensor(X_cat[start:end], dtype=torch.long, device=device)
                recon = artifact.model(num_t, cat_t).detach().cpu().numpy()
            else:
                recon = artifact.model(num_t).detach().cpu().numpy()

            recon = np.nan_to_num(recon, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32, copy=False)
            diff = recon - Xs[start:end]
            out[start:end] = np.mean(diff * diff, axis=1).astype(np.float64, copy=False)

    return out


def reconstruction_error_feature_mse(
    artifact: AutoEncoderArtifact,
    X_num: np.ndarray,
    X_cat: Optional[np.ndarray],
    batch_size: int = 8192,
) -> np.ndarray:
    """Per-feature mean squared reconstruction error in standardized space."""
    Xs = standardize_apply(X_num, artifact.mean_, artifact.std_, artifact.lower_, artifact.upper_)
    device = torch.device(artifact.device)

    if X_cat is None:
        X_cat = np.empty((Xs.shape[0], 0), dtype=np.int64)

    sse = np.zeros((Xs.shape[1],), dtype=np.float64)
    n = Xs.shape[0]
    artifact.model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            num_t = torch.tensor(Xs[start:end], dtype=torch.float32, device=device)
            if artifact.requires_cats:
                cat_t = torch.tensor(X_cat[start:end], dtype=torch.long, device=device)
                recon = artifact.model(num_t, cat_t).detach().cpu().numpy()
            else:
                recon = artifact.model(num_t).detach().cpu().numpy()
            recon = np.nan_to_num(recon, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32, copy=False)
            diff = recon - Xs[start:end]
            sse += np.sum(diff * diff, axis=0).astype(np.float64, copy=False)
    denom = max(1, n)
    return (sse / float(denom)).astype(np.float64, copy=False)


def latent_features(
    artifact: AutoEncoderArtifact,
    X_num: np.ndarray,
    X_cat: Optional[np.ndarray],
    batch_size: int = 8192,
) -> np.ndarray:
    Xs = standardize_apply(X_num, artifact.mean_, artifact.std_, artifact.lower_, artifact.upper_)
    device = torch.device(artifact.device)

    if X_cat is None:
        X_cat = np.empty((Xs.shape[0], 0), dtype=np.int64)

    n = Xs.shape[0]
    out = np.empty((n, artifact.model.encoder[-1].out_features), dtype=np.float32)  # bottleneck_dim

    artifact.model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            num_t = torch.tensor(Xs[start:end], dtype=torch.float32, device=device)
            if artifact.requires_cats:
                cat_t = torch.tensor(X_cat[start:end], dtype=torch.long, device=device)
                z = artifact.model.encode(num_t, cat_t)
            else:
                z = artifact.model.encoder(num_t)
            z_np = z.detach().cpu().numpy().astype(np.float32, copy=False)
            z_np = np.nan_to_num(z_np, nan=0.0, posinf=0.0, neginf=0.0)
            z_np = np.clip(z_np, a_min=-1e4, a_max=1e4)
            out[start:end] = z_np

    return out.astype(np.float64, copy=False)
