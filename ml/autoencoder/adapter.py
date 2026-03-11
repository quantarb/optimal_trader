from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ml.base import FitSpec, Model, print_model_section

from .config import AutoEncoderConfig
from .trainer import AutoEncoderArtifact, train_autoencoder, reconstruction_error_rows, latent_features, standardize_apply


class TorchAutoEncoder(Model):
    """Torch AutoEncoder (numeric-only or hybrid with categorical embeddings).

    IMPORTANT (no autodetect):
      - You MUST pass numeric_cols and categorical_cols explicitly to fit()/latent()/reconstruct()/recon_error().
      - Dtypes are NOT used to infer schema.

    Recommended usage:
      ae.fit(df_train, spec, numeric_cols=[...], categorical_cols=[...])
      z = ae.latent(df_any, numeric_cols=[...], categorical_cols=[...])
      err = ae.recon_error(df_any, numeric_cols=[...], categorical_cols=[...])
    """

    def __init__(self, cfg: AutoEncoderConfig | None = None) -> None:
        self.cfg = cfg or AutoEncoderConfig()
        self._artifact: AutoEncoderArtifact | None = None
        self._metrics: Dict[str, Any] = {}
        self._is_fit = False
        self._latent_nn = None

    # -------------------------
    # Schema + DF prep utilities
    # -------------------------
    @staticmethod
    def _pull_from_index(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
        available_cols = set(df.columns)
        idx_names = list(getattr(df.index, "names", []) or [])
        needed_from_index = [c for c in cols if c in idx_names and c not in available_cols]
        if needed_from_index:
            return df.reset_index(level=needed_from_index)
        return df.copy()

    @staticmethod
    def _require_cols(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing {len(missing)} {name} columns. Example: {missing[:10]}")

    def _build_matrices(
        self,
        df: pd.DataFrame,
        *,
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str],
        is_fit_time: bool,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, List[object]], List[int]]:
        """Return X_num, X_cat, cat_mappings, cat_cardinalities.

        - At fit time, we factorize categoricals and store mappings.
        - At inference time, we map unseen categories to 0.
        """
        df2 = self._pull_from_index(df, list(numeric_cols) + list(categorical_cols))

        self._require_cols(df2, numeric_cols, "numeric")
        self._require_cols(df2, categorical_cols, "categorical")

        X_num = df2[list(numeric_cols)].to_numpy(dtype=np.float32, copy=False)

        if len(categorical_cols) == 0:
            return X_num, None, {}, []

        if is_fit_time:
            cat_mappings: Dict[str, List[object]] = {}
            cat_indices = []
            cat_cardinalities = []
            for col in categorical_cols:
                codes, uniques = pd.factorize(df2[col], sort=True)
                # Reserve 0 for unknown/missing, shift known categories to 1..K.
                codes = np.asarray(codes, dtype=np.int64)
                codes = np.where(codes >= 0, codes + 1, 0).astype(np.int64, copy=False)
                cat_indices.append(codes)
                cat_cardinalities.append(int(len(uniques)) + 1)
                cat_mappings[col] = uniques.tolist()
            X_cat = np.stack(cat_indices, axis=1)
            return X_num, X_cat, cat_mappings, cat_cardinalities

        # inference-time mapping using stored mappings
        assert self._artifact is not None
        cat_indices = []
        for col in categorical_cols:
            mapping_list = self._artifact.cat_mappings.get(col)
            if not mapping_list:
                raise RuntimeError(
                    f"Categorical mapping for '{col}' not found in artifact. "
                    "This usually means you changed categorical_cols between fit() and inference."
                )
            # Match fit-time convention: known categories 1..K, unknown/missing -> 0.
            mapping = {v: (i + 1) for i, v in enumerate(mapping_list)}
            codes = df2[col].map(mapping).fillna(0).astype(int).to_numpy(dtype=np.int64, copy=False)
            cat_indices.append(codes)
        X_cat = np.stack(cat_indices, axis=1)
        return X_num, X_cat, {}, []

    # -------------
    # Public API
    # -------------
    def fit(
        self,
        df_train: pd.DataFrame,
        spec: FitSpec,
        *,
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str] = (),
        verbose: bool = True,
    ) -> "TorchAutoEncoder":
        # no autodetect; verify numeric/cat subsets of spec.feature_cols
        feat = list(spec.feature_cols)
        if any(c not in feat for c in numeric_cols):
            raise ValueError("All numeric_cols must be included in spec.feature_cols")
        if any(c not in feat for c in categorical_cols):
            raise ValueError("All categorical_cols must be included in spec.feature_cols")

        X_num, X_cat, cat_mappings, cat_cardinalities = self._build_matrices(
            df_train, numeric_cols=numeric_cols, categorical_cols=categorical_cols, is_fit_time=True
        )

        self._artifact = train_autoencoder(
            X_num=X_num,
            X_cat=X_cat,
            cat_cardinalities=cat_cardinalities,
            numeric_cols=list(numeric_cols),
            cat_cols=list(categorical_cols),
            cat_mappings=cat_mappings,
            cfg=self.cfg,
        )

        # Training-set error metrics (use artifact calibration if present)
        scores = getattr(self._artifact, "train_error_sorted", None)
        if scores is None or len(scores) == 0:
            scores = reconstruction_error_rows(self._artifact, X_num, X_cat, batch_size=self.cfg.batch_size)
        self._metrics = {
            "recon_error_mean": float(np.mean(scores)),
            "recon_error_p95": float(np.percentile(scores, 95)),
        }
        feat_mse = getattr(self._artifact, "train_feature_mse", None)
        if feat_mse is not None and len(feat_mse) == len(self._artifact.numeric_cols):
            pairs = list(zip(self._artifact.numeric_cols, np.asarray(feat_mse, dtype=float)))
            pairs.sort(key=lambda x: x[1], reverse=True)
            self._metrics["feature_mse_top"] = [(str(col), float(val)) for col, val in pairs[:10]]
        self._is_fit = True
        if verbose:
            self.summarize()
        self._latent_nn = None
        return self

    def _get_latent_nn(self):
        if not self._is_fit or self._artifact is None:
            raise RuntimeError("Model must be fit first.")
        ref = getattr(self._artifact, "latent_ref", None)
        if ref is None or len(ref) == 0:
            raise RuntimeError("latent_ref is unavailable; re-fit AE to build latent calibration reference.")
        if self._latent_nn is None:
            from sklearn.neighbors import NearestNeighbors
            self._latent_nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            self._latent_nn.fit(np.asarray(ref, dtype=np.float32))
        return self._latent_nn

    def standardized_numeric(self, df: pd.DataFrame, *, numeric_cols: Sequence[str], categorical_cols: Sequence[str] = ()) -> np.ndarray:
        if not self._is_fit or self._artifact is None:
            raise RuntimeError("Model must be fit first.")
        if list(numeric_cols) != self._artifact.numeric_cols:
            raise ValueError("numeric_cols must match exactly the columns used during fit().")
        if list(categorical_cols) != self._artifact.cat_cols:
            raise ValueError("categorical_cols must match exactly the columns used during fit().")

        X_num, _, _, _ = self._build_matrices(df, numeric_cols=numeric_cols, categorical_cols=(), is_fit_time=False)
        Xs = standardize_apply(
            X_num,
            self._artifact.mean_,
            self._artifact.std_,
            self._artifact.lower_,
            self._artifact.upper_,
        )
        return Xs

    def latent(self, df: pd.DataFrame, *, numeric_cols: Sequence[str], categorical_cols: Sequence[str] = ()) -> np.ndarray:
        if not self._is_fit or self._artifact is None:
            raise RuntimeError("Model must be fit first.")
        if list(numeric_cols) != self._artifact.numeric_cols:
            raise ValueError("numeric_cols must match exactly the columns used during fit().")
        if list(categorical_cols) != self._artifact.cat_cols:
            raise ValueError("categorical_cols must match exactly the columns used during fit().")

        X_num, X_cat, _, _ = self._build_matrices(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols, is_fit_time=False)
        z = latent_features(self._artifact, X_num, X_cat, batch_size=self.cfg.batch_size)
        return z

    def recon_error(self, df: pd.DataFrame, *, numeric_cols: Sequence[str], categorical_cols: Sequence[str] = ()) -> np.ndarray:
        if not self._is_fit or self._artifact is None:
            raise RuntimeError("Model must be fit first.")
        if list(numeric_cols) != self._artifact.numeric_cols:
            raise ValueError("numeric_cols must match exactly the columns used during fit().")
        if list(categorical_cols) != self._artifact.cat_cols:
            raise ValueError("categorical_cols must match exactly the columns used during fit().")
        X_num, X_cat, _, _ = self._build_matrices(
            df, numeric_cols=numeric_cols, categorical_cols=categorical_cols, is_fit_time=False
        )
        scores = reconstruction_error_rows(self._artifact, X_num, X_cat, batch_size=self.cfg.batch_size)
        return scores.astype(float, copy=False)

    def recon_error_matrix(
        self,
        df: pd.DataFrame,
        *,
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str] = (),
        space: str = "standardized",
    ) -> pd.DataFrame:
        """Per-row per-feature squared reconstruction error.

        Args:
          space:
            - standardized: compare in AE standardized input space.
            - original: compare in original numeric feature scale.
        """
        if not self._is_fit or self._artifact is None:
            raise RuntimeError("Model must be fit first.")
        if list(numeric_cols) != self._artifact.numeric_cols:
            raise ValueError("numeric_cols must match exactly the columns used during fit().")
        if list(categorical_cols) != self._artifact.cat_cols:
            raise ValueError("categorical_cols must match exactly the columns used during fit().")
        if space not in {"standardized", "original"}:
            raise ValueError("space must be 'standardized' or 'original'.")

        X_num, X_cat, _, _ = self._build_matrices(
            df, numeric_cols=numeric_cols, categorical_cols=categorical_cols, is_fit_time=False
        )
        Xs = standardize_apply(
            X_num,
            self._artifact.mean_,
            self._artifact.std_,
            self._artifact.lower_,
            self._artifact.upper_,
        )

        import torch

        device = torch.device(self._artifact.device)
        n = Xs.shape[0]
        recon_std = np.empty_like(Xs, dtype=np.float32)

        self._artifact.model.eval()
        with torch.no_grad():
            for start in range(0, n, self.cfg.batch_size):
                end = min(start + self.cfg.batch_size, n)
                num_t = torch.tensor(Xs[start:end], dtype=torch.float32, device=device)
                if self._artifact.requires_cats:
                    if X_cat is None:
                        raise RuntimeError("Model requires categorical inputs but X_cat is None")
                    cat_t = torch.tensor(X_cat[start:end], dtype=torch.long, device=device)
                    recon = self._artifact.model(num_t, cat_t).detach().cpu().numpy()
                else:
                    recon = self._artifact.model(num_t).detach().cpu().numpy()
                recon_std[start:end] = recon.astype(np.float32, copy=False)

        if space == "standardized":
            diff = recon_std - Xs
        else:
            recon_orig = (recon_std * self._artifact.std_) + self._artifact.mean_
            X_orig = np.nan_to_num(X_num, nan=self._artifact.mean_)
            diff = recon_orig - X_orig

        diff = np.nan_to_num(diff, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float64, copy=False)

        return pd.DataFrame(
            (diff * diff).astype(np.float64, copy=False),
            index=df.index,
            columns=self._artifact.numeric_cols,
        )

    def recon_error_percentile(
        self,
        df: pd.DataFrame,
        *,
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str] = (),
    ) -> np.ndarray:
        """Row-wise reconstruction-error percentile in [0, 1].

        Uses global calibration against AE train-error distribution when available,
        otherwise falls back to in-batch percentile ranking.
        """
        mse = self.recon_error(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols)
        ref = getattr(self._artifact, "train_error_sorted", None)
        if ref is not None and len(ref):
            ref_arr = np.asarray(ref, dtype=np.float64)
            pct = np.searchsorted(ref_arr, np.asarray(mse, dtype=np.float64), side="right") / float(len(ref_arr))
            return np.clip(pct, 0.0, 1.0).astype(float)
        return pd.Series(mse, index=df.index).rank(pct=True).to_numpy(dtype=float)

    def weighted_recon_error(
        self,
        df: pd.DataFrame,
        *,
        numeric_cols: Sequence[str],
        feature_weights: Dict[str, float] | pd.Series,
        categorical_cols: Sequence[str] = (),
        space: str = "standardized",
    ) -> np.ndarray:
        """Row-wise weighted reconstruction MSE."""
        err_df = self.recon_error_matrix(
            df,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            space=space,
        )
        if isinstance(feature_weights, pd.Series):
            w = feature_weights.reindex(err_df.columns).fillna(0.0).to_numpy(dtype=np.float64)
        else:
            w = np.asarray([float(feature_weights.get(c, 0.0)) for c in err_df.columns], dtype=np.float64)
        w = np.clip(w, a_min=0.0, a_max=None)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        denom = float(np.sum(w))
        if denom <= 0:
            raise ValueError("feature_weights must contain positive values.")
        vals = err_df.to_numpy(dtype=np.float64, copy=False)
        vals = np.nan_to_num(vals, nan=0.0, posinf=1e6, neginf=0.0)
        return (vals @ w / denom).astype(float, copy=False)

    def fracture_score(
        self,
        df: pd.DataFrame,
        *,
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str] = (),
        top_k: int | None = None,
        top_frac: float | None = 0.05,
        space: str = "standardized",
    ) -> np.ndarray:
        """Row-wise mean of top-K (or top fraction) largest feature squared errors."""
        err = self.recon_error_matrix(
            df,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            space=space,
        ).to_numpy(dtype=np.float64, copy=False)
        err = np.nan_to_num(err, nan=0.0, posinf=1e6, neginf=0.0)
        n_feat = err.shape[1]
        if top_k is None:
            if top_frac is None or top_frac <= 0:
                raise ValueError("Provide top_k or positive top_frac.")
            top_k = max(1, int(np.ceil(n_feat * float(top_frac))))
        top_k = int(max(1, min(top_k, n_feat)))
        part = np.partition(err, kth=n_feat - top_k, axis=1)[:, -top_k:]
        return np.mean(part, axis=1).astype(float, copy=False)

    def precision_scaled_recon_error(
        self,
        df: pd.DataFrame,
        *,
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str] = (),
        eps: float = 1e-8,
        space: str = "standardized",
    ) -> np.ndarray:
        """Row-wise recon error weighted by inverse optimal-train feature residual variance."""
        if not self._is_fit or self._artifact is None:
            raise RuntimeError("Model must be fit first.")
        ref = getattr(self._artifact, "feature_weights", None)
        if ref is None:
            raise RuntimeError("feature_weights unavailable; re-fit AE to build precision weights.")
        w = np.asarray(ref, dtype=np.float64)
        w = np.clip(w, a_min=eps, a_max=None)
        w = np.nan_to_num(w, nan=eps, posinf=1.0 / eps, neginf=eps)
        err = self.recon_error_matrix(
            df,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            space=space,
        ).to_numpy(dtype=np.float64, copy=False)
        err = np.nan_to_num(err, nan=0.0, posinf=1e6, neginf=0.0)
        return (err @ w / float(np.sum(w))).astype(float, copy=False)

    def latent_distance(
        self,
        df: pd.DataFrame,
        *,
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str] = (),
    ) -> np.ndarray:
        """Distance to nearest optimal-manifold latent reference point."""
        z = self.latent(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols)
        nn = self._get_latent_nn()
        d, _ = nn.kneighbors(np.asarray(z, dtype=np.float32), return_distance=True)
        return d[:, 0].astype(float, copy=False)

    def latent_distance_percentile(
        self,
        df: pd.DataFrame,
        *,
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str] = (),
    ) -> np.ndarray:
        """Percentile rank of latent distance vs train optimal manifold distances."""
        d = self.latent_distance(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols)
        ref = getattr(self._artifact, "latent_dist_train_sorted", None)
        if ref is not None and len(ref):
            ref_arr = np.asarray(ref, dtype=np.float64)
            pct = np.searchsorted(ref_arr, np.asarray(d, dtype=np.float64), side="right") / float(len(ref_arr))
            return np.clip(pct, 0.0, 1.0).astype(float)
        return pd.Series(d, index=df.index).rank(pct=True).to_numpy(dtype=float)

    def latent_mahalanobis_distance(
        self,
        df: pd.DataFrame,
        *,
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str] = (),
    ) -> np.ndarray:
        """Mahalanobis distance in latent space vs optimal-train latent covariance."""
        if not self._is_fit or self._artifact is None:
            raise RuntimeError("Model must be fit first.")
        mu = getattr(self._artifact, "latent_mean", None)
        cov_inv = getattr(self._artifact, "latent_cov_inv", None)
        if mu is None or cov_inv is None or len(mu) == 0:
            raise RuntimeError("Latent Mahalanobis calibration unavailable; re-fit AE.")
        z = self.latent(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols).astype(np.float64, copy=False)
        z = np.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6)
        centered = z - np.asarray(mu, dtype=np.float64)
        inv = np.asarray(cov_inv, dtype=np.float64)
        inv = np.nan_to_num(inv, nan=0.0, posinf=0.0, neginf=0.0)
        d2 = np.einsum("ij,jk,ik->i", centered, inv, centered, optimize=True)
        d2 = np.clip(d2, a_min=0.0, a_max=None)
        return np.sqrt(d2).astype(float, copy=False)

    def latent_mahalanobis_percentile(
        self,
        df: pd.DataFrame,
        *,
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str] = (),
    ) -> np.ndarray:
        """Percentile rank of latent Mahalanobis vs optimal-train latent Mahalanobis distances."""
        d = self.latent_mahalanobis_distance(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols)
        ref = getattr(self._artifact, "latent_mahal_train_sorted", None)
        if ref is not None and len(ref):
            ref_arr = np.asarray(ref, dtype=np.float64)
            pct = np.searchsorted(ref_arr, np.asarray(d, dtype=np.float64), side="right") / float(len(ref_arr))
            return np.clip(pct, 0.0, 1.0).astype(float)
        return pd.Series(d, index=df.index).rank(pct=True).to_numpy(dtype=float)

    def familiarity(
        self,
        df: pd.DataFrame,
        *,
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str] = (),
        quantile: float = 99.9,
        mode: str = "quantile_soft",
    ) -> np.ndarray:
        """Row-wise familiarity score in [0, 1], where higher means more familiar.

        Modes:
          - quantile_soft (default): familiarity = clip(1 - err / q_train, 0, 1),
            where q_train is train-error quantile (default q99.9).
          - reciprocal_soft: familiarity = 1 / (1 + err / q_train),
            smoother than clipped linear scaling.
          - cdf: familiarity = 1 - empirical CDF percentile against train errors.
          - latent_reciprocal_soft: familiarity from latent-distance / q_train_latent.
          - hybrid_reciprocal_soft: geometric mean of recon + latent reciprocal familiarity.
        """
        mse = self.recon_error(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols).astype(float, copy=False)
        ref = getattr(self._artifact, "train_error_sorted", None)

        if mode in {"quantile_soft", "reciprocal_soft"}:
            if ref is not None and len(ref):
                cutoff = float(np.percentile(np.asarray(ref, dtype=np.float64), float(quantile)))
            else:
                cutoff = float(np.percentile(np.asarray(mse, dtype=np.float64), float(quantile)))
            cutoff = max(cutoff, 1e-12)
            ratio = mse / cutoff
            if mode == "quantile_soft":
                return np.clip(1.0 - ratio, 0.0, 1.0).astype(float, copy=False)
            return (1.0 / (1.0 + ratio)).astype(float, copy=False)

        if mode == "cdf":
            mse_pct = self.recon_error_percentile(
                df,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
            )
            return (1.0 - mse_pct).astype(float, copy=False)

        if mode in {"latent_reciprocal_soft", "hybrid_reciprocal_soft"}:
            d = self.latent_distance(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols).astype(float, copy=False)
            d_ref = getattr(self._artifact, "latent_dist_train_sorted", None)
            if d_ref is not None and len(d_ref):
                d_cut = float(np.percentile(np.asarray(d_ref, dtype=np.float64), float(quantile)))
            else:
                d_cut = float(np.percentile(np.asarray(d, dtype=np.float64), float(quantile)))
            d_cut = max(d_cut, 1e-12)
            fam_lat = (1.0 / (1.0 + (d / d_cut))).astype(float, copy=False)
            if mode == "latent_reciprocal_soft":
                return fam_lat

            if ref is not None and len(ref):
                r_cut = float(np.percentile(np.asarray(ref, dtype=np.float64), float(quantile)))
            else:
                r_cut = float(np.percentile(np.asarray(mse, dtype=np.float64), float(quantile)))
            r_cut = max(r_cut, 1e-12)
            fam_rec = (1.0 / (1.0 + (mse / r_cut))).astype(float, copy=False)
            return np.sqrt(fam_lat * fam_rec).astype(float, copy=False)

        raise ValueError(
            "mode must be 'quantile_soft', 'reciprocal_soft', 'cdf', "
            "'latent_reciprocal_soft', or 'hybrid_reciprocal_soft'"
        )

    def reconstruct(self, df: pd.DataFrame, *, numeric_cols: Sequence[str], categorical_cols: Sequence[str] = ()) -> pd.DataFrame:
        if not self._is_fit or self._artifact is None:
            raise RuntimeError("Model must be fit first.")
        if list(numeric_cols) != self._artifact.numeric_cols:
            raise ValueError("numeric_cols must match exactly the columns used during fit().")
        if list(categorical_cols) != self._artifact.cat_cols:
            raise ValueError("categorical_cols must match exactly the columns used during fit().")

        X_num, X_cat, _, _ = self._build_matrices(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols, is_fit_time=False)
        Xs = standardize_apply(
            X_num,
            self._artifact.mean_,
            self._artifact.std_,
            self._artifact.lower_,
            self._artifact.upper_,
        )

        # Run model in batches for reconstruction (standardized)
        import torch
        device = torch.device(self._artifact.device)
        n = Xs.shape[0]
        out = np.empty_like(Xs, dtype=np.float32)

        self._artifact.model.eval()
        with torch.no_grad():
            for start in range(0, n, self.cfg.batch_size):
                end = min(start + self.cfg.batch_size, n)
                num_t = torch.tensor(Xs[start:end], dtype=torch.float32, device=device)
                if self._artifact.requires_cats:
                    if X_cat is None:
                        raise RuntimeError("Model requires categorical inputs but X_cat is None")
                    cat_t = torch.tensor(X_cat[start:end], dtype=torch.long, device=device)
                    recon = self._artifact.model(num_t, cat_t).detach().cpu().numpy()
                else:
                    recon = self._artifact.model(num_t).detach().cpu().numpy()
                out[start:end] = recon.astype(np.float32, copy=False)

        # inverse standardize -> original scale
        recon_original = (out * self._artifact.std_) + self._artifact.mean_
        return pd.DataFrame(recon_original, index=df.index, columns=self._artifact.numeric_cols)

    def summarize(self) -> None:
        if not self._is_fit or self._artifact is None:
            return

        print_model_section("AUTOENCODER REPORT")
        bottleneck_dim = getattr(self._artifact.model.encoder[-1], "out_features", self.cfg.bottleneck_dim)
        n_layers = int(getattr(self.cfg, "n_layers", 2))
        summary = f"""\
### 1. MODEL STATE
- Type: {'HYBRID' if self._artifact.requires_cats else 'NUMERIC_ONLY'}
- Numeric inputs: {len(self._artifact.numeric_cols)}
- Categorical inputs: {len(self._artifact.cat_cols)}
- Encoder layers: {n_layers}
- Bottleneck dim: {bottleneck_dim}
- MSE (Mean): {self._metrics['recon_error_mean']:.6f}
- MSE (P95):  {self._metrics['recon_error_p95']:.6f}
"""
        print(summary)
        feat_mse = getattr(self._artifact, "train_feature_mse", None)
        if feat_mse is not None and len(feat_mse) == len(self._artifact.numeric_cols):
            print("### 2. TOP FEATURE RECONSTRUCTION MSE (TRAIN, STANDARDIZED)")
            pairs = list(zip(self._artifact.numeric_cols, np.asarray(feat_mse, dtype=float)))
            pairs.sort(key=lambda x: x[1], reverse=True)
            for col, val in pairs[:10]:
                print(f"- {col}: {val:.6f}")
        print("============================================================")

    def metrics_report(self) -> dict:
        return dict(self._metrics)
