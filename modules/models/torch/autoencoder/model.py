from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init


def _build_halving_dims(in_dim: int, n_layers: int, min_layer_dim: int) -> list[int]:
    dims: list[int] = []
    prev = int(in_dim)
    for _ in range(int(n_layers)):
        nxt = max(int(min_layer_dim), prev // 2)
        dims.append(nxt)
        prev = nxt
    return dims


def _build_mlp_encoder(in_dim: int, dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = int(in_dim)
    for i, d in enumerate(dims):
        layers.append(nn.Linear(prev, int(d)))
        if i < len(dims) - 1:
            layers.append(nn.BatchNorm1d(int(d)))
            layers.append(nn.ReLU())
        prev = int(d)
    return nn.Sequential(*layers)


def _build_mlp_decoder(out_dim: int, dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = int(dims[-1])
    for d in reversed(dims[:-1]):
        layers.append(nn.Linear(prev, int(d)))
        layers.append(nn.BatchNorm1d(int(d)))
        layers.append(nn.ReLU())
        prev = int(d)
    layers.append(nn.Linear(prev, int(out_dim)))
    return nn.Sequential(*layers)


class NumericAutoEncoder(nn.Module):
    def __init__(self, in_dim: int, n_layers: int, min_layer_dim: int = 2) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.n_layers = int(n_layers)
        self.layer_dims = _build_halving_dims(self.in_dim, self.n_layers, int(min_layer_dim))
        self.bottleneck_dim = int(self.layer_dims[-1])

        self.encoder = _build_mlp_encoder(self.in_dim, self.layer_dims)
        self.decoder = _build_mlp_decoder(self.in_dim, self.layer_dims)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x_num)
        return self.decoder(z)


class DynamicHybridAutoEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        cat_cardinalities: list[int],
        embed_dim: int,
        n_layers: int,
        min_layer_dim: int = 2,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("DynamicHybridAutoEncoder requires embed_dim > 0")

        self.in_dim = int(in_dim)
        self.embed_dim = int(embed_dim)
        self.n_layers = int(n_layers)

        self.cat_cardinalities = [int(x) for x in cat_cardinalities]

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, self.embed_dim) for cardinality in self.cat_cardinalities
        ])
        for emb in self.cat_embeddings:
            init.normal_(emb.weight, mean=0.0, std=0.01)

        self.cat_feature_dim = len(self.cat_cardinalities) * self.embed_dim
        self.cat_norm = nn.LayerNorm(self.cat_feature_dim) if self.cat_feature_dim > 0 else nn.Identity()

        total_in = self.in_dim + self.cat_feature_dim
        self.layer_dims = _build_halving_dims(total_in, self.n_layers, int(min_layer_dim))
        self.bottleneck_dim = int(self.layer_dims[-1])

        self.encoder = _build_mlp_encoder(total_in, self.layer_dims)
        self.decoder = _build_mlp_decoder(self.in_dim, self.layer_dims)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, x_num: torch.Tensor, x_cats: torch.Tensor) -> torch.Tensor:
        if x_cats.ndim != 2 or x_cats.shape[1] != len(self.cat_embeddings):
            raise ValueError(
                f"x_cats must have shape (N, {len(self.cat_embeddings)}); got {tuple(x_cats.shape)}"
            )

        embed_vecs = [emb(x_cats[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_features = torch.cat(embed_vecs, dim=1) if embed_vecs else torch.empty((x_num.shape[0], 0), device=x_num.device)
        cat_features = self.cat_norm(cat_features)

        x_combined = torch.cat([x_num, cat_features], dim=1)
        z = self.encoder(x_combined)
        return self.decoder(z)

    def encode(self, x_num: torch.Tensor, x_cats: torch.Tensor) -> torch.Tensor:
        embed_vecs = [emb(x_cats[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_features = torch.cat(embed_vecs, dim=1) if embed_vecs else torch.empty((x_num.shape[0], 0), device=x_num.device)
        cat_features = self.cat_norm(cat_features)
        x_combined = torch.cat([x_num, cat_features], dim=1)
        return self.encoder(x_combined)
