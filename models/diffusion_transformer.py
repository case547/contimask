from __future__ import annotations

import torch
import torch.nn as nn

from models.embeddings import DiffusionStepEmbedding, SinusoidalTimeEmbedding


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        dropout: float,
        n_features: int,
        time_embed_L: int,
        T_diff: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features

        self.input_proj = nn.Linear(n_features, d_model)
        self.time_embed = SinusoidalTimeEmbedding(L=time_embed_L, d_model=d_model)
        self.step_embed = DiffusionStepEmbedding(T_diff=T_diff, L=time_embed_L, d_model=d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.denoise_head = nn.Linear(d_model, n_features)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def encode(
        self,
        t: torch.Tensor,
        X: torch.Tensor,
        data_mask: torch.Tensor,
        s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = X.shape

        tokens = self.input_proj(X) + self.time_embed(t)  # (B, T, d_model)
        if s is not None:
            tokens = tokens + self.step_embed(s)  # (B, 1, d_model) broadcasts over T

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, T+1, d_model)

        # src_key_padding_mask: True = ignore this position
        pad = ~data_mask.any(dim=-1)  # (B, T) True for fully padded rows
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=X.device)
        src_key_padding_mask = torch.cat([cls_pad, pad], dim=1)  # (B, T+1)

        out = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)
        return out  # (B, T+1, d_model)

    def denoise(
        self,
        t: torch.Tensor,
        X: torch.Tensor,
        data_mask: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encode(t, X, data_mask, s=s)
        return self.denoise_head(out[:, 1:, :])  # (B, T, n_features)

    def classify(
        self,
        t: torch.Tensor,
        X: torch.Tensor,
        data_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encode(t, X, data_mask, s=None)
        return self.cls_head(out[:, 0, :])  # (B, 1)
