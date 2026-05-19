from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, L: int, d_model: int):
        super().__init__()
        self.L = L
        self.proj = nn.Linear(2 * L, d_model)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, T) values in [0, 1]
        device = t.device
        freqs = 2 * math.pi * (2.0 ** torch.arange(self.L, device=device))  # (L,)
        t_feats = t.unsqueeze(-1) * freqs  # (B, T, L)
        feats = torch.cat([t_feats.sin(), t_feats.cos()], dim=-1)  # (B, T, 2L)
        return self.proj(feats)  # (B, T, d_model)


class DiffusionStepEmbedding(nn.Module):
    def __init__(self, T_diff: int, L: int, d_model: int):
        super().__init__()
        self.T_diff = T_diff
        self.L = L
        self.proj = nn.Linear(2 * L, d_model)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: (B,) integer steps in [0, T_diff)
        device = s.device
        s_norm = s.float() / self.T_diff  # (B,)
        freqs = 2 * math.pi * (2.0 ** torch.arange(self.L, device=device))  # (L,)
        s_feats = s_norm.unsqueeze(-1) * freqs  # (B, L)
        feats = torch.cat([s_feats.sin(), s_feats.cos()], dim=-1)  # (B, 2L)
        return self.proj(feats).unsqueeze(1)  # (B, 1, d_model)
