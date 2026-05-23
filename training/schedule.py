from __future__ import annotations

import torch


class DDPMSchedule:
    def __init__(self, T: int = 1000, beta_min: float = 1e-4, beta_max: float = 0.02):
        betas = torch.linspace(beta_min, beta_max, T)
        alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(alphas, dim=0)  # (T,)

    def q_sample(
        self,
        x0: torch.Tensor,
        s: torch.Tensor,
        data_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x0: (B, T_seq, F) clean normalised features
        s:  (B,) diffusion step indices in [0, T_diff)
        data_mask: (B, T_seq, F) binary, 1 where observed
        Returns (x_noisy, eps) -- noise only applied to observed positions
        """
        B = x0.shape[0]
        device = x0.device
        alpha_bar = self.alpha_bars.to(device)[s].view(B, 1, 1)  # (B, 1, 1)
        eps = torch.randn_like(x0)
        x_noisy = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * eps
        x_noisy = x_noisy * data_mask  # zero out unobserved positions
        return x_noisy, eps
