"""
Sepsis attribution utilities: forward_func wrapper and ContiMask runner.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn

import config
from attribution.mask_conti import ContiMask
from attribution.perturbation_conti import Deletion, MaskFunctionFourier

logger = logging.getLogger(__name__)


def make_forward_func(model: nn.Module, device: str = "cpu") -> Callable:
    """Return a forward_func compatible with ContiMask's (t, X, data_mask) signature.

    Args:
        model: A DiffusionTransformer (or any module with a ``classify`` method
               that accepts ``(t, X, data_mask)`` and returns logits of shape ``(B, 1)``).
        device: Device string used to move tensors before inference.

    Returns:
        forward_func callable with signature::

            forward_func(t, X, data_mask) -> probabilities

        * Standard input: ``X.dim() == 3``, shape ``(B, T, F)``
          → returns ``(B,)`` float tensor in [0, 1].
        * K-sample input: ``X.dim() == 4``, shape ``(K, B, T, F)``
          → returns ``(K, B)`` float tensor in [0, 1].
    """
    model = model.to(device)
    model.eval()

    def forward_func(
        t: torch.Tensor,
        X: torch.Tensor,
        data_mask: torch.Tensor,
    ) -> torch.Tensor:
        if X.dim() == 4:
            K, B, T, F = X.shape
            with torch.no_grad():
                logit = model.classify(
                    t.view(K * B, T).to(device),
                    X.view(K * B, T, F).to(device),
                    data_mask.view(K * B, T, F).to(device),
                )
            return torch.sigmoid(logit).view(K, B).cpu()
        # Standard (B, T, F) case
        with torch.no_grad():
            logit = model.classify(
                t.to(device),
                X.to(device),
                data_mask.to(device),
            )
        return torch.sigmoid(logit).squeeze(-1).cpu()  # (B,)

    return forward_func


@torch.no_grad()
def compute_del_odds_change(
    model: nn.Module,
    t: torch.Tensor,
    X: torch.Tensor,
    data_mask: torch.Tensor,
    mask: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Average change in log-odds when the attribution mask is applied as Deletion.

    Del zeros both X and data_mask where mask=0 — the transformer treats those
    positions as never observed.  Returns mean(logit_del - logit_orig) as a scalar.
    """
    was_training = model.training
    model = model.to(device).eval()
    t, X, data_mask, mask = (
        t.to(device),
        X.to(device),
        data_mask.to(device),
        mask.to(device),
    )
    try:
        logit_orig = model.classify(t, X, data_mask).squeeze(-1)  # (B,)
        logit_del = model.classify(t, X * mask, data_mask * mask).squeeze(-1)  # (B,)
        return (logit_del - logit_orig).mean()
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def compute_imp_odds_change(
    model: nn.Module,
    t: torch.Tensor,
    X: torch.Tensor,
    data_mask: torch.Tensor,
    mask: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Average change in log-odds when the attribution mask is applied as Imputation.

    Imp zeros X where mask=0 (inserting the feature mean in z-score space) but
    leaves data_mask unchanged.  Returns mean(logit_imp - logit_orig) as a scalar.
    """
    was_training = model.training
    model = model.to(device).eval()
    t, X, data_mask, mask = (
        t.to(device),
        X.to(device),
        data_mask.to(device),
        mask.to(device),
    )
    try:
        logit_orig = model.classify(t, X, data_mask).squeeze(-1)  # (B,)
        logit_imp = model.classify(t, X * mask, data_mask).squeeze(-1)  # (B,)
        return (logit_imp - logit_orig).mean()
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def subsample_mask(
    mask: torch.Tensor,
    data_mask: torch.Tensor,
    target_area: float = config.MASK_TARGET_AREA,
    seed: int | None = 42,
) -> torch.Tensor:
    """Randomly subsample active mask entries to exactly target_area of observed entries.

    If the mask already covers <= target_area of observed entries, that batch item is
    returned verbatim (mask=1 & data_mask=0 positions are preserved as-is). For items
    that are subsampled, the output is built from scratch so mask=1 & data_mask=0
    positions are zeroed.
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    B, T, F = mask.shape
    out = mask.clone()

    for b in range(B):
        n_observed = int(data_mask[b].sum().item())
        n_target = round(target_area * n_observed)

        active_indices = (
            (mask[b] * data_mask[b]).view(-1).nonzero(as_tuple=False).view(-1)
        )  # shape: (n_active,)
        n_active = len(active_indices)

        if n_active <= n_target:
            continue

        perm = torch.randperm(n_active, generator=gen)
        keep = active_indices[perm[:n_target]]

        flat = torch.zeros(T * F, dtype=mask.dtype)
        flat[keep] = 1.0
        out[b] = flat.view(T, F)

    return out


def run_attribution(
    model: nn.Module,
    t: torch.Tensor,
    X: torch.Tensor,
    data_mask: torch.Tensor,
    n_epoch: int = config.MASK_EPOCHS,
    K: int = 10,
    lr: float = 0.01,
    lambda_l1: float = config.LAMBDA_1,
    lambda_tv: float = config.LAMBDA_2,
    target_area: float = config.MASK_TARGET_AREA,
    mask_hidden_dim: int = config.MASK_HIDDEN_DIM,
    mask_L: int = config.MASK_L,
    n_features: int = 39,
    seed: int | None = 42,
    device: str = "cpu",
) -> tuple[ContiMask, torch.Tensor]:
    """Run ContiMask attribution for a batch of time series.

    Args:
        model: Trained DiffusionTransformer.
        t: Time tensor of shape ``(B, T)``.
        X: Feature tensor of shape ``(B, T, F)``.
        data_mask: Observation mask of shape ``(B, T, F)``.
        n_epoch: Number of optimisation epochs.
        K: Number of Monte-Carlo samples for the perturbation.
        lr: Adam learning rate for the mask network.
        lambda_l1: Sparsity regularisation weight.
        lambda_tv: Total-variation regularisation weight.
        target_area: Target fraction of the time series kept by the mask.
        mask_hidden_dim: Hidden dimension of the Fourier mask network.
        mask_L: Number of Fourier frequencies.
        n_features: Feature dimensionality (output channels of the mask net).
        seed: Seed for the post-hoc random subsampling step; ``None`` for
            non-deterministic behaviour.
        device: Compute device.

    Returns:
        A tuple ``(explainer, mask_values)`` where

        * ``explainer`` is the fitted :class:`~attribution.mask_conti.ContiMask` instance.
        * ``mask_values`` is a ``(B, T, F)`` float tensor of mask values in [0, 1]
          evaluated at the input time points.
    """
    logger.debug(
        "run_attribution: n_epoch=%d  K=%d  target_area=%.2f  device=%s",
        n_epoch,
        K,
        target_area,
        device,
    )
    forward_func = make_forward_func(model, device=device)

    pert_mask = MaskFunctionFourier(mask_hidden_dim, n_features, mask_L)

    perturbation_func = Deletion(device=device)

    explainer = ContiMask(
        forward_func, perturbation_func, pert_mask=pert_mask, device=device
    )

    explainer.attribute(
        t,
        X,
        data_mask,
        n_epoch,
        lr,
        K,
        lambda_l1,
        lambda_tv,
        popsize=config.PGPE_POP_SIZE,
        target_area=target_area,
        optimization_strategy="evotorch",
        deletion_mode=True,
    )

    with torch.no_grad():
        mask_values = (explainer.pert_mask(t.to(device)) > 0.5).float().cpu()
    mask_values = subsample_mask(mask_values, data_mask.cpu(), target_area, seed)
    return explainer, mask_values
