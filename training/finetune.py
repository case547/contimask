from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from utils.tools import EarlyStopping

logger = logging.getLogger(__name__)


def _build_optimizer(
    model: nn.Module,
    lr_ratios: list[float],
    base_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    embedding_params = (
        list(model.input_proj.parameters())
        + list(model.time_embed.parameters())
        + list(model.step_embed.parameters())
        + [model.cls_token]
    )
    param_groups = [{"params": embedding_params, "lr": base_lr * lr_ratios[0]}]
    for i, layer in enumerate(model.transformer.layers):
        ratio = lr_ratios[i]
        if ratio == 0.0:
            for p in layer.parameters():
                p.requires_grad_(False)
        else:
            param_groups.append(
                {"params": list(layer.parameters()), "lr": base_lr * ratio}
            )
    param_groups.append({"params": list(model.cls_head.parameters()), "lr": base_lr})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def _finetune_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pos_weight: torch.Tensor,
    grad_clip: float,
    device: str,
) -> float:
    model.train()
    total = 0.0
    for t, X, dm, labels in loader:
        t, X, dm = t.to(device), X.to(device), dm.to(device)
        labels = labels.float().to(device)
        logit = model.classify(t, X, dm).squeeze(-1)  # (B,)
        loss = F.binary_cross_entropy_with_logits(
            logit, labels, pos_weight=pos_weight.to(device)
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    all_preds, all_labels = [], []
    for t, X, dm, labels in loader:
        t, X, dm = t.to(device), X.to(device), dm.to(device)
        prob = torch.sigmoid(model.classify(t, X, dm).squeeze(-1)).cpu()
        all_preds.append(prob)
        all_labels.append(labels)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    if len(set(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, preds))


def finetune(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr_ratios: list[float],
    base_lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    checkpoint_path: str,
    grad_clip: float = 1.0,
    device: str = "cpu",
) -> float:
    model.to(device)

    # Compute pos_weight from training labels
    all_labels = torch.cat([labels for _, _, _, labels in train_loader])
    n_pos = all_labels.sum().item()
    n_neg = len(all_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)])
    logger.info(
        "Finetune: n_pos=%d  n_neg=%d  pos_weight=%.2f  device=%s",
        int(n_pos),
        int(n_neg),
        pos_weight.item(),
        device,
    )

    optimizer = _build_optimizer(model, lr_ratios, base_lr, weight_decay)
    es = EarlyStopping(
        patience, path=checkpoint_path, verbose=True, trace_func=logger.info
    )

    best_auc = 0.0
    for epoch in range(1, max_epochs + 1):
        train_loss = _finetune_epoch(
            model, train_loader, optimizer, pos_weight, grad_clip, device
        )
        val_auc = _evaluate(model, val_loader, device)
        logger.info(
            "Finetune epoch %d/%d  train=%.4f  val_auc=%.4f",
            epoch,
            max_epochs,
            train_loss,
            val_auc,
        )

        es(-val_auc)  # EarlyStopping minimises; negate AUC so higher = better
        if es.counter == 0:
            es.save_checkpoint(-val_auc, model, epoch)
            best_auc = val_auc
        if es.early_stop:
            logger.info("Early stopping.")
            break

    logger.info(
        "Finetune done. Best val AUC: %.4f @ epoch %d.", best_auc, es.best_epoch
    )
    return best_auc
