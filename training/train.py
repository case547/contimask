from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from data.dataset import SepsisDataset, compute_norm_stats
from data.split import stratified_patient_split
from models.diffusion_transformer import DiffusionTransformer
from training.finetune import finetune
from training.pretrain import pretrain
from training.schedule import DDPMSchedule
from utils.tools import setup_logging

logger = logging.getLogger(__name__)


def build_model() -> DiffusionTransformer:
    return DiffusionTransformer(
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        ffn_dim=config.FFN_DIM,
        dropout=config.DROPOUT,
        n_features=config.N_FEATURES,
        time_embed_L=config.TIME_EMBED_L,
        T_diff=config.T_DIFF,
    )


def build_loaders(data_dir: str, checkpoint_dir: Path):
    raw_ds = SepsisDataset(data_dir)
    train_files, val_files, _ = stratified_patient_split(
        raw_ds.psv_files, raw_ds.labels
    )

    norm_stats_path = checkpoint_dir / "norm_stats.pt"
    train_ds = SepsisDataset(data_dir, train_files)
    if norm_stats_path.exists():
        logger.info("Norm stats: loaded from cache (%s)", norm_stats_path)
        norm_stats = torch.load(norm_stats_path, map_location="cpu")
    else:
        logger.info(
            "Norm stats: computing from %d training patients...", len(train_files)
        )
        norm_stats = compute_norm_stats(train_ds)
        torch.save(norm_stats, norm_stats_path)
        logger.info("Norm stats: saved to %s", norm_stats_path)
    train_ds.norm_stats = norm_stats

    val_ds = SepsisDataset(data_dir, val_files, norm_stats)
    logger.info("Split: train=%d  val=%d patients", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=["pretrain", "finetune", "both"], default="both"
    )
    parser.add_argument("--data_dir", default="~/msc_ai/individual-project/sepsis_data")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--pretrain_epochs", type=int, default=config.PRETRAIN_EPOCHS)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    setup_logging(log_path=str(checkpoint_dir / "train.log"))
    logger.info("Phase: %s  device: %s", args.phase, args.device)

    data_dir = str(Path(args.data_dir).expanduser())
    train_loader, val_loader = build_loaders(data_dir, checkpoint_dir)
    model = build_model()

    pretrain_ckpt = checkpoint_dir / "pretrained_backbone.pt"
    finetune_ckpt = checkpoint_dir / "best_model.pt"

    if args.phase in ("pretrain", "both"):
        schedule = DDPMSchedule(T=config.T_DIFF)
        pretrain(
            model,
            train_loader,
            val_loader,
            schedule,
            max_epochs=args.pretrain_epochs,
            lr=config.PRETRAIN_LR,
            patience=config.EARLY_STOPPING_PATIENCE,
            checkpoint_path=str(pretrain_ckpt),
            grad_clip=config.GRAD_CLIP,
            device=args.device,
        )

    if args.phase in ("finetune", "both"):
        if pretrain_ckpt.exists() and args.phase == "finetune":
            model.load_state_dict(torch.load(pretrain_ckpt, map_location="cpu"))
            logger.info("Loaded pretrained weights from %s", pretrain_ckpt)
        finetune(
            model,
            train_loader,
            val_loader,
            config.FINETUNE_LR_RATIOS,
            base_lr=config.FINETUNE_LR,
            weight_decay=config.WEIGHT_DECAY,
            max_epochs=config.FINETUNE_MAX_EPOCHS,
            patience=config.EARLY_STOPPING_PATIENCE,
            checkpoint_path=str(finetune_ckpt),
            grad_clip=config.GRAD_CLIP,
            device=args.device,
        )


if __name__ == "__main__":
    main()
