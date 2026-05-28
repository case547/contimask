from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from data.dataset import SepsisDataset
from data.split import stratified_patient_split
from training.finetune import _evaluate
from training.train import build_model
from utils.tools import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="~/msc_ai/individual-project/sepsis_data")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    setup_logging(log_path=str(checkpoint_dir / "eval.log"))
    data_dir = str(Path(args.data_dir).expanduser())

    raw_ds = SepsisDataset(data_dir)
    _, _, test_files = stratified_patient_split(raw_ds.psv_files, raw_ds.labels)

    norm_stats = torch.load(checkpoint_dir / "norm_stats.pt", map_location="cpu")
    test_ds = SepsisDataset(data_dir, psv_files=test_files, norm_stats=norm_stats)
    logger.info("Test set: %d patients  device: %s", len(test_ds), args.device)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE)

    model = build_model()
    model.load_state_dict(
        torch.load(checkpoint_dir / "best_model.pt", map_location="cpu")
    )
    model.to(args.device)

    auc = _evaluate(model, test_loader, args.device)
    logger.info("Test AUROC: %.4f", auc)


if __name__ == "__main__":
    main()
