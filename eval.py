from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from data.dataset import SepsisDataset
from data.split import stratified_patient_split
from training.finetune import _evaluate
from training.train import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="~/msc_ai/individual-project/sepsis_data")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    data_dir = str(Path(args.data_dir).expanduser())

    raw_ds = SepsisDataset(data_dir)
    _, _, test_files = stratified_patient_split(raw_ds.psv_files, raw_ds.labels)

    norm_stats = torch.load(checkpoint_dir / "norm_stats.pt", map_location="cpu")
    test_ds = SepsisDataset(data_dir, psv_files=test_files, norm_stats=norm_stats)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE)

    model = build_model()
    model.load_state_dict(
        torch.load(checkpoint_dir / "best_model.pt", map_location="cpu")
    )
    model.to(args.device)

    auc = _evaluate(model, test_loader, args.device)
    print(f"Test AUROC: {auc:.4f}")


if __name__ == "__main__":
    main()
