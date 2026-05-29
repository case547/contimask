from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import config
from data.dataset import SepsisDataset
from data.split import stratified_patient_split
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

    model = build_model()
    model.load_state_dict(
        torch.load(checkpoint_dir / "best_model.pt", map_location="cpu")
    )
    model.to(args.device)

    model.eval()
    all_preds, all_labels, all_files = [], [], []
    patient_idx = 0
    with torch.no_grad():
        for t, X, dm, labels in DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False):
            probs = torch.sigmoid(
                model.classify(
                    t.to(args.device), X.to(args.device), dm.to(args.device)
                ).squeeze(-1)
            ).cpu()
            for prob, label in zip(probs.tolist(), labels.tolist()):
                all_preds.append(prob)
                all_labels.append(int(label))
                all_files.append(str(test_ds.psv_files[patient_idx]))
                patient_idx += 1

    csv_path = checkpoint_dir / "test_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["psv_file", "true_label", "predicted_prob"]
        )
        writer.writeheader()
        for fname, label, prob in zip(all_files, all_labels, all_preds):
            writer.writerow(
                {"psv_file": fname, "true_label": label, "predicted_prob": prob}
            )

    auc = float(roc_auc_score(all_labels, all_preds))
    logger.info("Test AUROC: %.4f", auc)
    logger.info("Saved test predictions to %s", csv_path)


if __name__ == "__main__":
    main()
