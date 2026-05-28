from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from data.dataset import SepsisDataset
from data.split import stratified_patient_split
from sepsis_attribution import (
    compute_del_odds_change,
    compute_imp_odds_change,
    make_forward_func,
    run_attribution,
)
from training.train import build_model
from utils.tools import setup_logging

logger = logging.getLogger(__name__)


def _score_all(
    model: torch.nn.Module, test_ds: SepsisDataset, device: str
) -> torch.Tensor:
    """Return predicted sepsis probabilities for every patient in test_ds."""
    forward_func = make_forward_func(model, device=device)
    all_probs = []
    for t, X, dm, _ in DataLoader(test_ds, batch_size=config.BATCH_SIZE):
        all_probs.append(forward_func(t, X, dm).cpu())
    return torch.cat(all_probs)  # (N,)


def _run_group(
    name: str,
    indices: torch.Tensor,
    probs: torch.Tensor,
    test_ds: SepsisDataset,
    model: torch.nn.Module,
    device: str,
) -> tuple[list[float], list[float]]:
    del_vals, imp_vals = [], []
    logger.info("Running attribution for %s...", name)
    t0 = time.perf_counter()

    for rank, idx in enumerate(indices.tolist()):
        t, X, dm, _ = test_ds[idx]
        t, X, dm = t.unsqueeze(0), X.unsqueeze(0), dm.unsqueeze(0)

        t1 = time.perf_counter()
        _, mask = run_attribution(model, t, X, dm, device=device)
        elapsed = time.perf_counter() - t1

        del_val = compute_del_odds_change(model, t, X, dm, mask, device=device).item()
        imp_val = compute_imp_odds_change(model, t, X, dm, mask, device=device).item()

        del_vals.append(del_val)
        imp_vals.append(imp_val)
        logger.info(
            "  [%3d/100] prob=%.3f  del=%+.4f  imp=%+.4f  (%.0fs)",
            rank + 1,
            probs[idx],
            del_val,
            imp_val,
            elapsed,
        )

    total_elapsed = time.perf_counter() - t0
    logger.info("Finished %s attribution. Total time: %.1fs", name, total_elapsed)

    return del_vals, imp_vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="~/msc_ai/individual-project/sepsis_data")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    setup_logging(log_path=str(checkpoint_dir / "attribute.log"))
    logger.info("Device: %s", args.device)
    data_dir = str(Path(args.data_dir).expanduser())

    model = build_model()
    model.load_state_dict(
        torch.load(checkpoint_dir / "best_model.pt", map_location="cpu")
    )
    model.to(args.device)

    raw_ds = SepsisDataset(data_dir)
    _, _, test_files = stratified_patient_split(raw_ds.psv_files, raw_ds.labels)

    norm_stats = torch.load(checkpoint_dir / "norm_stats.pt", map_location="cpu")
    test_ds = SepsisDataset(data_dir, test_files, norm_stats)

    logger.info("Test set: %d patients. Scoring...", len(test_ds))
    probs = _score_all(model, test_ds, args.device)

    sorted_idx = probs.argsort(descending=True)
    top_idx = sorted_idx[:100]
    bottom_idx = sorted_idx[-100:]

    del_top, imp_top = _run_group(
        "top 100 (most likely septic)", top_idx, probs, test_ds, model, args.device
    )
    del_bot, imp_bot = _run_group(
        "bottom 100 (least likely septic)",
        bottom_idx,
        probs,
        test_ds,
        model,
        args.device,
    )

    logger.info("=== Odds-change results ===")
    logger.info("Del odds change  |  top 100:    %+.4f", sum(del_top) / len(del_top))
    logger.info("Del odds change  |  bottom 100: %+.4f", sum(del_bot) / len(del_bot))
    logger.info("Imp odds change  |  top 100:    %+.4f", sum(imp_top) / len(imp_top))
    logger.info("Imp odds change  |  bottom 100: %+.4f", sum(imp_bot) / len(imp_bot))


if __name__ == "__main__":
    main()
