from __future__ import annotations

import argparse
import json
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


def compute_retention(
    masks: list[torch.Tensor],
    data_masks: list[torch.Tensor],
) -> torch.Tensor:
    F = masks[0].shape[-1]
    totals = torch.zeros(F)
    counts = torch.zeros(F)
    for mask, dm in zip(masks, data_masks):
        m = mask.squeeze(0)        # (T, F)
        d = dm.squeeze(0)          # (T, F)
        n_obs = d.sum(dim=0)       # (F,)
        n_ret = (m * d).sum(dim=0) # (F,)
        observed = n_obs > 0
        totals[observed] += n_ret[observed] / n_obs[observed]
        counts[observed] += 1.0
    return torch.where(counts > 0, totals / counts, torch.zeros(F))


def _run_group(
    name: str,
    indices: torch.Tensor,
    probs: torch.Tensor,
    test_ds: SepsisDataset,
    model: torch.nn.Module,
    device: str,
) -> tuple[list[float], list[float], list[torch.Tensor], list[torch.Tensor]]:
    del_vals, imp_vals = [], []
    masks_list, dms_list = [], []
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
        masks_list.append(mask.cpu())
        dms_list.append(dm.cpu())
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

    return del_vals, imp_vals, masks_list, dms_list


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

    del_top, imp_top, masks_top, dms_top = _run_group(
        "top 100 (most likely septic)", top_idx, probs, test_ds, model, args.device
    )
    del_bot, imp_bot, masks_bot, dms_bot = _run_group(
        "bottom 100 (least likely septic)",
        bottom_idx,
        probs,
        test_ds,
        model,
        args.device,
    )

    retention_top = compute_retention(masks_top, dms_top)
    retention_bot = compute_retention(masks_bot, dms_bot)

    top_files = [str(test_ds.psv_files[i]) for i in top_idx.tolist()]
    bot_files = [str(test_ds.psv_files[i]) for i in bottom_idx.tolist()]

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    logger.info("=== Odds-change results ===")
    logger.info("Del odds change  |  top 100:    %+.4f", _mean(del_top))
    logger.info("Del odds change  |  bottom 100: %+.4f", _mean(del_bot))
    logger.info("Imp odds change  |  top 100:    %+.4f", _mean(imp_top))
    logger.info("Imp odds change  |  bottom 100: %+.4f", _mean(imp_bot))

    logger.info("=== Retention — top 100 (ranked) ===")
    ranked = sorted(
        zip(config.FEATURE_COLS, retention_top.tolist()), key=lambda x: -x[1]
    )
    for feat, rate in ranked:
        logger.info("  %-25s %.4f", feat, rate)

    results = {
        "feature_names": config.FEATURE_COLS,
        "retention_top100": dict(zip(config.FEATURE_COLS, retention_top.tolist())),
        "retention_bottom100": dict(zip(config.FEATURE_COLS, retention_bot.tolist())),
        "top100_psv_files": top_files,
        "bottom100_psv_files": bot_files,
    }
    json_path = checkpoint_dir / "attribution_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved attribution results → %s", json_path)


if __name__ == "__main__":
    main()
