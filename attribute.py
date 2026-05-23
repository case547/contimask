from __future__ import annotations

import argparse
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


def _score_all(model: torch.nn.Module, test_ds: SepsisDataset, device: str) -> torch.Tensor:
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
    print(f"\nRunning attribution for {name}...")
    for rank, idx in enumerate(indices.tolist()):
        t, X, dm, _ = test_ds[idx]
        t, X, dm = t.unsqueeze(0), X.unsqueeze(0), dm.unsqueeze(0)

        _, mask = run_attribution(model, t, X, dm, device=device)

        del_val = compute_del_odds_change(model, t, X, dm, mask, device=device).item()
        imp_val = compute_imp_odds_change(model, t, X, dm, mask, device=device).item()

        del_vals.append(del_val)
        imp_vals.append(imp_val)
        print(f"  [{rank + 1:3d}/100] prob={probs[idx]:.3f}  del={del_val:+.4f}  imp={imp_val:+.4f}")

    return del_vals, imp_vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="~/msc_ai/individual-project/sepsis_data/training_setA")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    data_dir = str(Path(args.data_dir).expanduser())

    model = build_model()
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt", map_location="cpu"))
    model.to(args.device)

    raw_ds = SepsisDataset(data_dir)
    labels = [int(raw_ds[i][3].item()) for i in range(len(raw_ds))]
    _, _, test_files = stratified_patient_split(raw_ds.psv_files, labels)

    norm_stats = torch.load(checkpoint_dir / "norm_stats.pt", map_location="cpu")
    test_ds = SepsisDataset(data_dir, psv_files=test_files, norm_stats=norm_stats)

    print(f"Scoring {len(test_ds)} test patients...")
    probs = _score_all(model, test_ds, device=args.device)

    sorted_idx = probs.argsort(descending=True)
    top_idx = sorted_idx[:100]
    bottom_idx = sorted_idx[-100:]

    del_top, imp_top = _run_group("top 100 (most likely sepsis)", top_idx, probs, test_ds, model, args.device)
    del_bot, imp_bot = _run_group("bottom 100 (least likely sepsis)", bottom_idx, probs, test_ds, model, args.device)

    print("\n=== Odds-change results ===")
    print(f"Del odds change  |  top 100:    {sum(del_top) / len(del_top):+.4f}")
    print(f"Del odds change  |  bottom 100: {sum(del_bot) / len(del_bot):+.4f}")
    print(f"Imp odds change  |  top 100:    {sum(imp_top) / len(imp_top):+.4f}")
    print(f"Imp odds change  |  bottom 100: {sum(imp_bot) / len(imp_bot):+.4f}")


if __name__ == "__main__":
    main()
