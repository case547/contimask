from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import config


def _timestep_count(path: Path) -> int:
    df = pd.read_csv(path, sep="|", usecols=["ICULOS"])
    return int((df["ICULOS"] <= config.MAX_SEQ_LEN).sum())


class SepsisDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        psv_files: Optional[list] = None,
        norm_stats: Optional[dict] = None,
        min_timesteps: int = 3,
    ):
        self.data_dir = Path(data_dir)
        if psv_files is None:
            psv_files = sorted(self.data_dir.glob("*.psv"))
            if min_timesteps > 1:
                psv_files = [f for f in psv_files if _timestep_count(f) >= min_timesteps]
        self.psv_files = list(psv_files)
        self.norm_stats = norm_stats

    def __len__(self) -> int:
        return len(self.psv_files)

    def __getitem__(self, idx: int):
        df = pd.read_csv(self.psv_files[idx], sep="|")

        label = int(df["SepsisLabel"].any())

        # Truncate to first MAX_SEQ_LEN hours (filter by ICULOS value, not row count)
        rows = df[df["ICULOS"] <= config.MAX_SEQ_LEN]
        n = len(rows)

        pad = config.MAX_SEQ_LEN - n

        # Time axis: ICULOS normalised to [0, 1]
        t_real = (
            torch.tensor(rows["ICULOS"].values, dtype=torch.float32)
            / config.MAX_SEQ_LEN
        )
        t = F.pad(t_real, (0, pad))

        # Features and mask - pad tail rows with zeros
        feature_tensor = torch.tensor(
            rows[config.FEATURE_COLS].values, dtype=torch.float32
        )
        observed = ~torch.isnan(feature_tensor)
        X = F.pad(torch.nan_to_num(feature_tensor, nan=0.0), (0, 0, 0, pad))
        data_mask = F.pad(observed.float(), (0, 0, 0, pad))

        if self.norm_stats is not None:
            mean = self.norm_stats["mean"]
            std = self.norm_stats["std"]
            X = (X - mean) * data_mask
            X = X / std
            X = X * data_mask  # re-zero positions masked out

        return t, X, data_mask, torch.tensor(label, dtype=torch.float32)


def compute_norm_stats(dataset: SepsisDataset) -> dict:
    all_X = []
    all_mask = []
    for idx in range(len(dataset)):
        _, X, data_mask, _ = dataset[idx]
        all_X.append(X)
        all_mask.append(data_mask)
    all_X = torch.stack(all_X)  # (N, T, F)
    all_mask = torch.stack(all_mask)  # (N, T, F)
    denom = all_mask.sum(dim=(0, 1)).clamp(min=1.0)
    mean = (all_X * all_mask).sum(dim=(0, 1)) / denom
    sq_diff = ((all_X - mean) ** 2) * all_mask
    std = torch.sqrt(sq_diff.sum(dim=(0, 1)) / denom).clamp(min=1e-6)
    return {"mean": mean, "std": std}
