from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import config


def _patient_meta(path: Path) -> tuple[int, int]:
    """Return (n_timesteps_within_window, binary_label) for a PSV file."""
    df = pd.read_csv(path, sep="|", usecols=["ICULOS", "SepsisLabel"])
    n = int((df["ICULOS"] <= config.MAX_SEQ_LEN).sum())
    label = int(df["SepsisLabel"].any())
    return n, label


class SepsisDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        psv_files: list[Path] | None = None,
        norm_stats: dict[str, torch.Tensor] | None = None,
        min_timesteps: int = 3,
    ):
        self.data_dir = Path(data_dir)
        if psv_files is None:
            psv_files = sorted(self.data_dir.glob("*.psv"))
            meta = [_patient_meta(f) for f in psv_files]
            if min_timesteps > 1:
                psv_files = [
                    f for f, (n, _) in zip(psv_files, meta) if n >= min_timesteps
                ]
                meta = [(n, label) for n, label in meta if n >= min_timesteps]
            self.labels: list[int] = [label for _, label in meta]
        else:
            self.labels = [
                int(
                    pd.read_csv(f, sep="|", usecols=["SepsisLabel"])[
                        "SepsisLabel"
                    ].any()
                )
                for f in psv_files
            ]
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


def compute_norm_stats(dataset: SepsisDataset) -> dict[str, torch.Tensor]:
    count = torch.zeros(config.N_FEATURES)
    total = torch.zeros(config.N_FEATURES)
    for idx in range(len(dataset)):
        _, X, data_mask, _ = dataset[idx]
        count += data_mask.sum(dim=0)           # (F,)
        total += (X * data_mask).sum(dim=0)     # (F,)
    mean = total / count.clamp(min=1.0)

    sq_diff = torch.zeros(config.N_FEATURES)
    for idx in range(len(dataset)):
        _, X, data_mask, _ = dataset[idx]
        sq_diff += (((X - mean) ** 2) * data_mask).sum(dim=0)
    std = torch.sqrt(sq_diff / count.clamp(min=1.0)).clamp(min=1e-6)

    return {"mean": mean, "std": std}
