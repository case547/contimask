# Sepsis DiffusionTransformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a DiffusionTransformer on PhysioNet 2019 sepsis data and wrap it as a ContiMask `forward_func` for attribution.

**Architecture:** Transformer encoder with sinusoidal time embeddings and a learned CLS token, pretrained with a DDPM denoising objective then fine-tuned with BCE. All 39 features (34 time-varying + 5 static) are treated uniformly as a constant time series — static features repeat at every real timestep and are attributed over by ContiMask like any other feature.

**Tech Stack:** Python 3.10, PyTorch 2.x, scikit-learn (stratified split + AUC), pandas (PSV loading), pytest, existing `utils/tools.py` (EarlyStopping), existing `attribution/` (ContiMask framework).

**Spec:** `docs/superpowers/specs/2026-05-18-sepsis-diffusiontransformer-option1-design.md`

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `config.py` | **Create** | All hyperparameters as module-level constants |
| `data/__init__.py` | **Create** | Empty package marker |
| `data/dataset.py` | **Create** | `SepsisDataset`, `compute_norm_stats` |
| `data/split.py` | **Create** | `stratified_patient_split` |
| `models/__init__.py` | **Create** | Empty package marker |
| `models/embeddings.py` | **Create** | `SinusoidalTimeEmbedding`, `DiffusionStepEmbedding` |
| `models/diffusion_transformer.py` | **Create** | `DiffusionTransformer` (`encode`, `denoise`, `classify`) |
| `training/__init__.py` | **Create** | Empty package marker |
| `training/schedule.py` | **Create** | `DDPMSchedule` |
| `training/pretrain.py` | **Create** | `pretrain()` loop (Phase 1) |
| `training/finetune.py` | **Create** | `finetune()` loop (Phase 2) |
| `training/train.py` | **Create** | CLI entry point (`--phase pretrain\|finetune\|both`) |
| `sepsis_attribution.py` | **Create** | `make_forward_func`, `run_attribution` |
| `tests/conftest.py` | **Create** | Shared fixtures (mock PSV directory) |
| `tests/data/test_dataset.py` | **Create** | Dataset unit tests |
| `tests/data/test_split.py` | **Create** | Split unit tests |
| `tests/models/test_embeddings.py` | **Create** | Embedding shape tests |
| `tests/models/test_diffusion_transformer.py` | **Create** | Model forward pass tests |
| `tests/training/test_schedule.py` | **Create** | DDPM schedule tests |
| `tests/training/test_loops.py` | **Create** | Pretrain/finetune loop tests |
| `tests/test_attribution.py` | **Create** | `forward_func` K>1 path test + odds-change tests |
| `pyproject.toml` | **Modify** | Add `scikit-learn`, `pandas`, `pytest` deps |
| `checkpoints/` | **Create dir** | Saved model weights |

---

## Task 1: Project setup

**Files:**
- Create: `config.py`
- Create: `checkpoints/.gitkeep`
- Create: `tests/__init__.py`, `tests/data/__init__.py`, `tests/models/__init__.py`, `tests/training/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add missing dependencies to `pyproject.toml`**

```toml
dependencies = [
    "evotorch==0.5.1",
    "isort>=8.0.1",
    "matplotlib==3.10.1",
    "numpy==1.26.4",
    "pandas>=2.0",
    "pip==25.0",
    "pytest>=8.0",
    "ruff>=0.15.12",
    "scikit-learn>=1.4",
    "torch>=2.11.0",
    "torchvision>=0.26.0",
    "tqdm>=4.67.3",
]
```

- [ ] **Step 2: Install new deps**

```bash
cd /home/justin/msc_ai/individual-project/contimask
uv sync
```

Expected: resolves without error.

- [ ] **Step 3: Create `config.py`**

```python
# Architecture
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 4
FFN_DIM = 256
DROPOUT = 0.1
TIME_EMBED_L = 64
T_DIFF = 1000
MAX_SEQ_LEN = 72
N_FEATURES = 39

# Training
BATCH_SIZE = 64
PRETRAIN_LR = 1e-4
FINETUNE_LR = 1e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
PRETRAIN_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 10
FINETUNE_LR_RATIOS = [0.1, 0.3, 0.7, 1.0]  # per transformer layer; 0.0 = frozen

# ContiMask attribution
MASK_HIDDEN_DIM = 16
MASK_L = 12
MASK_TARGET_AREA = 0.1
MASK_EPOCHS = 200

FEATURE_COLS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime",
]
```

- [ ] **Step 4: Create directory structure**

```bash
mkdir -p checkpoints tests/data tests/models tests/training
touch data/__init__.py models/__init__.py training/__init__.py
touch tests/__init__.py tests/data/__init__.py tests/models/__init__.py tests/training/__init__.py
touch checkpoints/.gitkeep
```

- [ ] **Step 5: Verify config imports**

```bash
python -c "import config; print(config.N_FEATURES, len(config.FEATURE_COLS))"
```

Expected: `39 39`

---

## Task 2: SepsisDataset

**Files:**
- Create: `data/dataset.py`
- Create: `tests/conftest.py`
- Create: `tests/data/test_dataset.py`

- [ ] **Step 1: Write failing tests**

`tests/conftest.py`:
```python
import tempfile
from pathlib import Path

import pytest


def _write_psv(path: Path, n_rows: int, has_sepsis: bool = False) -> None:
    header = (
        "HR|O2Sat|Temp|SBP|MAP|DBP|Resp|EtCO2|BaseExcess|HCO3|FiO2|pH|PaCO2|SaO2|"
        "AST|BUN|Alkalinephos|Calcium|Chloride|Creatinine|Bilirubin_direct|Glucose|"
        "Lactate|Magnesium|Phosphate|Potassium|Bilirubin_total|TroponinI|Hct|Hgb|"
        "PTT|WBC|Fibrinogen|Platelets|Age|Gender|Unit1|Unit2|HospAdmTime|ICULOS|SepsisLabel"
    )
    rows = []
    for i in range(n_rows):
        tv = ["NaN"] * 34
        static = ["65.0", "0", "NaN", "NaN", "-1.0"]
        iculos = str(i + 1)
        label = "1" if (has_sepsis and i == n_rows - 1) else "0"
        rows.append("|".join(tv + static + [iculos, label]))
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


@pytest.fixture()
def mock_data_dir(tmp_path):
    _write_psv(tmp_path / "p000001.psv", n_rows=50, has_sepsis=False)
    _write_psv(tmp_path / "p000002.psv", n_rows=80, has_sepsis=True)   # >72 rows, sepsis
    _write_psv(tmp_path / "p000003.psv", n_rows=10, has_sepsis=False)  # short stay
    return tmp_path
```

`tests/data/test_dataset.py`:
```python
import torch
from data.dataset import SepsisDataset


def test_dataset_length(mock_data_dir):
    ds = SepsisDataset(mock_data_dir)
    assert len(ds) == 3


def test_item_shapes(mock_data_dir):
    ds = SepsisDataset(mock_data_dir)
    t, X, data_mask, label = ds[0]
    assert t.shape == (72,)
    assert X.shape == (72, 39)
    assert data_mask.shape == (72, 39)


def test_time_values(mock_data_dir):
    ds = SepsisDataset(mock_data_dir)
    t, X, data_mask, label = ds[0]  # 50-row patient
    assert abs(t[0].item() - 1 / 72) < 1e-5
    assert abs(t[49].item() - 50 / 72) < 1e-5


def test_padding(mock_data_dir):
    ds = SepsisDataset(mock_data_dir)
    t, X, data_mask, label = ds[0]  # 50-row patient, padded to 72
    assert data_mask[50:, :].sum().item() == 0
    assert X[50:, :].sum().item() == 0.0


def test_truncation(mock_data_dir):
    ds = SepsisDataset(mock_data_dir)
    t, X, data_mask, label = ds[1]  # 80-row patient, truncated to 72
    assert data_mask.sum().item() > 0
    assert t[71].item() <= 1.0


def test_label_from_full_file(mock_data_dir):
    ds = SepsisDataset(mock_data_dir)
    _, _, _, label0 = ds[0]
    _, _, _, label1 = ds[1]
    assert label0.item() == 0
    assert label1.item() == 1


def test_static_features_observed_at_real_rows(mock_data_dir):
    ds = SepsisDataset(mock_data_dir)
    t, X, data_mask, label = ds[0]  # 50 real rows
    # Age (col 34) and Gender (col 35) are non-NaN in mock data
    assert data_mask[:50, 34].all()
    assert data_mask[:50, 35].all()
    # Unit1 (col 36) and Unit2 (col 37) are NaN in mock data
    assert data_mask[:50, 36].sum().item() == 0


def test_normalization_applied(mock_data_dir):
    ds_raw = SepsisDataset(mock_data_dir)
    from data.dataset import compute_norm_stats
    norm_stats = compute_norm_stats(ds_raw)
    ds_norm = SepsisDataset(mock_data_dir, norm_stats=norm_stats)
    _, X_norm, _, _ = ds_norm[0]
    _, X_raw, _, _ = ds_raw[0]
    # Normalized and raw should differ
    assert not torch.allclose(X_norm, X_raw)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/justin/msc_ai/individual-project/contimask
python -m pytest tests/data/test_dataset.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'data'`

- [x] **Step 3: Implement `data/dataset.py`**

> **Deviations from original plan:**
> - Windowing uses `df[df["ICULOS"] <= config.MAX_SEQ_LEN]` (value filter, as planned), but without the `min(len(...), MAX_SEQ_LEN)` cap — ICULOS filter alone is sufficient.
> - Tensor construction uses `F.pad` rather than pre-allocation + slice assignment.
> - Static feature broadcasting (copying row-0 value/mask to all real rows) was omitted. The PSV format already stores static features consistently across all rows, so no enforcement is needed.

```python
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import config


class SepsisDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        psv_files: Optional[list] = None,
        norm_stats: Optional[dict] = None,
    ):
        self.data_dir = Path(data_dir)
        if psv_files is None:
            psv_files = sorted(self.data_dir.glob("*.psv"))
        self.psv_files = list(psv_files)
        self.norm_stats = norm_stats

    def __len__(self) -> int:
        return len(self.psv_files)

    def __getitem__(self, idx: int):
        df = pd.read_csv(self.psv_files[idx], sep="|")

        label = int(df["SepsisLabel"].any())

        rows = df[df["ICULOS"] <= config.MAX_SEQ_LEN]
        n = len(rows)
        pad = config.MAX_SEQ_LEN - n

        t_real = torch.tensor(rows["ICULOS"].values, dtype=torch.float32) / config.MAX_SEQ_LEN
        t = F.pad(t_real, (0, pad))

        feat_tensor = torch.tensor(rows[config.FEATURE_COLS].values, dtype=torch.float32)
        observed = ~torch.isnan(feat_tensor)
        X = F.pad(torch.nan_to_num(feat_tensor, nan=0.0), (0, 0, 0, pad))
        data_mask = F.pad(observed.float(), (0, 0, 0, pad))

        if self.norm_stats is not None:
            mean = self.norm_stats["mean"]
            std = self.norm_stats["std"]
            X = (X - mean) * data_mask
            X = X / std
            X = X * data_mask

        return t, X, data_mask, torch.tensor(label, dtype=torch.float32)


def compute_norm_stats(dataset: SepsisDataset) -> dict:
    all_X = []
    all_mask = []
    for idx in range(len(dataset)):
        _, X, data_mask, _ = dataset[idx]
        all_X.append(X)
        all_mask.append(data_mask)
    all_X = torch.stack(all_X)       # (N, T, F)
    all_mask = torch.stack(all_mask) # (N, T, F)
    denom = all_mask.sum(dim=(0, 1)).clamp(min=1.0)
    mean = (all_X * all_mask).sum(dim=(0, 1)) / denom
    sq_diff = ((all_X - mean) ** 2) * all_mask
    std = torch.sqrt(sq_diff.sum(dim=(0, 1)) / denom).clamp(min=1e-6)
    return {"mean": mean, "std": std}
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/data/test_dataset.py -v
```

Expected: all 8 tests pass.

---

## Task 3: Stratified split

**Files:**
- Create: `data/split.py`
- Create: `tests/data/test_split.py`

- [ ] **Step 1: Write failing tests**

`tests/data/test_split.py`:
```python
from pathlib import Path

import torch

from data.dataset import SepsisDataset
from data.split import stratified_patient_split


def test_split_sizes(mock_data_dir):
    ds = SepsisDataset(mock_data_dir)
    labels = [int(ds[i][3].item()) for i in range(len(ds))]
    train_files, val_files, test_files = stratified_patient_split(
        ds.psv_files, labels, val_frac=0.15, test_frac=0.15, seed=42
    )
    total = len(train_files) + len(val_files) + len(test_files)
    assert total == len(ds.psv_files)


def test_split_no_overlap(mock_data_dir):
    ds = SepsisDataset(mock_data_dir)
    labels = [int(ds[i][3].item()) for i in range(len(ds))]
    train_files, val_files, test_files = stratified_patient_split(
        ds.psv_files, labels, val_frac=0.15, test_frac=0.15, seed=42
    )
    all_files = set(train_files) | set(val_files) | set(test_files)
    assert len(all_files) == len(ds.psv_files)


def test_split_returns_path_lists(mock_data_dir):
    ds = SepsisDataset(mock_data_dir)
    labels = [int(ds[i][3].item()) for i in range(len(ds))]
    train_files, val_files, test_files = stratified_patient_split(
        ds.psv_files, labels, seed=42
    )
    assert all(isinstance(p, Path) for p in train_files)
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/data/test_split.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'data.split'`

- [ ] **Step 3: Implement `data/split.py`**

```python
from __future__ import annotations

from pathlib import Path

from sklearn.model_selection import train_test_split


def stratified_patient_split(
    psv_files: list[Path],
    labels: list[int],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    test_size = test_frac / (1.0 - val_frac)
    train_val_files, test_files, train_val_labels, _ = train_test_split(
        psv_files, labels, test_size=test_frac, stratify=labels, random_state=seed
    )
    # If too few samples for stratification, fall back to unstratified
    try:
        train_files, val_files = train_test_split(
            train_val_files,
            train_val_labels,
            test_size=val_frac / (1.0 - test_frac),
            stratify=train_val_labels,
            random_state=seed,
        )
    except ValueError:
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=val_frac / (1.0 - test_frac),
            random_state=seed,
        )
    return list(train_files), list(val_files), list(test_files)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/data/test_split.py -v
```

Expected: all 3 tests pass.

---

## Task 4: Embeddings

**Files:**
- Create: `models/embeddings.py`
- Create: `tests/models/test_embeddings.py`

- [ ] **Step 1: Write failing tests**

`tests/models/test_embeddings.py`:
```python
import math

import torch

from models.embeddings import DiffusionStepEmbedding, SinusoidalTimeEmbedding


def test_sinusoidal_output_shape():
    emb = SinusoidalTimeEmbedding(L=64, d_model=64)
    t = torch.rand(4, 72)
    out = emb(t)
    assert out.shape == (4, 72, 64)


def test_sinusoidal_no_nan():
    emb = SinusoidalTimeEmbedding(L=64, d_model=64)
    t = torch.rand(4, 72)
    out = emb(t)
    assert not torch.isnan(out).any()


def test_sinusoidal_different_times_differ():
    emb = SinusoidalTimeEmbedding(L=64, d_model=64)
    t1 = torch.zeros(1, 72)
    t2 = torch.ones(1, 72)
    assert not torch.allclose(emb(t1), emb(t2))


def test_step_embed_output_shape():
    emb = DiffusionStepEmbedding(T_diff=1000, L=64, d_model=64)
    s = torch.randint(0, 1000, (4,))
    out = emb(s)
    assert out.shape == (4, 1, 64)


def test_step_embed_no_nan():
    emb = DiffusionStepEmbedding(T_diff=1000, L=64, d_model=64)
    s = torch.randint(0, 1000, (4,))
    assert not torch.isnan(emb(s)).any()
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/models/test_embeddings.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'models.embeddings'`

- [ ] **Step 3: Implement `models/embeddings.py`**

```python
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, L: int, d_model: int):
        super().__init__()
        self.L = L
        self.proj = nn.Linear(2 * L, d_model)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, T) values in [0, 1]
        device = t.device
        freqs = 2 * math.pi * (2.0 ** torch.arange(self.L, device=device))  # (L,)
        t_feats = t.unsqueeze(-1) * freqs  # (B, T, L)
        feats = torch.cat([t_feats.sin(), t_feats.cos()], dim=-1)  # (B, T, 2L)
        return self.proj(feats)  # (B, T, d_model)


class DiffusionStepEmbedding(nn.Module):
    def __init__(self, T_diff: int, L: int, d_model: int):
        super().__init__()
        self.T_diff = T_diff
        self.L = L
        self.proj = nn.Linear(2 * L, d_model)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: (B,) integer steps in [0, T_diff)
        device = s.device
        s_norm = s.float() / self.T_diff  # (B,)
        freqs = 2 * math.pi * (2.0 ** torch.arange(self.L, device=device))  # (L,)
        s_feats = s_norm.unsqueeze(-1) * freqs  # (B, L)
        feats = torch.cat([s_feats.sin(), s_feats.cos()], dim=-1)  # (B, 2L)
        return self.proj(feats).unsqueeze(1)  # (B, 1, d_model)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/models/test_embeddings.py -v
```

Expected: all 5 tests pass.

---

## Task 5: DiffusionTransformer

**Files:**
- Create: `models/diffusion_transformer.py`
- Create: `tests/models/test_diffusion_transformer.py`

- [ ] **Step 1: Write failing tests**

`tests/models/test_diffusion_transformer.py`:
```python
import torch
import pytest

import config
from models.diffusion_transformer import DiffusionTransformer


@pytest.fixture()
def model():
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


def _batch(B=2, T=72, F=39, actual_len=50):
    t = torch.zeros(B, T)
    t[:, :actual_len] = torch.linspace(1 / T, actual_len / T, actual_len)
    X = torch.randn(B, T, F)
    X[:, actual_len:] = 0.0
    data_mask = torch.zeros(B, T, F)
    data_mask[:, :actual_len] = 1.0
    return t, X, data_mask


def test_classify_shape(model):
    t, X, dm = _batch()
    out = model.classify(t, X, dm)
    assert out.shape == (2, 1)


def test_classify_no_nan(model):
    t, X, dm = _batch()
    assert not torch.isnan(model.classify(t, X, dm)).any()


def test_denoise_shape(model):
    t, X, dm = _batch()
    s = torch.randint(0, config.T_DIFF, (2,))
    out = model.denoise(t, X, dm, s)
    assert out.shape == (2, 72, 39)


def test_denoise_no_nan(model):
    t, X, dm = _batch()
    s = torch.randint(0, config.T_DIFF, (2,))
    assert not torch.isnan(model.denoise(t, X, dm, s)).any()


def test_classify_fully_padded_batch(model):
    t = torch.zeros(2, 72)
    X = torch.zeros(2, 72, 39)
    dm = torch.zeros(2, 72, 39)
    # Even with all-padded input, should not crash (may produce NaN, but no error)
    out = model.classify(t, X, dm)
    assert out.shape == (2, 1)


def test_padding_mask_excludes_tail(model):
    # Two items: one with 10 real rows, one with 72 real rows
    t, X, dm = _batch(B=2, actual_len=10)
    t2, X2, dm2 = _batch(B=2, actual_len=72)
    out1 = model.classify(t, X, dm)
    out2 = model.classify(t2, X2, dm2)
    # Just verify both run without error and shapes are correct
    assert out1.shape == out2.shape == (2, 1)
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/models/test_diffusion_transformer.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'models.diffusion_transformer'`

- [ ] **Step 3: Implement `models/diffusion_transformer.py`**

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from models.embeddings import DiffusionStepEmbedding, SinusoidalTimeEmbedding


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        dropout: float,
        n_features: int,
        time_embed_L: int,
        T_diff: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features

        self.input_proj = nn.Linear(n_features, d_model)
        self.time_embed = SinusoidalTimeEmbedding(L=time_embed_L, d_model=d_model)
        self.step_embed = DiffusionStepEmbedding(T_diff=T_diff, L=time_embed_L, d_model=d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.denoise_head = nn.Linear(d_model, n_features)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def encode(
        self,
        t: torch.Tensor,
        X: torch.Tensor,
        data_mask: torch.Tensor,
        s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # t: (B, T), X: (B, T, F), data_mask: (B, T, F)
        B, T, _ = X.shape

        tokens = self.input_proj(X) + self.time_embed(t)  # (B, T, d_model)
        if s is not None:
            tokens = tokens + self.step_embed(s)  # broadcasts (B, 1, d_model) over T

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, T+1, d_model)

        # src_key_padding_mask: True = ignore this position
        pad = ~data_mask.any(dim=-1)  # (B, T) True for fully padded
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=X.device)
        src_key_padding_mask = torch.cat([cls_pad, pad], dim=1)  # (B, T+1)

        out = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)
        return out  # (B, T+1, d_model)

    def denoise(
        self,
        t: torch.Tensor,
        X: torch.Tensor,
        data_mask: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encode(t, X, data_mask, s=s)  # (B, T+1, d_model)
        return self.denoise_head(out[:, 1:, :])  # (B, T, n_features)

    def classify(
        self,
        t: torch.Tensor,
        X: torch.Tensor,
        data_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encode(t, X, data_mask, s=None)  # (B, T+1, d_model)
        return self.cls_head(out[:, 0, :])  # (B, 1)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/models/test_diffusion_transformer.py -v
```

Expected: all 6 tests pass.

---

## Task 6: DDPM noise schedule

**Files:**
- Create: `training/schedule.py`
- Create: `tests/training/test_schedule.py`

- [ ] **Step 1: Write failing tests**

`tests/training/test_schedule.py`:
```python
import torch

from training.schedule import DDPMSchedule


def test_alpha_bars_shape():
    schedule = DDPMSchedule(T=1000)
    assert schedule.alpha_bars.shape == (1000,)


def test_alpha_bars_decreasing():
    schedule = DDPMSchedule(T=1000)
    assert (schedule.alpha_bars[1:] <= schedule.alpha_bars[:-1]).all()


def test_q_sample_shapes():
    schedule = DDPMSchedule(T=1000)
    x0 = torch.randn(4, 72, 39)
    dm = torch.ones(4, 72, 39)
    s = torch.randint(0, 1000, (4,))
    x_noisy, eps = schedule.q_sample(x0, s, dm)
    assert x_noisy.shape == x0.shape
    assert eps.shape == x0.shape


def test_unobserved_positions_stay_zero():
    schedule = DDPMSchedule(T=1000)
    x0 = torch.randn(2, 72, 39)
    dm = torch.zeros(2, 72, 39)
    dm[:, :10, :] = 1.0  # only first 10 real
    x0 = x0 * dm
    s = torch.tensor([100, 500])
    x_noisy, _ = schedule.q_sample(x0, s, dm)
    assert x_noisy[:, 10:, :].abs().sum().item() == 0.0


def test_high_noise_step_corrupts():
    schedule = DDPMSchedule(T=1000)
    torch.manual_seed(0)
    x0 = torch.ones(1, 10, 5)
    dm = torch.ones(1, 10, 5)
    s = torch.tensor([999])  # max noise
    x_noisy, _ = schedule.q_sample(x0, s, dm)
    assert not torch.allclose(x_noisy, x0)
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/training/test_schedule.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'training.schedule'`

- [ ] **Step 3: Implement `training/schedule.py`**

```python
from __future__ import annotations

import torch


class DDPMSchedule:
    def __init__(self, T: int = 1000, beta_min: float = 1e-4, beta_max: float = 0.02):
        betas = torch.linspace(beta_min, beta_max, T)
        alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(alphas, dim=0)  # (T,)

    def q_sample(
        self,
        x0: torch.Tensor,
        s: torch.Tensor,
        data_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x0: (B, T_seq, F) clean normalised features, NaN already filled with 0
        s:  (B,) diffusion step indices in [0, T_diff)
        data_mask: (B, T_seq, F) binary, 1 where observed
        Returns: (x_noisy, eps) — eps is the sampled noise
        """
        B = x0.shape[0]
        device = x0.device
        alpha_bar = self.alpha_bars.to(device)[s].view(B, 1, 1)  # (B, 1, 1)
        eps = torch.randn_like(x0)
        x_noisy = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * eps
        x_noisy = x_noisy * data_mask  # zero out unobserved positions
        return x_noisy, eps
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/training/test_schedule.py -v
```

Expected: all 5 tests pass.

---

## Task 7: Pretrain loop

**Files:**
- Create: `training/pretrain.py`
- Extend: `tests/training/test_loops.py`

- [ ] **Step 1: Write failing tests**

`tests/training/test_loops.py`:
```python
import torch
from torch.utils.data import DataLoader, TensorDataset

import config
from models.diffusion_transformer import DiffusionTransformer
from training.pretrain import pretrain
from training.schedule import DDPMSchedule


def _synthetic_loader(n=16, T=72, F=39, batch_size=8):
    t = torch.rand(n, T)
    X = torch.randn(n, T, F)
    dm = torch.ones(n, T, F)
    labels = torch.zeros(n)
    ds = TensorDataset(t, X, dm, labels)
    return DataLoader(ds, batch_size=batch_size)


def test_pretrain_returns_loss(tmp_path):
    model = DiffusionTransformer(
        d_model=16, n_heads=2, n_layers=2, ffn_dim=32,
        dropout=0.0, n_features=config.N_FEATURES,
        time_embed_L=4, T_diff=100,
    )
    loader = _synthetic_loader(n=16)
    schedule = DDPMSchedule(T=100)
    val_loss = pretrain(
        model, loader, loader, schedule,
        max_epochs=2, lr=1e-3, patience=5,
        checkpoint_path=str(tmp_path / "pretrained.pt"),
        device="cpu",
    )
    assert isinstance(val_loss, float)
    assert val_loss > 0


def test_pretrain_saves_checkpoint(tmp_path):
    model = DiffusionTransformer(
        d_model=16, n_heads=2, n_layers=2, ffn_dim=32,
        dropout=0.0, n_features=config.N_FEATURES,
        time_embed_L=4, T_diff=100,
    )
    loader = _synthetic_loader(n=16)
    schedule = DDPMSchedule(T=100)
    pretrain(
        model, loader, loader, schedule,
        max_epochs=2, lr=1e-3, patience=5,
        checkpoint_path=str(tmp_path / "pretrained.pt"),
        device="cpu",
    )
    assert (tmp_path / "pretrained.pt").exists()
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/training/test_loops.py::test_pretrain_returns_loss -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'training.pretrain'`

- [ ] **Step 3: Implement `training/pretrain.py`**

```python
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.schedule import DDPMSchedule
from utils.tools import EarlyStopping


def _pretrain_epoch(
    model: nn.Module,
    loader: DataLoader,
    schedule: DDPMSchedule,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: str,
) -> float:
    model.train()
    total = 0.0
    for t, X, data_mask, _ in loader:
        t = t.to(device)
        X = X.to(device)
        data_mask = data_mask.to(device)
        B = X.shape[0]
        s = torch.randint(0, schedule.alpha_bars.shape[0], (B,), device=device)
        X_noisy, eps = schedule.q_sample(X, s, data_mask)
        eps_pred = model.denoise(t, X_noisy, data_mask, s)
        loss = ((eps_pred - eps) ** 2 * data_mask).sum() / data_mask.sum().clamp(min=1)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def _val_epoch(
    model: nn.Module,
    loader: DataLoader,
    schedule: DDPMSchedule,
    device: str,
) -> float:
    model.eval()
    total = 0.0
    for t, X, data_mask, _ in loader:
        t = t.to(device)
        X = X.to(device)
        data_mask = data_mask.to(device)
        B = X.shape[0]
        s = torch.randint(0, schedule.alpha_bars.shape[0], (B,), device=device)
        X_noisy, eps = schedule.q_sample(X, s, data_mask)
        eps_pred = model.denoise(t, X_noisy, data_mask, s)
        loss = ((eps_pred - eps) ** 2 * data_mask).sum() / data_mask.sum().clamp(min=1)
        total += loss.item()
    return total / len(loader)


def pretrain(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    schedule: DDPMSchedule,
    max_epochs: int,
    lr: float,
    patience: int,
    checkpoint_path: str,
    grad_clip: float = 1.0,
    device: str = "cpu",
) -> float:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    es = EarlyStopping(patience=patience, path=checkpoint_path, verbose=True)

    for epoch in range(max_epochs):
        train_loss = _pretrain_epoch(model, train_loader, schedule, optimizer, grad_clip, device)
        val_loss = _val_epoch(model, val_loader, schedule, device)
        print(f"Pretrain epoch {epoch+1}/{max_epochs}  train={train_loss:.4f}  val={val_loss:.4f}")
        es(val_loss)
        if es.counter == 0:
            es.save_checkpoint(val_loss, model)
        if es.early_stop:
            print("Early stopping.")
            break

    return es.val_loss_min
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/training/test_loops.py::test_pretrain_returns_loss tests/training/test_loops.py::test_pretrain_saves_checkpoint -v
```

Expected: both pass.

---

## Task 8: Finetune loop

**Files:**
- Extend: `training/finetune.py`
- Extend: `tests/training/test_loops.py`

- [ ] **Step 1: Add finetune tests to `tests/training/test_loops.py`**

Append to `tests/training/test_loops.py`:
```python
from training.finetune import finetune


def test_finetune_returns_auc(tmp_path):
    model = DiffusionTransformer(
        d_model=16, n_heads=2, n_layers=2, ffn_dim=32,
        dropout=0.0, n_features=config.N_FEATURES,
        time_embed_L=4, T_diff=100,
    )
    # Imbalanced: 2 positives, 14 negatives
    labels = torch.cat([torch.ones(2), torch.zeros(14)])
    t = torch.rand(16, 72)
    X = torch.randn(16, 72, config.N_FEATURES)
    dm = torch.ones(16, 72, config.N_FEATURES)
    ds = TensorDataset(t, X, dm, labels)
    loader = DataLoader(ds, batch_size=8)
    auc = finetune(
        model, loader, loader,
        lr_ratios=[1.0, 1.0, 1.0, 1.0],
        base_lr=1e-3,
        weight_decay=0.01,
        max_epochs=2,
        patience=5,
        checkpoint_path=str(tmp_path / "best.pt"),
        device="cpu",
    )
    assert 0.0 <= auc <= 1.0


def test_finetune_saves_checkpoint(tmp_path):
    model = DiffusionTransformer(
        d_model=16, n_heads=2, n_layers=2, ffn_dim=32,
        dropout=0.0, n_features=config.N_FEATURES,
        time_embed_L=4, T_diff=100,
    )
    labels = torch.cat([torch.ones(2), torch.zeros(14)])
    t = torch.rand(16, 72)
    X = torch.randn(16, 72, config.N_FEATURES)
    dm = torch.ones(16, 72, config.N_FEATURES)
    ds = TensorDataset(t, X, dm, labels)
    loader = DataLoader(ds, batch_size=8)
    finetune(
        model, loader, loader,
        lr_ratios=[1.0, 1.0, 1.0, 1.0],
        base_lr=1e-3,
        weight_decay=0.01,
        max_epochs=2,
        patience=5,
        checkpoint_path=str(tmp_path / "best.pt"),
        device="cpu",
    )
    assert (tmp_path / "best.pt").exists()
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/training/test_loops.py::test_finetune_returns_auc -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'training.finetune'`

- [ ] **Step 3: Implement `training/finetune.py`**

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from utils.tools import EarlyStopping


def _build_optimizer(
    model: nn.Module,
    lr_ratios: list[float],
    base_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    param_groups = []
    embedding_params = (
        list(model.input_proj.parameters())
        + list(model.time_embed.parameters())
        + list(model.step_embed.parameters())
        + [model.cls_token]
    )
    param_groups.append({"params": embedding_params, "lr": base_lr * lr_ratios[0]})
    for i, layer in enumerate(model.transformer.layers):
        ratio = lr_ratios[i]
        if ratio == 0.0:
            for p in layer.parameters():
                p.requires_grad_(False)
        else:
            param_groups.append({"params": list(layer.parameters()), "lr": base_lr * ratio})
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

    optimizer = _build_optimizer(model, lr_ratios, base_lr, weight_decay)
    es = EarlyStopping(patience=patience, path=checkpoint_path, verbose=True)

    best_auc = 0.0
    for epoch in range(max_epochs):
        _finetune_epoch(model, train_loader, optimizer, pos_weight, grad_clip, device)
        val_auc = _evaluate(model, val_loader, device)
        print(f"Finetune epoch {epoch+1}/{max_epochs}  val_auc={val_auc:.4f}")
        es(-val_auc)  # EarlyStopping minimises; negate so higher AUC = improvement
        if es.counter == 0:
            es.save_checkpoint(-val_auc, model)
            best_auc = val_auc
        if es.early_stop:
            print("Early stopping.")
            break

    return best_auc
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/training/test_loops.py -v
```

Expected: all 4 tests pass.

---

## Task 9: Train CLI

**Files:**
- Create: `training/train.py`

- [ ] **Step 1: Implement `training/train.py`**

```python
from __future__ import annotations

import argparse
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
    labels = [int(raw_ds[i][3].item()) for i in range(len(raw_ds))]
    train_files, val_files, test_files = stratified_patient_split(
        raw_ds.psv_files, labels
    )

    # Compute norm stats on training split only
    train_ds_raw = SepsisDataset(data_dir, psv_files=train_files)
    norm_stats = compute_norm_stats(train_ds_raw)
    torch.save(norm_stats, checkpoint_dir / "norm_stats.pt")

    train_ds = SepsisDataset(data_dir, psv_files=train_files, norm_stats=norm_stats)
    val_ds = SepsisDataset(data_dir, psv_files=val_files, norm_stats=norm_stats)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pretrain", "finetune", "both"], default="both")
    parser.add_argument("--data_dir", default="~/msc_ai/individual-project/sepsis_data/training_setA")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--pretrain_epochs", type=int, default=config.PRETRAIN_EPOCHS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    train_loader, val_loader = build_loaders(args.data_dir, checkpoint_dir)
    model = build_model()

    pretrain_ckpt = checkpoint_dir / "pretrained_backbone.pt"
    finetune_ckpt = checkpoint_dir / "best_model.pt"

    if args.phase in ("pretrain", "both"):
        schedule = DDPMSchedule(T=config.T_DIFF)
        pretrain(
            model, train_loader, val_loader, schedule,
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
            print(f"Loaded pretrained weights from {pretrain_ckpt}")
        finetune(
            model, train_loader, val_loader,
            lr_ratios=config.FINETUNE_LR_RATIOS,
            base_lr=config.FINETUNE_LR,
            weight_decay=config.WEIGHT_DECAY,
            max_epochs=200,
            patience=config.EARLY_STOPPING_PATIENCE,
            checkpoint_path=str(finetune_ckpt),
            grad_clip=config.GRAD_CLIP,
            device=args.device,
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test CLI entry**

```bash
cd /home/justin/msc_ai/individual-project/contimask
python training/train.py --help
```

Expected: usage message printed, no import errors.

---

## Task 10: forward_func and attribution runner

**Files:**
- Create: `sepsis_attribution.py`
- Create: `tests/test_attribution.py`

- [ ] **Step 1: Write failing tests**

`tests/test_attribution.py`:
```python
import torch

import config
from models.diffusion_transformer import DiffusionTransformer
from sepsis_attribution import make_forward_func


def _model():
    return DiffusionTransformer(
        d_model=16, n_heads=2, n_layers=2, ffn_dim=32,
        dropout=0.0, n_features=config.N_FEATURES,
        time_embed_L=4, T_diff=100,
    ).eval()


def test_forward_func_standard_shape():
    model = _model()
    ff = make_forward_func(model, device="cpu")
    t = torch.rand(2, 72)
    X = torch.randn(2, 72, 39)
    dm = torch.ones(2, 72, 39)
    out = ff(t, X, dm)
    assert out.shape == (2,)
    assert ((out >= 0) & (out <= 1)).all()


def test_forward_func_k_sample_shape():
    model = _model()
    ff = make_forward_func(model, device="cpu")
    K, B, T, F = 5, 2, 72, 39
    t = torch.rand(K, B, T)
    X = torch.randn(K, B, T, F)
    dm = torch.ones(K, B, T, F)
    out = ff(t, X, dm)
    assert out.shape == (K, B)


def test_forward_func_k_sample_values_in_range():
    model = _model()
    ff = make_forward_func(model, device="cpu")
    K, B = 3, 4
    t = torch.rand(K, B, 72)
    X = torch.randn(K, B, 72, 39)
    dm = torch.ones(K, B, 72, 39)
    out = ff(t, X, dm)
    assert ((out >= 0) & (out <= 1)).all()
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_attribution.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'sepsis_attribution'`

- [ ] **Step 3: Implement `sepsis_attribution.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

import config
from attribution.mask_conti import ContiMask
from attribution.perturbation_conti import Deletion, MaskFunctionFourier


def make_forward_func(model: nn.Module, device: str = "cpu") -> Callable:
    """
    Returns a forward_func compatible with ContiMask's (t, X, data_mask) signature.
    Handles both standard (B, T, F) inputs and the K-sample (K, B, T, F) path
    used by the Deletion perturbation.
    """
    model.eval()

    def forward_func(t: torch.Tensor, X: torch.Tensor, data_mask: torch.Tensor) -> torch.Tensor:
        if X.dim() == 4:
            K, B, T, F = X.shape
            with torch.no_grad():
                logit = model.classify(
                    t.view(K * B, T).to(device),
                    X.view(K * B, T, F).to(device),
                    data_mask.view(K * B, T, F).to(device),
                )
            return torch.sigmoid(logit).view(K, B).cpu()
        with torch.no_grad():
            logit = model.classify(
                t.to(device), X.to(device), data_mask.to(device)
            )
        return torch.sigmoid(logit).squeeze(-1).cpu()  # (B,)

    return forward_func


def run_attribution(
    model: nn.Module,
    t: torch.Tensor,
    X: torch.Tensor,
    data_mask: torch.Tensor,
    perturbation: str = "D",
    device: str = "cpu",
    n_epochs: int = config.MASK_EPOCHS,
    target_area: float = config.MASK_TARGET_AREA,
    results_dir: str = "results/sepsis_attribution",
    patient_id: str = "unknown",
) -> torch.Tensor:
    """
    Run ContiMask attribution for a single patient.

    perturbation: "D" (Deletion), "GB" (GaussianBlur), or "FMA" (FadeMovingAverage)
    Returns: mask tensor of shape (1, T, N_FEATURES)
    """
    from attribution.perturbation_conti import FadeMovingAverage, GaussianBlur

    forward_func = make_forward_func(model, device=device)

    pert_mask = MaskFunctionFourier(
        hidden_dim=config.MASK_HIDDEN_DIM,
        L=config.MASK_L,
        features=config.N_FEATURES,
    ).to(device)

    if perturbation == "D":
        pert = Deletion(device=device)
    elif perturbation == "GB":
        pert = GaussianBlur(device=device)
    elif perturbation == "FMA":
        pert = FadeMovingAverage(device=device)
    else:
        raise ValueError(f"Unknown perturbation: {perturbation}")

    explainer = ContiMask(
        forward_func=forward_func,
        perturbation_func=pert,
        pert_mask=pert_mask,
        device=device,
    )
    explainer.attribute(
        t=t.to(device),
        X=X.to(device),
        data_mask=data_mask.to(device),
        n_epoch=n_epochs,
        target_area=target_area,
        optimization_strategy="evotorch",
    )

    fitted_mask = explainer.get_mask(t.to(device))

    import os
    os.makedirs(f"{results_dir}/{patient_id}", exist_ok=True)
    torch.save(fitted_mask, f"{results_dir}/{patient_id}/mask_{perturbation}.pt")
    torch.save(explainer.hist, f"{results_dir}/{patient_id}/hist_{perturbation}.pt")

    return fitted_mask
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_attribution.py -v
```

Expected: all 3 tests pass.

---

## Task 11: Full test suite + smoke test

**Files:** No new files.

- [ ] **Step 1: Run full test suite**

```bash
cd /home/justin/msc_ai/individual-project/contimask
python -m pytest tests/ -v --tb=short
```

Expected: all tests pass.

- [ ] **Step 2: Smoke-test data pipeline on real data**

```bash
python -c "
from data.dataset import SepsisDataset
import os
data_dir = os.path.expanduser('~/msc_ai/individual-project/sepsis_data/training_setA')
ds = SepsisDataset(data_dir)
t, X, dm, label = ds[0]
print('Patients:', len(ds))
print('t shape:', t.shape)
print('X shape:', X.shape)
print('data_mask shape:', dm.shape)
print('Static cols always observed (first patient):', dm[:50, 34].all().item(), dm[:50, 35].all().item())
print('Label:', label.item())
"
```

Expected:
```
Patients: 20338
t shape: torch.Size([72])
X shape: torch.Size([72, 39])
data_mask shape: torch.Size([72, 39])
Static cols always observed (first patient): True True
Label: 0.0
```

- [ ] **Step 3: Smoke-test model forward pass**

```bash
python -c "
import torch, config
from models.diffusion_transformer import DiffusionTransformer
model = DiffusionTransformer(config.D_MODEL, config.N_HEADS, config.N_LAYERS,
    config.FFN_DIM, config.DROPOUT, config.N_FEATURES, config.TIME_EMBED_L, config.T_DIFF)
t = torch.rand(4, 72); X = torch.randn(4, 72, 39); dm = torch.ones(4, 72, 39)
dm[:, 50:] = 0
out = model.classify(t, X, dm)
print('classify output shape:', out.shape)
assert not torch.isnan(out).any(), 'NaN in output!'
print('OK')
"
```

Expected: `classify output shape: torch.Size([4, 1])` then `OK`

- [ ] **Step 4: Smoke-test training pipeline (2 epochs)**

```bash
cd /home/justin/msc_ai/individual-project/contimask
python training/train.py \
  --phase both \
  --pretrain_epochs 2 \
  --data_dir ~/msc_ai/individual-project/sepsis_data/training_setA \
  --checkpoint_dir checkpoints
```

Expected: prints per-epoch loss/AUC, creates `checkpoints/pretrained_backbone.pt` and `checkpoints/best_model.pt`.

- [ ] **Step 5: Smoke-test attribution on one test patient**

```bash
python -c "
import torch
from pathlib import Path
from data.dataset import SepsisDataset
from models.diffusion_transformer import DiffusionTransformer
import config

# Load model
model = DiffusionTransformer(config.D_MODEL, config.N_HEADS, config.N_LAYERS,
    config.FFN_DIM, config.DROPOUT, config.N_FEATURES, config.TIME_EMBED_L, config.T_DIFF)
model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location='cpu'))
model.eval()

# Load one test patient
ds = SepsisDataset(
    '~/msc_ai/individual-project/sepsis_data/training_setA',
    norm_stats=torch.load('checkpoints/norm_stats.pt')
)
t, X, dm, label = ds[0]
t, X, dm = t.unsqueeze(0), X.unsqueeze(0), dm.unsqueeze(0)

from sepsis_attribution import run_attribution
mask = run_attribution(model, t, X, dm, perturbation='FMA', device='cpu', n_epochs=5)
print('Mask shape:', mask.shape)
assert mask.shape == (1, 72, 39), f'Unexpected shape: {mask.shape}'
print('Attribution smoke test passed.')
"
```

Expected: `Mask shape: torch.Size([1, 72, 39])` then `Attribution smoke test passed.`

---

## Task 12: Odds-change metrics

**Files:**
- Modify: `sepsis_attribution.py` (replace `compute_del_odds_change` and `compute_imp_odds_change` stubs)
- Test: `tests/test_attribution.py` (5 tests already written, currently failing with `NotImplementedError`)

**Background:** The paper reports average change in log-odds as the primary attribution quality metric.
- **Del odds change** — apply the attribution mask as Deletion: zero both `X` and `data_mask` where `mask=0`. The transformer treats deleted positions as never observed. Metric: `mean(logit_del - logit_orig)` over the batch.
- **Imp odds change** — apply the attribution mask as Imputation: zero `X` where `mask=0` but leave `data_mask` unchanged. Since features are z-score normalised, 0 equals the feature mean. Metric: `mean(logit_imp - logit_orig)` over the batch.

The two metrics are semantically distinct: Del changes the attention pattern (transformer ignores fully-deleted timesteps); Imp only changes feature values while the transformer's view of which timesteps exist stays the same.

- [ ] **Step 1: Confirm the failing tests**

```bash
uv run python -m pytest tests/test_attribution.py -k "odds" -v
```

Expected: 5 tests collected, all FAILED with `NotImplementedError`.

- [ ] **Step 2: Implement the two functions in `sepsis_attribution.py`**

Replace the two stub functions (lines 63–68 of the current file) with:

```python
@torch.no_grad()
def compute_del_odds_change(
    model: nn.Module,
    t: torch.Tensor,
    X: torch.Tensor,
    data_mask: torch.Tensor,
    mask: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Average change in log-odds when the attribution mask is applied as Deletion.

    Del zeros both X and data_mask where mask=0 — the transformer treats those
    positions as never observed.  Returns mean(logit_del - logit_orig) as a scalar.
    """
    model = model.to(device).eval()
    t, X, data_mask, mask = (
        t.to(device), X.to(device), data_mask.to(device), mask.to(device)
    )
    logit_orig = model.classify(t, X, data_mask).squeeze(-1)          # (B,)
    logit_del = model.classify(t, X * mask, data_mask * mask).squeeze(-1)  # (B,)
    return (logit_del - logit_orig).mean()


@torch.no_grad()
def compute_imp_odds_change(
    model: nn.Module,
    t: torch.Tensor,
    X: torch.Tensor,
    data_mask: torch.Tensor,
    mask: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Average change in log-odds when the attribution mask is applied as Imputation.

    Imp zeros X where mask=0 (inserting the feature mean in z-score space) but
    leaves data_mask unchanged.  Returns mean(logit_imp - logit_orig) as a scalar.
    """
    model = model.to(device).eval()
    t, X, data_mask, mask = (
        t.to(device), X.to(device), data_mask.to(device), mask.to(device)
    )
    logit_orig = model.classify(t, X, data_mask).squeeze(-1)            # (B,)
    logit_imp = model.classify(t, X * mask, data_mask).squeeze(-1)      # (B,)
    return (logit_imp - logit_orig).mean()
```

- [ ] **Step 3: Run the odds-change tests**

```bash
uv run python -m pytest tests/test_attribution.py -k "odds" -v
```

Expected: 5 tests pass.

- [ ] **Step 4: Run the full test suite**

```bash
uv run python -m pytest tests/ -v
```

Expected: 39 passed, 0 failed.

- [ ] **Step 5: Commit**

```bash
git add sepsis_attribution.py
git commit -m "feat: implement compute_del_odds_change and compute_imp_odds_change"
```
