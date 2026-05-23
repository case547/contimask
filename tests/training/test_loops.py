import torch
from torch.utils.data import DataLoader, TensorDataset

import config
from models.diffusion_transformer import DiffusionTransformer
from training.finetune import finetune
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
        weight_decay=0.01,
        base_lr=1e-3,
        max_epochs=2,
        patience=5,
        checkpoint_path=str(tmp_path / "best.pt"),
        device="cpu",
    )
    assert (tmp_path / "best.pt").exists()
