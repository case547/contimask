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
