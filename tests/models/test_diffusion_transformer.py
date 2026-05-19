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
    out = model.classify(t, X, dm)
    assert out.shape == (2, 1)


def test_padding_mask_excludes_tail(model):
    t, X, dm = _batch(B=2, actual_len=10)
    t2, X2, dm2 = _batch(B=2, actual_len=72)
    out1 = model.classify(t, X, dm)
    out2 = model.classify(t2, X2, dm2)
    assert out1.shape == out2.shape == (2, 1)
