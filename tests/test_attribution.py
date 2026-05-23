import torch

import config
from models.diffusion_transformer import DiffusionTransformer
from sepsis_attribution import (
    compute_del_odds_change,
    compute_imp_odds_change,
    make_forward_func,
)


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


# ---------------------------------------------------------------------------
# Odds-change metric tests
# ---------------------------------------------------------------------------

def _batch(B=4, T=72, F=39):
    """Return (t, X, data_mask, binary_mask) with a 50 % hard mask."""
    t = torch.rand(B, T)
    X = torch.randn(B, T, F)
    dm = torch.ones(B, T, F)
    mask = (torch.rand(B, T, F) > 0.5).float()  # 50 % kept
    return t, X, dm, mask


def test_del_odds_change_is_scalar():
    model = _model()
    t, X, dm, mask = _batch()
    result = compute_del_odds_change(model, t, X, dm, mask, device="cpu")
    assert result.shape == ()  # scalar tensor


def test_del_odds_change_full_mask_is_zero():
    """When the mask keeps everything, Del perturbation leaves X unchanged,
    so the log-odds change should be exactly 0."""
    model = _model()
    t, X, dm, _ = _batch()
    full_mask = torch.ones_like(dm)
    result = compute_del_odds_change(model, t, X, dm, full_mask, device="cpu")
    assert result.abs() < 1e-4


def test_imp_odds_change_is_scalar():
    model = _model()
    t, X, dm, mask = _batch()
    result = compute_imp_odds_change(model, t, X, dm, mask, device="cpu")
    assert result.shape == ()


def test_imp_odds_change_full_mask_is_zero():
    """When the mask keeps everything, zero-imputation changes nothing,
    so the log-odds change should be exactly 0."""
    model = _model()
    t, X, dm, _ = _batch()
    full_mask = torch.ones_like(dm)
    result = compute_imp_odds_change(model, t, X, dm, full_mask, device="cpu")
    assert result.abs() < 1e-4


def test_imp_odds_change_zero_mask_uses_mean():
    """With an all-zero mask every feature is imputed with 0 (≈ feature mean
    in normalised data).  The result must still be a finite scalar."""
    model = _model()
    t, X, dm, _ = _batch()
    zero_mask = torch.zeros_like(dm)
    result = compute_imp_odds_change(model, t, X, dm, zero_mask, device="cpu")
    assert result.shape == ()
    assert result.isfinite()
