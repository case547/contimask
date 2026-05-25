import torch

from sepsis_attribution import subsample_mask


def test_excess_trimmed():
    """Active observed entries are reduced to exactly n_target."""
    T, F = 4, 5
    mask = torch.ones(1, T, F)        # all 20 entries active
    data_mask = torch.ones(1, T, F)   # all 20 entries observed
    # target_area=0.5 → n_target = round(0.5 * 20) = 10
    out = subsample_mask(mask, data_mask, target_area=0.5, seed=42)
    assert out.dtype == torch.float32
    assert out.sum().item() == 10


def test_already_at_target():
    """Mask with exactly n_target active observed entries is returned unchanged."""
    T, F = 4, 5
    data_mask = torch.ones(1, T, F)
    n_target = round(0.5 * T * F)    # 10
    mask = torch.zeros(1, T, F)
    mask.view(1, -1)[0, :n_target] = 1.0
    out = subsample_mask(mask, data_mask, target_area=0.5, seed=42)
    assert torch.equal(out, mask)


def test_below_target():
    """Mask with fewer active entries than n_target is returned unchanged."""
    T, F = 4, 5
    data_mask = torch.ones(1, T, F)
    mask = torch.zeros(1, T, F)
    mask[0, 0, 0] = 1.0
    mask[0, 0, 1] = 1.0             # only 2 active, target is 10
    out = subsample_mask(mask, data_mask, target_area=0.5, seed=42)
    assert torch.equal(out, mask)


def test_reproducibility():
    """Same seed produces identical output on repeated calls."""
    T, F = 4, 5
    mask = torch.ones(1, T, F)
    data_mask = torch.ones(1, T, F)
    out1 = subsample_mask(mask, data_mask, target_area=0.5, seed=42)
    out2 = subsample_mask(mask, data_mask, target_area=0.5, seed=42)
    assert torch.equal(out1, out2)


def test_different_seeds_give_different_outputs():
    """Different seeds (with high probability) produce different outputs."""
    T, F = 20, 20
    mask = torch.ones(1, T, F)
    data_mask = torch.ones(1, T, F)
    out1 = subsample_mask(mask, data_mask, target_area=0.5, seed=0)
    out2 = subsample_mask(mask, data_mask, target_area=0.5, seed=1)
    assert not torch.equal(out1, out2)


def test_unobserved_positions_zeroed():
    """mask=1 positions where data_mask=0 become 0 after subsampling."""
    T, F = 4, 5
    data_mask = torch.zeros(1, T, F)
    data_mask[0, :2, :] = 1.0       # only 10 entries observed, n_target=5
    mask = torch.ones(1, T, F)      # mask=1 everywhere including unobserved
    out = subsample_mask(mask, data_mask, target_area=0.5, seed=42)
    # No output entry should be 1 where data_mask is 0
    assert (out * (1.0 - data_mask)).sum().item() == 0.0


def test_zero_target_area():
    """With target_area=0.0, all active entries are dropped (n_target=0)."""
    T, F = 4, 5
    mask = torch.ones(1, T, F)
    data_mask = torch.ones(1, T, F)
    out = subsample_mask(mask, data_mask, target_area=0.0, seed=42)
    assert torch.equal(out, torch.zeros_like(out))


def test_full_target_area():
    """With target_area=1.0, mask at or below target is returned unchanged."""
    T, F = 4, 5
    mask = torch.ones(1, T, F)
    data_mask = torch.ones(1, T, F)
    out = subsample_mask(mask, data_mask, target_area=1.0, seed=42)
    assert torch.equal(out, mask)


def test_batch_independence():
    """Result for item 0 in a batch-of-2 equals result for item 0 run alone."""
    T, F = 4, 5
    mask0 = torch.ones(1, T, F)
    dm0 = torch.ones(1, T, F)
    mask1 = torch.zeros(1, T, F)
    mask1[0, :2, :] = 1.0
    dm1 = torch.ones(1, T, F)

    mask_batch = torch.cat([mask0, mask1], dim=0)   # (2, T, F)
    dm_batch = torch.cat([dm0, dm1], dim=0)
    out_batch = subsample_mask(mask_batch, dm_batch, target_area=0.5, seed=42)
    out_alone = subsample_mask(mask0, dm0, target_area=0.5, seed=42)

    assert torch.equal(out_batch[0], out_alone[0])
    assert torch.equal(out_batch[1], mask1[0])
