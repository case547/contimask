import torch

from attribute import compute_retention


def test_macro_average():
    # Patient 0: feature 0 observed 4 times, retained 2 → rate 0.5
    # Patient 1: feature 0 observed 2 times, retained 2 → rate 1.0
    # Macro-average for feature 0: (0.5 + 1.0) / 2 = 0.75
    masks = [torch.zeros(1, 4, 2), torch.zeros(1, 4, 2)]
    masks[0][0, :2, 0] = 1.0
    masks[1][0, :2, 0] = 1.0

    dms = [torch.zeros(1, 4, 2), torch.zeros(1, 4, 2)]
    dms[0][0, :, 0] = 1.0   # patient 0: 4 observed
    dms[1][0, :2, 0] = 1.0  # patient 1: 2 observed

    retention = compute_retention(masks, dms)

    assert retention.shape == (2,)
    assert abs(retention[0].item() - 0.75) < 1e-5
    assert retention[1].item() == 0.0  # feature 1 never observed


def test_excludes_unobserved_patients():
    # Patient 0: feature 1 NOT observed → excluded from feature 1 average
    # Patient 1: feature 1 observed 2 times, retained 1 → rate 0.5
    masks = [torch.zeros(1, 4, 2), torch.zeros(1, 4, 2)]
    masks[1][0, 0, 1] = 1.0

    dms = [torch.zeros(1, 4, 2), torch.zeros(1, 4, 2)]
    dms[1][0, :2, 1] = 1.0

    retention = compute_retention(masks, dms)

    assert abs(retention[1].item() - 0.5) < 1e-5


def test_zero_retention():
    # Masks are all zero — nothing retained
    masks = [torch.zeros(1, 4, 3)]
    dms = [torch.ones(1, 4, 3)]

    retention = compute_retention(masks, dms)

    assert torch.equal(retention, torch.zeros(3))


def test_full_retention():
    # Masks equal data_mask — everything retained
    dm = torch.zeros(1, 4, 3)
    dm[0, :2, :] = 1.0  # 2 timesteps observed
    masks = [dm.clone()]
    dms = [dm.clone()]

    retention = compute_retention(masks, dms)

    assert torch.allclose(retention, torch.ones(3))


def test_no_observations_anywhere():
    # data_mask is all zeros → all features return 0.0, no division by zero
    masks = [torch.ones(1, 4, 2)]
    dms = [torch.zeros(1, 4, 2)]

    retention = compute_retention(masks, dms)

    assert torch.equal(retention, torch.zeros(2))


def test_mask_outside_data_mask_does_not_inflate():
    # mask=1 at unobserved positions should not count as retained
    masks = [torch.ones(1, 2, 1)]    # mask claims everything retained
    dms   = [torch.zeros(1, 2, 1)]   # nothing observed except one timestep
    dms[0][0, 0, 0] = 1.0            # only 1 of 2 timesteps observed
    # n_ret = (m * d).sum() = 1, n_obs = 1, rate = 1.0
    retention = compute_retention(masks, dms)
    assert abs(retention[0].item() - 1.0) < 1e-5
