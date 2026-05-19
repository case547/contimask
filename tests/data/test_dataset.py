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
    assert not torch.allclose(X_norm, X_raw)
