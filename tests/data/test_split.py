from pathlib import Path

from data.dataset import SepsisDataset
from data.split import stratified_patient_split


def test_split_sizes(split_data_dir):
    ds = SepsisDataset(split_data_dir)
    labels = [int(ds[i][3].item()) for i in range(len(ds))]
    train_files, val_files, test_files = stratified_patient_split(
        ds.psv_files, labels, val_frac=0.15, test_frac=0.15
    )
    total = len(train_files) + len(val_files) + len(test_files)
    assert total == len(ds.psv_files)


def test_split_no_overlap(split_data_dir):
    ds = SepsisDataset(split_data_dir)
    labels = [int(ds[i][3].item()) for i in range(len(ds))]
    train_files, val_files, test_files = stratified_patient_split(
        ds.psv_files, labels, val_frac=0.15, test_frac=0.15
    )
    all_files = set(train_files) | set(val_files) | set(test_files)
    assert len(all_files) == len(ds.psv_files)


def test_split_returns_path_lists(split_data_dir):
    ds = SepsisDataset(split_data_dir)
    labels = [int(ds[i][3].item()) for i in range(len(ds))]
    train_files, val_files, test_files = stratified_patient_split(ds.psv_files, labels)
    assert all(isinstance(p, Path) for p in train_files)
