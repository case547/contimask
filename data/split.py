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
    try:
        train_val_files, test_files, train_val_labels, _ = train_test_split(
            psv_files, labels, test_size=test_frac, stratify=labels, random_state=seed
        )
    except ValueError:
        train_val_files, test_files, train_val_labels, _ = train_test_split(
            psv_files, labels, test_size=test_frac, random_state=seed
        )

    try:
        train_files, val_files = train_test_split(
            train_val_files,
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
