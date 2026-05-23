from __future__ import annotations

from pathlib import Path

from sklearn.model_selection import train_test_split


def stratified_patient_split(
    psv_files: list[Path],
    labels: list[int],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[list[Path], list[Path], list[Path]]:
    train_frac = 1.0 - val_frac - test_frac
    # Split seeds from https://github.com/patrick-kidger/NeuralCDE/blob/master/experiments/datasets/common.py#L26

    try:
        train_files, valtest_files, _, valtest_labels = train_test_split(
            psv_files, labels, train_size=train_frac, stratify=labels, random_state=0
        )
    except ValueError:
        train_files, valtest_files, _, valtest_labels = train_test_split(
            psv_files, labels, train_size=train_frac, random_state=0
        )

    try:
        val_files, test_files = train_test_split(
            valtest_files,
            test_size=test_frac / (val_frac + test_frac),
            stratify=valtest_labels,
            random_state=1,
        )
    except ValueError:
        val_files, test_files = train_test_split(
            valtest_files,
            test_size=test_frac / (val_frac + test_frac),
            random_state=1,
        )

    return list(train_files), list(val_files), list(test_files)
