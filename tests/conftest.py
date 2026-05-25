from pathlib import Path

import pytest


def _write_psv(path: Path, n_rows: int, has_sepsis: bool = False) -> None:
    header = (
        "HR|O2Sat|Temp|SBP|MAP|DBP|Resp|EtCO2|BaseExcess|HCO3|FiO2|pH|PaCO2|SaO2|"
        "AST|BUN|Alkalinephos|Calcium|Chloride|Creatinine|Bilirubin_direct|Glucose|"
        "Lactate|Magnesium|Phosphate|Potassium|Bilirubin_total|TroponinI|Hct|Hgb|"
        "PTT|WBC|Fibrinogen|Platelets|Age|Gender|Unit1|Unit2|HospAdmTime|ICULOS|SepsisLabel"
    )
    rows = []
    for i in range(n_rows):
        tv = ["NaN"] * 34
        static = ["65.0", "0", "NaN", "NaN", "-1.0"]
        iculos = str(i + 1)
        label = "1" if (has_sepsis and i == n_rows - 1) else "0"
        rows.append("|".join(tv + static + [iculos, label]))
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


@pytest.fixture()
def mock_data_dir(tmp_path):
    _write_psv(tmp_path / "p000001.psv", n_rows=50, has_sepsis=False)
    _write_psv(tmp_path / "p000002.psv", n_rows=80, has_sepsis=True)  # >72 rows, sepsis
    _write_psv(tmp_path / "p000003.psv", n_rows=10, has_sepsis=False)  # short stay
    return tmp_path


@pytest.fixture()
def split_data_dir(tmp_path):
    # 10 patients (3 positive, 7 negative) — enough for a valid 70/15/15 stratified split
    for i in range(7):
        _write_psv(tmp_path / f"neg{i:03d}.psv", n_rows=10, has_sepsis=False)
    for i in range(3):
        _write_psv(tmp_path / f"pos{i:03d}.psv", n_rows=10, has_sepsis=True)
    return tmp_path
