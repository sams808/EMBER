"""
Shared pytest fixtures. Qt-specific autouse fixtures (synchronous worker,
hermetic QSettings, blocked dialogs) are added once qt_worker.py and the
first Qt-based workspace exist (M2) — kept out for now since importing them
here would require PySide6 before any Qt module is written.
"""
from pathlib import Path

import polars as pl
import pytest

from data_model import HanfordDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CSV_PATH = REPO_ROOT / "Hanford.csv"
REAL_ATTRS_PATH = REPO_ROOT / "Tank_attributes.csv"

requires_real_data = pytest.mark.skipif(
    not REAL_CSV_PATH.exists(), reason="Hanford.csv not present locally (gitignored dev seed data)"
)


def _raw_composition_rows() -> pl.DataFrame:
    """Small, deliberately-chosen composition rows:
    - a plain isotope (137Cs), a metastable isotope (113mCd)
    - both combined-isotope bug-fix cases (239/240Pu, 243/244Cm)
    - a legitimately non-elemental analyte (Total Alpha)
    - a genuine duplicate (WasteSiteId, Analyte) key across WastePhase
      (241-A-101 / Fe appears twice, in Solid and Liquid phase) to prove
      aggregation must group-by-sum rather than assume a unique key
    - both Ci and kg units, two tanks in two different farms (A, AN)
    """
    return pl.DataFrame({
        "WasteSiteId": [
            "241-A-101", "241-A-101", "241-A-101", "241-A-101", "241-A-101", "241-A-101",
            "241-AN-104", "241-AN-104", "241-AN-104",
        ],
        "Analyte": [
            "137Cs", "Fe", "239/240Pu", "113mCd", "Total Alpha", "Fe",
            "137Cs", "Na", "243/244Cm",
        ],
        "WastePhase": [
            "Liquid", "Solid", "Solid", "Liquid", "Liquid", "Liquid",
            "Sludge", "Sludge", "Sludge",
        ],
        "WasteType": ["T1"] * 6 + ["T2"] * 3,
        "Inventory": [100.0, 50.0, 0.002, 0.5, 0.01, 10.0, 200.0, 500.0, 0.001],
        "Units": ["Ci", "kg", "Ci", "kg", "Ci", "kg", "Ci", "kg", "Ci"],
    })


def _raw_attributes_rows() -> pl.DataFrame:
    return pl.DataFrame({
        "Name": ["241-A-101", "241-AN-104"],
        "TankType": ["SST-4", "DST"],
        "TankStatus": ["", ""],
        "TankIntegrity": ["Sound", "Assumed leaker"],
    })


@pytest.fixture
def sample_dataset() -> HanfordDataset:
    """A HanfordDataset with small, hand-built, already-cleaned data — no
    disk I/O. Exercises the real cleaning pipeline (_clean_dataframe /
    _clean_attributes_dataframe / _merge_tank_attributes) against fixture
    rows rather than mocking it away."""
    dataset = HanfordDataset()
    df = dataset._clean_dataframe(_raw_composition_rows())
    attrs = dataset._clean_attributes_dataframe(_raw_attributes_rows())
    dataset.attrs_df = attrs
    dataset.df = dataset._merge_tank_attributes(df, attrs)
    dataset.report = None
    return dataset


@pytest.fixture
def real_csv_paths():
    """(composition_path, attributes_path) for the real local dataset.
    Use with @requires_real_data."""
    return REAL_CSV_PATH, REAL_ATTRS_PATH
