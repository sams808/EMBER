import math

import numpy as np
import pytest

import matrix_science as ms


class TestLog10Safe:
    def test_positive_values(self):
        out = ms.log10_safe([100.0, 10.0, 1.0])
        assert out.tolist() == pytest.approx([2.0, 1.0, 0.0])

    def test_zero_and_negative_become_nan(self):
        out = ms.log10_safe([0.0, -5.0, 10.0])
        assert math.isnan(out[0])
        assert math.isnan(out[1])
        assert out[2] == pytest.approx(1.0)


class TestTopTanksByInventory:
    def test_ranks_by_total_kg(self, sample_dataset):
        assert ms.top_tanks_by_inventory(sample_dataset, "kg", 1) == ["241-AN-104"]

    def test_max_tanks_limits_result(self, sample_dataset):
        assert len(ms.top_tanks_by_inventory(sample_dataset, "kg", 1)) == 1
        assert len(ms.top_tanks_by_inventory(sample_dataset, "kg", 10)) == 2


class TestMatrixLongWide:
    def test_basic_inventory_mode(self, sample_dataset):
        long_pdf, wide = ms.matrix_long_wide(sample_dataset, unit="kg", top_n_elements=10)
        assert set(long_pdf["Element"]) == {"Na", "Fe", "Cd"}
        assert list(wide.columns) == ["WasteSiteId", "Na", "Fe", "Cd"]  # ordered by total desc
        wide = wide.set_index("WasteSiteId")
        assert wide.loc["241-A-101", "Fe"] == pytest.approx(60.0)  # duplicate-key sum
        assert wide.loc["241-A-101", "Na"] == pytest.approx(0.0)  # fill_value for absent
        assert wide.loc["241-AN-104", "Na"] == pytest.approx(500.0)

    def test_log10_inventory_mode(self, sample_dataset):
        long_pdf, _ = ms.matrix_long_wide(sample_dataset, unit="kg", value_mode="log10_inventory")
        row = long_pdf[long_pdf["Element"] == "Fe"].iloc[0]
        assert row["log10_Inventory_kg"] == pytest.approx(math.log10(60.0))

    def test_fraction_mode_denominator_is_displayed_elements_only(self, sample_dataset):
        # 241-A-101 has Fe=60, Cd=0.5 among the *displayed* top elements --
        # fraction denominator is 60.5, not the tank's true full kg total.
        long_pdf, _ = ms.matrix_long_wide(sample_dataset, unit="kg", value_mode="fraction")
        a101 = long_pdf[long_pdf["WasteSiteId"] == "241-A-101"].set_index("Element")
        assert a101.loc["Fe", "Fraction_kg"] == pytest.approx(60.0 / 60.5)
        assert a101.loc["Cd", "Fraction_kg"] == pytest.approx(0.5 / 60.5)

    def test_min_inventory_filters_elements(self, sample_dataset):
        long_pdf, _ = ms.matrix_long_wide(sample_dataset, unit="kg", min_inventory=1.0)
        assert "Cd" not in set(long_pdf["Element"])  # 0.5 kg excluded

    def test_top_n_elements_limits_columns(self, sample_dataset):
        _, wide = ms.matrix_long_wide(sample_dataset, unit="kg", top_n_elements=1)
        assert list(wide.columns) == ["WasteSiteId", "Na"]

    def test_tank_subset_restricts_rows(self, sample_dataset):
        long_pdf, _ = ms.matrix_long_wide(sample_dataset, unit="kg", tank_subset=["241-A-101"])
        assert set(long_pdf["WasteSiteId"]) == {"241-A-101"}

    def test_empty_when_unit_not_present(self, sample_dataset):
        long_pdf, wide = ms.matrix_long_wide(sample_dataset, unit="XYZ")
        assert long_pdf.empty and wide.empty
