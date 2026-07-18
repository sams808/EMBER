"""
glass_science.py is copied verbatim from Dataapp (already correctly cited,
proven working there) -- these tests re-verify it in Ember's own test
suite so it counts toward Ember's 100%-coverage house rule.
"""
import numpy as np
import pytest

import glass_science as gs


class TestOpticalBasicity:
    def test_pure_oxides_match_table(self):
        assert gs.optical_basicity([("SiO2", 1.0)])["basicity"] == pytest.approx(0.48)
        assert gs.optical_basicity([("Bi2O3", 100.0)])["basicity"] == pytest.approx(1.19)

    def test_oxygen_weighted_mixing(self):
        # 50:50 mol Na2O:SiO2 -> (0.5*1*1.11 + 0.5*2*0.48) / (0.5*1 + 0.5*2)
        res = gs.optical_basicity([("Na2O", 50.0), ("SiO2", 50.0)])
        assert res["basicity"] == pytest.approx((0.5 * 1.11 + 1.0 * 0.48) / 1.5, abs=1e-6)

    def test_wt_basis_differs_from_mol(self):
        mol = gs.optical_basicity([("Na2O", 50.0), ("SiO2", 50.0)], basis="mol")["basicity"]
        wt = gs.optical_basicity([("Na2O", 50.0), ("SiO2", 50.0)], basis="wt")["basicity"]
        assert mol != pytest.approx(wt)

    def test_empty_composition_raises(self):
        with pytest.raises(ValueError, match="Empty composition"):
            gs.optical_basicity([])

    def test_unknown_oxide_raises(self):
        with pytest.raises(ValueError, match="Known oxides"):
            gs.optical_basicity([("XeO4", 1.0)])

    def test_zero_total_fraction_raises_zero_oxygen(self):
        # Every component present but at 0 fraction -> den accumulates to
        # exactly 0 without tripping the per-component n_o<=0 guard.
        with pytest.raises(ValueError, match="zero oxygen content"):
            gs.optical_basicity([("SiO2", 0.0)])

    def test_oxide_with_no_oxygen_raises(self, monkeypatch):
        # Every real OPTICAL_BASICITY entry is an actual oxide (contains
        # O), so the n_o<=0 guard can't be reached with real table
        # contents -- monkeypatch in a non-oxide formula to exercise it.
        monkeypatch.setitem(gs.OPTICAL_BASICITY, "NaCl", 1.0)
        with pytest.raises(ValueError, match="contains no oxygen"):
            gs.optical_basicity([("NaCl", 1.0)])


class TestGlassnetAvailable:
    def test_returns_bool(self):
        assert isinstance(gs.glassnet_available(), bool)

    def test_exception_during_lookup_returns_false(self, monkeypatch):
        import importlib.util

        def boom(name):
            raise RuntimeError("boom")

        monkeypatch.setattr(importlib.util, "find_spec", boom)
        assert gs.glassnet_available() is False


class TestParseCompositionTable:
    def test_with_names(self):
        df = gs.parse_composition_table("name SiO2 Na2O\nA 70 30\nB 60 40")
        assert list(df.index) == ["A", "B"]
        assert df.loc["A", "SiO2"] == 70.0

    def test_without_names(self):
        df = gs.parse_composition_table("SiO2,Na2O\n70,30")
        assert df.iloc[0, 1] == 30.0
        assert list(df.index) == ["sample1"]

    def test_semicolon_separated(self):
        df = gs.parse_composition_table("SiO2;Na2O\n70;30")
        assert df.iloc[0, 0] == 70.0

    def test_tab_separated(self):
        df = gs.parse_composition_table("SiO2\tNa2O\n70\t30")
        assert df.iloc[0, 0] == 70.0

    def test_too_few_rows_raises(self):
        with pytest.raises(ValueError, match="header row"):
            gs.parse_composition_table("SiO2 Na2O")


@pytest.mark.skipif(not gs.glassnet_available(), reason="glasspy not installed")
class TestGlassnetPredict:
    def test_predicts_finite_tg(self):
        import pandas as pd
        df = pd.DataFrame([{"SiO2": 70.0, "Na2O": 30.0}])
        pred = gs.glassnet_predict(df)
        tg_cols = [c for c in pred.columns if "tg" in str(c).lower()]
        assert tg_cols
        assert np.isfinite(float(pred[tg_cols[0]].iloc[0]))
