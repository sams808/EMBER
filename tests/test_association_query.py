"""
Tests for association_query: the engine-backed cross-correlation.

The expected values are computed by hand or from the raw CSV with
constants written out here, never read back from the module under test.
"""
import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

import association_query as aq
import data_model as dm

from conftest import REAL_CSV_PATH, requires_real_data

M_NA, M_CL, M_F = 22.990, 35.45, 18.998
M_SO4 = 32.06 + 4 * 15.999
AVOGADRO, CI_TO_BQ, YEAR_S = 6.02214076e23, 3.7e10, 365.2422 * 86400.0


def _dataset(rows):
    """rows: (tank, phase, type, analyte, inventory, units, volume_kL, density)."""
    frame = pl.DataFrame({
        "WasteSiteId": [r[0] for r in rows],
        "WastePhase": [r[1] for r in rows],
        "WasteType": [r[2] for r in rows],
        "Analyte": [r[3] for r in rows],
        "Inventory": [float(r[4]) for r in rows],
        "Units": [r[5] for r in rows],
        "Volume": [float(r[6]) for r in rows],
        "ComponentDensity": [float(r[7]) for r in rows],
        "VolumeUnits": ["kL"] * len(rows),
        "DensityUnits": ["g/mL"] * len(rows),
        "Inventory_RSD": [float(r[8]) if len(r) > 8 else 0.25 for r in rows],
    })
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(frame)
    dataset.report = None
    return dataset


@pytest.fixture
def small():
    # 1000 mol Na per layer; SO4 and Cl in round mole numbers.
    na = 1000 * M_NA / 1000.0
    return _dataset([
        ("241-T-101", "Supernatant", "X (Liquid)", "Na", na, "kg", 10, 1.2),
        ("241-T-101", "Supernatant", "X (Liquid)", "SO4", 100 * M_SO4 / 1000.0, "kg", 10, 1.2),
        ("241-T-101", "Supernatant", "X (Liquid)", "Cl", 50 * M_CL / 1000.0, "kg", 10, 1.2),
        ("241-T-101", "Sludge (Liquid & Solid)", "Y (Solid)", "Na", 2 * na, "kg", 20, 1.5),
        ("241-T-101", "Sludge (Liquid & Solid)", "Y (Solid)", "SO4", 20 * M_SO4 / 1000.0, "kg", 20, 1.5),
        ("241-T-101", "Sludge (Liquid & Solid)", "Y (Solid)", "Cl", 0.0, "kg", 20, 1.5),
        ("241-T-102", "Saltcake Solid", "Z (Solid)", "Na", na, "kg", 5, 1.6),
        ("241-T-102", "Saltcake Solid", "Z (Solid)", "SO4", 10 * M_SO4 / 1000.0, "kg", 5, 1.6),
        # no Cl row at all in this layer: unreported, not zero
    ])


class TestMolarMass:
    def test_species_element_nuclide_and_pair(self):
        assert aq.member_molar_mass("SO4") == pytest.approx(96.056, abs=0.01)
        assert aq.member_molar_mass("Cl") == pytest.approx(35.45)
        assert aq.member_molar_mass("129I") == 129.0
        assert aq.member_molar_mass("79Se") == 79.0
        assert aq.member_molar_mass("239/240Pu") is None


class TestMatrix:
    def test_mol_per_mol_na_by_layer(self, small):
        mx = aq.association_matrix(small, ["SO4", "Cl"], level="layer")
        rows = mx.values.set_index("WastePhase")
        assert rows.loc["Supernatant", "SO4"] == pytest.approx(0.100, rel=1e-9)
        assert rows.loc["Supernatant", "Cl"] == pytest.approx(0.050, rel=1e-9)
        assert rows.loc["Sludge (Liquid & Solid)", "SO4"] == pytest.approx(0.010, rel=1e-9)

    def test_reported_zero_stays_zero_and_unreported_stays_blank(self, small):
        mx = aq.association_matrix(small, ["SO4", "Cl"], level="layer")
        rows = mx.values.set_index("WastePhase")
        assert rows.loc["Sludge (Liquid & Solid)", "Cl"] == 0.0
        assert math.isnan(rows.loc["Saltcake Solid", "Cl"])

    def test_tank_level_sums_layers_before_dividing(self, small):
        mx = aq.association_matrix(small, ["SO4", "Cl"], level="tank")
        rows = mx.values.set_index("WasteSiteId")
        # 120 mol SO4 over 3000 mol Na -- not the mean of the two layer ratios
        assert rows.loc["241-T-101", "SO4"] == pytest.approx(120 / 3000, rel=1e-9)

    def test_kg_basis_and_layer_size(self, small):
        mx = aq.association_matrix(small, ["SO4", "Cl"], basis="kg", level="layer")
        rows = mx.values.set_index("WastePhase")
        assert rows.loc["Supernatant", "SO4"] == pytest.approx(100 * M_SO4 / 1000.0, rel=1e-9)
        assert rows.loc["Supernatant", "Volume_kL"] == 10
        assert rows.loc["Supernatant", "WasteMass_kg"] == pytest.approx(10 * 1.2 * 1000.0)
        assert rows.loc["Saltcake Solid", "PhaseFamily"] == "Saltcake"

    def test_wt_pct_is_mass_over_layer_mass(self, small):
        mx = aq.association_matrix(small, ["SO4", "Cl"], basis="wt%", level="layer")
        rows = mx.values.set_index("WastePhase")
        assert rows.loc["Supernatant", "SO4"] == pytest.approx(
            100.0 * (100 * M_SO4 / 1000.0) / (10 * 1.2 * 1000.0), rel=1e-9)

    def test_refusals(self, small):
        with pytest.raises(ValueError):
            aq.association_matrix(small, ["SO4"])
        with pytest.raises(ValueError):
            aq.association_matrix(small, ["SO4", "Cl"], basis="ppm")
        with pytest.raises(ValueError):
            aq.association_matrix(small, ["SO4", "Cl"], level="farm")

    def test_layer_without_sodium_is_blank_and_said(self):
        ds = _dataset([
            ("241-T-101", "Supernatant", "X", "Na", 23.0, "kg", 10, 1.2),
            ("241-T-101", "Supernatant", "X", "SO4", 9.6, "kg", 10, 1.2),
            ("241-T-101", "Supernatant", "X", "Cl", 3.5, "kg", 10, 1.2),
            ("241-T-102", "Supernatant", "W", "SO4", 9.6, "kg", 10, 1.0),
            ("241-T-102", "Supernatant", "W", "Cl", 3.5, "kg", 10, 1.0),
        ])
        mx = aq.association_matrix(ds, ["SO4", "Cl"], level="tank")
        rows = mx.values.set_index("WasteSiteId")
        assert math.isnan(rows.loc["241-T-102", "SO4"])
        assert any("no sodium" in w for w in mx.warnings)

    def test_null_waste_type_layer_is_kept(self):
        ds = _dataset([
            ("241-T-101", "Supernatant", None, "Na", 23.0, "kg", 10, 1.2),
            ("241-T-101", "Supernatant", None, "SO4", 9.6, "kg", 10, 1.2),
            ("241-T-101", "Supernatant", None, "Cl", 3.5, "kg", 10, 1.2),
        ])
        mx = aq.association_matrix(ds, ["SO4", "Cl"], level="layer")
        assert len(mx.values) == 1
        assert mx.values.loc[0, "SO4"] > 0
        assert mx.values.loc[0, "Volume_kL"] == 10

    def test_element_reported_as_mass_here_and_nuclide_there_is_flagged(self):
        ds = _dataset([
            ("241-T-101", "Supernatant", "X", "Na", 23.0, "kg", 10, 1.2),
            ("241-T-101", "Supernatant", "X", "Se", 2.0, "kg", 10, 1.2),
            ("241-T-101", "Supernatant", "X", "79Se", 0.01, "Ci", 10, 1.2),
            ("241-T-101", "Supernatant", "X", "Cl", 3.5, "kg", 10, 1.2),
            ("241-T-102", "Supernatant", "X", "Na", 23.0, "kg", 10, 1.2),
            ("241-T-102", "Supernatant", "X", "79Se", 0.01, "Ci", 10, 1.2),
            ("241-T-102", "Supernatant", "X", "Cl", 3.5, "kg", 10, 1.2),
        ])
        mixed = aq.association_matrix(ds, ["Se", "Cl"], level="tank")
        assert any("Se (1 layer(s) with a reported mass, 1 nuclide-only)" in w for w in mixed.warnings)
        clean = aq.association_matrix(ds, ["79Se", "Cl"], level="tank")
        assert not any("Not one quantity" in w for w in clean.warnings)


class TestReplicates:
    def _frame(self, rows):
        return pd.DataFrame(rows, columns=["A", "B"])

    def test_rounding_is_one_composition_and_chemistry_is_not(self):
        groups = aq.replicate_groups(
            self._frame([[1.000, 2.000], [1.008, 1.990], [1.050, 2.000], [1.000, 2.000]]), ["A", "B"])
        assert groups.iloc[0] == groups.iloc[1] == groups.iloc[3]
        assert groups.iloc[2] != groups.iloc[0]

    def test_blank_only_matches_blank_and_zero_matches_zero(self):
        groups = aq.replicate_groups(
            self._frame([[1.0, np.nan], [1.0, 0.0], [1.0, np.nan], [1.0, 0.0], [1.0, 1e-9]]), ["A", "B"])
        assert groups.iloc[0] == groups.iloc[2]
        assert groups.iloc[1] == groups.iloc[3]
        assert len({groups.iloc[0], groups.iloc[1], groups.iloc[4]}) == 3

    def test_row_order_does_not_change_the_number_of_groups(self):
        rng = np.random.default_rng(7)
        base = rng.lognormal(size=(40, 2))
        frame = self._frame(np.vstack([base, base[:15] * 1.004]).tolist())
        n = aq.replicate_groups(frame, ["A", "B"]).nunique()
        shuffled = frame.sample(frac=1.0, random_state=3)
        assert aq.replicate_groups(shuffled, ["A", "B"]).nunique() == n == 40


def _matrix(frame, members, basis=aq.MOL_PER_MOL_NA, level="layer"):
    return aq.AssociationMatrix(values=frame, members=list(members),
                                id_columns=[c for c in frame.columns if c not in members],
                                basis=basis, level=level)


class TestCorrelate:
    def _frame(self):
        rng = np.random.default_rng(11)
        a = rng.lognormal(size=30)
        b = a * rng.lognormal(sigma=0.4, size=30)
        c = rng.lognormal(size=30)
        return pd.DataFrame({"PhaseFamily": ["Saltcake"] * 20 + ["Sludge"] * 10, "A": a, "B": b, "C": c})

    def test_matches_scipy_on_the_same_rows(self):
        from scipy import stats
        frame = self._frame()
        res = aq.correlate(_matrix(frame, ["A", "B", "C"]), distinct_only=False)
        expected = stats.spearmanr(frame["A"], frame["B"]).statistic
        assert res.correlation.loc["A", "B"] == pytest.approx(expected, abs=1e-12)
        assert res.correlation.loc["B", "A"] == res.correlation.loc["A", "B"]
        assert res.n_used.loc["A", "B"] == 30

    def test_copies_of_one_composition_do_not_move_the_coefficient(self):
        frame = self._frame()
        copies = pd.concat([frame] + [frame.iloc[[0]]] * 25, ignore_index=True)
        base = aq.correlate(_matrix(frame, ["A", "B", "C"]))
        deduped = aq.correlate(_matrix(copies, ["A", "B", "C"]))
        counted = aq.correlate(_matrix(copies, ["A", "B", "C"]), distinct_only=False)
        assert deduped.n_rows == 55 and deduped.n_distinct == 30
        assert deduped.correlation.loc["A", "C"] == pytest.approx(base.correlation.loc["A", "C"], abs=1e-12)
        assert counted.correlation.loc["A", "C"] != pytest.approx(base.correlation.loc["A", "C"], abs=1e-6)
        assert any("counted once" in n for n in deduped.notes)

    def test_family_filter_and_its_refusal_at_tank_level(self):
        frame = self._frame()
        res = aq.correlate(_matrix(frame, ["A", "B", "C"]), family="Sludge")
        assert res.n_rows == 10
        with pytest.raises(ValueError):
            aq.correlate(_matrix(frame.drop(columns="PhaseFamily"), ["A", "B", "C"], level="tank"), family="Sludge")

    def test_too_few_observations_is_blank_not_a_number(self):
        frame = self._frame().head(5)
        res = aq.correlate(_matrix(frame, ["A", "B", "C"]), min_n=8)
        assert math.isnan(res.correlation.loc["A", "B"])
        assert any("left blank" in n for n in res.notes)

    def test_log_pearson_leaves_zeros_out_and_spearman_keeps_them(self):
        frame = self._frame()
        frame.loc[:4, "A"] = 0.0
        log = aq.correlate(_matrix(frame, ["A", "B", "C"]), method="pearson (log10)", distinct_only=False)
        rank = aq.correlate(_matrix(frame, ["A", "B", "C"]), distinct_only=False)
        assert log.n_used.loc["A", "B"] == 25
        assert rank.n_used.loc["A", "B"] == 30

    def test_blank_cells_are_dropped_pairwise(self):
        frame = self._frame()
        frame.loc[:9, "C"] = np.nan
        res = aq.correlate(_matrix(frame, ["A", "B", "C"]), distinct_only=False)
        assert res.n_used.loc["A", "C"] == 20 and res.n_used.loc["A", "B"] == 30

    def test_inventory_basis_never_merges_rows(self):
        frame = self._frame()
        copies = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        res = aq.correlate(_matrix(copies, ["A", "B", "C"], basis="kg"))
        assert res.n_distinct == res.n_rows == 31

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            aq.correlate(_matrix(self._frame(), ["A", "B", "C"]), method="magic")


class TestQualityAndCoOccurrence:
    def test_member_quality_counts_tanks_and_uncertainty_classes(self):
        ds = _dataset([
            ("241-T-101", "Supernatant", "X", "Na", 23.0, "kg", 10, 1.2, 0.2),
            ("241-T-101", "Supernatant", "X", "Br", 1.0, "kg", 10, 1.2, 1.004),
            ("241-T-101", "Supernatant", "X", "Cl", 3.0, "kg", 10, 1.2, 0.0),
            ("241-T-102", "Supernatant", "X", "Na", 23.0, "kg", 10, 1.2, 0.2),
            ("241-T-102", "Supernatant", "X", "Cl", 0.0, "kg", 10, 1.2, 0.3),
        ])
        q = aq.member_quality(ds, ["Br", "Cl"]).set_index("Member")
        assert q.loc["Br", "N_tanks"] == 1 and q.loc["Br", "N_tanks_in_scope"] == 2
        assert q.loc["Br", "Default_100pct_uncertainty_pct"] == 100.0
        assert q.loc["Cl", "N_tanks"] == 2 and q.loc["Cl", "N_tanks_positive"] == 1
        assert q.loc["Cl", "Unassigned_uncertainty_pct"] == 50.0
        assert q.loc["Cl", "Total_kg"] == pytest.approx(3.0)
        assert q.loc["Br", "Reported_as"] == "Br (kg)"

    def test_co_occurrence_counts_members_above_their_own_quantile(self):
        frame = pd.DataFrame({"WasteSiteId": list("abcd"), "A": [1, 2, 3, 4], "B": [4, 3, 2, 10], "C": [1, 1, 1, 9]})
        table = aq.co_occurrence(_matrix(frame, ["A", "B", "C"], level="tank"), quantile=0.75)
        top = table.iloc[0]
        assert top["WasteSiteId"] == "d" and top["N_high"] == 3
        assert top["High_members"] == "A, B, C"
        assert table.set_index("WasteSiteId").loc["a", "N_high"] == 0


@pytest.fixture(scope="module")
def real():
    dataset = dm.HanfordDataset()
    dataset.load(REAL_CSV_PATH, use_cache=True)
    return dataset


@pytest.fixture(scope="module")
def raw():
    return pl.read_csv(REAL_CSV_PATH, infer_schema_length=0).with_columns(
        pl.col("Inventory").cast(pl.Float64))


@requires_real_data
class TestRealDataset:
    def _raw_tank_total(self, raw, analyte):
        return (raw.filter(pl.col("Analyte") == analyte).group_by("WasteSiteId")
                .agg(pl.col("Inventory").sum().alias(analyte)).to_pandas().set_index("WasteSiteId")[analyte])

    def test_mol_per_mol_na_equals_the_raw_rows(self, real, raw):
        mx = aq.association_matrix(real, ["SO4", "129I", "Cl"], level="tank")
        got = mx.values.set_index("WasteSiteId")
        na_mol = self._raw_tank_total(raw, "Na") * 1000.0 / M_NA
        so4 = (self._raw_tank_total(raw, "SO4") * 1000.0 / M_SO4 / na_mol).dropna()
        np.testing.assert_allclose(got.loc[so4.index, "SO4"], so4, rtol=1e-9)
        # 129I: atoms = activity / decay constant, T1/2 = 1.57e7 y
        atoms = self._raw_tank_total(raw, "129I") * CI_TO_BQ * (1.57e7 * YEAR_S / math.log(2))
        iodine = (atoms / AVOGADRO / na_mol).dropna()
        np.testing.assert_allclose(got.loc[iodine.index, "129I"], iodine, rtol=2e-3)

    def test_a_tank_without_the_analyte_is_blank(self, real, raw):
        mx = aq.association_matrix(real, ["SO4", "Br"], level="tank")
        reported = set(raw.filter(pl.col("Analyte") == "Br").get_column("WasteSiteId").to_list())
        got = mx.values.set_index("WasteSiteId")["Br"]
        assert set(got.dropna().index) == reported

    def test_every_layer_and_tank_is_present_once(self, real, raw):
        layers = aq.association_matrix(real, ["SO4", "Cl"], level="layer").values
        n_layers = raw.select(["WasteSiteId", "WastePhase", "WasteType"]).unique().height
        assert len(layers) == n_layers
        assert not layers.duplicated(["WasteSiteId", "WastePhase", "WasteType"]).any()
        tanks = aq.association_matrix(real, ["SO4", "Cl"], level="tank").values
        assert len(tanks) == raw.get_column("WasteSiteId").n_unique()

    def test_inventories_all_correlate_and_concentrations_do_not(self, real):
        members = ["F", "Cl", "SO4", "PO4", "Cr", "UTOTAL"]
        kg = aq.correlate(aq.association_matrix(real, members, basis="kg", level="tank"), distinct_only=False)
        per_na = aq.correlate(aq.association_matrix(real, members, level="tank"), distinct_only=False)
        off = ~np.eye(len(members), dtype=bool)
        assert kg.correlation.to_numpy()[off].min() > 0.3          # tank size, not chemistry
        assert per_na.correlation.to_numpy()[off].min() < -0.2

    def test_spearman_does_not_depend_on_the_molar_masses(self, real):
        members = ["F", "SO4", "PO4"]
        tank = aq.association_matrix(real, members, level="tank")
        res = aq.correlate(tank, distinct_only=False)
        rescaled = tank.values.copy()
        rescaled["SO4"] = rescaled["SO4"] * 96.06
        again = aq.correlate(_matrix(rescaled, members, level="tank"), distinct_only=False)
        np.testing.assert_allclose(res.correlation.to_numpy(), again.correlation.to_numpy(), atol=1e-12)

    def test_mixed_representation_is_said_for_selenium_only_where_it_applies(self, real, raw):
        from query_engine import run_query

        def flagged(kind, target, basis):
            res = run_query(real, subject=(kind, target), basis=basis)
            return [w for w in res.warnings if w.startswith("Not one quantity")]

        # Counted from the raw rows: layers with a positive Se mass, and
        # layers with positive 79Se activity but no positive Se mass.
        layer = ["WasteSiteId", "WastePhase", "WasteType"]
        mass = raw.filter((pl.col("Analyte") == "Se") & (pl.col("Inventory") > 0)).select(layer).unique()
        activity = raw.filter((pl.col("Analyte") == "79Se") & (pl.col("Inventory") > 0)).select(layer).unique()
        only_nuclide = activity.join(mass, on=layer, how="anti", nulls_equal=True)
        text = flagged("element", "Se", "g")
        assert len(text) == 1
        assert f"Se ({mass.height} layer(s) with a reported mass, {only_nuclide.height} nuclide-only)" in text[0]
        assert flagged("element", "U", "g") == []          # UTOTAL in every layer
        assert flagged("element", "I", "g") == []          # 129I only: one representation
        assert flagged("analyte", "79Se", "g") == []
        assert flagged("element", "Se", "kg") == []        # not a bridged basis
