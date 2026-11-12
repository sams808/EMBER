"""
site_summary: the figures behind the Site overview and the tank profile.
They must be the engine's figures, so each is checked against a direct
engine call on the real dataset, and against hand-built rows for the
rules that matter on a map: unreported is not zero, a zero stays a zero.
"""
import math

import polars as pl
import pytest

import association_query as aq
import data_model as dm
import site_summary as ss
from query_engine import run_query

from conftest import REAL_CSV_PATH, requires_real_data


@pytest.fixture(scope="module")
def real_dataset():
    dataset = dm.HanfordDataset()
    dataset.load(REAL_CSV_PATH, use_cache=True)
    return dataset


def _layered_dataset() -> dm.HanfordDataset:
    """Two tanks with stated volumes. 241-A-101 has two layers and reports
    SO4 in both; 241-AN-104 reports Na and a ZERO for SO4; 241-AN-105
    reports Na only (no SO4 record at all)."""
    rows = [
        # tank, analyte, phase, type, inventory, units, volume kL, density
        ("241-A-101", "Na", "Supernatant", "T1", 2300.0, "kg", 100.0, 1.2),
        ("241-A-101", "SO4", "Supernatant", "T1", 960.6, "kg", 100.0, 1.2),
        ("241-A-101", "Na", "Sludge", "T2", 2300.0, "kg", 50.0, 1.6),
        ("241-A-101", "SO4", "Sludge", "T2", 96.06, "kg", 50.0, 1.6),
        ("241-A-101", "137Cs", "Sludge", "T2", 40.0, "Ci", 50.0, 1.6),
        ("241-AN-104", "Na", "Supernatant", "T1", 4600.0, "kg", 400.0, 1.3),
        ("241-AN-104", "SO4", "Supernatant", "T1", 0.0, "kg", 400.0, 1.3),
        ("241-AN-105", "Na", "Supernatant", "T1", 1150.0, "kg", 300.0, 1.1),
    ]
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame({
        "WasteSiteId": [r[0] for r in rows],
        "Analyte": [r[1] for r in rows],
        "WastePhase": [r[2] for r in rows],
        "WasteType": [r[3] for r in rows],
        "Inventory": [r[4] for r in rows],
        "Units": [r[5] for r in rows],
        "Volume": [r[6] for r in rows],
        "VolumeUnits": ["kL"] * len(rows),
        "ComponentDensity": [r[7] for r in rows],
        "DensityUnits": ["g/mL"] * len(rows),
    }))
    dataset.report = None
    return dataset


class TestTankTable:
    def test_volume_is_the_sum_of_the_distinct_layers(self):
        table = ss.tank_table(_layered_dataset()).set_index("WasteSiteId")
        assert table.loc["241-A-101", "Volume_kL"] == pytest.approx(150.0)
        assert table.loc["241-A-101", "N_layers"] == 2
        assert table.loc["241-A-101", "WasteMass_kg"] == pytest.approx(100e3 * 1.2 + 50e3 * 1.6)
        assert table.loc["241-AN-104", "Volume_kL"] == pytest.approx(400.0)
        assert table.loc["241-A-101", "TankFarm"] == "A"

    def test_no_stated_volume_is_blank_not_zero(self, sample_dataset):
        table = ss.tank_table(sample_dataset)
        assert len(table) == 2
        assert table["Volume_kL"].isna().all()

    def test_scope_limits_the_tanks_and_their_layers(self):
        table = ss.tank_table(_layered_dataset(), {"WastePhase": ["Sludge"]})
        assert list(table["WasteSiteId"]) == ["241-A-101"]
        assert table["Volume_kL"].iloc[0] == pytest.approx(50.0)

    def test_layers_of_one_tank(self):
        layers = ss.tank_layers(_layered_dataset(), "241-A-101")
        assert list(layers["Waste phase"]) == ["Supernatant", "Sludge"]   # largest volume first
        assert list(layers["Volume (kL)"]) == [100.0, 50.0]
        assert ss.tank_layers(_layered_dataset(), "241-ZZ-999").empty


class TestTankValues:
    def test_unreported_is_absent_and_a_zero_stays_a_zero(self):
        values = ss.tank_values(_layered_dataset(), "SO4", "kg")
        assert values.values["241-A-101"] == pytest.approx(960.6 + 96.06)
        assert values.values["241-AN-104"] == 0.0
        assert "241-AN-105" not in values.values
        assert list(values.table["WasteSiteId"]) == ["241-A-101", "241-AN-104"]
        assert values.additive and values.unit == "kg"

    def test_mol_per_mol_na_by_hand(self):
        # 1056.66 kg SO4 / 96.06 = 11.0 kmol; 4600 kg Na / 22.99 = 200.09 kmol
        values = ss.tank_values(_layered_dataset(), "SO4", ss.MOL_PER_MOL_NA)
        so4 = (960.6 + 96.06) / (32.06 + 4 * 15.999)
        na = 4600.0 / 22.98976928
        assert values.values["241-A-101"] == pytest.approx(so4 / na, rel=2e-3)
        assert values.values["241-AN-104"] == 0.0
        assert not values.additive
        assert values.unit == "mol per mol Na"
        assert not any(d.startswith("Na:") for d in values.disclosures)

    def test_concentration_bases_use_the_whole_tank(self):
        dataset = _layered_dataset()
        per_litre = ss.tank_values(dataset, "SO4", "g/L")
        assert per_litre.values["241-A-101"] == pytest.approx(1056.66e3 / 150e3)
        weight = ss.tank_values(dataset, "SO4", "wt%")
        assert weight.values["241-A-101"] == pytest.approx(100.0 * 1056.66 / (120e3 + 80e3))

    def test_activity_basis_and_bridged_mass(self):
        dataset = _layered_dataset()
        curies = ss.tank_values(dataset, "137Cs", "Ci")
        assert curies.values == {"241-A-101": 40.0}
        mass = ss.tank_values(dataset, "137Cs", "kg")
        assert 0 < mass.values["241-A-101"] < 1e-3          # 40 Ci of 137Cs is under a gram
        assert any("bridged" in d for d in mass.disclosures)

    def test_a_component_with_no_activity_gives_no_values_and_says_why(self):
        values = ss.tank_values(_layered_dataset(), "SO4", "Ci")
        assert values.values == {}
        assert values.table.empty
        assert any("activity basis" in w for w in values.warnings)

    def test_waste_volume_is_a_component(self):
        values = ss.tank_values(_layered_dataset(), ss.WASTE_VOLUME)
        assert values.values["241-AN-104"] == pytest.approx(400.0)
        assert values.unit == "kL" and values.title == "Waste volume (kL)"

    def test_sodium_on_its_own_basis_is_one(self):
        values = ss.tank_values(_layered_dataset(), "Na", ss.MOL_PER_MOL_NA)
        assert all(v == pytest.approx(1.0) for v in values.values.values())

    def test_scope_is_honoured(self):
        values = ss.tank_values(_layered_dataset(), "SO4", "kg", {"WastePhase": ["Sludge"]})
        assert values.values == {"241-A-101": pytest.approx(96.06)}

    def test_unknown_basis_and_empty_component_are_refused(self):
        with pytest.raises(ValueError):
            ss.tank_values(_layered_dataset(), "SO4", "ppm")
        with pytest.raises(ValueError):
            ss.tank_values(_layered_dataset(), "  ", "kg")

    def test_an_unknown_component_is_reported_not_raised(self):
        values = ss.tank_values(_layered_dataset(), "Unobtainium", "kg")
        assert values.values == {}
        assert any("not reported" in d for d in values.disclosures)


class TestResolveComponent:
    def test_plain_names_and_case(self):
        dataset = _layered_dataset()
        assert ss.resolve_component(dataset, "sulfate") == "SO4"
        assert ss.resolve_component(dataset, "so4") == "SO4"
        assert ss.resolve_component(dataset, "na") == "Na"
        assert ss.resolve_component(dataset, "Cs-137") == "137Cs"
        assert ss.resolve_component(dataset, "waste volume") == ss.WASTE_VOLUME
        assert ss.resolve_component(dataset, " Xx ") == "Xx"

    def test_choices_start_with_the_volume_and_the_common_components(self):
        choices = ss.component_choices(_layered_dataset())
        assert choices[0] == ss.WASTE_VOLUME
        assert choices[1:4] == ["Na", "SO4", "137Cs"]
        assert len(choices) == len(set(choices))


class TestSiteStats:
    def test_counts_and_engine_totals(self):
        dataset = _layered_dataset()
        stats = ss.site_stats(dataset)
        assert (stats.n_tanks, stats.n_tanks_total, stats.n_records) == (3, 3, 8)
        assert stats.n_analytes == 3 and stats.n_layers == 4
        assert stats.volume_kL == pytest.approx(850.0)
        assert stats.sodium_kg == pytest.approx(2300.0 * 2 + 4600.0 + 1150.0)
        assert stats.cs137_ci == pytest.approx(40.0)
        assert stats.sr90_ci is None                       # not reported: blank, not zero
        assert stats.n_single_shell is None                # no attributes file

    def test_scope_narrows_every_figure(self):
        stats = ss.site_stats(_layered_dataset(), {"WastePhase": ["Sludge"]})
        assert (stats.n_tanks, stats.n_tanks_total) == (1, 3)
        assert stats.n_records == 3 and stats.n_records_total == 8
        assert stats.volume_kL == pytest.approx(50.0)
        assert stats.sodium_kg == pytest.approx(2300.0)

    def test_shell_split_from_the_attributes(self, sample_dataset):
        stats = ss.site_stats(sample_dataset)
        assert (stats.n_single_shell, stats.n_double_shell) == (1, 1)
        assert stats.volume_kL is None

    def test_tank_headlines(self):
        head = ss.tank_headlines(_layered_dataset(), ["241-AN-104", "241-A-101"])
        assert list(head["WasteSiteId"]) == ["241-AN-104", "241-A-101"]     # the order asked for
        row = head.set_index("WasteSiteId").loc["241-A-101"]
        assert row["Na_kg"] == pytest.approx(4600.0)
        assert row["Na_wt_pct"] == pytest.approx(100.0 * 4600.0 / (120e3 + 80e3))
        assert row["Cs137_Ci"] == pytest.approx(40.0)
        assert math.isnan(row["Sr90_Ci"])
        assert row["N_layers"] == 2


@requires_real_data
class TestAgainstTheEngineOnTheRealDataset:
    def test_tank_volumes_match_the_engine_backed_matrix(self, real_dataset):
        table = ss.tank_table(real_dataset).set_index("WasteSiteId")
        matrix = aq.association_matrix(real_dataset, ["Na", "Al"], basis="kg", level="tank")
        engine = matrix.values.set_index("WasteSiteId")
        assert len(table) == 177
        for tank in table.index:
            assert table.loc[tank, "Volume_kL"] == pytest.approx(engine.loc[tank, "Volume_kL"], rel=1e-12)
            assert table.loc[tank, "WasteMass_kg"] == pytest.approx(engine.loc[tank, "WasteMass_kg"], rel=1e-12)

    def test_site_volume_is_the_public_figure(self, real_dataset):
        # The tank farms hold about 53 to 56 million gallons; the dataset
        # itself states 200,631 kL.
        stats = ss.site_stats(real_dataset)
        assert stats.volume_kL == pytest.approx(200631.163, rel=1e-9)
        assert 52.0 < stats.volume_kL * ss.GALLONS_PER_KL / 1e6 < 57.0

    def test_stat_figures_are_the_engine_totals(self, real_dataset):
        stats = ss.site_stats(real_dataset)
        assert (stats.n_tanks, stats.n_single_shell, stats.n_double_shell) == (177, 149, 28)
        assert stats.n_records == real_dataset.require_df().height
        na = run_query(real_dataset, subject=("element", "Na"), basis="kg")
        assert stats.sodium_kg == pytest.approx(float(na.table["Total_kg"].sum()), rel=1e-12)
        na_wt = run_query(real_dataset, subject=("element", "Na"), basis="wt%")
        assert stats.sodium_wt_pct == pytest.approx(float(na_wt.table["wt_pct"].iloc[0]), rel=1e-12)
        for field, nuclide in (("cs137_ci", "137Cs"), ("sr90_ci", "90Sr")):
            direct = run_query(real_dataset, subject=("analyte", nuclide), basis="Ci")
            assert getattr(stats, field) == pytest.approx(float(direct.table["Total_Ci"].sum()), rel=1e-12)

    def test_scoped_stats_match_scoped_queries(self, real_dataset):
        scope = {"TankFarm": ["AN"]}
        stats = ss.site_stats(real_dataset, scope)
        assert (stats.n_tanks, stats.n_double_shell) == (7, 7)
        direct = run_query(real_dataset, subject=("element", "Na"), basis="kg", scope=scope)
        assert stats.sodium_kg == pytest.approx(float(direct.table["Total_kg"].sum()), rel=1e-12)

    def test_map_values_are_the_engine_values(self, real_dataset):
        sulfate = ss.tank_values(real_dataset, "SO4", ss.MOL_PER_MOL_NA)
        matrix = aq.association_matrix(real_dataset, ["SO4", "Na"], basis=aq.MOL_PER_MOL_NA, level="tank")
        engine = matrix.values.set_index("WasteSiteId")["SO4"].dropna()
        assert set(sulfate.values) == set(engine.index) and len(sulfate.values) == 177
        for tank, value in sulfate.values.items():
            assert value == pytest.approx(float(engine[tank]), rel=1e-12)

        cesium = ss.tank_values(real_dataset, "137Cs", "Ci")
        direct = run_query(real_dataset, subject=("analyte", "137Cs"), basis="Ci",
                           group_by=["WasteSiteId"]).table.set_index("WasteSiteId")["Total_Ci"]
        assert cesium.values == {t: pytest.approx(float(v)) for t, v in direct.items()}
        assert cesium.table["WasteSiteId"].iloc[0] == "241-AZ-101"
        assert sum(cesium.values.values()) == pytest.approx(
            ss.site_stats(real_dataset).cs137_ci, rel=1e-9)

    def test_sum_of_tank_inventories_is_the_site_total(self, real_dataset):
        sodium = ss.tank_values(real_dataset, "Na", "kg")
        assert sum(sodium.values.values()) == pytest.approx(ss.site_stats(real_dataset).sodium_kg, rel=1e-9)


class TestFormatting:
    def test_numbers(self):
        assert ss.format_number(None) == ss.BLANK
        assert ss.format_number(float("nan")) == ss.BLANK
        assert ss.format_number(0) == "0"
        assert ss.format_number(46597) == "46,600"
        assert ss.format_number(3894600.0) == "3,890,000"
        assert ss.format_number(1234.5, 4) == "1,234"
        assert ss.format_number(200.631, 4) == "200.6"
        assert ss.format_number(80.0) == "80"
        assert ss.format_number(0.045713) == "0.0457"
        assert ss.format_number(1.0) == "1"
        assert ss.format_number(-12.34) == "-12.3"
        assert ss.format_number(4.57e-6) == "4.57×10⁻⁶"

    def test_compact(self):
        assert ss.format_compact(3894600.0) == "3.89M"
        assert ss.format_compact(35700.0) == "35.7k"
        assert ss.format_compact(7110.0) == "7,110"
        assert ss.format_compact(0.0116) == "0.0116"

    def test_quantities(self):
        assert ss.format_volume(4284.0) == "4,280 kL"
        assert ss.format_volume(200631.0) == "201 ML"
        assert ss.format_gallons(200631.163) == "53 million gallons"
        assert ss.format_gallons(1670.0) == "441 thousand gallons"
        assert ss.format_mass(46421230.0) == "46,400 t"
        assert ss.format_mass(12.0) == "12 kg"
        assert ss.format_activity(27553180.0) == "27.6 MCi"
        assert ss.format_activity(41919.0) == "41.9 kCi"
        assert ss.format_activity(40.0) == "40 Ci"
        assert ss.format_mass(None) == ss.BLANK

    def test_activity_split_shares_the_unit_of_the_total(self):
        total, split, sentence = ss.format_activity_split([("137Cs", 27553180.0), ("90Sr", 32763170.0)])
        assert total == ss.format_activity(27553180.0 + 32763170.0) == "60.3 MCi"
        assert split == "137Cs 27.6 · 90Sr 32.8"
        assert sentence == "137Cs 27.6 MCi and 90Sr 32.8 MCi"
        assert ss.format_activity_split([("137Cs", 923000.0), ("90Sr", 30400.0)])[:2] == (
            "953 kCi", "137Cs 923 · 90Sr 30.4")
        assert ss.format_activity_split([("137Cs", 300.0), ("90Sr", None)]) == (
            "300 Ci", "137Cs only", "137Cs 300 Ci")
        assert ss.format_activity_split([("137Cs", None), ("90Sr", None)]) == (ss.BLANK, "", "")

    def test_ordinals(self):
        assert [ss.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 102, 113)] == [
            "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "102nd", "113th"]

    def test_scope_sentence(self):
        assert ss.scope_sentence({}, 177, 177) == "All 177 tanks, every waste phase and waste type."
        text = ss.scope_sentence({"TankFarm": ["AN"], "TankSystem": ["DST"]}, 7, 177)
        assert text == "Scope: farm AN; tank system double-shell. 7 of 177 tanks."
