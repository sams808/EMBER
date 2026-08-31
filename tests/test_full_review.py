"""
Whole-application adversarial review: repro tests.

Every test here failed against the code as committed at d02826b. They
divide into four families:

NULL LAYER KEYS. `WasteType` is optional (data_model.REQUIRED_COLUMNS
omits it) and the shipped Hanford.csv writes it as the literal "NA" on
869 rows -- 9 (tank, phase) layers. polars left-joins do not match null
keys by default, so every join keyed on the layer silently missed those
layers: restatement suppression never fired there (uranium came back
2.0004x its reported mass) and the layer denominator never matched
(120 of 177 tanks blanked, under a warning that blamed absent data).

RESTATEMENT OUTSIDE THE ENGINE. Four science modules re-implement the
raw "sum Inventory per group" the engine already fixed, so Tank
Explorer, Heatmaps, Correlations and Vitrification each published
activity that added a combined pair to its own members.

A TOTAL THAT CONTAINS ITS ITEMISATION. TOC is total organic carbon and
already holds the oxalate/formate/acetate/glycolate the BBI reports
beside it -- the third instance of "the BBI describes the same
inventory twice", and the first where the total is a proper SUPERSET of
the itemisation, so the keep-rule inverts.

SILENT TRUNCATION AND MIS-STATED FORMULAS. A global row cap that
removed whole tanks from a per-tank view, NBO/T reported at half the
quantity its label names, and a scoring formula rendered with a
hardcoded sign.
"""
import math

import polars as pl
import pytest

import data_model as dm
import oxide_science as oxsci
import query_engine as qe
import reference_data as rd
from query_engine import run_query

from conftest import REAL_CSV_PATH, requires_real_data


def _null_waste_type_dataset() -> dm.HanfordDataset:
    """One layer whose WasteType is absent, reporting a bulk uranium mass
    AND the isotopes that restate it -- the shape of 241-AX-101."""
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame({
        "WasteSiteId": ["241-AX-101"] * 4,
        "Analyte": ["UTOTAL", "238U", "235U", "234U"],
        "WastePhase": ["Sludge (Liquid & Solid)"] * 4,
        "WasteType": [None] * 4,
        "Inventory": [488.0, 0.1626, 0.00752, 0.0846],
        "Units": ["kg", "Ci", "Ci", "Ci"],
    }))
    dataset.report = None
    return dataset


def _null_waste_type_pair_dataset() -> dm.HanfordDataset:
    """A null-WasteType layer reporting a combined pair AND its members.
    Every layer key on the right-hand side of the activity join is null,
    so polars infers dtype Null for the column and the join raises."""
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame({
        "WasteSiteId": ["241-AX-101"] * 3,
        "Analyte": ["239/240Pu", "239Pu", "240Pu"],
        "WastePhase": ["Sludge (Liquid & Solid)"] * 3,
        "WasteType": [None] * 3,
        "Inventory": [2700.0, 2200.0, 500.0],
        "Units": ["Ci"] * 3,
    }))
    dataset.report = None
    return dataset


class TestNullLayerKeysDoNotDefeatSuppression:
    """A layer that does not name its waste type is still one layer."""

    def test_mass_suppression_fires_on_a_null_waste_type_layer(self):
        result = run_query(_null_waste_type_dataset(), subject=("element", "U"), basis="kg")
        # The isotopes restate UTOTAL, so the answer is the reported mass.
        assert float(result.table["Total_kg"].sum()) == pytest.approx(488.0, rel=1e-9)

    def test_activity_suppression_does_not_crash_on_a_null_waste_type_layer(self):
        result = run_query(_null_waste_type_pair_dataset(), subject=("element", "Pu"), basis="Ci")
        # Members are complete, so they are kept and the pair is set aside.
        assert float(result.table["Total_Ci"].sum()) == pytest.approx(2700.0, rel=1e-9)

    def test_a_polars_schema_error_is_never_raised_at_the_ui_boundary(self):
        """SchemaError is not a ValueError, so the `except ValueError`
        guards in qt_query/qt_heatmap/qt_explorer could not catch it."""
        try:
            run_query(_null_waste_type_pair_dataset(), subject=("element", "Pu"), basis="Ci")
        except Exception as exc:  # noqa: BLE001 - the point is the type
            pytest.fail(f"run_query raised {type(exc).__name__}: {exc}")

    def test_waste_type_column_may_be_absent_entirely(self):
        """WasteType is not in REQUIRED_COLUMNS; a user CSV omitting it
        made every restatement layer null-typed."""
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame({
            "WasteSiteId": ["241-AN-101"] * 3,
            "Analyte": ["239/240Pu", "239Pu", "240Pu"],
            "Inventory": [2700.0, 2200.0, 500.0],
            "Units": ["Ci"] * 3,
        }))
        dataset.report = None
        result = run_query(dataset, subject=("element", "Pu"), basis="Ci")
        assert float(result.table["Total_Ci"].sum()) == pytest.approx(2700.0, rel=1e-9)

    @requires_real_data
    def test_real_uranium_is_not_doubled_on_null_waste_type_tanks(self):
        dataset = dm.HanfordDataset()
        dataset.load_local_default()
        for tank in ("241-AX-101", "241-C-110"):
            scope = {"WasteSiteId": [tank]}
            grams = float(run_query(dataset, subject=("element", "U"), basis="g",
                                    scope=scope).table["Total_g"].sum())
            kilos = float(run_query(dataset, subject=("element", "U"), basis="kg",
                                    scope=scope).table["Total_kg"].sum())
            # The two bases describe one inventory; they may not disagree
            # by the 2.0004x that unsuppressed isotopes produced.
            assert grams / 1000.0 == pytest.approx(kilos, rel=0.01), tank


class TestLayerDenominatorMatchesNullKeyedGroups:
    """A group whose key is null still has a volume and a density."""

    @requires_real_data
    def test_null_keyed_group_gets_its_concentration(self):
        dataset = dm.HanfordDataset()
        dataset.load_local_default()
        result = run_query(dataset, subject=("element", "Na"), basis="wt%",
                           group_by=["TankStatus"])
        table = result.table
        blank = table[table["wt_pct"].isna()]
        assert blank.empty, f"{len(blank)} group(s) blanked despite having volume/density data"

    @requires_real_data
    def test_no_false_missing_denominator_warning(self):
        dataset = dm.HanfordDataset()
        dataset.load_local_default()
        result = run_query(dataset, subject=("element", "Na"), basis="wt%",
                           group_by=["TankStatus"])
        assert not [w for w in result.warnings if "lack layer volume/density" in w]


class TestTotalOrganicCarbonContainsItsItemisation:
    """TOC is the total; oxalate and friends are part of it."""

    def _toc_dataset(self) -> dm.HanfordDataset:
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame({
            "WasteSiteId": ["241-AN-101"] * 3,
            "Analyte": ["TOC", "Oxalate", "Formate"],
            "WastePhase": ["Supernatant"] * 3,
            "WasteType": ["T1"] * 3,
            "Inventory": [1000.0, 880.2, 450.2],
            "Units": ["kg"] * 3,
        }))
        dataset.report = None
        return dataset

    def test_carbon_keeps_the_total_and_sets_aside_the_itemisation(self):
        result = run_query(self._toc_dataset(), subject=("element", "C"), basis="kg")
        # TOC is registered as {"C": 1}, so its 1000 kg IS 1000 kg of carbon,
        # and it already contains the carbon in the oxalate and formate.
        assert float(result.table["Total_kg"].sum()) == pytest.approx(1000.0, rel=1e-9)

    def test_a_layer_without_the_total_keeps_its_itemisation(self):
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame({
            "WasteSiteId": ["241-AN-101"] * 2,
            "Analyte": ["Oxalate", "Formate"],
            "WastePhase": ["Supernatant"] * 2,
            "WasteType": ["T1"] * 2,
            "Inventory": [880.2, 450.2],
            "Units": ["kg"] * 2,
        }))
        dataset.report = None
        result = run_query(dataset, subject=("element", "C"), basis="kg")
        expected = (880.2 * rd.SPECIES_ELEMENT_MASS_FRACTIONS["Oxalate"]["C"]
                    + 450.2 * rd.SPECIES_ELEMENT_MASS_FRACTIONS["Formate"]["C"])
        assert float(result.table["Total_kg"].sum()) == pytest.approx(expected, rel=1e-9)

    def test_phosphorus_in_tributyl_phosphate_survives_carbon_suppression(self):
        """Suppression is scoped to the element the total describes; the
        other elements a member carries must still be booked."""
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame({
            "WasteSiteId": ["241-AN-101"] * 2,
            "Analyte": ["TOC", "Tributyl phosphate"],
            "WastePhase": ["Supernatant"] * 2,
            "WasteType": ["T1"] * 2,
            "Inventory": [1000.0, 266.3],
            "Units": ["kg"] * 2,
        }))
        dataset.report = None
        result = run_query(dataset, subject=("element", "P"), basis="kg")
        expected = 266.3 * rd.SPECIES_ELEMENT_MASS_FRACTIONS["Tributyl phosphate"]["P"]
        assert float(result.table["Total_kg"].sum()) == pytest.approx(expected, rel=1e-9)

    def test_inorganic_carbon_is_not_part_of_the_organic_total(self):
        """TIC as CO3 is carbonate; TOC does not contain it."""
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame({
            "WasteSiteId": ["241-AN-101"] * 2,
            "Analyte": ["TOC", "TIC as CO3"],
            "WastePhase": ["Supernatant"] * 2,
            "WasteType": ["T1"] * 2,
            "Inventory": [1000.0, 600.1],
            "Units": ["kg"] * 2,
        }))
        dataset.report = None
        result = run_query(dataset, subject=("element", "C"), basis="kg")
        expected = 1000.0 + 600.1 * rd.SPECIES_ELEMENT_MASS_FRACTIONS["TIC as CO3"]["C"]
        assert float(result.table["Total_kg"].sum()) == pytest.approx(expected, rel=1e-9)


class TestRestatementSuppressionOutsideTheEngine:
    """Four modules re-summed raw rows the engine already knows to set
    aside. Each published a combined pair on top of its own members."""

    def _pair_dataset(self) -> dm.HanfordDataset:
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame({
            "WasteSiteId": ["241-AN-101"] * 3,
            "TankFarm": ["AN"] * 3,
            "Analyte": ["239/240Pu", "239Pu", "240Pu"],
            "WastePhase": ["Sludge"] * 3,
            "WasteType": ["T1"] * 3,
            "Inventory": [2700.0, 2200.0, 500.0],
            "Units": ["Ci"] * 3,
        }))
        dataset.report = None
        return dataset

    def test_tank_profile_sets_aside_the_pair(self):
        import tank_science as tsci
        out = tsci.tank_profile(self._pair_dataset(), ["241-AN-101"], "Ci", top_n=10)
        assert float(out["Inventory_sum"].sum()) == pytest.approx(2700.0, rel=1e-9)

    def test_element_inventory_matrix_sets_aside_the_pair(self):
        import matrix_science as msci
        out = msci.element_inventory_matrix(self._pair_dataset(), "Ci",
                                            value_mode="inventory")
        numeric = out.select_dtypes("number")
        assert float(numeric.to_numpy().sum()) == pytest.approx(2700.0, rel=1e-9)

    def test_element_totals_by_unit_sets_aside_the_pair(self):
        import correlation_science as csci
        out = csci.element_totals_by_unit(self._pair_dataset(), "Ci")
        assert float(sum(out.values())) == pytest.approx(2700.0, rel=1e-9)

    def test_tank_category_summary_sets_aside_the_pair(self):
        import vitrification_science as vsci
        out = vsci.tank_category_summary(self._pair_dataset())
        assert float(out["Total_Ci_inventory"].sum()) == pytest.approx(2700.0, rel=1e-9)

    @requires_real_data
    def test_tank_explorer_agrees_with_the_engine_on_real_data(self):
        import tank_science as tsci
        dataset = dm.HanfordDataset()
        dataset.load_local_default()
        tank = "241-AN-101"
        profile = tsci.tank_profile(dataset, [tank], "Ci", top_n=200)
        pu = profile[profile["Element"] == "Pu"]["Inventory_sum"].sum()
        engine = float(run_query(dataset, subject=("element", "Pu"), basis="Ci",
                                 scope={"WasteSiteId": [tank]}).table["Total_Ci"].sum())
        assert float(pu) == pytest.approx(engine, rel=1e-6)


class TestPerTankTopNeverRemovesATank:
    """The Top-N control says "how many elements to show"; it removed
    whole tanks from the table, the chart and the export."""

    def _many_tank_dataset(self) -> dm.HanfordDataset:
        rows = []
        for tank in ("241-A-101", "241-B-101", "241-C-101"):
            for i, element in enumerate(("Na", "Fe", "Al", "Cr", "Ni")):
                rows.append((tank, element, 100.0 - i))
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame({
            "WasteSiteId": [r[0] for r in rows],
            "TankFarm": [r[0][4] for r in rows],
            "Analyte": [r[1] for r in rows],
            "WastePhase": ["Sludge"] * len(rows),
            "WasteType": ["T1"] * len(rows),
            "Inventory": [r[2] for r in rows],
            "Units": ["kg"] * len(rows),
        }))
        dataset.report = None
        return dataset

    def test_every_selected_tank_appears(self):
        import tank_science as tsci
        tanks = ["241-A-101", "241-B-101", "241-C-101"]
        out = tsci.tank_profile(self._many_tank_dataset(), tanks, "kg", top_n=2)
        assert sorted(out["WasteSiteId"].unique()) == tanks

    def test_top_n_is_applied_per_tank(self):
        import tank_science as tsci
        tanks = ["241-A-101", "241-B-101", "241-C-101"]
        out = tsci.tank_profile(self._many_tank_dataset(), tanks, "kg", top_n=2)
        counts = out.groupby("WasteSiteId").size()
        assert set(counts.unique()) == {2}

    @requires_real_data
    def test_real_farm_selection_keeps_all_its_tanks(self):
        import tank_science as tsci
        dataset = dm.HanfordDataset()
        dataset.load_local_default()
        tanks = sorted(tsci.tanks_in_farm(dataset, "C"))
        out = tsci.tank_profile(dataset, tanks, "kg", top_n=35)
        present = set(out["WasteSiteId"].unique())
        expected = {
            t for t in tanks
            if dataset.require_df().filter(
                (pl.col("WasteSiteId") == t) & (pl.col("Units") == "kg")
                & (pl.col("Inventory") > 0) & pl.col("Element").is_not_null()
            ).height
        }
        assert expected - present == set(), f"tanks dropped: {sorted(expected - present)}"


class TestOxideChemistryQuantities:

    def test_nbo_t_matches_the_classical_definition(self):
        """NBO/T = (2*O - 4*T)/T. For Na2O.SiO2 -- one bridging and two
        non-bridging oxygens per Si -- the textbook answer is 2."""
        out = oxsci.nbo_over_t({"Na2O": 1.0, "SiO2": 1.0})
        assert out["NBO_T"] == pytest.approx(2.0, rel=1e-9)

    def test_nbo_t_is_zero_for_pure_silica(self):
        out = oxsci.nbo_over_t({"SiO2": 1.0})
        assert out["NBO_T"] == pytest.approx(0.0, abs=1e-12)

    def test_components_without_a_basicity_value_are_disclosed(self):
        """The exclusion set was derived from the already-oxide-filtered
        map, so a non-oxide component could never be reported."""
        table = pl.DataFrame({
            "Component": ["SiO2", "Na2O", "Cl"],
            "Kind": ["oxide", "oxide", "element"],
            "Mol": [1.0, 1.0, 0.5],
            "Mass_kg": [60.0, 62.0, 17.7],
        }).to_pandas()
        summary = oxsci.composition_summary(table)
        assert "Cl" in summary["excluded_from_basicity"]


class TestScreeningFormulaReadout:
    """The readout is the only human-readable statement of the score."""

    def test_a_positive_weight_is_not_rendered_as_a_penalty(self):
        pytest.importorskip("PySide6")
        from qt_vitrification_screening import _format_score_formula
        negative = _format_score_formula({
            "glass_former_weight": 40.0, "modifier_weight": 20.0,
            "problem_weight": -45.0, "volatile_weight": -25.0, "redox_weight": -10.0,
        })
        positive = _format_score_formula({
            "glass_former_weight": 40.0, "modifier_weight": 20.0,
            "problem_weight": 45.0, "volatile_weight": -25.0, "redox_weight": -10.0,
        })
        assert negative != positive


class TestReferenceConstantsArePinnedToLiterature:
    """Every existing constant test read its expected value out of the
    table under test, so no mutation of the table could fail the suite."""

    def test_species_molar_masses_match_literature(self):
        literature = {
            "SO4": 96.06, "PO4": 94.97, "NO3": 62.00, "NO2": 46.01,
            "NH3": 17.03, "CN": 26.02, "Free OH": 17.01, "TIC as CO3": 60.01,
            "Oxalate": 88.02, "Formate": 45.02, "Acetate": 59.04,
        }
        for species, expected in literature.items():
            counts = rd.SPECIES_FORMULAS[species]
            mass = sum(rd.ATOMIC_MASSES[el] * n for el, n in counts.items())
            assert mass == pytest.approx(expected, rel=2e-3), species

    def test_specific_activities_match_literature(self):
        # ICRP-107 / NIST, Ci per gram.
        literature = {"137Cs": 87.0, "90Sr": 138.0, "239Pu": 0.0621, "241Am": 3.43}
        for nuclide, expected in literature.items():
            grams = rd.nuclide_grams_per_ci(nuclide)
            assert grams is not None, nuclide
            assert 1.0 / grams == pytest.approx(expected, rel=0.05), nuclide

    def test_curie_is_defined_in_becquerel(self):
        assert rd.CI_TO_BQ == 3.7e10


class TestAnalyteMassDisclosureNamesTheBridge:

    @requires_real_data
    def test_an_analyte_subject_on_a_mass_basis_names_icrp(self):
        dataset = dm.HanfordDataset()
        dataset.load_local_default()
        result = run_query(dataset, subject=("analyte", "137Cs"), basis="g")
        notes = " ".join(result.disclosures) + " ".join(result.warnings)
        assert "ICRP" in notes


class TestOxideCompositionConservesMass:
    """Converting a cation to its oxide may not manufacture mass."""

    @requires_real_data
    def test_sulfate_does_not_grow_into_more_sulfur_trioxide(self):
        dataset = dm.HanfordDataset()
        dataset.load_local_default()
        tank = "241-AW-106"
        table = oxsci.tank_oxide_composition(dataset, [tank])[tank]
        so3_kg = float(table.loc[table["Component"] == "SO3", "Mass_g"].sum()) / 1000.0
        so4_kg = float(
            dataset.require_df().filter(
                (pl.col("WasteSiteId") == tank) & (pl.col("Analyte") == "SO4")
                & (pl.col("Units") == "kg")
            )["Inventory"].sum()
        )
        # SO4 -> SO3 loses one oxygen: the ratio is exactly 80.06/96.06.
        assert so3_kg / so4_kg == pytest.approx(80.06 / 96.06, rel=2e-3)

    @requires_real_data
    def test_the_table_never_outweighs_the_inventory_it_came_from(self):
        dataset = dm.HanfordDataset()
        dataset.load_local_default()
        tank = "241-AW-106"
        table = oxsci.tank_oxide_composition(dataset, [tank])[tank]
        oxide_kg = float(table["Mass_g"].sum()) / 1000.0
        reported_kg = float(
            dataset.require_df().filter(
                (pl.col("WasteSiteId") == tank) & (pl.col("Units") == "kg")
            )["Inventory"].sum()
        )
        assert oxide_kg <= reported_kg

    def test_oxygen_and_hydrogen_are_not_double_booked(self):
        """The oxide formulas carry their own oxygen, so the apportioned
        free-oxygen pool must not also appear as a component."""
        assert "O" in oxsci.IMPLIED_BY_OXIDE_BASIS
        assert "H" in oxsci.IMPLIED_BY_OXIDE_BASIS

    def test_the_basis_is_disclosed(self):
        note = oxsci.oxide_basis_note()
        assert "apportioned" in note.lower()
        assert "oxygen" in note.lower()


class TestOxideMapEditReachesTheComputation:
    """The tab's one editable assumption has to be able to change a number."""

    def test_repopulating_keeps_the_users_choice(self, qtbot):
        pytest.importorskip("PySide6")
        from qt_vitrification_oxide import OxideChemistryTab
        tab = OxideChemistryTab(app_window=None)
        qtbot.addWidget(tab)
        tab._populate_oxide_map_table(["Fe", "Na"])
        tab.oxide_map_table.setItem(1, 1, __import__("PySide6.QtWidgets", fromlist=["QTableWidgetItem"])
                                   .QTableWidgetItem("Na2O2"))
        # A second build must not reseed the default over the edit.
        tab._populate_oxide_map_table(["Fe", "Na"])
        assert tab._current_oxide_map()["Na"] == "Na2O2"

    def test_an_edited_map_changes_the_converted_composition(self):
        before = oxsci.convert_composition_to_oxides({"Na": 100.0})
        after = oxsci.convert_composition_to_oxides({"Na": 100.0}, {"Na": "Na2O2"})
        assert set(before["Component"]) != set(after["Component"])
