import math

import pytest
import xraydb

import oxide_science as ox


class TestDefaultMaps:
    def test_every_default_formula_parses(self):
        for element, formula in ox.DEFAULT_OXIDE_MAP.items():
            if formula is None:
                continue
            counts = xraydb.chemparse(formula)
            assert counts.get(element, 0) > 0, f"{formula} (default for {element}) does not contain {element}"

    def test_alternatives_first_entry_matches_default(self):
        for element, alts in ox.ALTERNATIVE_OXIDES.items():
            assert alts[0] == ox.DEFAULT_OXIDE_MAP[element]

    def test_alternatives_all_contain_their_element(self):
        for element, alts in ox.ALTERNATIVE_OXIDES.items():
            for formula in alts:
                counts = xraydb.chemparse(formula)
                assert counts.get(element, 0) > 0

    def test_halides_and_noble_metals_default_to_none(self):
        for element in ["F", "Cl", "Br", "I", "Ru", "Rh", "Pd", "Ag", "Pt", "Au", "Tc"]:
            assert ox.DEFAULT_OXIDE_MAP[element] is None


class TestDefaultOxideRole:
    def test_formers(self):
        for formula in ["SiO2", "B2O3", "P2O5", "Al2O3"]:
            assert ox.default_oxide_role(formula) == "former"

    def test_modifiers(self):
        for formula in ["Na2O", "K2O", "Li2O", "Cs2O", "CaO", "MgO", "SrO", "BaO"]:
            assert ox.default_oxide_role(formula) == "modifier"

    def test_other(self):
        assert ox.default_oxide_role("Fe2O3") == "other"
        assert ox.default_oxide_role("ZrO2") == "other"


class TestOxideMassFromElementMass:
    def test_matches_hand_stoichiometry(self):
        # 1000 g Na -> Na2O: moles_Na = 1000/22.9898; moles_Na2O = moles_Na/2;
        # mass = moles_Na2O * (2*22.9898 + 15.9994).
        na_mw = xraydb.atomic_mass("Na")
        na2o_mw = 2 * xraydb.atomic_mass("Na") + xraydb.atomic_mass("O")
        expected = (1000.0 / na_mw / 2.0) * na2o_mw
        assert ox.oxide_mass_from_element_mass("Na", 1000.0, "Na2O") == pytest.approx(expected)

    def test_element_not_in_formula_raises(self):
        with pytest.raises(ValueError, match="does not contain"):
            ox.oxide_mass_from_element_mass("Fe", 100.0, "Na2O")


class TestConvertCompositionToOxides:
    def test_wt_and_mol_pct_sum_to_100(self):
        out = ox.convert_composition_to_oxides({"Na": 10.0, "Si": 5.0})
        assert out["Wt_pct"].sum() == pytest.approx(100.0)
        assert out["Mol_pct"].sum() == pytest.approx(100.0)
        assert set(out["Component"]) == {"Na2O", "SiO2"}
        assert (out["Kind"] == "oxide").all()

    def test_unmapped_halide_reported_as_elemental(self):
        out = ox.convert_composition_to_oxides({"Na": 10.0, "Cl": 2.0}).set_index("Component")
        assert out.loc["Cl", "Kind"] == "element"
        assert out.loc["Na2O", "Kind"] == "oxide"

    def test_custom_oxide_map_overrides_default(self):
        out = ox.convert_composition_to_oxides({"Fe": 10.0}, oxide_map={"Fe": "FeO"})
        assert out.iloc[0]["Component"] == "FeO"

    def test_custom_oxide_map_can_force_elemental(self):
        out = ox.convert_composition_to_oxides({"Na": 10.0}, oxide_map={"Na": None})
        assert out.iloc[0]["Kind"] == "element"
        assert out.iloc[0]["Component"] == "Na"

    def test_zero_and_negative_and_missing_inventory_skipped(self):
        out = ox.convert_composition_to_oxides({"Na": 10.0, "Si": 0.0, "Fe": -5.0, "Al": None})
        assert set(out["Component"]) == {"Na2O"}

    def test_empty_input_returns_empty_with_columns(self):
        out = ox.convert_composition_to_oxides({})
        assert out.empty
        assert list(out.columns) == ["Component", "Kind", "Source_elements", "Mass_g", "Wt_pct", "Mol", "Mol_pct"]

    def test_source_elements_merge_on_collision(self):
        # A hypothetical mixed oxide (spinel-like) containing both Fe and
        # Cr -- mapping both elements to the same formula must merge into
        # one row (summed mass/mol) rather than overwrite, with both
        # source elements recorded.
        out = ox.convert_composition_to_oxides(
            {"Fe": 10.0, "Cr": 5.0}, oxide_map={"Fe": "FeCr2O4", "Cr": "FeCr2O4"},
        )
        assert len(out) == 1
        row = out.iloc[0]
        assert row["Component"] == "FeCr2O4"
        assert set(row["Source_elements"].split(";")) == {"Cr", "Fe"}

    def test_element_not_in_default_map_at_all_treated_as_elemental(self):
        # A real element absent from DEFAULT_OXIDE_MAP entirely (not just
        # mapped to None) must still be reported, not silently dropped.
        assert "He" not in ox.DEFAULT_OXIDE_MAP
        out = ox.convert_composition_to_oxides({"He": 10.0})
        assert out.iloc[0]["Kind"] == "element"
        assert out.iloc[0]["Component"] == "He"


class TestTankOxideComposition:
    def test_per_tank_tables_built(self, oxide_dataset):
        tables = ox.tank_oxide_composition(oxide_dataset, ["241-A-101", "241-A-103"])
        assert set(tables["241-A-101"]["Component"]) == {"Na2O", "SiO2"}
        assert set(tables["241-A-103"]["Component"]) == {"Na2O", "SiO2"}

    def test_blend_sums_raw_kg_before_converting_not_average_of_percents(self, oxide_dataset):
        tables = ox.tank_oxide_composition(oxide_dataset, ["241-A-101", "241-A-103"])
        blend = tables["Blend (selected tanks)"].set_index("Component")
        # T1: Na=100,Si=50 (Na2O:SiO2 mass ratio distinct); T3: Na=60,Si=30.
        # Combined elemental Na=160, Si=80 -- convert once from that sum.
        expected = ox.convert_composition_to_oxides({"Na": 160.0, "Si": 80.0})
        expected_na2o_wt = expected.set_index("Component").loc["Na2O", "Wt_pct"]
        assert blend.loc["Na2O", "Wt_pct"] == pytest.approx(expected_na2o_wt)

    def test_single_tank_has_no_blend_table(self, oxide_dataset):
        tables = ox.tank_oxide_composition(oxide_dataset, ["241-A-101"])
        assert "Blend (selected tanks)" not in tables

    def test_no_tank_ids_returns_empty(self, oxide_dataset):
        assert ox.tank_oxide_composition(oxide_dataset, []) == {}

    def test_tank_with_no_kg_rows_skipped_not_crashed(self, oxide_dataset):
        tables = ox.tank_oxide_composition(oxide_dataset, ["241-A-101", "not-a-real-tank"])
        assert "not-a-real-tank" not in tables
        assert "241-A-101" in tables

    def test_no_matching_tanks_returns_empty(self, oxide_dataset):
        assert ox.tank_oxide_composition(oxide_dataset, ["nope-1", "nope-2"]) == {}


class TestNboOverT:
    def test_pure_silica_is_zero(self):
        # T=1 (1 Si per SiO2), O_total=2 (2 O per SiO2) -> (2-2*1)/1 = 0.
        out = ox.nbo_over_t({"SiO2": 1.0})
        assert out["T"] == pytest.approx(1.0)
        assert out["O_total"] == pytest.approx(2.0)
        assert out["NBO_T"] == pytest.approx(0.0)

    def test_soda_silicate_hand_computed(self):
        # 1 mol SiO2 + 1 mol Na2O: T = 1*1 (Si only, Na2O is a modifier);
        # O_total = 1*2 (SiO2) + 1*1 (Na2O) = 3; NBO/T = (3-2)/1 = 1.
        out = ox.nbo_over_t({"SiO2": 1.0, "Na2O": 1.0})
        assert out["T"] == pytest.approx(1.0)
        assert out["O_total"] == pytest.approx(3.0)
        assert out["NBO_T"] == pytest.approx(1.0)

    def test_other_role_oxide_excluded_even_if_large(self):
        base = ox.nbo_over_t({"SiO2": 1.0, "Na2O": 1.0})
        with_iron = ox.nbo_over_t({"SiO2": 1.0, "Na2O": 1.0, "Fe2O3": 50.0})
        assert with_iron["NBO_T"] == pytest.approx(base["NBO_T"])

    def test_no_formers_returns_nan(self):
        out = ox.nbo_over_t({"Na2O": 1.0})
        assert out["T"] == 0.0
        assert math.isnan(out["NBO_T"])

    def test_reclassified_former_uses_generalized_t_cation_count(self):
        # GeO2 is normally "other"; reclassifying it as a former should
        # count its 1 non-oxygen atom (Ge) as a T-cation via the fallback.
        out = ox.nbo_over_t({"GeO2": 1.0}, role_map={"GeO2": "former"})
        assert out["T"] == pytest.approx(1.0)
        assert out["O_total"] == pytest.approx(2.0)
        assert out["NBO_T"] == pytest.approx(0.0)

    def test_empty_role_map_uses_defaults(self):
        out = ox.nbo_over_t({"SiO2": 1.0}, role_map=None)
        assert out["NBO_T"] == pytest.approx(0.0)


class TestEnvelopeCheck:
    def test_pass_and_fail(self):
        out = ox.envelope_check(
            {"SiO2": 50.0, "Na2O": 20.0}, {"SiO2": (40.0, 60.0), "Na2O": (0.0, 10.0)},
        ).set_index("Oxide")
        assert out.loc["SiO2", "Status"] == "Pass"
        assert out.loc["Na2O", "Status"] == "Fail"

    def test_missing_from_composition_checked_as_zero(self):
        out = ox.envelope_check({}, {"B2O3": (5.0, 15.0)}).set_index("Oxide")
        assert out.loc["B2O3", "Wt_pct"] == pytest.approx(0.0)
        assert out.loc["B2O3", "Status"] == "Fail"

    def test_present_but_not_in_envelope_reported_not_specified(self):
        out = ox.envelope_check({"ZrO2": 3.0}, {"SiO2": (40.0, 60.0)}).set_index("Oxide")
        assert out.loc["ZrO2", "Status"] == "Not specified"

    def test_empty_envelope_returns_empty_with_columns(self):
        out = ox.envelope_check({"SiO2": 50.0}, {})
        assert out.empty
        assert list(out.columns) == ["Oxide", "Wt_pct", "Min_wt_pct", "Max_wt_pct", "Status"]


class TestEnvelopeJsonRoundTrip:
    def test_save_and_load(self, tmp_path):
        envelope = {"SiO2": (40.0, 60.0), "Na2O": (0.0, 20.0)}
        path = tmp_path / "envelope.json"
        ox.save_envelope_json(path, envelope)
        loaded = ox.load_envelope_json(path)
        assert loaded == envelope


class TestGlassnetInputRow:
    def test_oxide_only_columns(self):
        table = ox.convert_composition_to_oxides({"Na": 10.0, "Cl": 2.0})
        row = ox.glassnet_input_row(table)
        assert list(row.columns) == ["Na2O"]

    def test_empty_table_returns_empty(self):
        import pandas as pd
        assert ox.glassnet_input_row(pd.DataFrame()).empty

    def test_no_oxide_rows_returns_empty(self):
        table = ox.convert_composition_to_oxides({"Cl": 2.0})
        assert ox.glassnet_input_row(table).empty


class TestCompositionSummary:
    def test_basicity_nbo_t_computed(self):
        table = ox.convert_composition_to_oxides({"Na": 10.0, "Si": 20.0})
        summary = ox.composition_summary(table)
        assert summary["optical_basicity"] > 0
        assert not math.isnan(summary["NBO_T"])
        assert summary["excluded_from_basicity"] == []

    def test_pu_excluded_from_basicity_not_crashed(self):
        table = ox.convert_composition_to_oxides({"Na": 10.0, "Pu": 0.5})
        summary = ox.composition_summary(table)
        assert "PuO2" in summary["excluded_from_basicity"]
        assert summary["optical_basicity"] > 0  # Na2O alone still contributes

    def test_all_excluded_from_basicity_gives_nan(self):
        table = ox.convert_composition_to_oxides({"Pu": 0.5})
        summary = ox.composition_summary(table)
        assert math.isnan(summary["optical_basicity"])

    def test_envelope_wired_through(self):
        table = ox.convert_composition_to_oxides({"Na": 10.0, "Si": 20.0})
        summary = ox.composition_summary(table, envelope={"SiO2": (0.0, 100.0)})
        assert not summary["envelope_table"].empty

    def test_no_envelope_gives_empty_table(self):
        table = ox.convert_composition_to_oxides({"Na": 10.0})
        summary = ox.composition_summary(table)
        assert summary["envelope_table"].empty

    def test_zero_mol_known_oxide_falls_back_to_nan_not_crash(self):
        # A hand-built table (bypassing convert_composition_to_oxides,
        # which never emits a zero-Mol row) with a known oxide at exactly
        # Mol=0 -- glass_science.optical_basicity's "zero oxygen content"
        # ValueError must be caught, not propagate.
        import pandas as pd
        table = pd.DataFrame([{
            "Component": "SiO2", "Kind": "oxide", "Source_elements": "Si",
            "Mass_g": 0.0, "Wt_pct": 0.0, "Mol": 0.0, "Mol_pct": 0.0,
        }])
        summary = ox.composition_summary(table)
        assert math.isnan(summary["optical_basicity"])

    def test_empty_oxide_table_returns_safe_defaults(self):
        import pandas as pd
        summary = ox.composition_summary(pd.DataFrame())
        assert math.isnan(summary["optical_basicity"])
        assert math.isnan(summary["NBO_T"])
        assert summary["envelope_table"].empty
