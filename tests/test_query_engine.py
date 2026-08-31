"""
L1 query engine tests. Expected numbers are hand-derived from
reference_data's own constants inside each test (fraction * inventory),
never re-derived through the engine -- plus a real-CSV acceptance test of
the flagship salt-formers basket at the bottom.
"""
import polars as pl
import pytest

import data_model as dm
import reference_data as rd
from query_engine import (
    BASES,
    BASKET_PRESETS,
    DIMENSIONS,
    Basket,
    run_query,
)

from conftest import REAL_CSV_PATH, requires_real_data


def _dataset(rows: dict) -> dm.HanfordDataset:
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame(rows))
    dataset.report = None
    return dataset


def _chem_dataset() -> dm.HanfordDataset:
    """One tank, two layers; species + elemental + nuclide rows with known
    volumes/densities so every basis is hand-checkable.

    Layer A (Supernatant, T1): 2 kL x 1.5 g/mL -> 3.0e6 g
    Layer B (Sludge, T2):      1 kL x 1.0 g/mL -> 1.0e6 g
    """
    return _dataset({
        "WasteSiteId": ["241-A-101"] * 5 + ["241-A-101"],
        "Analyte": ["SO4", "Sulfide", "Cl", "129I", "Cr", "Fe"],
        "WastePhase": ["Supernatant"] * 5 + ["Sludge"],
        "WasteType": ["T1"] * 5 + ["T2"],
        "Inventory": [100.0, 5.0, 3.0, 2.0, 4.0, 7.0],
        "Units": ["kg", "kg", "kg", "Ci", "kg", "kg"],
        "Volume": [2.0] * 5 + [1.0],
        "ComponentDensity": [1.5] * 5 + [1.0],
    })


class TestApportionment:
    def test_element_gets_mass_fraction_of_species_not_full_mass(self):
        # THE legacy correctness fix: 100 kg of SO4 contributes ~33.4 kg of
        # S (32.06/96.06), never 100 kg.
        res = run_query(_chem_dataset(), subject=("element", "S"), basis="kg")
        frac = rd.SPECIES_ELEMENT_MASS_FRACTIONS["SO4"]["S"]
        expected = 100.0 * frac + 5.0  # SO4-apportioned + Sulfide (pure S basis)
        assert res.table["Total_kg"].iloc[0] == pytest.approx(expected)

    def test_analyte_subject_keeps_full_species_mass(self):
        res = run_query(_chem_dataset(), subject=("analyte", "SO4"), basis="kg")
        assert res.table["Total_kg"].iloc[0] == pytest.approx(100.0)

    def test_oxygen_gets_its_share_of_every_species(self):
        res = run_query(_chem_dataset(), subject=("element", "O"), basis="kg")
        expected = 100.0 * rd.SPECIES_ELEMENT_MASS_FRACTIONS["SO4"]["O"]
        assert res.table["Total_kg"].iloc[0] == pytest.approx(expected)

    def test_unregistered_species_is_excluded_with_a_warning(self):
        dataset = _dataset({
            "WasteSiteId": ["241-A-101", "241-A-101"],
            "Analyte": ["Cr2O7", "Cr"],
            "WastePhase": ["Supernatant"] * 2,
            "WasteType": ["T1"] * 2,
            "Inventory": [10.0, 4.0],
            "Units": ["kg", "kg"],
        })
        res = run_query(dataset, subject=("element", "Cr"), basis="kg")
        assert res.table["Total_kg"].iloc[0] == pytest.approx(4.0)  # only the elemental row
        assert any("Cr2O7" in w and "unregistered" in w for w in res.warnings)


class TestCiBridge:
    def test_ci_rows_bridge_to_grams_via_specific_activity(self, sample_dataset):
        res = run_query(sample_dataset, subject=("element", "Cs"), basis="g")
        expected = 300.0 * rd.nuclide_grams_per_ci("137Cs")  # 100 + 200 Ci
        assert res.table["Total_g"].iloc[0] == pytest.approx(expected)

    def test_combined_pair_is_excluded_from_mass_and_disclosed(self, sample_dataset):
        res = run_query(sample_dataset, subject=("element", "Pu"), basis="g")
        assert res.table.empty or res.table["Total_g"].fillna(0).eq(0).all()
        assert any("239/240Pu" in w for w in res.warnings)

    def test_combined_pair_still_counts_under_activity_basis(self, sample_dataset):
        res = run_query(sample_dataset, subject=("element", "Pu"), basis="Ci")
        assert res.table["Total_Ci"].iloc[0] == pytest.approx(0.002)

    def test_kg_reported_nuclide_contributes_mass(self, sample_dataset):
        # 113mCd is a kg row in the fixture: an isotope label reported in
        # kg must contribute its full mass to the element, not vanish
        # between the elemental and Ci paths.
        res = run_query(sample_dataset, subject=("element", "Cd"), basis="kg")
        assert res.table["Total_kg"].iloc[0] == pytest.approx(0.5)

    def test_kg_basis_excludes_ci_rows_with_warning(self, sample_dataset):
        res = run_query(sample_dataset, subject=("element", "Cs"), basis="kg")
        assert res.table.empty or res.table["Total_kg"].fillna(0).eq(0).all()
        assert any("excluded under the kg basis" in w for w in res.warnings)


class TestGroupingAndScope:
    def test_group_by_phase(self, sample_dataset):
        res = run_query(sample_dataset, subject=("element", "Fe"), basis="kg", group_by=["WastePhase"])
        by_phase = res.table.set_index("WastePhase")["Total_kg"]
        assert by_phase["Solid"] == pytest.approx(50.0)
        assert by_phase["Liquid"] == pytest.approx(10.0)

    def test_scope_filters_before_aggregation(self, sample_dataset):
        res = run_query(
            sample_dataset, subject=("element", "Fe"), basis="kg",
            scope={"WastePhase": ["Solid"]},
        )
        assert res.table["Total_kg"].iloc[0] == pytest.approx(50.0)

    def test_unknown_group_dims_are_refused(self, sample_dataset):
        # Review finding: a typo'd dimension silently returned an UNGROUPED
        # dataset-wide total that the caller would read as grouped. The
        # engine refuses loudly everywhere else -- this must match.
        with pytest.raises(ValueError, match="[Uu]nknown group"):
            run_query(sample_dataset, subject=("element", "Fe"), basis="kg", group_by=["WasteSiteID"])

    def test_two_dim_grouping(self, sample_dataset):
        res = run_query(
            sample_dataset, subject=("element", "Cs"), basis="Ci",
            group_by=["WasteSiteId", "WastePhase"],
        )
        assert len(res.table) == 2
        assert set(res.table["WasteSiteId"]) == {"241-A-101", "241-AN-104"}

    def test_analyte_dimension_shows_species_provenance(self):
        # Stage-2 dimension: grouping an ELEMENT subject by Analyte answers
        # "which species carry it" -- each species row keeps its apportioned
        # share, so the split sums to the ungrouped total.
        res = run_query(_chem_dataset(), subject=("element", "S"), basis="kg", group_by=["Analyte"])
        by_species = res.table.set_index("Analyte")["Total_kg"]
        frac = rd.SPECIES_ELEMENT_MASS_FRACTIONS["SO4"]["S"]
        assert by_species["SO4"] == pytest.approx(100.0 * frac)
        assert by_species["Sulfide"] == pytest.approx(5.0)

    def test_tank_type_dimension_groups_when_attributes_present(self, sample_dataset):
        res = run_query(sample_dataset, subject=("element", "Fe"), basis="kg", group_by=["TankType"])
        assert res.table["Total_kg"].sum() == pytest.approx(60.0)
        assert "TankType" in res.table.columns


class TestConcentrationBases:
    def test_wt_pct_uses_distinct_layer_masses_across_all_layers_of_the_group(self):
        # S mass sits in layer A only, but the tank's wt% denominator must
        # cover BOTH layers (3.0e6 + 1.0e6 g) -- concentration is relative
        # to everything in the group, not just layers containing S.
        res = run_query(_chem_dataset(), subject=("element", "S"), basis="wt%", group_by=["WasteSiteId"])
        frac = rd.SPECIES_ELEMENT_MASS_FRACTIONS["SO4"]["S"]
        mass_g = (100.0 * frac + 5.0) * 1000.0
        assert res.table["wt_pct"].iloc[0] == pytest.approx(mass_g / 4.0e6 * 100.0)

    def test_wt_pct_per_phase_uses_that_phase_layer_only(self):
        res = run_query(_chem_dataset(), subject=("element", "Fe"), basis="wt%", group_by=["WastePhase"])
        row = res.table.set_index("WastePhase")
        assert row.loc["Sludge", "wt_pct"] == pytest.approx(7000.0 / 1.0e6 * 100.0)

    def test_g_per_l_uses_layer_volumes(self):
        res = run_query(_chem_dataset(), subject=("analyte", "SO4"), basis="g/L", group_by=["WasteSiteId"])
        assert res.table["g_per_L"].iloc[0] == pytest.approx(100000.0 / 3000.0)  # 3 kL total

    def test_concentration_refuses_non_layer_grouping(self):
        # Isotope is a per-row dimension, so a layer denominator would be
        # double-counted across it. (Element has its own, earlier refusal:
        # it labels a subject's apportioned mass with the source row's
        # element -- see test_recheck_findings.)
        with pytest.raises(ValueError, match="layer-compatible"):
            run_query(_chem_dataset(), subject=("element", "S"), basis="wt%",
                      group_by=["Isotope"])


class TestWarnings:
    def test_mixed_decay_cohorts_warn(self):
        dataset = _dataset({
            "WasteSiteId": ["241-A-101", "241-AN-104"],
            "Analyte": ["137Cs", "137Cs"],
            "WastePhase": ["Supernatant"] * 2,
            "WasteType": ["T1"] * 2,
            "Inventory": [10.0, 20.0],
            "Units": ["Ci", "Ci"],
            "DecayDate": ["2015/07/01 12:00:00 AM", "2022/01/01 12:00:00 AM"],
        })
        res = run_query(dataset, subject=("element", "Cs"), basis="Ci")
        assert any("decay baselines" in w for w in res.warnings)

    def test_single_cohort_does_not_warn(self, sample_dataset):
        res = run_query(sample_dataset, subject=("element", "Cs"), basis="Ci")
        assert not any("decay baselines" in w for w in res.warnings)

    def test_rsd_zero_rows_flagged_as_unassigned_uncertainty(self):
        dataset = _dataset({
            "WasteSiteId": ["241-A-101"],
            "Analyte": ["Fe"],
            "WastePhase": ["Supernatant"],
            "WasteType": ["T1"],
            "Inventory": [10.0],
            "Units": ["kg"],
            "Inventory_RSD": [0.0],
        })
        res = run_query(dataset, subject=("element", "Fe"), basis="kg")
        assert any("no assigned uncertainty" in w for w in res.warnings)


class TestBaskets:
    def test_basket_sums_members_and_reports_each(self):
        res = run_query(_chem_dataset(), subject=("basket", BASKET_PRESETS["Salt formers"]), basis="g")
        frac = rd.SPECIES_ELEMENT_MASS_FRACTIONS["SO4"]["S"]
        expected_g = (
            (100.0 * frac + 5.0) * 1000.0     # S: apportioned SO4 + sulfide
            + 3.0 * 1000.0                     # Cl elemental
            + 4.0 * 1000.0                     # Cr elemental
            + 2.0 * rd.nuclide_grams_per_ci("129I")  # I: bridged
        )
        assert res.table["Total_g"].iloc[0] == pytest.approx(expected_g)
        members = set(res.member_table["Member"])
        assert {"S", "Cl", "Cr", "I"} <= members

    def test_member_disclosures_state_what_each_resolved_to(self):
        res = run_query(_chem_dataset(), subject=("basket", BASKET_PRESETS["Salt formers"]), basis="g")
        text = "\n".join(res.disclosures)
        assert "apportioned from SO4" in text
        assert "129I" in text and "lower bound" in text  # iodine: bridged Ci only
        assert "Br: not reported" in text                # bromine absent from this fixture

    def test_analyte_member_resolves_as_analyte(self):
        basket = Basket("test", ("NO3",))
        dataset = _dataset({
            "WasteSiteId": ["241-A-101"],
            "Analyte": ["NO3"],
            "WastePhase": ["Supernatant"],
            "WasteType": ["T1"],
            "Inventory": [50.0],
            "Units": ["kg"],
        })
        res = run_query(dataset, subject=("basket", basket), basis="kg")
        assert res.table["Total_kg"].iloc[0] == pytest.approx(50.0)  # full NO3 mass, not N-apportioned


class TestValidation:
    def test_unknown_basis_raises(self, sample_dataset):
        with pytest.raises(ValueError, match="basis"):
            run_query(sample_dataset, subject=("element", "Fe"), basis="mol")

    def test_dimensions_registry_covers_the_group_by_surface(self):
        assert "WastePhase" in DIMENSIONS and "Isotope" in DIMENSIONS
        assert len(BASES) == 5


class TestReviewFindings:
    """Repro tests for the Stage-1 code-review findings, written to fail
    against the pre-fix engine. Each documents the finding it pins."""

    def test_attribute_dim_grouping_without_attributes_refuses_not_crashes(self):
        # Finding: grouping by TankSystem with no Tank_attributes.csv loaded
        # escaped as polars ColumnNotFoundError (not ValueError), crashing
        # the Qt slot. Must be a ValueError the UI can catch and show.
        with pytest.raises(ValueError, match="Tank_attributes"):
            run_query(_chem_dataset(), subject=("element", "S"), basis="kg", group_by=["TankSystem"])

    def test_scope_on_a_missing_column_refuses_not_silently_ignores(self):
        # Finding: scope={"TankSystem": ["SST"]} on an attributes-less
        # dataset was dropped without a word -- unfiltered totals presented
        # as filtered. Refuse instead.
        with pytest.raises(ValueError, match="[Ss]cope"):
            run_query(_chem_dataset(), subject=("element", "S"), basis="kg",
                      scope={"TankSystem": ["SST"]})

    def test_kg_row_that_is_both_element_class_and_isotope_counts_once(self):
        # Finding: a kg row like "Cs-137" classifies as AnalyteClass
        # "element" AND parses as an isotope, so the kg-element and
        # kg-nuclide branches BOTH counted it -- exactly 2x the mass.
        dataset = _dataset({
            "WasteSiteId": ["241-A-101"],
            "Analyte": ["Cs-137"],
            "WastePhase": ["Supernatant"],
            "WasteType": ["T1"],
            "Inventory": [10.0],
            "Units": ["kg"],
        })
        res = run_query(dataset, subject=("element", "Cs"), basis="kg")
        assert res.table["Total_kg"].iloc[0] == pytest.approx(10.0)

    def test_element_basket_member_apportions_even_when_a_bare_analyte_row_exists(self):
        # Finding: member "S" fell into the exact-analyte path whenever a
        # bare "S" analyte row existed, silently dropping ALL apportioned
        # SO4/Sulfide mass (1.0 kg reported where the element view says
        # 39.4). The same subject string must resolve identically via the
        # Element kind and as a basket member.
        rows = {
            "WasteSiteId": ["241-A-101"] * 3,
            "Analyte": ["SO4", "Sulfide", "S"],
            "WastePhase": ["Supernatant"] * 3,
            "WasteType": ["T1"] * 3,
            "Inventory": [100.0, 5.0, 1.0],
            "Units": ["kg"] * 3,
        }
        as_element = run_query(_dataset(rows), subject=("element", "S"), basis="kg")
        as_member = run_query(_dataset(rows), subject=("basket", Basket("t", ("S",))), basis="kg")
        frac = rd.SPECIES_ELEMENT_MASS_FRACTIONS["SO4"]["S"]
        expected = 100.0 * frac + 5.0 + 1.0
        assert as_element.table["Total_kg"].iloc[0] == pytest.approx(expected)
        assert as_member.table["Total_kg"].iloc[0] == pytest.approx(expected)

    def test_isotope_member_still_resolves_as_the_exact_analyte(self):
        # Guard on the fix above: "129I" normalizes to element I, but the
        # Mobile long-lived basket means the exact nuclide -- an isotope
        # label must NOT widen to its whole element.
        dataset = _dataset({
            "WasteSiteId": ["241-A-101"] * 2,
            "Analyte": ["129I", "I"],
            "WastePhase": ["Supernatant"] * 2,
            "WasteType": ["T1"] * 2,
            "Inventory": [2.0, 50.0],
            "Units": ["Ci", "kg"],
        })
        res = run_query(dataset, subject=("basket", Basket("t", ("129I",))), basis="g")
        assert res.table["Total_g"].iloc[0] == pytest.approx(2.0 * rd.nuclide_grams_per_ci("129I"))

    def test_kg_nuclide_rows_get_a_real_disclosure_not_a_broken_one(self):
        # Finding: _member_disclosure had no branch for "kg-nuclide:" kinds
        # -> "Cd: ." (empty). And with a bridged Ci row alongside, the
        # unlisted kg mass made the "mass is a lower bound" claim false.
        dataset = _dataset({
            "WasteSiteId": ["241-A-101"] * 2,
            "Analyte": ["113mCd", "113mCd"],
            "WastePhase": ["Supernatant"] * 2,
            "WasteType": ["T1"] * 2,
            "Inventory": [0.5, 3.0],
            "Units": ["kg", "Ci"],
        })
        res = run_query(dataset, subject=("element", "Cd"), basis="g")
        text = "\n".join(res.disclosures)
        assert "kg nuclide rows" in text
        assert "lower bound" not in text  # kg mass IS reported for this member

    def test_hyphen_end_form_ci_label_bridges_like_the_canonical_form(self):
        # Finding: the Ci->g bridge keyed on the raw label, so "Cs-137"
        # (an end form parse_isotope explicitly supports) fell off the
        # bridge and was mis-warned as a combined-pair label.
        dataset = _dataset({
            "WasteSiteId": ["241-A-101"],
            "Analyte": ["Cs-137"],
            "WastePhase": ["Supernatant"],
            "WasteType": ["T1"],
            "Inventory": [100.0],
            "Units": ["Ci"],
        })
        res = run_query(dataset, subject=("element", "Cs"), basis="g")
        assert res.table["Total_g"].iloc[0] == pytest.approx(100.0 * rd.nuclide_grams_per_ci("137Cs"))
        assert not any("not mass-bridgeable" in w for w in res.warnings)

    def test_nuclide_grams_per_ci_accepts_end_form_labels(self):
        assert rd.nuclide_grams_per_ci("Cs-137") == rd.nuclide_grams_per_ci("137Cs")
        assert rd.nuclide_grams_per_ci("Cd-113m") == rd.nuclide_grams_per_ci("113mCd")
        assert rd.nuclide_grams_per_ci("239/240Pu") is None  # combined pair stays excluded

    def test_missing_layer_data_yields_blank_not_inf_with_warning(self):
        # Finding: null/absent Volume/ComponentDensity made wt% divide by
        # 0 -> inf, silently -- the one integrity failure the warning
        # system didn't cover.
        dataset = _dataset({
            "WasteSiteId": ["241-A-101"],
            "Analyte": ["Fe"],
            "WastePhase": ["Supernatant"],
            "WasteType": ["T1"],
            "Inventory": [10.0],
            "Units": ["kg"],
        })
        res = run_query(dataset, subject=("element", "Fe"), basis="wt%", group_by=["WasteSiteId"])
        assert res.table["wt_pct"].isna().all()
        assert any("volume/density" in w for w in res.warnings)

    def test_unexpected_volume_units_refuse_concentration_bases(self):
        # Finding: Volume was hardcoded as kL and density as g/mL with the
        # VolumeUnits/DensityUnits columns never consulted -- a gallons CSV
        # would be silently wrong by ~3.8x.
        dataset = _dataset({
            "WasteSiteId": ["241-A-101"],
            "Analyte": ["Fe"],
            "WastePhase": ["Supernatant"],
            "WasteType": ["T1"],
            "Inventory": [10.0],
            "Units": ["kg"],
            "Volume": [2.0],
            "VolumeUnits": ["kgal"],
            "ComponentDensity": [1.5],
        })
        with pytest.raises(ValueError, match="VolumeUnits"):
            run_query(dataset, subject=("element", "Fe"), basis="wt%", group_by=["WasteSiteId"])

    def test_mass_number_table_matches_the_shared_isotope_parser(self):
        # Behavior lock for the refactor replacing the hand-rolled label
        # parse with elements.parse_isotope.
        assert rd.NUCLIDE_MASS_NUMBER["137Cs"] == 137
        assert rd.NUCLIDE_MASS_NUMBER["113mCd"] == 113
        assert rd.NUCLIDE_MASS_NUMBER["239/240Pu"] is None
        assert rd.NUCLIDE_MASS_NUMBER["144Ce/Pr"] is None


@pytest.fixture(scope="module")
def real_dataset():
    dataset = dm.HanfordDataset()
    dataset.load(REAL_CSV_PATH, use_cache=True)
    return dataset


@requires_real_data
class TestSaltFormersAcceptance:
    """Stage 1 acceptance test from the blueprint roadmap: the shipped
    salt-formers basket runs end-to-end on the real CSV -- apportioned S
    from SO4, bridged 129I mass, per-member disclosure -- grouped by
    tank x phase on a wt% basis."""

    def test_flagship_query_runs_and_is_sane(self, real_dataset):
        res = run_query(
            real_dataset,
            subject=("basket", BASKET_PRESETS["Salt formers"]),
            basis="wt%",
            group_by=["WasteSiteId", "WastePhase"],
        )
        assert len(res.table) > 100                      # hundreds of tank x phase groups
        vals = res.table["wt_pct"].dropna()
        assert (vals >= 0).all() and (vals < 100).all()  # physically sane percentages

    def test_sulfur_is_apportioned_not_full_so4(self, real_dataset):
        # Manual cross-check on one tank x phase: engine S mass == hand
        # (SO4 kg x S fraction + Sulfide kg + Thiosulfate kg x fraction).
        res = run_query(
            real_dataset, subject=("element", "S"), basis="g",
            group_by=["WasteSiteId", "WastePhase"],
        )
        row = res.table.iloc[0]
        tank, phase = row["WasteSiteId"], row["WastePhase"]
        df = real_dataset.require_df().filter(
            (pl.col("WasteSiteId") == tank) & (pl.col("WastePhase") == phase) & (pl.col("Units") == "kg")
        )
        expected = 0.0
        for analyte, fracs in rd.SPECIES_ELEMENT_MASS_FRACTIONS.items():
            s_frac = fracs.get("S", 0.0)
            if s_frac > 0:
                expected += df.filter(pl.col("Analyte") == analyte)["Inventory"].sum() * s_frac * 1000.0
        assert row["Total_g"] == pytest.approx(expected)

    def test_iodine_discloses_bridged_lower_bound(self, real_dataset):
        res = run_query(real_dataset, subject=("basket", BASKET_PRESETS["Salt formers"]), basis="g")
        text = "\n".join(res.disclosures)
        assert "129I" in text and "lower bound" in text
        assert "apportioned from SO4" in text

    def test_mixed_decay_cohorts_warned_on_real_data(self, real_dataset):
        res = run_query(real_dataset, subject=("element", "Cs"), basis="Ci")
        assert any("decay baselines" in w for w in res.warnings)

    def test_member_table_covers_every_present_member(self, real_dataset):
        res = run_query(real_dataset, subject=("basket", BASKET_PRESETS["Salt formers"]), basis="g")
        assert {"F", "Cl", "S", "Cr", "I"} <= set(res.member_table["Member"])
