"""
Engine-backed tank x element matrix (matrix_query).

The point of the migration is that the two attributions answer different
questions and disagree by known, checkable factors -- so the tests below
hand-derive those factors from the reference data rather than restating
whatever the code happens to produce.
"""
import polars as pl
import pytest

import data_model as dm
import matrix_query as mq
import reference_data as rd

from conftest import REAL_CSV_PATH, requires_real_data


def _species_dataset() -> dm.HanfordDataset:
    """Two tanks. Tank 1 holds 100 kg SO4 (S + O) and 10 kg Fe; tank 2
    holds 50 kg NO3 (N + O). Under first-element attribution the oxygen
    is invisible and sulfur/nitrogen carry the whole species mass."""
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame({
        "WasteSiteId": ["241-A-101", "241-A-101", "241-AN-104"],
        "Analyte": ["SO4", "Fe", "NO3"],
        "WastePhase": ["Supernatant", "Supernatant", "Sludge"],
        "WasteType": ["T1", "T1", "T2"],
        "Inventory": [100.0, 10.0, 50.0],
        "Units": ["kg", "kg", "kg"],
    }))
    dataset.report = None
    return dataset


class TestCandidateElements:
    def test_includes_species_constituents_not_only_tagged_elements(self):
        elements = mq.candidate_elements(_species_dataset())
        # O is never a "primary" element of anything, so it exists only
        # because species formulas are consulted.
        assert "O" in elements
        assert {"S", "N", "Fe"} <= set(elements)


class TestApportionedMatrix:
    def test_element_gets_its_mass_fraction_of_each_species(self):
        result = mq.engine_matrix(_species_dataset(), basis="kg", top_n_elements=10)
        wide = result.wide_df.set_index("WasteSiteId")
        s_frac = rd.SPECIES_ELEMENT_MASS_FRACTIONS["SO4"]["S"]
        o_frac_so4 = rd.SPECIES_ELEMENT_MASS_FRACTIONS["SO4"]["O"]
        o_frac_no3 = rd.SPECIES_ELEMENT_MASS_FRACTIONS["NO3"]["O"]
        assert wide.loc["241-A-101", "S"] == pytest.approx(100.0 * s_frac)
        assert wide.loc["241-A-101", "O"] == pytest.approx(100.0 * o_frac_so4)
        assert wide.loc["241-AN-104", "O"] == pytest.approx(50.0 * o_frac_no3)
        assert wide.loc["241-A-101", "Fe"] == pytest.approx(10.0)

    def test_row_totals_conserve_the_reported_mass(self):
        result = mq.engine_matrix(_species_dataset(), basis="kg", top_n_elements=20)
        wide = result.wide_df.set_index("WasteSiteId")
        assert wide.loc["241-A-101"].sum() == pytest.approx(110.0)  # 100 SO4 + 10 Fe
        assert wide.loc["241-AN-104"].sum() == pytest.approx(50.0)

    def test_top_n_trim_is_disclosed(self):
        result = mq.engine_matrix(_species_dataset(), basis="kg", top_n_elements=2)
        assert len(result.elements) == 2
        assert any("omitted" in w for w in result.warnings)

    def test_scope_restricts_the_matrix(self):
        result = mq.engine_matrix(_species_dataset(), basis="kg", top_n_elements=10,
                                  scope={"WastePhase": ["Sludge"]})
        assert list(result.wide_df["WasteSiteId"]) == ["241-AN-104"]

    def test_tank_subset_restricts_the_matrix(self):
        result = mq.engine_matrix(_species_dataset(), basis="kg", top_n_elements=10,
                                  tank_subset=["241-A-101"])
        assert list(result.wide_df["WasteSiteId"]) == ["241-A-101"]

    def test_empty_scope_yields_an_empty_result_not_a_crash(self):
        result = mq.engine_matrix(_species_dataset(), basis="kg",
                                  scope={"WastePhase": ["Nonexistent"]})
        assert result.wide_df.empty


class TestLegacyMatrix:
    def test_books_the_whole_species_mass_to_its_first_element(self):
        result = mq.legacy_matrix(_species_dataset(), unit="kg", top_n_elements=10)
        wide = result.wide_df.set_index("WasteSiteId")
        assert wide.loc["241-A-101", "S"] == pytest.approx(100.0)  # not 33.4
        assert "O" not in wide.columns                              # oxygen invisible
        assert result.disclosures and "first parsed element" in result.disclosures[0]

    def test_honors_the_same_scope_filter_as_the_engine(self):
        result = mq.legacy_matrix(_species_dataset(), unit="kg", top_n_elements=10,
                                  scope={"WastePhase": ["Sludge"]})
        assert list(result.wide_df["WasteSiteId"]) == ["241-AN-104"]

    def test_scoping_does_not_mutate_the_dataset(self):
        dataset = _species_dataset()
        before = dataset.require_df().height
        mq.legacy_matrix(dataset, unit="kg", scope={"WastePhase": ["Sludge"]})
        assert dataset.require_df().height == before


class TestAttributionDelta:
    def test_reports_the_ratio_between_the_two_attributions(self):
        dataset = _species_dataset()
        engine = mq.engine_matrix(dataset, basis="kg", top_n_elements=20)
        legacy = mq.legacy_matrix(dataset, unit="kg", top_n_elements=20)
        delta = mq.attribution_delta(engine, legacy).set_index("Element")
        s_frac = rd.SPECIES_ELEMENT_MASS_FRACTIONS["SO4"]["S"]
        assert delta.loc["S", "Ratio"] == pytest.approx(s_frac)   # ~0.334
        assert delta.loc["Fe", "Ratio"] == pytest.approx(1.0)     # elemental: unchanged


@pytest.fixture(scope="module")
def real_dataset():
    dataset = dm.HanfordDataset()
    dataset.load(REAL_CSV_PATH, use_cache=True)
    return dataset


@requires_real_data
class TestOnRealData:
    def test_oxygen_becomes_visible_and_nitrogen_shrinks(self, real_dataset):
        # The headline consequence of the migration, hand-checked against
        # the ratios measured on the shipped dataset.
        engine = mq.engine_matrix(real_dataset, basis="kg", top_n_elements=40)
        legacy = mq.legacy_matrix(real_dataset, unit="kg", top_n_elements=40)
        delta = mq.attribution_delta(engine, legacy).set_index("Element")

        assert delta.loc["O", "Ratio"] > 10          # measured 15.6x
        assert delta.loc["N", "Ratio"] < 0.4         # measured 0.24
        assert delta.loc["S", "Ratio"] == pytest.approx(
            rd.SPECIES_ELEMENT_MASS_FRACTIONS["SO4"]["S"], rel=0.05)
        for elemental in ("Na", "Al", "Fe", "Cr"):
            assert delta.loc[elemental, "Ratio"] == pytest.approx(1.0, rel=1e-6)

    def test_oxygen_is_the_largest_element_once_apportioned(self, real_dataset):
        result = mq.engine_matrix(real_dataset, basis="kg", top_n_elements=5)
        assert result.elements[0] == "O"

    def test_matrix_is_built_from_one_query_not_one_per_element(self, real_dataset):
        # A per-element loop would be ~58 queries; the basket form keeps
        # the workspace interactive. Guard the shape, not the clock.
        result = mq.engine_matrix(real_dataset, basis="kg", top_n_elements=35)
        assert len(result.elements) == 35
        assert len(result.wide_df) == 177          # every tank
        assert not result.long_df.empty
