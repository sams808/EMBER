"""
"One inventory described two ways" — the general case.

The BBI states the same material more than once in three distinct ways,
and adding any of them to itself inflates a published number:

1. UTOTAL (kg) alongside the activity of each uranium isotope.
2. A combined-pair label (239/240Pu) alongside 239Pu and 240Pu
   individually -- the same ACTIVITY reported twice, which the mass-side
   fix did not cover.
3. Either of the above reached through a BASKET, where one member
   resolves as an element and another as an analyte, so per-member
   suppression never sees the overlap.
"""
import polars as pl
import pytest

import data_model as dm
import explorer_query as eq
import reference_data as rd
from query_engine import Basket, run_query

from conftest import REAL_CSV_PATH, requires_real_data


def _pu_dataset() -> dm.HanfordDataset:
    """One layer reporting the pair AND both its members, as the real
    dataset does in 181 of 181 such layers."""
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame({
        "WasteSiteId": ["241-AN-101"] * 3,
        "Analyte": ["239/240Pu", "239Pu", "240Pu"],
        "WastePhase": ["Sludge"] * 3,
        "WasteType": ["T1"] * 3,
        "Inventory": [2700.0, 2200.0, 500.0],
        "Units": ["Ci"] * 3,
    }))
    dataset.report = None
    return dataset


def _uranium_dataset() -> dm.HanfordDataset:
    grams_per_ci = rd.nuclide_grams_per_ci("238U")
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame({
        "WasteSiteId": ["241-A-101"] * 2,
        "Analyte": ["UTOTAL", "238U"],
        "WastePhase": ["Sludge"] * 2,
        "WasteType": ["T1"] * 2,
        "Inventory": [100.0, 100_000.0 / grams_per_ci],
        "Units": ["kg", "Ci"],
    }))
    dataset.report = None
    return dataset


class TestCombinedPairActivity:
    def test_the_pair_and_its_members_are_not_both_counted(self):
        result = run_query(_pu_dataset(), subject=("element", "Pu"), basis="Ci")
        assert result.table["Total_Ci"].iloc[0] == pytest.approx(2700.0), (
            "239/240Pu was added to 239Pu + 240Pu, counting the same activity twice"
        )

    def test_the_restatement_is_disclosed(self):
        result = run_query(_pu_dataset(), subject=("element", "Pu"), basis="Ci")
        assert any("restate" in w.lower() or "same activity" in w.lower()
                   for w in result.warnings), result.warnings

    def test_an_incomplete_itemisation_keeps_the_pair_not_the_fragment(self):
        # Pair + only ONE member: dropping the pair would lose the other
        # member's activity, so the pair is the exact representation.
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame({
            "WasteSiteId": ["241-AN-101"] * 2,
            "Analyte": ["239/240Pu", "239Pu"],
            "WastePhase": ["Sludge"] * 2,
            "WasteType": ["T1"] * 2,
            "Inventory": [2700.0, 2200.0],
            "Units": ["Ci"] * 2,
        }))
        dataset.report = None
        result = run_query(dataset, subject=("element", "Pu"), basis="Ci")
        assert result.table["Total_Ci"].iloc[0] == pytest.approx(2700.0)

    def test_a_pair_alone_is_untouched(self):
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame({
            "WasteSiteId": ["241-AN-101"],
            "Analyte": ["239/240Pu"],
            "WastePhase": ["Sludge"],
            "WasteType": ["T1"],
            "Inventory": [2700.0],
            "Units": ["Ci"],
        }))
        dataset.report = None
        result = run_query(dataset, subject=("element", "Pu"), basis="Ci")
        assert result.table["Total_Ci"].iloc[0] == pytest.approx(2700.0)

    def test_members_alone_are_untouched(self):
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame({
            "WasteSiteId": ["241-AN-101"] * 2,
            "Analyte": ["239Pu", "240Pu"],
            "WastePhase": ["Sludge"] * 2,
            "WasteType": ["T1"] * 2,
            "Inventory": [2200.0, 500.0],
            "Units": ["Ci"] * 2,
        }))
        dataset.report = None
        result = run_query(dataset, subject=("element", "Pu"), basis="Ci")
        assert result.table["Total_Ci"].iloc[0] == pytest.approx(2700.0)

    def test_suppression_is_per_layer(self):
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame({
            "WasteSiteId": ["241-AN-101", "241-AN-101", "241-AN-101", "241-AZ-102"],
            "Analyte": ["239/240Pu", "239Pu", "240Pu", "239/240Pu"],
            "WastePhase": ["Sludge"] * 3 + ["Supernatant"],
            "WasteType": ["T1"] * 3 + ["T2"],
            "Inventory": [2700.0, 2200.0, 500.0, 40.0],
            "Units": ["Ci"] * 4,
        }))
        dataset.report = None
        result = run_query(dataset, subject=("element", "Pu"), basis="Ci",
                           group_by=["WasteSiteId"])
        totals = result.table.set_index("WasteSiteId")["Total_Ci"]
        assert totals["241-AN-101"] == pytest.approx(2700.0)
        assert totals["241-AZ-102"] == pytest.approx(40.0)


class TestBasketsSeeTheOverlapToo:
    def test_a_basket_mixing_an_element_and_its_isotope_does_not_restate(self):
        basket = Basket("mix", ("U", "238U"))
        result = run_query(_uranium_dataset(), subject=("basket", basket), basis="g")
        assert result.table["Total_g"].iloc[0] == pytest.approx(100_000.0, rel=1e-6), (
            "a basket member resolving as an analyte bypassed the suppression"
        )

    def test_a_pattern_search_expanded_to_a_basket_does_not_restate(self):
        # explorer_query turns "regex ^U" into a basket of every matching
        # analyte, which is exactly the mixed element/analyte case.
        dataset = _uranium_dataset()
        resolved = eq.resolve_subject(dataset, "U", "regex")
        out = eq.target_tables(dataset, resolved, basis="g")
        assert out.tables["Target by tank"]["Total_g"].sum() == pytest.approx(
            100_000.0, rel=1e-6)

    def test_a_single_analyte_subject_is_still_never_suppressed(self):
        result = run_query(_uranium_dataset(), subject=("analyte", "238U"), basis="g")
        assert result.table["Total_g"].iloc[0] == pytest.approx(100_000.0, rel=1e-6)

    def test_unrelated_members_do_not_suppress_each_other(self):
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame({
            "WasteSiteId": ["241-A-101"] * 2,
            "Analyte": ["Fe", "137Cs"],
            "WastePhase": ["Sludge"] * 2,
            "WasteType": ["T1"] * 2,
            "Inventory": [10.0, 300.0],
            "Units": ["kg", "Ci"],
        }))
        dataset.report = None
        result = run_query(dataset, subject=("basket", Basket("b", ("Fe", "Cs"))), basis="g")
        expected = 10_000.0 + 300.0 * rd.nuclide_grams_per_ci("137Cs")
        assert result.table["Total_g"].iloc[0] == pytest.approx(expected)


class TestZeroMassDoesNotSuppress:
    def test_a_reported_zero_is_not_a_mass_that_includes_its_nuclides(self):
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame({
            "WasteSiteId": ["241-A-101"] * 2,
            "Analyte": ["Zr", "93Zr"],
            "WastePhase": ["Sludge"] * 2,
            "WasteType": ["T1"] * 2,
            "Inventory": [0.0, 1.0],
            "Units": ["kg", "Ci"],
        }))
        dataset.report = None
        result = run_query(dataset, subject=("element", "Zr"), basis="g")
        assert result.table["Total_g"].iloc[0] == pytest.approx(
            rd.nuclide_grams_per_ci("93Zr")), "a 0 kg row swallowed the bridged mass"


@requires_real_data
class TestRealDataset:
    @pytest.fixture(scope="module")
    def real_dataset(self):
        dataset = dm.HanfordDataset()
        dataset.load(REAL_CSV_PATH, use_cache=True)
        return dataset

    def test_plutonium_activity_is_not_inflated(self, real_dataset):
        # Engine reported 123,205 Ci; itemised members give 101,408 Ci.
        result = run_query(real_dataset, subject=("element", "Pu"), basis="Ci")
        total = float(result.table["Total_Ci"].sum())
        assert total < 110_000, f"Pu activity still inflated: {total:,.0f} Ci"

    def test_curium_activity_is_not_inflated(self, real_dataset):
        result = run_query(real_dataset, subject=("element", "Cm"), basis="Ci")
        assert float(result.table["Total_Ci"].sum()) < 600
