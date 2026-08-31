"""
Whole-dataset invariants of the L1 engine.

These are properties that must hold across the REAL dataset, not
hand-picked cases: they are the difference between "the apportionment
maths is right on the examples we thought of" and "the apportionment
maths cannot invent or lose mass anywhere".
"""
import polars as pl
import pytest

import data_model as dm
import reference_data as rd
from query_engine import run_query

from conftest import REAL_CSV_PATH, requires_real_data


class TestSpeciesFractions:
    def test_every_species_fractions_sum_to_one(self):
        # If a formula's fractions did not sum to 1, apportioning it would
        # create or destroy mass -- silently, and in proportion to how much
        # of that species the tanks hold.
        for analyte, fractions in rd.SPECIES_ELEMENT_MASS_FRACTIONS.items():
            assert sum(fractions.values()) == pytest.approx(1.0, abs=1e-12), analyte

    def test_no_species_has_a_zero_or_negative_fraction(self):
        for analyte, fractions in rd.SPECIES_ELEMENT_MASS_FRACTIONS.items():
            for element, frac in fractions.items():
                assert frac > 0.0, f"{analyte}/{element}"


@pytest.fixture(scope="module")
def real_dataset():
    dataset = dm.HanfordDataset()
    dataset.load(REAL_CSV_PATH, use_cache=True)
    return dataset


def _restated_total_mass(df: pl.DataFrame) -> float:
    """The mass a reported TOTAL already accounts for, so conservation is
    checked against what the engine may legitimately count rather than
    against the raw rows.

    Derived here from the raw frame rather than from the engine, so the
    test does not simply restate the code it is checking: for every layer
    holding a positive TOC row, each itemised organic anion in that layer
    contributes its CARBON only -- the oxygen and hydrogen in the same
    row are still booked, and so is the phosphorus in tributyl phosphate.
    """
    layer = ["WasteSiteId", "WastePhase", "WasteType"]
    total = 0.0
    for name, (element, members) in rd.SPECIES_TOTAL_MEMBERS.items():
        holds_total = (
            df.filter((pl.col("Analyte") == name) & (pl.col("Units") == "kg")
                      & (pl.col("Inventory") > 0))
            .select(layer).unique()
        )
        if holds_total.is_empty():
            continue
        itemised = df.filter(
            (pl.col("Units") == "kg") & pl.col("Analyte").is_in(list(members))
        ).join(holds_total, on=layer, how="semi", nulls_equal=True)
        for row in itemised.select(["Analyte", "Inventory"]).iter_rows(named=True):
            fraction = rd.SPECIES_ELEMENT_MASS_FRACTIONS[row["Analyte"]].get(element, 0.0)
            total += float(row["Inventory"]) * fraction
    return total


@requires_real_data
class TestMassConservation:
    def test_apportioning_every_element_returns_the_reported_mass(self, real_dataset):
        """Summing the engine's apportioned kg over EVERY element must
        equal the kg actually reported for the rows it draws from --
        no mass created, none lost. (Verified on the real dataset:
        149,487,676.7 kg both ways.)"""
        df = real_dataset.require_df()
        registered = list(rd.SPECIES_FORMULAS)

        elements = sorted(
            {el for fractions in rd.SPECIES_ELEMENT_MASS_FRACTIONS.values() for el in fractions}
            | set(df.filter(pl.col("Units") == "kg").get_column("Element").drop_nulls().unique().to_list())
        )
        apportioned = 0.0
        for element in elements:
            result = run_query(real_dataset, subject=("element", element), basis="kg")
            if not result.table.empty:
                apportioned += float(result.table["Total_kg"].fillna(0).sum())

        def reported(expr) -> float:
            return float(df.filter(expr).get_column("Inventory").sum() or 0.0)

        is_kg = pl.col("Units") == "kg"
        expected = (
            reported(is_kg & pl.col("Analyte").is_in(registered))          # registered species
            + reported(is_kg & (pl.col("AnalyteClass") == "element")       # bare elements
                       & pl.col("Isotope").is_null()
                       & ~pl.col("Analyte").is_in(registered))
            + reported(is_kg & pl.col("Isotope").is_not_null())            # nuclides reported in kg
            - _restated_total_mass(df)                                     # counted by their total
        )
        assert apportioned == pytest.approx(expected, rel=1e-9)

    def test_bridging_never_doubles_an_element_reported_both_ways(self, real_dataset):
        """A mass reported for an element already includes its
        radionuclides. Bridging curies to grams may only ADD where a layer
        reports activity and no mass, so the g-basis total can sit a little
        above the kg total, never near double it.

        Measured on the shipped dataset after the fix: U 1.0007, Th 1.0005
        (both were 1.999 and 2.003 when the bridged isotopes were being
        added to UTOTAL), Sr/Zr/Ni/C/H 1.0000. Samarium is the high end at
        1.093 because more of its layers report 151Sm activity without a
        bulk Sm mass -- that excess is real information, not a restatement.
        """
        both_ways = ["U", "Th", "Sm", "Sn", "Sr", "Zr", "Ni", "Se", "C", "H"]
        for element in both_ways:
            as_kg = run_query(real_dataset, subject=("element", element), basis="kg")
            as_g = run_query(real_dataset, subject=("element", element), basis="g")
            kg_total = float(as_kg.table["Total_kg"].sum())
            if kg_total <= 0:
                continue
            ratio = (float(as_g.table["Total_g"].sum()) / 1000.0) / kg_total
            assert ratio < 1.25, (
                f"{element}: g basis is {ratio:.3f}x the reported mass -- "
                f"its nuclides are being counted on top of the bulk measurement"
            )

    def test_species_provenance_split_sums_to_the_element_total(self, real_dataset):
        """Grouping an element subject by Analyte answers "which species
        carry it"; those shares must add back up to the ungrouped total,
        or the provenance view is telling a different story than the
        headline number."""
        for element in ("S", "N", "P", "U", "Cr"):
            total = run_query(real_dataset, subject=("element", element), basis="kg")
            by_species = run_query(real_dataset, subject=("element", element), basis="kg",
                                   group_by=["Analyte"])
            assert by_species.table["Total_kg"].sum() == pytest.approx(
                total.table["Total_kg"].sum(), rel=1e-9), element
