"""
Real-CSV-gated integration tests (skipped unless Hanford.csv is present
locally -- it's gitignored dev-seed data, so these only run in a dev
environment that has copied the real files in, never in a clean CI
checkout). These are deliberately NOT a re-verification of correctness
(that's what the synthetic-fixture unit tests in every other test_*.py
module are for, per the project's stated testing philosophy) -- they are
a small, targeted "does the real 46,894-row / 177-tank dataset actually
flow through every major workspace's entry point without crashing, with
sane real numbers" smoke pass. This formalizes, as an automated regression
test, what M6-M12 verified by hand via screenshots during development
(some of which had to be skipped in later milestones because the desktop
was in active use) -- the real-data pipeline behavior stays checked
either way.
"""
import math

import polars as pl
import pytest

import correlation_science as csci
import data_model as dm
import element_science as esci
import matrix_science as msci
import overview_science as osci
import oxide_science as oxsci
import structure_science as ssci
import tank_science as tsci
import vitrification_science as vsci
from conftest import REAL_ATTRS_PATH, REAL_CSV_PATH, requires_real_data

DEFAULT_ELEMENTS = "Cs, Sr, Tc, I, Se, U, Cr, Fe, Al, Na, P, B, Si"


@pytest.fixture(scope="module")
def real_dataset():
    dataset = dm.HanfordDataset()
    dataset.load(REAL_CSV_PATH, use_cache=False, attributes_path=REAL_ATTRS_PATH)
    return dataset


@requires_real_data
class TestDataModelRealData:
    def test_combined_isotope_mass_not_dropped_pu_and_cm(self, real_dataset):
        # The headline bug fix from M1: both combined-isotope analytes
        # must resolve to their element, not be silently dropped.
        for analyte, element in [("239/240Pu", "Pu"), ("243/244Cm", "Cm")]:
            rows = real_dataset.df.filter(pl.col("Analyte") == analyte)
            assert rows.height > 0, f"{analyte} missing from real data -- fixture assumption changed"
            assert rows["Element"].unique().to_list() == [element]

    def test_available_units_are_kg_and_ci(self, real_dataset):
        assert set(real_dataset.available_units()) == {"kg", "Ci"}

    def test_tank_and_farm_counts_are_sane(self, real_dataset):
        assert len(real_dataset.available_tanks()) == 177
        assert len(real_dataset.available_farms()) > 0


@requires_real_data
class TestOverviewRealData:
    def test_overview_and_audits_run_without_crash(self, real_dataset):
        assert not osci.overview(real_dataset).empty
        assert not osci.units_audit(real_dataset).empty
        assert not osci.top_elements(real_dataset, unit="kg", top_n=20).empty


@requires_real_data
class TestElementScienceRealData:
    def test_target_search_for_known_element(self, real_dataset):
        rows, resolved_mode, symbol = esci.target_rows(real_dataset, "Cs")
        assert rows.height > 0
        assert resolved_mode == "element"
        assert symbol == "Cs"


@requires_real_data
class TestTankScienceRealData:
    def test_tank_attributes_table_covers_real_tanks(self, real_dataset):
        attrs = tsci.tank_attributes_table(real_dataset)
        assert not attrs.empty
        assert set(attrs["WasteSiteId"]) <= set(real_dataset.available_tanks())


@requires_real_data
class TestMatrixScienceRealData:
    def test_element_inventory_matrix_kg(self, real_dataset):
        matrix = msci.element_inventory_matrix(real_dataset, unit="kg", top_n=20, value_mode="log10_plus1")
        assert not matrix.empty
        assert matrix.shape[0] == 177  # include_all_tanks defaults True


@requires_real_data
class TestCorrelationScienceRealData:
    def test_kg_correlation_workbench_default_elements(self, real_dataset):
        # Same default element list the Association Workbench UI ships
        # with (spot-checked manually via screenshots during M7).
        out = csci.kg_correlation_workbench(real_dataset, elements_text=DEFAULT_ELEMENTS, value_mode="log10_plus1")
        assert not out["element_stats"].empty
        assert not out["pair_stats"].empty
        corr = out["corr_matrix"].set_index("Element")
        for element in corr.index:
            assert corr.loc[element, element] == pytest.approx(1.0)
        # Cs/Tc/I are historically Ci-only in this dataset -- must be
        # present as zero-kg elements, not silently dropped.
        stats = out["element_stats"].set_index("Element")
        for element in ("Cs", "Tc", "I"):
            assert stats.loc[element, "Total_inventory_kg"] == pytest.approx(0.0)

    def test_control_for_total_inventory_runs_without_crash(self, real_dataset):
        out, _ = csci.element_correlation_scan(
            real_dataset, "Cs", unit="kg", min_overlap=0, control_for_total_inventory=True,
        )
        # Cs is Ci-only in this dataset, so a kg-basis scan legitimately
        # returns nothing -- the point of this test is that requesting
        # partial correlation against the real data doesn't crash either way.
        assert out is not None


@requires_real_data
class TestStructureScienceRealData:
    def test_structure_workbench_default_elements(self, real_dataset):
        out = ssci.structure_workbench(real_dataset, elements_text=DEFAULT_ELEMENTS, value_mode="log10_plus1", n_clusters=4)
        assert not out["tank_summary"].empty
        assert not out["network_nodes"].empty
        variance = out["pca_variance"]
        assert variance["ExplainedVarianceRatio"].sum() <= 1.0 + 1e-6


@requires_real_data
class TestOxideScienceRealData:
    def test_convert_a_real_tank_to_oxides(self, real_dataset):
        tank = real_dataset.available_tanks()[0]
        tables = oxsci.tank_oxide_composition(real_dataset, [tank])
        assert tank in tables
        table = tables[tank]
        if not table.empty:
            assert table["Wt_pct"].sum() == pytest.approx(100.0, abs=1e-6)
            summary = oxsci.composition_summary(table)
            assert math.isnan(summary["optical_basicity"]) or 0.0 <= summary["optical_basicity"] <= 2.0


@requires_real_data
class TestVitrificationScienceRealData:
    def test_tank_category_summary_scores_are_finite_and_clipped(self, real_dataset):
        summary = vsci.tank_category_summary(real_dataset)
        assert len(summary) == 177
        scores = summary["Vitrification_screening_score_proxy"]
        assert scores.notna().all()
        assert scores.between(-100.0, 100.0).all()

    def test_candidate_search_and_blend_partner_search_run_without_crash(self, real_dataset):
        candidates = vsci.vitrification_candidate_search(
            real_dataset, target_elements=["Cs", "Sr"], penalty_elements=["Cl", "S"], required_elements=[], top_n=10,
        )
        assert not candidates.empty
        base_tank = candidates.iloc[0]["WasteSiteId"]
        blend = vsci.blend_partner_search(real_dataset, base_tank, top_n=10)
        assert not blend.empty
