"""
Repro tests for the Overview-migration review (commit 5b3905e), led by
the one that matters most: the bridged mass basis double-counted every
element the dataset reports BOTH as bulk mass and as its own nuclides.

Uranium is the clean case. The BBI reports UTOTAL (all the uranium, in
kg) and separately the activity of 232-238U. Bridging those activities to
grams and ADDING them to UTOTAL counts the same metal twice: on the real
dataset the per-tank ratio of bridged-isotope mass to UTOTAL mass is
0.9926-1.0050 across all 177 tanks, and 235U/UTOTAL = 0.706 wt%, i.e.
natural uranium. They are one inventory described two ways.
"""
import polars as pl
import pytest
from PySide6.QtWidgets import QMainWindow

import data_model as dm
import matrix_query as mq
import reference_data as rd
from qt_overview import OverviewPage
from query_engine import run_query

from conftest import REAL_CSV_PATH, requires_real_data


def _page(qtbot):
    window = QMainWindow()
    qtbot.addWidget(window)
    page = OverviewPage(window)
    qtbot.addWidget(page)
    return page


def _uranium_dataset() -> dm.HanfordDataset:
    """One layer holding 100 kg of uranium, whose 238U activity is
    reported separately -- the same metal, stated radiologically."""
    grams_per_ci = rd.nuclide_grams_per_ci("238U")
    ci_for_100kg = 100_000.0 / grams_per_ci
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame({
        "WasteSiteId": ["241-A-101", "241-A-101"],
        "Analyte": ["UTOTAL", "238U"],
        "WastePhase": ["Sludge", "Sludge"],
        "WasteType": ["T1", "T1"],
        "Inventory": [100.0, ci_for_100kg],
        "Units": ["kg", "Ci"],
    }))
    dataset.report = None
    return dataset


def _iodine_dataset() -> dm.HanfordDataset:
    """Iodine is reported ONLY as 129I activity -- no kg row anywhere --
    so bridging it to grams adds real information and must survive."""
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame({
        "WasteSiteId": ["241-A-101"],
        "Analyte": ["129I"],
        "WastePhase": ["Sludge"],
        "WasteType": ["T1"],
        "Inventory": [2.0],
        "Units": ["Ci"],
    }))
    dataset.report = None
    return dataset


class TestBridgedMassDoesNotDoubleCount:
    def test_reported_element_mass_already_includes_its_nuclides(self):
        result = run_query(_uranium_dataset(), subject=("element", "U"), basis="g")
        assert result.table["Total_g"].iloc[0] == pytest.approx(100_000.0, rel=1e-6), (
            "the 238U activity was added to UTOTAL, counting the same uranium twice"
        )

    def test_the_restatement_is_disclosed_not_just_dropped(self):
        result = run_query(_uranium_dataset(), subject=("element", "U"), basis="g")
        text = " ".join(result.warnings + result.disclosures)
        assert "restate" in text.lower() or "already includes" in text.lower(), (
            f"nothing tells the reader the Ci rows were set aside: {result.warnings}"
        )

    def test_activity_basis_is_untouched(self):
        # Only the MASS bases overlap; summing activities is still right.
        result = run_query(_uranium_dataset(), subject=("element", "U"), basis="Ci")
        assert result.table["Total_Ci"].iloc[0] > 0

    def test_kg_basis_is_untouched(self):
        result = run_query(_uranium_dataset(), subject=("element", "U"), basis="kg")
        assert result.table["Total_kg"].iloc[0] == pytest.approx(100.0)

    def test_an_element_reported_only_in_curies_still_bridges(self):
        result = run_query(_iodine_dataset(), subject=("element", "I"), basis="g")
        assert result.table["Total_g"].iloc[0] == pytest.approx(
            2.0 * rd.nuclide_grams_per_ci("129I"))

    def test_suppression_is_per_layer_not_dataset_wide(self):
        # Layer A reports the metal in kg; layer B reports only activity.
        # B's bridged mass is new information and must be kept.
        grams_per_ci = rd.nuclide_grams_per_ci("238U")
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame({
            "WasteSiteId": ["241-A-101", "241-A-101", "241-AN-104"],
            "Analyte": ["UTOTAL", "238U", "238U"],
            "WastePhase": ["Sludge", "Sludge", "Supernatant"],
            "WasteType": ["T1", "T1", "T2"],
            "Inventory": [100.0, 50_000.0 / grams_per_ci, 7_000.0 / grams_per_ci],
            "Units": ["kg", "Ci", "Ci"],
        }))
        dataset.report = None
        result = run_query(dataset, subject=("element", "U"), basis="g",
                           group_by=["WasteSiteId"])
        totals = result.table.set_index("WasteSiteId")["Total_g"]
        assert totals["241-A-101"] == pytest.approx(100_000.0, rel=1e-6)
        assert totals["241-AN-104"] == pytest.approx(7_000.0, rel=1e-6)

    def test_analyte_subject_is_never_suppressed(self):
        # Asking for the isotope itself must return the isotope's mass.
        result = run_query(_uranium_dataset(), subject=("analyte", "238U"), basis="g")
        assert result.table["Total_g"].iloc[0] == pytest.approx(100_000.0, rel=1e-6)


@requires_real_data
class TestRealDatasetUranium:
    @pytest.fixture(scope="module")
    def real_dataset(self):
        dataset = dm.HanfordDataset()
        dataset.load(REAL_CSV_PATH, use_cache=True)
        return dataset

    def test_bridged_uranium_matches_the_reported_uranium(self, real_dataset):
        # UTOTAL is 677,042 kg. The g basis must not report ~1,354 t.
        as_kg = run_query(real_dataset, subject=("element", "U"), basis="kg")
        as_g = run_query(real_dataset, subject=("element", "U"), basis="g")
        kg_total = float(as_kg.table["Total_kg"].sum())
        g_total_in_kg = float(as_g.table["Total_g"].sum()) / 1000.0
        assert g_total_in_kg == pytest.approx(kg_total, rel=0.02), (
            f"uranium: {g_total_in_kg:,.0f} kg on the g basis vs {kg_total:,.0f} kg reported"
        )

    def test_thorium_too(self, real_dataset):
        as_kg = run_query(real_dataset, subject=("element", "Th"), basis="kg")
        as_g = run_query(real_dataset, subject=("element", "Th"), basis="g")
        assert float(as_g.table["Total_g"].sum()) / 1000.0 == pytest.approx(
            float(as_kg.table["Total_kg"].sum()), rel=0.02)

    def test_the_two_workspaces_still_agree_on_uranium(self, real_dataset):
        # Heatmaps offers kg only; Overview's default is the bridged basis.
        # After the fix those must not differ by 2x.
        heatmap = mq.engine_matrix(real_dataset, basis="kg", top_n_elements=60)
        overview = mq.element_totals(real_dataset, basis="g", top_n=60)
        hm_u = heatmap.wide_df["U"].sum(skipna=True)
        ov_u = float(overview.table.set_index("Element").loc["U", "Total_g"]) / 1000.0
        assert ov_u == pytest.approx(hm_u, rel=0.02)


class TestOverviewSurfacesItsDisclosures:
    def test_warnings_reach_the_page(self, qtbot):
        page = _page(qtbot)
        page.on_dataset_changed(_uranium_dataset())
        qtbot.wait(20)
        notes = page._table_views["Notes"].dataframe()
        assert not notes.empty, "element_totals' warnings and disclosures are discarded"
        assert (notes["kind"] == "disclosure").any()

    def test_a_trim_is_visible_on_the_page(self, qtbot):
        page = _page(qtbot)
        page.on_dataset_changed(_uranium_dataset())
        page.top_n_spin.setValue(5)
        page.unit_combo.setCurrentText("kg")
        page.refresh()
        qtbot.wait(20)
        # With only U present nothing is trimmed, but the surface must exist.
        assert hasattr(page, "warnings_label")


class TestOverviewHandlesUnexpectedUnits:
    def test_a_unit_the_engine_does_not_know_does_not_raise(self, qtbot):
        # available_units() returns whatever the CSV holds; passing "MT"
        # to the engine as a basis raised straight out of refresh() and
        # aborted the shell's whole dataset broadcast.
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame({
            "WasteSiteId": ["241-A-101"],
            "Analyte": ["Na"],
            "WastePhase": ["Sludge"],
            "WasteType": ["T1"],
            "Inventory": [5.0],
            "Units": ["MT"],
        }))
        dataset.report = None
        page = _page(qtbot)
        page.on_dataset_changed(dataset)  # must not raise
        qtbot.wait(20)
        page.unit_combo.setCurrentText("MT")
        page.refresh()  # must not raise
        qtbot.wait(20)


@requires_real_data
class TestDebugBundleAgreesWithTheScreen:
    def test_exported_top_elements_uses_the_same_attribution(self, qtbot, tmp_path):
        dataset = dm.HanfordDataset()
        dataset.load(REAL_CSV_PATH, use_cache=True)
        dataset.output_root = tmp_path
        page = _page(qtbot)
        page.on_dataset_changed(dataset)
        page.unit_combo.setCurrentText("kg")
        page.refresh()
        qtbot.wait(20)
        page._export_debug_bundle()

        import pandas as pd
        bundle = next(iter(tmp_path.glob("debug_bundle_*")))
        exported = pd.read_csv(bundle / "top_elements_all.csv")
        on_screen = page._table_views["Top elements"].dataframe()
        assert exported["Element"].iloc[0] == on_screen["Element"].iloc[0], (
            "the debug bundle's top element contradicts the one on screen"
        )


class TestComposerTitleOverrideFindsTheRightSlot:
    """A grammar-rendered panel puts its title on the subfigure suptitle
    and its disclosure line on the axes title. The composer's title
    override aimed at the axes, so "hide the title" erased the
    disclosures and left the heading standing."""

    def _composed(self, qtbot):
        import composer_store as cs
        import figure_composer as fcomp
        import pandas as pd
        from plot_grammar import PlotSpec, render_plot
        from qt_widgets import PlotWidget

        panel = PlotWidget()
        qtbot.addWidget(panel)
        df = pd.DataFrame({"Element": ["O", "Na"], "Total_kg": [3.0, 1.0]})
        spec = PlotSpec(mark="bar", x="Total_kg", y="Element",
                        title="Top elements by inventory",
                        subtitle="Apportioned (stoichiometric)")
        render_plot(panel, df, spec)
        qtbot.wait(20)
        store = cs.ComposerStore()
        fn, args, kwargs = panel._last_recipe
        store.add_recipe(fn, args, kwargs, caption="c", source="s")
        out = PlotWidget()
        qtbot.addWidget(out)
        composed = fcomp.compose_figure(out, store.items(), cols=1)
        return store, out, composed

    def test_hiding_the_title_hides_the_title_not_the_disclosure(self, qtbot):
        import figure_composer as fcomp
        store, out, composed = self._composed(qtbot)
        store.set_title_override(0, "")
        fcomp.apply_text_override(out, composed, 0, store.items()[0], "title")
        qtbot.wait(20)
        cell = composed.cells[0]
        assert cell.figure.get_suptitle() == "", "the heading survived 'hide title'"
        assert "Apportioned" in cell.ax.get_title(), "the disclosure line was erased instead"

    def test_replacing_the_title_does_not_stack_two_headings(self, qtbot):
        import figure_composer as fcomp
        store, out, composed = self._composed(qtbot)
        store.set_title_override(0, "My caption")
        fcomp.apply_text_override(out, composed, 0, store.items()[0], "title")
        qtbot.wait(20)
        cell = composed.cells[0]
        assert cell.figure.get_suptitle() == "My caption"
        assert "Top elements" not in cell.ax.get_title()
