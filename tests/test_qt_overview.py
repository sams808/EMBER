import pandas as pd
import pytest
from PySide6.QtWidgets import QMainWindow

import data_model as dm
from plot_grammar import PlotSpec, render_plot
from qt_overview import OverviewPage
from qt_widgets import PlotWidget


def _make_page(qtbot):
    app_window = QMainWindow()  # only used for .statusBar()
    qtbot.addWidget(app_window)
    page = OverviewPage(app_window)
    qtbot.addWidget(page)
    return page


class TestOverviewPage:
    def test_refresh_without_dataset_shows_message_not_crash(self, qtbot):
        page = _make_page(qtbot)
        page.refresh()  # QMessageBox.information neutralized by conftest; must not raise

    def test_on_dataset_changed_populates_unit_combo(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        qtbot.wait(20)  # let the plot's canvas.draw_idle() flush before teardown
        items = [page.unit_combo.itemText(i) for i in range(page.unit_combo.count())]
        assert items == ["All", "Ci", "kg"]

    def test_on_dataset_changed_triggers_refresh(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        qtbot.wait(20)
        overview_table = page._table_views["Overview"].dataframe()
        assert overview_table.iloc[0]["rows"] == 9

    def test_unit_filter_changes_the_basis_of_top_elements(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        qtbot.wait(20)
        page.unit_combo.setCurrentText("kg")
        page.refresh()
        qtbot.wait(20)
        assert page._element_totals.value_column == "Total_kg"
        top = page._table_views["Top elements"].dataframe()
        assert "Element" in top.columns and "Total_kg" in top.columns

    def test_all_units_bridges_onto_one_mass_axis(self, qtbot, sample_dataset):
        # "All" used to rank kilograms against curies on one axis, which
        # compares nothing; it now means the engine's bridged mass basis.
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        qtbot.wait(20)
        page.unit_combo.setCurrentText("All")
        page.refresh()
        qtbot.wait(20)
        assert page._element_totals.basis == "g"
        assert page._element_totals.value_column == "Total_g"
        # Assert a BRIDGED NUMBER, not just the basis string: the fixture's
        # caesium exists only as 100 + 200 Ci of 137Cs, so its appearance
        # here at the ICRP-107 mass is the only proof the bridge ran.
        import reference_data as rd
        totals = page._element_totals.table.set_index("Element")
        assert totals.loc["Cs", "Total_g"] == pytest.approx(
            300.0 * rd.nuclide_grams_per_ci("137Cs"))

    def test_top_elements_are_apportioned_not_first_element(self, qtbot):
        # The whole point of the migration: this page must agree with
        # Heatmaps about what the waste contains.
        import polars as pl
        import reference_data as rd
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame({
            "WasteSiteId": ["241-A-101"],
            "Analyte": ["SO4"],
            "WastePhase": ["Supernatant"],
            "WasteType": ["T1"],
            "Inventory": [100.0],
            "Units": ["kg"],
        }))
        dataset.report = None
        page = _make_page(qtbot)
        page.on_dataset_changed(dataset)
        page.unit_combo.setCurrentText("kg")
        page.refresh()
        qtbot.wait(20)
        top = page._table_views["Top elements"].dataframe().set_index("Element")
        assert top.loc["S", "Total_kg"] == pytest.approx(
            100.0 * rd.SPECIES_ELEMENT_MASS_FRACTIONS["SO4"]["S"])
        assert "O" in top.index  # oxygen exists only under apportionment

    def test_farm_plot_labels_match_the_data_actually_plotted(self, qtbot, sample_dataset):
        # Review finding: _update_plot labeled the farm plot with the LIVE
        # unit combo while plotting the rollup cached at the last Refresh --
        # a chart titled "(kg)" could carry Ci values. Labels must state the
        # unit the cached data was computed in, not whatever the combo says
        # right now.
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)  # refresh resolves to kg
        qtbot.wait(20)
        page.unit_combo.setCurrentText("Ci")  # no refresh happens here
        page.plot_type_combo.setCurrentText("Farm rollup by inventory")
        qtbot.wait(20)
        assert "kg" in page.plot.ax.get_xlabel()
        assert "Ci" not in page.plot.ax.get_xlabel()

    def test_farm_table_keeps_farms_that_lack_rows_in_the_resolved_unit(self, qtbot):
        # Review finding: the rollup replaced the complete unit-blind farm
        # audit, so a farm reported only in the other unit silently vanished
        # from the table -- an audit view must list every farm.
        import data_model as dm
        import polars as pl
        dataset = dm.HanfordDataset()
        rows = {
            "WasteSiteId": ["241-A-101", "241-SX-101"],
            "Analyte": ["Fe", "137Cs"],
            "WastePhase": ["Solid", "Liquid"],
            "WasteType": ["T1", "T2"],
            "Inventory": [50.0, 200.0],
            "Units": ["kg", "Ci"],  # SX farm exists ONLY in Ci
        }
        dataset.df = dataset._clean_dataframe(pl.DataFrame(rows))
        dataset.report = None

        page = _make_page(qtbot)
        page.on_dataset_changed(dataset)
        qtbot.wait(20)
        farms_table = page._table_views["Tank farms"].dataframe()
        assert "SX" in set(farms_table["TankFarm"])

    def test_export_without_dataset_shows_message_not_crash(self, qtbot):
        page = _make_page(qtbot)
        page._export_debug_bundle()  # must not raise

    def test_export_writes_debug_bundle(self, qtbot, sample_dataset, tmp_path):
        page = _make_page(qtbot)
        sample_dataset.output_root = tmp_path
        page.on_dataset_changed(sample_dataset)
        qtbot.wait(20)
        page._export_debug_bundle()
        bundles = list(tmp_path.glob("debug_bundle_*"))
        assert len(bundles) == 1
        assert (bundles[0] / "overview.csv").exists()


class TestBarChartSizing:
    """Regression coverage for the Overview "Top elements by inventory"
    chart, now drawn by the plot grammar: with Top N left at its default
    of 40, a fixed-height figure crammed every y-tick label into the same
    vertical space, making them overlap into illegibility even though the
    panel had plenty of unused space below the chart. These pin the
    sizing contract wherever that chart is rendered from."""

    def _df(self, n):
        return pd.DataFrame({
            "Element": [f"E{i}" for i in range(n)],
            "Total_kg": [float(n - i) for i in range(n)],
        })

    def _draw(self, panel, n):
        render_plot(panel, self._df(n), PlotSpec(
            mark="bar", x="Total_kg", y="Element",
            title="Top elements by inventory", log_x=True, top_n=n,
        ))

    def test_more_bars_grows_figure_height(self, qtbot):
        panel = PlotWidget()
        qtbot.addWidget(panel)
        self._draw(panel, 3)
        qtbot.wait(20)
        small_height = panel.figure.get_size_inches()[1]

        self._draw(panel, 40)
        qtbot.wait(20)
        assert panel.figure.get_size_inches()[1] > small_height

    def test_canvas_pixel_height_grows_too(self, qtbot):
        # The figure's declared inches size changing isn't sufficient on its
        # own -- Figure.set_size_inches(forward=True) is a no-op for a
        # canvas with no pyplot manager, so the actual on-screen widget must
        # be resized too (via PlotWidget.set_figure_size_inches).
        panel = PlotWidget()
        qtbot.addWidget(panel)
        self._draw(panel, 3)
        qtbot.wait(20)
        small_px = panel.canvas.height()

        self._draw(panel, 40)
        qtbot.wait(20)
        assert panel.canvas.height() > small_px

    def test_does_not_exceed_visible_panel_height(self, qtbot):
        # Growing the figure to fit every label is only a win if the whole
        # figure stays visible -- there's no scroll area around the canvas,
        # so a figure taller than the panel just gets silently clipped,
        # hiding some of the requested Top N with no indication more exists.
        panel = PlotWidget()
        qtbot.addWidget(panel)
        panel.resize(800, 500)  # deliberately too short for 40 bars at ~0.26in/bar
        panel.show()
        qtbot.waitExposed(panel)
        qtbot.wait(20)  # let any deferred draw_idle from show()/resize() flush

        try:
            self._draw(panel, 40)
            qtbot.wait(20)

            available_px = panel.height() - panel.toolbar.height() - 24
            assert panel.canvas.height() <= available_px + 1  # +1: rounding
        finally:
            panel.hide()
            qtbot.wait(20)  # settle before qtbot's teardown deletes the widget
