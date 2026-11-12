from contextlib import contextmanager

import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMainWindow

import data_model as dm
from plot_grammar import PlotSpec, render_plot
from qt_overview import OverviewPage
from qt_widgets import PlotWidget

from conftest import REAL_CSV_PATH, requires_real_data


@contextmanager
def _on_screen(qtbot, widget):
    """Show a widget for the length of a block, then hide it and let the
    event loop settle. A shown chart schedules a redraw when it is hidden;
    it must run before the test's teardown deletes the canvas, or it fires
    during the next test on a deleted object."""
    widget.show()
    qtbot.waitExposed(widget)
    try:
        yield widget
    finally:
        widget.hide()
        qtbot.wait(40)


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


class TestSiteDashboard:
    """The landing page as a dashboard: headline tiles, the farm map and
    the ranked list, all under the scope and linked to one another."""

    @contextmanager
    def _shown(self, qtbot, dataset):
        page = _make_page(qtbot)
        page.resize(1200, 800)
        with _on_screen(qtbot, page):
            page.on_dataset_changed(dataset)
            qtbot.wait(20)
            yield page

    def test_first_paint_is_an_empty_state(self, qtbot):
        page = _make_page(qtbot)
        assert page.map_stack.currentWidget() is page.map_empty
        assert page.tile_tanks.value() == "—"
        assert page.header.title_label.text() == "Site overview"

    def test_tiles_state_the_dataset(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        qtbot.wait(20)
        assert page.tile_tanks.value() == "2"
        assert page.tile_tanks.hint_label.text() == "1 SST · 1 DST"
        assert "1 single-shell" in page.tile_tanks.toolTip()
        assert page.tile_activity.hint_label.text() == "137Cs only"
        assert page.tile_records.value() == "9"
        assert page.tile_sodium.value() == "500 kg"
        assert page.tile_activity.value() == "300 Ci"
        assert "All 2 tanks" in page.header.subtitle_label.text()
        assert page.map_stack.currentWidget() is page.farm_map

    def test_map_shows_every_tank_and_leaves_unreported_hollow(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)     # default: Na, inventory in kg
        qtbot.wait(20)
        assert sorted(page.farm_map.tanks()) == ["241-A-101", "241-AN-104"]
        assert page.farm_map.value_of("241-AN-104") == pytest.approx(500.0)
        assert page.farm_map.class_of("241-A-101") is None       # no sodium record: hollow
        assert page.rank_list.tanks() == ["241-AN-104"]
        assert "1 without" in page.rank_subtitle.text()

    def test_colour_by_recolours_the_map_through_the_engine(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        page.set_colour_by("137Cs", "Ci")
        qtbot.wait(20)
        assert page.farm_map.value_of("241-A-101") == pytest.approx(100.0)
        assert page.farm_map.value_of("241-AN-104") == pytest.approx(200.0)
        assert page.rank_list.tanks() == ["241-AN-104", "241-A-101"]
        assert "137Cs" in page.rank_subtitle.text() and "Ci" in page.rank_subtitle.text()

    def test_a_typed_component_is_resolved(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        page.component_combo.setCurrentText("fe")
        page.component_combo.lineEdit().editingFinished.emit()
        qtbot.wait(20)
        assert page.farm_map.value_of("241-A-101") == pytest.approx(60.0)

    def test_scale_changes_without_a_new_query(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        page.set_colour_by("137Cs", "Ci")
        token = page._dashboard_token
        page.scale_combo.setCurrentIndex(page.scale_combo.findData("linear"))
        assert page._dashboard_token == token
        assert len(page.farm_map.breaks()) == 4

    def test_linked_selection_both_ways(self, qtbot, sample_dataset):
        with self._shown(qtbot, sample_dataset) as page:
            page.set_colour_by("137Cs", "Ci")
            qtbot.wait(20)
            fmap = page.farm_map
            qtbot.mouseClick(fmap, Qt.LeftButton, pos=fmap.tile_center("241-A-101"))
            assert page.rank_list.selected_tanks() == ["241-A-101"]
            assert "241-A-101" in page.selection_label.text() and "100 Ci" in page.selection_label.text()
            assert not page.open_btn.isHidden()
            page.rank_list.selectRow(0)                       # first row of the list: 241-AN-104
            assert fmap.selection() == ["241-AN-104"]
            page.clear_btn.click()
            assert fmap.selection() == [] and page.rank_list.selected_tanks() == []

    def test_linked_hover_both_ways(self, qtbot, sample_dataset):
        from PySide6.QtCore import QEvent, QPointF
        from PySide6.QtGui import QMouseEvent
        from PySide6.QtWidgets import QApplication
        with self._shown(qtbot, sample_dataset) as page:
            page.set_colour_by("137Cs", "Ci")
            qtbot.wait(20)
            page.rank_list.tankHovered.emit("241-A-101")
            assert page.farm_map._highlight == {"241-A-101"}
            pos = page.farm_map.tile_center("241-AN-104")
            QApplication.sendEvent(page.farm_map, QMouseEvent(
                QEvent.MouseMove, QPointF(pos), QPointF(page.farm_map.mapToGlobal(pos)),
                Qt.NoButton, Qt.NoButton, Qt.NoModifier))
            assert page.rank_list._linked_row == 0

    def test_double_click_opens_the_tank_through_the_window(self, qtbot, sample_dataset):
        with self._shown(qtbot, sample_dataset) as page:
            opened = []
            page.app_window.open_tank = opened.append
            qtbot.mouseDClick(page.farm_map, Qt.LeftButton, pos=page.farm_map.tile_center("241-AN-104"))
            assert opened == ["241-AN-104"]
            page.rank_list.tankActivated.emit("241-AN-104")
            assert opened == ["241-AN-104", "241-AN-104"]

    def test_scope_narrows_tiles_and_fades_the_other_tanks(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        page.set_colour_by("137Cs", "Ci")
        page.scope.set_filter("TankFarm", ["AN"])
        page.on_scope_changed()
        qtbot.wait(20)
        assert page.tile_tanks.value() == "1"
        assert "of 2" in page.tile_tanks.hint_label.text()
        assert "farm AN" in page.header.subtitle_label.text()
        assert sorted(page.farm_map.tanks()) == ["241-A-101", "241-AN-104"]     # still drawn
        assert page.farm_map.colour_of("241-A-101") is None
        assert page.rank_list.tanks() == ["241-AN-104"]

    def test_hidden_behind_another_page_it_waits_and_refreshes_when_shown(self, qtbot, sample_dataset):
        from PySide6.QtWidgets import QStackedWidget
        window = QMainWindow()
        qtbot.addWidget(window)
        stack = QStackedWidget()
        window.setCentralWidget(stack)
        other = QLabel("another workspace")
        page = OverviewPage(window)
        stack.addWidget(other)
        stack.addWidget(page)
        with _on_screen(qtbot, window):
            page.on_dataset_changed(sample_dataset)
            assert page._stale and page._stats is None            # nothing computed while hidden
            stack.setCurrentWidget(page)
            qtbot.waitUntil(lambda: page._stats is not None, timeout=5000)
            assert page.tile_tanks.value() == "2"
            stack.setCurrentWidget(other)
            page.scope.set_filter("TankFarm", ["AN"])
            page.on_scope_changed()
            assert page._stale and page.tile_tanks.value() == "2"
            stack.setCurrentWidget(page)
            qtbot.waitUntil(lambda: page.tile_tanks.value() == "1", timeout=5000)
            qtbot.wait(20)

    def test_a_stale_result_is_dropped(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        qtbot.wait(20)
        before = page.farm_map.value_of("241-AN-104")
        page._on_dashboard_done({"token": page._dashboard_token - 1, "dataset": sample_dataset,
                                 "scope": {}, "with_stats": False, "error": "", "values": None})
        assert page.farm_map.value_of("241-AN-104") == before

    def test_notes_are_one_calm_line_without_internal_names(self, qtbot):
        import polars as pl
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame({
            "WasteSiteId": ["241-A-101", "241-A-101", "241-A-101"],
            "Analyte": ["Na", "Acetone", "239/240Pu"],
            "WastePhase": ["Sludge"] * 3,
            "WasteType": ["T1"] * 3,
            "Inventory": [5.0, 1.0, 2.0],
            "Units": ["kg", "kg", "Ci"],
            "Inventory_RSD": [0.0, 0.0, 0.0],
        }))
        dataset.report = None
        page = _make_page(qtbot)
        page.on_dataset_changed(dataset)
        qtbot.wait(20)
        banner = page.warnings_label
        assert banner is page.notes_banner and not banner.isHidden()
        assert len(banner.notes()) > 1
        headline = banner.summary_label.text()
        assert "notes qualify the figures" in headline
        for internal in ("reference_data", "Inventory_RSD", "SPECIES_FORMULAS"):
            assert internal not in headline
            assert internal not in banner.text()
        # the Notes table keeps the wording of the engine
        raw = " ".join(page._table_views["Notes"].dataframe()["text"])
        assert "Inventory_RSD" in raw

    def test_export_carries_the_mapped_values(self, qtbot, sample_dataset, tmp_path):
        page = _make_page(qtbot)
        sample_dataset.output_root = tmp_path
        page.on_dataset_changed(sample_dataset)
        page.set_colour_by("137Cs", "Ci")
        qtbot.wait(20)
        page._export_debug_bundle()
        bundle = next(iter(tmp_path.glob("debug_bundle_*")))
        mapped = pd.read_csv(bundle / "map_values.csv")
        assert list(mapped["WasteSiteId"]) == ["241-AN-104", "241-A-101"]
        assert set(mapped["Unit"]) == {"Ci"}


@pytest.fixture(scope="module")
def real_dataset():
    dataset = dm.HanfordDataset()
    dataset.load(REAL_CSV_PATH, use_cache=True)
    return dataset


@requires_real_data
class TestSiteDashboardOnTheRealDataset:
    def test_tiles_are_the_engine_totals(self, qtbot, real_dataset):
        import site_summary as ss
        from query_engine import run_query
        page = _make_page(qtbot)
        page.on_dataset_changed(real_dataset)
        qtbot.wait(20)
        assert page.tile_tanks.value() == "177"
        assert page.tile_tanks.hint_label.text() == "149 SST · 28 DST"
        assert "149 single-shell tanks (SST) and 28 double-shell tanks (DST)" in page.tile_tanks.toolTip()
        assert page.tile_volume.value() == "200.6 ML"
        assert page.tile_volume.hint_label.text() == "53 million gallons"
        assert page.tile_records.value() == f"{real_dataset.require_df().height:,}"
        assert page.tile_analytes.value() == str(real_dataset.require_df().get_column("Analyte").n_unique())
        na = float(run_query(real_dataset, subject=("element", "Na"), basis="kg").table["Total_kg"].sum())
        assert page.tile_sodium.value() == ss.format_mass(na)
        cs = float(run_query(real_dataset, subject=("analyte", "137Cs"), basis="Ci").table["Total_Ci"].sum())
        sr = float(run_query(real_dataset, subject=("analyte", "90Sr"), basis="Ci").table["Total_Ci"].sum())
        assert page.tile_activity.value() == ss.format_activity(cs + sr)
        assert page.tile_activity.hint_label.text() == (
            f"137Cs {ss.format_number(cs / 1e6)} · 90Sr {ss.format_number(sr / 1e6)}")
        na_wt = float(run_query(real_dataset, subject=("element", "Na"), basis="wt%").table["wt_pct"].iloc[0])
        assert page.tile_sodium.hint_label.text() == f"{ss.format_number(na_wt)} wt% of waste"
        assert len(page.farm_map.tanks()) == 177

    def test_sulfate_map_ranks_and_scopes(self, qtbot, real_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(real_dataset)
        page.set_colour_by("SO4", "mol/mol Na")
        qtbot.wait(20)
        fmap = page.farm_map
        assert len(page.rank_list.tanks()) == 177
        assert fmap.class_count() == 5 and len(fmap.legend_labels()) == 4
        top = page.rank_list.tanks()[0]
        assert fmap.rank_of(top) == 1 and fmap.class_of(top) == 4
        assert "1st of 177 tanks" in fmap.tooltip_text(top)
        page.scope.set_filter("TankFarm", ["AN"])
        page.on_scope_changed()
        qtbot.wait(20)
        assert page.tile_tanks.value() == "7"
        assert len(page.rank_list.tanks()) == 7
        assert len(fmap.tanks()) == 177
        assert fmap.colour_of("241-AW-106") is None
