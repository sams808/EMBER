"""
Repro tests for the full-app recheck (two parallel error hunts over the
whole session's work). Grouped by the thing each one protects.
"""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from PySide6.QtWidgets import QMainWindow

import data_model as dm
import matrix_query as mq
from plot_grammar import PlotSpec, render_plot
from qt_heatmap import HeatmapPage
from qt_overview import OverviewPage
from qt_widgets import PlotWidget
from query_engine import run_query

from conftest import REAL_CSV_PATH, requires_real_data


def _panel(qtbot):
    panel = PlotWidget()
    qtbot.addWidget(panel)
    return panel


def _page(qtbot, cls):
    window = QMainWindow()
    qtbot.addWidget(window)
    page = cls(window)
    qtbot.addWidget(page)
    return page


def _dataset() -> dm.HanfordDataset:
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame({
        "WasteSiteId": ["241-A-101", "241-A-101", "241-AN-104"],
        "Analyte": ["Na", "SO4", "Na"],
        "WastePhase": ["Sludge", "Sludge", "Supernatant"],
        "WasteType": ["T1", "T1", "T2"],
        "Inventory": [10.0, 100.0, 90.0],
        "Units": ["kg", "kg", "kg"],
    }))
    dataset.report = None
    return dataset


class TestElementGroupingDoesNotMislabel:
    """group_by=["Element"] labelled the SUBJECT's apportioned mass with
    the SOURCE ROW's primary element: querying oxygen and grouping by
    element produced a row labelled "N" holding 3.1x the real nitrogen."""

    def test_element_grouping_is_refused_for_an_element_subject(self):
        with pytest.raises(ValueError, match="Element"):
            run_query(_dataset(), subject=("element", "O"), basis="kg",
                      group_by=["Element"])

    def test_the_refusal_points_at_the_dimension_that_works(self):
        try:
            run_query(_dataset(), subject=("element", "O"), basis="kg", group_by=["Element"])
        except ValueError as exc:
            assert "Analyte" in str(exc)

    def test_analyte_grouping_still_works(self):
        result = run_query(_dataset(), subject=("element", "O"), basis="kg",
                           group_by=["Analyte"])
        assert set(result.table["Analyte"]) == {"SO4"}

    def test_an_analyte_subject_may_still_group_by_element(self):
        # There the label and the value describe the same row.
        result = run_query(_dataset(), subject=("analyte", "SO4"), basis="kg",
                           group_by=["Element"])
        assert not result.table.empty


class TestUnparseableMassIsWarnedAboutEverywhere:
    def test_run_query_reports_analytes_the_parser_cannot_resolve(self):
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame({
            "WasteSiteId": ["241-A-101"] * 2,
            "Analyte": ["Na", "Aroclor-1254"],
            "WastePhase": ["Sludge"] * 2,
            "WasteType": ["T1"] * 2,
            "Inventory": [90.0, 10.0],
            "Units": ["kg", "kg"],
        }))
        dataset.report = None
        result = run_query(dataset, subject=("element", "Na"), basis="kg")
        assert any("cannot resolve" in w.lower() or "unresolvable" in w.lower()
                   for w in result.warnings), result.warnings


class TestLegacyMatrixDoesNotFabricateZeros:
    def test_as_reported_wide_matrix_leaves_absent_cells_blank(self):
        result = mq.legacy_matrix(_dataset(), unit="kg", top_n_elements=10)
        wide = result.wide_df.set_index("WasteSiteId")
        # 241-AN-104 has no sulfur row at all.
        assert "S" in wide.columns
        assert pd.isna(wide.loc["241-AN-104", "S"]), (
            "the as-reported matrix published a measured 0 kg where nothing was measured"
        )

    def test_as_reported_colorbar_uses_a_unit_honest_label(self, qtbot):
        page = _page(qtbot, HeatmapPage)
        page.on_dataset_changed(_dataset())
        page.attribution_combo.setCurrentText(mq.AS_REPORTED)
        page.set_mode("inventory")
        page.run()
        qtbot.wait(20)
        label = page.plot.figure.axes[1].get_ylabel()
        assert "Inventory_kg" not in label, f"raw column name leaked to the colorbar: {label}"
        assert "kg" in label


class TestOverviewHonorsTheGlobalScope:
    def test_scoped_overview_matches_a_scoped_query(self, qtbot):
        page = _page(qtbot, OverviewPage)
        page.on_dataset_changed(_dataset())
        page.unit_combo.setCurrentText("kg")
        page.scope.set_filter("WastePhase", ["Supernatant"])
        page.refresh()
        qtbot.wait(20)
        totals = page._element_totals.table.set_index("Element")["Total_kg"]
        assert totals["Na"] == pytest.approx(90.0), (
            "Overview ignored the global scope and reported every farm"
        )

    def test_the_scope_is_stamped_on_the_figure(self, qtbot):
        page = _page(qtbot, OverviewPage)
        page.on_dataset_changed(_dataset())
        page.unit_combo.setCurrentText("kg")
        page.scope.set_filter("WastePhase", ["Supernatant"])
        page.refresh()
        qtbot.wait(20)
        assert "Supernatant" in page.plot.ax.get_title()

    def test_a_scope_change_refreshes_the_page(self, qtbot):
        page = _page(qtbot, OverviewPage)
        page.on_dataset_changed(_dataset())
        page.unit_combo.setCurrentText("kg")
        page.refresh()
        qtbot.wait(20)
        page.scope.set_filter("WastePhase", ["Supernatant"])
        page.on_scope_changed()
        qtbot.wait(20)
        totals = page._element_totals.table.set_index("Element")["Total_kg"]
        assert totals["Na"] == pytest.approx(90.0)


def _many_elements_dataset() -> dm.HanfordDataset:
    """8 elements, so a top-N of 5 (the spin boxes' minimum) really trims."""
    analytes = ["Na", "Fe", "Cr", "Al", "Si", "Ca", "Ni", "Zr"]
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame({
        "WasteSiteId": ["241-A-101"] * len(analytes),
        "Analyte": analytes,
        "WastePhase": ["Sludge"] * len(analytes),
        "WasteType": ["T1"] * len(analytes),
        "Inventory": [float(100 - 5 * i) for i in range(len(analytes))],
        "Units": ["kg"] * len(analytes),
    }))
    dataset.report = None
    return dataset


class TestTrimsReachTheFigure:
    def test_heatmap_states_its_element_trim_on_the_figure(self, qtbot):
        page = _page(qtbot, HeatmapPage)
        page.on_dataset_changed(_many_elements_dataset())
        page.top_n_spin.setValue(5)   # 5 of 8 elements
        page.run()
        qtbot.wait(20)
        text = (page.plot.figure.get_suptitle() or "") + page.plot.ax.get_title()
        assert "omitted" in text or "of" in text.lower()
        assert any("omitted" in w for w in page._result.warnings)

    def test_max_tanks_trim_is_warned_about(self):
        result = mq.engine_matrix(_dataset(), basis="kg", top_n_elements=10,
                                  tank_subset=["241-A-101"])
        assert any("tank" in w.lower() for w in result.warnings), result.warnings

    def test_overview_states_its_trim_on_the_figure(self, qtbot):
        page = _page(qtbot, OverviewPage)
        page.on_dataset_changed(_many_elements_dataset())
        page.unit_combo.setCurrentText("kg")
        page.top_n_spin.setValue(5)   # 5 of 8 elements
        page.refresh()
        qtbot.wait(20)
        text = (page.plot.figure.get_suptitle() or "") + page.plot.ax.get_title()
        assert "omitted" in text


class TestDebugBundleIsHonestAboutItsContents:
    def test_the_all_file_is_not_a_silent_top_n(self, qtbot, tmp_path):
        dataset = _dataset()
        dataset.output_root = tmp_path
        page = _page(qtbot, OverviewPage)
        page.on_dataset_changed(dataset)
        page.unit_combo.setCurrentText("kg")
        page.top_n_spin.setValue(1)
        page.refresh()
        qtbot.wait(20)
        page._export_debug_bundle()
        bundle = next(iter(tmp_path.glob("debug_bundle_*")))
        exported = pd.read_csv(bundle / "top_elements_all.csv")
        assert len(exported) > 1, "a file named _all held only the on-screen top 1"
        assert (bundle / "notes.csv").exists(), "the bundle carries no warnings file"


class TestGrammarMarksDiscloseTheirDrops:
    def test_scatter_reports_points_it_could_not_draw(self, qtbot):
        panel = _panel(qtbot)
        df = pd.DataFrame({"x": [1.0, 2.0, 0.0, -5.0], "y": [1.0, 2.0, 3.0, 4.0]})
        render_plot(panel, df, PlotSpec(mark="scatter", x="x", y="y",
                                        title="T", log_x=True))
        qtbot.wait(20)
        assert "not plottable" in panel.ax.get_title()

    def test_faceted_bars_do_not_sum_intensive_values(self, qtbot):
        panel = _panel(qtbot)
        df = pd.DataFrame({
            "Tank": ["T1", "T1", "T2", "T2"],
            "Farm": ["A"] * 4,
            "wt_pct": [40.0, 45.0, 10.0, 5.0],
        })
        render_plot(panel, df, PlotSpec(mark="bar", x="wt_pct", y="Tank", facet="Farm",
                                        title="T", intensive=True))
        qtbot.wait(20)
        widths = sorted(p.get_width() for ax in panel.figure.axes for p in ax.patches)
        assert widths == pytest.approx([10.0, 45.0]), (
            f"intensive values were summed across rows: {widths}"
        )

    def test_faceted_bars_disclose_unplottable_categories(self, qtbot):
        panel = _panel(qtbot)
        df = pd.DataFrame({
            "Tank": ["T1", "T2", "T3"],
            "Farm": ["A"] * 3,
            "Total_kg": [10.0, 0.0, -3.0],
        })
        render_plot(panel, df, PlotSpec(mark="bar", x="Total_kg", y="Tank",
                                        facet="Farm", title="T"))
        qtbot.wait(20)
        footer = " ".join(t.get_text() for t in panel.figure.texts)
        assert "not plottable" in footer


class TestLogScaleHidesNothingSilently:
    def test_measured_zeros_dropped_by_a_log_scale_are_disclosed(self, qtbot):
        panel = _panel(qtbot)
        df = pd.DataFrame({
            "Tank": ["T1", "T1", "T2"],
            "Element": ["Bi", "Na", "Bi"],
            "Total_kg": [0.0, 5.0, 3.0],
        })
        render_plot(panel, df, PlotSpec(mark="heatmap", x="Total_kg", y="Tank",
                                        hue="Element", title="T", log_x=True))
        qtbot.wait(20)
        text = (panel.figure.get_suptitle() or "") + panel.ax.get_title()
        assert "zero" in text.lower(), (
            "a measured zero was blanked by the log transform with no note"
        )


@requires_real_data
class TestRealDataCrossWorkspaceAgreement:
    def test_overview_heatmaps_and_query_agree_under_a_scope(self, qtbot):
        dataset = dm.HanfordDataset()
        dataset.load(REAL_CSV_PATH, use_cache=True)

        overview = _page(qtbot, OverviewPage)
        overview.on_dataset_changed(dataset)
        overview.unit_combo.setCurrentText("kg")
        overview.scope.set_filter("TankFarm", ["AN"])
        overview.refresh()
        qtbot.wait(50)

        direct = run_query(dataset, subject=("element", "Na"), basis="kg",
                           scope={"TankFarm": ["AN"]})
        ov_na = float(overview._element_totals.table.set_index("Element").loc["Na", "Total_kg"])
        assert ov_na == pytest.approx(float(direct.table["Total_kg"].sum()), rel=1e-9)
